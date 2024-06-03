"""Script to run an experiment with LEAP

Usage:
pdm run python -m circuit_finder.experiments.run_leap_experiment [ARGS]

Run with flag '-h', '--help' to see the arguments.
"""

import sys

sys.path.append("/workspace/circuit-finder")

import transformer_lens as tl
import pandas as pd
import json

from simple_parsing import ArgumentParser
from dataclasses import dataclass
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.data_loader import load_datasets_from_json
from circuit_finder.constants import device
from tqdm import tqdm

from typing import Literal
from pathlib import Path
from jaxtyping import Int, Float
from torch import Tensor
from circuit_finder.core.types import MetricFn
from circuit_finder.pretrained import (
    load_attn_saes,
    load_hooked_mlp_transcoders,
)
from circuit_finder.patching.indirect_leap import (
    preprocess_attn_saes,
    IndirectLEAP,
    LEAPConfig,
)
from circuit_finder.metrics import batch_avg_answer_diff

from circuit_finder.constants import ProjectDir

import torch
from einops import rearrange, einsum
from functools import partial


def get_mask(
    graph, module_name, layer, feature_acts: Float[Tensor, "batch seq n_features"]
):
    """
    Returns mask : [seq, n_features].
    mask[seq, feature_id] is False whenever {module_name}.{layer}.{seq}.{feature_id} is a node of graph.
    All other entries are True
    """
    nodes = graph.get_src_nodes()
    positions = [
        int(node.split(".")[2])
        for node in nodes
        if node.split(".")[:2] == [module_name, str(layer)]
    ]
    feature_ids = [
        int(node.split(".")[3])
        for node in nodes
        if node.split(".")[:2] == [module_name, str(layer)]
    ]
    mask = torch.ones_like(feature_acts[0]).bool()
    mask[positions, feature_ids] = False
    return mask


def get_metric_with_ablation(
    model: tl.HookedTransformer,
    graph: EAPGraph,
    tokens: Int[Tensor, "batch seq"],
    metric: MetricFn,
    transcoders,  # list
    attn_saes,  # list
    ablate_nodes: str | bool = "zero",  # options [False, "bm", "zero"]
    ablate_errors: str | bool = False,  # options [False, "bm", "zero"]
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
    freeze_attention: bool = False,
):
    """Cache the activations of the model on a circuit, then return the metric when ablated"""
    assert ablate_errors in [False, "bm", "zero"]
    assert ablate_nodes in [False, "bm", "zero"]
    # Do clean FP to get batchmean (bm) feature acts, and reconstructions
    # No patching is done during this FP, so the hook fns return nothing
    # Note: MLP transcoders are a bit more fiddly than attn-SAEs, requiring us to cache at both mlp_in and mlp_out
    mlp_bm_feature_act_cache = {}
    attn_bm_feature_act_cache = {}
    mlp_recons_cache = {}
    mlp_error_cache = {}
    attn_error_cache = {}

    def mlp_in_cache_hook(act, hook, layer):
        assert hook.name.endswith("ln2.hook_normalized")
        t = transcoders[layer]
        feature_acts = torch.relu(
            einsum(
                act - t.b_dec,
                t.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + t.b_enc
        )
        recons = feature_acts @ t.W_dec + t.b_dec_out

        mlp_bm_feature_act_cache[layer] = feature_acts.mean(0)
        mlp_recons_cache[layer] = recons

    def mlp_out_cache_hook(act, hook, layer):
        assert hook.name.endswith("mlp_out")
        mlp_error_cache[layer] = act - mlp_recons_cache[layer]

    def attn_cache_hook(act, hook, layer):
        assert hook.name.endswith("hook_z")
        sae = attn_saes[layer]
        z_concat = rearrange(act, "b s n_heads d_head -> b s (n_heads d_head)")
        feature_acts = torch.relu(
            einsum(
                z_concat - sae.b_dec,
                sae.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + sae.b_enc
        )
        recons_concat = feature_acts @ sae.W_dec + sae.b_dec
        recons = rearrange(
            recons_concat,
            "b s (n_heads d_head) -> b s n_heads d_head",
            n_heads=model.cfg.n_heads,
        )

        attn_bm_feature_act_cache[layer] = feature_acts.mean(0)
        attn_error_cache[layer] = act - recons

    model.reset_hooks()
    for layer in range(model.cfg.n_layers):
        model.add_hook(
            f"blocks.{layer}.ln2.hook_normalized",
            partial(mlp_in_cache_hook, layer=layer),
            "fwd",
        )
        model.add_hook(
            f"blocks.{layer}.hook_mlp_out",
            partial(mlp_out_cache_hook, layer=layer),
            "fwd",
        )
        model.add_hook(
            f"blocks.{layer}.attn.hook_z", partial(attn_cache_hook, layer=layer), "fwd"
        )

    # Run forward pass and populate all caches
    _, pattern_cache = model.run_with_cache(
        tokens, return_type="loss", names_filter=lambda x: x.endswith("pattern")
    )
    assert len(pattern_cache) > 0
    assert len(mlp_bm_feature_act_cache) > 0
    assert len(attn_bm_feature_act_cache) > 0
    assert len(mlp_recons_cache) > 0
    assert len(mlp_error_cache) > 0
    assert len(attn_error_cache) > 0

    # Now do FP where we patch nodes not in graph with their batchmeaned values
    model.reset_hooks()
    mlp_ablated_recons = {}

    def freeze_pattern_hook(act, hook):
        assert hook.name.endswith("pattern")
        return pattern_cache[hook.name]

    def mlp_out_ablated_recons_cache_hook(act, hook, layer):
        assert hook.name.endswith("ln2.hook_normalized")
        t = transcoders[layer]
        feature_acts = torch.relu(
            einsum(
                act - t.b_dec,
                t.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + t.b_enc
        )
        mask = get_mask(graph, "mlp", layer, feature_acts)
        ablated_feature_acts = feature_acts.clone()
        ablated_feature_acts[:, mask] = mlp_bm_feature_act_cache[layer][mask]
        mlp_ablated_recons[layer] = ablated_feature_acts @ t.W_dec + t.b_dec_out

    def mlp_patch_hook(act, hook, layer):
        assert hook.name.endswith("mlp_out")

        if ablate_errors == "bm":
            return mlp_ablated_recons[layer] + mlp_error_cache[layer].mean(
                dim=0, keepdim=True
            )
        elif ablate_errors == "zero":
            return mlp_ablated_recons[layer]
        else:
            return mlp_ablated_recons[layer] + mlp_error_cache[layer]

    def attn_patch_hook(act, hook, layer):
        assert hook.name.endswith("hook_z")
        sae = attn_saes[layer]
        z_concat = rearrange(act, "b s n_heads d_head -> b s (n_heads d_head)")
        feature_acts = torch.relu(
            einsum(
                z_concat - sae.b_dec,
                sae.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + sae.b_enc
        )

        mask = get_mask(graph, "attn", layer, feature_acts)
        ablated_feature_acts = feature_acts.clone()
        ablated_feature_acts[:, mask] = attn_bm_feature_act_cache[layer][mask]
        ablated_recons_concat = ablated_feature_acts @ sae.W_dec + sae.b_dec
        ablated_recons = rearrange(
            ablated_recons_concat,
            "b s (n_heads d_head) -> b s n_heads d_head",
            n_heads=model.cfg.n_heads,
        )

        if ablate_errors == "bm":
            return ablated_recons + attn_error_cache[layer].mean(dim=0, keepdim=True)
        elif ablate_errors == "zero":
            return ablated_recons
        else:
            return ablated_recons + attn_error_cache[layer]

    for layer in range(first_ablated_layer, model.cfg.n_layers):
        model.add_hook(
            f"blocks.{layer}.ln2.hook_normalized",
            partial(mlp_out_ablated_recons_cache_hook, layer=layer),
            "fwd",
        )
        model.add_hook(
            f"blocks.{layer}.hook_mlp_out", partial(mlp_patch_hook, layer=layer), "fwd"
        )
        model.add_hook(
            f"blocks.{layer}.attn.hook_z", partial(attn_patch_hook, layer=layer), "fwd"
        )
        if freeze_attention:
            model.add_hook(
                f"blocks.{layer}.attn.hook_pattern", freeze_pattern_hook, "fwd"
            )

    return metric(model, tokens)


@dataclass
class LeapExperimentConfig:
    dataset_path: str = "datasets/subject_verb_agreement.json"
    save_dir: str = "results/subject_verb_agreement"
    seed: int = 1
    batch_size: int = 8
    total_dataset_size: int = 1024
    ablate_nodes: bool | str = "bm"
    ablate_errors: bool | str = "bm"
    ablate_tokens: Literal["clean", "corrupt"] = "corrupt"
    # NOTE: This specifies what to do with error nodes when calculating faithfulness curves.
    # Options are:
    # - ablate_errors = False  ->  we don't ablate error nodes
    # - ablate_errors = "bm"   ->  we batchmean-ablate error nodes
    # - ablate_errors = "zero" ->  we zero-ablate error nodes (warning: this gives v bad performance)

    first_ablate_layer: int = 2
    # NOTE:This specifies which layer to start ablating at.
    # Marks et al don't ablate the first 2 layers
    # TODO: Find reference for the above

    verbose: bool = False


def run_leap_experiment(config: LeapExperimentConfig):
    # Define save dir
    save_dir = ProjectDir / config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save the config
    with open(save_dir / "config.json", "w") as jsonfile:
        json.dump(config.__dict__, jsonfile)

    # Load models
    model = tl.HookedTransformer.from_pretrained(
        "gpt2",
        device="cuda",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )

    attn_saes = load_attn_saes()
    attn_saes = preprocess_attn_saes(attn_saes, model)  # type: ignore
    transcoders = load_hooked_mlp_transcoders()

    # Define dataset
    dataset_path = config.dataset_path
    if not dataset_path.startswith("/"):
        # Assume relative path
        dataset_path = ProjectDir / dataset_path
    else:
        dataset_path = Path(dataset_path)
    train_loader, _ = load_datasets_from_json(
        model=model,
        path=dataset_path,
        device=device,  # type: ignore
        batch_size=config.batch_size,
        random_seed=config.seed,
    )

    for idx, batch in tqdm(enumerate(train_loader)):
        # Set up batch dir
        batch_dir = save_dir / f"batch_{idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save the config
        with open(batch_dir / "config.json", "w") as jsonfile:
            json.dump(config.__dict__, jsonfile)

        # Parse the batch
        key = batch.key
        clean_tokens = batch.clean
        answer_tokens = batch.answers
        wrong_answer_tokens = batch.wrong_answers
        corrupt_tokens = batch.corrupt
        ablate_tokens = (
            clean_tokens if config.ablate_tokens == "clean" else corrupt_tokens
        )

        # Define metric
        def metric_fn(model, tokens):
            logits = model(tokens)
            last_token_logits = logits[:, -1, :]
            return batch_avg_answer_diff(last_token_logits, batch)

        # Save the dataset.
        with open(batch_dir / "dataset.json", "w") as jsonfile:
            json.dump(
                {
                    "clean": model.to_string(clean_tokens),
                    "answer": model.to_string(answer_tokens),
                    "wrong_answer": model.to_string(wrong_answer_tokens),
                    "corrupt": model.to_string(corrupt_tokens),
                },
                jsonfile,
            )

        # NOTE: First, get the ceiling of the patching metric.
        # TODO: Replace 'last_token_logit' with logit difference
        ceiling = metric_fn(model, clean_tokens).item()

        # NOTE: Second, get floor of patching metric using empty graph, i.e. ablate everything
        empty_graph = EAPGraph([])
        floor = get_metric_with_ablation(
            model,
            empty_graph,
            ablate_tokens,
            metric_fn,
            transcoders,
            attn_saes,
            ablate_errors=False,  # Do not ablate errors when running forward pass
            first_ablated_layer=config.first_ablate_layer,
        ).item()
        clear_memory()

        # now sweep over thresholds to get graphs with variety of numbers of nodes
        # for each graph we calculate faithfulness
        num_nodes_list = []
        metrics_list = []

        # Sweep over thresholds
        # TODO: make configurable
        thresholds = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
        for threshold in thresholds:
            # Setup LEAP algorithm
            model.reset_hooks()
            cfg = LEAPConfig(
                threshold=threshold, contrast_pairs=False, chained_attribs=True
            )
            leap = IndirectLEAP(
                cfg=cfg,
                tokens=clean_tokens,
                model=model,
                metric=metric_fn,
                attn_saes=attn_saes,  # type: ignore
                transcoders=transcoders,
                corrupt_tokens=corrupt_tokens,
            )

            # Populate the graph
            leap.metric_step()
            for layer in reversed(range(1, leap.n_layers)):
                leap.mlp_step(layer)
                leap.ov_step(layer)

            # Save the graph
            graph = EAPGraph(leap.graph)
            num_nodes = len(graph.get_src_nodes())
            with open(
                batch_dir / f"leap-graph_threshold={threshold}.json", "w"
            ) as jsonfile:
                json.dump(graph.to_json(), jsonfile)

            # Delete tensors to save memory
            del leap
            clear_memory()

            # Calculate the metric under ablation
            metric = get_metric_with_ablation(
                model,
                graph,
                ablate_tokens,
                metric_fn,
                transcoders,
                attn_saes,
                ablate_errors=config.ablate_errors,  # type: ignore
                first_ablated_layer=config.first_ablate_layer,
            ).item()

            # Log the data
            num_nodes_list.append(num_nodes)
            metrics_list.append(metric)

        faith = [(metric - floor) / (ceiling - floor) for metric in metrics_list]

        # Save the result as a dataframe
        data = pd.DataFrame({"num_nodes": num_nodes_list, "faithfulness": faith})
        data.to_csv(batch_dir / "leap_experiment_results.csv", index=False)
        break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    args = parser.parse_args()

    run_leap_experiment(args.config)
