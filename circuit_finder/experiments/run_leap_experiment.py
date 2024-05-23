"""Script to run an experiment with LEAP

Usage:
pdm run python -m circuit_finder.experiments.run_leap_experiment [ARGS]

Run with flag '-h', '--help' to see the arguments.
"""

import torch
import transformer_lens as tl
import pandas as pd
import json

from jaxtyping import Int, Float
from torch import Tensor
from eindex import eindex
from simple_parsing import ArgumentParser
from dataclasses import dataclass
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.patching.ablate import get_metric_with_ablation
from tqdm import tqdm

from pathlib import Path
from circuit_finder.pretrained import (
    load_attn_saes,
    load_mlp_transcoders,
)
from circuit_finder.patching.leap import (
    preprocess_attn_saes,
    LEAP,
    LEAPConfig,
)

from circuit_finder.constants import ProjectDir


@dataclass
class LeapExperimentConfig:
    dataset_path: str = "datasets/ioi/ioi_vanilla_template_prompts.json"
    save_dir: str = "results/leap_experiment"
    seed: int = 1
    batch_size: int = 4
    total_dataset_size: int = 1024
    ablate_errors: bool | str = False
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


def logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer: Int[Tensor, " batch"],
    wrong_answer: Int[Tensor, " batch"],
):
    # Get last-token logits
    logits = logits[:, -1, :]
    correct_logits = eindex(logits, correct_answer, "batch [batch]")
    wrong_logits = eindex(logits, wrong_answer, "batch [batch]")
    return (correct_logits - wrong_logits).mean()


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
    transcoders = load_mlp_transcoders()

    # Define dataset
    dataset_path = config.dataset_path
    if not dataset_path.startswith("/"):
        # Assume relative path
        dataset_path = ProjectDir / dataset_path
    else:
        dataset_path = Path(dataset_path)
    # train_loader, _ = load_datasets_from_json(
    #     model=model,
    #     path=dataset_path,
    #     device=device,  # type: ignore
    #     batch_size=config.batch_size,
    #     train_test_size=(config.total_dataset_size, config.total_dataset_size),
    #     random_seed=config.seed,
    # )

    # # Get a batch on which we run the experiment
    # batch: PromptPairBatch = next(iter(train_loader))
    # clean_tokens = batch.clean
    # answer_tokens = batch.answers
    # wrong_answer_tokens = batch.wrong_answers
    # corrupt_tokens = batch.corrupt
    # # TODO: handle case where answer_token is list
    # # This is applicable e.g. for 'GreaterThan' which has many correct and wrong answers
    # # and we are expected to sum over them.
    # # NOTE: for now, we only support single answer and wrong answer

    clean_tokens = model.to_tokens(
        ["When John and Mary were at the store, John gave a bottle to"]
    )
    answer_tokens = model.to_tokens(["Mary"], prepend_bos=False)
    wrong_answer_tokens = model.to_tokens(["John"], prepend_bos=False)
    corrupt_tokens = model.to_tokens(
        ["When Alice and Bob were at the store, Charlie gave a bottle to"]
    )

    assert isinstance(answer_tokens, torch.Tensor), "Only single answer supported"
    assert isinstance(
        wrong_answer_tokens, torch.Tensor
    ), "Only single wrong answer supported"

    # Define metric
    def metric_fn(model, tokens):
        logits = model(tokens, return_type="logits")
        return logit_diff(
            logits,
            answer_tokens.squeeze(dim=-1),
            wrong_answer_tokens.squeeze(dim=-1),
        )

    # Save the dataset.
    with open(save_dir / "dataset.json", "w") as jsonfile:
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
    print(f"Ceiling: {ceiling}")

    # NOTE: Second, get floor of patching metric using empty graph, i.e. ablate everything
    empty_graph = EAPGraph([])
    floor = get_metric_with_ablation(
        model,
        empty_graph,
        clean_tokens,
        metric_fn,
        transcoders,
        attn_saes,
        ablate_errors=False,  # Do not ablate errors when running forward pass
        first_ablated_layer=config.first_ablate_layer,
    ).item()
    print(f"Floor: {floor}")
    clear_memory()

    # now sweep over thresholds to get graphs with variety of numbers of nodes
    # for each graph we calculate faithfulness
    num_nodes_list = []
    metrics_list = []

    # Sweep over thresholds
    # TODO: make configurable
    thresholds = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
    for threshold in tqdm(thresholds):
        # Setup LEAP algorithm
        model.reset_hooks()
        cfg = LEAPConfig(
            threshold=threshold, contrast_pairs=False, chained_attribs=True
        )
        leap = LEAP(
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
        with open(save_dir / f"leap-graph_threshold={threshold}.json", "w") as jsonfile:
            json.dump(graph.to_json(), jsonfile)

        # Delete tensors to save memory
        del leap
        clear_memory()

        # Calculate the metric under ablation
        metric = get_metric_with_ablation(
            model,
            empty_graph,
            clean_tokens,
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
    data.to_csv(save_dir / "leap_experiment_results.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    args = parser.parse_args()

    run_leap_experiment(args.config)
