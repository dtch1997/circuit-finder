# flake8: ignore
"""Script to run an experiment with LEAP

Usage:
pdm run python -m circuit_finder.experiments.run_leap_experiment [ARGS]

Run with flag '-h', '--help' to see the arguments.
"""

import sys

sys.path.append("/workspace/circuit-finder")

import pathlib
import pickle
import json
import torch
import transformer_lens as tl

from pprint import pprint
from simple_parsing import ArgumentParser
from dataclasses import dataclass, replace, asdict
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.patching.ablate import get_metric_with_ablation
from circuit_finder.data_loader import load_datasets_from_json, PromptPairBatch
from circuit_finder.experiments.run_dataset_sweep import SELECTED_DATASETS

from eindex import eindex
from circuit_finder.pretrained import (
    load_model,
    load_attn_saes,
    load_hooked_mlp_transcoders,
)
from circuit_finder.patching.indirect_leap import (
    preprocess_attn_saes,
    IndirectLEAP,
    LEAPConfig,
)
from circuit_finder.core.types import Model
from circuit_finder.metrics import batch_avg_answer_diff
from circuit_finder.constants import ProjectDir
from circuit_finder.patching.ablate import (
    splice_model_with_saes_and_transcoders,
    get_metric_with_ablation,
    AblateType,
)
from circuit_finder.plotting import make_html_graph

THRESHOLDS = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]


def batch_to_str_dict(batch: PromptPairBatch, model):
    return {
        "clean": model.to_string(batch.clean),
        "answer": model.to_string(batch.answers),
        "wrong_answer": model.to_string(batch.wrong_answers),
        "corrupt": model.to_string(batch.corrupt),
    }


def load_models():
    model = load_model()
    attn_sae_dict = load_attn_saes()
    # TODO: get rid of need to preprocess attn saes
    attn_sae_dict = preprocess_attn_saes(attn_sae_dict, model)
    hooked_mlp_transcoder_dict = load_hooked_mlp_transcoders()
    attn_saes = list(attn_sae_dict.values())
    transcoders = list(hooked_mlp_transcoder_dict.values())

    return model, attn_saes, transcoders


def compute_logit_diff(model, clean_tokens, answer_tokens, wrong_answer_tokens):
    clean_logits = model(clean_tokens)
    last_logits = clean_logits[:, -1, :]
    correct_logits = eindex(last_logits, answer_tokens, "batch [batch]")
    wrong_logits = eindex(last_logits, wrong_answer_tokens, "batch [batch]")
    return correct_logits - wrong_logits


@dataclass
class LeapExperimentConfig:
    leap_config: LEAPConfig
    ablate_act_type: str = "unstructured"  # "structured", "tokenwise"
    feature_ablate_type: str | None = "value"
    error_ablate_type: str | None = None
    first_ablated_layer: int = 2
    thresholds: tuple[float] = tuple(THRESHOLDS)
    metric_fn_name: str = "logit_diff"
    batch_size: int = 1
    save_dir_prefix: str = "leap_experiment_results"


@dataclass(frozen=True)
class LeapExperimentResult:
    config: LEAPConfig
    batch: PromptPairBatch
    clean_metric: float
    graph_ablated_metric: float
    fully_ablated_metric: float
    graph: EAPGraph
    error_graph: EAPGraph


def run_leap(
    leap_config: LEAPConfig,
    batch: PromptPairBatch,
    model: Model,
    attn_saes,
    hooked_mlp_transcoders,
    ablate_cache: tl.ActivationCache,
    *,
    save_dir: pathlib.Path,
    metric_fn_name: str = "logit_diff",
    feature_ablate_type: AblateType = "value",
    error_ablate_type: AblateType = None,
    first_ablated_layer: int = 2,
):
    # Setup the tokens
    clean_tokens = batch.clean
    answer_tokens = batch.answers
    wrong_answer_tokens = batch.wrong_answers
    corrupt_tokens = batch.corrupt

    # Save the batch
    batch_dict = batch_to_str_dict(batch, model)
    with open(save_dir / "batch.json", "w") as f:
        json.dump(batch_dict, f)

    # Setup the metric function
    if metric_fn_name == "logit_diff":

        def metric_fn(model, tokens):
            # Get the last-token logits
            logits = model(tokens)[:, -1, :]
            logit_diff = batch_avg_answer_diff(logits, batch)
            return logit_diff.mean()
    else:
        raise ValueError(f"Unknown metric_fn_name: {metric_fn_name}")

    # Get the clean metric
    with torch.no_grad():
        clean_metric = metric_fn(model, clean_tokens).item()

    # Get the fully ablated metric
    with torch.no_grad():
        empty_graph = EAPGraph([])
        fully_ablated_metric = get_metric_with_ablation(
            model,
            empty_graph,
            clean_tokens,
            metric_fn,
            hooked_mlp_transcoders,
            attn_saes,
            ablate_cache,
            feature_ablate_type=feature_ablate_type,
            error_ablate_type=error_ablate_type,
            first_ablated_layer=first_ablated_layer,
        ).item()

    # Run LEAP
    leap = IndirectLEAP(
        cfg=leap_config,
        tokens=clean_tokens,
        model=model,
        metric=metric_fn,
        attn_saes=attn_saes,  # type: ignore
        transcoders=hooked_mlp_transcoders,
        corrupt_tokens=corrupt_tokens,
    )
    leap.run()
    graph = EAPGraph(leap.graph)
    error_graph = EAPGraph(leap.error_graph)

    # Save the graph
    make_html_graph(leap, html_file=str(save_dir.absolute() / "graph.html"))

    # Run ablation experiment
    graph_ablated_metric = get_metric_with_ablation(
        model,  # type: ignore
        graph,
        clean_tokens,
        metric_fn,
        hooked_mlp_transcoders,
        attn_saes,
        ablate_cache,
        feature_ablate_type=feature_ablate_type,
        error_ablate_type=error_ablate_type,
        first_ablated_layer=first_ablated_layer,
    ).item()

    # TODO: noising, denoising plots.

    return LeapExperimentResult(
        config=leap_config,
        batch = batch,
        clean_metric=clean_metric,
        graph_ablated_metric=graph_ablated_metric,
        fully_ablated_metric=fully_ablated_metric,
        graph=graph,
        error_graph=error_graph,
    )


def run_leap_experiment(config: LeapExperimentConfig):
    model = load_model()
    attn_saes = load_attn_saes()
    attn_saes = preprocess_attn_saes(attn_saes, model)
    hooked_mlp_transcoders = load_hooked_mlp_transcoders()

    # Sweep over datasets
    sweep_dir = ProjectDir / "results" / config.save_dir_prefix
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(asdict(config), f)

    for dataset_path in SELECTED_DATASETS:
        dataset_name = pathlib.Path(dataset_path).stem

        # Load the dataset
        train_loader, _ = load_datasets_from_json(
            model,
            ProjectDir / dataset_path,
            device=torch.device("cuda"),
            batch_size=config.batch_size,
            # NOTE: Do not specify total_dataset_size.
            # It leads to some weird indexing bug I don't understand
        )
        batch = next(iter(train_loader))

        # Load ablate cache
        if config.ablate_act_type == "unstructured":
            with open(ProjectDir / "data" / "c4_mean_acts.pkl", "rb") as file:
                ablate_cache = pickle.load(file)
        elif config.ablate_act_type == "structured":
            with open(
                ProjectDir / "data" / f"{dataset_name}_mean_acts.pkl", "rb"
            ) as file:
                ablate_cache = pickle.load(file)
        elif config.ablate_act_type == "tokenwise":
            with open(
                ProjectDir / "data" / f"{dataset_name}_tokenwise_acts.pkl", "rb"
            ) as file:
                ablate_cache = pickle.load(file)

        elif config.ablate_act_type == "corrupt":
            assert config.batch_size == 1
            _, ablate_cache = model.run_with_cache(batch.corrupt)
            ablate_cache.remove_batch_dim()
        else:
            raise ValueError(f"Unknown ablate_act_type: {config.ablate_act_type}")

        # Sweep over thresholds
        for threshold in THRESHOLDS:
            try:
                # Main script
                cfg = replace(config.leap_config, threshold=threshold)
                pprint(cfg)

                save_dir = sweep_dir / f"dataset={dataset_name}_threshold={threshold}"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save the config
                with open(save_dir / "config.json", "w") as f:
                    json.dump(asdict(cfg), f)

                print(f"Running experiment on dataset: {dataset_name}")
                leap_experiment_result = run_leap(
                    cfg,
                    batch,
                    model,
                    attn_saes,
                    hooked_mlp_transcoders,
                    ablate_cache,
                    save_dir=save_dir,
                    metric_fn_name=config.metric_fn_name,
                    feature_ablate_type=config.feature_ablate_type,  # type: ignore
                    error_ablate_type=config.error_ablate_type,  # type: ignore
                    first_ablated_layer=config.first_ablated_layer,
                )
                clear_memory()

                if leap_experiment_result is None:
                    continue

                with open(save_dir / "result.pkl", "wb") as f:
                    pickle.dump(leap_experiment_result, f)
            except Exception as e:
                print(e)
                clear_memory()
                continue


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    args = parser.parse_args()

    config = args.config
    print(config)
    run_leap_experiment(config)
