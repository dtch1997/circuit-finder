"""Script to run an experiment with LEAP

Usage:
pdm run python -m circuit_finder.experiments.run_leap_experiment [ARGS]

Run with flag '-h', '--help' to see the arguments.
"""

import transformer_lens as tl
import pandas as pd
import json

from simple_parsing import ArgumentParser
from dataclasses import dataclass
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.patching.ablate import get_metric_with_ablation
from circuit_finder.data_loader import load_datasets_from_json
from circuit_finder.constants import device
from tqdm import tqdm

from typing import Literal
from pathlib import Path
from circuit_finder.pretrained import (
    load_model,
    load_attn_saes,
    load_mlp_transcoders,
)
from circuit_finder.patching.leap import (
    preprocess_attn_saes,
    LEAP,
    LEAPConfig,
)
from circuit_finder.metrics import batch_avg_answer_diff

from circuit_finder.constants import ProjectDir


@dataclass
class LeapExperimentConfig:
    dataset_path: str = "datasets/ioi/ioi_vanilla_template_prompts.json"
    save_dir: str = "results/leap_experiment"
    seed: int = 1
    batch_size: int = 4
    total_dataset_size: int = 1024
    ablate_nodes: bool | str = "bm"
    ablate_errors: bool | str = "bm"
    ablate_tokens: Literal["clean", "corrupt"] = "clean"
    # NOTE: This specifies what to do with error nodes when calculating faithfulness curves.
    # Options are:
    # - ablate_errors = False  ->  we don't ablate error nodes
    # - ablate_errors = "bm"   ->  we batchmean-ablate error nodes
    # - ablate_errors = "zero" ->  we zero-ablate error nodes (warning: this gives v bad performance)

    first_ablate_layer: int = 2
    # NOTE:This specifies which layer to start ablating at.
    # Marks et al don't ablate the first 2 layers
    # TODO: Find reference for the above

    freeze_attention_pattern: bool = False

    verbose: bool = False


def run_leap_experiment(config: LeapExperimentConfig):
    # Define save dir
    save_dir = ProjectDir / config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save the config
    with open(save_dir / "config.json", "w") as jsonfile:
        json.dump(config.__dict__, jsonfile)

    # Load models
    model = load_model(requires_grad=True)
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
    train_loader, _ = load_datasets_from_json(
        model=model,
        path=dataset_path,
        device=device,  # type: ignore
        batch_size=config.batch_size,
        train_test_size=(config.total_dataset_size, config.total_dataset_size),
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
            # Config options
            ablate_nodes=config.ablate_nodes,
            ablate_errors=config.ablate_errors,
            first_ablated_layer=config.first_ablate_layer,
            freeze_attention=config.freeze_attention_pattern,
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
                # Config options
                ablate_nodes=config.ablate_nodes,
                ablate_errors=config.ablate_errors,
                first_ablated_layer=config.first_ablate_layer,
                freeze_attention=config.freeze_attention_pattern,
            ).item()

            # Log the data
            num_nodes_list.append(num_nodes)
            metrics_list.append(metric)

        faith = [(metric - floor) / (ceiling - floor) for metric in metrics_list]

        # Save the result as a dataframe
        data = pd.DataFrame(
            {
                "num_nodes": num_nodes_list,
                "faithfulness": faith,
                "metric": metrics_list,
            }
        )
        data["floor"] = floor
        data["ceiling"] = ceiling
        data.to_csv(batch_dir / "leap_experiment_results.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    args = parser.parse_args()

    run_leap_experiment(args.config)
