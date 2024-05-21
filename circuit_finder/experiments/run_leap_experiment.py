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
from circuit_finder.data_loader import load_datasets_from_json
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.patching.leap import last_token_logit
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

from circuit_finder.constants import ProjectDir, device

from circuit_finder.patching.patched_fp import patched_fp


@dataclass
class LeapExperimentConfig:
    dataset_path: str = "datasets/ioi/ioi_prompts.json"
    save_dir: str = "results/leap_experiment"
    seed: int = 0
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


def run_leap_experiment(config: LeapExperimentConfig):
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
    train_loader, _ = load_datasets_from_json(
        model=model,
        path=dataset_path,
        device=device,
        batch_size=config.batch_size,
        train_test_size=(config.total_dataset_size, config.total_dataset_size),
        random_seed=config.seed,
    )

    # Define save dir
    save_dir = ProjectDir / config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(train_loader))
    clean_tokens = batch.clean
    corrupt_tokens = batch.corrupt

    # Save the dataset.
    with open(save_dir / "dataset.json", "w") as jsonfile:
        json.dump(
            {
                "clean": model.to_string(clean_tokens),
                "corrupt": model.to_string(corrupt_tokens),
            },
            jsonfile,
        )

    # NOTE: First, get the ceiling of the patching metric.
    model.reset_hooks()
    # TODO: Replace 'last_token_logit' with logit difference
    ceiling = last_token_logit(model, clean_tokens).item()

    # NOTE: Second, get floor of patching metric using empty graph, i.e. ablate everything
    empty_graph = EAPGraph([])
    floor = patched_fp(
        model,
        empty_graph,
        clean_tokens,
        last_token_logit,
        transcoders,
        attn_saes,
        ablate_errors=config.ablate_errors,  # if bm, error nodes are mean-ablated
        first_ablated_layer=config.first_ablate_layer,  # Marks et al don't ablate first 2 layers
    ).item()
    clear_memory()

    # now sweep over thresholds to get graphs with variety of numbers of nodes
    # for each graph we calculate faithfulness
    num_nodes_list = []
    metrics_list = []

    # TODO: make configurable
    for threshold in tqdm(
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.03, 0.06, 0.1]
    ):
        model.reset_hooks()
        cfg = LEAPConfig(
            threshold=threshold, contrast_pairs=False, chained_attribs=True
        )
        leap = LEAP(
            cfg,
            clean_tokens,
            model,
            attn_saes,
            transcoders,
            corrupt_tokens=corrupt_tokens,
        )
        leap.get_graph(verbose=False)
        graph = EAPGraph(leap.graph)

        # Save the graph
        with open(save_dir / f"leap-graph_threshold={threshold}.json", "w") as jsonfile:
            json.dump(graph.to_json(), jsonfile)

        num_nodes = len(graph.get_src_nodes())

        del leap
        clear_memory()

        metric = patched_fp(
            model,
            graph,
            clean_tokens,
            last_token_logit,
            transcoders,
            attn_saes,
            ablate_errors=config.ablate_errors,  # if bm, error nodes are mean-ablated
            first_ablated_layer=config.first_ablate_layer,  # Marks et al don't ablate first 2 layers
        )

        clear_memory()

        # Log the data
        num_nodes_list.append(num_nodes)
        metrics_list.append(metric.item())

    faith = [(metric - floor) / (ceiling - floor) for metric in metrics_list]

    # Save the result as a dataframe
    data = pd.DataFrame({"num_nodes": num_nodes_list, "faithfulness": faith})
    data.to_csv(save_dir / "leap_experiment_results.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    args = parser.parse_args()

    run_leap_experiment(args.config)
