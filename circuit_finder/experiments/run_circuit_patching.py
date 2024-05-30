"""Script to plot ablation curves for multiple thresholds."""

import sys

sys.path.append("/workspace/circuit-finder")

import pandas as pd
import pickle
from circuit_finder.constants import ProjectDir
from circuit_finder.experiments.run_leap_experiment import (
    LeapExperimentResult,
    THRESHOLDS,
    datasets,
)

from circuit_finder.experiments.run_leap_experiment import (
    datasets,
    compute_logit_diff,
    load_models,
)

from circuit_finder.patching.ablate import (
    splice_model_with_saes_and_transcoders,
    filter_sae_acts_and_errors,
    get_ablation_result,
)


if __name__ == "__main__":
    model, attn_saes, transcoders = load_models()
    metric_fn_name = "logit_diff"

    for dataset_name, dataset in datasets.items():
        # Setup the tokens
        clean_tokens = dataset.clean_tokens(model)
        answer_tokens = dataset.answer_tokens(model)
        wrong_answer_tokens = dataset.wrong_answer_tokens(model)
        corrupt_tokens = dataset.corrupt_tokens(model)

        # Setup the metric function
        if metric_fn_name == "logit_diff":

            def metric_fn(model, tokens):
                logit_diff = compute_logit_diff(
                    model, tokens, answer_tokens, wrong_answer_tokens
                )
                return logit_diff.mean()
        else:
            raise ValueError(f"Unknown metric_fn_name: {metric_fn_name}")

        # Set up the dataframe
        rows = []

        for threshold in THRESHOLDS:
            print(
                "Processing dataset: ",
                dataset_name,
                " with threshold: ",
                threshold,
            )

            # Load the saved result
            result_fp = f"dataset={dataset_name}_threshold={threshold}.pkl"
            with open(ProjectDir / "leap_experiment_results" / result_fp, "rb") as file:
                result = pickle.load(file)

            # Get all nodes in the graph
            graph = result.graph
            all_nodes = graph.get_src_nodes()
            all_nodes = [n for n in all_nodes if "metric" not in n]

            # Get the clean and corrupt cache
            with splice_model_with_saes_and_transcoders(
                model, transcoders, attn_saes
            ) as spliced_model:
                _, clean_cache = model.run_with_cache(
                    clean_tokens, names_filter=filter_sae_acts_and_errors
                )
                _, corrupt_cache = model.run_with_cache(
                    corrupt_tokens, names_filter=filter_sae_acts_and_errors
                )

            # Get the noising ablation result
            noising_result = get_ablation_result(
                model,
                transcoders,
                attn_saes,
                clean_tokens=clean_tokens,
                corrupt_tokens=corrupt_tokens,
                clean_cache=clean_cache,
                corrupt_cache=corrupt_cache,
                nodes=all_nodes,
                metric_fn=metric_fn,
                setting="noising",
            )

            for c, m in zip(noising_result.coefficient, noising_result.metric):
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "threshold": threshold,
                        "n_nodes": len(all_nodes),
                        "setting": "noising",
                        "coefficient": c,
                        "metric": m,
                    }
                )

            # Get the denoising ablation result
            denoising_result = get_ablation_result(
                model,
                transcoders,
                attn_saes,
                clean_tokens=clean_tokens,
                corrupt_tokens=corrupt_tokens,
                clean_cache=clean_cache,
                corrupt_cache=corrupt_cache,
                nodes=all_nodes,
                metric_fn=metric_fn,
                setting="denoising",
            )

            for c, m in zip(denoising_result.coefficient, denoising_result.metric):
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "threshold": threshold,
                        "setting": "denoising",
                        "coefficient": c,
                        "metric": m,
                    }
                )

        df = pd.DataFrame(rows)
        # Save dataframe
        df_fp = f"dataset={dataset_name}.csv"
        df.to_csv(ProjectDir / "leap_experiment_results" / df_fp, index=False)
