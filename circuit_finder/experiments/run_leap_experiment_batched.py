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
import pandas as pd
import json
import torch
import transformer_lens as tl

from simple_parsing import ArgumentParser
from dataclasses import dataclass
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.patching.ablate import get_metric_with_ablation
from circuit_finder.data_loader import load_datasets_from_json, PromptPairBatch
from circuit_finder.constants import device
from tqdm import tqdm

from typing import Literal
from eindex import eindex
from pathlib import Path
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
)
from circuit_finder.plotting import make_html_graph

THRESHOLDS = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06]


def batch_to_str_dict(batch: PromptPairBatch, model):
    return {
        "clean": model.to_string(batch.clean),
        "answer": model.to_string(batch.answers),
        "wrong_answer": model.to_string(batch.wrong_answers),
        "corrupt": model.to_string(batch.corrupt),
    }


@dataclass(frozen=True)
class CleanCorruptExample:
    clean_prompt: str
    answer: str
    wrong_answer: str
    corrupt_prompt: str

    def clean_tokens(self, model: Model):
        return model.to_tokens(self.clean_prompt)

    def answer_tokens(self, model: Model):
        return model.to_tokens(self.answer, prepend_bos=False).squeeze(0)

    def wrong_answer_tokens(self, model: Model):
        return model.to_tokens(self.wrong_answer, prepend_bos=False).squeeze(0)

    def corrupt_tokens(self, model: Model):
        return model.to_tokens(self.corrupt_prompt)


datasets: dict[str, CleanCorruptExample] = {
    # Datasets reported in "Have Faith in Faithfulness" (Hanna et al, 2023)
    "gender-bias": CleanCorruptExample(
        clean_prompt="The doctor is ready, you can go see",
        answer=" him",
        wrong_answer=" her",
        corrupt_prompt="The nurse is ready, you can go see",
    ),
    # TODO: Subject-verb Agreement
    "ioi": CleanCorruptExample(
        clean_prompt="When John and Mary went to the shop, John gave a bottle to",
        answer=" Mary",
        wrong_answer=" John",
        corrupt_prompt="When Alice and Bob went to the shop, Charlie gave a bottle to",
    ),
    # TODO: Hypernymy
    # TODO: Greater-than
    # Datasets Jacob came up with himself
    "for-loop": CleanCorruptExample(
        clean_prompt="for y in x: print(",
        answer="y",
        wrong_answer="x",
        corrupt_prompt="for x in y: print(",
    ),
    # "if-else": CleanCorruptExample(
    #     clean_prompt="if x>5: print(x)",
    #     answer="else",
    #     wrong_answer="if",
    #     corrupt_prompt="while x>5: print(x)",
    # ),
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


def get_clean_and_corrupt_metric(
    model,
    metric_fn,
    clean_tokens,
    corrupt_tokens,
):
    with torch.no_grad():
        clean_metric = metric_fn(model, clean_tokens).item()
        corrupt_metric = metric_fn(model, corrupt_tokens).item()

    return clean_metric, corrupt_metric


@dataclass(frozen=True)
class LeapExperimentResult:
    config: LEAPConfig
    clean_metric: float
    corrupt_metric: float
    graph: EAPGraph
    error_graph: EAPGraph


def run_leap(
    leap_config: LEAPConfig,
    batch: PromptPairBatch,
    model: Model,
    attn_saes,
    hooked_mlp_transcoders,
    *,
    save_dir: pathlib.Path,
    metric_fn_name: str = "logit_diff",
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
    return

    # Setup the metric function
    if metric_fn_name == "logit_diff":

        def metric_fn(model, tokens):
            # Get the last-token logits
            logits = model(tokens)[:, -1, :]
            logit_diff = batch_avg_answer_diff(logits, batch)
            return logit_diff.mean()
    else:
        raise ValueError(f"Unknown metric_fn_name: {metric_fn_name}")

    clean_metric, corrupt_metric = get_clean_and_corrupt_metric(
        model, metric_fn, clean_tokens, corrupt_tokens
    )

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

    make_html_graph(leap, html_file=str(save_dir.absolute() / "graph.html"))

    return LeapExperimentResult(
        config=leap_config,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        graph=graph,
        error_graph=error_graph,
    )


if __name__ == "__main__":
    from circuit_finder.data.ioi import ABBA_DATASETS, BABA_DATASETS

    IOI_DATASETS = ABBA_DATASETS[:2] + BABA_DATASETS[:2]

    model = load_model()
    attn_saes = load_attn_saes()
    attn_saes = preprocess_attn_saes(attn_saes, model)
    hooked_mlp_transcoders = load_hooked_mlp_transcoders()
    metric_fn_name = "logit_diff"
    batch_size = 4
    cfg = LEAPConfig(
        threshold=0.001,
        contrast_pairs=True,
        qk_enabled=True,
        chained_attribs=True,
        allow_neg_feature_acts=True,
        store_error_attribs=True,
    )

    for dataset_path in IOI_DATASETS:
        dataset_name = pathlib.Path(dataset_path).stem
        train_loader, test_loader = load_datasets_from_json(
            model,
            ProjectDir / dataset_path,
            device=torch.device("cuda"),
            batch_size=batch_size,
            train_test_size=(batch_size, batch_size),
        )
        batch = next(iter(train_loader))
        for threshold in THRESHOLDS:
            cfg.threshold = threshold
            save_dir = (
                ProjectDir
                / "leap_experiment_results"
                / f"dataset={dataset_name}_threshold={threshold}_batch-size={batch_size}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running experiment on dataset: {dataset_name}")
            leap_experiment_result = run_leap(
                cfg,
                batch,
                model,
                attn_saes,
                hooked_mlp_transcoders,
                save_dir=save_dir,
                metric_fn_name=metric_fn_name,
            )

            if leap_experiment_result is None:
                continue

            print(f"Clean Metric: {leap_experiment_result.clean_metric}")
            print(f"Corrupt Metric: {leap_experiment_result.corrupt_metric}")

            with open(save_dir / "result.pkl", "wb") as f:
                pickle.dump(leap_experiment_result, f)
