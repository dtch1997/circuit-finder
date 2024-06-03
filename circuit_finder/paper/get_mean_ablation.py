# %%
import sys

sys.path.append("/workspace/circuit-finder")

# %%
from circuit_finder.pretrained import (
    load_model,
    load_attn_saes,
    load_hooked_mlp_transcoders,
)
from circuit_finder.patching.indirect_leap import preprocess_attn_saes

model = load_model()
attn_saes = load_attn_saes()
attn_saes = preprocess_attn_saes(attn_saes, model)
hooked_mlp_transcoders = load_hooked_mlp_transcoders()

transcoders = list(hooked_mlp_transcoders.values())
saes = list(attn_saes.values())

# %% [markdown]
# # C4

# %%
from datasets import load_dataset

dataset = load_dataset("c4", "en", streaming=True)

# %%
import torch
from transformer_lens import ActivationCache
from circuit_finder.patching.ablate import (
    splice_model_with_saes_and_transcoders,
    filter_sae_acts_and_errors,
)


def get_hf_cache(dataset):
    n_tokens = 0
    total_tokens = 100_000  # 100k tokens
    # total_tokens = 100
    print(f"Total tokens: {total_tokens}")

    # A bit of a hack, run once to get cache shapes
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        _, dummy_cache = model.run_with_cache(
            "Hello World", names_filter=filter_sae_acts_and_errors
        )

    zero_cache_dict = {
        hook_name: torch.zeros_like(act.sum(1).squeeze(0))
        for hook_name, act in dummy_cache.items()
    }
    # zero_cache = ActivationCache(zero_cache_dict, model)

    # Run the model
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        for element in dataset["train"]:
            text = element["text"]
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(
                text, names_filter=filter_sae_acts_and_errors
            )

            n_tokens += tokens.shape[1]
            for hook_name, act in cache.items():
                zero_cache_dict[hook_name] += act.sum(1).squeeze(0)

            if n_tokens >= total_tokens:
                break

    # Average the cache
    for hook_name, act in zero_cache_dict.items():
        zero_cache_dict[hook_name] /= n_tokens

    zero_cache = ActivationCache(zero_cache_dict, model)
    return zero_cache


# %%
cache = get_hf_cache(dataset)

# %%
for hook_name, act in cache.items():
    print(hook_name, act.shape)

# %%
# Save the cache

import pickle
from circuit_finder.constants import ProjectDir

data_dir = ProjectDir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
with open(ProjectDir / "data" / "c4_mean_acts.pkl", "wb") as file:
    pickle.dump(cache, file)

del cache

# %% [markdown]
# # Auto-Circuit Datasets

# %%
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
from circuit_finder.patching.ablate import get_metric_with_ablation

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
    AblateType,
)

from circuit_finder.experiments.run_dataset_sweep import SELECTED_DATASETS

batch_size = 8
print(ALL_DATASETS)


# %%
import torch
from transformer_lens import ActivationCache
from circuit_finder.patching.ablate import (
    splice_model_with_saes_and_transcoders,
    filter_sae_acts_and_errors,
)


def get_cache(train_loader):
    n_tokens = 0
    total_tokens = 100_000  # 100k tokens
    # total_tokens = 1c00
    print(f"Total tokens: {total_tokens}")

    # A bit of a hack, run once to get cache shapes
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        _, dummy_cache = model.run_with_cache(
            "Hello World", names_filter=filter_sae_acts_and_errors
        )

    zero_cache_dict = {
        hook_name: torch.zeros_like(act.sum(1).squeeze(0))
        for hook_name, act in dummy_cache.items()
    }
    # zero_cache = ActivationCache(zero_cache_dict, model)

    # Run the model
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        for batch in train_loader:
            tokens = batch.clean
            _, cache = model.run_with_cache(
                tokens, names_filter=filter_sae_acts_and_errors
            )

            n_tokens += tokens.shape[1] * tokens.shape[0]
            for hook_name, act in cache.items():
                zero_cache_dict[hook_name] += act.sum(1).sum(0)

            if n_tokens >= total_tokens:
                break

    print(n_tokens)

    # Average the cache
    for hook_name, act in zero_cache_dict.items():
        zero_cache_dict[hook_name] /= n_tokens

    zero_cache = ActivationCache(zero_cache_dict, model)
    return zero_cache


# %%
for dataset_path in ALL_DATASETS:
    print("Processing", dataset_path)
    train_loader, _ = load_datasets_from_json(
        model,
        ProjectDir / dataset_path,
        device=torch.device("cuda"),
        batch_size=batch_size,
    )
    cache = get_cache(train_loader)
    with open(
        ProjectDir / "data" / f"{pathlib.Path(dataset_path).stem}_mean_acts.pkl", "wb"
    ) as file:
        pickle.dump(cache, file)

    del cache

# %% [markdown]
# # Auto-Circuit Datasets, Tokenwise
# %%

import torch
from transformer_lens import ActivationCache
from circuit_finder.patching.ablate import (
    splice_model_with_saes_and_transcoders,
    filter_sae_acts_and_errors,
)


def get_cache(train_loader):
    n_tokens = 0
    n_examples = 0
    total_tokens = 100_000  # 100k tokens
    # total_tokens = 100
    print(f"Total tokens: {total_tokens}")

    # A bit of a hack, run once to get cache shapes
    batch = next(iter(train_loader))
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        _, dummy_cache = model.run_with_cache(
            batch.clean, names_filter=filter_sae_acts_and_errors
        )

    zero_cache_dict = {
        hook_name: torch.zeros_like(act.sum(0))
        for hook_name, act in dummy_cache.items()
    }

    # Run the model
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        for batch in train_loader:
            tokens = batch.clean
            _, cache = model.run_with_cache(
                tokens, names_filter=filter_sae_acts_and_errors
            )

            n_tokens += tokens.shape[1] * tokens.shape[0]
            n_examples += tokens.shape[0]
            for hook_name, act in cache.items():
                zero_cache_dict[hook_name] += act.sum(0)

            if n_tokens >= total_tokens:
                break

    print(n_tokens)

    # Average the cache
    for hook_name, act in zero_cache_dict.items():
        zero_cache_dict[hook_name] /= n_examples

    zero_cache = ActivationCache(zero_cache_dict, model)
    return zero_cache


# %%
for dataset_path in SELECTED_DATASETS:
    print("Processing", dataset_path)
    train_loader, _ = load_datasets_from_json(
        model,
        ProjectDir / dataset_path,
        device=torch.device("cuda"),
        batch_size=batch_size,
    )
    cache = get_cache(train_loader)
    with open(
        ProjectDir / "data" / f"{pathlib.Path(dataset_path).stem}_tokenwise_acts.pkl",
        "wb",
    ) as file:
        pickle.dump(cache, file)

# %%
for hook_name, act in cache.items():
    print(hook_name, act.shape)

del cache

# %%
