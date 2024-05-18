# %%
"""Demonstrate how to do linear edge attribution patching using LEAP"""

import transformer_lens as tl

from torch import Tensor
from jaxtyping import Int
from typing import Callable

from circuit_finder.pretrained import (
    load_attn_saes,
    load_mlp_transcoders,
)
from circuit_finder.patching.leap import (
    preprocess_attn_saes,
    LEAP,
    LEAPConfig,
)

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

print(len(attn_saes))
print(len(transcoders))
print(transcoders.keys())
print(attn_saes.keys())

MetricFn = Callable[[tl.HookedTransformer, Int[Tensor, "batch seq"]], Tensor]

# Load dataset
tokens = model.to_tokens(
    ["When John and Mary were at the store, John gave a bottle to Mary"]
)
corrupt_tokens = model.to_tokens(
    ["When Alice and Bob were at the store, Alice gave a bottle to Bob"]
)

# Set up LEAP
cfg = LEAPConfig(threshold=0.2, contrast_pairs=True)
leap = LEAP(cfg, tokens, model, attn_saes, transcoders, corrupt_tokens=corrupt_tokens)

# Run LEAP
leap.metric_step()
print("num edges = ", len(leap.graph))

for layer in reversed(range(1, model.cfg.n_layers)):
    print("layer : ", layer)
    leap.mlp_step(layer)
    print("num edges = ", len(leap.graph))
    leap.ov_step(layer)
    print("num edges = ", len(leap.graph))
    print()

# Inspect graph found via LEAP
attn_attn = [
    (edge, attrib)
    for (edge, attrib) in leap.graph[1:]
    if edge[0].startswith("attn") and edge[1].startswith("attn")
]
attn_mlp = [
    (edge, attrib)
    for (edge, attrib) in leap.graph[1:]
    if edge[0].startswith("attn") and edge[1].startswith("mlp")
]
mlp_attn = [
    (edge, attrib)
    for (edge, attrib) in leap.graph[1:]
    if edge[0].startswith("mlp") and edge[1].startswith("attn")
]
mlp_mlp = [
    (edge, attrib)
    for (edge, attrib) in leap.graph[1:]
    if edge[0].startswith("mlp") and edge[1].startswith("mlp")
]

print(len(attn_attn), len(attn_mlp), len(mlp_attn), len(mlp_mlp))
# %%
# Print edges found via LEAP
print(len(leap.graph))
for edge in leap.graph:
    print(edge)


#%% 
leap.graph
#%%
from circuit_finder.neuronpedia import get_neuronpedia_url_for_quick_list
get_neuronpedia_url_for_quick_list(layer=5,
                                   features=[6255],
                                   sae_family = "tres-dc")

# %%
