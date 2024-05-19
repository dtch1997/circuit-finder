"""Check that the LEAP algorithm returns a given circuit"""

import pytest
import transformer_lens as tl

from circuit_finder.pretrained import (
    load_attn_saes,
    load_mlp_transcoders,
)
from circuit_finder.patching.leap import (
    preprocess_attn_saes,
    LEAP,
    LEAPConfig,
)

from circuit_finder.constants import device


@pytest.mark.skip
def test_leap_e2e(snapshot):
    # Load models
    model = tl.HookedTransformer.from_pretrained(
        "gpt2",
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )

    attn_saes = load_attn_saes()
    attn_saes = preprocess_attn_saes(attn_saes, model)  # type: ignore
    transcoders = load_mlp_transcoders()

    # Load dataset
    tokens = model.to_tokens(
        ["When John and Mary were at the store, John gave a bottle to Mary"]
    )
    corrupt_tokens = model.to_tokens(
        ["When Alice and Bob were at the store, Alice gave a bottle to Bob"]
    )

    # Set up LEAP
    cfg = LEAPConfig(threshold=0.2, contrast_pairs=True)
    leap = LEAP(
        cfg, tokens, model, attn_saes, transcoders, corrupt_tokens=corrupt_tokens
    )

    # Run LEAP
    leap.metric_step()
    for layer in reversed(range(1, model.cfg.n_layers)):
        leap.mlp_step(layer)
        leap.ov_step(layer)

    graph = leap.graph
    assert graph == snapshot
