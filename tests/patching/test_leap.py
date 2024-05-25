from circuit_finder.core.hooked_sae_config import HookedSAEConfig
from circuit_finder.core.hooked_sae import HookedSAE
from circuit_finder.core.hooked_transcoder import (
    HookedTranscoder,
)
from circuit_finder.core.hooked_transcoder_config import HookedTranscoderConfig
from circuit_finder.patching.leap import LEAP, LEAPConfig

import pytest
import torch

from transformer_lens import HookedSAETransformer

MODEL = "solu-2l"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL)
    yield model
    model.reset_saes()


def get_site(act_name):
    site = act_name.split(".")[2:]
    site = ".".join(site)
    return site


def get_transcoder_config(model, act_name_in, act_name_out):
    site_to_size = {
        "ln2.hook_normalized": model.cfg.d_model,
        "hook_mlp_out": model.cfg.d_model,
    }
    site_in = get_site(act_name_in)
    assert site_in == "ln2.hook_normalized"
    d_in = site_to_size[site_in]

    site_out = get_site(act_name_out)
    d_out = site_to_size[site_out]

    return HookedTranscoderConfig(
        d_in=d_in,
        d_out=d_out,
        d_sae=d_in * 2,
        hook_name=act_name_in,
        hook_name_out=act_name_out,
    )


def get_attn_sae_config(model, act_name):
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]
    return HookedSAEConfig(d_in=d_in, d_sae=d_in * 2, hook_name=act_name)


@pytest.mark.xfail(reason="Unsure why not determinstic")
def test_leap(model, snapshot):
    config = LEAPConfig(threshold=0.01, contrast_pairs=False, chained_attribs=True)
    tokens = model.to_tokens(prompt)

    # NOTE: All attn saes and transcoders must be hooked in
    # Seed the random generation
    torch.manual_seed(0)
    attn_saes = {
        layer: HookedSAE(get_attn_sae_config(model, f"blocks.{layer}.attn.hook_z"))
        for layer in range(model.cfg.n_layers)
    }

    transcoders = {
        layer: HookedTranscoder(
            get_transcoder_config(
                model,
                f"blocks.{layer}.ln2.hook_normalized",
                f"blocks.{layer}.hook_mlp_out",
            )
        )
        for layer in range(model.cfg.n_layers)
    }

    def metric_fn(model, tokens):
        return model(tokens, return_type="loss")

    leap = LEAP(
        config,
        tokens,
        model,
        attn_saes=attn_saes,
        transcoders=transcoders,
        metric=metric_fn,
    )

    leap.metric_step()
    for layer in reversed(range(1, model.cfg.n_layers)):
        leap.mlp_step(layer)
        leap.ov_step(layer)

    graph = leap.graph
    assert graph == snapshot
