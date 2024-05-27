import torch
import pytest

from circuit_finder.core import HookedSAE, HookedSAEConfig, HookedTranscoder, HookedTranscoderConfig
from circuit_finder.pretrained import load_model
from circuit_finder.patching.ablate import splice_model_with_saes_and_transcoders
from circuit_finder.patching.gradient_cache import get_gradient_cache, patch_model_gradients_from_cache


@pytest.fixture
def model():
    model = load_model("solu-1l", requires_grad=True)
    return model

@pytest.fixture
def prompt():
    return "Hello World!"

def get_sae_config(model, act_name):
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_resid_post": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]
    return HookedSAEConfig(d_in=d_in, d_sae=d_in * 2, hook_name=act_name, use_error_term=True)

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
        use_error_term=True
    )

def ce_loss(model, tokens):
    """ Returns the sum of logits at the last token position. """
    return model(tokens, return_type = "loss")


def test_patch_model_with_gradient_cache(model, prompt):

    saes = [
        HookedSAE(get_sae_config(model, "blocks.0.attn.hook_z")),
        HookedSAE(get_sae_config(model, "blocks.0.hook_resid_pre")),
        HookedSAE(get_sae_config(model, "blocks.0.hook_resid_post")),
    ]
    transcoders = [
        HookedTranscoder(get_transcoder_config(model, "blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out")),
    ]

    prompt_tokens = model.to_tokens(prompt)


    orig_grad_cache = get_gradient_cache(model, prompt_tokens, ce_loss)
    with splice_model_with_saes_and_transcoders(model, transcoders, saes):
        with patch_model_gradients_from_cache(model, orig_grad_cache):
            spliced_grad_cache = get_gradient_cache(model, prompt_tokens, ce_loss)

    for key, spliced_grad in spliced_grad_cache.items():
        if "sae" in key: 
            continue
        if "mlp.mlp" in key:
            continue
        if "embed" in key:
            continue
        orig_grad = orig_grad_cache[key]
        norm = torch.norm(orig_grad - spliced_grad)
        assert norm < 1e-4, "key: {}, norm: {}".format(key, norm)