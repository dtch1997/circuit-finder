from circuit_finder.core.hooked_transcoder import (
    HookedTranscoder,
    HookedTranscoderWrapper,
    HookedTranscoderReplacementContext,
)
from circuit_finder.core.hooked_transcoder_config import HookedTranscoderConfig


import einops
import pytest
import torch

from transformer_lens import HookedSAETransformer

MODEL = "solu-1l"
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


def test_forward_reconstructs_input(model):
    """Verfiy that the HookedSAE returns an output with the same shape as the input activations."""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    hooked_tc = HookedTranscoder(tc_cfg)

    _, cache = model.run_with_cache(prompt)
    x_in = cache[act_in]
    x_out = cache[act_out]

    sae_output = hooked_tc(x_in)
    assert sae_output.shape == x_out.shape


def test_run_with_cache(model):
    """Verifies that run_with_cache caches SAE activations"""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    tc_cfg.use_error_term = True
    hooked_tc = HookedTranscoder(tc_cfg)

    expected_hook_names = [
        "transcoder.hook_sae_input",
        "transcoder.hook_sae_recons",
        "transcoder.hook_sae_acts_pre",
        "transcoder.hook_sae_acts_post",
        "hook_sae_error",
        "hook_sae_output",
    ]

    with HookedTranscoderReplacementContext(model, [hooked_tc]):
        _, cache = model.run_with_cache(prompt)

    for hook_name in expected_hook_names:
        assert "blocks.0.mlp." + hook_name in cache


def test_error_term(model):
    """Verifies that that if we use error_terms, HookedSAE returns an output that is equal to the input activations."""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    tc_cfg.use_error_term = True
    hooked_tc = HookedTranscoder(tc_cfg)

    _, orig_cache = model.run_with_cache(prompt)

    # Test hooked transcoder wrapper
    wrapped_hook_tc = HookedTranscoderWrapper(hooked_tc, model.blocks[0].mlp)

    in_orig = orig_cache[act_in]
    out_orig = orig_cache[act_out]
    sae_out = wrapped_hook_tc(in_orig)
    assert sae_out.shape == out_orig.shape
    assert torch.allclose(sae_out, out_orig, atol=1e-6)

    # Test replacement context
    with HookedTranscoderReplacementContext(model, [hooked_tc]):
        _, spliced_cache = model.run_with_cache(prompt)

    out_orig = orig_cache[act_out]
    out_spliced = spliced_cache[act_out]

    assert out_orig.shape == out_spliced.shape
    assert torch.allclose(out_orig, out_spliced, atol=1e-6)


# def test_error_grads(model, act_name):
#     """Verifies that if we use error terms, the error term has a gradient"""
#     sae_cfg = get_sae_config(model, act_name)
#     sae_cfg.use_error_term = True
#     hooked_sae = HookedSAE(sae_cfg)

#     _, cache = model.run_with_cache(prompt, names_filter=act_name)
#     x = cache[act_name]

#     grad_cache = {}
#     hooked_sae.reset_hooks()

#     def backward_cache_hook(act, hook):
#         grad_cache[hook.name] = act.detach()

#     hooked_sae.add_hook("hook_sae_error", backward_cache_hook, "bwd")

#     sae_output = hooked_sae(x)
#     assert sae_output.shape == x.shape
#     assert torch.allclose(sae_output, x, atol=1e-6)

#     value = sae_output.sum()
#     value.backward()
#     hooked_sae.reset_hooks()

#     assert len(grad_cache) == 1
#     assert "hook_sae_error" in grad_cache

#     # NOTE: The output is linear in the error, hence analytic gradient is one
#     grad = grad_cache["hook_sae_error"]
#     analytic_grad = torch.ones_like(grad)
#     assert torch.allclose(grad, analytic_grad, atol=1e-6)


# def test_feature_grads_with_error_term(model, act_name):
#     """Verifies that pytorch backward computes the correct feature gradients when using error_terms. Motivated by the need to compute feature gradients for attribution patching."""

#     # Load SAE
#     sae_cfg = get_sae_config(model, act_name)
#     sae_cfg.use_error_term = True
#     hooked_sae = HookedSAE(sae_cfg)

#     # Get input activations
#     _, cache = model.run_with_cache(prompt, names_filter=act_name)
#     x = cache[act_name]

#     # Cache gradients with respect to feature acts
#     hooked_sae.reset_hooks()
#     grad_cache = {}

#     def backward_cache_hook(act, hook):
#         grad_cache[hook.name] = act.detach()

#     hooked_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")
#     hooked_sae.add_hook("hook_sae_output", backward_cache_hook, "bwd")

#     sae_output = hooked_sae(x)
#     assert torch.allclose(sae_output, x, atol=1e-6)
#     value = sae_output.sum()
#     value.backward()
#     hooked_sae.reset_hooks()

#     # Compute gradient analytically
#     if act_name.endswith("hook_z"):
#         reshaped_output_grad = einops.rearrange(
#             grad_cache["hook_sae_output"], "... n_heads d_head -> ... (n_heads d_head)"
#         )
#         analytic_grad = reshaped_output_grad @ hooked_sae.W_dec.T
#     else:
#         analytic_grad = grad_cache["hook_sae_output"] @ hooked_sae.W_dec.T

#     # Compare analytic gradient with pytorch computed gradient
#     assert torch.allclose(grad_cache["hook_sae_acts_post"], analytic_grad, atol=1e-6)
