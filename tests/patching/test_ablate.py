from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.patching.ablate import (
    splice_model_with_saes_and_transcoders,
    get_ablation_hooks,
    add_ablation_hooks_to_model,
)
from circuit_finder.core import (
    HookedSAE,
    HookedSAEConfig,
    HookedTranscoder,
    HookedTranscoderConfig,
)

import torch
import pytest

from transformer_lens import HookedSAETransformer

MODEL = "solu-1l"
prompt = "Hello World!"


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL)
    yield model
    model.reset_saes()


def get_site(act_name):
    site = act_name.split(".")[3:]
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
        use_error_term=True,
    )


def get_sae_config(model, act_name):
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]
    return HookedSAEConfig(
        d_in=d_in, d_sae=d_in * 2, hook_name=act_name, use_error_term=True
    )


# @pytest.mark.xfail
@pytest.mark.parametrize(
    "expected_norm",
    [1e-3, pytest.param(1e-6, marks=pytest.mark.xfail(reason="Unsure why fails"))],
)
def test_wrap_model_with_saes_and_transcoders(model, expected_norm):
    sae_config = get_sae_config(model, "blocks.0.attn.hook_z")
    sae = HookedSAE(sae_config)
    transcoder_config = get_transcoder_config(
        model,
        "blocks.0.mlp.ln2.hook_normalized",
        "blocks.0.mlp.hook_mlp_out",
    )
    transcoder = HookedTranscoder(transcoder_config)
    assert sae.cfg.use_error_term
    assert transcoder.cfg.use_error_term

    tokens = model.to_tokens(["Hello world"])
    orig_logits = model(tokens)
    with splice_model_with_saes_and_transcoders(model, [transcoder], [sae]):
        spliced_logits, cache = model.run_with_cache(tokens)

    assert orig_logits.shape == spliced_logits.shape
    assert torch.allclose(
        orig_logits, spliced_logits, atol=expected_norm
    ), f"Original and patched logits differ by L2 norm of {torch.linalg.norm(orig_logits - spliced_logits)}"

    expected_hook_names = [
        # Attention
        "blocks.0.attn.hook_z.hook_sae_input",
        "blocks.0.attn.hook_z.hook_sae_acts_pre",
        "blocks.0.attn.hook_z.hook_sae_acts_post",
        "blocks.0.attn.hook_z.hook_sae_recons",
        "blocks.0.attn.hook_z.hook_sae_output",
        "blocks.0.attn.hook_z.hook_sae_error",
        # MLP
        # Due to the weird way we had to hack this together, the first four are different
        "blocks.0.mlp.transcoder.hook_sae_input",
        "blocks.0.mlp.transcoder.hook_sae_recons",
        "blocks.0.mlp.transcoder.hook_sae_acts_pre",
        "blocks.0.mlp.transcoder.hook_sae_acts_post",
        "blocks.0.mlp.hook_sae_output",
        "blocks.0.mlp.hook_sae_error",
    ]

    for hook_name in expected_hook_names:
        assert hook_name in cache, "Missing hook: " + hook_name


def test_apply_ablation_hooks_matches_add_ablation_hooks_to_model(model):
    sae_config = get_sae_config(model, "blocks.0.attn.hook_z")
    sae = HookedSAE(sae_config)
    transcoder_config = get_transcoder_config(
        model,
        "blocks.0.mlp.ln2.hook_normalized",
        "blocks.0.mlp.hook_mlp_out",
    )
    transcoder = HookedTranscoder(transcoder_config)
    tokens = model.to_tokens(prompt)

    empty_graph = EAPGraph()

    with splice_model_with_saes_and_transcoders(model, [transcoder], [sae]):
        _, cache = model.run_with_cache(tokens)
        hooks = get_ablation_hooks(
            empty_graph,
            cache,
            ablate_errors="zero",
            ablate_nodes="zero",
            freeze_attention=False,
            first_ablated_layer=0,
        )
        with model.hooks(fwd_hooks=hooks):
            patched_logits_A = model(tokens)

    model.reset_hooks()
    add_ablation_hooks_to_model(
        model,
        empty_graph,
        tokens,
        [transcoder],
        [sae],
        ablate_errors="zero",
        ablate_nodes="zero",
        freeze_attention=False,
        first_ablated_layer=0,
    )
    patched_logits_B = model(tokens)

    assert torch.allclose(patched_logits_A, patched_logits_B, atol=1e-6)
