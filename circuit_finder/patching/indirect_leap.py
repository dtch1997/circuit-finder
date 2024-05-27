
# %%
# Load models

from circuit_finder.pretrained import load_attn_saes, load_hooked_mlp_transcoders, load_model
from circuit_finder.core import HookedSAEConfig, HookedSAE, HookedTranscoderConfig, HookedTranscoder

model = load_model("solu-1l", device="cuda", requires_grad=True)
print(model)

# %%
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
saes = [
    HookedSAE(get_sae_config(model, "blocks.0.attn.hook_z")),
    HookedSAE(get_sae_config(model, "blocks.0.hook_resid_pre")),
    HookedSAE(get_sae_config(model, "blocks.0.hook_resid_post")),
]
transcoders = [
    HookedTranscoder(get_transcoder_config(model, "blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out")),
]

# attn_saes = load_attn_saes(use_error_term=True)
# transcoders = load_hooked_mlp_transcoders(use_error_term=True)

# %% 
# Compute metric 
prompt = "Hello world!"
logits = model(prompt)
prompt_tokens = model.to_tokens(prompt)
metric = logits[0, -1].sum()
metric.backward()

def ce_loss(model, tokens):
    """ Returns the sum of logits at the last token position. """
    return model(tokens, return_type = "loss")

# %%
# Define hooks to cache the gradients

from circuit_finder.core.types import MetricFn, Model, Tokens
import transformer_lens as tl 

GradientCache = tl.ActivationCache

def get_gradient_cache(
    model: Model, 
    tokens: Tokens,
    metric_fn: MetricFn,
) -> GradientCache:
    
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    def names_filter(x):
        return True
    
    bwd_hooks = []
    for hook_name in model.hook_dict: 
        if names_filter(hook_name):
            bwd_hooks.append((hook_name, backward_cache_hook))

    model.zero_grad()
    with model.hooks(
        bwd_hooks = bwd_hooks
    ):
        metric = metric_fn(model, tokens)
        metric.backward()

    return tl.ActivationCache(grad_cache, model)




grad_cache = get_gradient_cache(model, prompt_tokens, ce_loss)
print(len(grad_cache))

# %%
from contextlib import contextmanager

@contextmanager
def patch_model_gradients_from_cache(
    model: Model,
    grad_cache: GradientCache,
):
    def backward_patch_hook(act, hook):
        patched_act = grad_cache[hook.name]
        assert act.shape == patched_act.shape
        return (patched_act,)
    
    def names_filter(x):
        return x in grad_cache
    
    bwd_hooks = []
    for hook_name in model.hook_dict: 
        if names_filter(hook_name):
            bwd_hooks.append((hook_name, backward_patch_hook))

    with model.hooks(
        bwd_hooks = bwd_hooks
    ):
        yield model

    pass  

# %%
import torch
from circuit_finder.patching.ablate import splice_model_with_saes_and_transcoders

# del orig_grad_cache
# del spliced_grad_cache

model.zero_grad()
model.reset_saes()
model.reset_hooks()
orig_grad_cache = get_gradient_cache(model, prompt_tokens, ce_loss)


# Splice SAEs, transcoders into the forward pass
model.reset_saes()
model.reset_hooks()
model.zero_grad()
with splice_model_with_saes_and_transcoders(model, transcoders, saes):
    with patch_model_gradients_from_cache(model, orig_grad_cache):
        spliced_grad_cache = get_gradient_cache(model, prompt_tokens, ce_loss)


for key, spliced_grad in spliced_grad_cache.items():
    if "sae" in key: 
        continue
    if "mlp.mlp" in key:
        continue
    orig_grad = orig_grad_cache[key]
    norm = torch.norm(orig_grad - spliced_grad)
    if norm.item() > 1e-4:
        print(f"Key: {key}, norm: {norm}")
    # assert torch.allclose(orig_grad, spliced_grad, atol=1e-6), f"Key: {key}"
# %%
