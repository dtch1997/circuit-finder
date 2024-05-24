'''Do the LEAP attribution estimates match what we get when we manually ablate?
Note: our prior should be that our estimates are bad, since we're approximating
nodes as only having an effect on the metric via some very small number of pathways.
In reality, changing a node will also change loads of other downstream nodes that 
we didn't include in our circuit (notably, error nodes!)'''

# %%  Imports and Downloads
import sys
sys.path.append("/root/circuit-finder")
print(sys.path)

import transformer_lens as tl
from torch import Tensor
from jaxtyping import Int
from typing import Callable
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.plotting import show_attrib_graph
import torch
import gc
from circuit_finder.patching.leap import last_token_logit
from tqdm import tqdm
import plotly.express as px
from einops import *

from circuit_finder.pretrained import (
    load_attn_saes,
    load_mlp_transcoders,

)
from circuit_finder.patching.leap import (
    preprocess_attn_saes,
    LEAP,
    LEAPConfig,
)

from circuit_finder.patching.patched_fp import patched_fp

# Load models
model = tl.HookedTransformer.from_pretrained(
    "gpt2",
    device="cuda",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)

clean_model_clone = tl.HookedTransformer.from_pretrained(
    "gpt2",
    device="cuda",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)

attn_saes = load_attn_saes()
attn_saes = preprocess_attn_saes(attn_saes, model)  # type: ignore
transcoders = load_mlp_transcoders()

#%% Define tokens and run LEAP to get graph
tokens = model.to_tokens(
    ["When John and Mary were at the store, John gave a bottle to Mary"])

model.reset_hooks()
cfg = LEAPConfig(threshold=0.01, 
                contrast_pairs=False, 
                chained_attribs=True)
leap = LEAP(cfg, tokens, model, attn_saes, transcoders, corrupt_tokens=None)
leap.get_graph(verbose=False)
print(len(leap.graph))

#%% Cache clean layernorm scales and attention patterns, so we can freeze them
model.reset_hooks()
loss, cache = model.run_with_cache(
    tokens, 
    return_type=None,
    names_filter = lambda x: (x.endswith("pattern") or x.endswith("scale"))
)

def freeze_pattern_hook(act, hook):
    assert hook.name.endswith('pattern')
    return cache[hook.name]

#%%
# Set these as desired
freeze_pattern = True
freeze_layernorm = True

for up_module_name in ["attn", "mlp"]:
    for down_module_name in ["attn", "mlp"]:

        # get clean metric for comparison
        model.reset_hooks()
        clean_m = leap.metric(model, tokens)

        good_edge_attribs = [(edge, vals) for (edge, vals) in leap.graph 
                    if (edge[0].startswith(down_module_name) and edge[1].startswith(up_module_name))]

        true_attribs = []
        approx_attribs = []
        true_nn_attribs = []
        approx_nn_attribs = []

        for (edge, vals) in tqdm(good_edge_attribs):

            down_module, down_layer, down_pos, down_feature_id = edge[0].split('.')
            down_layer, down_pos, down_feature_id = int(down_layer), int(down_pos), int(down_feature_id)
            
            up_module, up_layer, up_pos, up_feature_id = edge[1].split('.')
            up_layer, up_pos, up_feature_id = int(up_layer), int(up_pos), int(up_feature_id)
            
            # Get the change in the output of the upstream head, when we ablate the upstream node
            up_out_diff = {}
            if up_module == "mlp":
                up_hook_pt = f'blocks.{up_layer}.ln2.hook_normalized'
                up_sae = transcoders[up_layer]
                def up_cache_hook(act, hook):
                    assert hook.name.endswith('ln2.hook_normalized')
                    up_facts = torch.relu(
                        (act[:, up_pos] - up_sae.b_dec) @ up_sae.W_enc[:, up_feature_id] 
                        + up_sae.b_enc[up_feature_id]) #  [b]
                    up_out_diff[0] = einsum(up_facts, 
                                    up_sae.W_dec[up_feature_id, :],
                                    "b, d -> b d")
                    
            if up_module == "attn":
                up_hook_pt = f'blocks.{up_layer}.attn.hook_z'
                up_sae = attn_saes[up_layer]
                def up_cache_hook(act, hook):
                    # hook.name = hook_z
                    concat = rearrange(act[:, up_pos], "b n_heads d_head -> b (n_heads d_head)")
                    up_facts = torch.relu(
                        (concat - up_sae.b_dec) @ up_sae.W_enc[:, up_feature_id] + up_sae.b_enc[up_feature_id]) #  [b]
                    W_dec_z = rearrange(up_sae.W_dec[up_feature_id,:],
                                            "(n_heads d_head) -> n_heads d_head",
                                            n_heads = model.cfg.n_heads)
                    W_dec_resid = einsum(W_dec_z , model.W_O[up_layer],
                                        "n_heads d_head, n_heads d_head d_model -> d_model")
                    up_out_diff[0] = einsum(up_facts, 
                                    W_dec_resid,
                                    "b, d -> b d")
            
            # Take the upstream change that we found, and feed it into the downstream head
            # to work out the change in the output of the downstream head
            down_out_diff = {}
            if down_module == "mlp":
                down_sae = transcoders[down_layer]
                down_in_pt  = f'blocks.{down_layer}.hook_resid_mid'
                down_out_pt = f'blocks.{down_layer}.hook_mlp_out'
            
                def down_cache_hook(act, hook):
                    clean_normed_input = clean_model_clone.blocks[down_layer].ln2(act)
                    down_clean_facts = torch.relu(
                        (clean_normed_input - down_sae.b_dec) @ down_sae.W_enc[:, down_feature_id] 
                        + down_sae.b_enc[down_feature_id]) # [b s]

                    ablated_input = act.clone()
                    ablated_input[:, up_pos] -= up_out_diff[0]
                    if freeze_layernorm:
                        ablated_normed_input = ablated_input  / cache[f'blocks.{down_layer}.ln2.hook_scale']
                    else:
                        ablated_normed_input = clean_model_clone.blocks[down_layer].ln2(ablated_input)

                    down_ablated_facts = torch.relu(
                        (ablated_normed_input - down_sae.b_dec) @ down_sae.W_enc[:, down_feature_id] 
                        + down_sae.b_enc[down_feature_id]) # [b s]
                    
                    down_facts_diff = down_ablated_facts - down_clean_facts # [b s]

                    global true_nn_attribs, approx_nn_attribs
                    true_nn_attribs += [down_facts_diff[:,down_pos].mean().cpu().item()]
                    approx_nn_attribs += [vals[1].cpu().item()]

                    down_out_diff[0] = einsum(down_facts_diff,
                                                down_sae.W_dec[down_feature_id, :],
                                                "b s, d -> b s d")

            elif down_module == "attn":
                down_sae = attn_saes[down_layer]
                down_in_pt = f'blocks.{down_layer}.hook_resid_pre'  
                down_out_pt = f'blocks.{down_layer}.attn.hook_z'

                z_cache = {}
                def z_cache_hook(act, hook):
                    z_cache[0] = act
                clean_model_clone.reset_hooks()
                clean_model_clone.add_hook(f'blocks.{down_layer}.attn.hook_z', z_cache_hook, "fwd")

                def down_cache_hook(act, hook):
                    clean_normed_input = clean_model_clone.blocks[down_layer].ln1(act)
                    clean_model_clone.blocks[down_layer].attn(clean_normed_input, clean_normed_input, clean_normed_input)
                    clean_z = z_cache[0]
                    clean_z_concat = rearrange(clean_z, "b s n_heads d_head -> b s (n_heads d_head)")
                    down_clean_facts = torch.relu(
                        (clean_z_concat - down_sae.b_dec) @ down_sae.W_enc[:, down_feature_id]
                        + down_sae.b_enc[down_feature_id]) # [b s]
                    
                    ablated_input = act.clone()
                    ablated_input[:, up_pos] -= up_out_diff[0]
                    if freeze_layernorm:
                        ablated_normed_input = ablated_input  / cache[f'blocks.{down_layer}.ln2.hook_scale']
                    else:
                        ablated_normed_input = clean_model_clone.blocks[down_layer].ln2(ablated_input)
                    clean_model_clone.blocks[down_layer].attn(ablated_normed_input, ablated_normed_input, ablated_normed_input)
                    ablated_z = z_cache[0]
                    ablated_concat = rearrange(ablated_z, "b s n_heads d_head -> b s (n_heads d_head)")
                    down_ablated_facts = torch.relu(
                        (ablated_concat - down_sae.b_dec) @ down_sae.W_enc[:, down_feature_id]
                        + down_sae.b_enc[down_feature_id]) # [b s]
                    
                    all_down_facts_diff = down_ablated_facts - down_clean_facts # [b s]
                    down_facts_diff = torch.zeros_like(all_down_facts_diff)
                    down_facts_diff[:, down_pos] = all_down_facts_diff[:, down_pos]

                    global true_nn_attribs, approx_nn_attribs
                    true_nn_attribs += [down_facts_diff[:,down_pos].mean().cpu().item()]
                    approx_nn_attribs += [vals[1].cpu().item()]

                    W_dec_z = rearrange(up_sae.W_dec[down_feature_id,:],
                                            "(n_heads d_head) -> n_heads d_head",
                                            n_heads = model.cfg.n_heads)
                    W_dec_resid = einsum(W_dec_z , model.W_O[down_layer],
                                        "n_heads d_head, n_heads d_head d_model -> d_model")
                    down_out_diff_concat = einsum(down_facts_diff,
                                                W_dec_resid,
                                                "b s, d -> b s d")
                    
                    down_out_diff[0] = rearrange(down_out_diff_concat, 
                                                "b s (n_heads d_head) -> b s n_heads d_head",
                                                n_heads = model.cfg.n_heads)
            
            else:
                print("error -- down module was", down_module)

            # Patch in the new output of the downstream head
            def down_patch_hook(act, hook):
                return act + down_out_diff[0]

            # Add all the hooks
            model.reset_hooks()
            model.add_hook(up_hook_pt, up_cache_hook, "fwd")
            model.add_hook(down_in_pt, down_cache_hook, "fwd")
            model.add_hook(down_out_pt, down_patch_hook, "fwd")
            if freeze_pattern:
                for layer in range(model.cfg.n_layers):
                    model.add_hook(f'blocks.{layer}.attn.hook_pattern', freeze_pattern_hook, "fwd")
            if freeze_layernorm:
                def freeze_ln_hook(act, hook):
                    return cache[hook.name]
                model.add_hook(f'ln_final.hook_scale', freeze_ln_hook, "fwd")

            m = leap.metric(model, tokens)

            del up_out_diff, down_out_diff
            torch.cuda.empty_cache()
            gc.collect()

            true_attribs += [(clean_m - m).cpu().item()]
            approx_attribs += [vals[3].cpu().item()]

        px.scatter(x=true_attribs, y=approx_attribs,
                labels={'x': 'True edge->metric attrib', 'y': 'LEAP edge->metric attrib'},
                title = f'{up_module} -> {down_module}').show()
        px.scatter(x=[-i for i in true_nn_attribs], y=approx_nn_attribs,
                labels={'x': 'True node->node attrib', 'y': 'LEAP node->node attrib'},
                title = f'{up_module} -> {down_module}').show()
#%%
