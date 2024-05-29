#%%
import sys
sys.path.append("/root/circuit-finder")
import transformer_lens as tl
from functools import partial
from einops import rearrange, einsum
from circuit_finder.pretrained import (
    load_attn_saes,
    load_resid_saes,
    load_hooked_mlp_transcoders,
)
from circuit_finder.core.hooked_transcoder import HookedTranscoderReplacementContext

# Initialize SAEs
attn_saes = load_attn_saes(use_error_term=True)
mlp_transcoders = load_hooked_mlp_transcoders(use_error_term=True)

# Load model
model = tl.HookedSAETransformer.from_pretrained("gpt2").cuda()

#%%
def logit_diff(logits, tokens, correct_str, wrong_str):
    correct_token = model.to_tokens(correct_str)[0,1]
    wrong_token = model.to_tokens(wrong_str)[0,1]
    logits = logits[0,-1]
    return logits[correct_token ] - logits[wrong_token]
#%%
def ablation_hook(act, hook, id_to_ablate):
    assert hook.name.endswith("hook_sae_acts_post")
    act[:, :, id_to_ablate] = 0
    return act

def patch_hook(act, hook, module, layer, feature_id, patch_pt, scale):
    if module == "mlp":
        decoder_col = mlp_transcoders[layer].W_dec[feature_id, :]
    elif module == "attn":
        decoder_col_concat = attn_saes[layer].W_dec[feature_id, :]
        decoder_col = rearrange(
            decoder_col_concat, 
            "(n_heads d_head -> n_heads d_head)",
            n_heads=model.cfg.n_heads
            )
    return act + scale*decoder_col

def run_with_ablations(
        model, 
        prompts,
        attn_saes, 
        mlp_transcoders, 
        ablation_list, # elements (module, layer, feature_id)
        patch_list # elements (module, layer, feature_id, patch_pt, scale)
        ):
    
    # Reset all hooks
    model.reset_hooks()
    for t in mlp_transcoders.values():
        t.reset_hooks()
    for a in attn_saes.values():
        a.reset_hooks()

    # Add ablation hooks to saes/transcoders
    for module, layer, feature_id in ablation_list:
        temp_ablate_hook = partial(
            ablation_hook, 
            id_to_ablate=feature_id
            )
        
        if module == "mlp":     
            mlp_transcoders[layer].add_hook(
                "hook_sae_acts_post",
                temp_ablate_hook,
                "fwd"
            )

        elif module == "attn":
            attn_saes[layer].add_hook(
                "hook_sae_acts_post",
                temp_ablate_hook,
                "fwd"
            )

        else:
            print("modules must be attn or mlp")

    # Now add patching hooks to model
    for module, layer, feature_id, patch_pt, scale in patch_list:
        temp_patch_hook = partial(patch_hook, 
                                  module=module, 
                                  layer=layer, 
                                  feature_id=feature_id, 
                                  patch_pt=patch_pt, 
                                  scale=scale)
        model.add_hook(patch_pt, temp_patch_hook, "fwd")

    # Run model
    with HookedTranscoderReplacementContext(
        model,  # type: ignore
        transcoders=mlp_transcoders.values()
    ) as context:          

        with model.saes(saes=attn_saes.values()):
            logits  = model(prompts)
    
    return logits


# %%
ablation_list = [("mlp", 5, 10087), ("mlp", 5, 6344),
                ("mlp", 0, 20782), ("mlp", 0, 6646) ]
patch_list = [("mlp", 5, 10087, 'blocks.{6}.')]
prompts = ["The favourable prisoner was released on good"]
logits = run_with_ablations(model, 
                            prompts,
                            attn_saes, 
                            mlp_transcoders, 
                            ablation_list,
                            patch_list)

logit_diff(logits, prompts, " behaviour", " behavior")
# %%

