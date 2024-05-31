import sys
sys.path.append("/root/circuit-finder")
from functools import partial
from einops import rearrange, einsum
from circuit_finder.core.hooked_transcoder import HookedTranscoderReplacementContext
import torch

def get_decoder_col(model,
                    attn_saes, 
                    mlp_transcoders, 
                    module, 
                    layer, 
                    feature_id):
    if module == "mlp":
        decoder_col = mlp_transcoders[layer].W_dec[feature_id, :]
    elif module == "attn":
        decoder_col_concat = attn_saes[layer].W_dec[feature_id, :]
        decoder_col_z = rearrange(
            decoder_col_concat, 
            "(n_heads d_head) -> n_heads d_head",
            n_heads=model.cfg.n_heads
            )
        decoder_col = einsum(decoder_col_z,
                            model.W_O[layer],
                            "n_heads d_head, n_heads d_head d_model -> d_model")
    else: print(module, "not a valdid module")
    return decoder_col

def run_with_ablations(
        model, 
        prompts,
        attn_saes, 
        mlp_transcoders, 
        ablation_list = [], # elements (module, layer, feature_id, pos)
        patch_list = [], # elements (module, layer, feature_id, pos, patch_pt, scale)
        grab_and_delete_list = [], # elements (module, layer, feature_id, pos, patch_pt, scale)
        cache_names_filter = []
        ):
    
    # Define the hook functions we'll need for ablating & patching
    def ablation_hook(act, hook,  # hook.name will determine what module/layer we're looking at
                      feature_id, pos
                      ):
        assert hook.name.endswith("hook_sae_acts_post")
        _, layer, module = hook.name.split('.')[:3]
        layer = int(layer)
        print(f"ablating L{layer} {module} {feature_id} at pos {pos} with mean activation {act[:, pos, feature_id].mean().item():.2f}")
        act[:, pos, feature_id] = 0
        return act

    def patch_hook(act, hook, 
                   module, layer, feature_id,   #these tell us which feature to patch
                   pos, patch_pt, scale   #these tell us where to patch the feature, and by how much
                   ):
        act[:, pos, :] += scale* get_decoder_col(model, attn_saes, mlp_transcoders, 
                                                 module, layer, feature_id)
        return act
    
    grab_cache = {}
    def grab_hook(act, hook, feature_id, pos):
        assert hook.name.endswith("hook_sae_acts_post")
        _, layer, module = hook.name.split('.')[:3]
        layer = int(layer)
        grab_cache[module, layer, feature_id, pos] = act[:, pos, feature_id] # [batch]

    def delete_grabbed_feature_hook(act, hook, 
                                   module, layer, feature_id,  #these tell us which feature to patch
                                   pos, patch_pt): #these tell us where to patch the feature, and by how much
        assert len(grab_cache) > 0
        decoder_col = get_decoder_col(model, attn_saes, mlp_transcoders, 
                                                 module, layer, feature_id
                                                 ).unsqueeze(0) # [1, d_model]
        scale = grab_cache[module, layer, feature_id, pos].unsqueeze(1) # [batch, 1]
        act[:, pos, :] -= scale * decoder_col
        return act
    
    # Reset all hooks
    model.reset_hooks()
    for t in mlp_transcoders.values():
        t.reset_hooks()
    for a in attn_saes.values():
        a.reset_hooks()

    # Add ablation hooks to saes/transcoders
    for module, layer, feature_id, pos in ablation_list:
        temp_ablate_hook = partial(
            ablation_hook, 
            feature_id=feature_id,
            pos=pos
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
    for module, layer, feature_id, pos, patch_pt, scale in patch_list:
        temp_patch_hook = partial(patch_hook, 
                                  module=module, 
                                  layer=layer, 
                                  feature_id=feature_id,
                                  pos=pos, 
                                  patch_pt=patch_pt, 
                                  scale=scale)
        model.add_hook(patch_pt, temp_patch_hook, "fwd")

    for module, layer, feature_id, pos, patch_pt in grab_and_delete_list:
        if module == "attn":
            attn_saes[layer].add_hook("hook_sae_acts_post",
                                    partial(grab_hook, feature_id=feature_id, pos=pos))
        elif module == "mlp":
            mlp_transcoders[layer].add_hook("hook_sae_acts_post",
                                    partial(grab_hook, feature_id=feature_id, pos=pos))  
        else: print("module must be attn or mlp")        
        
        temp_patch_hook = partial(delete_grabbed_feature_hook, 
                                  module=module, 
                                  layer=layer, 
                                  feature_id=feature_id,
                                  pos=pos, 
                                  patch_pt=patch_pt)
        model.add_hook(patch_pt, temp_patch_hook, "fwd")


    # Run model
    with HookedTranscoderReplacementContext(
        model,  # type: ignore
        transcoders=mlp_transcoders.values()
    ) as context:          

        with model.saes(saes=attn_saes.values()):
            logits, cache = model.run_with_cache(
                prompts, 
                return_type="logits",
                names_filter=cache_names_filter,
                prepend_bos=True)

        return logits, cache
    