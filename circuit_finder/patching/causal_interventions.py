#%%

import transformer_lens as tl
from functools import partial
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
def ablation_hook(act, hook, id_to_ablate):
    assert hook.name.endswith("hook_sae_acts_post")
    act[:, :, id_to_ablate] = 0
    return act
#%%
def run_with_ablations(
        model, 
        attn_saes, 
        mlp_transcoders, 
        ablation_list
        ):
    
    # Reset hooks on all saes/transcoders
    for t in mlp_transcoders.values():
        t.reset_hooks()
    for a in attn_saes.values():
        a.reset_hooks()

    # Add ablation hooks to all saes/transcoders
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

    # Run model
    with HookedTranscoderReplacementContext(
        model,  # type: ignore
        transcoders=mlp_transcoders.values()
    ) as context:          

        with model.saes(saes=attn_saes.values()):
            logits  = model("Hi my name is Jacob")
    
    return logits


# %%
ablation_list = [("mlp", 10, id) for id in range(20)]
logits = run_with_ablations(model, 
                            attn_saes, 
                            mlp_transcoders, 
                            ablation_list)


