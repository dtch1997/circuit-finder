from transformer_lens import HookedSAE, HookedSAETransformer, HookedSAEConfig

MODEL = "solu-1l"
prompt = "Hello World!"

model = HookedSAETransformer.from_pretrained(MODEL)


def get_sae_config(model, act_name):
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]
    return HookedSAEConfig(d_in=d_in, d_sae=d_in * 2, hook_name=act_name)


sae = HookedSAE(get_sae_config(model, "blocks.0.hook_resid_pre"))
model.add_sae(sae)
