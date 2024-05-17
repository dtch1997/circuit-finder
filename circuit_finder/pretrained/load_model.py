import transformer_lens as tl

from circuit_finder.constants import GPT_2_SMALL, device


def load_model(
    name: str = GPT_2_SMALL, device: str = device
) -> tl.HookedSAETransformer:
    """
    Load a `HookedSAETransformer` model with the necessary config to perform edge patching
    (with separate edges to Q, K, and V). Sets `requires_grad` to `False` for all model
    weights.
    """
    tl_model = tl.HookedSAETransformer.from_pretrained(
        name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_attn_in = True
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.eval()
    for param in tl_model.parameters():
        param.requires_grad = False
    return tl_model
