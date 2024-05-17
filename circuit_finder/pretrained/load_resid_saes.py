"""Load Joseph Bloom's residual SAEs"""

import torch
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.train_sae_on_language_model import LanguageModelSAERunnerConfig

from circuit_finder.core.hooked_sae import HookedSAE
from circuit_finder.core.hooked_sae_config import HookedSAEConfig
from circuit_finder.core.types import LayerIndex, ALL_GPT_2_SMALL_LAYERS


def get_hook_point_from_layer(layer: LayerIndex) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def resid_sae_cfg_to_hooked_sae_cfg(resid_sae_cfg: LanguageModelSAERunnerConfig):
    new_cfg = {
        "d_sae": resid_sae_cfg.d_sae,
        "d_in": resid_sae_cfg.d_in,
        "hook_name": resid_sae_cfg.hook_point,
    }
    return HookedSAEConfig.from_dict(new_cfg)


def load_resid_saes(
    layers: list[int] = ALL_GPT_2_SMALL_LAYERS + [12],
) -> dict[LayerIndex, HookedSAE]:
    layer_to_sae = {}
    for layer in layers:
        if 0 <= layer < 12:
            hook_point = f"blocks.{layer}.hook_resid_pre"
        elif layer == 12:
            hook_point = "blocks.11.hook_resid_post"
        else:
            raise ValueError(f"Invalid layer: {layer}")

        (sae_dict, _) = get_gpt2_res_jb_saes(hook_point)
        sae = sae_dict[hook_point]

        # convert to HookedSAE
        cfg = resid_sae_cfg_to_hooked_sae_cfg(sae.cfg)
        hooked_sae = HookedSAE(cfg)
        state_dict = sae.state_dict()

        # NOTE: sae-lens uses a 'scaling factor'
        # For now, just check this is 1 and then remove it
        torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        )
        state_dict.pop("scaling_factor")
        hooked_sae.load_state_dict(state_dict)

        layer_to_sae[layer] = hooked_sae
    return layer_to_sae
