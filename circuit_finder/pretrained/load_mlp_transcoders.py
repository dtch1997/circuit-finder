from transcoders_slim.transcoder import Transcoder
from transcoders_slim.load_pretrained import load_pretrained
from transcoders_slim.sae_training.config import LanguageModelSAERunnerConfig

from circuit_finder.core.types import LayerIndex
from circuit_finder.constants import ALL_GPT_2_SMALL_LAYERS, device
from circuit_finder.core.hooked_transcoder import (
    HookedTranscoder,
    HookedTranscoderConfig,
)


def get_filenames(layers: list[int]) -> list[str]:
    return [
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.ln2.hook_normalized_24576.pt"
        for layer in layers
    ]


def parse_layer_of_module_name(module_name: str) -> LayerIndex:
    return int(module_name.split(".")[1])


def load_mlp_transcoders(
    layers: list[int] = ALL_GPT_2_SMALL_LAYERS,
    device: str = device,
) -> dict[LayerIndex, Transcoder]:
    transcoders_dict = load_pretrained(get_filenames(layers))
    transcoders = {}
    for module_name, transcoder in transcoders_dict.items():
        layer = parse_layer_of_module_name(module_name)
        transcoders[layer] = transcoder.to(device)
    return transcoders


def ts_tc_cfg_to_hooked_tc_cfg(
    resid_sae_cfg: LanguageModelSAERunnerConfig,
) -> HookedTranscoderConfig:
    new_cfg = {
        "d_sae": resid_sae_cfg.d_sae,
        "d_in": resid_sae_cfg.d_in,
        "d_out": resid_sae_cfg.d_out,
        "hook_name": resid_sae_cfg.hook_point,
        "hook_name_out": resid_sae_cfg.out_hook_point,
    }
    return HookedTranscoderConfig.from_dict(new_cfg)


def ts_tc_to_hooked_tc(
    sl_sae: Transcoder,
) -> HookedTranscoder:
    state_dict = sl_sae.state_dict()
    cfg = ts_tc_cfg_to_hooked_tc_cfg(sl_sae.cfg)
    tl_sae = HookedTranscoder(cfg)
    tl_sae.load_state_dict(state_dict)
    return tl_sae


def load_hooked_mlp_transcoders(
    layers: list[int] = ALL_GPT_2_SMALL_LAYERS,
    device: str = device,
) -> dict[LayerIndex, HookedTranscoder]:
    transcoders_dict = load_pretrained(get_filenames(layers))
    hooked_transcoders = {}
    for module_name, transcoder in transcoders_dict.items():
        layer = parse_layer_of_module_name(module_name)
        hooked_transcoder = ts_tc_to_hooked_tc(transcoder).to(device)
        hooked_transcoders[layer] = hooked_transcoder

    return hooked_transcoders
