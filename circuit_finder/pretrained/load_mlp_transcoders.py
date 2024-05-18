from transcoders_slim.transcoder import Transcoder
from transcoders_slim.load_pretrained import load_pretrained
from circuit_finder.core.types import LayerIndex
from circuit_finder.constants import ALL_GPT_2_SMALL_LAYERS, device


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
