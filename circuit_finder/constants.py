import torch
from circuit_finder.core.types import LayerIndex

GPT_2_SMALL: str = "gpt2"
ALL_GPT_2_SMALL_LAYERS: list[LayerIndex] = list(range(12))

device = "cuda" if torch.cuda.is_available() else "cpu"
