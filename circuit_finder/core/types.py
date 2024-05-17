import torch

HookName = str
LayerIndex = int

ALL_GPT_2_SMALL_LAYERS: list[LayerIndex] = list(range(12))

device = "cuda" if torch.cuda.is_available() else "cpu"
