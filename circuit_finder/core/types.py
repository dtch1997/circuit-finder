import torch
from typing import Protocol
from transformer_lens.hook_points import HookPoint

HookName = str
LayerIndex = int

ALL_GPT_2_SMALL_LAYERS: list[LayerIndex] = list(range(12))

device = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError
