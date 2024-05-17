import torch
from jaxtyping import Float
from typing import Protocol, Callable
from transformer_lens.hook_points import HookPoint

HookName = str
LayerIndex = int
HookNameFilterFn = Callable[[HookName], bool]
Logits = Float[torch.Tensor, "batch seq d_vocab"]
Metric = Float[torch.Tensor, "batch"]
MetricFn = Callable[[Logits], Metric]


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError
