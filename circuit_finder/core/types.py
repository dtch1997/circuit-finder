import torch
import transformer_lens as tl

from jaxtyping import Float, Int
from typing import Protocol, Callable, Literal, TypeGuard
from transformer_lens.hook_points import HookPoint

HookName = str
ModuleName = Literal["mlp", "attn", "metric", "mlp_error", "attn_error"]

LayerIndex = int
FeatureIndex = int
TokenIndex = int
HookNameFilterFn = Callable[[HookName], bool]

Node = str
Edge = tuple[Node, Node]  # (dest, src)
Attrib = tuple[float, float, float, float]

Tokens = Int[torch.Tensor, "batch seq"]

Logits = Float[torch.Tensor, "batch seq d_vocab"]
Model = tl.HookedTransformer
Metric = Float[torch.Tensor, " ()"]
MetricFn = Callable[[Model, Tokens], Metric]
SaeFamily = Literal["res-jb", "att-kk", "tres-dc"]


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError


def is_valid_module_name(str) -> TypeGuard[ModuleName]:
    return str in ["mlp", "attn", "metric", "mlp_error", "attn_error"]


def parse_node_name(
    node: Node,
) -> tuple[ModuleName, LayerIndex, TokenIndex, FeatureIndex]:
    """Parse a node name into its components."""
    module, layer, pos, feature_id = node.split(".")
    assert is_valid_module_name(module)
    return module, int(layer), int(pos), int(feature_id)


def get_node_name(
    module: ModuleName, layer: LayerIndex, pos: TokenIndex, feature_id: FeatureIndex
) -> Node:
    """Get a node name from its components."""
    return f"{module}.{layer}.{pos}.{feature_id}"
