"""Implement"""

import transformer_lens as tl
from circuit_finder.core.types import LayerIndex
from typing import Callable
from dataclasses import dataclass

from circuit_finder.core import HookedSAE

HookName = str
ModuleType = str
NodeType = str
sae_module_type = [
    "mlp",
    "mlp.transcoder",
    "hook_resid_pre",
    "hook_resid_post",
    "attn.hook_z",
]
sae_node_types = ["hook_sae_error", "hook_sae_output"]


@dataclass(frozen=True)
class Node:
    layer: LayerIndex
    module_type: ModuleType
    node_type: NodeType

    @property
    def hook_name(self):
        return f"blocks.{self.layer}.{self.module_type}.{self.node_type}"

    @staticmethod
    def from_hook_name(hook_name: str) -> "Node":
        parts = hook_name.split(".")
        layer = int(parts[1])
        node_type = parts[-1]
        module_type = ".".join(parts[2:-1])
        return Node(layer, module_type, node_type)


def get_gradient_hook_point_for_sae(sae: HookedSAE) -> HookName:
    """Get the gradient hook point for an SAE.

    Return a hook name in a standard HookedTransformer that will have the same gradient as sae_output.
    In practice this will usually be the residual stream.
    """
    sae_node = Node.from_hook_name(sae.cfg.hook_name)
    if sae_node.module_type in ("mlp", "mlp.transcoder"):  # blocks.{layer}.mlp.
        # Return hook resid post
        return f"blocks.{sae_node.layer}.hook_resid_post"
    elif sae_node.module_type == "attn":  # blocks.{layer}.attn.hook_z
        # Return hook resid mid
        return f"blocks.{sae_node.layer}.hook_resid_mid"
    elif sae_node.module_type in ("hook_resid_pre", "hook_resid_post"):
        # Here we can just return the same hook point
        return sae.cfg.hook_name

    raise ValueError(f"Unknown SAE module type {sae_node.module_type}")


class ForwardBackwardCache:
    """Context manager to cache forward and backward activations of a model"""

    def __init__(
        self, model: tl.HookedSAETransformer, filter_fn: str | Callable[[str], bool]
    ):
        self.model = model
        self.filter_fn = filter_fn
        self.act_cache = {}
        self.grad_cache = {}

    def __enter__(self):
        # TODO: change this to use ```with model.hooks(): ... '''
        self.model.reset_hooks()

        def forward_cache_hook(act, hook):
            self.act_cache[hook.name] = act.detach()

        def backward_cache_hook(act, hook):
            self.grad_cache[hook.name] = act.detach()

        self.model.add_hook(self.filter_fn, forward_cache_hook, "fwd")
        self.model.add_hook(self.filter_fn, backward_cache_hook, "bwd")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.reset_hooks()
