from typing import Literal, Protocol

import torch
from jaxtyping import Float
from sae_lens import SparseAutoencoder
from transformer_lens.hook_points import HookPoint


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError


ForwardHookData = tuple[str, TransformerLensForwardHook]
BackwardHookData = tuple[str, TransformerLensBackwardHook]

NodeType = Literal["feature", "error", "all"]


class SAESplicer:
    """Splices a single SAE into the computational graph of a HookedTransformer

    Usage:

    sae_patcher = SAEPatcher(sae)
    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()]
        bwd_hooks=[sae_patcher.get_backward_hook()]
    ):
        _, cache = model.run_with_cache(text)
        cache = sae_patcher.update_cache(cache)
    """

    sae: SparseAutoencoder
    sae_feature_acts: Float[torch.Tensor, "n_batch n_token d_sae"]
    sae_errors: Float[torch.Tensor, "n_batch n_token d_model"]

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae
        self.sae_feature_acts = torch.tensor([])
        self.sae_errors = torch.tensor([])

    def _forward_hook_fn(
        self,
        orig: Float[torch.Tensor, "n_batch n_token d_model"],
        hook: HookPoint,  # noqa: ARG002
    ) -> Float[torch.Tensor, "n_batch n_token d_model"]:
        """Forward hook to patch the SAE into the computational graph

        orig: Original activations
        """

        a_orig = orig
        a_sae, z_sae = self.sae(a_orig)[:2]
        # keep pyright happy
        assert isinstance(z_sae, torch.Tensor)
        a_err = a_orig - a_sae.detach()

        # Track the gradients
        assert z_sae.requires_grad
        z_sae.retain_grad()
        assert a_err.requires_grad
        a_err.retain_grad()

        # Store values for later use
        self.sae_feature_acts = z_sae
        self.sae_errors = a_err

        a_rec = a_sae + a_err
        return a_rec

    @property
    def sae_nodes(self) -> Float[torch.Tensor, "n_batch n_token (d_sae+d_model)"]:
        return torch.cat([self.sae_feature_acts, self.sae_errors], dim=-1)

    def _backward_hook_fn(
        self,
        orig: Float[torch.Tensor, "n_batch n_token d_model"],
        hook: HookPoint,  # noqa: ARG002
    ) -> tuple[Float[torch.Tensor, "n_batch n_token d_model"]]:
        """Implement pass-through gradients

        orig: gradient w.r.t output
        return: gradient w.r.t input
        """
        # NOTE: We stopped the gradient in the forward pass
        # So we need to restore the gradient here

        # NOTE: Transformer lens un-tuples the gradient before passing it in
        # So we need to re-tuple it here
        # (Valid as of transformer_lens v1.5.4)
        return (orig,)

    def get_forward_hook(self) -> ForwardHookData:
        """Return a forward hook that patches the activation."""
        return (self.sae.cfg.hook_point, self._forward_hook_fn)

    def get_backward_hook(self) -> BackwardHookData:
        """Return a backward hook that patches the gradients."""
        return (self.sae.cfg.hook_point, self._backward_hook_fn)

    def get_node_values(
        self, node_type: NodeType = "all"
    ) -> Float[torch.Tensor, "n_batch n_token n_nodes"]:
        if node_type == "feature":
            return self.sae_feature_acts
        elif node_type == "error":
            return self.sae_errors
        elif node_type == "all":
            return torch.cat([self.sae_feature_acts, self.sae_errors], dim=-1)
        else:
            raise ValueError(f"Invalid node_type: {node_type}")

    def get_node_grads(
        self, node_type: NodeType = "all"
    ) -> Float[torch.Tensor, "n_batch n_token n_nodes"]:
        if node_type == "feature":
            if self.sae_feature_acts.grad is None:
                raise RuntimeError("Gradients are not available.")
            return self.sae_feature_acts.grad
        elif node_type == "error":
            if self.sae_errors.grad is None:
                raise RuntimeError("Gradients are not available.")
            return self.sae_errors.grad
        elif node_type == "all":
            if self.sae_feature_acts.grad is None or self.sae_errors.grad is None:
                raise RuntimeError("Gradients are not available.")
            return torch.cat([self.sae_feature_acts.grad, self.sae_errors.grad], dim=-1)
        else:
            raise ValueError(f"Invalid node_type: {node_type}")

    # @contextmanager
    # def splice(self, model: HookedTransformer) -> HookedTransformer:
    #     with model.hooks(
    #         fwd_hooks=[self.get_forward_hook()],
    #         bwd_hooks=[self.get_backward_hook()],
    #     ):
    #         yield model

    # def update_cache(self, cache: ActivationCache) -> ActivationCache:
    #     cache[self.sae.cfg.hook_point + ".sae_acts"] = self.sae_feature_acts
    #     cache[self.sae.cfg.hook_point + ".sae_acts_grads"] = self.sae_feature_acts.grad
    #     cache[self.sae.cfg.hook_point + ".sae_errs"] = self.sae_errors
    #     cache[self.sae.cfg.hook_point + ".sae_errs_grads"] = self.sae_errors.grad
    #     return cache


# class SAESplicer:
#     """Patches many SAEs into the computational graph of a HookedTransformer

#     Usage:

#     sae_splicer = sae_splicer(sae)
#     with sae_splicer.splice(model):
#         ...
#     """

#     sae_dict: dict[ModuleName, SparseAutoencoder]
#     sae_feature_acts: Float[torch.Tensor, "n_batch n_token d_sae"]
#     sae_errors: Float[torch.Tensor, "n_batch n_token d_model"]
