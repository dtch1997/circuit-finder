"""Useful functions from Neel's IOI Colab tutorial.

https://colab.research.google.com/drive/1HvD24vl-WFnydnUYaFkGfT2jKnN7Y6me#scrollTo=EPD6V1OIl6mJ
"""

import torch
from transformer_lens import HookedTransformer

import numpy as np
import eindex
import einops
import transformer_lens as tl
from transformer_lens import utils
from transformer_lens import ActivationCache

from torch import Tensor
from jaxtyping import Int, Float
from circuit_finder.core.types import HookNameFilterFn, MetricFn


def last_token_prediction_loss(model: HookedTransformer, text: str) -> torch.Tensor:
    """Compute the prediction loss of the last token in the text"""
    loss = model(text, return_type="loss", loss_per_token=True)
    return loss[0, -1]


def get_answer_tokens(
    answers: list[tuple[str, str]],
    model: tl.HookedSAETransformer,
) -> Int[torch.Tensor, "batch 2"]:
    # Define the answer tokens (same shape as the answers)
    return torch.concat(
        [model.to_tokens(names, prepend_bos=False).T for names in answers]  # type: ignore
    )


def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer_tokens: Float[Tensor, " batch"],
    wrong_answer_tokens: Float[Tensor, " batch"],
    per_prompt=False,
):
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    last_token_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    correct_logits = eindex.eindex(
        last_token_logits, correct_answer_tokens, "batch [batch]"
    )
    incorrect_logits = eindex.eindex(
        last_token_logits, wrong_answer_tokens, "batch [batch]"
    )

    logit_diff: Float[Tensor, " batch"] = correct_logits - incorrect_logits
    if per_prompt:
        return logit_diff
    else:
        return logit_diff.mean()


def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: tl.ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    """
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    """
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    average_logit_diff = (
        einops.einsum(
            scaled_residual_stack,
            logit_diff_directions,
            "... batch d_model, batch d_model -> ...",
        )
        / batch_size
    )
    return average_logit_diff


def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    """
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape (k, tensor.ndim).
    """
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()


def patching_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> Float[Tensor, " ()"]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """
    last_token_logits = logits[:, -1, :]
    correct_logit = eindex.eindex(
        last_token_logits, answer_tokens[:, 0], "batch [batch]"
    )
    incorrect_logit = eindex.eindex(
        last_token_logits, answer_tokens[:, 1], "batch [batch]"
    )
    logit_diff: Float[Tensor, " batch"] = correct_logit - incorrect_logit
    logit_diff = logit_diff.mean()

    return (logit_diff - corrupted_logit_diff) / (
        clean_logit_diff - corrupted_logit_diff
    )


def get_cache_fwd_and_bwd(
    model: tl.HookedSAETransformer,
    tokens: Int[Tensor, "batch seq"],
    metric_fn: MetricFn,
    filter_fn: HookNameFilterFn,
) -> tuple[float, ActivationCache, ActivationCache]:
    """
    Get a cache of the activations and gradients when running a model on an input.
    """

    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_fn, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_fn, backward_cache_hook, "bwd")

    value = metric_fn(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


def attr_patch_sae_acts(
    clean_cache: ActivationCache,
    clean_grad_cache: ActivationCache,
    site: str,
    layer: int,
):
    clean_sae_acts_post = clean_cache[
        utils.get_act_name(site, layer) + ".hook_sae_acts_post"
    ]
    clean_grad_sae_acts_post = clean_grad_cache[
        utils.get_act_name(site, layer) + ".hook_sae_acts_post"
    ]
    sae_act_attr = clean_grad_sae_acts_post * (0 - clean_sae_acts_post)
    return sae_act_attr
