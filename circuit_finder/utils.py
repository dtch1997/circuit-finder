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


from torch import Tensor
from jaxtyping import Int, Float


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
        [model.to_tokens(names, prepend_bos=False).T for names in answers]
    )


def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt=False,
):
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    last_token_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    correct_logits = eindex.eindex(
        last_token_logits, answer_tokens[:, 0], "batch [batch]"
    )
    incorrect_logits = eindex.eindex(
        last_token_logits, answer_tokens[:, 1], "batch [batch]"
    )

    logits: Float[Tensor, " batch"] = correct_logits - incorrect_logits
    if per_prompt:
        return logits
    else:
        return logits.mean()


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
) -> float:
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
