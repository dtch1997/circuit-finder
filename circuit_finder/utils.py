import torch
from transformer_lens import HookedTransformer


def last_token_prediction_loss(model: HookedTransformer, text: str) -> torch.Tensor:
    """Compute the prediction loss of the last token in the text"""
    loss = model(text, return_type="loss", loss_per_token=True)
    return loss[0, -1]
