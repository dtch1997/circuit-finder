import pytest
import torch
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached
from circuit_finder.utils import last_token_prediction_loss


@pytest.fixture()
def model(device: str) -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL).to(device)


@pytest.fixture()
def sae(device) -> SparseAutoencoder:
    return SparseAutoencoder(build_sae_cfg()).to(device)


# @pytest.fixture()
# def sae_dict(sae: SparseAutoencoder) -> dict[ModuleName, SparseAutoencoder]:
#     sae_dict = {ModuleName(sae.cfg.hook_point): sae}
#     return sae_dict


@pytest.fixture()
def device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


@pytest.fixture()
def metric_fn():
    return last_token_prediction_loss


@pytest.fixture()
def text() -> str:
    return "Hello world"
