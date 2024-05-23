from circuit_finder.patching.utils import get_backward_cache
import pytest

from transformer_lens import HookedSAETransformer

MODEL = "solu-1l"
prompt = "Hello World!"


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL)
    yield model
    model.reset_hooks()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_backward_cache(model, act_name):
    # Get the gradients using the backward cache
    with get_backward_cache(model, act_name) as backward_cache:
        loss = model(prompt, return_type="loss").mean()
        loss.backward()
    assert act_name in backward_cache
