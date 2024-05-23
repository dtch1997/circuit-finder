from typing import Callable, Iterator

from contextlib import contextmanager
from transformer_lens import ActivationCache, HookedTransformer


def print_hooks(model: HookedTransformer):
    """Utility to print all the hooks currently attached to model"""
    for hook_name, hook_point in model.hook_dict.items():
        for lens_handle in hook_point.fwd_hooks:
            print(f"{hook_name} | fwd: {lens_handle.hook}")
        for lens_handle in hook_point.bwd_hooks:
            print(f"{hook_name} | bwd: {lens_handle.hook}")


@contextmanager
def get_backward_cache(
    model: HookedTransformer,
    names_filter: str | Callable[[str], bool],
) -> Iterator[ActivationCache]:
    """Utility to get the backward cache from a model.

    Usage:
    metric = model(tokens)
    backward_cache = get_backward_cache(model, "hook_z", metric)
    """
    try:
        grad_cache = {}
        hook_names = []
        if isinstance(names_filter, str):
            hook_names = [
                hook_name for hook_name in model.hook_dict if names_filter in hook_name
            ]
        else:
            hook_names = [
                hook_name for hook_name in model.hook_dict if names_filter(hook_name)
            ]

        def grad_cache_hook(grad, hook):
            grad_cache[hook.name] = grad.detach()

        bwd_hooks = [(hook_pt, grad_cache_hook) for hook_pt in hook_names]
        with model.hooks(bwd_hooks=bwd_hooks):  # type: ignore
            yield ActivationCache(grad_cache, model)

    finally:
        pass


# @contextmanager
# def apply_saes_and_transcoders(
#     model: HookedTransformer,
#     saes: dict[int, HookedSAE],
#     transcoders: dict[int, HookedTranscoder],
# ) -> Iterator[HookedTransformer]:
#     with HookedTranscoderReplacementContext(
#         model,  # type: ignore
#         transcoders=transcoders,
#     ) as context:
#         with model.saes(saes=saes):
#             yield model
