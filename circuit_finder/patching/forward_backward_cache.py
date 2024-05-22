import transformer_lens as tl
from typing import Callable


class ForwardBackwardCache:
    """Context manager to cache forward and backward activations of a model"""

    model: tl.HookedTransformer

    def __init__(
        self, model: tl.HookedTransformer, filter_fn: str | Callable[[str], bool]
    ):
        self.model = model
        self.filter_fn = filter_fn
        self.act_cache_dict = {}
        self.grad_cache_dict = {}

    def __enter__(self):
        # TODO: change this to use ```with model.hooks(): ... '''
        self.model.reset_hooks()

        def forward_cache_hook(act, hook):
            self.act_cache_dict[hook.name] = act.detach()

        def backward_cache_hook(act, hook):
            self.grad_cache_dict[hook.name] = act.detach()

        self.model.add_hook(self.filter_fn, forward_cache_hook, "fwd")
        self.model.add_hook(self.filter_fn, backward_cache_hook, "bwd")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.reset_hooks()

    @property
    def act_cache(self):
        return tl.ActivationCache(self.act_cache_dict, self.model)

    @property
    def grad_cache(self):
        return tl.ActivationCache(self.grad_cache_dict, self.model)
