from circuit_finder.core.types import MetricFn, Model, Tokens
import transformer_lens as tl 
from contextlib import contextmanager

GradientCache = tl.ActivationCache

def get_gradient_cache(
    model: Model, 
    tokens: Tokens,
    metric_fn: MetricFn,
) -> GradientCache:
    
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    def names_filter(x):
        return True
    
    bwd_hooks = []
    for hook_name in model.hook_dict: 
        if names_filter(hook_name):
            bwd_hooks.append((hook_name, backward_cache_hook))

    model.zero_grad()
    with model.hooks(
        bwd_hooks = bwd_hooks
    ):
        metric = metric_fn(model, tokens)
        metric.backward()

    return tl.ActivationCache(grad_cache, model)

@contextmanager
def patch_model_gradients_from_cache(
    model: Model,
    grad_cache: GradientCache,
):
    def backward_patch_hook(act, hook):
        patched_act = grad_cache[hook.name]
        assert act.shape == patched_act.shape
        return (patched_act,)
    
    def names_filter(x):
        return x in grad_cache
    
    bwd_hooks = []
    for hook_name in model.hook_dict: 
        if names_filter(hook_name):
            bwd_hooks.append((hook_name, backward_patch_hook))

    with model.hooks(
        bwd_hooks = bwd_hooks
    ):
        yield model

    pass  
