from jaxtyping import Float
from torch import Tensor
from circuit_finder.core.types import Model, Tokens, MetricFn
from circuit_finder.

# A name of a component in a model
# e.g. SAE activations at a given module for all token indices and features
Component = str

ComponentActivation = Float[Tensor, "batch t f"]

# A tensor that contains edge attributions for all edges between two components
EdgeAttribution = Float[Tensor, "batch t_up f_up t_down f_down"]

# A dictionary that
EdgeAttributionDict = dict[Component, EdgeAttribution]

def list_all_components(model):
    attn_sae_components = [
        f"blocks.{layer}.attn.hook_z.hook_sae_acts_post" for layer in model.cfg.layers
    ]
    transcoder_components = [
        f"blocks.0.{layer}.transcoder.hook_z.hook_sae_acts_post" for layer in model.cfg.layers
    ]
    return attn_sae_components + transcoder_components

def get_d_metric_d_component(
    model, tokens, metric_fn
) -> dict[Component, ComponentActivation]:
    return {}

def get_d_dest_d_src(
    model, d_component: Component, u_component: Component
) -> EdgeAttribution:
    return {}
    

def get_edge_attribution(
    model: Model, 
    tokens: Tokens,
    metric_fn: MetricFn,
    *,
    corrupt_tokens: Tokens | None = None,
) -> EdgeAttributionDict:
    

    # First, we need dMetric / dComponent for all components

    return {}