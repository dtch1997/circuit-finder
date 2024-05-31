import torch
from jaxtyping import Float, Int
from functools import partial
from circuit_finder.core.types import (
    Model,
    Node,
    MetricFn,
    Tokens,
    LayerIndex,
    ModuleName,
    TokenIndex,
    FeatureIndex,
    get_hook_name,
    parse_node_name,
)
from einops import repeat
from collections import namedtuple
from circuit_finder.patching.eap_graph import EAPGraph
from torch import Tensor
from typing import Literal
import transformer_lens as tl
import numpy as np

from typing import Iterator
from contextlib import contextmanager
from circuit_finder.core.hooked_sae import HookedSAE
from circuit_finder.core.hooked_transcoder import HookedTranscoder
from circuit_finder.core.hooked_transcoder import HookedTranscoderReplacementContext

AblateType = Literal["value", "zero"] | None


def get_complement_mask(
    graph, module_name, layer, feature_acts: Float[Tensor, "seq n_features"]
) -> Int[Tensor, "seq n_features"]:
    """
    Returns mask : [seq, n_features].
    mask[seq, feature_id] is False whenever {module_name}.{layer}.{seq}.{feature_id} is a node of graph.
    All other entries are True
    """
    # NOTE:  Feature_acts only used for shape.
    nodes = graph.get_src_nodes()
    positions = [
        int(node.split(".")[2])
        for node in nodes
        if node.split(".")[:2] == [module_name, str(layer)]
    ]
    feature_ids = [
        int(node.split(".")[3])
        for node in nodes
        if node.split(".")[:2] == [module_name, str(layer)]
    ]
    mask = torch.ones_like(feature_acts).bool()
    mask[positions, feature_ids] = False
    return mask


def get_metric_with_ablation(
    model: tl.HookedSAETransformer,
    graph: EAPGraph,
    tokens: Tokens,
    metric: MetricFn,
    transcoders: dict[LayerIndex, HookedTranscoder],
    attn_saes: dict[LayerIndex, HookedSAE],
    ablate_cache: tl.ActivationCache,
    *,
    feature_ablate_type: AblateType = "value",  # options [False, "bm", "zero"]
    error_ablate_type: AblateType = "value",  # options [False, "bm", "zero"]
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
):
    # Splice the SAEs and transcoders into the model
    with splice_model_with_saes_and_transcoders(
        model, list(transcoders.values()), list(attn_saes.values())
    ):
        # Patch the model with corrupt activations
        hooks = get_circuit_ablation_hooks(
            graph,
            ablate_cache,
            feature_ablate_type=feature_ablate_type,
            error_ablate_type=error_ablate_type,
            first_ablated_layer=first_ablated_layer,
        )

        # Run the forward pass on the clean tokens
        with model.hooks(fwd_hooks=hooks):
            return metric(model, tokens)


@contextmanager
def splice_model_with_saes_and_transcoders(
    model: tl.HookedSAETransformer,
    transcoders: list[HookedTranscoder],
    attn_saes: list[HookedSAE],
) -> Iterator[tl.HookedSAETransformer]:
    # Check if error term is used
    for tc in transcoders:
        if not tc.cfg.use_error_term:
            print(
                f"Warning: Transcoder {tc} does not use error term. Inference will not be exact."
            )
            tc.cfg.use_error_term = True
    for sae in attn_saes:
        if not sae.cfg.use_error_term:
            print(
                f"Warning: SAE {sae} does not use error term. Inference will not be exact."
            )
            sae.cfg.use_error_term = True

    try:
        with HookedTranscoderReplacementContext(
            model,  # type: ignore
            transcoders=transcoders,
        ) as context:
            with model.saes(saes=attn_saes):  # type: ignore
                yield model

    finally:
        pass


def filter_sae_acts_and_errors(name: str):
    return "hook_sae_acts_post" in name or "hook_sae_error" in name


def get_node_act(
    cache: tl.ActivationCache,
    node: Node,
):
    """Get the activation of a node from the cache."""
    module_name, layer, token_idx, feature_idx = parse_node_name(node)
    act_name = get_hook_name(module_name, layer)
    return cache[act_name][:, token_idx, feature_idx]


def node_patch_hook(
    act,
    hook,
    *,
    token_idx: TokenIndex | None = None,
    feature_idx: FeatureIndex | None = None,
    value: float | torch.Tensor = 0.0,
):
    """Patches a node by setting its activation to a fixed value.

    Can be converted into a TransformerLens hook by using partial
    """
    act[:, token_idx, feature_idx] = value
    return act


def node_delta_patch_hook(
    act, hook, reference_value, coefficient, *, token_idx, feature_idx
):
    """Patches a node by interpolating towards a reference value."""
    a = act[:, token_idx, feature_idx]
    delta = reference_value - a
    act[:, token_idx, feature_idx] = a + coefficient * delta
    return act


def get_node_patch_hook(node: Node, value: float):
    module_name, layer, token_idx, feature_idx = parse_node_name(node)
    hook_name = get_hook_name(module_name, layer)

    def hook_fn(act, hook):
        return node_patch_hook(
            act, hook, token_idx=token_idx, feature_idx=feature_idx, value=value
        )

    return hook_name, hook_fn


def get_circuit_node_patch_hooks(
    clean_cache: tl.ActivationCache,
    corrupt_cache: tl.ActivationCache,
    nodes: list[Node],
    coefficient: float,
):
    """Return a list of patch hooks"""

    fwd_hooks = []
    for node in nodes:
        module_name, layer, token_idx, feature_idx = parse_node_name(node)
        hook_name = get_hook_name(module_name, layer)
        clean_value = get_node_act(clean_cache, node)
        corrupt_value = get_node_act(corrupt_cache, node)

        # Interpolate between clean, corrupt to get the value
        # c = 0 --> corrupt value
        # c = 1 --> clean value
        value = (clean_value - corrupt_value) * coefficient + corrupt_value

        # Define the hook function
        hook_fn = partial(
            node_patch_hook,
            token_idx=token_idx,
            feature_idx=feature_idx,
            value=value,
        )
        fwd_hooks.append((hook_name, hook_fn))
    return fwd_hooks


def get_circuit_node_delta_patch_hooks(
    corrupt_cache: tl.ActivationCache,
    nodes: list[Node],
    coefficient: float,
):
    """Return a list of patch hooks"""

    fwd_hooks = []
    for node in nodes:
        module_name, layer, token_idx, feature_idx = parse_node_name(node)
        hook_name = get_hook_name(module_name, layer)
        corrupt_value = get_node_act(corrupt_cache, node)

        # Define the hook function
        hook_fn = partial(
            node_delta_patch_hook,
            reference_value=corrupt_value,
            coefficient=coefficient,
            token_idx=token_idx,
            feature_idx=feature_idx,
        )
        fwd_hooks.append((hook_name, hook_fn))
    return fwd_hooks


AblationResult = namedtuple("AblationResult", ["coefficient", "metric"])


def get_ablation_result(
    model: Model,
    transcoders,
    attn_saes,
    *,
    clean_tokens,
    corrupt_tokens,
    clean_cache,
    corrupt_cache,
    nodes,
    metric_fn: MetricFn,
    coefficients=np.linspace(0, 1, 11),
    setting="noising",
):
    coefs = []
    metrics = []

    if setting == "noising":
        with splice_model_with_saes_and_transcoders(
            model,  # type: ignore
            transcoders,
            attn_saes,
        ) as spliced_model:
            for coefficient in coefficients:
                fwd_hooks = get_circuit_node_patch_hooks(
                    clean_cache, corrupt_cache, nodes, coefficient
                )
                with model.hooks(fwd_hooks=fwd_hooks):
                    metric = metric_fn(model, clean_tokens).item()
                    metrics.append(metric)
                    coefs.append(coefficient)
    elif setting == "denoising":
        with splice_model_with_saes_and_transcoders(
            model,  # type: ignore
            transcoders,
            attn_saes,
        ) as spliced_model:
            for coefficient in coefficients:
                # NOTE: In the noising setting, the clean and corrupt caches are swapped
                fwd_hooks = get_circuit_node_patch_hooks(
                    corrupt_cache, clean_cache, nodes, coefficient
                )
                with model.hooks(fwd_hooks=fwd_hooks):
                    # NOTE: In the noising setting, the clean and corrupt tokens are swapped
                    metric = metric_fn(model, corrupt_tokens).item()
                    metrics.append(metric)
                    # NOTE: in the noising setting, the role of the coefficient is inverted
                    coefs.append(1 - coefficient)

    return AblationResult(coefs, metrics)


def get_circuit_ablation_hooks(
    graph: EAPGraph,
    cache: tl.ActivationCache,  # The cache of activations to patch with.
    *,
    feature_ablate_type: AblateType = "value",
    error_ablate_type: AblateType = "value",
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
) -> list[tuple]:
    """Return hooks to ablate a model outside a graph"""

    def patch_act_post_hook(act, hook, layer: LayerIndex, module_name: ModuleName):
        assert hook.name.endswith("hook_sae_acts_post")
        feature_acts = cache[hook.name]

        # Add the token dimension back
        if len(feature_acts.shape) == 1:
            feature_acts = repeat(
                feature_acts, "n_features -> seq n_features", seq=act.shape[1]
            )
        assert (
            len(feature_acts.shape) == 2
        ), f"Hook {hook.name} had shape {feature_acts.shape}"

        # NOTE: only patch the features outside the graph.
        mask = get_complement_mask(graph, module_name, layer, feature_acts)

        ablate_acts: Float[Tensor, "seq n_features"]

        if feature_ablate_type == "value":
            ablate_acts = feature_acts
            act[:, mask] = ablate_acts[mask]
        elif feature_ablate_type == "zero":
            ablate_acts = torch.zeros_like(feature_acts)
            act[:, mask] = ablate_acts[mask]
        elif feature_ablate_type is None:
            pass
        else:
            raise ValueError(f"Unknown feature_ablate_type: {feature_ablate_type}")

        return act

    def patch_error_hook(act, hook, layer, module_name):
        assert hook.name.endswith("hook_sae_error")
        error_acts = cache[hook.name]

        # Add the token dimension back
        if len(error_acts.shape) == 1:
            error_acts = repeat(
                error_acts, "n_features -> seq n_features", seq=act.shape[1]
            )
        assert (
            len(error_acts.shape) == 2
        ), f"Hook {hook.name} had shape {error_acts.shape}"

        ablate_acts: Float[Tensor, "seq n_features"]
        if error_ablate_type == "value":
            ablate_acts = error_acts
            act[:] = ablate_acts
        elif error_ablate_type == "zero":
            ablate_acts = torch.zeros_like(error_acts)
            act[:] = ablate_acts
        elif error_ablate_type is None:
            pass
        else:
            raise ValueError(f"Unknown error_ablate_type: {error_ablate_type}")

        return act

    fwd_hooks = []
    for layer in range(first_ablated_layer, cache.model.cfg.n_layers):
        fwd_hooks.append(
            (
                f"blocks.{layer}.attn.hook_z.hook_sae_acts_post",
                partial(patch_act_post_hook, layer=layer, module_name="attn"),
            )
        )
        fwd_hooks.append(
            (
                f"blocks.{layer}.attn.hook_z.hook_sae_error",
                partial(patch_error_hook, layer=layer, module_name="attn"),
            )
        )
        # MLP transcoder hooks
        fwd_hooks.append(
            (
                f"blocks.{layer}.mlp.transcoder.hook_sae_acts_post",
                partial(patch_act_post_hook, layer=layer, module_name="mlp"),
            )
        )
        fwd_hooks.append(
            (
                f"blocks.{layer}.mlp.hook_sae_error",
                partial(patch_error_hook, layer=layer, module_name="mlp"),
            )
        )

    return fwd_hooks
