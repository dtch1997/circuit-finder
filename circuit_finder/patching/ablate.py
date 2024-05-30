import torch
from jaxtyping import Float
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
from collections import namedtuple
from circuit_finder.patching.eap_graph import EAPGraph
from torch import Tensor
import transformer_lens as tl
import numpy as np

from typing import Iterator
from contextlib import contextmanager
from circuit_finder.core.hooked_sae import HookedSAE
from circuit_finder.core.hooked_transcoder import HookedTranscoder
from circuit_finder.core.hooked_transcoder import HookedTranscoderReplacementContext


def get_mask(
    graph, module_name, layer, feature_acts: Float[Tensor, "batch seq n_features"]
):
    """
    Returns mask : [seq, n_features].
    mask[seq, feature_id] is False whenever {module_name}.{layer}.{seq}.{feature_id} is a node of graph.
    All other entries are True
    """
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
    mask = torch.ones_like(feature_acts[0]).bool()
    mask[positions, feature_ids] = False
    return mask


def get_metric_with_ablation(
    model: tl.HookedSAETransformer,
    graph: EAPGraph,
    clean_tokens: Tokens,
    metric: MetricFn,
    transcoders: dict[LayerIndex, HookedTranscoder],
    attn_saes: dict[LayerIndex, HookedSAE],
    *,
    corrupt_tokens: Tokens | None = None,
    ablate_nodes: str | bool = "zero",  # options [False, "bm", "zero"]
    ablate_errors: str | bool = False,  # options [False, "bm", "zero"]
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
    freeze_attention: bool = False,
):
    if corrupt_tokens is None:
        corrupt_tokens = clean_tokens

    # Splice the SAEs and transcoders into the model
    with splice_model_with_saes_and_transcoders(
        model, list(transcoders.values()), list(attn_saes.values())
    ):
        # Cache the activations on the corrupt tokens
        _, cache = model.run_with_cache(corrupt_tokens)

        # Patch the model with corrupt activations
        hooks = get_ablation_hooks(
            graph,
            cache,
            ablate_nodes=ablate_nodes,
            ablate_errors=ablate_errors,
            freeze_attention=freeze_attention,
            first_ablated_layer=first_ablated_layer,
        )

        # Run the forward pass on the clean tokens
        with model.hooks(fwd_hooks=hooks):
            return metric(model, clean_tokens)


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
        # Interpolate between clean, corrupt to get the value
        # c = 0 --> corrupt value
        # c = 1 --> clean value
        clean_value = get_node_act(clean_cache, node)
        corrupt_value = get_node_act(corrupt_cache, node)
        value = (clean_value - corrupt_value) * coefficient + corrupt_value

        # Define the hook function
        module_name, layer, token_idx, feature_idx = parse_node_name(node)
        hook_name = get_hook_name(module_name, layer)
        hook_fn = partial(
            node_patch_hook, token_idx=token_idx, feature_idx=feature_idx, value=value
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


def get_ablation_hooks(
    graph: EAPGraph,
    cache: tl.ActivationCache,
    *,
    ablate_nodes: str | bool = "zero",  # options [False, "bm", "zero"]
    ablate_errors: str | bool = False,  # options [False, "bm", "zero"]
    freeze_attention: bool = False,
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
) -> list[tuple]:
    """Return hooks to ablate a model outside a graph"""

    def patch_act_post_hook(act, hook, layer: LayerIndex, module_name: ModuleName):
        assert hook.name.endswith("hook_sae_acts_post")
        feature_acts = cache[hook.name]
        mask = get_mask(graph, module_name, layer, feature_acts)

        if ablate_nodes == "bm":
            ablate_acts = cache[hook.name].mean(0, keepdim=True)
        elif ablate_nodes == "zero":
            ablate_acts = torch.zeros_like(act)
        else:
            ablate_acts = act

        act[:, mask] = ablate_acts[:, mask]
        return act

    def patch_error_hook(act, hook, layer, module_name):
        assert hook.name.endswith("hook_sae_error")

        if ablate_errors == "bm":
            ablate_acts = cache[hook.name].mean(0, keepdim=True)
        elif ablate_errors == "zero":
            ablate_acts = torch.zeros_like(act)
        else:
            ablate_acts = act

        act = ablate_acts
        return act

    def patch_pattern_hook(act, hook, layer, module_name):
        assert hook.name.endswith("attn.hook_pattern")
        if freeze_attention:
            return cache[hook.name]
        else:
            return act

    fwd_hooks = []
    for layer in range(first_ablated_layer, cache.model.cfg.n_layers):
        # Attention pattern hooks
        fwd_hooks.append(
            (
                f"blocks.{layer}.attn.hook_pattern",
                partial(patch_pattern_hook, layer=layer, module_name="attn"),
            )
        )
        # Attention SAE hooks
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


# def add_ablation_hooks_to_model(
#     model: tl.HookedTransformer,
#     graph: EAPGraph,
#     tokens: Int[Tensor, "batch seq"],
#     transcoders,  # list
#     attn_saes,  # list
#     ablate_nodes: str | bool = "zero",  # options [False, "bm", "zero"]
#     ablate_errors: str | bool = False,  # options [False, "bm", "zero"]
#     first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
#     freeze_attention: bool = False,
# ) -> tl.HookedTransformer:
#     """Cache the activations of the model on a circuit, then return the model with relevant ablation hooks added"""
#     assert ablate_errors in [False, "bm", "zero"]
#     assert ablate_nodes in [False, "bm", "zero"]
#     # Do clean FP to get batchmean (bm) feature acts, and reconstructions
#     # No patching is done during this FP, so the hook fns return nothing
#     # Note: MLP transcoders are a bit more fiddly than attn-SAEs, requiring us to cache at both mlp_in and mlp_out
#     mlp_bm_feature_act_cache = {}
#     attn_bm_feature_act_cache = {}
#     mlp_recons_cache = {}
#     mlp_error_cache = {}
#     attn_error_cache = {}

#     def mlp_in_cache_hook(act, hook, layer):
#         assert hook.name.endswith("ln2.hook_normalized")
#         t = transcoders[layer]
#         feature_acts = torch.relu(
#             einsum(
#                 act - t.b_dec,
#                 t.W_enc,
#                 "b s d_model, d_model n_features -> b s n_features",
#             )
#             + t.b_enc
#         )
#         recons = feature_acts @ t.W_dec + t.b_dec_out

#         mlp_bm_feature_act_cache[layer] = feature_acts.mean(0)
#         mlp_recons_cache[layer] = recons

#     def mlp_out_cache_hook(act, hook, layer):
#         assert hook.name.endswith("mlp_out")
#         mlp_error_cache[layer] = act - mlp_recons_cache[layer]

#     def attn_cache_hook(act, hook, layer):
#         assert hook.name.endswith("hook_z")
#         sae = attn_saes[layer]
#         z_concat = rearrange(act, "b s n_heads d_head -> b s (n_heads d_head)")
#         feature_acts = torch.relu(
#             einsum(
#                 z_concat - sae.b_dec,
#                 sae.W_enc,
#                 "b s d_model, d_model n_features -> b s n_features",
#             )
#             + sae.b_enc
#         )
#         recons_concat = feature_acts @ sae.W_dec + sae.b_dec
#         recons = rearrange(
#             recons_concat,
#             "b s (n_heads d_head) -> b s n_heads d_head",
#             n_heads=model.cfg.n_heads,
#         )

#         attn_bm_feature_act_cache[layer] = feature_acts.mean(0)
#         attn_error_cache[layer] = act - recons

#     model.reset_hooks()
#     for layer in range(model.cfg.n_layers):
#         model.add_hook(
#             f"blocks.{layer}.ln2.hook_normalized",
#             partial(mlp_in_cache_hook, layer=layer),
#             "fwd",
#         )
#         model.add_hook(
#             f"blocks.{layer}.hook_mlp_out",
#             partial(mlp_out_cache_hook, layer=layer),
#             "fwd",
#         )
#         model.add_hook(
#             f"blocks.{layer}.attn.hook_z", partial(attn_cache_hook, layer=layer), "fwd"
#         )

#     # Run forward pass and populate all caches
#     _, pattern_cache = model.run_with_cache(
#         tokens, return_type="loss", names_filter=lambda x: x.endswith("pattern")
#     )
#     assert len(pattern_cache) > 0
#     assert len(mlp_bm_feature_act_cache) > 0
#     assert len(attn_bm_feature_act_cache) > 0
#     assert len(mlp_recons_cache) > 0
#     assert len(mlp_error_cache) > 0
#     assert len(attn_error_cache) > 0

#     # Now do FP where we patch nodes not in graph with their batchmeaned values
#     model.reset_hooks()
#     mlp_ablated_recons = {}

#     def freeze_pattern_hook(act, hook):
#         assert hook.name.endswith("pattern")
#         return pattern_cache[hook.name]

#     def mlp_out_ablated_recons_cache_hook(act, hook, layer):
#         assert hook.name.endswith("ln2.hook_normalized")
#         t = transcoders[layer]
#         feature_acts = torch.relu(
#             einsum(
#                 act - t.b_dec,
#                 t.W_enc,
#                 "b s d_model, d_model n_features -> b s n_features",
#             )
#             + t.b_enc
#         )
#         mask = get_mask(graph, "mlp", layer, feature_acts)
#         ablated_feature_acts = feature_acts.clone()
#         ablated_feature_acts[:, mask] = mlp_bm_feature_act_cache[layer][mask]
#         mlp_ablated_recons[layer] = ablated_feature_acts @ t.W_dec + t.b_dec_out

#     def mlp_patch_hook(act, hook, layer):
#         assert hook.name.endswith("mlp_out")

#         if ablate_nodes == "bm":
#             recons = mlp_ablated_recons[layer]
#         elif ablate_nodes == "zero":
#             recons = torch.zeros_like(mlp_ablated_recons[layer])
#         else:
#             # NOTE: This requires us to change the way we're computing the cached objects.
#             raise NotImplementedError

#         if ablate_errors == "bm":
#             error = mlp_error_cache[layer].mean(dim=0, keepdim=True)
#         elif ablate_errors == "zero":
#             error = torch.zeros_like(mlp_error_cache[layer])
#         else:
#             error = mlp_error_cache[layer]

#         return recons + error

#     def attn_patch_hook(act, hook, layer):
#         assert hook.name.endswith("hook_z")
#         sae = attn_saes[layer]
#         z_concat = rearrange(act, "b s n_heads d_head -> b s (n_heads d_head)")
#         feature_acts = torch.relu(
#             einsum(
#                 z_concat - sae.b_dec,
#                 sae.W_enc,
#                 "b s d_model, d_model n_features -> b s n_features",
#             )
#             + sae.b_enc
#         )

#         mask = get_mask(graph, "attn", layer, feature_acts)
#         ablated_feature_acts = feature_acts.clone()
#         ablated_feature_acts[:, mask] = attn_bm_feature_act_cache[layer][mask]
#         ablated_recons_concat = ablated_feature_acts @ sae.W_dec + sae.b_dec
#         ablated_recons = rearrange(
#             ablated_recons_concat,
#             "b s (n_heads d_head) -> b s n_heads d_head",
#             n_heads=model.cfg.n_heads,
#         )

#         if ablate_nodes == "bm":
#             recons = ablated_recons
#         elif ablate_nodes == "zero":
#             recons = torch.zeros_like(ablated_recons)
#         else:  # no ablation
#             raise NotImplementedError

#         if ablate_errors == "bm":
#             error = attn_error_cache[layer].mean(dim=0, keepdim=True)
#         elif ablate_errors == "zero":
#             error = torch.zeros_like(attn_error_cache[layer])
#         else:
#             error = attn_error_cache[layer]

#         return recons + error

#     for layer in range(first_ablated_layer, model.cfg.n_layers):
#         model.add_hook(
#             f"blocks.{layer}.ln2.hook_normalized",
#             partial(mlp_out_ablated_recons_cache_hook, layer=layer),
#             "fwd",
#         )
#         model.add_hook(
#             f"blocks.{layer}.hook_mlp_out", partial(mlp_patch_hook, layer=layer), "fwd"
#         )
#         model.add_hook(
#             f"blocks.{layer}.attn.hook_z", partial(attn_patch_hook, layer=layer), "fwd"
#         )
#         if freeze_attention:
#             model.add_hook(
#                 f"blocks.{layer}.attn.hook_pattern", freeze_pattern_hook, "fwd"
#             )

#     return model
