from einops import einsum, rearrange
import torch
from jaxtyping import Int, Float
from functools import partial
from circuit_finder.core.types import MetricFn, Tokens
from circuit_finder.patching.eap_graph import EAPGraph
from torch import Tensor
import transformer_lens as tl

from transcoders_slim.transcoder import Transcoder
from circuit_finder.pretrained.load_mlp_transcoders import ts_tc_to_hooked_tc
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
    model: tl.HookedTransformer,
    graph: EAPGraph,
    tokens: Int[Tensor, "batch seq"],
    metric: MetricFn,
    transcoders,  # list
    attn_saes,  # list
    ablate_nodes: str | bool = "zero",  # options [False, "bm", "zero"]
    ablate_errors: str | bool = False,  # options [False, "bm", "zero"]
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
    freeze_attention: bool = False,
):
    model = add_ablation_hooks_to_model(
        model,
        graph,
        tokens,
        transcoders,
        attn_saes,
        ablate_nodes,
        ablate_errors,
        first_ablated_layer,
        freeze_attention,
    )
    return metric(model, tokens)


def get_forward_cache(
    model: tl.HookedSAETransformer,
    tokens: Tokens,
    transcoders: list[Transcoder] | list[HookedTranscoder],
    attn_saes: list[HookedSAE],
) -> tl.ActivationCache:
    hooked_transcoders = []
    for tc in transcoders:
        if isinstance(tc, Transcoder):
            tc = ts_tc_to_hooked_tc(tc)
        hooked_transcoders.append(tc)

    # Run model
    with HookedTranscoderReplacementContext(
        model,  # type: ignore
        transcoders=hooked_transcoders,
    ) as context:
        with model.saes(saes=attn_saes):  # type: ignore
            _, cache = model.run_with_cache(tokens)

    return cache


def add_ablation_hooks_to_model(
    model: tl.HookedTransformer,
    graph: EAPGraph,
    tokens: Int[Tensor, "batch seq"],
    transcoders,  # list
    attn_saes,  # list
    ablate_nodes: str | bool = "zero",  # options [False, "bm", "zero"]
    ablate_errors: str | bool = False,  # options [False, "bm", "zero"]
    first_ablated_layer: int = 2,  # Marks et al don't ablate first 2 layers
    freeze_attention: bool = False,
) -> tl.HookedTransformer:
    """Cache the activations of the model on a circuit, then return the model with relevant ablation hooks added"""
    assert ablate_errors in [False, "bm", "zero"]
    assert ablate_nodes in [False, "bm", "zero"]
    # Do clean FP to get batchmean (bm) feature acts, and reconstructions
    # No patching is done during this FP, so the hook fns return nothing
    # Note: MLP transcoders are a bit more fiddly than attn-SAEs, requiring us to cache at both mlp_in and mlp_out
    mlp_bm_feature_act_cache = {}
    attn_bm_feature_act_cache = {}
    mlp_recons_cache = {}
    mlp_error_cache = {}
    attn_error_cache = {}

    def mlp_in_cache_hook(act, hook, layer):
        assert hook.name.endswith("ln2.hook_normalized")
        t = transcoders[layer]
        feature_acts = torch.relu(
            einsum(
                act - t.b_dec,
                t.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + t.b_enc
        )
        recons = feature_acts @ t.W_dec + t.b_dec_out

        mlp_bm_feature_act_cache[layer] = feature_acts.mean(0)
        mlp_recons_cache[layer] = recons

    def mlp_out_cache_hook(act, hook, layer):
        assert hook.name.endswith("mlp_out")
        mlp_error_cache[layer] = act - mlp_recons_cache[layer]

    def attn_cache_hook(act, hook, layer):
        assert hook.name.endswith("hook_z")
        sae = attn_saes[layer]
        z_concat = rearrange(act, "b s n_heads d_head -> b s (n_heads d_head)")
        feature_acts = torch.relu(
            einsum(
                z_concat - sae.b_dec,
                sae.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + sae.b_enc
        )
        recons_concat = feature_acts @ sae.W_dec + sae.b_dec
        recons = rearrange(
            recons_concat,
            "b s (n_heads d_head) -> b s n_heads d_head",
            n_heads=model.cfg.n_heads,
        )

        attn_bm_feature_act_cache[layer] = feature_acts.mean(0)
        attn_error_cache[layer] = act - recons

    model.reset_hooks()
    for layer in range(model.cfg.n_layers):
        model.add_hook(
            f"blocks.{layer}.ln2.hook_normalized",
            partial(mlp_in_cache_hook, layer=layer),
            "fwd",
        )
        model.add_hook(
            f"blocks.{layer}.hook_mlp_out",
            partial(mlp_out_cache_hook, layer=layer),
            "fwd",
        )
        model.add_hook(
            f"blocks.{layer}.attn.hook_z", partial(attn_cache_hook, layer=layer), "fwd"
        )

    # Run forward pass and populate all caches
    _, pattern_cache = model.run_with_cache(
        tokens, return_type="loss", names_filter=lambda x: x.endswith("pattern")
    )
    assert len(pattern_cache) > 0
    assert len(mlp_bm_feature_act_cache) > 0
    assert len(attn_bm_feature_act_cache) > 0
    assert len(mlp_recons_cache) > 0
    assert len(mlp_error_cache) > 0
    assert len(attn_error_cache) > 0

    # Now do FP where we patch nodes not in graph with their batchmeaned values
    model.reset_hooks()
    mlp_ablated_recons = {}

    def freeze_pattern_hook(act, hook):
        assert hook.name.endswith("pattern")
        return pattern_cache[hook.name]

    def mlp_out_ablated_recons_cache_hook(act, hook, layer):
        assert hook.name.endswith("ln2.hook_normalized")
        t = transcoders[layer]
        feature_acts = torch.relu(
            einsum(
                act - t.b_dec,
                t.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + t.b_enc
        )
        mask = get_mask(graph, "mlp", layer, feature_acts)
        ablated_feature_acts = feature_acts.clone()
        ablated_feature_acts[:, mask] = mlp_bm_feature_act_cache[layer][mask]
        mlp_ablated_recons[layer] = ablated_feature_acts @ t.W_dec + t.b_dec_out

    def mlp_patch_hook(act, hook, layer):
        assert hook.name.endswith("mlp_out")

        if ablate_nodes == "bm":
            recons = mlp_ablated_recons[layer]
        elif ablate_nodes == "zero":
            recons = torch.zeros_like(mlp_ablated_recons[layer])
        else:
            # NOTE: This requires us to change the way we're computing the cached objects.
            raise NotImplementedError

        if ablate_errors == "bm":
            error = mlp_error_cache[layer].mean(dim=0, keepdim=True)
        elif ablate_errors == "zero":
            error = torch.zeros_like(mlp_error_cache[layer])
        else:
            error = mlp_error_cache[layer]

        return recons + error

    def attn_patch_hook(act, hook, layer):
        assert hook.name.endswith("hook_z")
        sae = attn_saes[layer]
        z_concat = rearrange(act, "b s n_heads d_head -> b s (n_heads d_head)")
        feature_acts = torch.relu(
            einsum(
                z_concat - sae.b_dec,
                sae.W_enc,
                "b s d_model, d_model n_features -> b s n_features",
            )
            + sae.b_enc
        )

        mask = get_mask(graph, "attn", layer, feature_acts)
        ablated_feature_acts = feature_acts.clone()
        ablated_feature_acts[:, mask] = attn_bm_feature_act_cache[layer][mask]
        ablated_recons_concat = ablated_feature_acts @ sae.W_dec + sae.b_dec
        ablated_recons = rearrange(
            ablated_recons_concat,
            "b s (n_heads d_head) -> b s n_heads d_head",
            n_heads=model.cfg.n_heads,
        )

        if ablate_nodes == "bm":
            recons = ablated_recons
        elif ablate_nodes == "zero":
            recons = torch.zeros_like(ablated_recons)
        else:  # no ablation
            raise NotImplementedError

        if ablate_errors == "bm":
            error = attn_error_cache[layer].mean(dim=0, keepdim=True)
        elif ablate_errors == "zero":
            error = torch.zeros_like(attn_error_cache[layer])
        else:
            error = attn_error_cache[layer]

        return recons + error

    for layer in range(first_ablated_layer, model.cfg.n_layers):
        model.add_hook(
            f"blocks.{layer}.ln2.hook_normalized",
            partial(mlp_out_ablated_recons_cache_hook, layer=layer),
            "fwd",
        )
        model.add_hook(
            f"blocks.{layer}.hook_mlp_out", partial(mlp_patch_hook, layer=layer), "fwd"
        )
        model.add_hook(
            f"blocks.{layer}.attn.hook_z", partial(attn_patch_hook, layer=layer), "fwd"
        )
        if freeze_attention:
            model.add_hook(
                f"blocks.{layer}.attn.hook_pattern", freeze_pattern_hook, "fwd"
            )

    return model
