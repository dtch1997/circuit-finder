# type: ignore
"""Linear Edge Attribution Patching.

Similar to Edge Attribution Patching, calculates the effect of edges on some downstream metric.
However, takes advantage of linearity afforded by transcoders and MLPs to parallelize
"""
#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("/root/circuit-finder")
import torch
import transformer_lens as tl
from typing import Callable
from transcoders_slim.transcoder import Transcoder
from torch import Tensor
from jaxtyping import Int, Float
from dataclasses import dataclass
from einops import rearrange, einsum, repeat
import gc
from eindex import eindex

from circuit_finder.core.types import (
    Node,
    Edge,
    Attrib,
    Tokens,
    LayerIndex,
    FeatureIndex,
    TokenIndex,
    ModuleName,
    parse_node_name,
    HookNameFilterFn,
)
from circuit_finder.constants import device
from circuit_finder.core.hooked_sae import HookedSAE


def preprocess_attn_saes(
    attn_saes_in: dict[LayerIndex, HookedSAE],
    model: tl.HookedTransformer,
) -> dict[LayerIndex, HookedSAE]:
    """Preprocess the SAEs to have the same feature dimension."""
    # NOTE: currently do this by chopping off features, but this loses information
    # A simple fix is to extend all SAEs to have the same feature dimension by adding zeros
    # TODO: implement fix
    attn_saes = {}
    for layer, sae in attn_saes_in.items():
        # chop off features so all have same size TODO fix this
        sae.W_enc = torch.nn.Parameter(sae.W_enc[:, :24576], requires_grad=False)
        sae.b_enc = torch.nn.Parameter(sae.b_enc[:24576], requires_grad=False)
        sae.W_dec = torch.nn.Parameter(sae.W_dec[:24576, :], requires_grad=False)
        sae.b_dec = torch.nn.Parameter(sae.b_dec, requires_grad=False)

        # normalize so decoder ols have norm=1 when viewed as resid vectors
        # TODO: Daniel doesn't understand why this is necessary
        W_dec_z = rearrange(
            sae.W_dec, "nf (nhead dhead) -> nf nhead dhead", nhead=model.cfg.n_heads
        )
        W_dec_resid = einsum(
            W_dec_z, model.W_O[layer], "nf nhead dhead, nhead dhead dmodel -> nf dmodel"
        )
        norms = W_dec_resid.norm(dim=-1, keepdim=True)  # [nf, 1]

        normed_sae = sae
        normed_sae.W_dec = torch.nn.Parameter(sae.W_dec / norms)
        normed_sae.W_enc = torch.nn.Parameter(sae.W_enc * norms.T)
        normed_sae.b_enc = torch.nn.Parameter(sae.b_enc * norms.squeeze())


        attn_saes[layer] = normed_sae

    return attn_saes


@dataclass
class LEAPConfig:
    threshold: float = 0.01
    contrast_pairs: bool = False
    chained_attribs: bool = True
    store_error_attribs: bool = False
    qk_enabled : bool = True


class IndirectLEAP:
    """
    Linear Edge Attribution Patching

    Similar to edge attribution patching, but fully linearises the network to compute attributions exactly
    """

    # Models
    model: tl.HookedTransformer
    attn_saes: dict[LayerIndex, tl.HookedSAE]
    transcoders: dict[LayerIndex, Transcoder]

    # Data
    tokens: Tokens
    corrupt_tokens: Tokens

    # Intermediate computations
    mlp_feature_acts: Float[Tensor, "seq layer d_trans"]
    mlp_is_active: Float[Tensor, "seq layer d_trans"]
    attn_feature_acts: Float[Tensor, "seq layer d_sae"]
    attn_is_active: Float[Tensor, "seq layer d_sae"]
    mlp_errors: Float[Tensor, "seq layer d_model"]
    attn_errors: Float[Tensor, "seq layer d_model"]

    # Graph
    # (edge, vals)
    # vals = (node_node_grad, node_node_attrib, edge_metric_grad, edge_metric_attrib)
    graph: list[tuple[Edge, Attrib]]

    def __init__(
        self,
        cfg: LEAPConfig,
        tokens: Int[torch.Tensor, "batch seq"],
        model: tl.HookedTransformer,
        attn_saes: dict[LayerIndex, tl.HookedSAE],  # layer index: attn-out SAE
        transcoders: dict[LayerIndex, Transcoder],  # layer index: mlp transcoder
        metric: Callable[
            [tl.HookedTransformer, Tokens],
            Float[Tensor, "batch d_model"],
        ],
        corrupt_tokens: Int[
            torch.Tensor, "batch seq"
        ] = None,  # only specify if contrast_pairs = True
    ):
        self.cfg = cfg
        self.tokens = tokens
        self.model = model
        self.attn_saes = attn_saes
        self.transcoders = transcoders
        self.metric = metric
        self.corrupt_tokens = corrupt_tokens

        if self.cfg.contrast_pairs:
            assert self.tokens.shape == self.corrupt_tokens.shape

        # Store some params for convenience
        self.d_trans = transcoders[0].W_enc.size(1)
        self.d_sae = attn_saes[0].W_enc.size(1)
        self.batch, self.n_seq = tokens.size()
        self.d_model = model.cfg.d_model
        self.n_layers = model.cfg.n_layers
        self.layers = range(model.cfg.n_layers)

        # Cache feature acts, pattern, layernorm scales, SAE errors, active feature IDs
        self.get_initial_cache()
        
        # Cache gradients at outputs of heads, for use in calculating sum-over-paths attribs
        self.get_grads()
      
        # Kissane et al's attention SAEs are trained at hook_z, concatenated across head_idx dimension.
        # Here we process the decoder matrices so their columns can be viewed as residual vectors.
        attn_all_W_dec_z_cat = torch.stack(
            [attn_saes[i].W_dec for i in self.layers]
        )  # [layer d_sae d_model]
        attn_all_W_dec_z = rearrange(
            attn_all_W_dec_z_cat,
            "layer d_sae (n_heads d_head) -> layer d_sae n_heads d_head",
            n_heads=model.cfg.n_heads,
        )
        self.attn_all_W_dec_resid = einsum(
            attn_all_W_dec_z,
            model.W_O,
            "layer d_sae n_heads d_head , layer n_heads d_head d_model -> layer d_sae d_model",
        )

        # NOTE: We'll store (edge, vals) pairs here.
        # Initialise by making the metric an important node, by hand.
        # edge = (downstream_node, upstream_node)
        # node = "{module_name}.{layer}.{pos}.{feature_id}"
        # vals = (node_node_grad, node_node_attrib, edge_metric_grad, edge_metric_attrib)
        self.graph: list[tuple[Edge, Attrib]] = [
            (("null", f"metric.{self.n_layers}.{seq}.0"), (0.0, 0.0, 1.0, 0.0), None)
            for seq in range(self.n_seq)
        ]

        self.error_graph = []

    def get_initial_cache(self):
        """Run model on tokens. Grab acts at mlp_in, and run them through
        transcoders to get all transcoder feature acts. Grab acts at hook_z,
        concatenate on head_idx dim, and run through attn_sae to get all
        attention SAE feature acts"""
        # This code is verbose but totally trivial!

        # Run model and cache the acts we need
        names_filter: HookNameFilterFn = lambda x: (
            x.endswith("ln2.hook_normalized")
            or x.endswith("mlp_out")
            or x.endswith("hook_z")
            or x.endswith("pattern")
            or x.endswith("hook_scale")
            or (x.endswith("ln1.hook_normalized") and self.cfg.qk_enabled)
            or (x.endswith("_k") and self.cfg.qk_enabled)
            or (x.endswith("_q") and self.cfg.qk_enabled)
        )
        _, cache = self.model.run_with_cache(
            self.tokens, return_type="loss", names_filter=names_filter
        )

        if self.cfg.contrast_pairs:
            names_filter = lambda x: (
                x.endswith("ln2.hook_normalized")
                or x.endswith("mlp_out")
                or x.endswith("hook_z")
            )
            _, corrupt_cache = self.model.run_with_cache(
                self.corrupt_tokens, return_type="loss", names_filter=names_filter
            )

        # Save attention patterns (avg over batch and head) and layernorm scales (avg over batch)
        if self.cfg.qk_enabled:
            self.pattern : Float[Tensor, "layer head q_pos k_pos"] = torch.stack(
            [cache["pattern", layer].mean(0) for layer in self.layers]
            )

            self.q : Float[Tensor, "layer pos n_heads_head"] = torch.stack(
                [cache[f'blocks.{layer}.attn.hook_q'].mean(0) for layer in self.layers]
            )
            self.k : Float[Tensor, "layer pos n_heads_head"] = torch.stack(
                [cache[f'blocks.{layer}.attn.hook_k'].mean(0) for layer in self.layers]
            )

            self.attn_in : Float[Tensor, "layer pos d_model"] = torch.stack(
            [cache[f'blocks.{layer}.ln1.hook_normalized'].mean(0) for layer in self.layers]
            )

        self.headmean_pattern: Float[Tensor, "layer q_pos k_pos"] = torch.stack(
            [cache["pattern", layer].mean(0) for layer in self.layers]
        )

        self.attn_layernorm_scales: Float[Tensor, "layer pos"] = torch.stack(
            [
                cache[f"blocks.{layer}.ln1.hook_scale"].mean(dim=0)
                for layer in self.layers
            ]
        )  # [layer pos]
        self.mlp_layernorm_scales: Float[Tensor, "layer pos"] = torch.stack(
            [
                cache[f"blocks.{layer}.ln2.hook_scale"].mean(dim=0)
                for layer in self.layers
            ]
        )  # [layer pos]

        # Save feature acts
        # Initialise empty tensors to store feature acts and is_active (both batchmeaned)
        self.mlp_feature_acts: Float[Tensor, "seq layer d_trans"] = torch.empty(
            self.n_seq, self.n_layers, self.d_trans, device=device
        )
        self.mlp_is_active: Float[Tensor, "seq layer d_trans"] = torch.empty(
            self.n_seq, self.n_layers, self.d_trans, device=device
        )
        self.attn_feature_acts: Float[Tensor, "seq layer d_sae"] = torch.empty(
            self.n_seq, self.n_layers, self.d_sae, device=device
        )
        self.attn_is_active: Float[Tensor, "seq layer d_sae"] = torch.empty(
            self.n_seq, self.n_layers, self.d_sae, device=device
        )
        self.mlp_errors: Float[Tensor, "seq layer d_model"] = torch.empty(
            self.n_seq, self.n_layers, self.d_model, device=device
        )
        self.attn_errors: Float[Tensor, "seq layer d_model"] = torch.empty(
            self.n_seq, self.n_layers, self.d_model, device=device
        )

        # Add feature acts to the empty tensors, layer by layer
        for layer in range(self.n_layers):
            mlp_in_pt = f"blocks.{layer}.ln2.hook_normalized"
            mlp_out_pt = f"blocks.{layer}.hook_mlp_out"
            attn_out_pt = f"blocks.{layer}.attn.hook_z"

            # Get MLP feature acts and recons errors
            mlp_recons, mlp_feature_acts = self.transcoders[layer](cache[mlp_in_pt])[:2]
            if self.cfg.contrast_pairs:  # in contrast pairs case, feature_acts now really refers to the change in feature acts
                mlp_feature_acts -= self.transcoders[layer](corrupt_cache[mlp_in_pt])[1]
            self.mlp_feature_acts[:, layer, :] = mlp_feature_acts.mean(0)
            self.mlp_is_active[:, layer, :] = (mlp_feature_acts > 0).float().mean(0)
            self.mlp_errors[:, layer, :] = (cache[mlp_out_pt] - mlp_recons).mean(0)

            # Get attention feature acts and recons errors (remember to be careful with z concatenation!)
            z_concat = rearrange(
                cache[attn_out_pt],
                "batch seq n_heads d_head -> batch seq (n_heads d_head)",
            )
            attn_recons, sae_cache = self.attn_saes[layer].run_with_cache(
                z_concat, names_filter="hook_sae_acts_post"
            )
            attn_feature_acts = sae_cache["hook_sae_acts_post"]

            if self.cfg.contrast_pairs:
                z_concat = rearrange(
                    corrupt_cache[attn_out_pt],
                    "batch seq n_heads d_head -> batch seq (n_heads d_head)",
                )
                attn_recons, sae_cache = self.attn_saes[layer].run_with_cache(
                    z_concat, names_filter="hook_sae_acts_post"
                )
                attn_feature_acts -= sae_cache["hook_sae_acts_post"]

            self.attn_feature_acts[:, layer, :] = attn_feature_acts.mean(0)
            self.attn_is_active[:, layer, :] = (attn_feature_acts > 0).float().mean(0)
            z_error = rearrange(
                (attn_recons - z_concat).mean(0),
                "seq (n_heads d_head) -> seq n_heads d_head",
                n_heads=self.model.cfg.n_heads,
            )
            resid_error = einsum(
                z_error,
                self.model.W_O[layer],
                "seq n_heads d_head, n_heads d_head d_model -> seq d_model",
            )
            self.attn_errors[:, layer, :] = resid_error

        # Get ids of active features
        self.mlp_active_feature_ids = torch.where(self.mlp_is_active.sum(0) > 0)
        self.attn_active_feature_ids = torch.where(self.attn_is_active.sum(0) > 0)

        del cache
        torch.cuda.empty_cache()
        gc.collect()

    def get_grads(self):
        grad_cache = {}
        def grad_cache_hook(grad, hook):
            assert hook.name.endswith("mlp_out") or hook.name.endswith("attn_out")
            grad_cache[hook.name] = grad.sum(0) # sum not mean!!

        self.model.reset_hooks()
        for layer in self.layers:
            self.model.add_hook(f'blocks.{layer}.hook_mlp_out', grad_cache_hook, "bwd")
            self.model.add_hook(f'blocks.{layer}.hook_attn_out', grad_cache_hook, "bwd")

        m = self.metric(self.model, self.tokens)
        m.backward()
        
        self.mlp_out_grads : Float[Tensor, "layer seq d_model"] = torch.stack(
            [
                grad_cache[f'blocks.{layer}.hook_mlp_out']
                for layer in self.layers
            ]
        )

        self.attn_out_grads : Float[Tensor, "layer seq d_model"] = torch.stack(
            [
                grad_cache[f'blocks.{layer}.hook_mlp_out']
                for layer in self.layers
            ]
        )


    def metric_step(self):
        """Step 0 of circuit discovery: get attributions from each node to the metric."""
        (imp_down_feature_ids, imp_down_pos) = (
            self.get_imp_properties("metric", self.n_layers)
        )
        self.model.reset_hooks()
        self.model.zero_grad()
        grad_cache = {}
        hook_pt = f"blocks.{self.n_layers-1}.hook_resid_post"

        def grad_cache_hook(grad, hook):
            grad_cache[hook.name] = grad.sum(0)  # [seq, d_model] # sum not mean!

        self.model.add_hook(hook_pt, grad_cache_hook, "bwd")
        m = self.metric(self.model, self.tokens)
        m.backward()  # TODO don't actually need backward through whole model. If m is linear, we can disable autograd!
        grad = grad_cache[hook_pt][imp_down_pos]
        self.compute_and_save_attribs(
            grad,
            "metric",
            self.n_layers,
            imp_down_feature_ids,
            imp_down_pos,
            edge_type=None
                            )

    def mlp_step(self, down_layer):
        """For each imp node at this MLP, compute attrib wrt all previous nodes"""

        # Get the imp features coming out of the MLP, and the positions at which they're imp
        (imp_down_feature_ids, imp_down_pos) = (
            self.get_imp_properties("mlp", down_layer)
        )
        # For each imp downstream (feature_id, pos) pair, get batchmeaned is_active, and the corresponding encoder row
        imp_active = self.mlp_is_active[
            imp_down_pos, down_layer, imp_down_feature_ids
        ]  # [imp_id]
        imp_enc_cols = self.transcoders[down_layer].W_enc[
            :, imp_down_feature_ids
        ]  # [d_model, imp_id]
        imp_layernorm_scales = self.mlp_layernorm_scales[
            down_layer, imp_down_pos
        ]  # [imp_id, 1]

        # The grad of these imp feature acts is just the corresponding row of W_enc (scaled by layernorm)
        grad = einsum(
            imp_active, imp_enc_cols, "imp_id, d_model imp_id -> imp_id d_model"
        )
        grad /= imp_layernorm_scales

        self.compute_and_save_attribs(
            grad,
            "mlp",
            down_layer,
            imp_down_feature_ids,
            imp_down_pos,
            edge_type=None
        )


    def ov_step(self, down_layer):
        """For each imp node at this attention layer, compute attrib wrt all previous nodes *via the OV circuit*"""

        # Get the imp features coming out of the attention layer, and the positions at which they're imp
        (imp_down_feature_ids, imp_down_pos) = (
            self.get_imp_properties("attn", down_layer)
        )

        # For each imp downstream (feature_id, pos) pair, get batchmeaned is_active, encoder row, and pattern
        imp_active = self.attn_is_active[
            imp_down_pos, down_layer, imp_down_feature_ids
        ]  # [imp_id]
        imp_enc_rows = self.attn_saes[down_layer].W_enc[
            :, imp_down_feature_ids
        ]  # [d_model, imp_id]
        imp_enc_rows_z = rearrange(
            imp_enc_rows,
            "(head_id d_head) imp_id -> head_id d_head imp_id",
            head_id=self.model.cfg.n_heads,
        )
        imp_patterns = self.headmean_pattern[
            down_layer, :, imp_down_pos, :
        ]  # [head imp_id, kpos]
        imp_layernorm_scales = self.attn_layernorm_scales[down_layer].unsqueeze(
            1
        )  # [kpos, 1, 1]

        grad = einsum(
            imp_active,
            imp_enc_rows_z,
            self.model.W_V[down_layer],
            imp_patterns,
            "imp_id, head_id d_head imp_id, head_id d_model d_head, head_id imp_id kpos -> kpos imp_id d_model",
        )
        grad /= imp_layernorm_scales

        self.compute_and_save_attribs(
            grad,
            "attn",
            down_layer,
            imp_down_feature_ids,
            imp_down_pos,
            edge_type="ov"
        )

    def pattern_grad_approx(self, pattern):
        # Pattern can have any size since we apply the function elementwise.
        # True pattern derivative is p - p^2, but this underestimates effect of decreasing
        # a high score. Whereas using derivative = p leads to an overestimate. 
        # So we do something hacky :)
        # Create masks for the conditions
        mask_greater = pattern > 0.5
        mask_lesser = pattern < 0.5
        
        # Apply the function elementwise
        result = torch.where(mask_greater, pattern - pattern**2, pattern**2)
        
        return result
        

    def q_step(self, down_layer):
        # Get the imp features coming out of the attention layer, and the positions at which they're imp
        (imp_down_feature_ids, imp_down_pos) = (
            self.get_imp_properties("attn", down_layer)
        )

        # For each imp downstream (feature_id, pos) pair, get batchmeaned is_active, encoder row, and pattern
        imp_active = self.attn_is_active[
            imp_down_pos, down_layer, imp_down_feature_ids
        ]  # [imp_id]
        imp_enc_rows = self.attn_saes[down_layer].W_enc[
            :, imp_down_feature_ids
        ]  # [d_model, imp_id]
        imp_enc_rows_z = rearrange(
            imp_enc_rows,
            "(head_id d_head) imp_id -> head_id d_head imp_id",
            head_id=self.model.cfg.n_heads,
        )
        imp_layernorm_scales = self.attn_layernorm_scales[
            down_layer, imp_down_pos
        ]  # [imp_id, 1]

        # TODO: W_dec normalisation?

        # This will make no sense until I send you the equations
        # The cryptic variable names just follow what's handwritten in front of me
        kp = einsum(self.k[down_layer], 
                    self.pattern_grad_approx(self.pattern[down_layer]),
                    "kseq n_heads d_head, n_heads qseq kseq -> n_heads qseq kseq d_head")
        kp_centred = kp - kp.mean(dim=-2, keepdim=True)

        dfa = einsum(
            self.attn_in[down_layer], 
            model.W_V[down_layer], 
            imp_enc_rows_z,
            "kseq d_model_in, n_heads d_model_in d_head, n_heads d_head imp_id -> kseq n_heads imp_id")
        
        grad = einsum(
            model.W_Q[down_layer],
            kp_centred,
            dfa,
            "n_heads d_model d_head, n_heads qseq kseq d_head, kseq n_heads imp_id -> imp_id qseq d_model "
        )  /  self.model.cfg.d_head**0.5

        grad = eindex(grad, torch.tensor(imp_down_pos, dtype=torch.long),
                      "imp_id [imp_id] d_model") # [imp_id, d_model]
        
        grad /= imp_layernorm_scales
        # grad *= imp_active.unsqueeze(1)
        
        seq_expand_grad = torch.zeros([self.n_seq, len(imp_down_pos), self.d_model]).cuda()

        for i, pos in enumerate(imp_down_pos):
            seq_expand_grad[pos, i, :] = grad[i, :]
                
        self.compute_and_save_attribs(
            seq_expand_grad,
            "attn",
            down_layer,
            imp_down_feature_ids,
            imp_down_pos,
            edge_type="q"
        )
    
    def k_step(self, down_layer):
        # Get the imp features coming out of the attention layer, and the positions at which they're imp
        (imp_down_feature_ids, imp_down_pos) = (
            self.get_imp_properties("attn", down_layer)
        )

        # For each imp downstream (feature_id, pos) pair, get batchmeaned is_active, encoder row, and pattern
        imp_active = self.attn_is_active[
            imp_down_pos, down_layer, imp_down_feature_ids
        ]  # [imp_id]
        imp_enc_rows = self.attn_saes[down_layer].W_enc[
            :, imp_down_feature_ids
        ]  # [d_model, imp_id]
        imp_enc_rows_z = rearrange(
            imp_enc_rows,
            "(head_id d_head) imp_id -> head_id d_head imp_id",
            head_id=self.model.cfg.n_heads,
        )
        imp_layernorm_scales = self.attn_layernorm_scales[
            down_layer, imp_down_pos
        ]  # [imp_id, 1]

        # TODO: W_dec normalisation?

        # This will make no sense until I send you the equations
        # The cryptic variable names just follow what's handwritten in front of me
        p_WK_q = einsum(
            self.pattern_grad_approx(self.pattern[down_layer]),
            self.model.W_K[down_layer],
            self.q[down_layer][imp_down_pos, :, :], # [head imp kseq]
            "n_heads qseq kseq, n_heads d_model d_head, imp_id n_heads d_head -> imp_id n_heads kseq d_model")

        dfa = einsum(
            self.attn_in[down_layer], 
            model.W_V[down_layer], 
            imp_enc_rows_z,
            "kseq d_model_in, n_heads d_model_in d_head, n_heads d_head imp_id -> kseq n_heads imp_id")
        
        grad = einsum(
            p_WK_q, dfa,
            "imp_id n_heads kseq d_model, kseq n_heads imp_id -> kseq imp_id d_model"
            ) / self.model.cfg.d_head**0.5

        grad /= imp_layernorm_scales
        # grad *= imp_active.unsqueeze(1)

        self.compute_and_save_attribs(
            grad,
            "attn",
            down_layer,
            imp_down_feature_ids,
            imp_down_pos,
            edge_type="k"
        )



    """ Helper functions """

    # TODO: This should be made an attribute of the graph instead
    def get_imp_properties(
        self, down_module: ModuleName, down_layer: LayerIndex
    ) -> tuple[list[FeatureIndex], list[TokenIndex]]:
        """Get the feature indices and token positiosn of important nodes at a given layer

        Returns:
            imp_feature_ids : list of feature indices of important nodes.
            imp_pos : list of token positions of important nodes.

        These can be zipped together to get the (feature_id, pos) pairs of important nodes.

        module : name of upstream module."""
        imp_feature_ids: list[FeatureIndex] = []
        imp_pos: list[TokenIndex] = []

        # Get all nodes that are currently in the graph
        up_nodes_set: set[Node] = set()
        for edge in [e for e, _ , _ in self.graph]:
            _, upstream = edge
            up_nodes_set.add(upstream)
        up_nodes_deduped: list[Node] = list(up_nodes_set)

        # Filter by module and layer
        # TODO: It seems like we could do this previously but ig it doesn't matter.
        for node in up_nodes_deduped:
            module_, layer_, pos, feature_id = parse_node_name(node)
            if module_ == down_module and layer_ == down_layer:
                imp_feature_ids += [int(feature_id)]
                imp_pos += [int(pos)]
        return imp_feature_ids, imp_pos

    def get_active_mlp_W_dec(
        self, down_layer: LayerIndex
    ) -> tuple[Tensor, Tensor, Tensor]:
        """so we don't have to dot with *every* upstream feature"""
        mlp_up_active_layers = self.mlp_active_feature_ids[0][
            self.mlp_active_feature_ids[0] < down_layer
        ]
        mlp_up_active_feature_ids = self.mlp_active_feature_ids[1][
            self.mlp_active_feature_ids[0] < down_layer
        ]
        mlp_active_W_dec = torch.stack(
            [self.transcoders[i].W_dec for i in range(down_layer)]
        )[mlp_up_active_layers, mlp_up_active_feature_ids, :]
        return mlp_active_W_dec, mlp_up_active_layers, mlp_up_active_feature_ids

    def get_active_attn_W_dec(
        self, down_layer: LayerIndex, down_module: ModuleName
    ) -> tuple[Tensor, Tensor, Tensor]:
        """so we don't have to dot with *every* upstream feature"""
        max_layer = down_layer
        if down_module == "mlp":  # mlp sees attn from same layer!
            max_layer += 1
        attn_up_active_layers = self.attn_active_feature_ids[0][
            self.attn_active_feature_ids[0] < max_layer
        ]
        attn_up_active_feature_ids = self.attn_active_feature_ids[1][
            self.attn_active_feature_ids[0] < max_layer
        ]
        attn_active_W_dec = self.attn_all_W_dec_resid[
            attn_up_active_layers, attn_up_active_feature_ids, :
        ]
        return attn_active_W_dec, attn_up_active_layers, attn_up_active_feature_ids

    def get_up_active_stuff(self, down_module, down_layer, up_module):
        # Returns
        up_active_W_dec: Float[Tensor, "up_active_id n_features d_model"]
        up_active_layers: Float[Tensor, " up_active_id "]
        up_active_feature_ids: Float[Tensor, " up_active_id"]
        up_active_feature_acts: Float[Tensor, " seq up_active_id"]

        assert up_module in ["attn", "mlp"]
        if up_module == "mlp":
            up_active_W_dec, up_active_layers, up_active_feature_ids = (
                self.get_active_mlp_W_dec(down_layer)
            )
            up_active_feature_acts = self.mlp_feature_acts[
                :, up_active_layers, up_active_feature_ids
            ]  # [pos, up_active_id]

        elif up_module == "attn":
            up_active_W_dec, up_active_layers, up_active_feature_ids = (
                self.get_active_attn_W_dec(down_layer, down_module)
            )
            up_active_feature_acts = self.attn_feature_acts[
                :, up_active_layers, up_active_feature_ids
            ]
        return (
            up_active_W_dec,
            up_active_layers,
            up_active_feature_ids,
            up_active_feature_acts,
        )

    def compute_and_save_attribs(
        self,
        grad,
        down_module,
        down_layer,
        imp_down_feature_ids,
        imp_down_pos,
        edge_type=None #options "q", "k", None
    ):
        """grad : [... imp_id, d_model]"""

        for up_module in ["mlp", "attn"]:
            (   up_active_W_dec,
                up_active_layers,
                up_active_feature_ids,
                up_active_feature_acts,
            ) = self.get_up_active_stuff(down_module, down_layer, up_module)

            if down_module == "mlp":
                imp_head_out_metric_grads = self.mlp_out_grads[down_layer, imp_down_pos, :]   # [imp d_model]
                imp_W_dec = self.transcoders[down_layer].W_dec[imp_down_feature_ids, :]
                imp_node_metric_grads = einsum(imp_head_out_metric_grads, imp_W_dec,
                                               "imp_id d_model, imp_id d_model -> imp_id")

            elif down_module == "attn":
                imp_head_out_metric_grads = self.attn_out_grads[down_layer, imp_down_pos, :]   # [imp d_model]
                imp_W_dec = self.attn_all_W_dec_resid[down_layer, imp_down_feature_ids]   # [imp d_model]
                imp_node_metric_grads = einsum(imp_head_out_metric_grads, imp_W_dec,
                                               "imp_id d_model, imp_id d_model -> imp_id")
                
            elif down_module == "metric":
                imp_node_metric_grads = torch.ones(len(imp_down_pos)).cuda()

            # split attrib calc into two cases depending on downstream module type
            # this is because sequence index requires different treatment in the attn case
            if down_module in ["mlp", "metric"]:
                up_active_feature_acts: Float[Tensor, " imp_id up_active_id"] = (
                    up_active_feature_acts[imp_down_pos]
                )   

                node_node_grads = einsum(
                    up_active_W_dec,
                    grad,
                    "up_active_id d_model, imp_id d_model -> imp_id up_active_id",
                )

                node_node_attribs = einsum(
                    node_node_grads,
                    up_active_feature_acts,
                    "imp_id up_active_id, imp_id up_active_id -> imp_id up_active_id",
                )
                if down_layer == 12 and up_module == "attn":
                    self.nn_attribs = node_node_attribs

                edge_metric_grads = einsum(
                    imp_node_metric_grads,
                    node_node_grads,
                    "imp_id, imp_id up_active_id -> imp_id up_active_id",
                )

                edge_metric_attribs = einsum(
                    edge_metric_grads,
                    up_active_feature_acts,
                    "imp_id up_active_id, imp_id up_active_id -> imp_id up_active_id",
                )

                if up_module == "mlp":
                    imp_errors: Float[Tensor, " imp_id layer d_model"] = (
                        self.mlp_errors[imp_down_pos]
                    )
                elif up_module == "attn":
                    imp_errors = self.attn_errors[imp_down_pos]
                error_attribs = einsum(
                    imp_errors[:, :down_layer],
                    grad,
                    "imp_id layer d_model, imp_id d_model -> imp_id layer",
                )

            elif down_module in ["attn"]:
                # Compute attribs of imp nodes wrt all upstream MLP nodes
                node_node_grads = einsum(
                    up_active_W_dec,
                    grad,
                    "up_active_id d_model, seq imp_id d_model -> seq imp_id up_active_id",
                )

                node_node_attribs = einsum(
                    node_node_grads,
                    up_active_feature_acts,
                    "seq imp_id up_active_id, seq up_active_id -> seq imp_id up_active_id",
                )

                edge_metric_grads = einsum(
                    imp_node_metric_grads,
                    node_node_grads,
                    "imp_id, seq imp_id up_active_id -> seq imp_id up_active_id",
                )

                edge_metric_attribs = einsum(
                    edge_metric_grads,
                    up_active_feature_acts,
                    "seq imp_id up_active_id, seq up_active_id -> seq imp_id up_active_id",
                )

                # calculate error attribs (optional)
                error_attribs = None
                if self.cfg.store_error_attribs:
                    if up_module == "mlp":
                        imp_errors = self.mlp_errors[:, :down_layer]
                    elif up_module == "attn":
                        imp_errors = self.attn_errors[:, :down_layer]

                    error_attribs = einsum(
                        imp_errors,
                        grad,
                        "seq layer d_model, seq imp_id d_model -> seq imp_id layer",
                    )
            else:
                print(down_module)
                raise ValueError("down_module must be one of ['mlp', 'attn', 'metric']")

            # attrib can be at most the value of the original downstream feature act
            # mlp_attribs = torch.min(mlp_attribs, imp_mlp_feature_acts)

            # now add important edges to graph, alongside their (node_node_grad, edge_metric_grad, edge_metric_attrib)
            self.add_to_graph(
                node_node_grads,
                node_node_attribs,
                edge_metric_grads,
                edge_metric_attribs,
                error_attribs,
                imp_down_feature_ids,
                imp_down_pos,
                down_module,
                down_layer,
                up_module,
                up_active_layers,
                up_active_feature_ids,
                edge_type=edge_type
            )

    def add_to_graph(
        self,
        node_node_grads,  # [seq, imp_id, up_active_id] if down_module==attn.  otherwise [imp_id, up_active_id]
        node_node_attribs,
        edge_metric_grads,
        edge_metric_attribs,
        error_attribs,  # [seq, imp_id, layer] if down_module==attn.  otherwise [imp_id, layer]
        imp_down_feature_ids,
        imp_down_pos,
        down_module_name: ModuleName,
        down_layer: LayerIndex,
        up_module_name: ModuleName,
        up_active_layers,
        up_active_feature_ids,
        edge_type=None
    ):
        # If there are no important nodes at this down_layer, do nothing
        if len(imp_down_pos) == 0:
            return
        # Convert lists to PyTorch tensors
        imp_down_feature_ids = torch.tensor(
            imp_down_feature_ids, dtype=torch.long, device="cuda"
        )
        imp_down_pos = torch.tensor(imp_down_pos, dtype=torch.long, device="cuda")

        # Create a mask where attribs are greater than the threshold TODO add option for chained
        if self.cfg.chained_attribs:
            mask = edge_metric_attribs > self.cfg.threshold
        else:
            mask = node_node_attribs > self.cfg.threshold
        # Use the mask to find the relevant indices
        if down_module_name in ["mlp", "metric"]:
            assert (
                len(node_node_attribs.size()) == 2
            ), "attribs must be 2D for mlp and metric"
            imp_ids, up_active_ids = torch.where(mask)
            node_node_grads_values = node_node_grads[imp_ids, up_active_ids].flatten()
            node_node_attribs_values = node_node_attribs[
                imp_ids, up_active_ids
            ].flatten()
            edge_metric_grads_values = edge_metric_grads[
                imp_ids, up_active_ids
            ].flatten()
            edge_metric_attribs_values = edge_metric_attribs[
                imp_ids, up_active_ids
            ].flatten()

        elif down_module_name == "attn":
            assert (
                len(node_node_attribs.size()) == 3
            ), "attribs must be 2D for mlp and metric"
            up_seqs, imp_ids, up_active_ids = torch.where(mask)
            node_node_grads_values = node_node_grads[
                up_seqs, imp_ids, up_active_ids
            ].flatten()
            node_node_attribs_values = node_node_attribs[
                up_seqs, imp_ids, up_active_ids
            ].flatten()
            edge_metric_grads_values = edge_metric_grads[
                up_seqs, imp_ids, up_active_ids
            ].flatten()
            edge_metric_attribs_values = edge_metric_attribs[
                up_seqs, imp_ids, up_active_ids
            ].flatten()
        else:
            raise ValueError(
                "down_module_name must be one of ['mlp', 'attn', 'metric']"
            )

        # Get corresponding down_feature_ids and seqs using tensor indexing
        down_feature_ids = imp_down_feature_ids[imp_ids]
        down_seqs = imp_down_pos[imp_ids]

        # metric and mlp don't mix positions
        if down_module_name in ["mlp", "metric"]:
            up_seqs = down_seqs

        up_feature_ids = up_active_feature_ids[up_active_ids]
        up_layer_ids = up_active_layers[up_active_ids]

        # Construct edges based on the indices and mask
        edges = [
            (
                f"{down_module_name}.{down_layer}.{down_seqs[i]}.{down_feature_ids[i]}",
                f"{up_module_name}.{up_layer_ids[i]}.{up_seqs[i]}.{up_feature_ids[i]}",
            )
            for i in range(node_node_grads_values.size(0))
        ]

        # Append to the graph
        for edge, nn_grad, nn_attrib, em_grad, em_attrib in zip(
            edges,
            node_node_grads_values,
            node_node_attribs_values,
            edge_metric_grads_values,
            edge_metric_attribs_values,
        ):
            # don't bother adding nodes at pos=0, since this is BOS token
            if not edge[1].split(".")[2] == "0":
                self.graph.append(
                    (
                        edge,
                        (
                            nn_grad.item(),
                            nn_attrib.item(),
                            em_grad.item(),
                            em_attrib.item(),
                        ),
                        edge_type
                    )
                )  # type: ignore

        # Add errors to separate graph
        # TODO do we care about chained attribs for these too?
        if self.cfg.store_error_attribs:
            if down_module_name in ["mlp", "metric"]:
                # error_attribs : [imp_id, layer]
                for imp_id in range(error_attribs.size(0)):
                    for up_layer in range(error_attribs.size(1)):
                        edge = (
                            f"{down_module_name}.{down_layer}.{imp_down_pos[imp_id]}.{imp_down_feature_ids[imp_id]}",
                            f"{up_module_name}_error.{up_layer}.{imp_down_pos[imp_id]}.{0}",
                        )
                        attrib = error_attribs[imp_id, up_layer]
                        self.error_graph.append((edge, attrib.item()))

            if down_module_name in ["attn"]:
                # error_attribs : [seq, imp_id, layer]
                for imp_id in range(error_attribs.size(1)):
                    for up_layer in range(error_attribs.size(2)):
                        for up_seq in range(error_attribs.size(0)):
                            edge = (
                                f"{down_module_name}.{down_layer}.{imp_down_pos[imp_id]}.{imp_down_feature_ids[imp_id]}",
                                f"{up_module_name}_error.{up_layer}.{up_seq}.{0}",
                            )
                            attrib = error_attribs[up_seq, imp_id, up_layer]
                            self.error_graph.append((edge, attrib.item()))

    def run(self):
        self.metric_step()
        for layer in reversed(range(1, self.n_layers)):
            self.mlp_step(layer)
            self.ov_step(layer)
            if self.cfg.qk_enabled:
                self.q_step(layer)
                self.k_step(layer)


# # %%
# #Imports and downloads
# import sys

# sys.path.append("/root/circuit-finder")
# print(sys.path)

# import transformer_lens as tl
# from torch import Tensor
# from jaxtyping import Int
# from typing import Callable
# from circuit_finder.patching.eap_graph import EAPGraph
# from circuit_finder.plotting import show_attrib_graph
# import torch
# import gc
# from tqdm import tqdm

# from circuit_finder.pretrained import (
#     load_attn_saes,
#     load_mlp_transcoders,
# )

# # Load models
# model = tl.HookedTransformer.from_pretrained(
#     "gpt2",
#     device="cuda",
#     fold_ln=True,
#     center_writing_weights=True,
#     center_unembed=True,
# )

# attn_saes = load_attn_saes()
# attn_saes = preprocess_attn_saes(attn_saes, model)  # type: ignore
# transcoders = load_mlp_transcoders()

# # # Define dataset
# tokens = model.to_tokens(
#     [    
#         "When John and Mary were at the store, John gave a bottle to Mary",
#         "When Linda and Tom were in the park, Linda threw a ball to Tom",
#         "While Sarah and Jamie are on the run, Jamie gives a gun to Sarah",
#         "When Hugh and Susan came to the party, Susan passed a drink to Hugh",
#         "Since Tom and Jim are best of friends, Tom gives a hug to Jim",
#     ]
# )

# tokens = model.to_tokens(
#     [    "in the centre of the road was a car, with black rubber tyres" ])

# # corrupt_tokens = model.to_tokens(
# #     [
# #         "When Alice and Bob were at the store, Charlie gave a bottle to Dan",
# #         "When Alice and Bob were in the park, Charlie threw a ball to Dan",
# #         "When Alice and Bob are on the run, Charlie gives a gun to Dan",
# #         "When Alice and Bob came to the party, Charlie passed a drink to Dan",
# #         "Since Alice and Bob are best of friends, Charlie gives a hug to Dan",
# #     ]
# # )

# def last_token_logit(metric, tokens):
#     logits = model(tokens)[:,-2]
#     correct_logit = eindex(logits, tokens[:,-1], "batch [batch]")
#     wrong_logit = eindex(logits, tokens[:,-6], "batch [batch]")
#     return correct_logit.mean() - wrong_logit.mean()



#%%
# model.reset_hooks()
# cfg = LEAPConfig(threshold=0.1, contrast_pairs=False, chained_attribs=True)
# leap = IndirectLEAP(
#     cfg, tokens[:1], model, attn_saes, transcoders, last_token_logit, corrupt_tokens=None
# )
# leap.run()
# print(len(leap.graph))

# # %%
# from circuit_finder.plotting import make_html_graph
# from circuit_finder.patching.eap_graph import EAPGraph
# graph = EAPGraph(leap.graph)
# make_html_graph(graph, attrib_type="em")
# # %%
# types = graph.get_edge_types()
# print("num ov edges: ", len([t for t in types if t=="ov"]))
# print("num q edges: ", len([t for t in types if t=="q"]))
# print("num k edges: ", len([t for t in types if t=="k"]))
# print("num mlp edges: ", len([t for t in types if t==None]))




