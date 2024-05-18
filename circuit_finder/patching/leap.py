# type: ignore
"""Linear Edge Attribution Patching.

Similar to Edge Attribution Patching, calculates the effect of edges on some downstream metric.
However, takes advantage of linearity afforded by transcoders and MLPs to parallelize
"""

# %%
import gc
from msilib import Feature
import torch
import transformer_lens as tl

from transcoders_slim.transcoder import Transcoder
from torch import Tensor
from jaxtyping import Int, Float
from dataclasses import dataclass
from einops import rearrange, einsum
from typing import Literal

from circuit_finder.core.types import LayerIndex, MetricFn, HookNameFilterFn
from circuit_finder.constants import device


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def last_token_logit(model, tokens):
    """just a simple metric for testing"""
    logits = model(tokens, return_type="logits")[:, -2, :]
    correct_logits = logits[torch.arange(logits.size(0)), tokens[:, -1]]
    return correct_logits.mean()


FeatureIndex = int
TokenIndex = int
Node = str
Edge = tuple[Node, Node]  # (downstream, upstream)
Attrib = float
ModuleName = Literal["mlp", "attn", "metric"]


def parse_node_name(
    node: Node,
) -> tuple[ModuleName, LayerIndex, TokenIndex, FeatureIndex]:
    """Parse a node name into its components."""
    module, layer, pos, feature_id = node.split(".")
    return module, int(layer), int(pos), int(feature_id)


def get_node_name(
    module: ModuleName, layer: LayerIndex, pos: TokenIndex, feature_id: FeatureIndex
) -> Node:
    """Get a node name from its components."""
    return f"{module}.{layer}.{pos}.{feature_id}"


def preprocess_attn_saes(
    attn_saes_in: dict[LayerIndex, tl.HookedSAE],
    model: tl.HookedTransformer,
) -> dict[LayerIndex, tl.HookedSAE]:
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


# %%
@dataclass
class LEAPConfig:
    threshold: float = 0.01
    contrast_pairs: bool = False


class LEAP:
    """
    Linear Edge Attribution Patching

    Similar to edge attribution patching, but fully linearises the network to compute attributions exactly
    """

    # Models
    model: tl.HookedTransformer
    attn_saes: dict[LayerIndex, tl.HookedSAE]
    transcoders: dict[LayerIndex, Transcoder]

    # Data
    tokens: Int[Tensor, "batch seq"]
    corrupt_tokens: Int[Tensor, "batch seq"]

    # Intermediate computations
    mlp_feature_acts: Float[Tensor, "seq layer d_trans"]
    mlp_is_active: Float[Tensor, "seq layer d_trans"]
    attn_feature_acts: Float[Tensor, "seq layer d_sae"]
    attn_is_active: Float[Tensor, "seq layer d_sae"]
    mlp_errors: Float[Tensor, "seq layer d_model"]
    attn_errors: Float[Tensor, "seq layer d_model"]

    # Graph
    graph: list[tuple[Edge, Attrib]]

    def __init__(
        self,
        cfg: LEAPConfig,
        tokens: Int[torch.Tensor, "batch seq"],
        model: tl.HookedTransformer,
        attn_saes: dict[LayerIndex, tl.HookedSAE],  # layer index: attn-out SAE
        transcoders: dict[LayerIndex, Transcoder],  # layer index: mlp transcoder
        metric: MetricFn = last_token_logit,
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

        # NOTE: We'll store (edge, attrib) pairs here.
        # Initialise by making the metric an important node, by hand.
        # edge = (downstream_node, upstream_node)
        # node = "{module_name}.{layer}.{pos}.{feature_id}"
        self.graph: list[tuple[Edge, Attrib]] = [
            (("null", f"metric.{self.n_layers}.{self.n_seq-2}.0"), 0)
        ]

    def get_important_edges(self) -> list[Edge]:
        return [edge for edge, _ in self.graph]

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
        self.pattern: Float[Tensor, "layer head q_pos k_pos"] = torch.stack(
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
                "seq n_heads d_heads, n_heads d_head d_model -> seq d_model",
            )
            self.attn_errors[:, layer, :] = resid_error

        # Get ids of active features
        self.mlp_active_feature_ids = torch.where(self.mlp_is_active.sum(0) > 0)
        self.attn_active_feature_ids = torch.where(self.attn_is_active.sum(0) > 0)

    def metric_step(self):
        """Step 0 of circuit discovery: get attributions from each node to the metric.

        TODO currently, if metric depends on multiple token positions, this will
        sum the gradient over those positions. Do we want to be more general, i.e.
        store separate gradients at each position? Or maybe we don't care..."""
        imp_down_feature_ids, imp_down_pos = self.get_imp_feature_ids_and_pos(
            "metric", self.n_layers
        )

        self.model.blocks[self.model.cfg.n_layers - 1].mlp.b_out.grad = None
        m = self.metric(self.model, self.tokens)
        m.backward()  # TODO don't actually need backward through whole model. If m is linear, we can disable autograd!

        # Sneaky way to get d(metric)/d(resid_post_final)
        grad = self.model.blocks[self.model.cfg.n_layers - 1].mlp.b_out.grad.unsqueeze(
            0
        )
        self.compute_and_save_attribs(
            grad, "metric", self.n_layers, imp_down_feature_ids, imp_down_pos
        )

    def mlp_step(self, down_layer):
        """For each imp node at this MLP, compute attrib wrt all previous nodes"""

        # Get the imp features coming out of the MLP, and the positions at which they're imp
        imp_down_feature_ids, imp_down_pos = self.get_imp_feature_ids_and_pos(
            "mlp", down_layer
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
            grad, "mlp", down_layer, imp_down_feature_ids, imp_down_pos
        )

    def ov_step(self, down_layer):
        """For each imp node at this attention layer, compute attrib wrt all previous nodes *via the OV circuit*"""

        # Get the imp features coming out of the attention layer, and the positions at which they're imp
        imp_down_feature_ids, imp_down_pos = self.get_imp_feature_ids_and_pos(
            "attn", down_layer
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
        imp_patterns = self.pattern[
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
            grad, "attn", down_layer, imp_down_feature_ids, imp_down_pos
        )

    """ Helper functions """

    def get_imp_feature_ids_and_pos(
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
        for edge, _ in self.get_important_edges():
            _, upstream = edge
            up_nodes_set.add(upstream)
        up_nodes_deduped: list[Node] = list(up_nodes_set)

        # up_nodes_deduped: list[Node] = list(set([edge[1] for (edge, _) in self.graph]))
        for up_node in up_nodes_deduped:
            down_module_, down_layer_, pos, feature_id = parse_node_name(up_node)
            # Filter by module and layer
            # TODO: It seems like we could do this previously but ig it doesn't matter.
            if down_module_ == down_module and down_layer_ == str(down_layer):
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

    def compute_and_save_attribs(
        self, grad, down_module, down_layer, imp_down_feature_ids, imp_down_pos
    ):
        """grad : [... imp_id, d_model]"""

        # ----- UPSTREAM MLP ATTRIBS ------
        # Get upstream feature acts, since we'll need this for attrib computation
        mlp_active_W_dec, mlp_up_active_layers, mlp_up_active_feature_ids = (
            self.get_active_mlp_W_dec(down_layer)
        )
        imp_mlp_feature_acts = self.mlp_feature_acts[
            :, mlp_up_active_layers, mlp_up_active_feature_ids
        ]  # [pos, up_active_id]

        if down_module in ["mlp", "metric"]:  # down mlp doesn't mix positions
            imp_mlp_feature_acts = imp_mlp_feature_acts[
                imp_down_pos
            ]  # [imp_id, up_active_id]

            # Compute attribs of imp nodes wrt all upstream MLP nodes
            mlp_attribs = einsum(
                imp_mlp_feature_acts,  # TODO RENAME TO ACTIVE_MLP_FEATURE_ACTS
                mlp_active_W_dec,
                grad,
                "imp_id up_active_id, up_active_id d_model, imp_id d_model -> imp_id up_active_id",
            )
        elif down_module in ["attn"]:
            # Compute attribs of imp nodes wrt all upstream MLP nodes
            mlp_attribs = einsum(
                imp_mlp_feature_acts,  # TODO RENAME TO ACTIVE_MLP_FEATURE_ACTS
                mlp_active_W_dec,
                grad,
                "seq up_active_id, up_active_id d_model, seq imp_id d_model -> seq imp_id up_active_id",
            )
        else:
            print(down_module)
            raise ValueError("down_module must be one of ['mlp', 'attn', 'metric']")

        # attrib can be at most the value of the original downstream feature act
        # mlp_attribs = torch.min(mlp_attribs, imp_mlp_feature_acts)
        self.add_to_graph(
            mlp_attribs,
            imp_down_feature_ids,
            imp_down_pos,
            down_module_name=down_module,
            down_layer=down_layer,
            up_module_name="mlp",
            up_active_layers=mlp_up_active_layers,
            up_active_feature_ids=mlp_up_active_feature_ids,
        )

        del mlp_attribs
        clear_mem()

        # ----- UPSTREAM ATTENTION ATTRIBS (do exactly the same for attention) -----
        attn_active_W_dec, attn_up_active_layers, attn_up_active_feature_ids = (
            self.get_active_attn_W_dec(down_layer, down_module)
        )
        imp_attn_feature_acts = self.attn_feature_acts[
            :, attn_up_active_layers, attn_up_active_feature_ids
        ]  # [imp_id, up_active_id]

        if down_module in ["mlp", "metric"]:  # down mlp doesn't mix positions
            imp_attn_feature_acts = imp_attn_feature_acts[
                imp_down_pos
            ]  # [imp_id, up_active_id]
            attn_attribs = einsum(
                imp_attn_feature_acts,
                attn_active_W_dec,
                grad,
                "imp_id up_active_id, up_active_id d_model, imp_id d_model -> imp_id up_active_id",
            )
        elif down_module in ["attn"]:
            attn_attribs = einsum(
                imp_attn_feature_acts,
                attn_active_W_dec,
                grad,
                "seq up_active_id, up_active_id d_model, seq imp_id d_model -> seq imp_id up_active_id",
            )
        else:
            raise ValueError("down_module must be one of ['mlp', 'attn', 'metric']")

        self.add_to_graph(
            attn_attribs,
            imp_down_feature_ids,
            imp_down_pos,
            down_module_name=down_module,
            down_layer=down_layer,
            up_module_name="attn",
            up_active_layers=attn_up_active_layers,
            up_active_feature_ids=attn_up_active_feature_ids,
        )

        del attn_attribs
        clear_mem()

    def add_to_graph(
        self,
        attribs,
        imp_down_feature_ids,
        imp_down_pos,
        down_module_name: ModuleName,
        down_layer: LayerIndex,
        up_module_name: ModuleName,
        up_active_layers,
        up_active_feature_ids,
    ):
        # Convert lists to PyTorch tensors
        imp_down_feature_ids = torch.tensor(
            imp_down_feature_ids, dtype=torch.long, device=attribs.device
        )
        imp_down_pos = torch.tensor(
            imp_down_pos, dtype=torch.long, device=attribs.device
        )

        # Create a mask where attribs are greater than the threshold
        mask = attribs > self.cfg.threshold
        # Use the mask to find the relevant indices
        if down_module_name in ["mlp", "metric"]:
            imp_ids, up_active_ids = torch.where(mask)
            attrib_values = attribs[imp_ids, up_active_ids].flatten()

        elif down_module_name == "attn":
            up_seqs, imp_ids, up_active_ids = torch.where(mask)
            attrib_values = attribs[up_seqs, imp_ids, up_active_ids].flatten()
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
            for i in range(attrib_values.size(0))
        ]

        # Append to the graph
        for edge, value in zip(edges, attrib_values):
            # don't bother adding nodes at pos=0, since this is BOS token
            if not edge[1].split(".")[2] == "0":
                self.graph.append((edge, value.item()))  # type: ignore
