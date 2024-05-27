#%%
import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph

from circuit_finder.constants import ProjectDir
from circuit_finder.pretrained import load_model, load_hooked_mlp_transcoders, load_attn_saes
from circuit_finder.auto_circuit.utils.graph_utils import patchable_model
from circuit_finder.patching.ablate import splice_model_with_saes_and_transcoders

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_model("gpt2")
saes = load_attn_saes()
transcoders = load_hooked_mlp_transcoders()
saes = list(saes.values())
transcoders = list(transcoders.values())

path = (ProjectDir / "datasets/ioi/ioi_vanilla_template_prompts.json").absolute()
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=16,
    train_test_size=(128, 128),
)

with splice_model_with_saes_and_transcoders(model, transcoders, saes) as spliced_model:
    model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=False,
        device=device,
    )

    attribution_scores: PruneScores = mask_gradient_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

fig = draw_seq_graph(
    model, attribution_scores, 3.5, layer_spacing=True, orientation="v"
)
fig.write_image(repo_path_to_abs_path("docs/assets/IOI_Attributions_Viz.png"), scale=4)