"""Script to run an experiment with LEAP

Usage:
pdm run python -m circuit_finder.experiments.run_leap_experiment [ARGS]

Run with flag '-h', '--help' to see the arguments.
"""
#%% Imports and Downloads
import sys
sys.path.append("/root/circuit-finder")
import torch
import transformer_lens as tl
import pandas as pd
import json

from jaxtyping import Int, Float
from torch import Tensor
from eindex import eindex
from simple_parsing import ArgumentParser
from dataclasses import dataclass
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.utils import clear_memory
from circuit_finder.patching.ablate import get_metric_with_ablation
from circuit_finder.data_loader import load_datasets_from_json
from circuit_finder.constants import device
from tqdm import tqdm

from pathlib import Path
from circuit_finder.pretrained import (
    load_attn_saes,
    load_mlp_transcoders,
)
from circuit_finder.patching.indirect_leap import (
    preprocess_attn_saes,
    IndirectLEAP,
    LEAPConfig,
)

from circuit_finder.constants import ProjectDir
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.constants import ProjectDir
from circuit_finder.plotting import show_attrib_graph

model = tl.HookedTransformer.from_pretrained(
    "gpt2",
    device="cuda",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)

attn_saes = load_attn_saes()
attn_saes = preprocess_attn_saes(attn_saes, model)  # type: ignore
transcoders = load_mlp_transcoders()

#%%
@dataclass
class LeapExperimentConfig:
    dataset_path: str = "datasets/greaterthan_gpt2-small_prompts.json"
    save_dir: str = "results/leap_experiment"
    seed: int = 1
    batch_size: int = 4
    total_dataset_size: int = 1024
    ablate_errors: bool | str = False
    # NOTE: This specifies what to do with error nodes when calculating faithfulness curves.
    # Options are:
    # - ablate_errors = False  ->  we don't ablate error nodes
    # - ablate_errors = "bm"   ->  we batchmean-ablate error nodes
    # - ablate_errors = "zero" ->  we zero-ablate error nodes (warning: this gives v bad performance)

    first_ablate_layer: int = 2
    # NOTE:This specifies which layer to start ablating at.
    # Marks et al don't ablate the first 2 layers
    # TODO: Find reference for the above

    verbose: bool = False


def logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer: Int[Tensor, " batch"],
    wrong_answer: Int[Tensor, " batch"],
):
    # Get last-token logits
    logits = logits[:, -1, :]
    correct_logits = eindex(logits, correct_answer, "batch [batch]")
    wrong_logits = eindex(logits, wrong_answer, "batch [batch]")
    return (correct_logits - wrong_logits).mean()

def list_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer, # list of n_batch different Int[Tensor, " batch"]
    wrong_answer: Int[Tensor, " batch"] # list of n_batch different Int[Tensor, " batch"]
):
    # Get last-token logits
    logits = logits[:, -1, :]

    diff = 0
    for b in range(logits.size(0)):
        correct_logits = logits[b][correct_answer[b]]
        wrong_logits = logits[b][wrong_answer[b]]
        #TODO: do we want mean or sum here?
        diff += correct_logits.mean() - wrong_logits.mean()
    return diff
#%%

def run_leap_experiment(config: LeapExperimentConfig):
    # Define save dir
    save_dir = ProjectDir / config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save the config
    with open(save_dir / "config.json", "w") as jsonfile:
        json.dump(config.__dict__, jsonfile)

    # Define dataset
    dataset_path = config.dataset_path
    if not dataset_path.startswith("/"):
        # Assume relative path
        dataset_path = ProjectDir / dataset_path
    else:
        dataset_path = Path(dataset_path)
    train_loader, _ = load_datasets_from_json(
        model=model,
        path=dataset_path,
        device=device,  # type: ignore
        batch_size=config.batch_size,
        train_test_size=(config.total_dataset_size, config.total_dataset_size),
        random_seed=config.seed,
    )

    for idx, batch in tqdm(enumerate(train_loader)):
        # Set up batch dir
        batch_dir = save_dir / f"batch_{idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save the config
        with open(batch_dir / "config.json", "w") as jsonfile:
            json.dump(config.__dict__, jsonfile)

        # Parse the batch
        key = batch.key
        clean_tokens = batch.clean
        answer_tokens = batch.answers
        wrong_answer_tokens = batch.wrong_answers
        corrupt_tokens = batch.corrupt
        # TODO: handle case where answer_token is list
        # This is applicable e.g. for 'GreaterThan' which has many correct and wrong answers
        # and we are expected to sum over them.
        # NOTE: for now, we only support single answer and wrong answer

        # clean_tokens = model.to_tokens(
        #     ["When John and Mary were at the store, John gave a bottle to"]
        # )
        # answer_tokens = model.to_tokens(["Mary"], prepend_bos=False)
        # wrong_answer_tokens = model.to_tokens(["John"], prepend_bos=False)
        # corrupt_tokens = model.to_tokens(
        #     ["When Alice and Bob were at the store, Charlie gave a bottle to"]
        # )

        # Define metric
        def metric_fn(model, tokens):
            logits = model(tokens, return_type="logits")
            return list_logit_diff(
                logits,
                answer_tokens, #.squeeze(dim=-1),
                wrong_answer_tokens #.squeeze(dim=-1),
            )

        # Save the dataset.
        with open(batch_dir / "dataset.json", "w") as jsonfile:
            json.dump(
                {
                    "clean": model.tokenizer.batch_decode(clean_tokens),
                    "answer": model.tokenizer.batch_decode(answer_tokens),
                    "wrong_answer": model.tokenizer.batch_decode(wrong_answer_tokens),
                    "corrupt": model.tokenizer.batch_decode(corrupt_tokens),
                },
                jsonfile,
            )

        # NOTE: First, get the ceiling of the patching metric.
        # TODO: Replace 'last_token_logit' with logit difference
        model.reset_hooks()
        ceiling = metric_fn(model, clean_tokens).item()

        # NOTE: Second, get floor of patching metric using empty graph, i.e. ablate everything
        empty_graph = EAPGraph([])
        floor = get_metric_with_ablation(
            model,
            empty_graph,
            clean_tokens,
            metric_fn,
            transcoders,
            attn_saes,
            ablate_errors=False,  # Do not ablate errors when running forward pass
            first_ablated_layer=config.first_ablate_layer,
        ).item()
        clear_memory()

        # now sweep over thresholds to get graphs with variety of numbers of nodes
        # for each graph we calculate faithfulness
        num_nodes_list = []
        metrics_list = []

        # Sweep over thresholds
        # TODO: make configurable
        thresholds = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17]
        for threshold in thresholds:
            # Setup LEAP algorithm
            model.reset_hooks()
            cfg = LEAPConfig(
                threshold=threshold, contrast_pairs=False, chained_attribs=True
            )
            leap = IndirectLEAP(
                cfg=cfg,
                tokens=clean_tokens,
                model=model,
                metric=metric_fn,
                attn_saes=attn_saes,  # type: ignore
                transcoders=transcoders,
                corrupt_tokens=corrupt_tokens,
            )

            # Populate the graph
            leap.metric_step()
            for layer in reversed(range(1, leap.n_layers)):
                leap.mlp_step(layer)
                leap.ov_step(layer)

            # Save the graph
            graph = EAPGraph(leap.graph)
            num_nodes = len(graph.get_src_nodes())
            with open(
                batch_dir / f"leap-graph_threshold={threshold}.json", "w"
            ) as jsonfile:
                json.dump(graph.to_json(), jsonfile)

            # Delete tensors to save memory
            del leap
            clear_memory()

            # Calculate the metric under ablation
            metric = get_metric_with_ablation(
                model,
                graph,
                clean_tokens,
                metric_fn,
                transcoders,
                attn_saes,
                ablate_errors=config.ablate_errors,  # type: ignore
                first_ablated_layer=config.first_ablate_layer,
            ).item()

            # Log the data
            num_nodes_list.append(num_nodes)
            metrics_list.append(metric)

        faith = [(metric - floor) / (ceiling - floor) for metric in metrics_list]

        # Save the result as a dataframe
        data = pd.DataFrame({"num_nodes": num_nodes_list, "faithfulness": faith})
        data.to_csv(batch_dir / "leap_experiment_results.csv", index=False)












#%% RUN EXPERIMENT

cfg = LeapExperimentConfig(
    dataset_path = "datasets/greaterthan_gpt2-small_prompts.json",
    save_dir = "results/leap_experiment/jacob_05_25",
    seed = 1,
    batch_size = 10,
    total_dataset_size = 10,
    ablate_errors = False,
    first_ablate_layer = 2,
    verbose = False,
)

run_leap_experiment(cfg)



#%%
import json
import sys
sys.path.append("/root/circuit-finder")
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.constants import ProjectDir
from circuit_finder.plotting import show_attrib_graph

results_dir = ProjectDir / "results" / "leap_experiment" / "jacob_05_25" / "batch_0"
assert results_dir.exists()

config = json.load(open(results_dir / "config.json"))
for key, value in config.items():
    print(f"{key}: {value}")

#%%
import pandas as pd
pd.set_option('display.max_colwidth', None)

dataset = results_dir / "dataset.json"
with open(dataset, 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)
df.head()

# %%
threshold = 0.17
with open(results_dir / f"leap-graph_threshold={threshold}.json") as f:
    graph = EAPGraph.from_json(json.load(f))

from circuit_finder.plotting import make_html_graph
make_html_graph(graph,  attrib_type="em")
len(graph.get_edges())
# %%
## Print the distribution of nodes
import pandas as pd
from circuit_finder.core.types import parse_node_name

rows = []
for edge, edge_info in graph.graph:
    dest, src = edge
    if 'metric' in src: continue
    nn_grad, nn_attrib, em_grad, em_attrib = edge_info 
    src_module_type, src_layer, src_token, src_feature = parse_node_name(src)
    dest_module_type, dest_layer, dest_token, dest_feature = parse_node_name(dest)
    rows.append({
        "src_module_type": src_module_type,
        "dest_module_type": dest_module_type,
        "src_layer": src_layer,
        "dest_layer": dest_layer,
        "src_token": src_token,
        "dest_token": dest_token,
        "src_feature": src_feature,
        "dest_feature": dest_feature,
        "nn_grad": nn_grad,
        "nn_attrib": nn_attrib,
        "em_grad": em_grad,
        "em_attrib": em_attrib,
    })
df = pd.DataFrame(rows)
print(len(df))
df.head()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

total_attrib_df = df.groupby(["src_layer", "src_module_type"]).sum().reset_index()
print(total_attrib_df.columns)
# Plot the total edge attribution by src layer
sns.barplot(x="src_layer", y="nn_attrib", data=total_attrib_df, hue="src_module_type")
plt.title("Total edge attribution by src layer")
# %%
# Plot the number of nodes in each layer
sns.countplot(x="src_layer", data=df, hue="src_module_type")
plt.title("Number of nodes in each layer")

#%%
## Highest nodes by NN attrib

sorted_df = df.sort_values(by='em_attrib', ascending=False).head(30)

#%%
from circuit_finder.neuronpedia import get_neuronpedia_url_for_quick_list

layer, module = 8, "attn"
df_layer = sorted_df[(df.src_layer == layer) & (df.src_module_type == module)]
df_layer
#%%

features = sorted_df[
    (df.src_layer == layer) & 
    (df.src_module_type == module)
]['src_feature'].unique()

dash_type = "att-kk" if module=="attn" else "tres-dc"
print(get_neuronpedia_url_for_quick_list(layer, features, dash_type))



#%%
import sys
import json
sys.path.append("/root/circuit-finder")
from circuit_finder.patching.eap_graph import EAPGraph
with open("/root/circuit-finder/results/leap_experiment/batch_1/leap-graph_threshold=0.03.json") as f:
    graph = EAPGraph.from_json(json.load(f))


#%%
