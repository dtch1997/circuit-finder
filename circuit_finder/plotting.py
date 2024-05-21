# type: ignore
"""Generically useful functions for making plots with tensors"""

import torch
import transformer_lens.utils as utils

import plotly.express as px
import plotly.graph_objects as go

from typing import List

import networkx as nx
import matplotlib.pyplot as plt
from circuit_finder.core.types import parse_node_name

update_layout_set = {
    "xaxis_range",
    "yaxis_range",
    "hovermode",
    "xaxis_title",
    "yaxis_title",
    "colorbar",
    "colorscale",
    "coloraxis",
    "title_x",
    "bargap",
    "bargroupgap",
    "xaxis_tickformat",
    "yaxis_tickformat",
    "title_y",
    "legend_title_text",
    "xaxis_showgrid",
    "xaxis_gridwidth",
    "xaxis_gridcolor",
    "yaxis_showgrid",
    "yaxis_gridwidth",
}


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        labels={"x": xaxis, "y": yaxis},
        **kwargs_pre,
    ).update_layout(**kwargs_post)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = label

    fig.show(renderer)


def scatter(
    x, y, xaxis="", yaxis="", caxis="", renderer=None, return_fig=False, **kwargs
):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    )
    if return_fig:
        return fig
    fig.show(renderer)


def show_avg_logit_diffs(x_axis: List[str], per_prompt_logit_diffs: List[torch.tensor]):
    y_data = [
        per_prompt_logit_diff.mean().item()
        for per_prompt_logit_diff in per_prompt_logit_diffs
    ]
    error_y_data = [
        per_prompt_logit_diff.std().item()
        for per_prompt_logit_diff in per_prompt_logit_diffs
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_axis,
                y=y_data,
                error_y=dict(
                    type="data",  # specifies that the actual values are given
                    array=error_y_data,  # the magnitudes of the errors
                    visible=True,  # make error bars visible
                ),
            )
        ]
    )

    # Customize layout
    fig.update_layout(
        title_text="Logit Diff after Interventions",
        xaxis_title_text="Intervention",
        yaxis_title_text="Logit diff",
        plot_bgcolor="white",
    )

    # Show the figure
    fig.show()


def show_attrib_graph(graph):
    G = nx.DiGraph()
    for dest, src in graph.get_edges():
        if dest != "null":
            G.add_edge(src, dest)

    def get_node_position(
        node_name: str,
    ) -> tuple[float, float]:
        module, layer, token, feature = parse_node_name(node_name)
        x, y = token, layer
        if module == "mlp":
            x += 0.5
            y += 0.5
        return (x, y)

    def get_node_color(
        node_name: str,
    ) -> str:
        module, layer, token, feature = parse_node_name(node_name)
        if module == "mlp":
            return "red"
        elif module == "attn":
            return "blue"
        else:
            return "black"

    pos = {node: get_node_position(node) for node in G.nodes}
    color = [get_node_color(node) for node in G.nodes]

    plt.figure(3, figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color=color)
    plt.show()
