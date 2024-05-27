# type: ignore
"""Generically useful functions for making plots with tensors"""

import torch
import transformer_lens.utils as utils

import plotly.express as px
import plotly.graph_objects as go

from typing import List
from pyvis.network import Network
import numpy as np

import networkx as nx
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


def show_attrib_graph(graph, **kwargs):
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

    nx.draw(G, pos, with_labels=True, node_color=color, **kwargs)


def make_html_graph(
    graph, attrib_type="em", compact_nodes=False, node_offset=10, tokens=None
):
    edges = graph.get_edges()
    edge_types = graph.get_edge_types()  # Get edge types

    if attrib_type == "nn":
        attribs = graph.get_nn_attribs()
    elif attrib_type == "em":
        attribs = graph.get_em_attribs()
    else:
        print("Invalid attrib type chosen. Options are em and nn")
        return

    # Ensure the number of edges matches the number of edge types and attributes
    if len(edges) != len(edge_types) or len(edges) != len(attribs):
        print("Number of edges does not match number of edge types or attributes")
        return

    # Create a directed graph
    G = nx.DiGraph()

    # Color map for different modules
    color_map = {"metric": "green", "mlp": "blue", "attn": "orange"}

    # Track position offsets for nodes with the same (position, layer)
    offset_tracker = {"attn": {}, "mlp": {}, "metric": {}}

    positions = set()  # To track unique position values

    for downstream, upstream in edges:
        if downstream == "null" or upstream == "null":
            continue
        for node in [upstream, downstream]:
            if not G.has_node(node):
                module, layer, position, _id = node.split(".")
                layer = int(layer)
                position = int(position)
                positions.add(position)
                color = color_map[module]
                title = f"layer {layer} pos {position} id {_id}"

                # Determine the base position
                base_x = position * 150  # Increase x-axis spacing
                base_y = layer * 80  # Decrease y-axis spacing

                # Offset nodes based on module type
                if (position, layer) not in offset_tracker[module]:
                    offset_tracker[module][(position, layer)] = 0

                if module == "attn":
                    pos_x = (
                        base_x
                        - 10
                        - offset_tracker[module][(position, layer)] * node_offset
                    )  # Shift each "attn" node further left
                    pos_y = base_y - 15  # Shift each "attn" node further downward
                    url = f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{_id}"
                elif module == "mlp":
                    pos_x = (
                        base_x
                        + 15
                        + offset_tracker[module][(position, layer)] * node_offset
                    )  # Shift each "mlp" node further right
                    pos_y = base_y
                    url = (
                        f"https://www.neuronpedia.org/gpt2-small/{layer}-tres-dc/{_id}"
                    )
                else:
                    pos_x = (
                        base_x + offset_tracker[module][(position, layer)] * node_offset
                    )
                    pos_y = base_y
                    url = ""

                offset_tracker[module][(position, layer)] += 1

                # Add node with attributes
                G.add_node(
                    node, title=title, color=color, id=_id, x=pos_x, y=pos_y, url=url
                )
        G.add_edge(upstream, downstream)

    # Combine edges with their attributes and types
    edge_data = list(zip(edges, attribs, edge_types))

    # Clip the attribs using the 95th percentile
    p95_attrib = np.percentile(attribs, 95)
    clipped_attribs = np.clip(attribs, None, p95_attrib)

    # Normalize the clipped attribs to range from 0.2 to 1
    min_attrib = min(clipped_attribs)
    max_attrib = max(clipped_attribs)
    range_attrib = max_attrib - min_attrib

    normalized_attribs = [
        0.2 + 0.8 * (a - min_attrib) / range_attrib for a in clipped_attribs
    ]

    # Determine the threshold for the top 5% of edges by attribute value
    threshold = np.percentile(attribs, 95)

    # Map normalized attributes back to edges
    edge_data_normalized = [
        (e[0], e[1], norm_attrib, e[2], original_attrib)
        for e, norm_attrib, original_attrib in zip(
            edge_data, normalized_attribs, attribs
        )
    ]

    # Create pyvis network
    net = Network(notebook=True, directed=True, cdn_resources="remote")
    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 5,
        "font": {
          "size": 0
        },
        "fixed": {
          "x": true,
          "y": true
        }
      },
      "interaction": {
        "hover": true
      },
      "physics": {
        "enabled": false
      }
    }
    """)

    # Add nodes to the pyvis network
    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=data["id"],
            title=data["title"],
            color=data["color"],
            x=data["x"],
            y=-data["y"],
            url=data["url"],
        )

    # Add edges with normalized opacity, hover title, color based on edge type, and varying widths
    for (
        source,
        target,
    ), original_attrib, norm_attrib, edge_type, original_attrib in edge_data_normalized:
        if source == "null" or target == "null":
            continue
        gray_value = int(
            255 * (1 - norm_attrib)
        )  # Convert opacity to a grayscale value
        width = 3 if original_attrib >= threshold else 1
        if edge_type == "q":
            color = f"rgba(0, 255, 0, {norm_attrib})"  # Green with normalized opacity
        elif edge_type == "k":
            color = f"rgba(255, 0, 0, {norm_attrib})"  # Red with normalized opacity
        else:
            color = f"rgba({gray_value}, {gray_value}, {gray_value}, {norm_attrib})"
        title = f"attrib: {original_attrib:.3f}"  # Round original attrib to 3 decimal places
        net.add_edge(
            target, source, color=color, title=title, width=width
        )  # Reversed edge direction

    # Add faint vertical lines to delineate different values of `position`
    max_position = max(positions)
    for position in range(max_position + 1):
        x = (position + 0.5) * 150  # Adjust to place lines between integer positions
        start_node = f"line_{position}_start"
        end_node = f"line_{position}_end"
        net.add_node(
            start_node,
            label="",
            x=x,
            y=0,
            size=0,
            color="rgba(0,0,0,0)",
            fixed=True,
            physics=False,
        )
        net.add_node(
            end_node,
            label="",
            x=x,
            y=-1000,
            size=0,
            color="rgba(0,0,0,0)",
            fixed=True,
            physics=False,
        )
        net.add_edge(
            start_node,
            end_node,
            color="rgba(200, 200, 200, 0.5)",
            width=0.5,
            physics=False,
        )

        # Add labels at the bottom to indicate the positions
        label = tokens[position] if tokens and position < len(tokens) else str(position)
        label_node = f"label_{position}"
        net.add_node(
            label_node,
            label=label,
            size=0,
            x=(position * 150),
            y=20,
            color="black",
            fixed=True,
            physics=False,
            font={"size": 30},
        )

    # Generate the HTML file
    html_file = "graph.html"
    net.show(html_file)

    # Inject JavaScript for click event handling using PyVis API
    with open(html_file, "a") as f:
        f.write("""
        <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function() {
            network.on("click", function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    var node = network.body.nodes[nodeId];
                    var url = node.options.url;
                    if (url) {
                        window.open(url, '_blank');
                    }
                }
            });
        });
        </script>
        """)
    print(f"Generated {html_file}. Open this file in Live Server to view the graph.")
