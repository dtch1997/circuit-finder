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


from circuit_finder.patching.eap_graph import EAPGraph
import networkx as nx
from pyvis.network import Network
import numpy as np


import networkx as nx
from pyvis.network import Network
import numpy as np

def make_html_graph(leap, attrib_type="em", node_offset=8.0, show_error_nodes=False):
    graph = EAPGraph(leap.graph)
    tokens = leap.model.to_str_tokens(leap.model.tokenizer.batch_decode(leap.tokens), prepend_bos=False)
    corrupt_tokens = leap.model.to_str_tokens(leap.model.tokenizer.batch_decode(leap.corrupt_tokens), prepend_bos=False)

    error_graph = leap.error_graph if (len(leap.error_graph) > 0) and show_error_nodes else None
    
    edges = graph.get_edges()
    error_edges = [edge for edge, attrib, edge_type in error_graph] if error_graph else []

    edge_types = graph.get_edge_types()  # Get edge types
    error_edge_types = [edge_type for edge, attrib, edge_type in error_graph] if error_graph else []

    if attrib_type == "nn":
        attribs = graph.get_nn_attribs()
        error_attribs = [attrib[0] for edge, attrib, edge_type in error_graph] if error_graph else []
    elif attrib_type == "em":
        attribs = graph.get_em_attribs()
        error_attribs = [attrib[1] for edge, attrib, edge_type in error_graph] if error_graph else []
    else:
        print("Invalid attrib type chosen. Options are em and nn")
        return

    # Ensure the number of edges matches the number of edge types and attributes
    if len(edges) != len(edge_types) or len(edges) != len(attribs):
        print("Number of edges does not match number of edge types or attributes")
        return

    # Handle error edges and attributes
    if error_edges is None:
        error_edges = []
    if error_attribs is None:
        error_attribs = []
    if error_edge_types is None:
        error_edge_types = []

    # Ensure the number of error edges matches the number of error edge types and error attributes
    if len(error_edges) != len(error_edge_types) or len(error_edges) != len(error_attribs):
        print("Number of error edges does not match number of error edge types or attributes")
        return

    # Create a directed graph
    G = nx.DiGraph()
    
    # Color map for different modules (using RGB tuples)
    color_map = {
        'metric': (0, 255, 0),  # Green
        'mlp': (0, 0, 255),  # Blue
        'attn': (255, 165, 0),  # Orange
        'mlp_error': (0, 0, 255),  # Blue
        'attn_error': (255, 165, 0)  # Orange
    }
    
    # Track position offsets for nodes with the same (position, layer)
    offset_tracker = {'attn': {}, 'mlp': {}, 'metric': {}, 'attn_error': {}, 'mlp_error': {}}
    
    positions = set()  # To track unique position values
    
    # Create a dictionary for total_attrib
    node_total_attrib = {node: total_attrib for node, total_attrib in leap.nodes}

    # Collect all total_attribs for clipping
    all_total_attribs = [total_attrib for total_attrib in node_total_attrib.values()]
    p90_total_attrib = np.percentile(np.abs(all_total_attribs), 90)
    clipped_total_attribs = np.clip(np.abs(all_total_attribs), None, p90_total_attrib)

    # Normalize the clipped total_attribs to range from 0.2 to 1
    min_total_attrib = min(clipped_total_attribs)
    max_total_attrib = max(clipped_total_attribs)
    range_total_attrib = p90_total_attrib - min_total_attrib
    normalized_total_attribs = {node: min(1, 0.2 + 0.8 * (total_attrib - min_total_attrib) / range_total_attrib) for node, total_attrib in node_total_attrib.items()}

    for downstream, upstream in edges + error_edges:
        if downstream == 'null' or upstream == 'null':
            continue
        for node in [upstream, downstream]:
            if not G.has_node(node):
                parts = node.split('.')
                module, layer, position = parts[0], int(parts[1]), int(parts[2])
                _id = parts[3] if len(parts) > 3 else ''
                positions.add(position)
                color = color_map[module]
                total_attrib = node_total_attrib.get(node, 8888)

                title = f"layer {layer} pos {position} id {_id} \ntotal_attrib: {total_attrib:.3f}"

                # Determine the base position
                base_x = position * 150  # Increase x-axis spacing
                base_y = layer * 80  # Decrease y-axis spacing
                
                # Offset nodes based on module type
                if (position, layer) not in offset_tracker[module]:
                    offset_tracker[module][(position, layer)] = 0
                
                if module == 'attn':
                    pos_x = base_x - 10 - offset_tracker[module][(position, layer)] * node_offset  # Shift each "attn" node further left
                    pos_y = base_y - 15  # Shift each "attn" node further downward
                    url = f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{_id}"
                elif module == 'mlp':
                    pos_x = base_x + 15 + offset_tracker[module][(position, layer)] * node_offset  # Shift each "mlp" node further right
                    pos_y = base_y
                    url = f"https://www.neuronpedia.org/gpt2-small/{layer}-tres-dc/{_id}"
                else:
                    pos_x = base_x + offset_tracker[module][(position, layer)] * node_offset
                    pos_y = base_y
                    url = ""
                
                offset_tracker[module][(position, layer)] += 1
                
                # Get normalized opacity for the node
                opacity = normalized_total_attribs.get(node, 0.2)

                # Add node with attributes
                shape = 'dot' if 'error' not in module else 'square'
                color_with_opacity = f"rgba({color[0]}, {color[1]}, {color[2]}, {opacity})"
                G.add_node(node, title=title, color=color_with_opacity, id=_id, x=pos_x, y=pos_y, url=url, shape=shape)
        G.add_edge(upstream, downstream)
    
    # Combine edges with their attributes and types
    edge_data = list(zip(edges, attribs, edge_types))
    error_edge_data = list(zip(error_edges, error_attribs, error_edge_types))

    # Clip the attribs using the 95th percentile
    all_attribs = attribs + error_attribs
    p95_attrib = np.percentile(np.abs(all_attribs), 95)
    clipped_attribs = np.clip(np.abs(all_attribs), None, p95_attrib)
    
    # Normalize the clipped attribs to range from 0.2 to 1
    min_attrib = min(clipped_attribs)
    max_attrib = max(clipped_attribs)
    range_attrib = max_attrib - min_attrib
    
    normalized_attribs = [0.2 + 0.8 * (a - min_attrib) / range_attrib for a in clipped_attribs]

    # Determine the threshold for the top 5% of edges by attribute value
    threshold = np.percentile(np.abs(all_attribs), 95)

    # Map normalized attributes back to edges
    edge_data_normalized = [(e[0], e[1], norm_attrib, e[2], original_attrib) for e, norm_attrib, original_attrib in zip(edge_data + error_edge_data, normalized_attribs, all_attribs)]
    
    # Create pyvis network
    net = Network(notebook=True, directed=True, cdn_resources='remote')
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
        net.add_node(node, label=data['id'], title=data['title'], color=data['color'], x=data['x'], y=-data['y'], url=data['url'], shape=data['shape'])
    
    # Track edge counts for parallel edges
    edge_count = {}

    # Add edges with normalized opacity, hover title, color based on edge type, and varying widths
    for (source, target), original_attrib, norm_attrib, edge_type, original_attrib in edge_data_normalized:
        if source == 'null' or target == 'null':
            continue
        gray_value = int(255 * (1 - norm_attrib))  # Convert opacity to a grayscale value
        width = 3 if abs(original_attrib) >= threshold else 1
        if edge_type == 'q':
            color = f"rgba(0, 255, 0, {norm_attrib})"  # Green with normalized opacity
        elif edge_type == 'k':
            color = f"rgba(255, 0, 0, {norm_attrib})"  # Red with normalized opacity
        else:
            color = f"rgba({gray_value}, {gray_value}, {gray_value}, {norm_attrib})"
        title = f"attrib: {original_attrib:.3f}"  # Round original attrib to 3 decimal places

        # Determine the edge count between the same pair of nodes
        if (source, target) not in edge_count:
            edge_count[(source, target)] = 0
        edge_offset = edge_count[(source, target)]
        edge_count[(source, target)] += 1

        # Determine the style based on whether the attrib is negative
        style = "dashed" if original_attrib < 0 else "curvedCCW"
        roundness = 0.2 * edge_offset if original_attrib >= 0 else 0.1 * edge_offset

        # Add offset to the edge
        net.add_edge(target, source, color=color, title=title, width=width, dashes=original_attrib < 0)  # Reversed edge direction
    
    # Add "clean tokens" label to the left
    net.add_node(f'clean_tokens_label', label='CLEAN TOKENS', size=0, x=-300, y=100, color='black', fixed=True, physics=False, font={'size': 30})
    # Add "corrupt tokens" label to the left
    net.add_node(f'corrupt_tokens_label', label='CORRUPT TOKENS', size=0, x=-300, y=100 + 30*min(len(tokens), 10) + 20, color='black', fixed=True, physics=False, font={'size': 30})

    # Add faint vertical lines to delineate different values of `position`
    max_position = max(positions)
    for position in range(max_position + 1):
        x = (position + 0.5) * 150  # Adjust to place lines between integer positions
        start_node = f'line_{position}_start'
        end_node = f'line_{position}_end'
        net.add_node(start_node, label='', x=x, y=0, size=0, color='rgba(0,0,0,0)', fixed=True, physics=False)
        net.add_node(end_node, label='', x=x, y=-1000, size=0, color='rgba(0,0,0,0)', fixed=True, physics=False)
        net.add_edge(start_node, end_node, color='rgba(200, 200, 200, 0.5)', width=0.5, physics=False)
        
        # Add labels at the bottom to indicate the positions
        if tokens and (position < len(tokens[0])):
            y_offset = 100
            for idx, token_set in enumerate(tokens[:10]):  # Limit to max 10 rows
                if position < len(token_set):
                    label = token_set[position]
                    net.add_node(f'label_{position}_{idx}', label=label, size=0, x=(position * 150), y=y_offset, color='black', fixed=True, physics=False, font={'size': 30})
                    y_offset += 30  # Increase the y offset for the next set of labels
            
        if corrupt_tokens and (position < len(corrupt_tokens[0])):
            y_offset = 100 + 30*min(len(tokens), 10) + 20
            for idx, token_set in enumerate(corrupt_tokens[:10]):  # Limit to max 10 rows
                if position < len(token_set):
                    label = token_set[position]
                    net.add_node(f'corrupt_label_{position}_{idx}', label=label, size=0, x=(position * 150), y=y_offset , color='red', fixed=True, physics=False, font={'size': 30})
                    y_offset += 30  # Increase the y offset for the next set of labels

    # Generate the HTML file
    html_file = 'graph.html'
    net.show(html_file)

    # Inject JavaScript for click event handling using PyVis API
    with open(html_file, 'a') as f:
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
    print(f'Generated {html_file}. Open this file in Live Server to view the graph.')
