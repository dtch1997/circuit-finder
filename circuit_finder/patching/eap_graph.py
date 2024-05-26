
import pandas as pd
from typing import Callable
from circuit_finder.core.types import (
    LayerIndex,
    FeatureIndex,
    TokenIndex,
    Node,
    Edge,
    Attrib,
    ModuleName,
    parse_node_name,
)


def convert_graph_to_dataframe(graph: "EAPGraph") -> pd.DataFrame:
    rows = []
    for edge, edge_info in graph.graph:
        dest, src = edge
        if "metric" in src:
            continue
        nn_grad, nn_attrib, em_grad, em_attrib = edge_info
        src_module_type, src_layer, src_token, src_feature = parse_node_name(src)
        dest_module_type, dest_layer, dest_token, dest_feature = parse_node_name(dest)
        rows.append(
            {
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
            }
        )
    df = pd.DataFrame(rows)
    return df


class EAPGraph:
    """A class representing a circuit of nodes."""

    graph: list[tuple[Edge, Attrib]]

    def __init__(self, graph: list[tuple[Edge, Attrib]] = []):
        """Initialize the graph"""
        self.graph = graph

    def get_edges(
        self, filter_fn: Callable[[Edge], bool] = lambda x: True
    ) -> list[Edge]:
        """Get the edges of the graph"""
        return sorted([edge for edge, _, _ in self.graph if filter_fn(edge)])
    
    def get_edge_types(
        self, filter_fn: Callable[[Edge], bool] = lambda x: True
    ) -> list[Edge]:
        unsorted_edges = [edge for edge, _ , _ in self.graph if filter_fn(edge)]
        unsorted_types = [
            type for edge, _, type in self.graph if filter_fn(edge)]
        
        combined = list(zip(unsorted_edges, unsorted_types))

        # Sort combined list based on the elements of unsorted_edges
        combined_sorted = sorted(combined, key=lambda x: x[0])

        # Unzip the combined list
        sorted_edges, sorted_types = zip(*combined_sorted)

        # Convert back to lists if needed
        sorted_edges = list(sorted_edges)
        sorted_types = list(sorted_types)
        return sorted_types

    def get_nodes(
        self, filter_fn: Callable[[Node], bool] = lambda x: True
    ) -> list[Node]:
        """Get the nodes of the graph"""
        node_set = set()
        for dest, src in self.get_edges():
            if filter_fn(src):
                node_set.add(src)
            if filter_fn(dest):
                node_set.add(dest)
        return sorted(list(node_set))

    def get_src_nodes(
        self, filter_fn: Callable[[Node], bool] = lambda x: True
    ) -> list[Node]:
        """Get the source nodes of the graph"""
        node_set = set()
        for _, src in self.get_edges():
            if filter_fn(src):
                node_set.add(src)
        return sorted(list(node_set))

    def get_dest_nodes(
        self, filter_fn: Callable[[Node], bool] = lambda x: True
    ) -> list[Node]:
        """Get the destination nodes of the graph"""
        node_set = set()
        for dest, _ in self.get_edges():
            if filter_fn(dest):
                node_set.add(dest)
        return sorted(list(node_set))

    def get_nn_attribs(
        self, filter_fn: Callable[[Node], bool] = lambda x: True
    ) -> list[Node]:
        """Get the node->node attributions of the graph"""
        unsorted_edges = [edge for edge, _ , _ in self.graph if filter_fn(edge)]
        unsorted_attribs = [
            attribs[1] for edge, attribs, _ in self.graph if filter_fn(edge)]
        
        combined = list(zip(unsorted_edges, unsorted_attribs))

        # Sort combined list based on the elements of unsorted_edges
        combined_sorted = sorted(combined, key=lambda x: x[0])

        # Unzip the combined list
        sorted_edges, sorted_attribs = zip(*combined_sorted)

        # Convert back to lists if needed
        sorted_edges = list(sorted_edges)
        sorted_attribs = list(sorted_attribs)
        return sorted_attribs

    def get_em_attribs(
        self, filter_fn: Callable[[Node], bool] = lambda x: True
    ) -> list[Node]:
        """Get the node->node attributions of the graph"""
        unsorted_edges = [edge for edge, _, _ in self.graph if filter_fn(edge)]
        unsorted_attribs = [
            attribs[3] for edge, attribs, _ in self.graph if filter_fn(edge)]
        
        combined = list(zip(unsorted_edges, unsorted_attribs))

        # Sort combined list based on the elements of unsorted_edges
        combined_sorted = sorted(combined, key=lambda x: x[0])

        # Unzip the combined list
        sorted_edges, sorted_attribs = zip(*combined_sorted)

        # Convert back to lists if needed
        sorted_edges = list(sorted_edges)
        sorted_attribs = list(sorted_attribs)
        return sorted_attribs

    def to_json(self) -> dict:
        """Convert the graph to a JSON object"""
        return {
            "graph": [
                (str(dest), str(src), attrib) for (dest, src), attrib in self.graph
            ]
        }

    @staticmethod
    def from_json(json: dict) -> "EAPGraph":
        """Load the graph from a JSON object"""
        graph = [
            ((Node(dest), Node(src)), attrib) for dest, src, attrib in json["graph"]
        ]
        return EAPGraph(graph)

    def has_node(self, module_name, layer, feature) -> bool:
        """Check if a node is in the graph"""
        for node in self.get_nodes():
            _, layer_idx, _, feature_idx = parse_node_name(node)
            if layer_idx == layer and feature_idx == feature:
                return True
        return False


def get_feature_and_token_idx_of_nodes(
    graph: EAPGraph, module_name: ModuleName, layer_idx: LayerIndex
) -> tuple[list[FeatureIndex], list[TokenIndex]]:
    """Get the feature and token indices of upstream nodes for all nodes at a given module and layer

    Note: Only searches over upstream nodes in the graph.

    Returns:
        feature_indices : list of feature indices
        token_indices: list of token positions

    These can be zipped together to get the (token_idx, feature_idx) pairs of important nodes.

    Args:
        module_name : downstream module name.
        layer_idx : index of the downstream layer.
    """
    feature_indices: list[FeatureIndex] = []
    token_indices: list[TokenIndex] = []

    def filter_by_module_and_layer(node: Node) -> bool:
        (module, layer_idx, _, _) = parse_node_name(node)
        return module == module_name and layer_idx == layer_idx

    # Get all nodes that are currently in the graph
    nodes_set: set[Node] = set()
    for node in graph.get_src_nodes(filter_fn=filter_by_module_and_layer):
        nodes_set.add(node)
    nodes: list[Node] = list(nodes_set)

    # Filter by module and layer
    # TODO: It seems like we could do this previously but ig it doesn't matter.
    for node in nodes:
        (_, _, token_idx, feature_idx) = parse_node_name(node)
        feature_indices += [int(feature_idx)]
        token_indices += [int(token_idx)]
    return feature_indices, token_indices
