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
        return sorted([edge for edge, _ in self.graph if filter_fn(edge)])

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
        graph = [(Node(dest), Node(src), attrib) for dest, src, attrib in json["graph"]]
        return EAPGraph(graph)


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
