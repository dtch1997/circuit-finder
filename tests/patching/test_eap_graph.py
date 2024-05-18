from circuit_finder.patching.eap_graph import EAPGraph

edge_attribs = [
    # Note: these are written as (dest, src)
    (("null", "metric.12.14.0"), 0),
    (("metric.12.14.0", "attn.9.14.11368"), 5.972317218780518),
    (("metric.12.14.0", "attn.10.14.3849"), 1.9210439920425415),
    (("attn.10.14.3849", "mlp.0.4.5545"), 11.198338508605957),
]


def test_get_edges():
    eap_graph = EAPGraph(edge_attribs)
    assert set(eap_graph.get_edges()) == set([elem[0] for elem in edge_attribs])


def test_get_nodes():
    eap_graph = EAPGraph(edge_attribs)
    assert set(eap_graph.get_nodes()) == {
        "null",
        "metric.12.14.0",
        "attn.9.14.11368",
        "attn.10.14.3849",
        "mlp.0.4.5545",
    }


def test_get_src_nodes():
    eap_graph = EAPGraph(edge_attribs)
    assert set(eap_graph.get_src_nodes()) == {
        "metric.12.14.0",
        "attn.9.14.11368",
        "attn.10.14.3849",
        "mlp.0.4.5545",
    }


def test_get_dest_nodes():
    eap_graph = EAPGraph(edge_attribs)
    assert set(eap_graph.get_dest_nodes()) == {
        "null",
        "metric.12.14.0",
        "attn.10.14.3849",
    }
