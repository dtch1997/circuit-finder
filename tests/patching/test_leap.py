from circuit_finder.patching.leap import parse_node_name, get_node_name


def test_parse_node_name():
    node_name = "mlp.0.1.2"
    assert parse_node_name(node_name) == ("mlp", 0, 1, 2)


def test_get_node_name():
    components = ("mlp", 0, 1, 2)
    assert get_node_name(*components) == "mlp.0.1.2"
