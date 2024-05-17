from circuit_finder.pretrained import load_mlp_transcoders


def test_load_mlp_transcoders():
    transcoder = load_mlp_transcoders(layers=[0])
    assert transcoder[0] is not None
