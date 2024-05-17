from circuit_finder.pretrained import load_attn_saes


def test_load_attn_saes():
    sae = load_attn_saes(layers=[0])
    assert sae[0] is not None
