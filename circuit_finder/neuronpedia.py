import json
import urllib.parse

from circuit_finder.core.types import SaeFamily


def get_neuronpedia_url_for_quick_list(
    layer: int,
    features: list[int],
    sae_family: SaeFamily = "res-jb",
    name: str = "temporary_list",
):
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {
            "modelId": "gpt2-small",
            "layer": f"{layer}-{sae_family}",
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    return url
