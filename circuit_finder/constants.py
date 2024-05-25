import torch
import pathlib

GPT_2_SMALL: str = "gpt2"
ALL_GPT_2_SMALL_LAYERS: list[int] = list(range(12))

device = "cuda" if torch.cuda.is_available() else "cpu"

ProjectDir = pathlib.Path(__file__).parent.parent
