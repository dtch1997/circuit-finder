# --- context manager for replacing MLP sublayers with transcoders ---
import torch
import torch.nn as nn
import transformer_lens as tl
from typing import Sequence
from transcoders_slim.transcoder import  Transcoder
from circuit_finder.core.types import LayerIndex

MLP = nn.Module

class TranscoderWrapper(torch.nn.Module):
    """ Wrapper class around a transcoder """

    def __init__(self, 
        transcoder: Transcoder,
    ):
        super().__init__()
        self.transcoder = transcoder

    def forward(self, x):
        # TODO: Track error terms
        return self.transcoder(x)[0]

class TranscoderReplacementContext:
    """ Context manager to replace MLP sublayers with transcoders """
    model: tl.HookedTransformer
    transcoders: Sequence[Transcoder]
    layers: Sequence[LayerIndex]
    original_mlps: Sequence[MLP]

    def __init__(
        self, 
        model: tl.HookedTransformer,
        transcoders: Sequence[Transcoder]
    ):
        self.layers = [t.cfg.hook_point_layer for t in transcoders]
        self.original_mlps = [ model.blocks[layer].mlp for layer in self.layers ]
        self.transcoders = transcoders
        self.model = model
    
    def __enter__(self):
        for transcoder in self.transcoders:
           self.model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)

    def __exit__(self, exc_type, exc_value, exc_tb):
        for layer, mlp in zip(self.layers, self.original_mlps):
            self.model.blocks[layer].mlp = mlp

class ZeroAblationWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x*0.0

class ZeroAblationContext:
    def __init__(self, model, layers):
        self.original_mlps = [ model.blocks[i].mlp for i in layers ]
        
        self.layers = layers
        self.model = model
    
    def __enter__(self):
        for layer in self.layers:
           self.model.blocks[layer].mlp = ZeroAblationWrapper()

    def __exit__(self, exc_type, exc_value, exc_tb):
        for layer, mlp in zip(self.layers, self.original_mlps):
            self.model.blocks[layer].mlp = mlp