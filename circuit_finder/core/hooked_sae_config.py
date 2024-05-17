from dataclasses import dataclass
import transformer_lens as tl


@dataclass
class HookedSAEConfig(tl.HookedSAEConfig):
    retain_grad: bool = (
        False  # If true, retains gradient on all tensors in the forward pass.
    )
