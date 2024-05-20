import transformer_lens as tl
from dataclasses import dataclass

@dataclass
class HookedSAEConfig(tl.HookedSAEConfig):
    # NOTE: When this is True, the error gradient is allowed to flow back to the input layer.
    # This is useful to do attribution patching for many nodes in parallel. 
    # Additionally, it's the approach taken in Marks et al. 
    allow_error_grad_to_input: bool = False 