from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn

class NoiseInput(nn.Module):
    def __init__(self,
                 channels : int
                 ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs : Tensor, noise : Optional[Tensor] = None) -> Tensor:
        if noise is None:
            if hasattr(self, 'noise') and self.noise is not None:
                noise = self.noise # type: ignore
            else:
                noise = torch.randn(inputs.size(0), 1, inputs.size(2), inputs.size(3), 
                                    device=inputs.device, dtype=inputs.dtype)
        
        net = inputs + self.weight.view(1, -1, 1, 1) * noise
        return net
    