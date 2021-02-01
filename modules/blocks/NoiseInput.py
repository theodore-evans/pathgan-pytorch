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
        input_is_image = inputs.dim() == 4
        if noise is None:
            if hasattr(self, 'noise') and self.noise is not None:
                noise = self.noise # type: ignore
            else:
                noise_shape = (inputs.size(0), 1, inputs.size(2), inputs.size(3)) if input_is_image else (inputs.size(0), 1)
                noise = torch.randn(*noise_shape, device=inputs.device, dtype=inputs.dtype)
        
        shape = (1, -1, 1, 1) if input_is_image else (1, -1)
        net = inputs + self.weight.view(*shape) * noise
        return net