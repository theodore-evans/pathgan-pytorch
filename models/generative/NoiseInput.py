from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn

#https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
class NoiseInput(nn.Module):
    def __init__(self,
                 channels : int
                 ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, inputs : Tensor, noise : Optional[Tensor] = None) -> Tensor:
        if noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            if self.noise is not None:
                noise = self.noise
            else:
                noise = torch.randn(inputs.size(0), 1, inputs.size(2), inputs.size(3), device=inputs.device, dtype=inputs.dtype)
        
        net = inputs + self.weight.view(1, -1, 1, 1) * noise
        return net
    