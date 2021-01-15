from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn

#https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
class NoiseInput(nn.Module):
    def __init__(self,
                 in_channels : int
                 ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x : Tensor, noise : Optional[Tensor] = None) -> Tensor:
        if noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            if self.noise is None:
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            else:
                noise = self.noise
        
        net = input + self.weight.view(1, -1, 1, 1) * noise
        return net