from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.container import ModuleDict
import copy

from .Block import Block

#TODO: check whether subblocks are passed in as references, and implement with deep clones of a template if so
class ResidualBlock(Block):
    def __init__(self, 
                 num_blocks : int, 
                 block : Block,
                 **kwargs
                 ) -> None:
        
        super().__init__(block.in_channels, block.out_channels, None)
        
        for index in range(num_blocks):
            self.add_module(f'part_{index + 1}', copy.deepcopy(block))
        
    def forward(self, input, **kwargs) -> Tensor:
        net = super().forward(input, **kwargs)
        return input + net