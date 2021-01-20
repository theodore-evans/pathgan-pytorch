from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.container import ModuleDict
import copy

from .Block import Block

class ResidualBlock(Block):
    def __init__(self, 
                 num_blocks : int, 
                 block_template : Block,
                 **kwargs
                 ) -> None:
        
        super().__init__(block_template.in_channels, block_template.out_channels, None)
        
        for index in range(num_blocks):
            self.add_module(f'part_{index + 1}', copy.deepcopy(block_template))
        
    def forward(self, inputs, **kwargs) -> Tensor:
        net = super().forward(inputs, **kwargs)
        return inputs + net