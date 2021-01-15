from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.container import ModuleDict

from .Block import Block
from .ConvolutionalBlock import ConvolutionalBlock

class ResidualBlock(Block):
    def __init__(self, 
                 num_blocks : int, 
                 block : Block,
                 ) -> None:
            
        blocks = ModuleDict()
        for index in range(num_blocks):
            blocks[f'part_{index + 1}'] = block
            
        super().__init__(block.in_channels, block.out_channels, blocks)
        
    def forward(self, input, **kwargs):
        net = super().forward(input, **kwargs)
        return input + net
        
print(ResidualBlock(2, ConvolutionalBlock(200, 200, 3, 1, 0)))