from typing import Optional
from torch import Tensor
import torch.nn as nn

from Block import Block
from ConvolutionalBlock import ConvolutionalBlock

class ResidualBlock(Block):
    def __init__(self, 
                 num_blocks : int, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 output_padding: int = 0, 
                 transpose: bool = False, 
                 noise_input: bool = True,
                 normalization: Optional[str] = 'conditional',
                 regularization: Optional[str] = None,
                 activation: Optional[nn.Module] = nn.LeakyReLU(0.2)
                 ) -> None:
        
        conv_args = (in_channels, out_channels, kernel_size, stride, padding, output_padding, transpose, noise_input, normalization, regularization, activation)
        
        blocks = []
        for index in range(num_blocks):
            conv_block = (f'part_{index + 1}', ConvolutionalBlock(*conv_args))
            blocks.append(conv_block)
            
        super().__init__(blocks, False, None, None, None)
    
    def forward(self, input) -> Tensor:
        pass
