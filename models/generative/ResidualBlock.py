from typing import Optional
from torch import Tensor
import torch.nn as nn

from ConvolutionalBlock import ConvolutionalBlock

class ResidualBlock(nn.Module):
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
                 activation: Optional[nn.Module] = nn.LeakyReLU(0.2)
                 ) -> None:
        
        conv_args = (in_channels, out_channels, kernel_size, stride, padding, output_padding, transpose, noise_input, normalization, activation)
        
        super().__init__()
        for index in range(num_blocks):
            self.add_module(f'part_{index + 1}', ConvolutionalBlock(*conv_args))
    
    def forward(self, input) -> Tensor:
        pass