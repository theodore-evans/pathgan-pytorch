from typing import Optional
import torch.nn as nn
from torch import Tensor

from Placeholder import Placeholder
from ConditionalNorm import ConditionalNorm

class ConvolutionalBlock(nn.Module):
    def __init__(self, 
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
        
        super().__init__()
                
        conv_args = (in_channels, out_channels, kernel_size, stride, padding)
        conv_layer = nn.ConvTranspose2d(*conv_args, output_padding) if transpose else nn.Conv2d(*conv_args)
        self.add_module('conv_layer', conv_layer)
        
        if noise_input: 
            self.add_module(f'noise_input', Placeholder())
            
        if normalization == 'conditional': 
            self.add_module(f'conditional_instance_normalization', ConditionalNorm())
        
        if activation is not None:
            self.add_module(f'activation', activation)
    
    def forward(self, input: Tensor) -> Tensor:
        pass
    