from typing import Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import ModuleDict

from Block import Block

class ConvolutionalBlock(Block):
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
                 regularization: Optional[str] = None,
                 activation: Optional[nn.Module] = LeakyReLU(0.2),
                 ) -> None:
        
        conv_args = (in_channels, out_channels, kernel_size, stride, padding)
        conv_layer = ModuleDict({'conv_layer' : nn.ConvTranspose2d(*conv_args, output_padding) if transpose else nn.Conv2d(*conv_args)})
        
        super().__init__(in_channels, out_channels, conv_layer, noise_input, normalization, regularization, activation)
    
    def add_normalization(self, normalization, conv_layer): #is this necessary? we handle normalization in the Block initialization
        pass





