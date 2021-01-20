from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleDict

from .Block import Block

#TODO: refactor, perhaps phase out redundant class. i.e. merge with conv scale. 
#TODO: wrapper class for convolutional layer, with output padding/size, 'SAME' padding mode
class ConvolutionalBlock(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 transpose: bool = False,
                 output_padding: int = 0,
                 **kwargs,
                 ) -> None:
        
        conv_args = (in_channels, out_channels, kernel_size, stride, padding)

        conv_layer = nn.ConvTranspose2d(*conv_args, output_padding) if transpose else nn.Conv2d(*conv_args)
        
        layer_dict = ModuleDict({'conv_layer' : conv_layer})
        
        super().__init__(in_channels, out_channels, layer_dict, **kwargs)
