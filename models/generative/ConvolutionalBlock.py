from typing import Optional, Union
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
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 output_padding: Union[int, tuple] = 0,
                 transpose: bool = False,
                 pad_to_maintain_size: bool = False,
                 **kwargs
                 ) -> None:
        
        conv_args = (in_channels, out_channels, kernel_size, stride, padding)
        
        conv_layer = nn.ConvTranspose2d(*conv_args, output_padding) if transpose else nn.Conv2d(*conv_args)
            
        if pad_to_maintain_size:
            effective_filter_size = lambda i : conv_layer.dilation[i] * (conv_layer.kernel_size[i] - 1) + 1
            for dim in range(2):
                if conv_layer.stride[dim] != 1: 
                    raise Exception("In order to maintain input image size, stride must be 1")
                if effective_filter_size(dim) % 2 == 0:
                    raise Exception("In order to maintain input image size, effective filter size must be odd")
            conv_layer.padding = ( (effective_filter_size(0) - 1) // 2, (effective_filter_size(1) - 1) // 2 )
             
        module_dict = ModuleDict({'conv_layer' : conv_layer})
        
        super().__init__(in_channels, out_channels, module_dict = module_dict, **kwargs)