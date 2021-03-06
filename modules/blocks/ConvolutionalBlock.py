from typing import  Optional, Union
import torch.nn as nn
from torch.nn.modules.container import ModuleDict
from torch.tensor import Tensor

from modules.blocks.Block import Block
from modules.blocks.ConvolutionalScale import UpscaleConv2d, DownscaleConv2d
from modules.utils import apply_same_padding
from modules.types import size_2_t

default_layer_names = dict({nn.Conv2d : "conv_layer",
                            nn.ConvTranspose2d : "conv_transpose_layer",
                            UpscaleConv2d: "upscale_layer",
                            DownscaleConv2d: "downscale_layer"})

class ConvolutionalBlock(Block):
    def __init__(self,
                 conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d],
                 layer_name: Optional[str] = None,
                 same_padding: bool = True,
                 **kwargs
                 ) -> None:
        
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        
        if same_padding:
            apply_same_padding(conv_layer)
            
        if layer_name is None:
            layer_name = default_layer_names[type(conv_layer)]
            
        layers = ModuleDict({layer_name : conv_layer})
        
        super().__init__(in_channels, out_channels, layers, **kwargs)
        
    def forward(self, *args, **kwargs) -> Tensor:
        return super().forward(*args, **kwargs)
        
class UpscaleBlock(ConvolutionalBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: size_2_t,
                 **kwargs
                 )->None:
        
        super().__init__(UpscaleConv2d(in_channels, out_channels, kernel_size), **kwargs)
    
class DownscaleBlock(ConvolutionalBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: size_2_t,
                 **kwargs
                 )->None:
        
        super().__init__(DownscaleConv2d(in_channels, out_channels, kernel_size), **kwargs)
        