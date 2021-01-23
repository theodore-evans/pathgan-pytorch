from abc import ABC
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleDict
from torch.nn.modules.conv import Conv2d

from .Block import Block
from .ConvolutionalScale import ConvolutionalScale
from .utils import apply_same_padding

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
            layer_name = 'conv_layer' if isinstance(conv_layer, nn.Conv2d) else 'conv_transpose_layer'
            
        module_dict = ModuleDict({layer_name : conv_layer})
        
        super().__init__(in_channels, out_channels, module_dict, **kwargs)
class UpscaleBlock(ConvolutionalBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 **kwargs) -> None:

        layer_name = 'upscale_layer'
        layer = ConvolutionalScale(
            in_channels, out_channels, kernel_size, upscale=True)
        kwargs['regularization'] = lambda x: x
            
        super().__init__(layer, layer_name, **kwargs)
        
class DownscaleBlock(ConvolutionalBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 **kwargs) -> None:

        layer_name = 'downscale_layer'
        layer = ConvolutionalScale(
            out_channels, in_channels, kernel_size) #TODO: check why we switch in and out channels
        kwargs['regularization'] = lambda x: x
            
        super().__init__(layer, layer_name, **kwargs)
        