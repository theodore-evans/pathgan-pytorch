from typing import Callable, Optional, Union
import torch.nn as nn
from torch.nn.modules.container import ModuleDict
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.utils.spectral_norm import spectral_norm

from .Block import Block
from .ConvolutionalScale import UpscaleConv2d, DownscaleConv2d
from .utils import apply_same_padding

default_layer_names = dict({nn.Conv2d : "conv_layer",
                            nn.ConvTranspose2d : "conv_transpose_layer",
                            UpscaleConv2d: "upscale_layer",
                            DownscaleConv2d: "downscale_layer"})

class ConvolutionalBlock(Block):
    def __init__(self,
                 conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d],
                 layer_name: Optional[str] = None,
                 same_padding: bool = True,
                 regularization: Callable[..., nn.Module] = None,
                 **kwargs
                 ) -> None:
        
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        
        if same_padding:
            apply_same_padding(conv_layer)
            
        if layer_name is None:
            layer_name = default_layer_names[type(conv_layer)]
            
        layers = ModuleDict({layer_name : conv_layer})
        
        if regularization is spectral_norm:
            dim = 1 if isinstance(conv_layer, (UpscaleConv2d, nn.ConvTranspose2d)) else 1
            regularization = lambda x: spectral_norm(x, name = 'filter', dim = dim)
        
        super().__init__(in_channels, out_channels, layers, **kwargs)
        