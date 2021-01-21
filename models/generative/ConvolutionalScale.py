import torch
from torch import Tensor
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import ModuleDict
from torch import nn
from typing import Optional, Tuple, Union

from .Block import Block
from .ConvolutionalBlock import ConvolutionalBlock

def kernel_padding_hook(module, *args):
    weights = F.pad(module.kernel, [1, 1, 1, 1])
    module.conv_layer.weight = nn.Parameter(
            weights[:, :, 1:, 1:] + weights[:, :, 1:, :-1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1])

class ConvolutionalScale(ConvolutionalBlock):
    def __init__(self,
                 *args,
                 **kwargs) -> None:

        kwargs['transpose'] = kwargs['upscale']
        del kwargs['upscale']

        super().__init__(*args, **kwargs)
        
        # Set the stride to 2 for both Up and Downscaling
        self.conv_layer.stride = 2

        # Additional weight initialization used by authors for Up and Downscaling
        weights = F.pad(self.conv_layer.weight, [1, 1, 1, 1])
        self.conv_layer.weight = nn.Parameter(
            weights[:, :, 1:, 1:] + weights[:, :, 1:, :-1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1])

        # The filter is incremented in both directions
        filter_size = self.conv_layer.kernel_size[0] + 1
        self.conv_layer.kernel_size = (filter_size, filter_size)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv_layer(input)

    @staticmethod
    def calculate_padding_upscale(input_size: int,
                          stride: int, kernel_size: int) -> Tuple[int, int]:
        # Rest = out_pad - 2 * pad
        output_size = 2 * input_size
        rest = output_size - (input_size - 1) * stride  - (kernel_size -1) - 1
        if rest == 0:
            return 0, 0
        elif rest < 0:
            if rest % 2 == 0:
                return rest // -2, 0
            else:
                return rest // -2 + 1, 1
        else:
            return 0, rest

class ConvolutionalScaleVanilla(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: Union[int,tuple],
                 output_padding: int = 0,
                 upscale: bool = False,
                 **kwargs
                 ) -> None:
        
        stride = 2
        regularization = None
        self.upscale = upscale

        weights_shape = (in_channels, out_channels, kernel_size, kernel_size) if upscale else (out_channels, in_channels, kernel_size, kernel_size)
        modules = ModuleDict()
        conv_args = (in_channels, out_channels, kernel_size + 1, stride, padding)
        
        modules['conv_layer'] = nn.ConvTranspose2d(*conv_args, output_padding) if upscale else nn.Conv2d(*conv_args)
        
        super().__init__(in_channels, out_channels, modules, **kwargs)

        self.register_parameter(name='kernel', param = nn.parameter.Parameter(torch.ones(weights_shape)))
        self.register_forward_hook(kernel_padding_hook)
        self.conv_layer = spectral_norm(self.conv_layer)

    def forward(self, input : Tensor, **kwargs) -> Tensor: 
        net = input
        batch_size, channels, image_width, image_height = input.shape
        for module in self.children():
            if type(module) == ConvTranspose2d:
                net = module(net, output_size=(batch_size, channels, image_width * 2, image_height * 2))
            else:
                net = module(net, **kwargs)
        return net

