import torch
from torch import Tensor
from torch.nn.utils.spectral_norm import spectral_norm
from .ConvolutionalBlock import ConvolutionalBlock
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import ModuleDict
from torch import nn
from .Block import Block
from typing import Optional

def kernel_padding_hook(module, *args):
    weights = F.pad(module.kernel, [1, 1, 1, 1])
    module.weight = nn.Parameter(
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

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.conv_layer(input)

    @staticmethod
    def calculate_padding_upscale(input_size: int,
                          stride: int, kernel_size: int) -> int:
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
                 padding: int,
                 output_padding: int = 0,
                 upscale: bool = False,
                 noise_input: bool = True,
                 normalization: Optional[str] = 'conditional',
                 regularization: Optional[str] = None,
                 activation: Optional[nn.Module] = LeakyReLU(0.2),
                 ) -> None:
        
        stride = 2
        regularization = None

        weights_shape = (in_channels, out_channels, kernel_size, kernel_size) if upscale else (out_channels, in_channels, kernel_size, kernel_size)

        # Initialize the kernel as a seperate parameter to mutate shape later
        conv_args = (in_channels, out_channels, kernel_size + 1, stride, padding)
        conv_layer = ModuleDict({'conv_layer' : nn.ConvTranspose2d(*conv_args, output_padding) if upscale else nn.Conv2d(*conv_args)})
        
        super().__init__(in_channels, out_channels, conv_layer, noise_input, normalization, regularization, activation)

        self.register_parameter(name='kernel', param = nn.parameter.Parameter(torch.empty(weights_shape)))
        self.register_forward_hook(kernel_padding_hook)
        self.conv_layer = spectral_norm(self.conv_layer)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.conv_layer(input)

