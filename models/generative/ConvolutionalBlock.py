from typing import Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from models.generative.Placeholder import Placeholder
from models.generative.ConditionalNorm import ConditionalNorm

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
                 activation: Optional[nn.Module] = nn.LeakyReLU(0.2),
                 init: Optional[str] = 'orthogonal',
                 **kwargs
                 ) -> None:

        super().__init__()

        conv_args = (in_channels, out_channels, kernel_size, stride, padding)
        conv_layer = nn.ConvTranspose2d(
            *conv_args, output_padding) if transpose else nn.Conv2d(*conv_args)

        # Different init modes applied only to the weight tensor of the Conv Layer
        if init == 'orthogonal':
            nn.init.orthogonal_(conv_layer.weight)
        elif init == 'xavier':
            nn.init.xavier_uniform_(conv_layer.weight)
        elif init == 'normal':
            nn.init.normal_(conv_layer.weight, std=0.02)

        # Bias is initialized with constant 0 values, still trainable
        nn.init.constant_(conv_layer.bias, 0.)

        if normalization == 'spectral':
            self.add_module('conv_layer', spectral_norm(
                conv_layer, n_power_iterations=1))
        else:
            self.add_module('conv_layer', conv_layer)

        if noise_input:
            self.add_module(f'noise_input', Placeholder())
            
        if normalization == 'conditional': 
            self.add_module(f'conditional_instance_normalization', ConditionalNorm())
        
        if activation is not None:
            self.add_module(f'activation', activation)
    
    def add_normalization(self, normalization, conv_layer):
        pass

    def forward(self, input: Tensor) -> Tensor:
        pass


class ConvolutionalScale(ConvolutionalBlock):
    def __init__(self,
                 *args, 
                 **kwargs) -> None :
        
        kwargs['transpose'] = kwargs['upscale']
        del kwargs['upscale']
        
        super().__init__(*args, **kwargs)

        # Set the stride to 2 for both Up and Downscaling
        self.conv_layer.stride = (2,2)

        # Additional weight initialization used by authors for Up and Downscaling
        weights = F.pad(self.conv_layer.weight, (1,1,1,1))
        self.conv_layer.weight = nn.Parameter(weights[:,:,1:,1:] + weights[:,:,1:,:-1] + weights[:,:,:-1,1:] + weights[:,:,:-1,:-1])

        # The filter is incremented in both directions
        filter_size = self.conv_layer.kernel_size[0] + 1
        self.conv_layer.kernel_size = (filter_size, filter_size)



