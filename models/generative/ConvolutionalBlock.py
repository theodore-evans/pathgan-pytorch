from typing import Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

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
                 activation: Optional[nn.Module] = nn.LeakyReLU(0.2),
                 **kwargs
                 ) -> None:

        conv_args = (in_channels, out_channels, kernel_size, stride, padding)
        conv_layer = nn.ConvTranspose2d(*conv_args, output_padding) if transpose else nn.Conv2d(*conv_args)
        
        super().__init__([('conv_layer', conv_layer)], noise_input, normalization, regularization, activation)
    
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



