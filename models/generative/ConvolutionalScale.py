
from torch import Tensor
from ConvolutionalBlock import ConvolutionalBlock

class ConvolutionalScale(ConvolutionalBlock):
    def __init__(self,
                 *args, 
                 **kwargs) -> None :
        
        kwargs['transpose'] = kwargs['upscale']
        del kwargs['upscale']
        
        super().__init__(*args, **kwargs)

        # Set the stride to 2 for both Up and Downscaling
        self.conv_layer.stride = 2

        # Additional weight initialization used by authors for Up and Downscaling
        weights = F.pad(self.conv_layer.weight, (1,1,1,1))
        self.conv_layer.weight = nn.Parameter(weights[:,:,1:,1:] + weights[:,:,1:,:-1] + weights[:,:,:-1,1:] + weights[:,:,:-1,:-1])

        # The filter is incremented in both directions
        filter_size = self.conv_layer.kernel_size[0] + 1
        self.conv_layer.kernel_size = (filter_size, filter_size)