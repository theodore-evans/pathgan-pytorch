
from torch import Tensor
from .ConvolutionalBlock import ConvolutionalBlock
import torch.nn.functional as F
from torch import nn


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
        weights = F.pad(self.conv_layer.weight, (1, 1, 1, 1))
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

