from typing import Union
import torch.nn as nn
import torch.nn.functional as F

class ConvScaleDecorator(nn.Module):
    def __init__(self,
                 conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d],
                 ) -> None:

        # Additional weight initialization used by authors for Up and Downscaling
        weights = F.pad(conv_layer.weight, [1, 1, 1, 1])
        conv_layer.weight = nn.Parameter(
            weights[:, :, 1:, 1:] + weights[:, :, 1:, :-1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1])

        # The filter is incremented in both directions
        filter_size = conv_layer.kernel_size[0] + 1
        conv_layer.kernel_size = (filter_size, filter_size)