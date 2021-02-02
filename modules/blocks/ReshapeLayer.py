import torch.nn as nn
import torch
from torch.tensor import Tensor
from modules.blocks.Block import Block
from modules.types import size_2_t
from modules.utils import pair
class ReshapeLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 image_shape: size_2_t,
                 ) -> None:
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        out_channels_divides_in_channels = in_channels % out_channels == 0
        
        if not out_channels_divides_in_channels:
            raise ValueError("Output channels must divide input channels")

        self.image_shape = pair(image_shape)
        image_shape_factorises_in_channels = in_channels == out_channels * self.image_shape[0] * self.image_shape[1]
        if not image_shape_factorises_in_channels:
            raise ValueError("input channels must factorise into out channels, image height and width")
                
    def forward(self, inputs: Tensor, *args, **kwargs) -> Tensor:
        if inputs.size(1) != self.in_channels:
            raise ValueError("in_channels of block does not match in_channels of input")
        output_shape = (-1, self.out_channels, self.image_shape[0], self.image_shape[1])
        return torch.reshape(inputs, output_shape)