from typing import Optional
import torch
import math
from torch.tensor import Tensor
from modules.blocks.Block import Block
from modules.types import size_2_t
from modules.utils import pair
class ReshapeBlock(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 image_shape: Optional[size_2_t] = None,
                 **kwargs
                 ) -> None:
        
        super().__init__(in_channels, out_channels)
        
        out_channels_divides_in_channels = in_channels % out_channels == 0
        
        if not out_channels_divides_in_channels:
            raise ValueError("Output channels must divide input channels")
        
        image_area = in_channels // out_channels
        
        if image_shape is not None:
            self.image_shape = pair(image_shape)
            image_shape_factorises_in_channels = image_area == self.image_shape[0] * self.image_shape[1]
            if not image_shape_factorises_in_channels:
                raise ValueError("input channels must factorise into out channels, image height and width")
        else:
            if not is_perfect_square(image_area):
                raise ValueError('''Image width * height must be perfect square, 
                                 unless a non-square image shape is explicitly provided''')
            image_side = int(math.sqrt(in_channels // out_channels))
            self.image_shape = (image_side, image_side)
                
    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.size(1) != self.in_channels:
            raise ValueError("in_channels of block does not match in_channels of input")
        output_shape = (inputs.size(0), self.out_channels, self.image_shape[0], self.image_shape[1])
        return torch.reshape(inputs, output_shape)
    
def is_perfect_square(n: int):
    root = math.sqrt(n)
    return int(root + 0.5) ** 2 == n