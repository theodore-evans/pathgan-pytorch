from torch import Tensor
import copy

from modules.blocks.Block import Block

class ResidualBlock(Block):
    def __init__(self,
                 num_blocks : int,
                 block_template : Block
                 ) -> None:
        
        super().__init__(block_template.in_channels, block_template.out_channels)
        
        for index in range(num_blocks):
            self.add_module(f'part_{index + 1}', copy.deepcopy(block_template))
        
    def forward(self, inputs, **kwargs) -> Tensor:
        net = super().forward(inputs, **kwargs)
        return inputs + net