from torch.nn.modules.container import ModuleDict
import torch.nn as nn

from .Block import Block
class DenseBlock(Block):
    def __init__(self, 
                 in_channels : int, 
                 out_channels : int, 
                 **kwargs) -> None:
        
        dense_layer = ModuleDict({'dense_layer' : nn.Linear(in_channels, out_channels)})
        
        super().__init__(in_channels, out_channels, dense_layer, **kwargs)