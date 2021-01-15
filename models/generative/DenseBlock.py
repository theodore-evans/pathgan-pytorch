from torch import Tensor
from torch.nn.modules.container import ModuleDict
from typing import Optional
import torch.nn as nn

from .Block import Block

class DenseBlock(Block):
    def __init__(self, 
                 in_channels : int, 
                 out_channels : int, 
                 noise_input : bool = False, 
                 normalization : str = 'conditional', 
                 regularization : str = None,
                 activation : Optional[nn.Module] = nn.LeakyReLU(0.2)):
        
        dense_layer = ModuleDict({'dense_layer' : nn.Linear(in_channels, out_channels)})
        
        super().__init__(in_channels, out_channels, dense_layer, noise_input, normalization, regularization, activation)

