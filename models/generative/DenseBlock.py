from models.generative.Block import Block
from typing import Optional
import torch.nn as nn

from ConditionalNorm import ConditionalNorm

class DenseBlock(Block):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 noise_input : bool = False, 
                 normalization : str = 'conditional', 
                 regularization : str = None,
                 activation : Optional[nn.Module] = nn.LeakyReLU(0.2)):
                
        dense_layer = ('dense_layer', nn.Linear(in_features, out_features))
        super().__init__([dense_layer], noise_input, normalization, regularization, activation)
