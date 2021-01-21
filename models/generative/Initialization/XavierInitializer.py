from typing import Union
import torch.nn as nn
from torch.nn.init import calculate_gain, xavier_uniform_
from torch.nn.modules.activation import LeakyReLU, ReLU


from .AbstractInitializer import AbstractInitializer

class XavierInitializer(AbstractInitializer):        
    def __init__(self, module: nn.Module):
        self.module = module
    
    def initialize_weights(self) -> None:
        nonlinearity_lookup = dict({
                nn.Tanh : 'tanh',
                nn.ReLU : 'relu',
                nn.LeakyReLU : 'leaky_relu'
            })
        
        if self.module.activation is not None and type(self.module.activation) in nonlinearity_lookup:
            gain = calculate_gain(nonlinearity_lookup[type(self.module.activation)])
        else:
            gain = 1
        
        super().initialize_weights(initialization = lambda x: xavier_uniform_(x, gain))