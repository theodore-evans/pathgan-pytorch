from copy import deepcopy
from typing import Callable, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.container import ModuleDict

from .Normalization.AbstractNormalization import AbstractNormalization
from .Initialization.AbstractInitializer import AbstractInitializer

class Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 module_dict: Optional[ModuleDict] = None,
                 latent_dim: Optional[int] = None,
                 regularization: Optional[Callable[[nn.Module], nn.Module]] = None,
                 noise_input: Optional[Callable[[int], nn.Module]] = None,
                 normalization: Optional[Callable[..., AbstractNormalization]] = None,
                 activation: Optional[nn.Module] = None,
                 initializer: Optional[Callable[[nn.Module], AbstractInitializer]] = None
                 ) -> None:
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if regularization is None:
            regularization = lambda x: x
        
        if module_dict is not None:
            for module_name, module in module_dict.items():
                self.add_module(module_name, regularization(module))
                        
        if noise_input is not None:
            self.noise_input = noise_input(out_channels)
        
        if normalization is not None:
            if latent_dim is not None:
                self.normalization = normalization(out_channels, latent_dim)
            else:
                self.normalization = normalization(out_channels)
            
        if activation is not None:
            self.activation = activation
        
        if initializer is not None:
            self.initializer = initializer(self)
            self.initializer.initialize_weights()
    
    def forward(self, inputs: Tensor, latent_input: Optional[Tensor] = None) -> Tensor: 
        net = inputs
        for module in self.children():
            net = module(net, latent_input) if isinstance(module, AbstractNormalization) else module(net)
        return net