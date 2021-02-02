from copy import deepcopy
from typing import Callable, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.container import ModuleDict

from modules.normalization.AbstractNormalization import AbstractNormalization

from modules.types import regularization_t, noise_input_t, normalization_t, activation_t, initialization_t

class Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: ModuleDict = None,
                 latent_dim: Optional[int] = None,
                 regularization: regularization_t = None,
                 noise_input: noise_input_t = None,
                 normalization: normalization_t = None,
                 activation: activation_t = None,
                 initialization: initialization_t = None
                 ) -> None:
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        if regularization is None:
            regularization = lambda x: x
            
        if layers is not None:
            for name, layer in layers.items():
                self.add_module(name, regularization(layer))
                        
        self._add_layer_epilogue(noise_input, normalization, regularization, activation)
        
        self.initialize(initialization)

    def _add_layer_epilogue(self, noise_input, normalization, regularization, activation):
        if noise_input is not None:
            self.noise_input = noise_input(self.out_channels)
            
        if normalization is not None:
            if self.latent_dim is not None:
                self.normalization = normalization(self.out_channels, self.latent_dim, regularization)
            else:
                self.normalization = normalization(self.out_channels, regularization)
            
        if activation is not None:
            self.activation = activation

    def initialize(self, initialization):
        if initialization is not None:
            initialization(self).initialize_weights()
    
    def forward(self, inputs: Tensor, latent_input: Optional[Tensor] = None) -> Tensor:
        net = inputs
        for module in self.children():
            net = module(net, latent_input) if isinstance(module, (Block, AbstractNormalization)) else module(net)
        return net