from typing import List, Optional
import math
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleDict
from torch.nn.utils import spectral_norm

from .NoiseInput import NoiseInput
from .AdaIN import AdaIN

class Block(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 modules: ModuleDict,
                 noise_input: bool = False,
                 normalization: Optional[str] = None,
                 regularization: Optional[str] = None,
                 activation: Optional[nn.Module] = None,
                 ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        for module_name, module in modules.items():  
            if regularization == 'spectral': 
                self.add_module(module_name, spectral_norm(module))
            else:
                self.add_module(module_name, module)
                
        if noise_input:
            self.add_module(f'noise_input', NoiseInput(out_channels))
            
        if normalization == 'conditional':
            #hack
            latent_dim = 100
            inter_dim = math.ceil(float(latent_dim) + float(out_channels) / 2 )
            self.add_module(f'conditional_instance_normalization', AdaIN(out_channels, latent_dim, inter_dim))
        
        if activation is not None:
            self.add_module(f'activation', activation)
    
    #TODO: how to pass latent variable in?
    def forward(self, input : Tensor) -> Tensor: 
        net = input
        for module in self.modules():
            net = module(net)
        return net