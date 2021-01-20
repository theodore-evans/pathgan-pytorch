from typing import Optional
import math
from torch import Tensor

import torch.nn as nn
from torch.nn.modules.container import ModuleDict
from torch.nn.utils import spectral_norm

from .NoiseInput import NoiseInput
from .AdaIN import AdaIN

class Block(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 modules: Optional[ModuleDict],
                 noise_input: bool = False,
                 normalization: Optional[str] = None,
                 regularization: Optional[str] = None,
                 activation: Optional[nn.Module] = None,
                 ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #TODO: Spectral norm should be declared explicitly, some moudules use spec norm and orthogonal norm
        if modules is not None:
            for module_name, module in modules.items():  
                if regularization == 'spectral': 
                    self.add_module(module_name, spectral_norm(module))
                else:
                    self.add_module(module_name, module)
                
        if noise_input:
            self.noise_input = NoiseInput(out_channels)
            
        if normalization == 'conditional':
            #hack
            latent_dim = 100
            inter_dim = math.ceil((float(latent_dim) + float(out_channels)) / 2 )
            self.conditional_instance_normalization = AdaIN(out_channels, latent_dim, inter_dim)
        
        if activation is not None:
            self.activation = activation
    
    def forward(self, inputs : Tensor, **kwargs) -> Tensor: 
        net = inputs
        for module in self.children():
            net = module(net, **kwargs)
        return net