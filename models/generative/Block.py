from typing import Callable, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.container import ModuleDict, ModuleList

class Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modules: Optional[ModuleDict] = None,
                 regularization: Optional[Callable[[nn.Module], nn.Module]] = lambda x: x,
                 noise_input: Optional[Callable[[int], nn.Module]] = None,
                 normalization: Optional[Callable[[int], nn.Module]] = None,  
                 activation: Optional[nn.Module] = None,
                 initialization: Optional[Callable] = None
                 ) -> None:
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #TODO: Spectral norm should be declared explicitly, some moudules use spec norm and orthogonal norm
        if modules is not None:
            for module_name, module in modules.items():
                self.add_module(module_name, regularization(module))
                        
        if noise_input is not None:
            self.noise_input = noise_input(out_channels)
        
        if normalization is not None:
            self.normalization = normalization(out_channels)
            
        if activation is not None:
            self.activation = activation
        
        if initialization is not None:
            initialization(self)
    
    def forward(self, inputs : Tensor, **kwargs) -> Tensor: 
        net = inputs
        for module in self.children():
            net = module(net, **kwargs)
        return net