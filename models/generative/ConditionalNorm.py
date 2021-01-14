from typing import Optional
from torch import Tensor
import torch.nn as nn

class ConditionalNorm(nn.Module):
    def __init__(self, 
                 in_channels : int,
                 latent_dim : int, 
                 inter_dim : int, 
                 dense_activation : Optional[nn.Module] = nn.ReLU(), 
                 gamma_activation : Optional[nn.Module] = nn.ReLU(), 
                 beta_activation : Optional[nn.Module] = None
                 ) -> None:
        super().__init__()
        
        out_channels = in_channels
        
        self.add_module('dense_layer', nn.Linear(latent_dim, inter_dim))
        if dense_activation is not None:
            self.add_module('dense_layer_activation', dense_activation)
        
        self.add_module('dense_layer_gamma', nn.Linear(inter_dim, out_channels))
        if gamma_activation is not None:
            self.add_module('dense_layer_gamma_activation', gamma_activation)
        
        self.add_module('dense_layer_beta', nn.Linear(inter_dim, out_channels))
        if beta_activation is not None:
            self.add_module('dense_layer_beta_activation', beta_activation)
    
    def forward(self, input) -> Tensor:
        return input #TODO: implement