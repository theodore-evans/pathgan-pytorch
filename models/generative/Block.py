from typing import List, Optional, Tuple, Union
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from Placeholder import Placeholder
from ConditionalNorm import ConditionalNorm

class Block(nn.Module):
    def __init__(self,
                 layers: List[Tuple[str, nn.Module]],
                 noise_input: bool = False,
                 normalization: Optional[str] = None,
                 regularization: Optional[str] = None,
                 activation: Optional[nn.Module] = None
                 ) -> None:
        
        super().__init__()
        
        for layer_name, layer in layers:  
            if regularization == 'spectral':
                self.add_module(layer_name, spectral_norm(layer, n_power_iterations=1))
            else:
                self.add_module(layer_name, layer)

        if noise_input:
            self.add_module(f'noise_input', Placeholder())
            
        if normalization == 'conditional': 
            self.add_module(f'conditional_instance_normalization', Placeholder())
        
        if activation is not None:
            self.add_module(f'activation', activation)