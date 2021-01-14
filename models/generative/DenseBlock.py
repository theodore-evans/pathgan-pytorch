from typing import Optional
import torch.nn as nn

from ConditionalNorm import ConditionalNorm

class DenseBlock(nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 normalization : str = 'conditional', 
                 activation : Optional[nn.Module] = nn.LeakyReLU(0.2)):
        
        super().__init__()
        self.add_module('dense_layer', nn.Linear(in_features, out_features))
        
        if normalization == 'conditional':
            self.add_module('conditional_instance_normalization', ConditionalNorm())
            
        if activation is not None: 
            self.add_module('activation', activation)