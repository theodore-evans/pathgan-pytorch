import torch.nn as nn
from torch.nn.init import calculate_gain, xavier_uniform_

from modules.initialization.AbstractInitialization import AbstractInitialization

class XavierInitialization(AbstractInitialization):    
    def __init__(self, module: nn.Module):
        self.module = module
    
    def initialize_weights(self) -> None:
        gain_param_lookup = dict({
                nn.Tanh : dict({'nonlinearity' : 'tanh'}),
                nn.ReLU : dict({'nonlinearity' :'relu'}),
                nn.LeakyReLU : dict({'nonlinearity' : 'leaky_relu', 'param' : 0.2})
            })
        
        gain = 1
        
        if hasattr(self.module, 'activation'):
            activation = self.module.activation
            if activation is not None and type(activation) in gain_param_lookup:
                gain = calculate_gain(**gain_param_lookup[type(activation)])
        
        def has_parameter(param: str) -> bool:
            m = self.module
            return hasattr(m, param) and isinstance(getattr(m, param), nn.Parameter)
        
        if has_parameter('weight'):
            xavier_uniform_(self.module.weight, gain) #type: ignore
        
        if has_parameter('bias'):
            nn.init.constant_(self.module.bias, 0.) #type: ignore
