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
        
        super().initialize_weights(initialization = lambda x: xavier_uniform_(x, gain))