import torch.nn as nn
from torch.nn.init import calculate_gain, xavier_uniform_

from modules.initialization.Initialization import Initialization

class XavierInitialization(Initialization):
    def __init__(self, module: nn.Module):
        self.module = module
        self.weight_init = lambda weight : xavier_uniform_(weight, self._gain())
        self.bias_init = lambda bias : nn.init.constant_(bias, 0.)

    def _gain(self):
        gain_param_lookup = dict({
                nn.Tanh : dict({'nonlinearity' : 'tanh'}),
                nn.ReLU : dict({'nonlinearity' :'relu'}),
                nn.LeakyReLU : dict({'nonlinearity' : 'leaky_relu', 'param' : 0.2})
            })

        if hasattr(self.module, 'activation'):
            activation = self.module.activation
            if activation is not None and type(activation) in gain_param_lookup:
                return calculate_gain(**gain_param_lookup[type(activation)])
        
        return 1