import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = Dense(in_features=200, out_features=1024)
        self.adaIn1 = AdaIn()
        self.dense2 = 


class Dense(nn.Linear): 
    def __init__(self, in_features, out_features, bias=True, alpha=0.2, spectral=True, init='xavier', regularizer=None):
        super().__init__(in_features, out_features, bias=bias)
        self.alpha=alpha
        pass # implement spectral normalisation, xavier initialisation and regularisation
    
    def forward(self, x):
        x = F.leaky_relu(input=x, negative_slope=self.alpha)


class AdaIn(nn.)