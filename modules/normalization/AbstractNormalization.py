from abc import ABC

import torch.nn as nn
class AbstractNormalization(nn.Module, ABC):
    def __init__(self, channels, latent_dim, regularization):
        super().__init__()
        
    def forward(self, inputs, *args, **kwargs):
        pass