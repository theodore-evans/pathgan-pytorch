import torch.nn as nn
from torch.tensor import Tensor
from torch.nn.utils.spectral_norm import spectral_norm
from modules.initialization.XavierInitializer import XavierInitializer
from torch.nn.modules.activation import ReLU
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.ResidualBlock import ResidualBlock

class Mapping(nn.Module):
    def __init__(
        self,
        z_dim: int = 200,
        w_dim: int = 200,
        layers: int = 4) -> None:
        super().__init__()

        default_kwargs = {
            'normalization': None,
            'regularization': spectral_norm,
            'noise_input': None,
            'activation': ReLU(),
            'initializer': XavierInitializer
        }

        for layer in range(layers):
            inner_res = DenseBlock(in_channels=z_dim, out_channels=z_dim, **default_kwargs)
            res_block = ResidualBlock(num_blocks=2, block_template= inner_res, **default_kwargs)
            self.add_module(f'ResDense_{layer}',res_block)
           	
        w_map = DenseBlock(in_channels=z_dim,out_channels=w_dim, **default_kwargs)
        self.add_module('w_map', w_map)

    def forward(self, data: Tensor, **kwargs):
        for m in self.children():
            data = m(data, *kwargs)
        return data  
