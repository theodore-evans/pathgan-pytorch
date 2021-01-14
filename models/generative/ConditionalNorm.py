from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.instancenorm import InstanceNorm2d

from DenseBlock import DenseBlock

class ConditionalNorm(nn.Module):
    def __init__(self, 
                 channels : int,
                 latent_dim : int, 
                 inter_dim : int, 
                 dense_activation : Optional[nn.Module] = nn.ReLU(), 
                 gamma_activation : Optional[nn.Module] = nn.ReLU(), 
                 beta_activation : Optional[nn.Module] = None
                 ) -> None:
        super().__init__()
        
        self.instance_norm = InstanceNorm2d(channels)
        self.dense_layer = DenseBlock(latent_dim, inter_dim, activation=dense_activation)
        self.gamma_layer = DenseBlock(inter_dim, channels, activation=gamma_activation)
        self.beta_layer = DenseBlock(inter_dim, channels, activation=beta_activation)

    def forward(self, input, latent_input) -> Tensor:
        normalized_input = self.instance_norm(input)
        #https://zhangruochi.com/Components-of-StyleGAN/2020/10/13/
        
        intemediate_result = self.dense_layer(latent_input)
        style_scale = self.gamma_layer(intemediate_result)[:, :, None, None]
        style_shift = self.beta_layer(intemediate_result)[:, :, None, None]
        
        transformed_input = style_scale * normalized_input + style_shift
        return transformed_input
    
