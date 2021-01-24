from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.modules.instancenorm import InstanceNorm2d

from .AbstractNormalization import AbstractNormalization

class AdaptiveInstanceNormalization(AbstractNormalization, nn.Module):
    def __init__(self, 
                 channels : int,
                 latent_dim : int, 
                 intermediate_channels : Optional[int] = None, 
                 dense_activation : Optional[nn.Module] = nn.ReLU(), 
                 gamma_activation : Optional[nn.Module] = nn.ReLU(), 
                 beta_activation : Optional[nn.Module] = None
                 ) -> None:
        
        nn.Module.__init__(self)
        
        self.latent_dim = latent_dim

        self.instance_norm = InstanceNorm2d(channels)
        
        if intermediate_channels is not None:
            self.dense_layer = nn.Linear(latent_dim, intermediate_channels)
            in_channels = intermediate_channels
            self.dense_activation = dense_activation if dense_activation is not None else lambda x:x
        else: 
            self.dense_layer = None
            in_channels = latent_dim
            
        out_channels = channels
        
        self.gamma_layer = nn.Linear(in_channels, out_channels)
        self.gamma_activation = gamma_activation if gamma_activation is not None else lambda x:x
        
        self.beta_layer = nn.Linear(in_channels, out_channels)
        self.beta_activation = beta_activation if beta_activation is not None else lambda x:x

    def forward(self, 
                input_tensor: Tensor,
                latent_input: Optional[Tensor]
                ) -> Tensor:
        
        if latent_input is None:
            batch_size = input_tensor.shape[0]
            latent_input = torch.zeros(batch_size, self.latent_dim)
            
        normalized_input = self.instance_norm(input_tensor)
        
        if self.dense_layer is not None:
            intermediate_result = self.dense_activation(self.dense_layer(latent_input))
        else: intermediate_result = latent_input
        
        gamma = self.gamma_activation(self.gamma_layer(intermediate_result))
        beta = self.beta_activation(self.beta_layer(intermediate_result))
        
        style_shift_transform = beta[:, : , None , None]
        style_scale_transform = gamma[:, :, None, None]
        
        transformed_input = style_scale_transform * normalized_input + style_shift_transform
        return transformed_input