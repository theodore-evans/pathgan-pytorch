from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import InstanceNorm2d
from torch.nn.modules.normalization import LayerNorm

from modules.normalization.AbstractNormalization import AbstractNormalization
from modules.types import regularization_t

class AdaptiveInstanceNormalization(AbstractNormalization):
    def __init__(self,
                 channels : int,
                 latent_dim : int,
                 regularization: regularization_t = None,
                 intermediate_layer: bool = True,
                 dense_activation : Optional[nn.Module] = nn.ReLU(),
                 gamma_activation : Optional[nn.Module] = nn.ReLU(),
                 beta_activation : Optional[nn.Module] = None
                 ) -> None:
        
        super().__init__(channels, latent_dim, regularization)
        
        self.latent_dim = latent_dim
        self.channels = channels
        
        if regularization is None:
            regularization = lambda x: x
        
        if intermediate_layer:
            intermediate_channels = (latent_dim + channels) // 2
            self.dense_layer = regularization(nn.Linear(latent_dim, intermediate_channels))
            in_channels = intermediate_channels
            self.dense_activation = dense_activation
        else:
            self.dense_layer = None
            in_channels = latent_dim
            
        out_channels = channels
        
        self.gamma_layer = regularization(nn.Linear(in_channels, out_channels))
        self.gamma_activation = gamma_activation
        
        self.beta_layer = regularization(nn.Linear(in_channels, out_channels))
        self.beta_activation = beta_activation

    def forward(self,
                inputs: Tensor,
                latent_input: Optional[Tensor]
                ) -> Tensor:
        
        if latent_input is None:
            latent_input = torch.zeros(inputs.shape[0], self.latent_dim)
        
        valid_input_size = inputs.dim() in (2,4)
        
        if not valid_input_size:
            raise ValueError("Expecting input dimension of either 4 or 2")
        
        input_is_image = inputs.dim() == 4
        norm_2d = InstanceNorm2d(self.channels, affine=False)
        norm_1d = LayerNorm(inputs.size(), elementwise_affine=False)
        normalize = norm_2d if input_is_image else norm_1d
        
        if self.dense_layer is not None:
            intermediate_result = self.dense_layer(latent_input)
            if self.dense_activation is not None:
                intermediate_result = self.dense_activation(intermediate_result)
        else: intermediate_result = latent_input
        
        gamma = self.gamma_layer(intermediate_result)
        if self.gamma_activation is not None:
            gamma = self.gamma_activation(gamma)
            
        beta = self.beta_layer(intermediate_result)
        if self.beta_activation is not None:
            beta = self.beta_activation(beta)
        
        style_shift_transform = beta[:, :, None , None] if input_is_image else beta
        style_scale_transform = gamma[:, :, None, None] if input_is_image else gamma
        
        transformed_input = style_scale_transform * normalize(inputs) + style_shift_transform
        return transformed_input