from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.instancenorm import InstanceNorm2d

class AdaIN(nn.Module):
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
        
        def layer_with_activation(layer : nn.Module, activation : Optional[nn.Module]):
            return layer if activation is None else nn.Sequential(layer, activation)
        
        self.dense_layer = layer_with_activation(nn.Linear(latent_dim, inter_dim), dense_activation)
        self.gamma_layer = layer_with_activation(nn.Linear(inter_dim, channels), gamma_activation)
        self.beta_layer = layer_with_activation(nn.Linear(inter_dim, channels), beta_activation)

    def forward(self, input, latent_input) -> Tensor:
        #https://zhangruochi.com/Components-of-StyleGAN/2020/10/13/
        normalized_input = self.instance_norm(input)
        
        intermediate_result = self.dense_layer(latent_input)
        
        gamma = self.gamma_layer(intermediate_result)[:, :, None, None]
        beta = self.beta_layer(intermediate_result)[:, :, None, None]
        
        transformed_input = gamma * normalized_input + beta
        return transformed_input
    
class ConditionalNormTests:
    def run_tests(self) -> None:
        w_channels = 50
        image_channels = 20
        image_size = 30
        n_test = 10
        adain = AdaIN(image_channels, w_channels)
        test_w = torch.randn(n_test, w_channels)
        assert adain.style_scale_transform(test_w).shape == adain.style_shift_transform(test_w).shape
        assert adain.style_scale_transform(test_w).shape[-1] == image_channels
        assert tuple(adain(torch.randn(n_test, image_channels, image_size, image_size), test_w).shape) == (n_test, image_channels, image_size, image_size)

        w_channels = 3
        image_channels = 2
        image_size = 3
        n_test = 1
        adain = AdaIN(image_channels, w_channels)

        adain.style_scale_transform.weight.data = torch.ones_like(adain.style_scale_transform.weight.data) / 4
        adain.style_scale_transform.bias.data = torch.zeros_like(adain.style_scale_transform.bias.data)
        adain.style_shift_transform.weight.data = torch.ones_like(adain.style_shift_transform.weight.data) / 5
        adain.style_shift_transform.bias.data = torch.zeros_like(adain.style_shift_transform.bias.data)
        test_input = torch.ones(n_test, image_channels, image_size, image_size)
        test_input[:, :, 0] = 0
        test_w = torch.ones(n_test, w_channels)
        test_output = adain(test_input, test_w)
        assert(torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))) < 1e-4)
        assert(torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))) < 1e-4)
        print("Success!")