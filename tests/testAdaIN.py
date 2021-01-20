import unittest
import torch
from models.generative.AdaIN import AdaIN

class TestAdaIN(unittest.TestCase):
    #unit tests from https://zhangruochi.com/Components-of-StyleGAN/2020/10/13/
    def test_layer_setup(self) -> None:
        w_channels = 50
        image_channels = 20
        image_size = 30
        n_test = 10
        adain = AdaIN(image_channels, w_channels, gamma_activation=None)
        test_w = torch.randn(n_test, w_channels)
        assert adain.gamma_layer(test_w).shape == adain.beta_layer(test_w).shape
        assert adain.gamma_layer(test_w).shape[-1] == image_channels
        assert tuple(adain(torch.randn(n_test, image_channels, image_size, image_size), test_w).shape) == (n_test, image_channels, image_size, image_size)
    
    def test_conditional_norm(self) -> None:
        w_channels = 3
        image_channels = 2
        image_size = 3
        n_test = 1
        adain = AdaIN(image_channels, w_channels, gamma_activation=None)

        adain.gamma_layer.weight.data = torch.ones_like(adain.gamma_layer.weight.data) / 4
        adain.gamma_layer.bias.data = torch.zeros_like(adain.gamma_layer.bias.data)
        adain.beta_layer.weight.data = torch.ones_like(adain.beta_layer.weight.data) / 5
        adain.beta_layer.bias.data = torch.zeros_like(adain.beta_layer.bias.data)
        test_input = torch.ones(n_test, image_channels, image_size, image_size)
        test_input[:, :, 0] = 0
        test_w = torch.ones(n_test, w_channels)
        test_output = adain(test_input, test_w)
        assert(torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))) < 1e-4)
        assert(torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))) < 1e-4)