import unittest
import torch
import torch.nn as nn
from modules.normalization.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization

class TestAdaIN(unittest.TestCase):
    def setUp(self) -> None:
        self.w_channels = 3
        self.image_channels = 2
        image_size = 3
        n_test = 1
        self.input_shape = (n_test, self.image_channels, image_size, image_size)
        self.test_input = torch.ones(self.input_shape)
        self.test_w = torch.ones(n_test, self.w_channels)
        self.adain = AdaptiveInstanceNormalization(self.image_channels, self.w_channels, intermediate_layer=False)

        nn.init.constant_(self.adain.gamma_layer.weight, 0.25) #type: ignore
        nn.init.constant_(self.adain.beta_layer.weight, 0.2) #type: ignore
        
        for bias in (self.adain.gamma_layer.bias, self.adain.beta_layer.bias):
            nn.init.zeros_(bias)
            
        if self.adain.dense_layer is not None:
            nn.init.constant_(self.adain.dense_layer.weight, 0.2) #type: ignore
            nn.init.zeros_(self.adain.dense_layer.bias) #type: ignore

        self.test_w = torch.ones(n_test, self.w_channels)
        
    def test_dense_layer_shape_correct(self):
        self.assertEqual(self.adain.gamma_layer(self.test_w).shape, self.adain.beta_layer(self.test_w).shape)
        self.assertEqual(self.adain.gamma_layer(self.test_w).shape[-1], self.image_channels)
        self.assertEqual(tuple(self.adain(self.test_input, self.test_w).shape), self.input_shape)
    
    def test_conditional_norm_is_calculated_correctly(self) -> None:
        self.test_input[:, :, 0] = 0
        test_output = self.adain(self.test_input, self.test_w)
        self.assertLess(torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))), 1e-4) # type: ignore
        self.assertLess(torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))), 1e-4) # type: ignore
        # from https://zhangruochi.com/Components-of-StyleGAN/2020/10/13/ (deeplearning.ai course material)
        
    def test_works_for_dense_layers(self) -> None:
        dense_input = torch.rand(10, 200)
        latent_input = torch.zeros(10, 200)
        adain = AdaptiveInstanceNormalization(200, 200)
        out = adain(dense_input, latent_input)
        self.assertEqual(out.shape, dense_input.shape)