import unittest
import torch
from torch import nn

from modules.blocks.NoiseInput import NoiseInput

class TestNoiseInput(unittest.TestCase):
    def setUp(self) -> None:
        self.test_noise_channels = 3000
        self.test_noise_samples = 20
        self.fake_images = torch.randn(self.test_noise_samples, self.test_noise_channels, 10, 10)
        self.inject_noise = NoiseInput(self.test_noise_channels)
        nn.init.ones_(self.inject_noise.weight)

    def test_weight_is_correct_shape(self):
        self.assertEqual(type(self.inject_noise.weight), nn.parameter.Parameter)
        self.assertEqual(tuple(self.inject_noise.weight.shape), (1, self.test_noise_channels, 1, 1))
        self.inject_noise.weight = nn.Parameter(torch.ones_like(self.inject_noise.weight), requires_grad=True)
        
    def test_output_changed_by_noise(self):
        self.assertGreater(torch.abs((self.inject_noise(self.fake_images) - self.fake_images)).mean(), 0.1)
    
    def test_per_channel_change(self):
        self.assertGreater(torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(0)).mean(), 1e-4)
        self.assertLess(torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(1)).mean(), 1e-4)
        self.assertGreater(torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(2)).mean(), 1e-4)
        self.assertGreater(torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(3)).mean(), 1e-4)
        
    def test_per_channel_change_is_gaussian(self):
        per_channel_change = (self.inject_noise(self.fake_images) - self.fake_images).mean(1).std()
        self.assertLess(per_channel_change > 0.9 and per_channel_change, 1.1)
    
    def test_weights_are_being_used(self):
        self.inject_noise.weight = nn.Parameter(torch.zeros_like(self.inject_noise.weight), requires_grad=True)
        self.assertLess(torch.abs((self.inject_noise(self.fake_images) - self.fake_images)).mean(), 1e-4)
        self.assertEqual(len(self.inject_noise.weight.shape), 4)