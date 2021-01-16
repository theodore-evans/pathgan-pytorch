import unittest
import torch
from torch import nn
from models.generative.NoiseInput import NoiseInput

class TestNoiseInput(unittest.TestCase):
    def setUp(self) -> None:
        self.test_noise_channels = 3000
        self.test_noise_samples = 20
        self.fake_images = torch.randn(self.test_noise_samples, self.test_noise_channels, 10, 10)
        self.inject_noise = NoiseInput(self.test_noise_channels)
        return super().setUp()
    
    def test_noise_input(self):
        assert torch.abs(self.inject_noise.weight.std() - 1) < 0.1
        assert torch.abs(self.inject_noise.weight.mean()) < 0.1
        assert type(self.inject_noise.weight) == torch.nn.parameter.Parameter
        
        assert tuple(self.inject_noise.weight.shape) == (1, self.test_noise_channels, 1, 1)
        self.inject_noise.weight = nn.Parameter(torch.ones_like(self.inject_noise.weight), requires_grad=True)
        # Check that something changed
        assert torch.abs((self.inject_noise(self.fake_images) - self.fake_images)).mean() > 0.1
        # Check that the change is per-channel
        assert torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(0)).mean() > 1e-4
        assert torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(1)).mean() < 1e-4
        assert torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(2)).mean() > 1e-4
        assert torch.abs((self.inject_noise(self.fake_images) - self.fake_images).std(3)).mean() > 1e-4
        # Check that the per-channel change is roughly normal
        per_channel_change = (self.inject_noise(self.fake_images) - self.fake_images).mean(1).std()
        assert per_channel_change > 0.9 and per_channel_change < 1.1
        # Make sure that the weights are being used at all
        self.inject_noise.weight = nn.Parameter(torch.zeros_like(self.inject_noise.weight), requires_grad=True)
        assert torch.abs((self.inject_noise(self.fake_images) - self.fake_images)).mean() < 1e-4
        assert len(self.inject_noise.weight.shape) == 4