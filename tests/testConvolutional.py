import unittest
import torch
from torch import nn
from torch.nn.utils import spectral_norm 
from torch.nn.utils.spectral_norm import SpectralNorm

from models.generative.ConvolutionalBlock import ConvolutionalBlock
from models.generative.ConvolutionalScale import DownscaleConv2d, FusedScale, UpscaleConv2d

class TestConvolutional(unittest.TestCase):
    def setUp(self) -> None:
        self.input_shape = (64, 3, 224, 224)
        
        self.image_shape = (self.input_shape[2], self.input_shape[3])
        self.doubled_image_shape = tuple(size * 2 for size in self.image_shape)
        self.halved_image_shape = tuple( -(-size // 2) for size in self.image_shape) # ceil(size/2)
        
        self.data = torch.rand(self.input_shape)

    def test_vanilla(self):
        conv = ConvolutionalBlock(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3))
        out = conv(self.data)
        self.assertEqual(out.shape, (64, 6, *self.image_shape),
                         "Dimensions Should Match")

    def test_downscale(self):
        scale = DownscaleConv2d(in_channels=3, out_channels=6, kernel_size=5)
        out = scale(self.data)
        
        self.assertEqual(out.shape, (64, 6, *self.halved_image_shape),
                         "Dimensions Should Match")
    
    def test_upscale_requires_correct_arguments(self):
        with self.assertRaises(ValueError): 
            UpscaleConv2d(in_channels=3, out_channels=6, kernel_size=2)
        
    def test_upscale(self):
        scale = UpscaleConv2d(in_channels=3, out_channels=6, kernel_size=3)
        out = scale(self.data)
        
        self.assertEqual(out.shape, (64, 6, *self.doubled_image_shape),
                         "Dimensions Should Match")

    def test_upscale_block(self):
        scale_block = ConvolutionalBlock(UpscaleConv2d(in_channels=3, out_channels=6, kernel_size=3))
        out = scale_block(self.data)
        self.assertEqual(out.shape, (64, 6, *self.doubled_image_shape),
                         "Dimensions Should Match")
    
    def test_upscale_kernel_size(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        scale = UpscaleConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        self.assertEqual(scale.weight.shape, (in_channels, out_channels, kernel_size - 1, kernel_size - 1), 
                         "kernel should be reduced in weights")
        scale(self.data)
        self.assertEqual(scale.filter.shape, (3, 6, kernel_size, kernel_size), 
                         "kernel in forward pass should match constructor argument")
    
    def test_fused_scale_hook(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        scale = UpscaleConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, fused_scale=True)
        
        has_fused_scale_hook, _ = self.has_hook(scale, FusedScale, '_forward_pre_hooks')
        self.assertTrue(has_fused_scale_hook, "Fused scale transform should be registered as pre-forward hook")
        self.assertEqual(type(scale._forward_pre_hooks.popitem(last=False)[1]), FusedScale,
                         "Fused scale hook should be first pre-forward hook")
        
        
    def test_spectral_norm(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        scale = spectral_norm(UpscaleConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))
        has_spectral_norm_hook, spectral_norm_hook_id = self.has_hook(scale, SpectralNorm, '_forward_pre_hooks')
        self.assertTrue(has_spectral_norm_hook, "Layer should have spectral norm pre-forward hook")
        
        has_fused_scale_hook, fused_scale_hook_id = self.has_hook(scale, FusedScale, '_forward_pre_hooks')
        if has_fused_scale_hook:
            self.assertGreater(spectral_norm_hook_id, fused_scale_hook_id, 
                               "Spectral norm hook should follow fused scale hook")
        
    def has_hook(self, module: nn.Module, hook_class: type, hook_type: str = '_forward_pre_hooks'):
        hooks = getattr(module, hook_type)
        for k, hook in hooks.items():
            if isinstance(hook, hook_class):
                return True, k
        return False, None
            