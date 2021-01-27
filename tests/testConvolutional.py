
import copy
import unittest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm 
from torch.nn.utils.spectral_norm import SpectralNorm
from torch.nn.init import constant_

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
        
        has_fused_scale_hook, _, _ = self.has_hook(scale, FusedScale, '_forward_pre_hooks')
        self.assertTrue(has_fused_scale_hook, "Fused scale transform should be registered as pre-forward hook")
        
    def test_spectral_norm_hook(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        scale = spectral_norm(UpscaleConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, fused_scale=False))
        has_spectral_norm_hook, _, _ = self.has_hook(scale, SpectralNorm, '_forward_pre_hooks')
        self.assertTrue(has_spectral_norm_hook, 
                        "Layer should have spectral norm pre-forward hook")
        
    def test_spectral_norm_with_fused_scale_hook(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        fn = lambda x: spectral_norm(x, name = 'filter', dim = 1)
        scale = fn(UpscaleConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, fused_scale=True))
        _, spectral_norm_hook_id, spectral_parameter = self.has_hook(scale, SpectralNorm, '_forward_pre_hooks')
                
        _, fused_scale_hook_id, fused_scale_parameter = self.has_hook(scale, FusedScale, '_forward_pre_hooks')
        self.assertGreater(spectral_norm_hook_id, fused_scale_hook_id,
                           "Spectral norm hook should follow fused scale hook")
        self.assertEqual(spectral_parameter, fused_scale_parameter,
                         "Spectral norm hook should act on same parameter as fused scale")
        
    def test_fused_scale_with_spectral_norm(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        fn = lambda x: spectral_norm(x, name = 'filter', dim = 0)
        scale = fn(DownscaleConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, fused_scale=True))
        nn.init.constant_(scale.weight, 1)
        dummy = copy.deepcopy(scale)
        data = torch.ones(self.input_shape)
        
        FusedScale(name='filter')(dummy, None)
        SpectralNorm(name='filter', dim=0)(dummy, None)
        dummy_out = F.conv2d(data, dummy.filter, dummy.bias, stride=2, padding=dummy.padding)
        scale_out = scale(data)
        self.assertTrue(torch.equal(dummy_out, scale_out), 
                               "fused scale and spectral norm should be applied in ConvolutionScale forward pass")
        
        
    def has_hook(self, module: nn.Module, hook_class: type, hook_type: str = '_forward_pre_hooks'):
        hooks = getattr(module, hook_type)
        for k, hook in hooks.items():
            if isinstance(hook, hook_class):
                return True, k, hook.name
        return False, None, None
            