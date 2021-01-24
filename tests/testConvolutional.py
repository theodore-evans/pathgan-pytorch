import unittest
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from models.generative.ConvolutionalBlock import ConvolutionalBlock, UpscaleBlock
from models.generative.ConvolutionalScale import ConvolutionalScale

class TestConvolutional(unittest.TestCase):
    def setUp(self) -> None:
        self.input_shape = (64, 3, 224, 224)
        
        self.image_shape = (self.input_shape[2], self.input_shape[3])
        self.doubled_image_shape = tuple(size * 2 for size in self.image_shape)
        self.halved_image_shape = tuple( -(-size // 2) for size in self.image_shape) # ceil(size/2)
        
        self.data = torch.rand(self.input_shape)

    def test_vanilla(self):
        conv = ConvolutionalBlock(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1))
        out = conv(self.data)
        self.assertEqual(out.shape, (64, 6, *self.image_shape),
                         "Dimensions Should Match")

    def test_downscale(self):
        scale = ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=5,
                                   stride=2, padding=2, normalization=None, upscale=False)
        out = scale(self.data)
        
        self.assertEqual(out.shape, (64, 6, *self.halved_image_shape),
                         "Dimensions Should Match")
    
    def test_upscale_requires_correct_arguments(self):
        with self.assertRaises(ValueError) : ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=2,
                                   stride=2, padding=1, normalization=None, upscale=True)
        
    def test_upscale(self):
        scale = ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=3,
                                   stride=2, padding=1, normalization=None, upscale=True)
        out = scale(self.data)
        
        self.assertEqual(out.shape, (64, 6, *self.doubled_image_shape),
                         "Dimensions Should Match")

    def test_upscale_block(self):
        scale_block = UpscaleBlock(in_channels=3, out_channels=6, kernel_size=3)
        out = scale_block(self.data)
        self.assertEqual(out.shape, (64, 6, *self.doubled_image_shape),
                         "Dimensions Should Match")
    
    def test_upscale_kernel_size(self):
        in_channels, out_channels, kernel_size = 3, 6, 3
        scale = ConvolutionalScale(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=2, padding=0, normalization=None, upscale=True, output_padding=1)

        self.assertEqual(scale.weight.shape, (in_channels, out_channels, kernel_size - 1, kernel_size - 1), "kernel should be reduced in weights")
        self.assertEqual(scale.W_.shape, (3, 6, kernel_size, kernel_size), "kernel in forward pass should match constructor argument")
        