import unittest
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from models.generative.ConvolutionalBlock import ConvolutionalBlock, UpscaleBlock
from models.generative.ConvolutionalScale import ConvolutionalScale

class TestConvolutional(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.rand((64, 3, 224, 224))

    def test_vanilla(self):
        conv = ConvolutionalBlock(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1))
        out = conv(self.data)
        self.assertEqual(out.shape, (64, 6, 224, 224),
                         "Dimensions Should Match")

    def test_downscale(self):
        scale = ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=4,
                                   stride=2, padding=2, normalization=None, upscale=False)
        out = scale(self.data)
        self.assertEqual(out.shape, (64, 6, 112, 112),
                         "Dimensions Should Match")
    
    def test_upscale_requires_correct_arguments(self):
        self.assertRaises(Exception, ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=2,
                                   stride=2, padding=1, normalization=None, upscale=True),
                          "Should not be able to instantiate an upscale layer with bad parameters")
        
    def test_upscale(self):
        scale = ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=2,
                                   stride=2, padding=1, normalization=None, upscale=True)
        out = scale(self.data)
        self.assertEqual(out.shape, (64, 6, 448, 448),
                         "Dimensions Should Match")

    def test_upscale_block(self):
        scale_block = UpscaleBlock(in_channels=3, out_channels=6, kernel_size=2)
        out = scale_block(self.data)
        self.assertEqual(out.shape, (64, 6, 448, 448),
                         "Dimensions Should Match")
    
    def test_weights(self):
        in_channels, out_channels, kernel_size = 3, 6, 2
        scale = ConvolutionalScale(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=2, padding=0, normalization=None, upscale=True, output_padding=1)

        self.assertEqual(scale.weight.shape, (in_channels, out_channels, kernel_size + 1, kernel_size + 1), "Kernel Should be Extended")

        