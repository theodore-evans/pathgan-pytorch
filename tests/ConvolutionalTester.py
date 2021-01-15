import unittest
import torch
import numpy as np

from models.generative.ConvolutionalBlock import ConvolutionalBlock
from models.generative.ConvolutionalScale import ConvolutionalScale


class ConvolutionalTester(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.rand((64, 3, 224, 224))

    def test_vanilla(self):
        conv = ConvolutionalBlock(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        out = conv(self.data)
        self.assertEqual(out.shape, (64, 6, 224, 224),
                         "Dimensions Should Match")

    def test_downscale(self):
        scale = ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=4,
                                   stride=2, padding=2, normalization=None, upscale=False)
        out = scale(self.data)
        self.assertEqual(out.shape, (64, 6, 112, 112),
                         "Dimensions Should Match")

    def test_upscale(self):
        scale = ConvolutionalScale(in_channels=3, out_channels=6, kernel_size=2,
                                   stride=2, padding=1, normalization=None, upscale=True, output_padding=1)
        out = scale(self.data)
        self.assertEqual(out.shape, (64, 6, 448, 448),
                         "Dimensions Should Match")

    def test_weights(self):
        in_channels, out_channels, kernel_size = 3, 6, 2
        scale = ConvolutionalScale(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=2, padding=1, normalization=None, upscale=True, output_padding=1)

        self.assertEqual(scale.conv_layer.weight.shape, (in_channels, out_channels, kernel_size + 1, kernel_size + 1), "Kernel Should be Extended")

    def test_padding_calculator(self):
        pads = ConvolutionalScale.calculate_padding_upscale(input_size=112,stride=2,kernel_size=3)
        self.assertEqual(pads, (1,1), "Paddings Should Match")
