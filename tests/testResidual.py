import unittest
import torch
from torch import nn

from modules.blocks.ConvolutionalBlock import ConvolutionalBlock
from modules.blocks.ResidualBlock import ResidualBlock

class TestResidual(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.rand((64, 3, 224, 224))

    def test_vanilla_deep_copy(self):
        conv = ConvolutionalBlock(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1), normalization =None)
        res = ResidualBlock(2, conv)
        res.part_1.conv_layer.weight[0,0,1,1] += 3.5
        out = res(self.data)
        self.assertEqual(out.shape, (64, 3, 224, 224),"Dimensions Should Match")
        assert res.part_1.conv_layer.weight[0,0,1,1] != res.part_2.conv_layer.weight[0,0,1,1] 
        
    # TODO: write more init tests to make sure residual block is working correctly
