from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm

from modules.blocks.Block import Block
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock

from modules.types import regularization_t, initialization_t
class AttentionBlock(Block):
    def __init__(self,
                 channels: int,
                 regularization: regularization_t = None,
                 initialization: initialization_t = None,
                 **kwargs,
                 ) -> None:

        super().__init__(channels, channels, None)

        f_g_channels = channels // 8
        kernel_size = 1
        stride = 1
        
        conv_block_args = ({'regularization' : regularization,
                            'initialization' : initialization})

        f_conv_layer = nn.Conv2d(channels, f_g_channels, kernel_size, stride)
        g_conv_layer = nn.Conv2d(channels, f_g_channels, kernel_size, stride)
        h_conv_layer = nn.Conv2d(channels, channels, kernel_size, stride)
        
        self.attention_f = ConvolutionalBlock(f_conv_layer, **conv_block_args)
        self.attention_g = ConvolutionalBlock(g_conv_layer, **conv_block_args)
        self.attention_h = ConvolutionalBlock(h_conv_layer, **conv_block_args)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        batch_size, channels, height, width = inputs.shape
        f_flat = self.attention_f(inputs).view(
            (batch_size, channels//8, -1)).permute(0, 2, 1)
        g_flat = self.attention_g(inputs).view(
            (batch_size, channels//8, -1)).permute(0, 2, 1)
        h_flat = self.attention_h(inputs).view(
            (batch_size, channels, -1)).permute(0, 2, 1)

        s = torch.matmul(g_flat, f_flat.transpose(1,2))

        beta = F.softmax(s, dim=-1)

        o = torch.matmul(beta, h_flat)
        o = o.view((batch_size, height, width, channels)).permute(0, 3, 1, 2)

        y = self.gamma * o + inputs

        return y
