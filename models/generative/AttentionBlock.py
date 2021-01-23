from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Block import Block
from .ConvolutionalBlock import ConvolutionalBlock

class AttentionBlock(Block):
    def __init__(self,
                 channels: int,
                 regularization: Optional[str] = 'spectral',
                 **kwargs,
                 ) -> None:
        f_g_channels = channels // 8

        super().__init__(channels, channels, None)

        conv_kwargs = dict({'kernel_size': 1, 'stride': 1})
        kwargs = ({'regularization': regularization})

        self.attention_f = ConvolutionalBlock(nn.Conv2d(
            in_channels=channels, out_channels=f_g_channels, **conv_kwargs), **kwargs)
        self.attention_g = ConvolutionalBlock(nn.Conv2d(
            in_channels=channels, out_channels=f_g_channels, **conv_kwargs), **kwargs)
        self.attention_h = ConvolutionalBlock(nn.Conv2d(
            in_channels=channels, out_channels=channels, **conv_kwargs), **kwargs)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
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
