from typing import Optional
from models.generative.ConvolutionalBlock import ConvolutionalBlock
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Block import Block

class AttentionBlock(Block):
    def __init__(self, 
                 channels: int,
                 regularization: Optional[str] = 'spectral'
                 ) -> None:
        
        self.gamma = nn.Parameter(0, requires_grad=True)
        f_g_channels = channels // 8

        kwargs = dict({'kernel_size' : 1, 'stride' : 1, 'padding' : 0, 'regularization' : regularization})
        
        self.attention_f = ConvolutionalBlock(in_channels=channels, out_channels=f_g_channels, **kwargs)
        self.attention_g = ConvolutionalBlock(in_channels=channels, out_channels=f_g_channels, **kwargs)
        self.attention_h = ConvolutionalBlock(in_channels=channels, out_channels=channels, **kwargs)

        super().__init__(channels, channels, None)
        
    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, channels, height, width = inputs.shape
        f_flat = self.attention_f(inputs).view((batch_size, channels//8, -1)).permute(0,2,1)
        g_flat = self.attention_f(inputs).view((batch_size, channels//8, -1)).permute(0,2,1)
        h_flat = self.attention_f(inputs).view((batch_size, channels, -1)).permute(0,2,1)

        s = torch.matmul(g_flat, f_flat.transpose())

        beta = F.softmax(s)

        o = torch.matmul(beta, h_flat)
        o = o.view((batch_size, height, width, channels)).permute(0,3,1,2)

        y = self.gamma * o + inputs

        return y


