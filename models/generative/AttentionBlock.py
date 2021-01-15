from models.generative.ConvolutionalBlock import ConvolutionalBlock
from torch import Tensor
import torch
import torch.nn as nn
from .Block import Block
from torch.nn.modules.container import ModuleDict


class AttentionBlock(Block):
    def __init__(self, channels: int,) -> None:
        super().__init__()

        self.gamma = nn.Parameter(0)
        f_g_channels = channels // 8

        blocks = ModuleDict()
        blocks['attention_f'] = ConvolutionalBlock(
            in_channels=channels, out_channels=f_g_channels, kernel_size=1, stride=1, padding=0, init='xavier', regularization='spectral')
        blocks['attention_g'] = ConvolutionalBlock(
            in_channels=channels, out_channels=f_g_channels, kernel_size=1, stride=1, padding=0, init='xavier', regularization='spectral')
        blocks['attention_h'] = ConvolutionalBlock(
            in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, init='xavier', regularization='spectral')
        
    #TODO: implement

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, channels, height, width = inputs.shape
        f_flat = self.attention_f(inputs).view((batch_size, channels//8, -1)).permute(0,2,1)
        g_flat = self.attention_f(inputs).view((batch_size, channels//8, -1)).permute(0,2,1)
        h_flat = self.attention_f(inputs).view((batch_size, channels, -1)).permute(0,2,1)

        s = torch.matmul(g_flat, f_flat.transpose())

        beta = torch.softmax(s)

        o = torch.matmul(beta, h_flat)
        o = o.view((batch_size, height, width, channels)).permute(0,3,1,2)

        y = self.gamma * o + inputs

        return y


