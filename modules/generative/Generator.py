from collections import OrderedDict
from models.generative.ConvolutionalScale import UpscaleConv2d
from models.generative.AttentionBlock import AttentionBlock
from models.generative.ConvolutionalBlock import ConvolutionalBlock
from .Block import Block
import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import Tensor

from .ResidualBlock import ResidualBlock
from .DenseBlock import DenseBlock

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class Generator(Model):
    def __init__(self) -> None:
        super().__init__()

        dense = DenseBlock(8,8)
        residual = ResidualBlock(8, dense)
        upscale = ConvolutionalBlock(UpscaleConv2d(8,8,3))
        attention = AttentionBlock(8)
        
        pathgan_blocks = OrderedDict({
            ("dense_block_1", dense) : 1,
            ("dense_block_2", dense) : 2,
            
            ("res_block_1", residual) : 3,
            ("upscale_block_1", upscale) : 4,
            
            ("res_block_2", residual) : 5,
            ("upscale_block_2", upscale) : 6,
            
            ("res_block_3", residual) : 7,
            ("attention_block_3", attention) : 8,
            ("upscale_block_3", upscale) : 9,
            
            ("res_block_4", residual) : 10,
            ("upscale_block_4", upscale) : 11,
            
            ("res_block_5", residual) : 12,
            ("upscale_block_5", upscale) : 13,
            
            ("sigmoid_block", upscale) : 14
        })
        
        for name, module in pathgan_blocks:
            self.add_module(name, module)
