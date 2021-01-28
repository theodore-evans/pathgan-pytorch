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
        
        pathgan_blocks = OrderedDict({
            ("dense_block_1", DenseBlock(8,8, activation=nn.LeakyReLU)) : 1,
            ("dense_block_2", DenseBlock(8,8, activation=nn.LeakyReLU)) : 2,
            
            ("res_block_1", ResidualBlock(1, DenseBlock(8,8, activation=nn.LeakyReLU))) : 3,
            ("upscale_block_1", ConvolutionalBlock(UpscaleConv2d(8,8,3), activation=nn.LeakyReLU)) : 4,
            
            ("res_block_2", ResidualBlock(1, DenseBlock(8,8, activation=nn.LeakyReLU))) : 5,
            ("upscale_block_2", ConvolutionalBlock(UpscaleConv2d(8,8,3), activation=nn.LeakyReLU)) : 6,
            
            ("res_block_3", ResidualBlock(1, DenseBlock(8,8, activation=nn.LeakyReLU))) : 7,
            ("attention_block_3", AttentionBlock(8)) : 8,
            ("upscale_block_3", ConvolutionalBlock(UpscaleConv2d(8,8,3), activation=nn.LeakyReLU)) : 9,
            
            ("res_block_4", ResidualBlock(1, DenseBlock(8,8, activation=nn.LeakyReLU))) : 10,
            ("upscale_block_4", ConvolutionalBlock(UpscaleConv2d(8,8,3), activation=nn.LeakyReLU)) : 11,
            
            ("res_block_5", ResidualBlock(1, DenseBlock(8,8, activation=nn.LeakyReLU))) : 12,
            ("upscale_block_5", ConvolutionalBlock(UpscaleConv2d(8,8,3), activation=nn.LeakyReLU)) : 13,
            
            ("sigmoid_block", ConvolutionalBlock(UpscaleConv2d(8,8,3), activation=nn.Sigmoid)) : 14
        })
        
        
        
        for name, module in pathgan_blocks:
            self.add_module(name, module)
