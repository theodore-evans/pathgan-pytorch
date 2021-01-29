from typing import Tuple, Union
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

from torch.nn.modules.conv import Conv2d

from modules.blocks.ResidualBlock import ResidualBlock
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.AttentionBlock import AttentionBlock
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock, UpscaleBlock
from modules.initialization.XavierInitializer import XavierInitializer

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class Generator(Model):
    def __init__(self) -> None:
        super().__init__()
        
        number_of_dense_blocks = 2
        number_of_synthesis_blocks = 5
        synthesis_block_with_attention = 3
        
        default_kwargs = {
            'normalization': None,
            'regularization': spectral_norm,
            'noise_input': None,
            'activation': nn.LeakyReLU(0.2),
            'initializer': XavierInitializer
        }
        
        for scope in range(1, number_of_dense_blocks + 1):
            self.add_dense_block(scope=scope, in_channels=8, out_channels=8, **default_kwargs)
        
        for scope in range(1, number_of_synthesis_blocks + 1):
            attention_block = True if scope == synthesis_block_with_attention else False
            self.add_synthesis_block(scope=scope, in_channels=8, out_channels=8, kernel_size=3, 
                                     attention_block=attention_block, **default_kwargs)
            
        self.add_sigmoid_block(in_channels=8, out_channels=8, kernel_size=3, **default_kwargs)
        
    def add_dense_block(self, scope, in_channels, out_channels, **kwargs):
        self.add_module(f"dense_block_{scope}", DenseBlock(in_channels, out_channels, **kwargs))
        
    def add_synthesis_block(self,
                            scope: int,
                            in_channels: int,
                            out_channels: int,
                            kernel_size: Union[int, Tuple[int,int]],
                            attention_block: bool = False,
                            blocks_in_residual = 2,
                            **kwargs):
        
        residual_block_template = ConvolutionalBlock(Conv2d(in_channels, out_channels, kernel_size), **kwargs)
        
        self.add_module(f"res_block_{scope}", ResidualBlock(blocks_in_residual, residual_block_template))
        if attention_block:
            self.add_module(f"attention_block_{scope}", AttentionBlock(out_channels))
        self.add_module(f"upscale_block_{scope}", UpscaleBlock(in_channels, out_channels, kernel_size, **kwargs))
        
    def add_sigmoid_block(self, in_channels, out_channels, kernel_size, **kwargs):
        kwargs['activation'] = nn.Sigmoid()
        sigmoid_block = ConvolutionalBlock(Conv2d(in_channels, out_channels, kernel_size), **kwargs)
        self.add_module("sigmoid_block", sigmoid_block)