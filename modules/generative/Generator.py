from typing import Tuple, Union
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

from torch.nn.modules.conv import Conv2d

from modules.blocks.ResidualBlock import ResidualBlock
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.AttentionBlock import AttentionBlock
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock, UpscaleBlock
from modules.initialization.XavierInitializer import XavierInitializer
from modules.blocks.NoiseInput import NoiseInput
from modules.normalization.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class Generator(Model):
    def __init__(self,
                 latent_dim: int = 200,
                 kernel_size: Union[int, Tuple[int,int]] = 3,
                 number_of_dense_layers: int = 2,
                 number_of_synthesis_blocks: int = 5,
                 synthesis_block_with_attention: int = 2) -> None:
        super().__init__()
        
        default_kwargs = {
            'normalization': AdaptiveInstanceNormalization,
            'regularization': spectral_norm,
            'noise_input': NoiseInput,
            'activation': nn.LeakyReLU(0.2),
            'initializer': XavierInitializer,
            'latent_dim' : latent_dim
        }
        
        in_channels = latent_dim
        dense_out_channels = [1024, 12544]
        
        for scope in range(number_of_dense_layers):
            out_channels=dense_out_channels[scope]
            self.add_dense_block(scope, in_channels, out_channels, **default_kwargs)
            in_channels = out_channels
        
        # TODO: implement input reshape
        
        in_channels = 256
        synthesis_out_channels = [512, 256, 128, 64, 32]
        
        for scope in range(number_of_synthesis_blocks):
            attention_block = True if scope == synthesis_block_with_attention else False
            out_channels = synthesis_out_channels[scope]
            self.add_synthesis_block(scope, in_channels, out_channels, kernel_size, 
                                     attention_block=attention_block, **default_kwargs)
            in_channels = out_channels
        
        image_channels = 3
        
        self.add_sigmoid_block(in_channels, image_channels, kernel_size, **default_kwargs)
        
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
        
        residual_block_template = ConvolutionalBlock(Conv2d(in_channels, in_channels, kernel_size), **kwargs)
        
        self.add_module(f"res_block_{scope}", ResidualBlock(blocks_in_residual, residual_block_template))
        if attention_block:
            self.add_module(f"attention_block_{scope}", AttentionBlock(in_channels))
        self.add_module(f"upscale_block_{scope}", UpscaleBlock(in_channels, out_channels, kernel_size, **kwargs))
        
    def add_sigmoid_block(self, in_channels, out_channels, kernel_size, **kwargs):
        kwargs['activation'] = nn.Sigmoid()
        sigmoid_block = ConvolutionalBlock(Conv2d(in_channels, out_channels, kernel_size), **kwargs)
        self.add_module("sigmoid_block", sigmoid_block)