
from typing import List, Tuple, Union

from torch.tensor import Tensor
from modules.blocks.ReshapeLayer import ReshapeLayer
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

from torch.nn.modules.conv import Conv2d

from modules.blocks.ResidualBlock import ResidualBlock
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.AttentionBlock import AttentionBlock
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock, UpscaleBlock
from modules.initialization.XavierInitialization import XavierInitialization
from modules.blocks.NoiseInput import NoiseInput
from modules.normalization.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization

from modules.types import size_2_t

class Generator(nn.Module):
    def __init__(self,
                 latent_dim: int = 200,
                 output_shape: Tuple[int,int,int] = (224,224,3),
                 dense_out_channels: list = [1024, 12544],
                 synthesis_out_channels: list = [512, 256, 128, 64, 32],
                 kernel_size: size_2_t = (3,3),
                 synthesis_block_with_attention: int = 2,
                 blocks_in_residual: int = 2,
                 **kwargs
                 ) -> None:

        self.latent_dim = latent_dim
        
        self.output_image_size = tuple(output_shape[i] for i in (0,1))
        self.image_channels = output_shape[2]
            
        self.dense_out_channels = dense_out_channels
        self.synthesis_out_channels = synthesis_out_channels
        self.num_dense_blocks = len(dense_out_channels)
        self.num_synthesis_blocks = len(synthesis_out_channels)
        
        self.kernel_size = kernel_size
        self.synthesis_block_with_attention = synthesis_block_with_attention
        self.blocks_in_residual = blocks_in_residual

        def image_size_is_valid():
            upscale_factor = 2 ** self.num_synthesis_blocks
            return all(side % upscale_factor == 0 for side in self.output_image_size)
        
        if not image_size_is_valid:
            raise ValueError(f'''Output image size must be a multiple of
                             2^{self.num_synthesis_blocks} (no. upscale layers)''')
        
        super().__init__()

        default_kwargs = {
            'normalization': AdaptiveInstanceNormalization,
            'regularization': spectral_norm,
            'noise_input': NoiseInput,
            'activation': nn.LeakyReLU(0.2),
            'initialization': XavierInitialization,
            'latent_dim' : latent_dim
        }
        
        for name, value in default_kwargs.items():
            if name not in kwargs:
                kwargs[name] = value
        
        next_in_channels = self.latent_dim
        next_in_channels = self.add_dense_blocks(next_in_channels, **kwargs)
        next_in_channels = self.add_reshape_layer(next_in_channels)
        next_in_channels = self.add_synthesis_blocks(next_in_channels, **kwargs)
        self.add_sigmoid_block(next_in_channels, **kwargs)
        
    def forward(self, inputs: Tensor, latent_input: Union[Tensor, List[Tensor]]) -> Tensor:
        net = inputs
        for block in self.children():
            net = block(net, latent_input=latent_input)
        return net
    
    def add_dense_blocks(self, in_channels, **kwargs):
        out_channels = in_channels
        for scope in range(self.num_dense_blocks):
            out_channels=self.dense_out_channels[scope]
            self.add_dense_block(scope, in_channels, out_channels, **kwargs)
            in_channels = out_channels
        return out_channels
    
    def add_dense_block(self, scope, in_channels, out_channels, **kwargs):
        self.add_module(f"dense_block_{scope}", DenseBlock(in_channels, out_channels, **kwargs))
    
    def add_reshape_layer(self, in_channels):
        upscale_factor = 2 ** self.num_synthesis_blocks
        image_shape = tuple(side // upscale_factor for side in self.output_image_size)
        out_channels = in_channels // (image_shape[0] * image_shape[1])
        self.add_module("reshape_block", ReshapeLayer(in_channels, out_channels, image_shape))
        return out_channels
    
    def add_synthesis_blocks(self, in_channels, **kwargs):
        out_channels = in_channels
        for scope in range(self.num_synthesis_blocks):
            has_attention_block = True if scope == self.synthesis_block_with_attention else False
            out_channels = self.synthesis_out_channels[scope]
            self.add_synthesis_block(scope, in_channels, out_channels, self.kernel_size,
                                     has_attention_block, **kwargs)
            in_channels = out_channels
        return out_channels
        
    def add_synthesis_block(self,
                            scope: int,
                            in_channels: int,
                            out_channels: int,
                            kernel_size: size_2_t,
                            has_attention_block: bool = False,
                            **kwargs):
        
        residual_block_template = ConvolutionalBlock(Conv2d(in_channels, in_channels, kernel_size), **kwargs)
        
        self.add_module(f"res_block_{scope}", ResidualBlock(self.blocks_in_residual, residual_block_template))
        if has_attention_block:
            self.add_module(f"attention_block_{scope}", AttentionBlock(in_channels, **kwargs))
        self.add_module(f"upscale_block_{scope}", UpscaleBlock(in_channels, out_channels, kernel_size, **kwargs))
        
    def add_sigmoid_block(self, in_channels, **kwargs):
        kwargs['activation'] = nn.Sigmoid()
        out_channels = self.image_channels
        sigmoid_block = ConvolutionalBlock(Conv2d(in_channels, out_channels, self.kernel_size), **kwargs)
        self.add_module("sigmoid_block", sigmoid_block)
        