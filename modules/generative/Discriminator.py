from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import ModuleDict
import torch
import torch.nn as nn

from modules.blocks.ResidualBlock import ResidualBlock
from modules.blocks.AttentionBlock import AttentionBlock
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock, DownscaleBlock
from modules.initialization.XavierInitialization import XavierInitialization
from modules.blocks.ConvolutionalScale import DownscaleConv2d

class DiscriminatorResnet(nn.Module):
    def __init__(
        self,
        layers: int = 5,
        attention_size: int = 28,
        starting_channels: int = 32,
        image_size: int = 224,
        orthogonal_reg_scale: float = 1e-4
    ) -> None:
        super().__init__()

        in_channels = 3
        out_channels = starting_channels
        current_size = image_size
        self.ortho_reg_scale = orthogonal_reg_scale

        self.conv_part = ModuleDict()
        self.dense_part = ModuleDict()

        default_kwargs = {
            'normalization': None,
            'regularization': spectral_norm,
            'noise_input': None,
            'activation': LeakyReLU(0.2),
            'initialization': XavierInitialization
        }

        for layer in range(layers):

            # Res Block
            inner_res = ConvolutionalBlock(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1), **default_kwargs)
            res_block = ResidualBlock(
                num_blocks=2, block_template=inner_res)

            self.conv_part.add_module(f'ResNet_{layer}', res_block)

            # Attention Block
            if attention_size is not None and current_size == attention_size:
                att_block = AttentionBlock(
                    channels=in_channels, **default_kwargs)
                self.conv_part.add_module('Attention', att_block)

            # Downsample
            down = DownscaleBlock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=5, **default_kwargs)

            self.conv_part.add_module(f'DownScale_{layer}', down)
            in_channels = out_channels
            out_channels *= 2
            current_size //= 2

        # First Dense
        flattened_shape = current_size * current_size * in_channels
        # Add orho regularizer later
        dense = DenseBlock(in_channels=flattened_shape, out_channels=in_channels,
                           **default_kwargs)
        self.dense_part.add_module('Dense_1', dense)
        # Second Dense
        dense = DenseBlock(in_channels=in_channels, out_channels=1,
                           **default_kwargs)
        self.dense_part.add_module('Dense_2', dense)

    def get_orthogonal_reg_loss(self):
        with torch.enable_grad():
            orth_loss = torch.zeros(1)
            for name, param in self.named_parameters():
                # Ignore bias, scalars and scale layers
                if 'bias' not in name and 'Scale' not in name and len(param.shape) > 1:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0])
                    orth_loss = orth_loss + (self.ortho_reg_scale * sym.abs().sum())
            return orth_loss

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = data.shape[0]
        # Pass from Res Layers

        for m in self.conv_part.children():
            data = m(data, **kwargs)
            print(data.shape)

        # Flatten
        data = data.reshape((batch_size, -1))

        # Pass from dense Layers
        for m in self.dense_part.children():
            data = m(data, **kwargs)

        return data
