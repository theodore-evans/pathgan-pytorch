from torch.nn.modules.container import ModuleDict
from models.generative.ConvolutionalBlock import ConvolutionalBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResidualBlock import ResidualBlock
from .AttentionBlock import AttentionBlock
from .DenseBlock import DenseBlock
from .ConvolutionalScale import ConvolutionalScaleVanilla

'''
TODO: Remove soon
output, logits = discriminator_resnet
(
        images=images,
        layers=self.layers, 5
        spectral=True,
        activation=leakyReLU, alpha = 0.2
        reuse=reuse,
        attention=self.attention, 28
        init=init, 
        regularizer=orthogonal_reg(self.regularizer_scale), scale 10e-4
        label=label_input, None
        label_t=self.label_t
)
'''

class DiscriminatorResnet(nn.Module):
    def __init__(
        self,
        layers: int = 5,
        attention_size: int = 28,
        starting_channels: int = 32,
        image_size: int = 224
    ) -> None:
        super().__init__()

        in_channels = 3
        out_channels = starting_channels
        current_size = image_size

        self.conv_part = ModuleDict()
        self.dense_part = ModuleDict()

        default_kwargs = {
            'normalization': None,
            'regularization': 'spectral'
        }

        for layer in range(layers):

            # Spectral norm, init mode, regularizer and activation should be added
            # init is xavier
            # normalization=None,
            # noise_input_f=False,
            # use_bias=True,
            # spectral=True,
            # activation=leakyRelu,
            # init='xavier',
            # regularizer= Orthogonal Reg,
            # TODOs: implement orho regularizer, inits and spectral norms

            # Res Block
            inner_res = ConvolutionalBlock(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, **default_kwargs)
            res_block = ResidualBlock(
                num_blocks=2, block_template=inner_res, **default_kwargs)

            self.conv_part.add_module(f'ResNet_{layer}', res_block)

            # Attention Block
            if attention_size is not None and current_size == attention_size:
                att_block = AttentionBlock(
                    channels=in_channels, **default_kwargs)
                self.conv_part.add_module('Attention', att_block)

            # Downsample

            down = ConvolutionalScaleVanilla(
                in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=1, padding=2, **default_kwargs)

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

    def initialize_modules(self):
        pass

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = data.shape[0]
        # Pass from Res Layers

        for m in self.conv_part.children():
            data = m(data, **kwargs)

        # Flatten
        data = data.reshape((batch_size, -1))

        # Pass from dense Layers
        for m in self.dense_part.children():
            data = m(data, **kwargs)

        return data