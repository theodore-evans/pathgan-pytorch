import torch
import torch.nn as nn
from models.generative.ResidualBlock import ResidualBlock
from models.generative.ConvolutionalBlock import ConvolutionalBlock
from models.generative.normalization.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization

res_block = ResidualBlock(2, ConvolutionalBlock(10, 10, 3, 1, 1, 
                                                latent_dim = 200,
                                                normalization=AdaptiveInstanceNormalization, 
                                                activation=nn.LeakyReLU(0.2)))
print(res_block)

input = torch.ones(10, 10, 3, 3)
print('input: ', input)

# output = res_block(input)
# print('output: ', output)