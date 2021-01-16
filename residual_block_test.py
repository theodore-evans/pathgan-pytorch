import torch
from models.generative.ResidualBlock import ResidualBlock
from models.generative.ConvolutionalBlock import ConvolutionalBlock

res_block = ResidualBlock(2, ConvolutionalBlock(12, 12, 3, 1, 0))
print(res_block)

input = torch.ones(10, 12, 3, 3)
print('input: ', input)

output = res_block(input)
print('output: ', output)