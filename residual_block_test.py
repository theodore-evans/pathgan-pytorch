from models.generative.ResidualBlock import ResidualBlock
from models.generative.ConvolutionalBlock import ConvolutionalBlock

print(ResidualBlock(2, ConvolutionalBlock(200, 200, 3, 1, 0)))

