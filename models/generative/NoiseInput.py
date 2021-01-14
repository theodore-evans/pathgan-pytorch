from torch import Tensor
import torch.nn as nn

class NoiseInput(nn.Module):
    def __init__(self,
                 in_channels : int):
        super().__init__()

    def forward(self, inputs) -> Tensor:
        return Tensor()