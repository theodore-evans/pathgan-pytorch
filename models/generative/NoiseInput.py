from torch import Tensor
import torch.nn as nn

class NoiseInput(nn.Module):
    def __init__(self,
                 in_channels : int
                 ) -> None:
        super().__init__()

    def forward(self, input) -> Tensor:
        return input #TODO: implement