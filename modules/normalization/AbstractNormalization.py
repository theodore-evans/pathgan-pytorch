from abc import ABC

from torch.tensor import Tensor
import torch.nn as nn

class AbstractNormalization(nn.Module, ABC):
    def __init__(self):
        super().__init__()