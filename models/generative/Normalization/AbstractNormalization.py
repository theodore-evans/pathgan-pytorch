from abc import ABC

from torch.tensor import Tensor

class AbstractNormalization(ABC):
    def __init__(self):
        pass
    
    def forward(self, inputs: Tensor, latent_input: Tensor):
        pass