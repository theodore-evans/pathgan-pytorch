import torch.nn as nn
import torch
from torch.tensor import Tensor

from .AbstractRegularizer import AbstractRegularizer

class OrthogonalRegularizer(AbstractRegularizer):        
    def __init__(self, module: nn.Module, regularizer_scale: float = 1e-4):
        super().__init__(module, regularizer_scale)
    
    def get_regularizer_loss(self) -> None:
        loss_func = self.get_orthogonality
        return super().get_regularizer_loss(loss_func)

    @staticmethod
    def get_orthogonality(param: Tensor) -> Tensor:
        param_flat = param.view(param.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0])
        return sym.abs().sum()