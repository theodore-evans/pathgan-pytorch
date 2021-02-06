import torch.nn as nn
import torch
from torch.tensor import Tensor

from .AbstractRegularizer import AbstractRegularizer

class OrthogonalRegularizer(AbstractRegularizer):        
    def __init__(self, regularizer_scale: float, device: torch.device):
        super().__init__(regularizer_scale, device)
    
    def get_regularizer_loss(self, module: nn.Module) -> Tensor:
        loss_func = self.get_orthogonality
        return super().get_regularizer_loss(module, loss_func)

    def get_orthogonality(self,param: Tensor) -> Tensor:
        param_flat = param.view(param.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0]).to(self.device)
        return sym.abs().sum()