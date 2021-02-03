from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from torch.tensor import Tensor


class AbstractRegularizer(ABC):
    @abstractmethod
    def __init__(self, regularizer_scale: float = 1e-4):
        self.regularizer_scale = regularizer_scale

    @abstractmethod
    def get_regularizer_loss(self, module:nn.Module, loss_function: Callable[[Tensor], Tensor] = None) -> Tensor:
        if loss_function is None:
            return torch.zeros(1)

        with torch.enable_grad():
            loss = torch.zeros(1)
            for name, param in module.named_parameters():
                # Ignore biases and one dimensional params
                if len(param.shape)!=1 and 'scale' not in name:
                    loss += self.regularizer_scale * loss_function(param)
            return loss
