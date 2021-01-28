from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch.nn as nn
from torch.tensor import Tensor

class AbstractInitializer(ABC):
    @abstractmethod
    def __init__(self, module: nn.Module):
        self.module = module
        
    @abstractmethod
    def initialize_weights(self, initialization: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        for m in self.module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                
                if initialization is not None:
                    initialization(m.weight)

                # Bias is initialized with constant 0 values, still trainable
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0.)