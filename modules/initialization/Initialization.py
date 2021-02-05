from typing import Callable, Optional

import torch.nn as nn
from torch.tensor import Tensor

from modules.utils import get_parameters

init_method_t = Callable[[Tensor], Tensor]

class Initialization:
    def __init__(self,
                 module: nn.Module,
                 weight_init: Optional[init_method_t] = None,
                 bias_init: Optional[init_method_t] = None
                 ) -> None:
        self.module = module
        self.weight_init = weight_init
        self.bias_init = bias_init
        
    def initialize(self) -> None:
        weights = get_parameters(self.module, 'weight')
        for weight in weights:
            if self.weight_init is not None:
                self.weight_init(weight[1])
        
        biases = get_parameters(self.module, 'bias')
        for bias in biases:
            if self.bias_init is not None:
                self.bias_init(bias[1])