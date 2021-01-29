from abc import ABC

import torch.nn as nn

class AbstractNormalization(nn.Module, ABC):
    def __init__(self):
        super().__init__()