import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_layer = None
        self.conditional_normalization_layer = None