import torch.nn as nn

class ConditionalNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_layer_1 = None
        self.dense_layer_1_activation = nn.ReLU()
        
        self.dense_layer_gamma = None
        self.dense_layer_gamma_activation = nn.ReLU()
        
        self.dense_layer_beta = None
        self.dense_layer_gamma_activation = None

    def forward(self, input, latent_input):
        pass