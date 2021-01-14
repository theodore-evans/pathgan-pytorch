import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def initialise_weights(self, initialization : str = 'xavier') -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                # Different init modes applied only to the weight tensor of the Layer
                if initialization == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                elif initialization == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif initialization == 'normal':
                    nn.init.normal_(m.weight, std=0.02)

                # Bias is initialized with constant 0 values, still trainable
                nn.init.constant_(m.bias, 0.)

class Generator(Model):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    
