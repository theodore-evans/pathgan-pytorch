import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def initialise_weights(self) -> None:
        # default xavier init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                # TODO: optional, add dependency injection for other initialisations
                nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain(**(m.activation)))
    

class Generator(Model):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    
