from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleDict

from .Block import Block

#TODO: refactor, perhaps phase out redundant class. i.e. merge with conv scale. 
#TODO: wrapper class for convolutional layer, with output padding/size, 'SAME' padding mode
class ConvolutionalBlock(Block):
    def __init__(self,
                 transpose: bool = False,
                 **kwargs,
                 ) -> None:
        
        conv_layer = nn.ConvTranspose2d(**kwargs) if transpose else nn.Conv2d(**kwargs)
        module_dict = ModuleDict({'conv_layer' : conv_layer})
        
        super().__init__(module_dict = module_dict, **kwargs)
