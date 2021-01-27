import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union
from .utils import apply_same_padding

class ConvolutionalScale(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 same_padding: bool = True,
                 fused_scale: bool = True,
                 **kwargs):
        
        super().__init__(in_channels, out_channels, kernel_size, stride=2, **kwargs)
        
        if fused_scale:
            self.use_fused_scale()
        
        if same_padding:
            apply_same_padding(self)

    def use_fused_scale(self):
        channels = (self.in_channels, self.out_channels) 
        channels = channels if isinstance(self, UpscaleConv2d) else channels[::-1]
        reduced_kernel_size = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        self.weight = Parameter(torch.Tensor(*channels, *reduced_kernel_size))
        self.bias = Parameter(torch.Tensor(self.out_channels))
        self.filter = Parameter(torch.ones(*channels, *self.kernel_size))
        
        self.register_forward_pre_hook(FusedScale(name='filter'))

class FusedScale:
    def __init__(self, name: str = 'filter'):
        self.name = name
    
    def __call__(self, module, _):
        filter = self.fused_scale(module.weight)
        setattr(module, self.name+"_orig", Parameter(filter))

    def fused_scale(self, weight: Tensor) -> Tensor:
        padded = F.pad(weight, [1, 1, 1, 1])
        filter = padded[:, :, 1:, 1:] + padded[:, :, 1:, :-1] + padded[:, :, :-1, 1:] + padded[:, :, :-1, :-1]
        return filter
         
class UpscaleConv2d(ConvolutionalScale):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        **kwargs
        ) -> None:
            super().__init__(in_channels, out_channels, kernel_size, **kwargs)
                   
    def forward(self, inputs: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        
        if output_size is None:
            output_size = [inputs.size(2) * 2, inputs.size(3) * 2]

        params = (list(param) for param in (self.stride, self.padding, self.kernel_size, self.dilation))
        output_padding = self._output_padding(inputs, output_size, *params)
        
        return F.conv_transpose2d(inputs, self.filter, self.bias, self.stride, self.padding,
                                  output_padding=output_padding)

class DownscaleConv2d(ConvolutionalScale):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        **kwargs
        ) -> None:
            super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(inputs, self.filter, self.bias, self.stride, self.padding, self.dilation)