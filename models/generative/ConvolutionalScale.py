import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union
from torch.nn.utils.spectral_norm import SpectralNorm, spectral_norm
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
        self.filter = Parameter(torch.Tensor(*channels, *self.kernel_size))
        
        #self.filter = torch.zeros_like(self.weight)
        self.register_forward_pre_hook(FusedScale(name='filter'))
        
        #self._forward_pre_hooks.move_to_end(fused_scale_hook.id, False)
        
        #for k, hook in self._forward_pre_hooks.items():
        #    if isinstance(hook, SpectralNorm) and hook.name == 'weight':
        #        dim = 1 if isinstance(self, UpscaleConv2d) else 0
        #        self._forward_pre_hooks[k] = SpectralNorm(name=filter_name, dim=dim)
        
class FusedScale:
    def __init__(self, name: str = 'filter'):
        self.name = name
    
    def __call__(self, module, _):
        fused_scale_filter = F.pad(module.weight, [1, 1, 1, 1])
        fused_scale_filter = fused_scale_filter[:, :, 1:, 1:] + fused_scale_filter[:, :, 1:, :-1] + fused_scale_filter[:, :, :-1, 1:] + fused_scale_filter[:, :, :-1, :-1]
        setattr(module, self.name, Parameter(fused_scale_filter))
         
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