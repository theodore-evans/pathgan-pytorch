import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union

from torch.nn.utils.spectral_norm import SpectralNorm
from torch.utils.hooks import RemovableHandle
from .utils import apply_same_padding, max_singular_value

class ConvolutionalScale(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 same_padding: bool = True,
                 fused_scale: bool = True,
                 **kwargs):
        
        super().__init__(in_channels, out_channels, kernel_size, stride=2, **kwargs)
        
        self.filter = lambda x: x
        
        if same_padding:
            apply_same_padding(self)
        
        if fused_scale:
            self.use_fused_scale()
            
    def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        if isinstance(hook, SpectralNorm):
            self.filter = self.spectral_norm_wrapper(self.filter)
        return super().register_forward_pre_hook(hook)

    def use_fused_scale(self):
        channels = (self.in_channels, self.out_channels) 
        channels = channels if isinstance(self, UpscaleConv2d) else channels[::-1]
        reduced_kernel_size = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        self.weight = Parameter(torch.Tensor(*channels, *reduced_kernel_size))
        self.bias = Parameter(torch.Tensor(self.out_channels))
        self.filter = self.fused_scale_filter
        
    def fused_scale_filter(self, weight: Tensor) -> Tensor:
        w = F.pad(weight, [1, 1, 1, 1])
        w = w[:, :, 1:, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, 1:] + w[:, :, :-1, :-1]
        return w
    
    def spectral_norm_wrapper(self, filter_func: Callable):
        def normalized(weight: Tensor):
            return self.spectral_norm(filter_func(weight))
        return normalized

    def spectral_norm(self, weight: Tensor) -> Tensor:
        w_mat = weight.view(weight.size(0), -1)
        #TODO: self.u is not initialized. Two options if we are going to execute super().register_pre hook we can directly use self.weight_u
        # else we can initialize a buffer called self.u in use_fused scale method
        sigma, _u = max_singular_value(w_mat, self.u, 1)
        self.u.copy_(_u) # type: ignore
        return weight / sigma
         
class UpscaleConv2d(ConvolutionalScale):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        **kwargs
        ) -> None:
            super().__init__(in_channels, out_channels, kernel_size, **kwargs)
                   
    def forward(self, 
                inputs: Tensor, 
                output_size: Optional[List[int]] = None
                ) -> Tensor:
        
        if output_size is None:
            output_size = [inputs.size(2) * 2, inputs.size(3) * 2]

        params = (list(param) for param in (self.stride, self.padding,
                                            self.kernel_size, self.dilation))
        
        output_padding = self._output_padding(inputs, output_size, *params)
        
        return F.conv_transpose2d(inputs, self.filter(self.weight), self.bias,
                                  self.stride, self.padding, output_padding)

class DownscaleConv2d(ConvolutionalScale):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        **kwargs
        ) -> None: #TODO: Flip weight matrix here,
            super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(inputs, self.filter(self.weight), self.bias,
                        self.stride, self.padding, self.dilation)