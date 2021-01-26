# from .utils import max_singular_value
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from typing import Any, List, Optional, OrderedDict, Tuple, Union
from torch.nn.utils.spectral_norm import SpectralNorm

class ConvolutionalScale(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation = 1,
                 same_padding = True,
                 **kwargs):
        
        super().__init__(in_channels, out_channels, kernel_size, stride=2, **kwargs)
        
        channels = (out_channels, in_channels) if isinstance(self, DownscaleConv2d) else  (in_channels, out_channels)
        
        reduced_kernel_size = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        
        self.filter = torch.zeros_like(self.weight)
        self.weight = Parameter(torch.Tensor(*channels, *reduced_kernel_size))
        self.bias = Parameter(torch.Tensor(self.out_channels))
        self.register_buffer('u', torch.Tensor(1, channels[0]).normal_())
        
        fused_scale_hook = self.register_forward_pre_hook(FusedScale())
        self._forward_pre_hooks.move_to_end(fused_scale_hook.id, False)
        
        for k, hook in self._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == 'weight':
                self._forward_pre_hooks[k] = SpectralNorm(name='filter', dim=1)
        
        if same_padding:
            self.apply_same_padding()

    def apply_same_padding(self):
        effective_kernel_size = tuple(self.dilation[i] * (self.kernel_size[i] - 1) + 1 for i in range(2))
        padding = []
        for k in effective_kernel_size:
            if k % 2 == 0:
                raise ValueError("In order to correctly pad input, effective kernel size (dilation*(kernel-1)+1) must be odd")
            padding.append((k - 1) // 2)
        self.padding = tuple(padding)
    
    def fused_scale_filter(self) -> Tensor:
        weights = F.pad(self.weight, [1, 1, 1, 1])
        weights = weights[:, :, 1:, 1:] + weights[:, :, 1:, :-1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1]
        return weights
    
    # def spectral_norm(self, weights: Tensor) -> Tensor:
    #     w_mat = weights.view(weights.size(0), -1)
    #     sigma, _u = max_singular_value(w_mat, self.u, 1)
    #     self.u.copy_(_u) # type: ignore
    #     return weights / sigma
class FusedScale:
    def __call__(self, module, _):
        weights = F.pad(module.weight, [1, 1, 1, 1])
        weights = weights[:, :, 1:, 1:] + weights[:, :, 1:, :-1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1]
        setattr(module, 'filter', weights)
         
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