import torch
import torch.jit
from torch import Tensor
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.container import ModuleDict
from torch import nn
from typing import Tuple, Union, Dict, Any, Optional, List
from utils import max_singular_value
from warnings import warn

from .Block import Block

class ConvScaleBlock(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 2,
                 padding: Union[int, tuple] = 1,
                 upscale: bool = False,
                 **kwargs) -> None:

        layer__name = 'upscale_layer' if upscale else 'downscale_layer'
        layer_dict = ModuleDict()
        layer = ConvolutionalScale(
            in_channels, out_channels, kernel_size, stride, padding, upscale)
        layer_dict[layer__name] = layer
        kwargs['regularization'] = lambda x: x
        super().__init__(in_channels, out_channels, layer_dict, **kwargs)


class ConvolutionalScale(nn.Conv2d, nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 upscale: bool = False,
                 **kwargs) -> None:
        
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1) if isinstance(kernel_size, tuple) else (kernel_size + 1, kernel_size + 1)
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        
        self.upscale = upscale

        if self.upscale:
            ineq = lambda i: self.dilation[i] * (self.kernel_size[i] - 1) - 2 * self.padding[i]
            for i in range(2):
                if self.stride[i] != 2:
                    warn("Stride must be (2,2) for upscale layers. Using stride = (2,2) but please review arguments")
                    self.stride = (2,2)
                if ineq(i) < 0 or ineq(i) > 2:
                    raise Exception("Invalid parameters for upscale layer. For correct output padding,",
                                    "the following inequality must hold: ",
                                    "0 <= dilation * (kernel_size - 1) - 2 * padding <= 2")
        
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels, kernel_size, kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels, kernel_size, kernel_size))

        self.bias = Parameter(torch.Tensor(out_channels))
        if self.upscale:
            self.register_buffer('u', torch.Tensor(1, in_channels).normal_())
        else:
            self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    

    '''
    This was used in the original pathgan and in the stylegan repo:
    https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    Padding the weights and adding them by a sliding window results in a composite operation
    Fusing Conv + Scaling for Performance
    Don't know how it functions yet
    '''
    @property
    def W_(self):
        weights = F.pad(self.weight, [1, 1, 1, 1])
        weights = weights[:, :, 1:, 1:] + weights[:, :, 1:, :-
                                                  1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1]
        w_mat = weights.view(weights.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u, 1)
        self.u.copy_(_u)
        return weights / sigma

    def forward(self, inputs: Tensor, **kwargs: Dict[str, Any]) -> Tensor:
        if self.upscale:
            output_size = [inputs.size(2) * 2, inputs.size(3) * 2]
            output_padding = self._output_padding(
                inputs, output_size, list(self.stride), list(self.padding), list(self.kernel_size))
            return F.conv_transpose2d(inputs, self.W_, self.bias, self.stride, self.padding, output_padding=output_padding, groups=1, dilation=1)
        else:
            return F.conv2d(inputs, self.W_, self.bias, self.stride, self.padding, self.dilation, groups=1)