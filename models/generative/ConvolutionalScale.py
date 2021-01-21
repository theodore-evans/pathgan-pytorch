import torch
from torch import Tensor
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.parameter import Parameter
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import ModuleDict
from torch import nn
from typing import Union, Dict, Any, Optional, List
from utils import max_singular_value

from .Block import Block

class ConvScaleBlock(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 2,
                 padding: Union[int, tuple] = 1,
                 upscale: bool = False,
                 output_size: List[int] = None,
                 **kwargs) -> None:

        layer__name = 'upscale_layer' if upscale else 'downscale_layer'
        layer_dict = ModuleDict()
        layer = ConvolutionalScale(
            in_channels, out_channels, kernel_size, stride, padding, upscale, output_size)
        layer_dict[layer__name] = layer
        kwargs['regularization'] = lambda x: x
        super().__init__(in_channels, out_channels, layer_dict, **kwargs)


class ConvolutionalScale(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 2,
                 padding: Union[int, tuple] = 1,
                 upscale: bool = False,
                 output_size: List[int] = None,
                 **kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size + 1, kernel_size + 1)
        self.stride = (stride, stride)
        self.padding = padding if type(padding) != int else (padding, padding)
        self.upscale = upscale
        self.output_size = output_size
        self.dilation = 1

        if self.upscale:
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

    def forward(self, input: Tensor, **kwargs: Dict[str, Any]) -> Tensor:
        if self.upscale:
            output_padding = self._output_padding(
                input, self.output_size, self.stride, self.padding, self.kernel_size)
            return F.conv_transpose2d(input, self.W_, self.bias, self.stride, self.padding, output_padding=output_padding, groups=1, dilation=1)
        else:
            return F.conv2d(input, self.W_, self.bias, self.stride, self.padding, self.dilation, groups=1)

    '''
    This block was taken from torch source code 
    '''

    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        min_sizes = torch.jit.annotate(List[int], [])
        max_sizes = torch.jit.annotate(List[int], [])
        for d in range(k):
            dim_size = ((input.size(d + 2) - 1) * stride[d] -
                        2 * padding[d] + kernel_size[d])
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        res = torch.jit.annotate(List[int], [])
        for d in range(k):
            res.append(output_size[d] - min_sizes[d])

        ret = res
        return ret

