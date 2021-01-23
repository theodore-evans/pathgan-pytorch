import torch
import torch.jit
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Union

from .utils import max_singular_value
class ConvolutionalScale(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0,
                 upscale: bool = False,
                 **kwargs) -> None:

        # FIXME: this step doesn't make sense, as we are essentially incrementing the kernel size twice
        # which causes all sorts of problems further down the line, including making sure that 
        # _output_padding doesn't work correctly, which is why the tests are failing
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1) if isinstance(kernel_size, tuple) else (kernel_size + 1, kernel_size + 1)
        
        super().__init__(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        
        self.upscale = upscale
        
        if self.upscale:
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels, kernel_size[0], kernel_size[1])
                )
        else:
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1])
                )

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
        weights = weights[:, :, 1:, 1:] + weights[:, :, 1:, :-1] + weights[:, :, :-1, 1:] + weights[:, :, :-1, :-1]
        w_mat = weights.view(weights.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u, 1)
        self.u.copy_(_u) # type: ignore
        return weights / sigma

    def forward(self, inputs: Tensor) -> Tensor:
        if self.upscale:
            output_size = [inputs.size(2) * 2, inputs.size(3) * 2]
            output_padding = self._output_padding(
                inputs, output_size, list(self.stride), list(self.padding), list(self.kernel_size), list(self.dilation))
            return F.conv_transpose2d(inputs, self.W_, self.bias, self.stride, self.padding, output_padding=output_padding, groups=1, dilation=1)
        else:
            return F.conv2d(inputs, self.W_, self.bias, self.stride, self.padding, self.dilation, groups=1)