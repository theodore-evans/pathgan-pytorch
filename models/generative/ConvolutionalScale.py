import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union

from .utils import max_singular_value, apply_same_padding
class ConvolutionalScale(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0,
                 upscale: bool = False,
                 same_padding: bool = True,
                 **kwargs) -> None:

        self._upscale = upscale
       
        super().__init__(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        
        if same_padding:
                apply_same_padding(self) #FIXME: I don't really like that this is a global method, but it will have to do for now

        channels = (in_channels, out_channels) if upscale else (out_channels, in_channels)
        reduced_kernel_size = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        
        self.weight = Parameter(torch.Tensor(*channels, *reduced_kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))
        
        self.register_buffer('u', torch.Tensor(1, channels[0]).normal_())
        
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

    def forward(self, inputs: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        conv_parameters = dict({'stride': self.stride, 'padding': self.padding, 'dilation': self.dilation})
        
        weight = self.W_
        filter_size = (weight.size(2), weight.size(3))
        
        if self._upscale:
            if output_size is None:
                output_size = [inputs.size(2) * 2, inputs.size(3) * 2]
            params = (list(param) for param in (self.stride, self.padding, filter_size, self.dilation))
            output_padding = self._output_padding(inputs, output_size, *params)
            return F.conv_transpose2d(inputs, weight, self.bias, **conv_parameters, output_padding=output_padding, groups=1)
        else:
            return F.conv2d(inputs, weight, self.bias, **conv_parameters, groups=1)