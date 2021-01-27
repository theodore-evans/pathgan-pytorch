from typing import Tuple, Union
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import Tensor

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if Ip < 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u

def apply_same_padding(conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d]) -> None:
    effective_kernel_size = tuple(conv_layer.dilation[i] * (conv_layer.kernel_size[i] - 1) + 1 for i in range(2))
    #print(f'dilation: {conv_layer.dilation}, kernel_size: {conv_layer.kernel_size}, effective kernel size: {effective_kernel_size}')
    for dim in range(2):
        if effective_kernel_size[dim] % 2 == 0:
            raise ValueError("In order to correctly pad input, effective kernel size (dilation*(kernel-1)+1) must be odd")
    conv_layer.padding = tuple((k - 1) // 2 for k in effective_kernel_size)