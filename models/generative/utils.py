from typing import Union
import torch
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

def same_padding(conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d], inputs: Tensor) -> Tensor:
    in_height = inputs.size(2)
    in_width = inputs.size(3)
    filter_height, filter_width = conv_layer.kernel_size
    strides= conv_layer.stride

    if (in_height % strides[1] == 0):
        pad_along_height = max(filter_height - strides[0], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[0]), 0)
    if (in_width % strides[1] == 0):
        pad_along_width = max(filter_width - strides[1], 0)
    else:
        pad_along_width = max(filter_width - (in_width % strides[1]), 0)
    
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return F.pad(inputs, (pad_left, pad_right, pad_top, pad_bottom))