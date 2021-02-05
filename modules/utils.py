from typing import  List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import Tensor

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1) -> Tuple[Tensor, Tensor]:
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
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v) # type: ignore
    return sigma, _u

def apply_same_padding(conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d]) -> None:
        
    effective_kernel_size = tuple(conv_layer.dilation[i] * (conv_layer.kernel_size[i] - 1) + 1 for i in range(2))
    
    padding = []
    for k in effective_kernel_size:
        if k % 2 == 0:
            raise ValueError("In order to correctly pad input, effective kernel size (dilation*(kernel-1)+1) must be odd")
        padding.append((k - 1) // 2)
        
    conv_layer.padding = tuple(padding)
    
def pair(parameter: Union[int, Tuple[int,int]]) -> Tuple[int,int]:
    return parameter if isinstance(parameter, tuple) else (parameter, parameter)

def parameter_is_of_type(named_parameter: Tuple[str, Tensor], named_type: str) -> bool:
    parameter_type = named_parameter[0].split('.')[-1]
    return parameter_type in (named_type, named_type+'_orig')

def get_parameters(m: nn.Module, param: str, recurse: bool = False) -> List[Tuple[str, Tensor]]:
    return [np for np in m.named_parameters(recurse=recurse) if parameter_is_of_type(np, param)]