from modules.initialization.AbstractInitializer import AbstractInitializer
from modules.normalization.AbstractNormalization import AbstractNormalization
from typing import Callable, Optional, Tuple, TypeVar, Union
import torch.nn as nn

T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]

size_any_t = _scalar_or_tuple_any_t[int]
size_1_t = _scalar_or_tuple_1_t[int]
size_2_t = _scalar_or_tuple_2_t[int]

regularization_t = Optional[Callable[[nn.Module], nn.Module]]
noise_input_t = Optional[Callable[[int], nn.Module]]
normalization_t = Optional[Callable[..., AbstractNormalization]]
activation_t = Optional[nn.Module]
initialization_t = Optional[Callable[[nn.Module], AbstractInitializer]]