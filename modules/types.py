from typing import Tuple, TypeVar, Union

T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]

size_any_t = _scalar_or_tuple_any_t[int]
size_1_t = _scalar_or_tuple_1_t[int]
size_2_t = _scalar_or_tuple_2_t[int]