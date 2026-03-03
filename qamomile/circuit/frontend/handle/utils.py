"""Utility helpers for handle types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .array import Vector
    from .handle import Handle

__all__ = ["get_size"]


def get_size(arr: Vector[Handle]) -> int:
    """Get array size as Python int.

    Args:
        arr: A Vector of handle elements

    Returns:
        The size of the array as an integer

    Raises:
        ValueError: If the array doesn't have a fixed size
    """
    size = arr.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    if hasattr(size, "init_value"):
        return int(size.init_value)
    raise ValueError("Array must have fixed size")
