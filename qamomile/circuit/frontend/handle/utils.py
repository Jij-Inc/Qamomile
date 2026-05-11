"""Utility helpers for handle types."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .array import Vector
    from .handle import Handle

__all__ = ["get_size"]

_H = TypeVar("_H", bound="Handle")


def get_size(arr: Vector[_H]) -> int:
    """Return the size of a Vector handle as a Python integer.

    Resolves the leading axis of *arr.shape* through the three forms a
    ``Vector`` shape entry can take: a plain Python integer, a constant
    ``Value`` produced by partial evaluation, or a symbolic ``UInt`` handle
    that nevertheless carries an ``init_value`` known at build time.

    Args:
        arr (Vector[Handle]): Vector handle whose first axis size is
            requested.

    Returns:
        int: The first-axis size as a plain Python ``int``.

    Raises:
        ValueError: If the shape cannot be resolved to a concrete integer
            (e.g., the Vector is genuinely symbolic at this point in the
            pipeline).
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
