"""Provide shared rectangular-shape discovery for array-like user inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _rectangular_array_shape(value: Any) -> tuple[int, ...]:
    """Return the concrete rectangular shape of an array-like value.

    Args:
        value (Any): Object with a ``shape`` attribute, nested sequence, or
            scalar candidate.

    Returns:
        tuple[int, ...]: Concrete dimensions; scalars have rank zero.

    Raises:
        ValueError: If nested sequences have inconsistent shapes.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dimension) for dimension in shape)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, range):
        return (len(value),)
    if len(value) == 0:
        return (0,)
    children = iter(value)
    first_shape = _rectangular_array_shape(next(children))
    for item in children:
        if _rectangular_array_shape(item) != first_shape:
            raise ValueError("Array inputs must be rectangular.")
    return (len(value), *first_shape)
