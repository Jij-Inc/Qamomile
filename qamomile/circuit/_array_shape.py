"""Provide shared rectangular-shape discovery for array-like user inputs."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

_ARRAY_PROTOCOL_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    OverflowError,
    RuntimeError,
)


def _rectangular_array_shape(value: Any) -> tuple[int, ...]:
    """Return the concrete rectangular shape of an array-like value.

    Args:
        value (Any): Object with a ``shape`` attribute, nested sequence, or
            scalar candidate.

    Returns:
        tuple[int, ...]: Concrete dimensions; scalars have rank zero.

    Raises:
        ValueError: If shape dimensions are not nonnegative integers or nested
            sequences have inconsistent shapes.
    """
    try:
        shape = getattr(value, "shape", None)
    except _ARRAY_PROTOCOL_ERRORS as error:
        raise ValueError("Could not read the array shape.") from error
    if shape is not None:
        try:
            shape_dimensions = tuple(shape)
        except _ARRAY_PROTOCOL_ERRORS as error:
            raise ValueError(
                "Array shape must be an iterable of nonnegative integers."
            ) from error
        dimensions: list[int] = []
        for dimension in shape_dimensions:
            if isinstance(dimension, bool) or not isinstance(dimension, Integral):
                raise ValueError("Array shape dimensions must be nonnegative integers.")
            try:
                normalized_dimension = int(dimension)
            except _ARRAY_PROTOCOL_ERRORS as error:
                raise ValueError(
                    "Array shape dimensions must be nonnegative integers."
                ) from error
            if normalized_dimension < 0:
                raise ValueError("Array shape dimensions must be nonnegative integers.")
            dimensions.append(normalized_dimension)
        return tuple(dimensions)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    try:
        length = len(value)
    except _ARRAY_PROTOCOL_ERRORS as error:
        raise ValueError("Could not inspect the array sequence length.") from error
    if isinstance(value, range):
        return (length,)
    if length == 0:
        return (0,)
    try:
        children = iter(value)
        first = next(children)
    except _ARRAY_PROTOCOL_ERRORS as error:
        raise ValueError("Could not iterate over the array sequence.") from error
    first_shape = _rectangular_array_shape(first)
    try:
        for item in children:
            if _rectangular_array_shape(item) != first_shape:
                raise ValueError("Array inputs must be rectangular.")
    except ValueError:
        raise
    except _ARRAY_PROTOCOL_ERRORS as error:
        raise ValueError("Could not iterate over the array sequence.") from error
    return (length, *first_shape)
