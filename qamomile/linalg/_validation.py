"""Share private validation for host-side linear-algebra values."""

from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike


def coerce_finite_complex_scalar(value: object, name: str) -> complex:
    """Convert one numeric scalar to a canonical finite complex value.

    Args:
        value (object): Candidate real or complex numeric scalar.
        name (str): Human-readable value name used in diagnostics.

    Returns:
        complex: Finite Python complex value whose zero components have a
            positive sign.

    Raises:
        TypeError: If ``value`` is Boolean or not a numeric scalar.
        ValueError: If a numeric scalar cannot be represented as a finite
            Python complex value.
    """
    if _is_boolean_scalar(value) or not isinstance(
        value,
        (numbers.Complex, np.number),
    ):
        raise TypeError(f"{name} must be a numeric scalar.")
    try:
        scalar = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must fit in a finite complex value.") from error
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise ValueError(f"{name} must be finite.")
    return complex(
        scalar.real if scalar.real else 0.0,
        scalar.imag if scalar.imag else 0.0,
    )


def coerce_nonnegative_finite_real(value: object, name: str) -> float:
    """Convert one numeric scalar to a canonical non-negative finite float.

    Args:
        value (object): Candidate real numeric scalar.
        name (str): Human-readable value name used in diagnostics.

    Returns:
        float: Finite non-negative Python float with a positive zero sign.

    Raises:
        TypeError: If ``value`` is Boolean or not a real numeric scalar.
        ValueError: If an accepted real scalar cannot be represented as a
            finite non-negative Python float.
    """
    if _is_boolean_scalar(value) or not isinstance(
        value,
        (numbers.Real, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real numeric scalar.")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and non-negative.") from error
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar if scalar else 0.0


def as_finite_square_complex_matrix(matrix: ArrayLike) -> np.ndarray:
    """Convert one value to a finite square complex128 matrix.

    Args:
        matrix (ArrayLike): Candidate dense matrix.

    Returns:
        np.ndarray: Finite square complex128 matrix.

    Raises:
        ValueError: If conversion fails or the result is nonsquare or
            non-finite.
    """
    try:
        dense = np.asarray(matrix, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("matrix must contain numeric values.") from error
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(f"matrix must be 2D square; got shape {dense.shape}.")
    if not np.all(np.isfinite(dense)):
        raise ValueError("matrix entries must all be finite.")
    return dense


def _is_boolean_scalar(value: object) -> bool:
    """Return whether a value is a Python or NumPy Boolean scalar.

    Args:
        value (object): Candidate scalar value.

    Returns:
        bool: Whether ``value`` is Boolean rather than numeric input.
    """
    return isinstance(value, (bool, np.bool_))
