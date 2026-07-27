"""Test validation shared by host-side LCU representations."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from qamomile.linalg import (
    PauliLCU,
    PauliLCUTerm,
    PeriodicShiftLCU,
    PeriodicShiftLCUTerm,
)


class _UnconvertibleReal:
    """Model a registered real scalar whose float conversion fails."""

    def __float__(self) -> float:
        """Reject conversion after the numeric runtime type check."""
        raise ValueError("cannot convert this real scalar")


numbers.Real.register(_UnconvertibleReal)


def _pauli_error_bound(value: object) -> float:
    """Return one Pauli LCU error bound after validation.

    Args:
        value (object): Candidate truncation error bound.

    Returns:
        float: Canonical validated error bound.
    """
    return PauliLCU(
        1,
        (),
        truncation_error_bound=value,  # type: ignore[arg-type]
    ).truncation_error_bound


def _periodic_error_bound(value: object) -> float:
    """Return one periodic-shift LCU error bound after validation.

    Args:
        value (object): Candidate truncation error bound.

    Returns:
        float: Canonical validated error bound.
    """
    return PeriodicShiftLCU(
        (1,),
        (),
        truncation_error_bound=value,  # type: ignore[arg-type]
    ).truncation_error_bound


def _pauli_coefficient(value: object) -> complex:
    """Return one validated Pauli LCU coefficient.

    Args:
        value (object): Candidate LCU coefficient.

    Returns:
        complex: Canonical validated coefficient.
    """
    return PauliLCUTerm(value, ()).coefficient  # type: ignore[arg-type]


def _periodic_coefficient(value: object) -> complex:
    """Return one validated periodic-shift LCU coefficient.

    Args:
        value (object): Candidate LCU coefficient.

    Returns:
        complex: Canonical validated coefficient.
    """
    return PeriodicShiftLCUTerm(value, (0,)).coefficient  # type: ignore[arg-type]


def _pauli_from_matrix(matrix: Any) -> PauliLCU:
    """Build a Pauli LCU from one matrix candidate.

    Args:
        matrix (Any): Candidate dense matrix.

    Returns:
        PauliLCU: Validated Pauli decomposition.
    """
    return PauliLCU.from_matrix(matrix)


def _periodic_from_matrix(matrix: Any) -> PeriodicShiftLCU:
    """Build a periodic-shift LCU from one matrix candidate.

    Args:
        matrix (Any): Candidate dense matrix.

    Returns:
        PeriodicShiftLCU: Validated periodic-shift decomposition.
    """
    return PeriodicShiftLCU.from_matrix(matrix, register_sizes=(1,))


_ERROR_BOUND_BUILDERS: tuple[Callable[[object], float], ...] = (
    _pauli_error_bound,
    _periodic_error_bound,
)
_COEFFICIENT_BUILDERS: tuple[Callable[[object], complex], ...] = (
    _pauli_coefficient,
    _periodic_coefficient,
)
_MATRIX_BUILDERS: tuple[Callable[[Any], object], ...] = (
    _pauli_from_matrix,
    _periodic_from_matrix,
)


@pytest.mark.parametrize("builder", _ERROR_BOUND_BUILDERS)
def test_error_bounds_canonicalize_negative_zero(
    builder: Callable[[object], float],
) -> None:
    """Every LCU representation stores zero bounds with a positive sign."""
    result = builder(-0.0)

    assert result == 0.0
    assert math.copysign(1.0, result) == 1.0


@pytest.mark.parametrize("builder", _ERROR_BOUND_BUILDERS)
@pytest.mark.parametrize("value", [True, "1", 1j])
def test_error_bounds_reject_non_real_types(
    builder: Callable[[object], float],
    value: object,
) -> None:
    """Every LCU representation reports non-real values as type errors."""
    with pytest.raises(TypeError, match="real numeric scalar"):
        builder(value)


@pytest.mark.parametrize("builder", _ERROR_BOUND_BUILDERS)
@pytest.mark.parametrize(
    "value",
    [-1.0, np.nan, np.inf, 10**1000, _UnconvertibleReal()],
)
def test_error_bounds_reject_invalid_real_values(
    builder: Callable[[object], float],
    value: object,
) -> None:
    """Every LCU representation reports invalid numeric values consistently."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        builder(value)


@pytest.mark.parametrize("builder", _COEFFICIENT_BUILDERS)
def test_coefficients_canonicalize_signed_zero_components(
    builder: Callable[[object], complex],
) -> None:
    """Every LCU term canonicalizes the sign of a zero imaginary part."""
    result = builder(complex(-1.0, -0.0))

    assert result == complex(-1.0, 0.0)
    assert math.copysign(1.0, result.imag) == 1.0


@pytest.mark.parametrize("builder", _COEFFICIENT_BUILDERS)
@pytest.mark.parametrize("value", [True, "1"])
def test_coefficients_reject_non_numeric_types(
    builder: Callable[[object], complex],
    value: object,
) -> None:
    """Every LCU term reports non-numeric values as type errors."""
    with pytest.raises(TypeError, match="numeric scalar"):
        builder(value)


@pytest.mark.parametrize("builder", _COEFFICIENT_BUILDERS)
@pytest.mark.parametrize("value", [np.nan, np.inf, 10**1000])
def test_coefficients_reject_invalid_numeric_values(
    builder: Callable[[object], complex],
    value: object,
) -> None:
    """Every LCU term reports unrepresentable coefficients as value errors."""
    with pytest.raises(ValueError, match="finite"):
        builder(value)


@pytest.mark.parametrize("builder", _MATRIX_BUILDERS)
def test_matrix_consumers_share_numeric_conversion_errors(
    builder: Callable[[Any], object],
) -> None:
    """Every dense LCU path rejects nonnumeric matrix entries consistently."""
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        builder([["not numeric", 0], [0, 1]])
