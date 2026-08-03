"""Tests for shared Pauli-evolution emission helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
    is_zero_evolution_time,
)


@pytest.mark.parametrize(
    "gamma",
    [
        0,
        0.0,
        np.int64(0),
        np.float32(0.0),
        np.float64(0.0),
    ],
)
def test_zero_evolution_time_accepts_real_numeric_scalars(gamma: object) -> None:
    """Python and NumPy real zeros all select the identity specialization."""
    assert is_zero_evolution_time(gamma)


@pytest.mark.parametrize(
    "gamma",
    [
        True,
        False,
        np.bool_(False),
        np.float32(1.0),
        object(),
    ],
)
def test_zero_evolution_time_rejects_nonzero_or_nonnumeric_values(
    gamma: object,
) -> None:
    """Booleans, nonzero numbers, and backend-like objects are not zero time."""
    assert not is_zero_evolution_time(gamma)
