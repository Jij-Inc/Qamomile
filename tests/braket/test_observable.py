"""Tests for Qamomile-to-Braket observable conversion."""

import numpy as np
import pytest

pytestmark = pytest.mark.braket
pytest.importorskip("braket.circuits")

import qamomile.observable as qm_o  # noqa: E402
from qamomile.braket import hamiltonian_to_braket_observable  # noqa: E402


def test_mixed_hamiltonian_preserves_terms_and_coefficients() -> None:
    """A mixed Hamiltonian becomes a coefficient-preserving Braket sum."""
    hamiltonian = 0.5 * qm_o.X(0) * qm_o.Y(2) - 1.25 * qm_o.Z(1)
    hamiltonian.constant = 0.75

    observable = hamiltonian_to_braket_observable(hamiltonian)

    coefficients = sorted(float(term.coefficient) for term in observable.summands)
    assert np.allclose(
        coefficients,
        [-1.25, 0.5, 0.75],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("value", [0.0, 2.5, -3.0])
def test_scalar_hamiltonian_uses_identity(value: float) -> None:
    """Scalar Hamiltonians remain representable as Braket observables."""
    hamiltonian = qm_o.Hamiltonian.identity(value)

    observable = hamiltonian_to_braket_observable(hamiltonian)

    assert np.isclose(
        float(observable.coefficient),
        value,
        rtol=0.0,
        atol=1e-12,
    )


def test_non_hermitian_coefficient_is_rejected() -> None:
    """Observable conversion rejects a non-real Hamiltonian coefficient."""
    hamiltonian = 1j * qm_o.X(0)

    with pytest.raises(ValueError, match="real coefficients"):
        hamiltonian_to_braket_observable(hamiltonian)
