"""Tests for QURI Parts observable conversion.

Covers hamiltonian_to_quri_operator and to_quri_operator with various
Hamiltonian configurations including edge cases.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

# Skip entire module if QURI Parts operator support is not installed.
pytest.importorskip("quri_parts.core.operator")

import qamomile.observable as qm_o
from qamomile.quri_parts.observable import (
    hamiltonian_to_quri_operator,
    to_quri_operator,
)


class TestHamiltonianToQuriOperator:
    """Tests for hamiltonian_to_quri_operator conversion."""

    def test_single_z_term(self) -> None:
        """Verify single Z(0) Hamiltonian produces one Z term at qubit 0."""
        h = qm_o.Z(0)
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    def test_single_x_term(self) -> None:
        """Verify single X(0) Hamiltonian produces one X term at qubit 0."""
        h = qm_o.X(0)
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    def test_single_y_term(self) -> None:
        """Verify single Y(0) Hamiltonian produces one Y term at qubit 0."""
        h = qm_o.Y(0)
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    @pytest.mark.parametrize("coeff", [0.5, -1.0, 2.3, 1j])
    def test_single_term_with_coefficient(self, coeff: complex) -> None:
        """Verify coefficients are preserved through conversion."""
        h = coeff * qm_o.Z(0)
        op = hamiltonian_to_quri_operator(h)
        # Single term, coefficient should match
        assert len(op) == 1
        for _, c in op.items():
            assert np.isclose(c, coeff)

    def test_identity_operator_in_term(self) -> None:
        """Verify Pauli.I operators within a term are skipped correctly."""
        h = qm_o.Hamiltonian()
        # Manually construct a term with I(0)*Z(1)
        h.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.I, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
            1.0,
        )
        op = hamiltonian_to_quri_operator(h)
        # I(0)*Z(1) should produce just Z(1), not a two-qubit term
        assert len(op) == 1

    def test_multi_qubit_product(self) -> None:
        """Verify X(0)*Z(1) produces one two-qubit term."""
        h = qm_o.X(0) * qm_o.Z(1)
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    def test_sum_of_terms(self) -> None:
        """Verify Z(0) + X(1) produces two separate terms."""
        h = qm_o.Z(0) + qm_o.X(1)
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 2

    def test_hamiltonian_with_constant_term(self) -> None:
        """Verify constant term is included as PAULI_IDENTITY."""
        from quri_parts.core.operator import PAULI_IDENTITY

        h = 1.5 * qm_o.Z(0)
        h.constant = 0.75
        op = hamiltonian_to_quri_operator(h)

        # Should have Z(0) term + identity (constant) term
        assert len(op) == 2
        assert np.isclose(op[PAULI_IDENTITY], 0.75)

    def test_zero_constant_is_excluded(self) -> None:
        """Verify zero constant term is NOT included in output."""
        h = qm_o.Z(0)
        h.constant = 0.0
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    def test_near_zero_constant_is_excluded(self) -> None:
        """Verify near-zero constant (below threshold) is excluded."""
        h = qm_o.Z(0)
        h.constant = 1e-16
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    def test_empty_hamiltonian(self) -> None:
        """Verify empty Hamiltonian produces empty Operator."""
        h = qm_o.Hamiltonian()
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 0

    def test_empty_hamiltonian_with_constant(self) -> None:
        """Verify empty Hamiltonian with only constant produces identity term."""
        from quri_parts.core.operator import PAULI_IDENTITY

        h = qm_o.Hamiltonian()
        h.constant = 3.14
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1
        assert np.isclose(op[PAULI_IDENTITY], 3.14)

    def test_complex_hamiltonian(self) -> None:
        """Verify a realistic Hamiltonian with mixed terms converts correctly."""
        h = 0.5 * qm_o.Z(0) * qm_o.Z(1) + 0.3 * qm_o.X(0) + 0.3 * qm_o.X(1)
        h.constant = -0.25
        op = hamiltonian_to_quri_operator(h)

        # Z0*Z1 + X0 + X1 + constant = 4 terms
        assert len(op) == 4

    @pytest.mark.parametrize("qubit_index", [0, 1, 5, 10])
    def test_various_qubit_indices(self, qubit_index: int) -> None:
        """Verify conversion works for various qubit index values."""
        h = qm_o.Z(qubit_index)
        op = hamiltonian_to_quri_operator(h)
        assert len(op) == 1

    def test_random_coefficients(self) -> None:
        """Verify conversion with random coefficients preserves values."""
        rng = np.random.default_rng(seed=42)
        coeffs = rng.uniform(-2.0, 2.0, size=3)

        h = coeffs[0] * qm_o.Z(0) + coeffs[1] * qm_o.X(1) + coeffs[2] * qm_o.Y(2)
        op = hamiltonian_to_quri_operator(h)

        # 3 distinct terms
        assert len(op) == 3

        # Verify total coefficient magnitude is preserved
        total_original = sum(abs(c) for c in coeffs)
        total_converted = sum(abs(c) for _, c in op.items())
        assert np.isclose(total_original, total_converted, atol=1e-12)


class TestToQuriOperatorAlias:
    """Tests for to_quri_operator convenience alias."""

    def test_alias_produces_same_result(self) -> None:
        """Verify to_quri_operator returns the same result as hamiltonian_to_quri_operator."""
        h = 0.5 * qm_o.Z(0) * qm_o.Z(1) + qm_o.X(0)
        h.constant = 1.0

        op1 = hamiltonian_to_quri_operator(h)
        op2 = to_quri_operator(h)

        assert len(op1) == len(op2)
        for label, coeff in op1.items():
            assert np.isclose(coeff, op2[label])
