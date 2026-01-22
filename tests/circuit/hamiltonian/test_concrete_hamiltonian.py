"""Tests for ConcreteHamiltonian and Pauli algebra."""

import pytest

from qamomile.circuit.observable.concrete import (
    ConcreteHamiltonian,
    PauliString,
    _canonicalize_pauli_string,
    _multiply_paulis_same_qubit,
)
from qamomile.circuit.ir.types.hamiltonian import PauliKind


class TestPauliMultiplication:
    """Test Pauli multiplication rules."""

    def test_pauli_squares_to_identity(self):
        """Test that P^2 = I for all Paulis."""
        for p in [PauliKind.X, PauliKind.Y, PauliKind.Z]:
            result, phase = _multiply_paulis_same_qubit(p, p)
            assert result == PauliKind.I
            assert phase == 1

    def test_identity_multiplication(self):
        """Test that I * P = P * I = P."""
        for p in [PauliKind.I, PauliKind.X, PauliKind.Y, PauliKind.Z]:
            result, phase = _multiply_paulis_same_qubit(PauliKind.I, p)
            assert result == p
            assert phase == 1

            result, phase = _multiply_paulis_same_qubit(p, PauliKind.I)
            assert result == p
            assert phase == 1

    def test_xy_equals_iz(self):
        """Test XY = iZ."""
        result, phase = _multiply_paulis_same_qubit(PauliKind.X, PauliKind.Y)
        assert result == PauliKind.Z
        assert phase == 1j

    def test_yx_equals_minus_iz(self):
        """Test YX = -iZ."""
        result, phase = _multiply_paulis_same_qubit(PauliKind.Y, PauliKind.X)
        assert result == PauliKind.Z
        assert phase == -1j

    def test_yz_equals_ix(self):
        """Test YZ = iX."""
        result, phase = _multiply_paulis_same_qubit(PauliKind.Y, PauliKind.Z)
        assert result == PauliKind.X
        assert phase == 1j

    def test_zx_equals_iy(self):
        """Test ZX = iY."""
        result, phase = _multiply_paulis_same_qubit(PauliKind.Z, PauliKind.X)
        assert result == PauliKind.Y
        assert phase == 1j


class TestCanonicalizePauliString:
    """Test Pauli string canonicalization."""

    def test_empty_string(self):
        """Test canonicalization of empty string."""
        result, phase = _canonicalize_pauli_string([])
        assert result == ()
        assert phase == 1

    def test_single_pauli(self):
        """Test single Pauli canonicalization."""
        result, phase = _canonicalize_pauli_string([(PauliKind.Z, 0)])
        assert result == ((PauliKind.Z, 0),)
        assert phase == 1

    def test_identity_removed(self):
        """Test that identity Paulis are removed."""
        result, phase = _canonicalize_pauli_string([(PauliKind.I, 0)])
        assert result == ()
        assert phase == 1

    def test_same_qubit_multiplication(self):
        """Test that same-qubit Paulis are multiplied."""
        # Z * Z = I (should be removed)
        result, phase = _canonicalize_pauli_string([
            (PauliKind.Z, 0),
            (PauliKind.Z, 0),
        ])
        assert result == ()
        assert phase == 1

    def test_xy_same_qubit(self):
        """Test X * Y on same qubit gives iZ."""
        result, phase = _canonicalize_pauli_string([
            (PauliKind.X, 0),
            (PauliKind.Y, 0),
        ])
        assert result == ((PauliKind.Z, 0),)
        assert phase == 1j

    def test_different_qubits_sorted(self):
        """Test that different qubits are sorted by index."""
        result, phase = _canonicalize_pauli_string([
            (PauliKind.X, 2),
            (PauliKind.Z, 0),
            (PauliKind.Y, 1),
        ])
        # Should be sorted by qubit index
        assert len(result) == 3
        assert result[0][1] == 0  # qubit 0 first
        assert result[1][1] == 1  # qubit 1 second
        assert result[2][1] == 2  # qubit 2 last
        assert phase == 1


class TestConcreteHamiltonian:
    """Test ConcreteHamiltonian class."""

    def test_zero_hamiltonian(self):
        """Test zero Hamiltonian creation."""
        h = ConcreteHamiltonian.zero()
        assert len(h) == 0
        assert h.num_qubits == 0

    def test_identity_hamiltonian(self):
        """Test identity Hamiltonian creation."""
        h = ConcreteHamiltonian.identity(2.0)
        assert len(h) == 1
        assert h.constant == 2.0

    def test_single_pauli(self):
        """Test single Pauli creation."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        assert len(h) == 1
        assert h.num_qubits == 1
        assert ((PauliKind.Z, 0),) in h.terms

    def test_addition(self):
        """Test Hamiltonian addition."""
        h1 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0)
        h2 = ConcreteHamiltonian.single_pauli(PauliKind.X, 1)
        h = h1 + h2

        assert len(h) == 2
        assert ((PauliKind.Z, 0),) in h.terms
        assert ((PauliKind.X, 1),) in h.terms

    def test_addition_same_term(self):
        """Test addition of same terms combines coefficients."""
        h1 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h2 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 2.0)
        h = h1 + h2

        assert len(h) == 1
        assert h.terms[((PauliKind.Z, 0),)] == 3.0

    def test_subtraction(self):
        """Test Hamiltonian subtraction."""
        h1 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 3.0)
        h2 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h = h1 - h2

        assert len(h) == 1
        assert h.terms[((PauliKind.Z, 0),)] == 2.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h_scaled = h * 2.0

        assert h_scaled.terms[((PauliKind.Z, 0),)] == 2.0

    def test_scalar_rmul(self):
        """Test reverse scalar multiplication."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h_scaled = 3.0 * h

        assert h_scaled.terms[((PauliKind.Z, 0),)] == 3.0

    def test_hamiltonian_multiplication(self):
        """Test Hamiltonian multiplication (tensor product)."""
        h1 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0)
        h2 = ConcreteHamiltonian.single_pauli(PauliKind.Z, 1)
        h = h1 * h2

        assert len(h) == 1
        # Should have Z_0 * Z_1 term
        expected_key = ((PauliKind.Z, 0), (PauliKind.Z, 1))
        assert expected_key in h.terms

    def test_negation(self):
        """Test Hamiltonian negation."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 2.0)
        h_neg = -h

        assert h_neg.terms[((PauliKind.Z, 0),)] == -2.0

    def test_num_qubits(self):
        """Test num_qubits calculation."""
        h = ConcreteHamiltonian()
        h = h.add_term([(PauliKind.Z, 0)], 1.0)
        h = h.add_term([(PauliKind.X, 2)], 1.0)
        h = h.add_term([(PauliKind.Y, 5)], 1.0)

        assert h.num_qubits == 6  # max index is 5, so 6 qubits

    def test_to_sparse_dict(self):
        """Test conversion to sparse dictionary."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h = h + ConcreteHamiltonian.single_pauli(PauliKind.X, 1, 0.5)

        sparse = h.to_sparse_dict()
        assert "ZI" in sparse
        assert "IX" in sparse
        assert sparse["ZI"] == 1.0
        assert sparse["IX"] == 0.5

    def test_add_term_pauli_algebra(self):
        """Test that add_term applies Pauli algebra."""
        h = ConcreteHamiltonian()
        # Add X_0 * Y_0 which should become iZ_0
        h = h.add_term([(PauliKind.X, 0), (PauliKind.Y, 0)], 1.0)

        assert len(h) == 1
        assert ((PauliKind.Z, 0),) in h.terms
        assert h.terms[((PauliKind.Z, 0),)] == 1j


class TestConcreteHamiltonianIterator:
    """Test iteration over ConcreteHamiltonian."""

    def test_iterate_terms(self):
        """Test iterating over Hamiltonian terms."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h = h + ConcreteHamiltonian.single_pauli(PauliKind.X, 1, 2.0)

        terms = list(h)
        assert len(terms) == 2

        # Check that all terms are (pauli_string, coeff) tuples
        for pauli_string, coeff in terms:
            assert isinstance(pauli_string, tuple)
            assert isinstance(coeff, (int, float, complex))


class TestRemapQubits:
    """Test qubit remapping functionality."""

    def test_remap_single_pauli(self):
        """Test remapping a single Pauli term."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h_remapped = h.remap_qubits({0: 5})

        assert len(h_remapped) == 1
        assert ((PauliKind.Z, 5),) in h_remapped.terms
        assert h_remapped.terms[((PauliKind.Z, 5),)] == 1.0

    def test_remap_two_qubit_hamiltonian(self):
        """Test remapping a two-qubit Hamiltonian."""
        # Z(0) * Z(1)
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0)
        h = h * ConcreteHamiltonian.single_pauli(PauliKind.Z, 1)

        # Remap: 0 -> 3, 1 -> 7
        h_remapped = h.remap_qubits({0: 3, 1: 7})

        assert len(h_remapped) == 1
        expected_key = ((PauliKind.Z, 3), (PauliKind.Z, 7))
        assert expected_key in h_remapped.terms

    def test_remap_preserves_coefficients(self):
        """Test that coefficients are preserved during remapping."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.X, 0, 2.5)
        h_remapped = h.remap_qubits({0: 10})

        assert h_remapped.terms[((PauliKind.X, 10),)] == 2.5

    def test_remap_empty_map_returns_same(self):
        """Test that empty mapping returns the same Hamiltonian."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h_remapped = h.remap_qubits({})

        # Should be the same object (optimization)
        assert h_remapped is h

    def test_remap_partial_mapping(self):
        """Test that unmapped qubits keep their original indices."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0)
        h = h + ConcreteHamiltonian.single_pauli(PauliKind.X, 1)

        # Only remap qubit 0, leave qubit 1 unchanged
        h_remapped = h.remap_qubits({0: 5})

        assert ((PauliKind.Z, 5),) in h_remapped.terms  # 0 -> 5
        assert ((PauliKind.X, 1),) in h_remapped.terms  # 1 unchanged

    def test_remap_multiple_terms(self):
        """Test remapping a Hamiltonian with multiple terms."""
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        h = h + ConcreteHamiltonian.single_pauli(PauliKind.X, 1, 0.5)
        h = h + ConcreteHamiltonian.identity(0.3)  # constant term

        h_remapped = h.remap_qubits({0: 2, 1: 3})

        assert len(h_remapped) == 3
        assert ((PauliKind.Z, 2),) in h_remapped.terms
        assert ((PauliKind.X, 3),) in h_remapped.terms
        assert () in h_remapped.terms  # constant term unchanged
        assert h_remapped.constant == 0.3

    def test_remap_with_index_swap(self):
        """Test remapping that swaps qubit indices."""
        # Z(0) * X(1)
        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0)
        h = h * ConcreteHamiltonian.single_pauli(PauliKind.X, 1)

        # Swap: 0 -> 1, 1 -> 0
        h_remapped = h.remap_qubits({0: 1, 1: 0})

        # Should now be X(0) * Z(1) (sorted by qubit index)
        assert len(h_remapped) == 1
        expected_key = ((PauliKind.X, 0), (PauliKind.Z, 1))
        assert expected_key in h_remapped.terms
