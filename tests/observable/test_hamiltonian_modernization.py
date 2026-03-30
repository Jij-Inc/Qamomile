"""
Tests for modernized Hamiltonian features.

This test suite validates:
- Factory methods (zero, identity, single_pauli)
- Iterator protocol (__iter__, __len__)
- Qubit remapping (remap_qubits)
- Pauli multiplication table
- Backward compatibility
"""

import pytest

from qamomile.observable.hamiltonian import (
    Hamiltonian,
    Pauli,
    PauliOperator,
    X,
    Y,
    Z,
    multiply_pauli_same_qubit,
    simplify_pauliop_terms,
)


class TestFactoryMethods:
    """Test factory methods for creating common Hamiltonians."""

    def test_zero_hamiltonian(self):
        """Test creating a zero Hamiltonian."""
        h = Hamiltonian.zero()
        assert len(h) == 0
        assert h.constant == 0.0
        assert h.num_qubits == 0

    def test_zero_hamiltonian_with_num_qubits(self):
        """Test zero Hamiltonian with specified number of qubits."""
        h = Hamiltonian.zero(num_qubits=5)
        assert len(h) == 0
        assert h.constant == 0.0
        assert h.num_qubits == 5

    def test_identity_hamiltonian(self):
        """Test creating an identity Hamiltonian."""
        h = Hamiltonian.identity()
        assert len(h) == 0
        assert h.constant == 1.0

    def test_identity_hamiltonian_with_coeff(self):
        """Test identity Hamiltonian with custom coefficient."""
        h = Hamiltonian.identity(coeff=2.5)
        assert len(h) == 0
        assert h.constant == 2.5

    def test_identity_hamiltonian_complex_coeff(self):
        """Test identity Hamiltonian with complex coefficient."""
        h = Hamiltonian.identity(coeff=1.0 + 2.0j)
        assert len(h) == 0
        assert h.constant == 1.0 + 2.0j

    def test_single_pauli_x(self):
        """Test creating a single Pauli X term."""
        h = Hamiltonian.single_pauli(Pauli.X, 0)
        assert len(h) == 1
        term_list = list(h.terms.items())
        ops, coeff = term_list[0]
        assert len(ops) == 1
        assert ops[0].pauli == Pauli.X
        assert ops[0].index == 0
        assert coeff == 1.0

    def test_single_pauli_y_with_coeff(self):
        """Test creating a single Pauli Y term with coefficient."""
        h = Hamiltonian.single_pauli(Pauli.Y, 3, coeff=1.5)
        assert len(h) == 1
        term_list = list(h.terms.items())
        ops, coeff = term_list[0]
        assert len(ops) == 1
        assert ops[0].pauli == Pauli.Y
        assert ops[0].index == 3
        assert coeff == 1.5

    def test_single_pauli_z_complex(self):
        """Test creating a single Pauli Z term with complex coefficient."""
        h = Hamiltonian.single_pauli(Pauli.Z, 2, coeff=1j)
        assert len(h) == 1
        term_list = list(h.terms.items())
        ops, coeff = term_list[0]
        assert len(ops) == 1
        assert ops[0].pauli == Pauli.Z
        assert ops[0].index == 2
        assert coeff == 1j


class TestIteratorProtocol:
    """Test iterator protocol implementation."""

    def test_iteration(self):
        """Test iterating over Hamiltonian terms."""
        h = X(0) + Y(1) + Z(2)
        terms = list(h)
        assert len(terms) == 3

        # Check that all terms are present
        found_x = found_y = found_z = False
        for ops, coeff in terms:
            if len(ops) == 1:
                if ops[0].pauli == Pauli.X and ops[0].index == 0:
                    found_x = True
                    assert coeff == 1.0
                elif ops[0].pauli == Pauli.Y and ops[0].index == 1:
                    found_y = True
                    assert coeff == 1.0
                elif ops[0].pauli == Pauli.Z and ops[0].index == 2:
                    found_z = True
                    assert coeff == 1.0

        assert found_x and found_y and found_z

    def test_iteration_empty(self):
        """Test iterating over an empty Hamiltonian."""
        h = Hamiltonian.zero()
        terms = list(h)
        assert len(terms) == 0

    def test_len(self):
        """Test __len__ method."""
        h = Hamiltonian()
        assert len(h) == 0

        h.add_term((PauliOperator(Pauli.X, 0),), 1.0)
        assert len(h) == 1

        h.add_term((PauliOperator(Pauli.Y, 1),), 1.0)
        assert len(h) == 2

    def test_len_multi_qubit_terms(self):
        """Test __len__ with multi-qubit terms."""
        h = X(0) * Y(1) + Z(2) * X(3)
        assert len(h) == 2


class TestRemapQubits:
    """Test qubit remapping functionality."""

    def test_empty_map_returns_self(self):
        """Test that empty mapping returns self."""
        h = X(0) + Y(1)
        h_remapped = h.remap_qubits({})
        assert h_remapped is h

    def test_simple_remap(self):
        """Test simple qubit remapping."""
        h = X(0)
        h_remapped = h.remap_qubits({0: 5})

        assert len(h_remapped) == 1
        for ops, coeff in h_remapped:
            assert len(ops) == 1
            assert ops[0].pauli == Pauli.X
            assert ops[0].index == 5
            assert coeff == 1.0

    def test_remap_multi_qubit(self):
        """Test remapping multi-qubit terms."""
        h = X(0) * Y(1)
        h_remapped = h.remap_qubits({0: 3, 1: 7})

        assert len(h_remapped) == 1
        for ops, coeff in h_remapped:
            assert len(ops) == 2
            # Find X and Y operators
            pauli_indices = {op.pauli: op.index for op in ops}
            assert pauli_indices[Pauli.X] == 3
            assert pauli_indices[Pauli.Y] == 7
            assert coeff == 1.0

    def test_remap_multiple_terms(self):
        """Test remapping multiple terms."""
        h = X(0) + Y(1) + Z(2)
        h_remapped = h.remap_qubits({0: 10, 1: 11, 2: 12})

        assert len(h_remapped) == 3
        indices_found = set()
        for ops, coeff in h_remapped:
            assert len(ops) == 1
            indices_found.add(ops[0].index)
            assert coeff == 1.0

        assert indices_found == {10, 11, 12}

    def test_remap_preserves_coefficients(self):
        """Test that remapping preserves coefficients."""
        h = 2.5 * X(0) + 1.5j * Y(1)
        h_remapped = h.remap_qubits({0: 5, 1: 6})

        coeff_map = {}
        for ops, coeff in h_remapped:
            assert len(ops) == 1
            coeff_map[ops[0].pauli] = coeff

        assert coeff_map[Pauli.X] == 2.5
        assert coeff_map[Pauli.Y] == 1.5j

    def test_remap_preserves_constant(self):
        """Test that remapping preserves constant term."""
        h = X(0) + 3.0
        h_remapped = h.remap_qubits({0: 5})

        assert h_remapped.constant == 3.0
        assert len(h_remapped) == 1

    def test_remap_partial_mapping(self):
        """Test partial qubit mapping (some indices not in map)."""
        h = X(0) * Y(1) * Z(2)
        # Only remap qubit 1, leave 0 and 2 unchanged
        h_remapped = h.remap_qubits({1: 10})

        assert len(h_remapped) == 1
        for ops, coeff in h_remapped:
            indices = {op.index for op in ops}
            assert indices == {0, 10, 2}  # 0 and 2 unchanged, 1 → 10

    def test_remap_with_swap(self):
        """Test remapping that swaps qubit indices."""
        h = X(0) * Y(1)
        h_remapped = h.remap_qubits({0: 1, 1: 0})

        # After swap, we should still have X and Y, but on swapped indices
        assert len(h_remapped) == 1
        for ops, coeff in h_remapped:
            pauli_indices = {op.pauli: op.index for op in ops}
            assert pauli_indices[Pauli.X] == 1  # X was on 0, now on 1
            assert pauli_indices[Pauli.Y] == 0  # Y was on 1, now on 0


class TestPauliMultiplicationTable:
    """Test Pauli multiplication using the lookup table."""

    def test_xy_equals_iz(self):
        """Test X * Y = iZ."""
        X0 = PauliOperator(Pauli.X, 0)
        Y0 = PauliOperator(Pauli.Y, 0)
        result, phase = multiply_pauli_same_qubit(X0, Y0)

        assert result.pauli == Pauli.Z
        assert result.index == 0
        assert phase == 1j

    def test_yx_equals_minus_iz(self):
        """Test Y * X = -iZ."""
        Y0 = PauliOperator(Pauli.Y, 0)
        X0 = PauliOperator(Pauli.X, 0)
        result, phase = multiply_pauli_same_qubit(Y0, X0)

        assert result.pauli == Pauli.Z
        assert result.index == 0
        assert phase == -1j

    def test_pauli_squares(self):
        """Test that Pauli operators square to identity."""
        for pauli in [Pauli.X, Pauli.Y, Pauli.Z]:
            op = PauliOperator(pauli, 0)
            result, phase = multiply_pauli_same_qubit(op, op)
            assert result.pauli == Pauli.I
            assert result.index == 0
            assert phase == 1

    def test_identity_multiplication(self):
        """Test identity multiplication properties."""
        I0 = PauliOperator(Pauli.I, 0)
        X0 = PauliOperator(Pauli.X, 0)

        result, phase = multiply_pauli_same_qubit(I0, X0)
        assert result.pauli == Pauli.X
        assert result.index == 0
        assert phase == 1

        result, phase = multiply_pauli_same_qubit(X0, I0)
        assert result.pauli == Pauli.X
        assert result.index == 0
        assert phase == 1

    def test_different_qubits_raises_error(self):
        """Test that multiplying operators on different qubits raises error."""
        X0 = PauliOperator(Pauli.X, 0)
        Y1 = PauliOperator(Pauli.Y, 1)

        with pytest.raises(ValueError, match="different qubits"):
            multiply_pauli_same_qubit(X0, Y1)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_mutable_add_term(self):
        """Test that add_term still mutates the Hamiltonian."""
        h = Hamiltonian()
        assert len(h) == 0

        h.add_term((PauliOperator(Pauli.X, 0),), 1.0)
        assert len(h) == 1

        h.add_term((PauliOperator(Pauli.Y, 1),), 1.0)
        assert len(h) == 2

    def test_legacy_construction(self):
        """Test legacy construction pattern."""
        H = Hamiltonian()
        H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
        H.add_term((PauliOperator(Pauli.Z, 2),), 1.0)

        assert len(H) == 2
        assert H.num_qubits == 3

    def test_X_Y_Z_functions(self):
        """Test legacy X, Y, Z helper functions."""
        h_x = X(0)
        h_y = Y(1)
        h_z = Z(2)

        assert len(h_x) == 1
        assert len(h_y) == 1
        assert len(h_z) == 1

        # Test arithmetic still works
        h = h_x + h_y + h_z
        assert len(h) == 3

    def test_hamiltonian_arithmetic(self):
        """Test Hamiltonian arithmetic operations."""
        h1 = X(0) + Y(1)
        h2 = Z(0)

        # Addition
        h3 = h1 + h2
        assert len(h3) == 3

        # Scalar multiplication
        h4 = 2.0 * h1
        for ops, coeff in h4:
            assert coeff == 2.0

        # Subtraction
        h5 = h1 - h2
        assert len(h5) == 3

        # Negation
        h6 = -h1
        for ops, coeff in h6:
            assert coeff == -1.0

    def test_terms_property(self):
        """Test that terms property still works."""
        h = X(0) + Y(1)
        terms = h.terms
        assert isinstance(terms, dict)
        assert len(terms) == 2

    def test_constant_term(self):
        """Test constant term handling."""
        h = X(0) + 5.0
        assert h.constant == 5.0
        assert len(h) == 1

    def test_simplify_pauliop_terms(self):
        """Test simplify_pauliop_terms function."""
        X0 = PauliOperator(Pauli.X, 0)
        Y0 = PauliOperator(Pauli.Y, 0)
        Z1 = PauliOperator(Pauli.Z, 1)

        ops, phase = simplify_pauliop_terms((X0, Y0, Z1))

        # X0 * Y0 = iZ0, so we should get (Z0, Z1) with phase 1j
        assert phase == 1j
        assert len(ops) == 2
        paulis = {op.pauli for op in ops}
        indices = {op.index for op in ops}
        assert Pauli.Z in paulis
        assert 0 in indices
        assert 1 in indices


class TestModernFeatures:
    """Test that modern features integrate well."""

    def test_factory_methods_with_arithmetic(self):
        """Test factory methods work with arithmetic."""
        h_id = Hamiltonian.identity(2.0)
        h_x = Hamiltonian.single_pauli(Pauli.X, 0)

        h = h_id + h_x
        assert h.constant == 2.0
        assert len(h) == 1

    def test_iterator_with_complex_hamiltonian(self):
        """Test iterator with complex multi-term Hamiltonian."""
        h = X(0) * Y(1) + Z(0) * Z(1) + 2.0 * X(2)

        term_count = 0
        for ops, coeff in h:
            term_count += 1
            assert isinstance(ops, tuple)
            assert isinstance(coeff, (int, float, complex))

        assert term_count == 3

    def test_remap_with_factory_methods(self):
        """Test remapping works with factory-created Hamiltonians."""
        h = Hamiltonian.single_pauli(Pauli.X, 0, coeff=2.0)
        h_remapped = h.remap_qubits({0: 10})

        for ops, coeff in h_remapped:
            assert ops[0].index == 10
            assert coeff == 2.0
