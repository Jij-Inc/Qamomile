"""Tests for QRAO21 (Quantum Random Access Optimization with (2,1,p)-QRAC)."""

import pytest
from math import sqrt

from qamomile.optimization.qrao import QRAC21Converter, QRAC21Encoder, SignRounder
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType


class TestQRAC21Encoder:
    """Tests for QRAC21Encoder."""

    def test_simple_encoding(self):
        """Test encoding of a simple 2-variable problem."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC21Encoder(spin_model)

        # Interacting variables must be on different qubits
        assert encoder.num_qubits >= 2
        assert len(encoder.pauli_encoding) == 2

        q0, _ = encoder.pauli_encoding[0]
        q1, _ = encoder.pauli_encoding[1]
        assert q0 != q1

    def test_qubit_reduction(self):
        """Test that QRAC21 reduces qubit count for independent variables."""
        # 4 independent variables -> 2 qubits (2 vars per qubit)
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC21Encoder(spin_model)
        assert encoder.num_qubits == 2

    def test_hamiltonian_scaling(self):
        """Test that Hamiltonian coefficients are correctly scaled."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += 2.0 * x
        problem += 3.0 * x * y

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC21Encoder(spin_model)

        # Check scaling factors
        assert encoder.linear_coeff_scale == pytest.approx(sqrt(2))
        assert encoder.quad_coeff_scale == pytest.approx(2.0)

        linear_coeffs = list(encoder.linear_hamiltonian.values())
        quad_coeffs = list(encoder.quad_hamiltonian.values())
        assert len(linear_coeffs) > 0 or len(quad_coeffs) > 0

    def test_pauli_types(self):
        """Test that only Z and X Paulis are used (not Y)."""
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC21Encoder(spin_model)

        for _, pauli_type in encoder.pauli_encoding.values():
            assert pauli_type in ('Z', 'X')


class TestQRAC21Converter:
    """Tests for QRAC21Converter."""

    def test_converter_from_binary_model(self):
        """Test converter initialization from BinaryModel."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y
        problem += x

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        assert converter.num_qubits > 0
        assert converter.spin_model is not None

    def test_cost_hamiltonian(self):
        """Test cost Hamiltonian generation."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        hamiltonian = converter.get_cost_hamiltonian()
        assert hamiltonian is not None

    def test_ansatz_kernel(self):
        """Test that ansatz kernel is generated correctly."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        kernel = converter.get_ansatz_kernel(depth=2)
        assert kernel is not None

        expected_params = converter.num_parameters(depth=2)
        assert expected_params == 2 * converter.num_qubits * 2

    def test_encoded_pauli_list(self):
        """Test encoded Pauli list generation."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)
        converter.get_cost_hamiltonian()

        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == converter.spin_model.num_bits


class TestQRAC21Decoding:
    """Tests for decoding functionality."""

    def test_decode_from_rounded(self):
        """Test decoding rounded spins to BinarySampleSet."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        spins = [1, 1]
        result = converter.decode_from_rounded(spins)

        assert len(result.samples) == 1
        assert result.vartype == VarType.SPIN

    def test_decode_to_binary(self):
        """Test conversion from spin to binary."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        spins = [1, -1]
        result = converter.decode_to_binary(spins)

        assert len(result.samples) == 1
        assert result.vartype == VarType.BINARY

        sample = result.samples[0]
        for idx, spin in zip(sorted(sample.keys()), spins):
            expected_binary = (1 - spin) // 2
            assert sample[idx] == expected_binary


class TestQRAC21EndToEnd:
    """End-to-end tests for QRAO21."""

    def test_full_workflow(self):
        """Test the complete QRAO workflow without actual quantum execution."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += -1.0 * x * y
        problem += -1.0 * y * z

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        assert converter.num_qubits >= 2
        assert len(converter.encoder.pauli_encoding) == 3

        kernel = converter.get_ansatz_kernel(depth=1)
        assert kernel is not None

        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]
        spins = rounder.round(mock_expectations)
        assert spins == [1, -1, 1]

        result = converter.decode_from_rounded(spins)
        assert len(result.samples) == 1
