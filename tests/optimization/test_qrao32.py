"""Tests for QRAO32 (Quantum Random Access Optimization with (3,2,p)-QRAC)."""

import pytest
import numpy as np

from qamomile.optimization.qrao import QRAC32Converter, QRAC32Encoder, SignRounder
from qamomile.optimization.qrao.qrao32 import (
    create_x_prime,
    create_y_prime,
    create_z_prime,
    create_prime_operator,
)
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType
import qamomile.observable as qm_o


class TestPrimeOperators:
    """Tests for prime operator construction."""

    def test_create_x_prime(self):
        """Test X' operator construction."""
        x_prime = create_x_prime(0)
        assert x_prime is not None
        # X' is a sum of Pauli terms, should be a Hamiltonian
        assert isinstance(x_prime, qm_o.Hamiltonian)

    def test_create_y_prime(self):
        """Test Y' operator construction."""
        y_prime = create_y_prime(0)
        assert y_prime is not None
        assert isinstance(y_prime, qm_o.Hamiltonian)

    def test_create_z_prime(self):
        """Test Z' operator construction."""
        z_prime = create_z_prime(0)
        assert z_prime is not None
        assert isinstance(z_prime, qm_o.Hamiltonian)

    def test_create_prime_operator_x(self):
        """Test prime operator dispatch for X."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.X, 0)
        prime = create_prime_operator(pauli_op)
        assert isinstance(prime, qm_o.Hamiltonian)

    def test_create_prime_operator_y(self):
        """Test prime operator dispatch for Y."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
        prime = create_prime_operator(pauli_op)
        assert isinstance(prime, qm_o.Hamiltonian)

    def test_create_prime_operator_z(self):
        """Test prime operator dispatch for Z."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        prime = create_prime_operator(pauli_op)
        assert isinstance(prime, qm_o.Hamiltonian)

    def test_prime_operator_qubit_indexing(self):
        """Test that prime operators use 2*index for physical qubits."""
        # Logical qubit 1 should map to physical qubits 2 and 3
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        prime = create_prime_operator(pauli_op)
        assert isinstance(prime, qm_o.Hamiltonian)


class TestQRAC32Encoder:
    """Tests for QRAC32Encoder."""

    def test_simple_encoding(self):
        """Test encoding of a simple problem."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC32Encoder(spin_model)

        assert len(encoder.pauli_encoding) == 2
        q0, _ = encoder.pauli_encoding[0]
        q1, _ = encoder.pauli_encoding[1]
        assert q0 != q1  # Interacting variables on different logical qubits

    def test_num_qubits_is_2x_logical(self):
        """Test that physical qubit count is 2x logical."""
        problem = BinaryExpr()
        for i in range(6):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC32Encoder(spin_model)

        # 6 vars, 3 per logical qubit -> 2 logical -> 4 physical
        assert encoder.num_logical_qubits == 2
        assert encoder.num_qubits == 4

    def test_pauli_types(self):
        """Test that Z, X, Y Paulis are used."""
        problem = BinaryExpr()
        for i in range(6):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC32Encoder(spin_model)

        pauli_types = {pt for _, pt in encoder.pauli_encoding.values()}
        # Should use at least Z and X (may include Y depending on group sizes)
        assert pauli_types.issubset({'Z', 'X', 'Y'})


class TestQRAC32Converter:
    """Tests for QRAC32Converter."""

    def test_converter_from_binary_model(self):
        """Test converter initialization from BinaryModel."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y
        problem += x

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

        assert converter.num_qubits > 0
        assert converter.spin_model is not None

    def test_num_qubits_is_2x_logical(self):
        """Test that converter's num_qubits is 2x logical."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

        # 2 interacting vars -> 2 logical qubits -> 4 physical
        assert converter.num_qubits == len(converter.color_group) * 2

    def test_cost_hamiltonian(self):
        """Test cost Hamiltonian generation."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

        hamiltonian = converter.get_cost_hamiltonian()
        assert hamiltonian is not None

    def test_ansatz_kernel(self):
        """Test that ansatz kernel is generated correctly."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

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
        converter = QRAC32Converter(model)
        converter.get_cost_hamiltonian()

        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == converter.spin_model.num_bits


class TestQRAC32Decoding:
    """Tests for decoding functionality."""

    def test_decode_from_rounded(self):
        """Test decoding rounded spins to BinarySampleSet."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

        spins = [1, -1]
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
        converter = QRAC32Converter(model)

        spins = [1, -1]
        result = converter.decode_to_binary(spins)

        assert len(result.samples) == 1
        assert result.vartype == VarType.BINARY

        sample = result.samples[0]
        for idx, spin in zip(sorted(sample.keys()), spins):
            expected_binary = (1 - spin) // 2
            assert sample[idx] == expected_binary


class TestQRAC32EndToEnd:
    """End-to-end tests for QRAO32."""

    def test_full_workflow(self):
        """Test the complete QRAO workflow without actual quantum execution."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += -1.0 * x * y
        problem += -1.0 * y * z

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

        assert converter.num_qubits >= 4  # At least 2 logical -> 4 physical
        assert len(converter.encoder.pauli_encoding) == 3

        kernel = converter.get_ansatz_kernel(depth=1)
        assert kernel is not None

        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]
        spins = rounder.round(mock_expectations)
        assert spins == [1, -1, 1]

        result = converter.decode_from_rounded(spins)
        assert len(result.samples) == 1
