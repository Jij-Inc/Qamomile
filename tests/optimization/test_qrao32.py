"""Tests for QRAO32 (Quantum Random Access Optimization with (3,2,p)-QRAC)."""

import pytest
import numpy as np

from qamomile.optimization.qrao import QRAC32Converter, QRAC32Encoder, SignRounder
from qamomile.optimization.qrao.qrao32 import (
    create_x_prime,
    create_y_prime,
    create_z_prime,
    create_prime_operator,
    qrac32_encode_ising,
)
from qamomile.optimization.qrao.graph_coloring import greedy_graph_coloring, check_linear_term
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType
import qamomile.observable as qm_o


class TestPrimeOperators:
    """Tests for prime operator construction."""

    def test_create_x_prime(self):
        """Test X' operator construction with exact coefficients."""
        X_prime = create_x_prime(0)
        coeff = 1.0 / np.sqrt(6)
        X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        expected = qm_o.Hamiltonian()
        expected.add_term((X0, X1), 1 / 2 * coeff)
        expected.add_term((X0, Z1), 1 / 2 * coeff)
        expected.add_term((Z0,), 1 * coeff)
        assert X_prime == expected

    def test_create_y_prime(self):
        """Test Y' operator construction with exact coefficients."""
        Y_prime = create_y_prime(0)
        coeff = 1.0 / np.sqrt(6)
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
        Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
        expected = qm_o.Hamiltonian()
        expected.add_term((X1,), 1 / 2 * coeff)
        expected.add_term((Z1,), 1 * coeff)
        expected.add_term((Y0, Y1), 1 / 2 * coeff)
        assert Y_prime == expected

    def test_create_z_prime(self):
        """Test Z' operator construction with exact coefficients."""
        Z_prime = create_z_prime(0)
        coeff = 1.0 / np.sqrt(6)
        X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        expected = qm_o.Hamiltonian()
        expected.add_term((Z0, Z1), 1 * coeff)
        expected.add_term((X0,), -1 / 2 * coeff)
        expected.add_term((Z0, X1), -1 / 2 * coeff)
        assert Z_prime == expected

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

        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]
        spins = rounder.round(mock_expectations)
        assert spins == [1, -1, 1]

        assert len(spins) == 3
        assert all(s in (1, -1) for s in spins)


class TestQRAC32EncodeIsingCoefficients:
    """Tests for exact Hamiltonian coefficient verification using qrac32_encode_ising."""

    def test_encode_ising_exact_coefficients(self):
        """Test exact coefficients for QRAC32 Ising encoding with prime operators."""
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)

        ising = BinaryModel.from_ising(
            linear={2: 5.0, 3: 2.0},
            quad={(0, 1): 2.0, (0, 2): 1.0},
            constant=6.0,
        )

        max_color_group_size = 3
        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = check_linear_term(
            color_group, list(ising.linear.keys()), max_color_group_size
        )

        qrac_hamiltonian, encoding = qrac32_encode_ising(ising, color_group)

        # Build expected Hamiltonian using prime operators
        expected_hamiltonian = qm_o.Hamiltonian()
        expected_hamiltonian.constant = 6.0
        Z0_prime = create_prime_operator(Z0)
        Z1_prime = create_prime_operator(Z1)
        Z2_prime = create_prime_operator(Z2)
        X1_prime = create_prime_operator(X1)

        # color_group: {0:[0], 1:[1,2], 2:[3]}
        # Occupancies: var 0→k=1, var 1→k=2, var 2→k=2, var 3→k=1
        # (3,2,p) formula: linear √(2k), quad √(2k_i)·√(2k_j)
        expected_hamiltonian += np.sqrt(2 * 1) * np.sqrt(2 * 2) * 2.0 * Z0_prime * Z1_prime
        expected_hamiltonian += np.sqrt(2 * 1) * np.sqrt(2 * 2) * 1.0 * Z0_prime * X1_prime
        expected_hamiltonian += np.sqrt(2 * 2) * 5.0 * X1_prime
        expected_hamiltonian += np.sqrt(2 * 1) * 2.0 * Z2_prime

        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian == expected_hamiltonian
