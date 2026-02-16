"""Tests for QRAO21 (Quantum Random Access Optimization with (2,1,p)-QRAC)."""

import pytest
import numpy as np
from math import sqrt

from qamomile.optimization.qrao import QRAC21Converter, QRAC21Encoder, SignRounder
from qamomile.optimization.qrao.qrao21 import qrac21_encode_ising
from qamomile.optimization.qrao.qrao31 import qrac31_encode_ising
from qamomile.optimization.qrao.graph_coloring import greedy_graph_coloring, check_linear_term
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType
import qamomile.observable as qm_o


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


class TestQRAC21EncodeIsingCoefficients:
    """Tests for exact Hamiltonian coefficient verification using qrac21_encode_ising."""

    def test_linear_term_with_graph_coloring(self):
        """Test exact coefficients for QRAC21 Ising encoding."""
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        X2 = qm_o.PauliOperator(qm_o.Pauli.X, 2)
        X3 = qm_o.PauliOperator(qm_o.Pauli.X, 3)
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)
        Z3 = qm_o.PauliOperator(qm_o.Pauli.Z, 3)

        ising = BinaryModel.from_ising(
            linear={2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
            quad={(0, 1): 2.0, (0, 2): 1.0},
            constant=6.0,
        )

        max_color_group_size = 2
        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = check_linear_term(
            color_group, list(ising.linear.keys()), max_color_group_size
        )

        qrac_hamiltonian, encoding = qrac21_encode_ising(ising, color_group)
        num_terms = len(ising.linear) + len(ising.quad)

        expected_hamiltonian = {
            (Z0, Z1): max_color_group_size * 2.0,
            (Z0, X1): max_color_group_size * 1.0,
            (X1,): np.sqrt(max_color_group_size) * 5.0,
            (Z2,): np.sqrt(max_color_group_size) * 2.0,
            (X2,): np.sqrt(max_color_group_size) * 1.0,
            (Z3,): np.sqrt(max_color_group_size) * 1.0,
            (X3,): np.sqrt(max_color_group_size) * 1.0,
        }
        assert len(qrac_hamiltonian.terms) == num_terms
        assert qrac_hamiltonian.num_qubits < ising.num_bits
        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian.terms == expected_hamiltonian

    def test_no_quadratic_terms(self):
        """Test QRAC21 encoding with only linear terms."""
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)

        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0},
            quad={},
            constant=6.0,
        )

        max_color_group_size = 2
        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = check_linear_term(
            color_group, list(ising.linear.keys()), max_color_group_size
        )

        qrac_hamiltonian, encoding = qrac21_encode_ising(ising, color_group)
        num_terms = len(ising.linear) + len(ising.quad)

        expected_hamiltonian = {
            (Z0,): np.sqrt(max_color_group_size) * 1.0,
            (X0,): np.sqrt(max_color_group_size) * 1.0,
            (Z1,): np.sqrt(max_color_group_size) * 5.0,
            (X1,): np.sqrt(max_color_group_size) * 2.0,
        }
        assert len(qrac_hamiltonian.terms) == num_terms
        assert qrac_hamiltonian.num_qubits < ising.num_bits
        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian.terms == expected_hamiltonian
