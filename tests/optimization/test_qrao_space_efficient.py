"""Tests for Space Efficient QRAC (Quantum Random Access Optimization)."""

import pytest
import numpy as np

from qamomile.optimization.qrao import (
    QRACSpaceEfficientConverter,
    QRACSpaceEfficientEncoder,
    SignRounder,
)
from qamomile.optimization.qrao.qrao_space_efficient import (
    numbering_space_efficient_encode,
    qrac_space_efficient_encode_ising,
)
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType
import qamomile.observable as qm_o


class TestSpaceEfficientEncoder:
    """Tests for QRACSpaceEfficientEncoder."""

    def test_sequential_numbering(self):
        """Test that variables are numbered sequentially."""
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRACSpaceEfficientEncoder(spin_model)

        # Variable 0 -> qubit 0, X
        # Variable 1 -> qubit 0, Y
        # Variable 2 -> qubit 1, X
        # Variable 3 -> qubit 1, Y
        assert encoder.pauli_encoding[0] == (0, 'X')
        assert encoder.pauli_encoding[1] == (0, 'Y')
        assert encoder.pauli_encoding[2] == (1, 'X')
        assert encoder.pauli_encoding[3] == (1, 'Y')

    def test_num_qubits(self):
        """Test qubit count calculation."""
        # 4 vars -> 2 qubits
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRACSpaceEfficientEncoder(spin_model)
        assert encoder.num_qubits == 2

    def test_odd_num_vars(self):
        """Test with odd number of variables."""
        # 3 vars -> 2 qubits (ceil(3/2))
        problem = BinaryExpr()
        for i in range(3):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRACSpaceEfficientEncoder(spin_model)
        assert encoder.num_qubits == 2

    def test_pauli_types(self):
        """Test that only X and Y Paulis are used."""
        problem = BinaryExpr()
        for i in range(6):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRACSpaceEfficientEncoder(spin_model)

        for _, pauli_type in encoder.pauli_encoding.values():
            assert pauli_type in ('X', 'Y')


class TestSpaceEfficientEncoding:
    """Tests for encoding functions."""

    def test_numbering_encode(self):
        """Test the numbering-based encoding function."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y + y * z

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encode = numbering_space_efficient_encode(spin_model)

        assert encode[0].pauli == qm_o.Pauli.X
        assert encode[0].index == 0
        assert encode[1].pauli == qm_o.Pauli.Y
        assert encode[1].index == 0
        assert encode[2].pauli == qm_o.Pauli.X
        assert encode[2].index == 1

    def test_same_qubit_interaction(self):
        """Test that same-qubit interactions produce Z terms."""
        # Variables 0 and 1 share qubit 0
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        hamiltonian, encoded_ope = qrac_space_efficient_encode_ising(spin_model)
        assert hamiltonian is not None

    def test_different_qubit_interaction(self):
        """Test that different-qubit interactions produce P_i * P_j terms."""
        # Variables 0 and 2 are on different qubits (0 and 1)
        x = binary(0)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * z

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        hamiltonian, encoded_ope = qrac_space_efficient_encode_ising(spin_model)
        assert hamiltonian is not None


class TestSpaceEfficientConverter:
    """Tests for QRACSpaceEfficientConverter."""

    def test_converter_from_binary_model(self):
        """Test converter initialization from BinaryModel."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y
        problem += x

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)

        assert converter.spin_model is not None

    def test_cost_hamiltonian(self):
        """Test cost Hamiltonian generation."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y
        problem += y * z

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)

        hamiltonian = converter.get_cost_hamiltonian()
        assert hamiltonian is not None
        assert converter.num_qubits > 0

    def test_num_qubits_after_encoding(self):
        """Test num_qubits is set after get_cost_hamiltonian."""
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)
        # Add interaction to make it non-trivial
        problem += binary(0) * binary(1)

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)
        converter.get_cost_hamiltonian()

        # 4 variables -> 2 qubits with 2:1 compression
        assert converter.num_qubits >= 2

    def test_encoded_pauli_list(self):
        """Test encoded Pauli list generation."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)
        converter.get_cost_hamiltonian()

        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == converter.spin_model.num_bits


class TestSpaceEfficientDecoding:
    """Tests for decoding functionality."""

    def test_decode_from_rounded(self):
        """Test decoding rounded spins to BinarySampleSet."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)

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
        converter = QRACSpaceEfficientConverter(model)

        spins = [1, -1]
        result = converter.decode_to_binary(spins)

        assert len(result.samples) == 1
        assert result.vartype == VarType.BINARY

        sample = result.samples[0]
        for idx, spin in zip(sorted(sample.keys()), spins):
            expected_binary = (1 - spin) // 2
            assert sample[idx] == expected_binary


class TestSpaceEfficientEndToEnd:
    """End-to-end tests for Space Efficient QRAC."""

    def test_full_workflow(self):
        """Test the complete workflow without actual quantum execution."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += -1.0 * x * y
        problem += -1.0 * y * z

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)

        hamiltonian = converter.get_cost_hamiltonian()
        assert hamiltonian is not None
        assert converter.num_qubits >= 1

        assert len(converter.encoder.pauli_encoding) == 3

        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]
        spins = rounder.round(mock_expectations)
        assert spins == [1, -1, 1]

        result = converter.decode_from_rounded(spins)
        assert len(result.samples) == 1


class TestSpaceEfficientEncodeIsingCoefficients:
    """Tests for exact Hamiltonian coefficient verification."""

    def test_numbering_encode_exact(self):
        """Test exact encoding map from numbering_space_efficient_encode."""
        ising = BinaryModel.from_ising(
            linear={2: 5.0, 3: 2.0},
            quad={(0, 1): 2.0, (0, 2): 1.0},
            constant=6.0,
        )
        encoding = numbering_space_efficient_encode(ising)
        expected_encoding = {
            0: qm_o.PauliOperator(qm_o.Pauli.X, 0),
            1: qm_o.PauliOperator(qm_o.Pauli.Y, 0),
            2: qm_o.PauliOperator(qm_o.Pauli.X, 1),
            3: qm_o.PauliOperator(qm_o.Pauli.Y, 1),
        }
        assert encoding == expected_encoding

    def test_encode_ising_exact_coefficients(self):
        """Test exact Hamiltonian coefficients for space-efficient encoding."""
        ising = BinaryModel.from_ising(
            linear={2: 5.0, 3: 2.0},
            quad={(0, 1): 2.0, (0, 2): 1.0},
            constant=6.0,
        )

        expected_hamiltonian = qm_o.Hamiltonian()
        expected_hamiltonian.constant = 6.0

        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.X, 1),), np.sqrt(3) * 5.0
        )
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.Y, 1),), np.sqrt(3) * 2.0
        )
        # (0,2) interaction: different qubits -> 3 * coeff * P_i * P_j
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.X, 1)),
            3 * 1.0,
        )
        # (0,1) interaction: same qubit -> sqrt(3) * coeff * Z
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.Z, 0),), np.sqrt(3) * 2.0
        )

        expected_encoding = {
            0: qm_o.PauliOperator(qm_o.Pauli.X, 0),
            1: qm_o.PauliOperator(qm_o.Pauli.Y, 0),
            2: qm_o.PauliOperator(qm_o.Pauli.X, 1),
            3: qm_o.PauliOperator(qm_o.Pauli.Y, 1),
        }

        hamiltonian, encoding = qrac_space_efficient_encode_ising(ising)

        assert hamiltonian == expected_hamiltonian
        assert encoding == expected_encoding
