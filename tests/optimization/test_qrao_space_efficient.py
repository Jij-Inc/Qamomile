"""Tests for Space Efficient QRAC (Quantum Random Access Optimization)."""

import math

import pytest
import numpy as np
import networkx as nx

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
        assert encoder.pauli_encoding[0] == (0, "X")
        assert encoder.pauli_encoding[1] == (0, "Y")
        assert encoder.pauli_encoding[2] == (1, "X")
        assert encoder.pauli_encoding[3] == (1, "Y")

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
            assert pauli_type in ("X", "Y")


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
        # x*y → spin: 1 quad (same qubit → Z) + 2 linear = 3 terms, constant=0.25
        assert len(hamiltonian.terms) == 3
        assert hamiltonian.constant == pytest.approx(0.25)
        # Same-qubit interaction should produce a Z term
        has_z = any(
            any(op.pauli == qm_o.Pauli.Z for op in key) for key in hamiltonian.terms
        )
        assert has_z

    def test_different_qubit_interaction(self):
        """Test that different-qubit interactions produce P_i * P_j terms."""
        # Use Ising model directly: 3 vars, interaction between 0 and 2
        # var 0 → X on qubit 0, var 2 → X on qubit 1 (different qubits)
        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 1.0},
            quad={(0, 2): 1.0},
            constant=0.0,
        )

        hamiltonian, encoded_ope = qrac_space_efficient_encode_ising(ising)
        # 3 linear + 1 quad = 4 terms
        assert len(hamiltonian.terms) == 4
        assert hamiltonian.constant == pytest.approx(0.0)
        # Vars 0 and 2 are on different qubits → product term X0·X1
        has_2body = any(len(key) == 2 for key in hamiltonian.terms)
        assert has_2body


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

        # x*y + x → spin: 2 bits, 1 quad, 2 linear
        assert converter.spin_model.num_bits == 2
        assert len(converter.spin_model.quad) == 1
        assert len(converter.spin_model.linear) == 2

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
        # x*y + y*z → spin: 2 quad + 3 linear = 5 terms, constant=0.5
        assert converter.num_qubits == 2
        assert len(hamiltonian.terms) == 5
        assert hamiltonian.constant == pytest.approx(0.5)

    def test_num_qubits_before_get_cost_hamiltonian(self):
        """Test num_qubits is valid even before get_cost_hamiltonian."""
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)
        problem += binary(0) * binary(1)

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)

        # num_qubits should be correct immediately (delegated to encoder)
        assert converter.num_qubits == 2

    def test_num_qubits_after_get_cost_hamiltonian(self):
        """Test num_qubits is consistent after get_cost_hamiltonian."""
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)
        problem += binary(0) * binary(1)

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)
        converter.get_cost_hamiltonian()

        # 4 variables -> 2 qubits with 2:1 compression (always ceil(n/2))
        assert converter.num_qubits == 2

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

    def test_encoded_pauli_list_before_get_cost_hamiltonian(self):
        """Test get_encoded_pauli_list works before get_cost_hamiltonian.

        Regression test: previously _num_qubits was 0 until get_cost_hamiltonian
        was called, causing Hamiltonians with num_qubits=0.
        """
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y + y * z

        model = BinaryModel(problem)
        converter = QRACSpaceEfficientConverter(model)

        # Call get_encoded_pauli_list BEFORE get_cost_hamiltonian
        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == converter.spin_model.num_bits

        # Each Hamiltonian should have the correct num_qubits (2, not 0)
        for h in pauli_list:
            assert h.num_qubits == 2


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
        # -x*y - y*z → spin: 2 quad + 3 linear = 5 terms, constant=-0.5
        assert converter.num_qubits == 2
        assert len(hamiltonian.terms) == 5
        assert hamiltonian.constant == pytest.approx(-0.5)

        assert len(converter.encoder.pauli_encoding) == 3

        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]
        spins = rounder.round(mock_expectations)
        assert spins == [1, -1, 1]

        assert len(spins) == 3
        assert all(s in (1, -1) for s in spins)


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

        # All qubits encode 2 variables (k=2 for all)
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.X, 1),), np.sqrt(2) * 5.0
        )
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.Y, 1),), np.sqrt(2) * 2.0
        )
        # (0,2) interaction: different qubits -> √k_0 * √k_2 * coeff * P_i * P_j
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.X, 1)),
            np.sqrt(2) * np.sqrt(2) * 1.0,
        )
        # (0,1) interaction: same qubit -> √k_0 * √k_1 * coeff * Z
        expected_hamiltonian.add_term(
            (qm_o.PauliOperator(qm_o.Pauli.Z, 0),), np.sqrt(2) * np.sqrt(2) * 2.0
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


class TestSpaceEfficientRandomGraphs:
    """Property-based tests with random Erdős–Rényi graphs."""

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_random_graph(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 15))
        G = nx.erdos_renyi_graph(n, 0.4, seed=seed)

        linear = {i: float(rng.uniform(-2, 2)) for i in range(n)}
        quad = {(u, v): float(rng.uniform(-2, 2)) for u, v in G.edges()}
        ising = BinaryModel.from_ising(linear=linear, quad=quad, constant=0.0)

        converter = QRACSpaceEfficientConverter(ising)
        hamiltonian = converter.get_cost_hamiltonian()

        # Space-efficient always uses ceil(n/2) qubits
        expected_qubits = math.ceil(n / 2)
        assert converter.num_qubits == expected_qubits
        assert hamiltonian.num_qubits == expected_qubits

        assert len(converter.encoder.pauli_encoding) == n
        assert len(hamiltonian.terms) <= len(linear) + len(quad)
        assert len(hamiltonian.terms) > 0

        # Pauli types must be X or Y only for space-efficient encoding
        for _, pauli_type in converter.encoder.pauli_encoding.values():
            assert pauli_type in ("X", "Y")

        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == n
