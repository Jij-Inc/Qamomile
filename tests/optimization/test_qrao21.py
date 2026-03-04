"""Tests for QRAO21 (Quantum Random Access Optimization with (2,1,p)-QRAC)."""

import pytest
import numpy as np
import networkx as nx

from qamomile.optimization.qrao import QRAC21Converter, QRAC21Encoder, SignRounder
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

    def test_pauli_types(self):
        """Test that only Z and X Paulis are used (not Y)."""
        problem = BinaryExpr()
        for i in range(4):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC21Encoder(spin_model)

        for _, pauli_type in encoder.pauli_encoding.values():
            assert pauli_type in ("Z", "X")


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

        # x*y + x → spin: 2 bits, 1 quad, 2 linear, constant=0.75
        assert converter.num_qubits == 2
        assert converter.spin_model.num_bits == 2
        assert len(converter.spin_model.quad) == 1
        assert len(converter.spin_model.linear) == 2

    def test_cost_hamiltonian(self):
        """Test cost Hamiltonian generation."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC21Converter(model)

        hamiltonian = converter.get_cost_hamiltonian()
        # x*y → spin: 1 quad + 2 linear = 3 terms, constant=0.25
        assert len(hamiltonian.terms) == 3
        assert hamiltonian.constant == pytest.approx(0.25)
        assert hamiltonian.num_qubits == 2

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
        # Each entry should be a non-trivial Hamiltonian with terms
        for obs in pauli_list:
            assert len(obs.terms) > 0


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

        # Path graph (0-1-2): 3 vars, 2 interactions → 2 qubits for QRAC21
        assert converter.num_qubits == 2
        assert len(converter.encoder.pauli_encoding) == 3

        hamiltonian = converter.get_cost_hamiltonian()
        assert hamiltonian.constant == pytest.approx(-0.5)

        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]
        spins = rounder.round(mock_expectations)
        assert spins == [1, -1, 1]


class TestQRAC21EncodeIsingCoefficients:
    """Tests for exact Hamiltonian coefficient verification using encoder.encode_ising."""

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

        encoder = QRAC21Encoder(ising)
        qrac_hamiltonian, encoding = encoder.encode_ising(ising)
        num_terms = len(ising.linear) + len(ising.quad)

        # color_group: {0:[0], 1:[1,2], 2:[3,4], 3:[5,6]}
        # Occupancies: var 0→k=1, vars 1-6→k=2
        expected_hamiltonian = {
            (Z0, Z1): np.sqrt(1) * np.sqrt(2) * 2.0,
            (Z0, X1): np.sqrt(1) * np.sqrt(2) * 1.0,
            (X1,): np.sqrt(2) * 5.0,
            (Z2,): np.sqrt(2) * 2.0,
            (X2,): np.sqrt(2) * 1.0,
            (Z3,): np.sqrt(2) * 1.0,
            (X3,): np.sqrt(2) * 1.0,
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

        encoder = QRAC21Encoder(ising)
        qrac_hamiltonian, encoding = encoder.encode_ising(ising)
        num_terms = len(ising.linear) + len(ising.quad)

        max_color_group_size = 2
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


class TestQRAC21RandomGraphs:
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

        converter = QRAC21Converter(ising)
        hamiltonian = converter.get_cost_hamiltonian()

        # Structural invariants
        assert hamiltonian.num_qubits == converter.num_qubits
        assert converter.num_qubits <= n
        assert len(converter.encoder.pauli_encoding) == n
        assert len(hamiltonian.terms) <= len(linear) + len(quad)
        assert len(hamiltonian.terms) > 0

        # Pauli types must be Z or X only for (2,1,p)-QRAC
        for _, pauli_type in converter.encoder.pauli_encoding.values():
            assert pauli_type in ("Z", "X")

        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == n
