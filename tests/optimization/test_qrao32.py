"""Tests for QRAO32 (Quantum Random Access Optimization with (3,2,p)-QRAC)."""

import pytest
import numpy as np
import networkx as nx

from qamomile.optimization.qrao import QRAC32Converter, QRAC32Encoder, SignRounder
from qamomile.optimization.qrao.qrao32 import (
    build_physical_qubit_map,
    create_x_prime,
    create_y_prime,
    create_z_prime,
    create_prime_operator,
    qrac32_encode_ising,
)
from qamomile.optimization.qrao.graph_coloring import (
    greedy_graph_coloring,
    check_linear_term,
)
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
        """Test prime operator dispatch for X matches create_x_prime."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.X, 0)
        prime = create_prime_operator(pauli_op, 0)
        assert prime == create_x_prime(0)

    def test_create_prime_operator_y(self):
        """Test prime operator dispatch for Y matches create_y_prime."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
        prime = create_prime_operator(pauli_op, 0)
        assert prime == create_y_prime(0)

    def test_create_prime_operator_z(self):
        """Test prime operator dispatch for Z matches create_z_prime."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        prime = create_prime_operator(pauli_op, 0)
        assert prime == create_z_prime(0)

    def test_prime_operator_qubit_indexing(self):
        """Test that prime operators use the given phys_start for physical qubits."""
        pauli_op = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        # Physical qubits should start at 2
        prime = create_prime_operator(pauli_op, 2)
        expected = create_z_prime(2)
        assert prime == expected
        # Verify a term key contains qubit indices 2 and 3
        has_correct_indices = any(
            any(op.index in (2, 3) for op in key) for key in prime.terms
        )
        assert has_correct_indices


class TestBuildPhysicalQubitMap:
    """Tests for build_physical_qubit_map utility."""

    def test_all_k1_groups(self):
        """All color groups have 1 variable -> 1 physical qubit each."""
        color_group = {0: [0], 1: [1], 2: [2]}
        mapping, total = build_physical_qubit_map(color_group)
        assert mapping == {0: 0, 1: 1, 2: 2}
        assert total == 3

    def test_all_k3_groups(self):
        """All color groups have 3 variables -> 2 physical qubits each."""
        color_group = {0: [0, 1, 2], 1: [3, 4, 5]}
        mapping, total = build_physical_qubit_map(color_group)
        assert mapping == {0: 0, 1: 2}
        assert total == 4

    def test_mixed_k1_and_k3(self):
        """Mixed k=1 and k=3 groups."""
        color_group = {0: [0], 1: [1, 2, 3], 2: [4]}
        mapping, total = build_physical_qubit_map(color_group)
        assert mapping == {0: 0, 1: 1, 2: 3}
        assert total == 4

    def test_k2_group(self):
        """k=2 group uses 2 physical qubits."""
        color_group = {0: [0, 1], 1: [2]}
        mapping, total = build_physical_qubit_map(color_group)
        assert mapping == {0: 0, 1: 2}
        assert total == 3


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

    def test_num_qubits_full_groups(self):
        """Test physical qubit count when all groups are fully populated."""
        problem = BinaryExpr()
        for i in range(6):
            problem += binary(i)

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC32Encoder(spin_model)

        # 6 vars, 3 per logical qubit -> 2 logical -> each k=3 -> 4 physical
        assert encoder.num_logical_qubits == 2
        assert encoder.num_qubits == 4

    def test_num_qubits_partial_groups(self):
        """Test physical qubit count with all k=1 groups."""
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y
        problem += z

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC32Encoder(spin_model)

        # x0 and x1 interact -> different groups, x2 also separate
        # color_group: {0:[0], 1:[1], 2:[2]}, all k=1 -> 1 phys qubit each
        assert encoder.num_logical_qubits == 3
        assert encoder.num_qubits == 3

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
        assert pauli_types.issubset({"Z", "X", "Y"})


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

        # x*y + x → spin: 2 bits, 1 quad, 2 linear
        # 2 vars interact → 2 color groups each with k=1 → 2 physical qubits
        assert converter.num_qubits == 2
        assert converter.spin_model.num_bits == 2
        assert len(converter.spin_model.quad) == 1
        assert len(converter.spin_model.linear) == 2

    def test_num_qubits_matches_physical_map(self):
        """Test that converter's num_qubits matches build_physical_qubit_map."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

        _, expected_total = build_physical_qubit_map(converter.color_group)
        assert converter.num_qubits == expected_total

    def test_cost_hamiltonian(self):
        """Test cost Hamiltonian generation."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y

        model = BinaryModel(problem)
        converter = QRAC32Converter(model)

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

        # Path graph 0-1-2: 3 color groups each with k=1 → 3 physical qubits
        assert converter.num_qubits == 3
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
        """Test exact coefficients for QRAC32 Ising encoding with mixed operators.

        color_group: {0:[0], 1:[1,2], 2:[3]}
        Physical mapping: color 0 -> phys 0 (k=1, 1 qubit),
                          color 1 -> phys 1,2 (k=2, 2 qubits),
                          color 2 -> phys 3 (k=1, 1 qubit)
        var 0 (k=1): regular Z on phys 0, scale=1
        var 1 (k=2): Z' prime on phys 1, scale=√(2·2)=2
        var 2 (k=2): X' prime on phys 1, scale=√(2·2)=2
        var 3 (k=1): regular Z on phys 3, scale=1
        """
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

        # Build expected Hamiltonian
        expected_hamiltonian = qm_o.Hamiltonian()
        expected_hamiltonian.constant = 6.0

        # var 0 (k=1): simple Z on physical qubit 0
        Z_phys0 = qm_o.Hamiltonian()
        Z_phys0.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)

        # var 1 (k=2): Z' prime starting at physical qubit 1
        Z_prime_1 = create_z_prime(1)
        # var 2 (k=2): X' prime starting at physical qubit 1
        X_prime_1 = create_x_prime(1)

        # var 3 (k=1): simple Z on physical qubit 3
        Z_phys3 = qm_o.Hamiltonian()
        Z_phys3.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 3),), 1.0)

        # quad (0,1): scale_0=1, scale_1=2, coeff=2.0
        expected_hamiltonian += 1.0 * 2.0 * 2.0 * Z_phys0 * Z_prime_1
        # quad (0,2): scale_0=1, scale_2=2, coeff=1.0
        expected_hamiltonian += 1.0 * 2.0 * 1.0 * Z_phys0 * X_prime_1
        # linear(2): scale=2, coeff=5.0
        expected_hamiltonian += 2.0 * 5.0 * X_prime_1
        # linear(3): scale=1, coeff=2.0
        expected_hamiltonian += 1.0 * 2.0 * Z_phys3

        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian == expected_hamiltonian


class TestQRAC32RandomGraphs:
    """Property-based tests with random Erdős–Rényi graphs."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_random_graph(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 15))
        G = nx.erdos_renyi_graph(n, 0.4, seed=seed)

        linear = {i: float(rng.uniform(-2, 2)) for i in range(n)}
        quad = {(u, v): float(rng.uniform(-2, 2)) for u, v in G.edges()}
        ising = BinaryModel.from_ising(linear=linear, quad=quad, constant=0.0)

        converter = QRAC32Converter(ising)
        hamiltonian = converter.get_cost_hamiltonian()

        # Structural invariants
        assert hamiltonian.num_qubits == converter.num_qubits
        # Physical qubits: at most 2 per logical group
        assert converter.num_qubits <= 2 * n
        assert converter.num_qubits >= converter.encoder.num_logical_qubits
        assert len(converter.encoder.pauli_encoding) == n
        # Note: QRAC32 prime operators are multi-term, so Hamiltonian
        # may have more terms than |linear|+|quad| after operator products
        assert len(hamiltonian.terms) > 0

        # Pauli types must be Z, X, or Y for (3,2,p)-QRAC
        for _, pauli_type in converter.encoder.pauli_encoding.values():
            assert pauli_type in ("Z", "X", "Y")

        pauli_list = converter.get_encoded_pauli_list()
        assert len(pauli_list) == n
