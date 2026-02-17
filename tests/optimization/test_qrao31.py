"""Tests for QRAO31 (Quantum Random Access Optimization with (3,1,p)-QRAC)."""

import pytest
import numpy as np

from qamomile.optimization.qrao import QRAC31Converter, QRAC31Encoder, SignRounder
from qamomile.optimization.qrao.qrao31 import qrac31_encode_ising
from qamomile.optimization.qrao.graph_coloring import greedy_graph_coloring, check_linear_term
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType
import qamomile.observable as qm_o


class TestQRAC31Encoder:
    """Tests for QRAC31Encoder."""

    def test_simple_encoding(self):
        """Test encoding of a simple 3-variable problem."""
        # Create a simple Ising model: s0*s1 + s1*s2
        # This should use 2 qubits (variables 0,1 interact, 1,2 interact)
        x = binary(0)
        y = binary(1)
        z = binary(2)

        problem = BinaryExpr()
        problem += x * y
        problem += y * z

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC31Encoder(spin_model)

        # With graph coloring, interacting variables go to different qubits
        # So we need at least 2 qubits for this problem
        assert encoder.num_qubits >= 2

        # Each variable should be encoded
        assert len(encoder.pauli_encoding) == 3

        # Check that interacting variables are on different qubits
        q0, _ = encoder.pauli_encoding[0]
        q1, _ = encoder.pauli_encoding[1]
        q2, _ = encoder.pauli_encoding[2]
        assert q0 != q1  # 0 and 1 interact
        assert q1 != q2  # 1 and 2 interact

    def test_qubit_reduction(self):
        """Test that QRAC31 reduces qubit count for independent variables."""
        # Create a problem with 6 independent variables (no interactions)
        # Should be able to encode in 2 qubits (3 vars per qubit)
        problem = BinaryExpr()
        for i in range(6):
            problem += binary(i)  # Only linear terms

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC31Encoder(spin_model)

        # 6 variables with no interactions -> 2 qubits (3 per qubit)
        assert encoder.num_qubits == 2


class TestQRAC31Converter:
    """Tests for QRAC31Converter."""

    def test_converter_from_binary_model(self):
        """Test converter initialization from BinaryModel."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y
        problem += x

        model = BinaryModel(problem)
        converter = QRAC31Converter(model)

        assert converter.num_qubits > 0
        assert converter.spin_model is not None

class TestSignRounder:
    """Tests for SignRounder."""

    def test_basic_rounding(self):
        """Test basic sign-based rounding."""
        rounder = SignRounder()

        expectations = [0.8, -0.3, 0.0, -0.9, 0.5]
        spins = rounder.round(expectations)

        assert spins == [1, -1, 1, -1, 1]

    def test_boundary_cases(self):
        """Test rounding at boundary values."""
        rounder = SignRounder()

        # Exactly 0 should round to +1
        assert rounder.round([0.0]) == [1]

        # Values close to boundaries
        assert rounder.round([1.0]) == [1]
        assert rounder.round([-1.0]) == [-1]
        assert rounder.round([0.001]) == [1]
        assert rounder.round([-0.001]) == [-1]


class TestEndToEnd:
    """End-to-end tests for QRAO31."""

    def test_full_workflow(self):
        """Test the complete QRAO workflow without actual quantum execution."""
        # Create a simple MaxCut-like problem
        x = binary(0)
        y = binary(1)
        z = binary(2)

        # Minimize: -x*y - y*z (equivalent to MaxCut on a path graph)
        problem = BinaryExpr()
        problem += -1.0 * x * y
        problem += -1.0 * y * z

        model = BinaryModel(problem)
        converter = QRAC31Converter(model)

        # Check encoding
        assert converter.num_qubits >= 2
        assert len(converter.encoder.pauli_encoding) == 3

        # Simulate rounding results (as if from VQE)
        # For MaxCut, optimal is alternating: [+1, -1, +1] or [-1, +1, -1]
        rounder = SignRounder()
        mock_expectations = [0.9, -0.8, 0.7]  # Would give [+1, -1, +1]
        spins = rounder.round(mock_expectations)

        assert spins == [1, -1, 1]

        assert len(spins) == 3
        assert all(s in (1, -1) for s in spins)


class TestHUBORejection:
    """Tests for HUBO rejection in QRAC converters."""

    def test_hubo_rejection(self):
        """QRAC converters reject higher-order (HUBO) problems."""
        model = BinaryModel.from_hubo({(0, 1, 2): 1.0})
        with pytest.raises(ValueError, match="higher-order"):
            QRAC31Converter(model)


class TestQRAC31EncodeIsingCoefficients:
    """Tests for exact Hamiltonian coefficient verification using qrac31_encode_ising."""

    def test_linear_term_with_graph_coloring(self):
        """Test exact coefficients for Ising model with linear terms added via graph coloring."""
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)

        # Ising model: J={(0,1):2.0, (0,2):1.0}, h={2:5.0, 3:2.0}, constant=6.0
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

        qrac_hamiltonian, encoding = qrac31_encode_ising(ising, color_group)
        num_terms = len(ising.linear) + len(ising.quad)

        # color_group: {0:[0], 1:[1,2], 2:[3]}
        # Occupancies: var 0→k=1, var 1→k=2, var 2→k=2, var 3→k=1
        expected_hamiltonian = {
            (Z0, Z1): np.sqrt(1) * np.sqrt(2) * 2.0,
            (Z0, X1): np.sqrt(1) * np.sqrt(2) * 1.0,
            (X1,): np.sqrt(2) * 5.0,
            (Z2,): np.sqrt(1) * 2.0,
        }
        assert len(qrac_hamiltonian.terms) == num_terms
        assert qrac_hamiltonian.num_qubits < ising.num_bits
        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian.terms == expected_hamiltonian

    def test_linear_term_with_extra_variables(self):
        """Test coefficients when linear-only variables fill multiple color groups."""
        X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
        X2 = qm_o.PauliOperator(qm_o.Pauli.X, 2)
        Y2 = qm_o.PauliOperator(qm_o.Pauli.Y, 2)
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
        Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)
        Z3 = qm_o.PauliOperator(qm_o.Pauli.Z, 3)

        ising = BinaryModel.from_ising(
            linear={2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
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

        qrac_hamiltonian, encoding = qrac31_encode_ising(ising, color_group)
        num_terms = len(ising.linear) + len(ising.quad)

        # color_group: {0:[0], 1:[1,2], 2:[3,4,5], 3:[6]}
        # Occupancies: var 0→k=1, var 1→k=2, var 2→k=2,
        #              var 3→k=3, var 4→k=3, var 5→k=3, var 6→k=1
        expected_hamiltonian = {
            (Z0, Z1): np.sqrt(1) * np.sqrt(2) * 2.0,
            (Z0, X1): np.sqrt(1) * np.sqrt(2) * 1.0,
            (X1,): np.sqrt(2) * 5.0,
            (Z2,): np.sqrt(3) * 2.0,
            (X2,): np.sqrt(3) * 1.0,
            (Y2,): np.sqrt(3) * 1.0,
            (Z3,): np.sqrt(1) * 1.0,
        }
        assert len(qrac_hamiltonian.terms) == num_terms
        assert qrac_hamiltonian.num_qubits < ising.num_bits
        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian.terms == expected_hamiltonian

    def test_no_quadratic_terms(self):
        """Test encoding with only linear terms (no quadratic interactions)."""
        Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
        X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
        Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
        Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)

        ising = BinaryModel.from_ising(
            linear={0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0},
            quad={},
            constant=6.0,
        )

        max_color_group_size = 3
        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = check_linear_term(
            color_group, list(ising.linear.keys()), max_color_group_size
        )

        qrac_hamiltonian, encoding = qrac31_encode_ising(ising, color_group)
        num_terms = len(ising.linear) + len(ising.quad)

        # color_group: {0:[0,1,2], 1:[3]}
        # Occupancies: vars 0,1,2→k=3, var 3→k=1
        expected_hamiltonian = {
            (Z0,): np.sqrt(3) * 1.0,
            (X0,): np.sqrt(3) * 1.0,
            (Y0,): np.sqrt(3) * 5.0,
            (Z1,): np.sqrt(1) * 2.0,
        }
        assert len(qrac_hamiltonian.terms) == num_terms
        assert qrac_hamiltonian.num_qubits < ising.num_bits
        assert len(encoding) == ising.num_bits
        assert qrac_hamiltonian.terms == expected_hamiltonian
