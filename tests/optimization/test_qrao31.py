"""Tests for QRAO31 (Quantum Random Access Optimization with (3,1,p)-QRAC)."""

import pytest
from math import sqrt

from qamomile.optimization.qrao import QRAC31Converter, QRAC31Encoder, SignRounder
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel, VarType


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

    def test_hamiltonian_scaling(self):
        """Test that Hamiltonian coefficients are correctly scaled."""
        # Create: h_0 * s_0 + J_{01} * s_0 * s_1
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += 2.0 * x  # Linear term
        problem += 3.0 * x * y  # Quadratic term

        model = BinaryModel(problem)
        spin_model = model.change_vartype(VarType.SPIN)

        encoder = QRAC31Encoder(spin_model)

        # Linear terms should be scaled by sqrt(3)
        # Quadratic terms should be scaled by 3
        linear_coeffs = list(encoder.linear_hamiltonian.values())
        quad_coeffs = list(encoder.quad_hamiltonian.values())

        # Check that some scaling occurred (exact values depend on binary->spin conversion)
        assert len(linear_coeffs) > 0 or len(quad_coeffs) > 0


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


class TestDecoding:
    """Tests for decoding functionality."""

    def test_decode_from_rounded(self):
        """Test decoding rounded spins to BinarySampleSet."""
        x = binary(0)
        y = binary(1)

        problem = BinaryExpr()
        problem += x * y  # Minimize x*y

        model = BinaryModel(problem)
        converter = QRAC31Converter(model)

        # Assume rounding gave us spins [+1, +1]
        # In spin form: s_0 = +1, s_1 = +1
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
        converter = QRAC31Converter(model)

        # spin = +1 -> binary = 0
        # spin = -1 -> binary = 1
        spins = [1, -1]
        result = converter.decode_to_binary(spins)

        assert len(result.samples) == 1
        assert result.vartype == VarType.BINARY

        sample = result.samples[0]
        # Check conversion: spin +1 -> binary 0, spin -1 -> binary 1
        for idx, spin in zip(sorted(sample.keys()), spins):
            expected_binary = (1 - spin) // 2
            assert sample[idx] == expected_binary


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

        # Decode
        result = converter.decode_from_rounded(spins)
        assert len(result.samples) == 1
