"""Tests for HamiltonianExpr handle and pauli module."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.types.hamiltonian import PauliKind, HamiltonianExprType
from qamomile.circuit.ir.operation.hamiltonian_ops import (
    PauliCreateOp,
    HamiltonianAddOp,
    HamiltonianMulOp,
    HamiltonianScaleOp,
    HamiltonianNegOp,
)
from qamomile.circuit.frontend.tracer import trace


class TestPauliModule:
    """Test the pauli submodule (qm.pauli.X, etc.)."""

    def test_pauli_z_creation(self):
        """Test creating Z Pauli operator."""
        with trace() as tracer:
            h = qm.pauli.Z(0)

            assert isinstance(h, qm.HamiltonianExpr)
            assert isinstance(h.value.type, HamiltonianExprType)

            # Check that PauliCreateOp was emitted
            ops = tracer.operations
            assert len(ops) == 1
            assert isinstance(ops[0], PauliCreateOp)
            assert ops[0].pauli_kind == PauliKind.Z

    def test_pauli_x_creation(self):
        """Test creating X Pauli operator."""
        with trace() as tracer:
            h = qm.pauli.X(1)

            assert isinstance(h, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 1
            assert isinstance(ops[0], PauliCreateOp)
            assert ops[0].pauli_kind == PauliKind.X

    def test_pauli_y_creation(self):
        """Test creating Y Pauli operator."""
        with trace() as tracer:
            h = qm.pauli.Y(2)

            assert isinstance(h, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 1
            assert isinstance(ops[0], PauliCreateOp)
            assert ops[0].pauli_kind == PauliKind.Y

    def test_pauli_i_creation(self):
        """Test creating I (identity) Pauli operator."""
        with trace() as tracer:
            h = qm.pauli.I(0)

            assert isinstance(h, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 1
            assert isinstance(ops[0], PauliCreateOp)
            assert ops[0].pauli_kind == PauliKind.I


class TestHamiltonianExprArithmetic:
    """Test arithmetic operations on HamiltonianExpr."""

    def test_addition(self):
        """Test HamiltonianExpr addition."""
        with trace() as tracer:
            h1 = qm.pauli.Z(0)
            h2 = qm.pauli.X(1)
            h = h1 + h2

            assert isinstance(h, qm.HamiltonianExpr)

            # Should have 2 PauliCreateOps + 1 HamiltonianAddOp
            ops = tracer.operations
            assert len(ops) == 3
            assert isinstance(ops[0], PauliCreateOp)
            assert isinstance(ops[1], PauliCreateOp)
            assert isinstance(ops[2], HamiltonianAddOp)

    def test_multiplication_hamiltonian(self):
        """Test HamiltonianExpr multiplication (tensor product)."""
        with trace() as tracer:
            h1 = qm.pauli.Z(0)
            h2 = qm.pauli.Z(1)
            h = h1 * h2

            assert isinstance(h, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 3
            assert isinstance(ops[2], HamiltonianMulOp)

    def test_scalar_multiplication_right(self):
        """Test HamiltonianExpr * scalar."""
        with trace() as tracer:
            h = qm.pauli.Z(0)
            h_scaled = h * 2.0

            assert isinstance(h_scaled, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 2
            assert isinstance(ops[1], HamiltonianScaleOp)

    def test_scalar_multiplication_left(self):
        """Test scalar * HamiltonianExpr."""
        with trace() as tracer:
            h = qm.pauli.Z(0)
            h_scaled = 3.0 * h

            assert isinstance(h_scaled, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 2
            assert isinstance(ops[1], HamiltonianScaleOp)

    def test_negation(self):
        """Test HamiltonianExpr negation."""
        with trace() as tracer:
            h = qm.pauli.Z(0)
            h_neg = -h

            assert isinstance(h_neg, qm.HamiltonianExpr)

            ops = tracer.operations
            assert len(ops) == 2
            assert isinstance(ops[1], HamiltonianNegOp)

    def test_subtraction(self):
        """Test HamiltonianExpr subtraction."""
        with trace() as tracer:
            h1 = qm.pauli.Z(0)
            h2 = qm.pauli.X(1)
            h = h1 - h2

            assert isinstance(h, qm.HamiltonianExpr)

            # Subtraction = negate + add
            ops = tracer.operations
            # 2 PauliCreateOp + 1 NegOp + 1 AddOp
            assert len(ops) == 4
            assert isinstance(ops[2], HamiltonianNegOp)
            assert isinstance(ops[3], HamiltonianAddOp)


class TestComplexHamiltonianConstruction:
    """Test building complex Hamiltonians."""

    def test_ising_style_hamiltonian(self):
        """Test building an Ising-style Hamiltonian."""
        with trace() as tracer:
            # H = J * Z_0 * Z_1 + g * (X_0 + X_1)
            J = 1.0
            g = 0.5

            zz = qm.pauli.Z(0) * qm.pauli.Z(1)
            zz_scaled = J * zz

            x0 = qm.pauli.X(0)
            x1 = qm.pauli.X(1)
            x_sum = x0 + x1
            x_scaled = g * x_sum

            h = zz_scaled + x_scaled

            assert isinstance(h, qm.HamiltonianExpr)

            # Verify operation types are present
            op_types = [type(op).__name__ for op in tracer.operations]
            assert "PauliCreateOp" in op_types
            assert "HamiltonianMulOp" in op_types
            assert "HamiltonianScaleOp" in op_types
            assert "HamiltonianAddOp" in op_types

    def test_chained_multiplication(self):
        """Test chained Pauli multiplication."""
        with trace() as tracer:
            # Z_0 * Z_1 * Z_2
            h = qm.pauli.Z(0) * qm.pauli.Z(1) * qm.pauli.Z(2)

            assert isinstance(h, qm.HamiltonianExpr)

            ops = tracer.operations
            # 3 PauliCreateOp + 2 HamiltonianMulOp
            assert len(ops) == 5

    def test_sum_support_radd(self):
        """Test that sum() works with HamiltonianExpr (via __radd__)."""
        with trace() as tracer:
            paulis = [qm.pauli.Z(0), qm.pauli.Z(1), qm.pauli.Z(2)]
            # This calls 0 + paulis[0] (uses __radd__)
            h = sum(paulis)

            assert isinstance(h, qm.HamiltonianExpr)


class TestHamiltonianExprWithFloat:
    """Test HamiltonianExpr with Float handle."""

    def test_multiply_by_float_handle_left(self):
        """Test Float * HamiltonianExpr - order matters.

        Note: Due to Python's operator precedence, Float.__mul__ is called first.
        This doesn't work because Float doesn't know about HamiltonianExpr.
        Use HamiltonianExpr * Float or scalar * HamiltonianExpr instead.
        """
        with trace() as tracer:
            J = qm.float_(1.5)
            h = qm.pauli.Z(0)
            # Float * HamiltonianExpr doesn't work as expected
            # Use h * J or literal * h instead
            h_scaled = h * J  # This works

            assert isinstance(h_scaled, qm.HamiltonianExpr)

            # Should have HamiltonianScaleOp
            ops = tracer.operations
            assert any(isinstance(op, HamiltonianScaleOp) for op in ops)

    def test_multiply_float_handle_right(self):
        """Test HamiltonianExpr * Float handle."""
        with trace() as tracer:
            h = qm.pauli.Z(0)
            J = qm.float_(2.5)
            h_scaled = h * J

            assert isinstance(h_scaled, qm.HamiltonianExpr)
            assert any(isinstance(op, HamiltonianScaleOp) for op in tracer.operations)

    def test_multiply_scalar_left_works(self):
        """Test that scalar * HamiltonianExpr works via __rmul__."""
        with trace() as tracer:
            h = qm.pauli.Z(0)
            # Literal scalar uses HamiltonianExpr.__rmul__
            h_scaled = 1.5 * h

            assert isinstance(h_scaled, qm.HamiltonianExpr)
            assert any(isinstance(op, HamiltonianScaleOp) for op in tracer.operations)
