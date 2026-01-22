"""Tests for Hamiltonian evaluation pass."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.types.hamiltonian import PauliKind
from qamomile.circuit.observable.concrete import ConcreteHamiltonian
from qamomile.circuit.transpiler.hamiltonian_eval import (
    HamiltonianEvaluator,
    evaluate_hamiltonian,
    HamiltonianEvalError,
)
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.block import Block


def build_block_from_trace(func):
    """Helper to build a Block from traced operations."""
    with trace() as tracer:
        result = func()

        # Create a simple block with the operations
        block = Block(
            name="test_block",
            input_values=[],
            operations=tracer.operations,
            output_values=[result.value] if result else [],
        )
        return block


class TestHamiltonianEvaluator:
    """Test the HamiltonianEvaluator class."""

    def test_evaluate_single_pauli(self):
        """Test evaluating a single Pauli operator."""
        block = build_block_from_trace(lambda: qm.pauli.Z(0))

        evaluator = HamiltonianEvaluator()
        result = evaluator.evaluate(block)

        assert result is not None
        assert isinstance(result, ConcreteHamiltonian)
        assert len(result) == 1
        assert ((PauliKind.Z, 0),) in result.terms
        assert result.terms[((PauliKind.Z, 0),)] == 1.0

    def test_evaluate_pauli_addition(self):
        """Test evaluating addition of Paulis."""
        block = build_block_from_trace(
            lambda: qm.pauli.Z(0) + qm.pauli.X(1)
        )

        result = evaluate_hamiltonian(block)

        assert result is not None
        assert len(result) == 2
        assert ((PauliKind.Z, 0),) in result.terms
        assert ((PauliKind.X, 1),) in result.terms

    def test_evaluate_pauli_multiplication(self):
        """Test evaluating multiplication of Paulis."""
        block = build_block_from_trace(
            lambda: qm.pauli.Z(0) * qm.pauli.Z(1)
        )

        result = evaluate_hamiltonian(block)

        assert result is not None
        assert len(result) == 1
        # Should have Z_0 * Z_1 term
        expected_key = ((PauliKind.Z, 0), (PauliKind.Z, 1))
        assert expected_key in result.terms

    def test_evaluate_scalar_multiplication(self):
        """Test evaluating scalar multiplication."""
        block = build_block_from_trace(
            lambda: 2.0 * qm.pauli.Z(0)
        )

        result = evaluate_hamiltonian(block)

        assert result is not None
        assert result.terms[((PauliKind.Z, 0),)] == 2.0

    def test_evaluate_negation(self):
        """Test evaluating negation."""
        block = build_block_from_trace(
            lambda: -qm.pauli.Z(0)
        )

        result = evaluate_hamiltonian(block)

        assert result is not None
        assert result.terms[((PauliKind.Z, 0),)] == -1.0

    def test_evaluate_complex_hamiltonian(self):
        """Test evaluating a complex Hamiltonian."""
        def build_hamiltonian():
            # H = 0.5 * Z_0 * Z_1 + 0.3 * X_0 + 0.3 * X_1
            zz = qm.pauli.Z(0) * qm.pauli.Z(1)
            h = 0.5 * zz
            h = h + 0.3 * qm.pauli.X(0)
            h = h + 0.3 * qm.pauli.X(1)
            return h

        block = build_block_from_trace(build_hamiltonian)

        result = evaluate_hamiltonian(block)

        assert result is not None
        assert len(result) == 3

        zz_key = ((PauliKind.Z, 0), (PauliKind.Z, 1))
        assert zz_key in result.terms
        assert abs(result.terms[zz_key] - 0.5) < 1e-10

        assert abs(result.terms[((PauliKind.X, 0),)] - 0.3) < 1e-10
        assert abs(result.terms[((PauliKind.X, 1),)] - 0.3) < 1e-10


class TestEvaluationWithBindings:
    """Test evaluation with parameter bindings."""

    def test_evaluate_with_bindings(self):
        """Test evaluation with bound parameters."""
        # This test would require passing Float parameters through
        # For now we test with literal values
        block = build_block_from_trace(
            lambda: 1.5 * qm.pauli.Z(0)
        )

        result = evaluate_hamiltonian(block, bindings={})

        assert result is not None
        assert abs(result.terms[((PauliKind.Z, 0),)] - 1.5) < 1e-10


class TestPauliAlgebraEvaluation:
    """Test that Pauli algebra is correctly applied during evaluation."""

    def test_pauli_squares_cancel(self):
        """Test that P*P = I in evaluation."""
        block = build_block_from_trace(
            lambda: qm.pauli.Z(0) * qm.pauli.Z(0)
        )

        result = evaluate_hamiltonian(block)

        assert result is not None
        # Z*Z = I, so we should have identity term
        assert () in result.terms  # Identity term
        assert result.terms[()] == 1.0

    def test_xy_equals_iz(self):
        """Test that X*Y = iZ in evaluation."""
        # Note: This tests the ConcreteHamiltonian multiplication
        # which applies Pauli algebra
        h1 = ConcreteHamiltonian.single_pauli(PauliKind.X, 0)
        h2 = ConcreteHamiltonian.single_pauli(PauliKind.Y, 0)
        result = h1 * h2

        assert len(result) == 1
        assert ((PauliKind.Z, 0),) in result.terms
        assert result.terms[((PauliKind.Z, 0),)] == 1j
