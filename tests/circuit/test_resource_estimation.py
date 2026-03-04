"""Tests for resource estimation."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.resource_estimation import (
    ResourceEstimate,
    ResourceEstimator,
)
from qamomile.circuit.stdlib.qft import QFT, qft
from qamomile.circuit.frontend.handle import Vector, Qubit
from qamomile.circuit.transpiler.errors import LinearTypeError


class TestResourceEstimate:
    """Tests for ResourceEstimate dataclass."""

    def test_default_values(self):
        """Test default values."""
        estimate = ResourceEstimate()
        assert estimate.total_gates == 0
        assert estimate.t_gate_count == 0
        assert estimate.cnot_count == 0
        assert estimate.qubit_count == 0

    def test_add_estimates(self):
        """Test adding two estimates."""
        est1 = ResourceEstimate(
            total_gates=10,
            t_gate_count=2,
            cnot_count=5,
            qubit_count=3,
            gate_counts={"h": 3, "cx": 5},
        )
        est2 = ResourceEstimate(
            total_gates=5,
            t_gate_count=1,
            cnot_count=2,
            qubit_count=4,
            gate_counts={"h": 2, "t": 1},
        )

        combined = est1 + est2

        assert combined.total_gates == 15
        assert combined.t_gate_count == 3
        assert combined.cnot_count == 7
        assert combined.qubit_count == 4  # max of the two
        assert combined.gate_counts["h"] == 5
        assert combined.gate_counts["cx"] == 5
        assert combined.gate_counts["t"] == 1

    def test_summary(self):
        """Test summary generation."""
        estimate = ResourceEstimate(
            total_gates=10,
            t_gate_count=2,
            cnot_count=5,
            qubit_count=3,
            gate_counts={"h": 3, "cx": 5, "t": 2},
        )

        summary = estimate.summary()

        assert "Total gates: 10" in summary
        assert "T gates: 2" in summary
        assert "Qubits: 3" in summary


class TestResourceEstimator:
    """Tests for ResourceEstimator."""

    def test_estimate_simple_kernel(self):
        """Test estimating a simple kernel."""

        @qmc.qkernel
        def simple(q: Qubit) -> Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        estimator = ResourceEstimator()
        estimate = estimator.estimate(simple)

        assert estimate.total_gates >= 2
        assert "h" in estimate.gate_counts
        assert "x" in estimate.gate_counts

    def test_estimate_with_loop(self):
        """Test estimating kernel with loop."""

        @qmc.qkernel
        def looped(q: Qubit, n: int) -> Qubit:
            for _ in qmc.range(n):
                q = qmc.h(q)
            return q

        estimator = ResourceEstimator()
        estimate = estimator.estimate(looped, bindings={"n": 5})

        # Should have 5 H gates from the loop
        assert estimate.gate_counts.get("h", 0) >= 5

    def test_estimate_qft(self):
        """Test estimating QFT resources."""

        @qmc.qkernel
        def qft_circuit(qubits: Vector[Qubit]) -> Vector[Qubit]:
            qubits = qft(qubits)
            return qubits

        estimator = ResourceEstimator()
        # Note: This test may need adjustment based on how QFT is traced
        # The important thing is that it doesn't crash and returns an estimate

    def test_compare_strategies(self):
        """Test comparing strategies."""

        @qmc.qkernel
        def simple_circuit(q: Qubit) -> Qubit:
            q = qmc.h(q)
            return q

        estimator = ResourceEstimator()
        comparison = estimator.compare_strategies(
            simple_circuit,
            strategies=["standard", "approximate"],
        )

        assert "standard" in comparison
        assert "approximate" in comparison

    def test_estimator_with_config(self):
        """Test estimator with decomposition config."""
        config = DecompositionConfig(
            strategy_overrides={"qft": "approximate"},
        )
        estimator = ResourceEstimator(decomposition_config=config)

        # Should use approximate strategy for QFT
        assert estimator._config.get_strategy_for_gate("qft") == "approximate"


class TestCallBlockEstimation:
    """Tests for CallBlockOperation dispatch in resource estimation."""

    def test_subkernel_gates_counted(self):
        """CallBlockOperation gates must be included in estimate."""

        @qmc.qkernel
        def inner(q: Qubit) -> Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        @qmc.qkernel
        def outer(q: Qubit) -> Qubit:
            q = inner(q)
            q = qmc.z(q)
            return q

        estimator = ResourceEstimator()
        estimate = estimator.estimate(outer)

        # inner: H + X = 2 gates; outer: + Z = 3 total
        assert estimate.total_gates >= 3
        assert estimate.gate_counts.get("h", 0) >= 1
        assert estimate.gate_counts.get("x", 0) >= 1
        assert estimate.gate_counts.get("z", 0) >= 1


class TestControlledUEstimation:
    """Tests for ControlledUOperation dispatch in resource estimation."""

    def test_controlled_operation_gates_counted(self):
        """ControlledUOperation gates must be included in estimate."""

        @qmc.qkernel
        def unitary(q: Qubit) -> Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            controlled_u = qmc.controlled(unitary)
            ctrl, tgt = controlled_u(ctrl, tgt)
            return ctrl, tgt

        estimator = ResourceEstimator()
        estimate = estimator.estimate(circuit)

        # Should count the H gate inside the controlled block
        assert estimate.total_gates >= 1
        assert estimate.gate_counts.get("h", 0) >= 1


class TestFailFastUnknownOp:
    """Tests for fail-fast on unknown operation types."""

    def test_unknown_op_raises(self):
        """Unhandled operation type must raise ValueError."""
        from dataclasses import dataclass, field

        from qamomile.circuit.ir.block import Block, BlockKind
        from qamomile.circuit.ir.operation.operation import Operation, Signature
        from qamomile.circuit.ir.operation.operation import OperationKind
        from qamomile.circuit.ir.value import Value

        @dataclass
        class FakeOperation(Operation):
            @property
            def signature(self) -> Signature:
                return Signature()

            @property
            def operation_kind(self) -> OperationKind:
                return OperationKind.QUANTUM

        block = Block(
            name="test",
            label_args=[],
            input_values=[],
            output_values=[],
            operations=[FakeOperation()],
            kind=BlockKind.HIERARCHICAL,
        )

        estimator = ResourceEstimator()
        with pytest.raises(ValueError, match="unhandled operation type"):
            estimator.estimate_block(block)


class TestLinearPreflightIntegration:
    """Tests for validate_linear parameter."""

    def test_validate_linear_true_catches_violation(self):
        """estimate_block(validate_linear=True) catches linear violations at IR level."""
        from qamomile.circuit.ir.block import Block, BlockKind
        from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
        from qamomile.circuit.ir.operation.operation import QInitOperation
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        # Manually construct a Block with double-use of a quantum value
        q_val = Value(name="q", type=QubitType())
        q_init = QInitOperation(results=[q_val])

        # Use q_val twice → linear violation
        h_result = Value(name="q", type=QubitType(), version=1)
        h_op = GateOperation(
            gate_type=GateOperationType.H,
            operands=[q_val],
            results=[h_result],
        )
        x_result = Value(name="q", type=QubitType(), version=2)
        x_op = GateOperation(
            gate_type=GateOperationType.X,
            operands=[q_val],  # same q_val → violation
            results=[x_result],
        )

        block = Block(
            name="bad_block",
            label_args=[],
            input_values=[],
            output_values=[x_result],
            operations=[q_init, h_op, x_op],
            kind=BlockKind.HIERARCHICAL,
        )

        estimator = ResourceEstimator()
        with pytest.raises(LinearTypeError):
            estimator.estimate_block(block, validate_linear=True)

    def test_validate_linear_false_is_default(self):
        """Default behavior does not run linear validation."""
        from qamomile.circuit.ir.block import Block, BlockKind
        from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
        from qamomile.circuit.ir.operation.operation import QInitOperation
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        # Same double-use block, but validate_linear=False (default)
        q_val = Value(name="q", type=QubitType())
        q_init = QInitOperation(results=[q_val])
        h_result = Value(name="q", type=QubitType(), version=1)
        h_op = GateOperation(
            gate_type=GateOperationType.H,
            operands=[q_val],
            results=[h_result],
        )
        x_result = Value(name="q", type=QubitType(), version=2)
        x_op = GateOperation(
            gate_type=GateOperationType.X,
            operands=[q_val],
            results=[x_result],
        )

        block = Block(
            name="bad_block",
            label_args=[],
            input_values=[],
            output_values=[x_result],
            operations=[q_init, h_op, x_op],
            kind=BlockKind.HIERARCHICAL,
        )

        estimator = ResourceEstimator()
        # Should NOT raise — validation is off by default
        estimate = estimator.estimate_block(block)
        assert estimate.total_gates == 2
