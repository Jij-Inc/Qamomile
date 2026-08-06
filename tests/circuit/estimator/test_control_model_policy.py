"""Tests for the fixed clean-ancilla coherent-control resource model."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.estimator._control_decomposition import (
    StaticCleanAncillaBatchProfile,
    static_clean_ancilla_batch_profile,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator.resource_estimator import (
    ResourceInterpreter,
    _ResourceEstimatorConfig,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.arithmetic_operations import (
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    Signature,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp


def _gate(gate_type: GateOperationType) -> GateOperation:
    """Build a minimal gate for fixed-model classification.

    Args:
        gate_type (GateOperationType): IR gate category to classify.

    Returns:
        GateOperation: Minimal gate operation without wire operands.
    """
    return GateOperation(gate_type=gate_type)


class _UnknownClassicalOperation(Operation):
    """Classical marker without resource-estimator semantics."""

    @property
    def signature(self) -> Signature:
        """Return an empty test-only signature."""
        return Signature()

    @property
    def operation_kind(self) -> OperationKind:
        """Classify the marker as classical."""
        return OperationKind.CLASSICAL


def test_controlled_unary_bookkeeping_matches_concrete_compiler() -> None:
    """Concrete unary bookkeeping is accepted by estimation and compilation."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def inner(target: qmc.Qubit, value: qmc.Float) -> qmc.Qubit:
        """Apply one gate after computing a classical structural value."""
        _rounded = qmc.ceil(value)
        return qmc.x(target)

    @qmc.qkernel
    def circuit(value: qmc.Float) -> qmc.Bit:
        """Control a body containing concrete unary bookkeeping."""
        control = qmc.qubit("control")
        target = qmc.qubit("target")
        control, target = qmc.control(inner)(control, target, value)
        return qmc.measure(target)

    estimate = circuit.estimate_resources(inputs={"value": 1.25})
    executable = QiskitTranspiler().transpile(
        circuit,
        bindings={"value": 1.25},
    )

    assert estimate.gates.total == 1
    assert len(executable.compiled_quantum) == 1


def test_unused_controlled_runtime_unary_math_is_zero_quantum_work() -> None:
    """Unused runtime unary math neither adds gates nor blocks emission."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def inner(target: qmc.Qubit, value: qmc.Float) -> qmc.Qubit:
        """Compute an unresolved unary value inside a controlled body."""
        _rounded = qmc.ceil(value)
        return target

    @qmc.qkernel
    def circuit(value: qmc.Float) -> qmc.Bit:
        """Invoke the unary body with a runtime parameter."""
        control = qmc.qubit("control")
        target = qmc.qubit("target")
        control, target = qmc.control(inner)(control, target, value)
        return qmc.measure(target)

    estimate = circuit.estimate_resources()
    executable = QiskitTranspiler().transpile(circuit, parameters=["value"])

    assert estimate.gates.total == 0
    assert len(executable.compiled_quantum) == 1


@pytest.mark.parametrize(
    "operation",
    [
        RuntimeClassicalExpr(kind=RuntimeOpKind.NOT),
        DecodeQFixedOperation(),
        _UnknownClassicalOperation(),
    ],
)
def test_fixed_model_rejects_unsupported_operation_under_control(
    operation: Operation,
) -> None:
    """Conservative profile work cannot become silent zero during evaluation."""
    block = Block(operations=[operation])
    interpreter = ResourceInterpreter(
        config=_ResourceEstimatorConfig(),
        bindings={},
    )

    with pytest.raises(ValueError, match="does not support this operation"):
        interpreter.eval_operation(
            operation,
            ExprResolver(block),
            controls=1,
        )


def test_estimator_treats_semantic_array_store_as_zero_control_work() -> None:
    """Resource analysis updates Store state without counting a quantum gate."""
    operation = StoreArrayElementOperation()
    block = Block(operations=[operation])
    interpreter = ResourceInterpreter(
        config=_ResourceEstimatorConfig(),
        bindings={},
    )

    estimate = interpreter.eval_operation(
        operation,
        ExprResolver(block),
        controls=1,
    )

    assert estimate.gates.total == 0


def test_fixed_model_classifies_gate_and_context_dependent_work() -> None:
    """The clean-ancilla model pins unit-gate and contextual work."""
    assert static_clean_ancilla_batch_profile(
        _gate(GateOperationType.H)
    ) == StaticCleanAncillaBatchProfile(work=1)
    assert static_clean_ancilla_batch_profile(
        _gate(GateOperationType.X)
    ) == StaticCleanAncillaBatchProfile(work=1)
    assert static_clean_ancilla_batch_profile(
        _gate(GateOperationType.CX)
    ) == StaticCleanAncillaBatchProfile(work=1)
    assert static_clean_ancilla_batch_profile(PauliEvolveOp()) is None
    assert static_clean_ancilla_batch_profile(ForOperation()) is None
