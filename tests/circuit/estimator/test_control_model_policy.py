"""Tests for the fixed clean-ancilla coherent-control resource model."""

import pytest

from qamomile.circuit.estimator._control_decomposition import (
    StaticCleanAncillaBatchProfile,
    static_clean_ancilla_batch_profile,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import (
    CInitOperation,
    Operation,
    QInitOperation,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)


def _gate(gate_type: GateOperationType) -> GateOperation:
    """Build a minimal gate for fixed-model classification.

    Args:
        gate_type (GateOperationType): IR gate category to classify.

    Returns:
        GateOperation: Minimal gate operation without wire operands.
    """
    return GateOperation(gate_type=gate_type)


@pytest.mark.parametrize(
    "operation",
    [
        ReturnOperation(),
        CInitOperation(),
        BinOp(kind=BinOpKind.ADD),
        CompOp(kind=CompOpKind.EQ),
        CondOp(kind=CondOpKind.AND),
        NotOp(),
        SliceArrayOperation(),
        ReleaseSliceViewOperation(),
        StoreArrayElementOperation(),
    ],
)
def test_fixed_model_classifies_classical_bookkeeping_as_zero_work(
    operation: Operation,
) -> None:
    """Every classical bookkeeping operation contributes no quantum work."""
    assert static_clean_ancilla_batch_profile(operation) == (
        StaticCleanAncillaBatchProfile()
    )


@pytest.mark.parametrize(
    "operation",
    [
        CastOperation(),
        QInitOperation(),
        ReturnQuantumArrayElementOperation(),
    ],
)
def test_fixed_model_classifies_quantum_bookkeeping_as_zero_work(
    operation: Operation,
) -> None:
    """Quantum identity and allocation markers contribute no gate work."""
    assert static_clean_ancilla_batch_profile(operation) == (
        StaticCleanAncillaBatchProfile()
    )


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
