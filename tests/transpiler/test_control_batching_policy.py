"""Tests for engine controlled-body fallback batching policy."""

from collections.abc import Iterator

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
    RuntimeClassicalExpr,
    RuntimeOpKind,
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import (
    CInitOperation,
    Operation,
    OperationKind,
    QInitOperation,
    Signature,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.emit_support.control_batching import (
    CONTROL_BATCH_DIRECT_AT_TWO_CONTROLS,
    CONTROL_BATCH_HEAVY_GATES,
    CONTROL_BATCH_MIN_WEIGHT,
    ControlBatchProfile,
    combine_control_batch_profiles,
    should_batch_controlled_body,
    static_controlled_batch_profile,
)


def _gate(gate_type: GateOperationType) -> GateOperation:
    """Build the minimal gate needed for policy-only classification.

    Args:
        gate_type (GateOperationType): IR gate category to classify.

    Returns:
        GateOperation: Minimal gate operation with no wire operands.
    """
    return GateOperation(gate_type=gate_type)


class _UnknownClassicalOperation(Operation):
    """Classical marker deliberately absent from the supported walker."""

    @property
    def signature(self) -> Signature:
        """Return an empty test-only signature."""
        return Signature()

    @property
    def operation_kind(self) -> OperationKind:
        """Classify the marker as classical without making it bookkeeping."""
        return OperationKind.CLASSICAL


@pytest.mark.parametrize(
    "operation",
    [
        ReturnOperation(),
        CInitOperation(),
        BinOp(kind=BinOpKind.ADD),
        CompOp(kind=CompOpKind.EQ),
        CondOp(kind=CondOpKind.AND),
        NotOp(),
        DictGetItemOperation(),
        UnaryMathOp(
            operands=[Value(type=UIntType(), name="input")],
            results=[Value(type=UIntType(), name="output")],
            kind=UnaryMathOpKind.CEIL,
        ),
        CastOperation(),
        QInitOperation(),
        ReturnQuantumArrayElementOperation(),
    ],
)
def test_static_batch_profile_classifies_bookkeeping_as_zero_work(
    operation: Operation,
) -> None:
    """Emitter and estimator share the complete zero-work classification."""
    assert static_controlled_batch_profile(operation) == ControlBatchProfile()
    assert (
        static_clean_ancilla_batch_profile(operation)
        == StaticCleanAncillaBatchProfile()
    )


@pytest.mark.parametrize(
    "operation",
    [
        StoreArrayElementOperation(),
        RuntimeClassicalExpr(kind=RuntimeOpKind.NOT),
        DecodeQFixedOperation(),
        _UnknownClassicalOperation(),
    ],
)
def test_static_batch_profile_keeps_unsupported_operations_visible(
    operation: Operation,
) -> None:
    """Unsupported markers contribute conservative work in both profiles."""
    assert static_controlled_batch_profile(operation) == ControlBatchProfile(
        weight=1,
        selects_exact_two=True,
    )
    assert static_clean_ancilla_batch_profile(
        operation
    ) == StaticCleanAncillaBatchProfile(work=1)


@pytest.mark.parametrize(
    "operation",
    [SliceArrayOperation(), ReleaseSliceViewOperation()],
)
def test_slice_markers_follow_their_pipeline_stage(
    operation: Operation,
) -> None:
    """Emit fails closed after strip, while pre-strip estimation counts zero."""
    assert static_controlled_batch_profile(operation) == ControlBatchProfile(
        weight=1,
        selects_exact_two=True,
    )
    assert (
        static_clean_ancilla_batch_profile(operation)
        == StaticCleanAncillaBatchProfile()
    )


def test_static_batch_profile_classifies_gate_and_contextual_work() -> None:
    """Engine policy distinguishes direct, shared, and contextual work."""
    assert static_controlled_batch_profile(
        _gate(GateOperationType.H)
    ) == ControlBatchProfile(weight=1, selects_exact_two=True)
    assert static_controlled_batch_profile(
        _gate(GateOperationType.CX)
    ) == ControlBatchProfile(
        weight=CONTROL_BATCH_MIN_WEIGHT,
        selects_exact_two=True,
    )
    assert static_controlled_batch_profile(PauliEvolveOp()) is None
    assert static_controlled_batch_profile(ForOperation()) is None
    assert CONTROL_BATCH_HEAVY_GATES == {
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
    }


def test_combined_batch_profile_stops_once_shared_ladder_is_decided() -> None:
    """Profile accumulation remains lazy until both decisions are resolved."""

    def profiles() -> Iterator[ControlBatchProfile]:
        """Yield enough profile data, then fail if composition consumes farther.

        Returns:
            Iterator[ControlBatchProfile]: Lazily generated operation profiles.
        """
        yield ControlBatchProfile(weight=1)
        yield ControlBatchProfile(weight=1)
        yield ControlBatchProfile(weight=1, selects_exact_two=True)
        raise AssertionError("batching profile consumed past its decision")

    assert combine_control_batch_profiles(profiles()) == ControlBatchProfile(
        weight=CONTROL_BATCH_MIN_WEIGHT,
        selects_exact_two=True,
    )


def test_batch_profile_completion_requires_both_saturated_fields() -> None:
    """Decision completion uses one shared predicate at every caller."""
    assert not ControlBatchProfile(weight=1, selects_exact_two=True).decision_complete
    assert not ControlBatchProfile(
        weight=CONTROL_BATCH_MIN_WEIGHT,
        selects_exact_two=False,
    ).decision_complete
    assert ControlBatchProfile(
        weight=CONTROL_BATCH_MIN_WEIGHT,
        selects_exact_two=True,
    ).decision_complete


def test_two_control_profitability_uses_emission_gate_type_policy() -> None:
    """Engine batching skips direct leaves but accepts other work."""
    direct_body = [_gate(GateOperationType.X), _gate(GateOperationType.Z)]
    shared_body = [_gate(GateOperationType.X), _gate(GateOperationType.CX)]
    direct_profile = combine_control_batch_profiles(
        profile
        for operation in direct_body
        if (profile := static_controlled_batch_profile(operation)) is not None
    )
    shared_profile = combine_control_batch_profiles(
        profile
        for operation in shared_body
        if (profile := static_controlled_batch_profile(operation)) is not None
    )

    assert CONTROL_BATCH_DIRECT_AT_TWO_CONTROLS == {
        GateOperationType.X,
        GateOperationType.Z,
    }
    assert direct_profile == ControlBatchProfile(
        weight=CONTROL_BATCH_MIN_WEIGHT,
        selects_exact_two=False,
    )
    assert shared_profile == ControlBatchProfile(
        weight=CONTROL_BATCH_MIN_WEIGHT,
        selects_exact_two=True,
    )
    assert not should_batch_controlled_body(
        num_controls=2,
        profile=direct_profile,
    )
    assert should_batch_controlled_body(
        num_controls=2,
        profile=shared_profile,
    )
    assert should_batch_controlled_body(
        num_controls=3,
        profile=direct_profile,
    )
    assert not should_batch_controlled_body(
        num_controls=3,
        profile=ControlBatchProfile(weight=1, selects_exact_two=True),
    )
