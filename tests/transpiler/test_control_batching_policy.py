"""Tests for engine controlled-body fallback batching policy."""

from collections.abc import Iterator

from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
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


def test_static_batch_profile_classifies_context_free_operations() -> None:
    """Engine policy distinguishes direct, shared, and contextual work."""
    assert static_controlled_batch_profile(ReturnOperation()) == ControlBatchProfile()
    assert static_controlled_batch_profile(QInitOperation()) == ControlBatchProfile()
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
