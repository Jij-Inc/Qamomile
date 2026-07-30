"""Tests for shared controlled-body batching policy."""

from collections.abc import Iterator

from qamomile.circuit._control_batching import (
    CONTROL_BATCH_MIN_WEIGHT,
    CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS,
    capped_controlled_batch_weight,
    controlled_body_benefits_from_two_control_batch,
    should_batch_controlled_body,
    static_controlled_batch_weight,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation


def _gate(gate_type: GateOperationType) -> GateOperation:
    """Build the minimal gate needed for policy-only classification.

    Args:
        gate_type (GateOperationType): IR gate category to classify.

    Returns:
        GateOperation: Minimal gate operation with no wire operands.
    """
    return GateOperation(gate_type=gate_type)


def test_static_batch_weight_classifies_context_free_operations() -> None:
    """Static policy distinguishes no-op, leaf, heavy, and contextual work."""
    assert static_controlled_batch_weight(ReturnOperation()) == 0
    assert static_controlled_batch_weight(QInitOperation()) == 0
    assert static_controlled_batch_weight(_gate(GateOperationType.H)) == 1
    assert static_controlled_batch_weight(PauliEvolveOp()) == CONTROL_BATCH_MIN_WEIGHT
    assert static_controlled_batch_weight(ForOperation()) is None


def test_capped_batch_weight_stops_at_shared_ladder_threshold() -> None:
    """Weight accumulation stops as soon as a shared ladder is justified."""

    def weights() -> Iterator[int]:
        """Yield enough work, then fail if the policy consumes too far.

        Returns:
            Iterator[int]: Lazily generated operation weights.
        """
        yield 0
        yield 1
        yield 1
        raise AssertionError("batching weight consumed past its threshold")

    assert capped_controlled_batch_weight(weights()) == CONTROL_BATCH_MIN_WEIGHT


def test_two_control_profitability_uses_shared_gate_type_policy() -> None:
    """Exact-two-control batching skips native leaves but accepts other work."""
    native_body = [_gate(GateOperationType.X), _gate(GateOperationType.Z)]
    non_native_body = [_gate(GateOperationType.X), _gate(GateOperationType.H)]

    assert CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS == {
        GateOperationType.X,
        GateOperationType.Z,
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.TOFFOLI,
        GateOperationType.RZZ,
    }
    assert not controlled_body_benefits_from_two_control_batch(native_body)
    assert controlled_body_benefits_from_two_control_batch(non_native_body)
    assert not should_batch_controlled_body(
        num_controls=2,
        body_weight=2,
        operations=native_body,
    )
    assert should_batch_controlled_body(
        num_controls=2,
        body_weight=2,
        operations=non_native_body,
    )
    assert should_batch_controlled_body(
        num_controls=3,
        body_weight=2,
        operations=native_body,
    )
    assert not should_batch_controlled_body(
        num_controls=3,
        body_weight=1,
        operations=non_native_body,
    )
