"""Tests for the fixed clean-ancilla coherent-control resource model."""

from qamomile.circuit.estimator._control_decomposition import (
    StaticCleanAncillaBatchProfile,
    static_clean_ancilla_batch_profile,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation


def _gate(gate_type: GateOperationType) -> GateOperation:
    """Build a minimal gate for fixed-model classification.

    Args:
        gate_type (GateOperationType): IR gate category to classify.

    Returns:
        GateOperation: Minimal gate operation without wire operands.
    """
    return GateOperation(gate_type=gate_type)


def test_fixed_model_classifies_context_free_operations() -> None:
    """The clean-ancilla model pins empty, unit-gate, and contextual work."""
    assert (
        static_clean_ancilla_batch_profile(ReturnOperation())
        == StaticCleanAncillaBatchProfile()
    )
    assert (
        static_clean_ancilla_batch_profile(QInitOperation())
        == StaticCleanAncillaBatchProfile()
    )
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
