"""Classify IR operations for coherent-control work analysis.

This module owns only semantic categories shared by resource estimation and
emission.  It deliberately does not assign decomposition weights: the
estimator and emitter translate the same category into their own model or
engine-specific cost after classification.
"""

from __future__ import annotations

import enum

from .arithmetic_operations import BinOp, CompOp, CondOp, NotOp, UnaryMathOp
from .callable import InvokeOperation
from .cast import CastOperation
from .classical_ops import DictGetItemOperation, ReturnQuantumArrayElementOperation
from .control_flow import HasNestedOps
from .gate import ControlledUOperation, GateOperation
from .global_phase import GlobalPhaseOperation
from .inverse_block import InverseBlockOperation
from .operation import CInitOperation, Operation, QInitOperation
from .pauli_evolve import PauliEvolveOp
from .return_operation import ReturnOperation
from .select import SelectOperation


class ControlWorkKind(enum.Enum):
    """Describe how an operation participates in coherent-control analysis.

    Values:
        BOOKKEEPING: The operation contributes no controlled quantum gate, but
            its validation or value/alias update may still have to execute.
        QUANTUM_LEAF: One primitive quantum gate whose decomposition policy is
            selected by the consumer.
        CONTEXT_DEPENDENT: A structured operation whose work must be resolved
            from its values, selected body, or nested regions.
        UNSUPPORTED: An operation outside the supported controlled-unitary
            language.  Consumers keep it visible so their normal error path
            rejects it rather than silently dropping it.
    """

    BOOKKEEPING = "bookkeeping"
    QUANTUM_LEAF = "quantum_leaf"
    CONTEXT_DEPENDENT = "context_dependent"
    UNSUPPORTED = "unsupported"


_CONTEXT_DEPENDENT_TYPES = (
    ControlledUOperation,
    GlobalPhaseOperation,
    HasNestedOps,
    InvokeOperation,
    InverseBlockOperation,
    PauliEvolveOp,
    SelectOperation,
)

_BOOKKEEPING_TYPES = (
    BinOp,
    CastOperation,
    CInitOperation,
    CompOp,
    CondOp,
    DictGetItemOperation,
    NotOp,
    QInitOperation,
    ReturnOperation,
    ReturnQuantumArrayElementOperation,
    UnaryMathOp,
)


def classify_control_work(operation: Operation) -> ControlWorkKind:
    """Return the shared coherent-control category for one IR operation.

    ``BOOKKEEPING`` means zero controlled quantum work, not that the operation
    may be discarded.  Emitters must still run bookkeeping semantics such as
    classical evaluation, cast alias propagation, and deferred borrow-return
    validation.

    Args:
        operation (Operation): IR operation inside a coherently controlled
            body.

    Returns:
        ControlWorkKind: Semantic category consumed by both estimation and
            emission policy.
    """
    if isinstance(operation, _BOOKKEEPING_TYPES):
        return ControlWorkKind.BOOKKEEPING
    if isinstance(operation, GateOperation):
        return ControlWorkKind.QUANTUM_LEAF
    if isinstance(operation, _CONTEXT_DEPENDENT_TYPES):
        return ControlWorkKind.CONTEXT_DEPENDENT
    return ControlWorkKind.UNSUPPORTED


__all__ = ["ControlWorkKind", "classify_control_work"]
