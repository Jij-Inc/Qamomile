"""Share backend-neutral controlled-body batching policy.

This module owns only the pure, structure-based part of shared-control
batching. Emitters and resource estimation retain their own value-resolution
adapters for loops, branches, nested calls, and controlled powers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation

CONTROL_BATCH_MIN_WEIGHT = 2

# These gates already have a direct or equally cheap fallback at exactly two
# composed controls. A shared AND carrier is therefore useful only when the
# body contains some other controlled leaf.
CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS: frozenset[GateOperationType] = frozenset(
    {
        GateOperationType.X,
        GateOperationType.Z,
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.TOFFOLI,
        GateOperationType.RZZ,
    }
)

_CONTEXT_DEPENDENT_OPERATION_TYPES = (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    ControlledUOperation,
    ForOperation,
    IfOperation,
    InvokeOperation,
    InverseBlockOperation,
)


def static_controlled_batch_weight(operation: Operation) -> int | None:
    """Return context-free batching weight for one controlled-body operation.

    ``None`` means that the caller must resolve classical state or a nested
    callable before deciding the weight. Unknown operation kinds conservatively
    count as one leaf so eligibility analysis never hides work that a later
    walker may reject.

    Args:
        operation (Operation): Operation inside a controlled body.

    Returns:
        int | None: Zero, one, or two for a context-free category, or ``None``
            when caller-specific value resolution is required.
    """
    if isinstance(operation, (ReturnOperation, QInitOperation)):
        return 0
    if isinstance(operation, (GateOperation, SelectOperation)):
        return 1
    if isinstance(operation, PauliEvolveOp):
        return CONTROL_BATCH_MIN_WEIGHT
    if isinstance(operation, _CONTEXT_DEPENDENT_OPERATION_TYPES):
        return None
    return 1


def capped_controlled_batch_weight(weights: Iterable[int]) -> int:
    """Sum controlled-body weights only up to the batching threshold.

    Args:
        weights (Iterable[int]): Nonnegative per-operation batching weights.

    Returns:
        int: Sum clamped to ``CONTROL_BATCH_MIN_WEIGHT``.
    """
    total = 0
    for weight in weights:
        total += weight
        if total >= CONTROL_BATCH_MIN_WEIGHT:
            return CONTROL_BATCH_MIN_WEIGHT
    return total


def controlled_body_benefits_from_two_control_batch(
    operations: Sequence[Operation],
) -> bool:
    """Return whether a two-control body benefits from a shared AND carrier.

    Args:
        operations (Sequence[Operation]): Controlled-body operations.

    Returns:
        bool: Whether at least one leaf lacks an equally cheap direct
            two-control fallback.
    """
    for operation in operations:
        if isinstance(operation, GateOperation):
            if operation.gate_type not in CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS:
                return True
        elif isinstance(
            operation,
            (
                ControlledUOperation,
                ForOperation,
                InvokeOperation,
                InverseBlockOperation,
                PauliEvolveOp,
                SelectOperation,
            ),
        ):
            return True
    return False


def should_batch_controlled_body(
    *,
    num_controls: int,
    body_weight: int,
    operations: Sequence[Operation],
) -> bool:
    """Return whether a controlled body should share one AND carrier.

    Args:
        num_controls (int): Concrete number of composed controls.
        body_weight (int): Context-resolved batching weight for the body.
        operations (Sequence[Operation]): Controlled-body operations used for
            the exact-two-control profitability guard.

    Returns:
        bool: Whether the common fallback should build one body-wide ladder.
    """
    if num_controls < 2 or body_weight < CONTROL_BATCH_MIN_WEIGHT:
        return False
    return num_controls != 2 or controlled_body_benefits_from_two_control_batch(
        operations
    )


__all__ = [
    "CONTROL_BATCH_MIN_WEIGHT",
    "CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS",
    "capped_controlled_batch_weight",
    "controlled_body_benefits_from_two_control_batch",
    "should_batch_controlled_body",
    "static_controlled_batch_weight",
]
