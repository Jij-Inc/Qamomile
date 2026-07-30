"""Define fixed coherent-control policies for resource estimation."""

from __future__ import annotations

import dataclasses

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
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation

CLEAN_ANCILLA_BATCH_MIN_WORK = 2

_CONTEXT_DEPENDENT_OPERATION_TYPES = (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    ControlledUOperation,
    ForOperation,
    GlobalPhaseOperation,
    IfOperation,
    InvokeOperation,
    InverseBlockOperation,
    PauliEvolveOp,
)


@dataclasses.dataclass(frozen=True, slots=True)
class StaticCleanAncillaBatchProfile:
    """Describe context-free work in the fixed clean-ancilla resource model.

    Args:
        work (int): Controlled work clamped to zero, one, or two.
    """

    work: int = 0

    def __post_init__(self) -> None:
        """Validate the fixed-model profile fields.

        Raises:
            TypeError: If ``work`` has an invalid Python type.
            ValueError: If ``work`` is outside the inclusive range zero to two.
        """
        if isinstance(self.work, bool) or not isinstance(self.work, int):
            raise TypeError("clean-ancilla batch work must be a plain Python int.")
        if not 0 <= self.work <= CLEAN_ANCILLA_BATCH_MIN_WORK:
            raise ValueError(
                "clean-ancilla batch work must be between zero and "
                f"{CLEAN_ANCILLA_BATCH_MIN_WORK}, got {self.work}."
            )


def static_clean_ancilla_batch_profile(
    operation: Operation,
) -> StaticCleanAncillaBatchProfile | None:
    """Classify one operation under the fixed clean-ancilla resource model.

    ``None`` means that the estimator must resolve classical state or a nested
    callable before choosing the modeled work. Unknown operation kinds
    conservatively contribute one unit of work.

    Args:
        operation (Operation): Operation inside a coherently controlled body.

    Returns:
        StaticCleanAncillaBatchProfile | None: Context-free fixed-model profile,
            or ``None`` when estimator-specific resolution is required.
    """
    if isinstance(operation, (ReturnOperation, QInitOperation)):
        return StaticCleanAncillaBatchProfile()
    if isinstance(operation, GateOperation):
        return StaticCleanAncillaBatchProfile(work=1)
    if isinstance(operation, SelectOperation):
        return None
    if isinstance(operation, _CONTEXT_DEPENDENT_OPERATION_TYPES):
        return None
    return StaticCleanAncillaBatchProfile(work=1)


__all__ = [
    "CLEAN_ANCILLA_BATCH_MIN_WORK",
    "StaticCleanAncillaBatchProfile",
    "static_clean_ancilla_batch_profile",
]
