"""Define fixed coherent-control policies for resource estimation."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_work import (
    ControlWorkKind,
    classify_control_work,
)
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)

CLEAN_ANCILLA_BATCH_MIN_WORK = 2


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
    # Resource estimation consumes the semantic IR before the transpiler's
    # slice-marker stripping pass. These two lifetime markers are therefore
    # valid zero-work structure here, while an emitter correctly treats either
    # marker as unsupported if it survives to the later emission stage.
    if isinstance(operation, (SliceArrayOperation, ReleaseSliceViewOperation)):
        return StaticCleanAncillaBatchProfile()

    work_kind = classify_control_work(operation)
    if work_kind is ControlWorkKind.BOOKKEEPING:
        return StaticCleanAncillaBatchProfile()
    if work_kind is ControlWorkKind.QUANTUM_LEAF:
        return StaticCleanAncillaBatchProfile(work=1)
    if work_kind is ControlWorkKind.CONTEXT_DEPENDENT:
        return None
    return StaticCleanAncillaBatchProfile(work=1)


__all__ = [
    "CLEAN_ANCILLA_BATCH_MIN_WORK",
    "StaticCleanAncillaBatchProfile",
    "static_clean_ancilla_batch_profile",
]
