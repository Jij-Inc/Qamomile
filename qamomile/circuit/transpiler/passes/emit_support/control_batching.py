"""Own controlled-body batching policy for engine fallback emission."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

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
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation

CONTROL_BATCH_MIN_WEIGHT = 2

# One leaf of these types already amortizes an outer control carrier in the
# current engine fallback. This is an emission choice, not a resource-estimator
# contract.
CONTROL_BATCH_HEAVY_GATES: frozenset[GateOperationType] = frozenset(
    {
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
    }
)

# Exact-two-control bodies containing only these gates keep the current direct
# emission path. Engines may evolve this set without changing resource models.
CONTROL_BATCH_DIRECT_AT_TWO_CONTROLS: frozenset[GateOperationType] = frozenset(
    {
        GateOperationType.X,
        GateOperationType.Z,
    }
)

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
class ControlBatchProfile:
    """Describe resolved work and the engine's exact-two-control choice.

    Args:
        weight (int): Controlled work clamped to zero, one, or two.
        selects_exact_two (bool): Whether the engine selects a shared carrier
            when exactly two outer controls are present.
    """

    weight: int = 0
    selects_exact_two: bool = False

    def __post_init__(self) -> None:
        """Validate the bounded profile fields.

        Raises:
            TypeError: If either field has an invalid Python type.
            ValueError: If ``weight`` is outside the inclusive range zero to
                two.
        """
        if isinstance(self.weight, bool) or not isinstance(self.weight, int):
            raise TypeError("control batch weight must be a plain Python int.")
        if not 0 <= self.weight <= CONTROL_BATCH_MIN_WEIGHT:
            raise ValueError(
                "control batch weight must be between zero and "
                f"{CONTROL_BATCH_MIN_WEIGHT}, got {self.weight}."
            )
        if not isinstance(self.selects_exact_two, bool):
            raise TypeError("selects_exact_two must be a bool.")

    @property
    def decision_complete(self) -> bool:
        """Return whether later work cannot change the batching decision.

        Returns:
            bool: True when work is saturated and exact-two batching has
                already been selected.
        """
        return self.weight == CONTROL_BATCH_MIN_WEIGHT and self.selects_exact_two


def static_controlled_batch_profile(
    operation: Operation,
) -> ControlBatchProfile | None:
    """Return context-free emission batching information for one operation.

    ``None`` means that emission must resolve classical state or a nested
    callable before deciding the weight. Unknown operation kinds count as one
    leaf so eligibility analysis never hides work rejected by the later walker.

    Args:
        operation (Operation): Operation inside a controlled body.

    Returns:
        ControlBatchProfile | None: Context-free emission profile, or ``None``
            when engine-specific value resolution is required.
    """
    if isinstance(operation, (ReturnOperation, QInitOperation)):
        return ControlBatchProfile()
    if isinstance(operation, GateOperation):
        return ControlBatchProfile(
            weight=(
                CONTROL_BATCH_MIN_WEIGHT
                if operation.gate_type in CONTROL_BATCH_HEAVY_GATES
                else 1
            ),
            selects_exact_two=(
                operation.gate_type not in CONTROL_BATCH_DIRECT_AT_TWO_CONTROLS
            ),
        )
    if isinstance(operation, SelectOperation):
        return None
    if isinstance(operation, _CONTEXT_DEPENDENT_OPERATION_TYPES):
        return None
    return ControlBatchProfile(weight=1, selects_exact_two=True)


def combine_control_batch_profiles(
    profiles: Iterable[ControlBatchProfile],
) -> ControlBatchProfile:
    """Combine sequential emission profiles up to the decision threshold.

    Args:
        profiles (Iterable[ControlBatchProfile]): Resolved operation profiles.

    Returns:
        ControlBatchProfile: Capped work and the union of exact-two choices.
    """
    weight = 0
    selects_exact_two = False
    for profile in profiles:
        weight = min(CONTROL_BATCH_MIN_WEIGHT, weight + profile.weight)
        selects_exact_two = selects_exact_two or profile.selects_exact_two
        combined = ControlBatchProfile(
            weight=weight,
            selects_exact_two=selects_exact_two,
        )
        if combined.decision_complete:
            break
    return ControlBatchProfile(weight=weight, selects_exact_two=selects_exact_two)


def should_batch_controlled_body(
    *,
    num_controls: int,
    profile: ControlBatchProfile,
) -> bool:
    """Return whether engine fallback should share one AND carrier.

    Args:
        num_controls (int): Concrete number of composed controls.
        profile (ControlBatchProfile): Context-resolved emission profile.

    Returns:
        bool: Whether engine fallback should build one body-wide ladder.
    """
    if num_controls < 2 or profile.weight < CONTROL_BATCH_MIN_WEIGHT:
        return False
    return num_controls != 2 or profile.selects_exact_two


__all__ = [
    "ControlBatchProfile",
    "CONTROL_BATCH_MIN_WEIGHT",
    "CONTROL_BATCH_DIRECT_AT_TWO_CONTROLS",
    "CONTROL_BATCH_HEAVY_GATES",
    "combine_control_batch_profiles",
    "should_batch_controlled_body",
    "static_controlled_batch_profile",
]
