"""Define the zero-qubit global-phase IR operation."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import ArrayValue, Value

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class GlobalPhaseOperation(Operation):
    """Multiply the complete quantum state by ``exp(i * phase)``.

    Global phase has no target qubit and does not create a new quantum value.
    Keeping the phase as the operation's sole ordinary operand lets generic IR
    passes substitute, serialize, and analyze it without a special value-field
    protocol. A surrounding controlled-unitary lowering turns the operation
    into an observable phase gate on the accumulated controls.

    Args:
        operands (list[Value]): Exactly one scalar ``FloatType`` phase angle in
            radians.
        results (list[Value]): Must be empty because global phase changes no
            qubit identity.

    Raises:
        ValueError: If the operand/result layout is invalid or the phase is not
            a scalar ``FloatType`` value.
    """

    def __post_init__(self) -> None:
        """Validate the zero-qubit global-phase operation contract.

        Raises:
            ValueError: If there is not exactly one scalar ``FloatType``
                operand or if any result is present.
        """
        if len(self.operands) != 1:
            raise ValueError(
                "global phase requires exactly one phase operand; "
                f"got {len(self.operands)}."
            )
        if self.results:
            raise ValueError(
                "global phase is a zero-qubit operation and must not produce "
                f"results; got {len(self.results)}."
            )

        phase = self.operands[0]
        if not isinstance(phase, Value) or isinstance(phase, ArrayValue):
            raise ValueError("global phase must be a scalar Value, not an array.")
        if not isinstance(phase.type, FloatType):
            raise ValueError(
                "global phase must be a FloatType angle in radians; "
                f"got {type(phase.type).__name__}."
            )

    @property
    def phase(self) -> Value:
        """Return the scalar phase-angle operand.

        Returns:
            Value: Scalar ``FloatType`` angle in radians.
        """
        return self.operands[0]

    @property
    def signature(self) -> Signature:
        """Return the zero-qubit operation signature.

        Returns:
            Signature: One scalar phase operand and no results.
        """
        return Signature(
            operands=[ParamHint(name="phase", type=FloatType())],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Classify global phase as a quantum operation.

        Returns:
            OperationKind: Always ``OperationKind.QUANTUM``.
        """
        return OperationKind.QUANTUM
