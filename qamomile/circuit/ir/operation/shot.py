"""Describe device execution context without fixing an engine realization."""

from dataclasses import dataclass

from qamomile.circuit.ir.types.primitives import UIntType

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclass
class GetCurrentShotOperation(Operation):
    """Read the zero-based index of the current device program repetition.

    The value is stable throughout one repetition and is supplied by the
    execution target, never by compile-time bindings or a host-side counter.

    Args:
        operands (list[Value]): Empty operand list.
        results (list[Value]): One nonconstant UInt result.
    """

    @property
    def signature(self) -> Signature:
        """Return the execution-context reader's signature.

        Returns:
            Signature: No inputs and one UInt output.
        """
        return Signature([], [ParamHint(name="shot", type=UIntType())])

    @property
    def operation_kind(self) -> OperationKind:
        """Keep this context read inside the device execution.

        Returns:
            OperationKind: HYBRID, since the value belongs to device execution.
        """
        return OperationKind.HYBRID
