"""Return operation for explicit block termination."""

import dataclasses

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class ReturnOperation(Operation):
    """Explicit return operation marking the end of a block with return values.

    This operation represents an explicit return statement in the IR.
    It takes the values to be returned as operands and produces no results
    (it is a terminal operation that transfers control flow back to the caller).

    operands: [Value, ...] - The values to return (may be empty for void returns)
    results: [] - Always empty (terminal operation)

    Example:
        A function that returns two values (a UInt and a Float):

        ReturnOperation(
            operands=[uint_value, float_value],
            results=[],
        )

        The signature would be:
            operands=[ParamHint("return_0", UIntType()), ParamHint("return_1", FloatType())]
            results=[]
    """

    @property
    def signature(self) -> Signature:
        """Return the signature with operands for each return value and no results."""
        return Signature(
            operands=[
                ParamHint(name=f"return_{i}", type=v.type)
                for i, v in enumerate(self.operands)
            ],
            results=[],  # Terminal operation - no results
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Return CLASSICAL as this is a control flow operation without quantum effects."""
        return OperationKind.CLASSICAL
