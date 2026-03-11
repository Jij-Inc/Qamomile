from __future__ import annotations

import dataclasses
import typing

from qamomile.circuit.ir.types.primitives import BitType, BlockType, UIntType
from qamomile.circuit.ir.value import Value

from .operation import Operation, OperationKind, ParamHint, Signature

if typing.TYPE_CHECKING:
    from .arithmetic_operations import PhiOp


@dataclasses.dataclass
class WhileOperation(Operation):
    """Represents a while loop operation.

    Example:
        while condition:
            body

    Attributes:
        operations: List of operations in the loop body
        operands[0]: initial condition (always present, Bit type)
        operands[1]: loop-carried condition (present only when the body
                     reassigns the condition variable via measurement +
                     if-else / if-only branching)

    The loop-carried operand (operands[1]) is added by the frontend
    ``while_loop`` builder when it detects that the condition variable
    was reassigned inside the body.  The transpiler emit pass uses this
    to alias the loop-carried classical bit back to the initial
    condition's classical bit so that Qiskit's ``while_loop`` reads the
    updated value on each iteration.
    """

    operations: list[Operation] = dataclasses.field(default_factory=list)

    @property
    def signature(self) -> Signature:
        # The signature declares both the initial condition and the
        # optional loop-carried condition.  At construction time
        # operands may have 1 or 2 entries; the emit pass asserts
        # exactly 2 when the loop body reassigns the condition.
        return Signature(
            operands=[
                ParamHint("condition", BitType()),
                ParamHint("loop_carried", BitType()),
            ],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForOperation(Operation):
    """Represents a for loop operation.

    Example:
        for i in range(start, stop, step):
            body

    Attributes:
        loop_var: Name of the loop variable (e.g., "i")
        operations: List of operations in the loop body
        operands[0]: start (UInt type)
        operands[1]: stop (UInt type)
        operands[2]: step (UInt type)
    """

    loop_var: str = ""
    operations: list[Operation] = dataclasses.field(default_factory=list)

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("start", UIntType()),
                ParamHint("stop", UIntType()),
                ParamHint("step", UIntType()),
            ],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForItemsOperation(Operation):
    """Represents iteration over dict/iterable items.

    Example:
        for (i, j), Jij in qmc.items(ising):
            body

    Attributes:
        key_vars: Names of key unpacking variables (e.g., ["i", "j"] for tuple keys)
        value_var: Name of value variable (e.g., "Jij")
        operations: List of operations in the loop body
        operands[0]: The dict/iterable value (DictValue type)

    Note:
        This operation is always unrolled at transpile time since quantum
        backends cannot natively iterate over classical data structures.
    """

    key_vars: list[str] = dataclasses.field(default_factory=list)
    value_var: str = ""
    key_is_vector: bool = False
    operations: list[Operation] = dataclasses.field(default_factory=list)

    @property
    def signature(self) -> Signature:
        # Signature is flexible - operand is the dict/iterable being iterated
        return Signature(
            operands=[ParamHint("iterable", BlockType())],  # BlockType as placeholder
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class IfOperation(Operation):
    """Represents an if-else conditional operation.

    Example:
        if condition:
            true_body
        else:
            false_body

    Attributes:
        true_operations: List of operations in the true branch
        false_operations: List of operations in the false branch (may be empty)
        phi_ops: List of PhiOp instances merging values from both branches
        operands[0]: condition (Bit type from measurement or comparison)
        results: Phi-merged output values from both branches
    """

    true_operations: list[Operation] = dataclasses.field(default_factory=list)
    false_operations: list[Operation] = dataclasses.field(default_factory=list)
    phi_ops: list[PhiOp] = dataclasses.field(default_factory=list)

    @property
    def condition(self) -> Value:
        return self.operands[0]

    @property
    def signature(self) -> Signature:
        result_hints = [
            ParamHint(name=f"result_{i}", type=r.type)
            for i, r in enumerate(self.results)
        ]
        return Signature(
            operands=[ParamHint("condition", BitType())], results=result_hints
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL
