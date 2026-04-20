import dataclasses
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types.primitives import BlockType

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    pass


@dataclasses.dataclass
class CallBlockOperation(Operation):
    # ``block`` is ``None`` only transiently: the self-recursive @qkernel
    # path emits a forward-ref call before its enclosing Block exists, and
    # ``QKernel.block`` back-patches it to a real Block before the property
    # returns.  Every pass outside that back-patch window sees a non-None
    # block; readers check explicitly (``if op.block is None: raise``) to
    # surface the invariant rather than silently dereferencing ``None``.
    block: Block | None = None

    def is_self_reference_to(self, block: "Block") -> bool:
        """Return True if this call points to the given block (self-ref)."""
        return self.block is block

    @property
    def signature(self) -> Signature:
        block = self.block
        if block is None:
            raise ValueError(
                "CallBlockOperation.block is not set; cannot build signature "
                "while the call is still a forward reference."
            )
        input_type = [
            ParamHint(name, value.type)
            for name, value in zip(block.label_args, block.input_values)
        ]
        return Signature(
            operands=[ParamHint(name="func", type=BlockType()), *input_type],
            results=[
                ParamHint(name=f"outputs_{i}", type=v.type)
                for i, v in enumerate(self.results)
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        # CallBlockOperation is CONTROL as it represents a function call
        return OperationKind.CONTROL
