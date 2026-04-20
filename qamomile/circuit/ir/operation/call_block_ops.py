import dataclasses
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types.primitives import BlockType

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    pass


@dataclasses.dataclass
class CallBlockOperation(Operation):
    block: Block = dataclasses.field(default=None)  # type: ignore[arg-type]

    def __post_init__(self):
        # ``block`` may be None transiently while a self-recursive @qkernel is
        # still building: the trace emits the self-call before the enclosing
        # Block exists, and ``QKernel.block`` back-patches it after
        # ``func_to_block`` returns.  Any pass that runs outside that window
        # must see a finalized block — the property and inline path below
        # enforce that.
        pass

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
