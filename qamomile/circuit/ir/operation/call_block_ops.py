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
        if self.block is None:
            raise ValueError("CallBlockOperation.block is required.")

    @property
    def signature(self) -> Signature:
        block = self.block
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
