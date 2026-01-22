import dataclasses
from typing import TYPE_CHECKING

from qamomile.circuit.ir.types.primitives import BlockType

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    pass


@dataclasses.dataclass
class CallBlockOperation(Operation):
    @property
    def signature(self) -> Signature:
        from qamomile.circuit.ir.block_value import BlockValue

        if not isinstance(self.operands[0], BlockValue):
            raise TypeError("The first operand must be a BlockValue.")
        block: BlockValue = self.operands[0]
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
