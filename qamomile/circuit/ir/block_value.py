from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, Any, cast

from qamomile.circuit.ir.types.primitives import BlockType
from .value import ArrayValue, Value
from .operation import Operation

if TYPE_CHECKING:
    from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation


@dataclasses.dataclass
class BlockValue(Value[BlockType]):
    """Represents a subroutine as a function block.

    def func_block(a: UInt, b: UInt) -> tuple[UInt]:
        ...

    BlockValue(
        name="func_block",
        inputs_type={"a": UIntType(), "b": UIntType()},
        outputs_type=(UIntType(), ),
        operations=[...],
    )

    Function to BlockValue conversion can be done via `func_to_block` function.
    Each Values in operations are dummy values.
    The execution of the BlockValue is corresponding to the BlockOperation.

    """

    type: BlockType = dataclasses.field(default_factory=BlockType)
    name: str = ""
    label_args: list[str] = dataclasses.field(default_factory=list)
    input_values: list[Value] = dataclasses.field(
        default_factory=list
    )  # store dummy values for inputs
    return_values: list[Value] = dataclasses.field(
        default_factory=list
    )  # store dummy values for returns
    operations: list[Operation] = dataclasses.field(default_factory=list)

    def call(self, **kwargs: Value) -> "CallBlockOperation":
        """Create a CallBlockOperation to call this BlockValue.

        Example:
            ```python
            block_value = BlockValue(
                name="func_block",
                inputs_type={"a": UIntType(), "b": UIntType()},
                outputs_type=(UIntType(), ),
                operations=[...],
            )
            a = Value(UIntType())
            b = Value(UIntType())
            call_op = block_value.call(a=a, b=b)
            ```
        """
        from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation

        inputs = [kwargs[label] for label in self.label_args]

        # Build callee-input → caller-arg UUID map for shape substitution.
        # Non-input array returns may have shape dims that reference callee
        # input Values; these must be normalized to caller-local Values so
        # that downstream operations (e.g. measure()) don't carry stale
        # callee-scoped metadata.
        input_uuid_map: dict[str, Value] = {}
        for block_input, call_arg in zip(self.input_values, inputs):
            input_uuid_map[block_input.uuid] = call_arg
            if isinstance(block_input, ArrayValue) and isinstance(call_arg, ArrayValue):
                for callee_dim, caller_dim in zip(block_input.shape, call_arg.shape):
                    input_uuid_map[callee_dim.uuid] = caller_dim

        # Use logical_id to track physical identity across SSA versions
        # All value types (Value, ArrayValue, TupleValue, DictValue) now have logical_id
        dummy_inputs = {v.logical_id: idx for idx, v in enumerate(self.input_values)}

        results = []
        for dummy_return in self.return_values:
            if dummy_return.logical_id in dummy_inputs:
                # If the return value is one of the input values, reuse the input value
                input_idx = dummy_inputs[dummy_return.logical_id]
                results.append(inputs[input_idx].next_version())
            elif isinstance(dummy_return, ArrayValue) and dummy_return.shape:
                # Substitute callee-local shape dims with caller-local values
                new_shape = tuple(
                    input_uuid_map.get(dim.uuid, dim) for dim in dummy_return.shape
                )
                if new_shape != dummy_return.shape:
                    results.append(dataclasses.replace(dummy_return, shape=new_shape))
                else:
                    results.append(dummy_return)
            else:
                results.append(dummy_return)

        return CallBlockOperation(
            operands=cast(list[Value[Any]], [self, *inputs]), results=results
        )
