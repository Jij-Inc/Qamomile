import inspect
import typing

import qamomile.circuit.ir.types as ir_types
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType, ValueType
from qamomile.circuit.ir.value import ArrayValue, Value

from qamomile.circuit.frontend.handle.primitives import (
    Bit,
    Float,
    Handle,
    Qubit,
    UInt,
)
from qamomile.circuit.frontend.tracer import Tracer, trace, get_current_tracer


TYPE_MAPPING: dict[typing.Any, typing.Any] = {
    int: UIntType,
    float: FloatType,
    bool: BitType,
}


def is_array_type(t) -> bool:
    """Check if type is a Vector, Matrix, or Tensor subclass."""
    # Handle generic aliases like Vector[Qubit], Matrix[Bit], etc.
    actual_type = getattr(t, "__origin__", t)

    return hasattr(actual_type, "__mro__") and any(
        c.__name__ in ("Vector", "Matrix", "Tensor", "ArrayBase")
        for c in getattr(actual_type, "__mro__", [])
    )


def _get_ndim(param_type) -> int:
    """Get the number of dimensions for an array type."""
    # Handle generic aliases like Vector[Qubit], Matrix[Bit], etc.
    actual_type = getattr(param_type, "__origin__", param_type)
    type_name = actual_type.__name__
    if "Vector" in type_name:
        return 1
    elif "Matrix" in type_name:
        return 2
    elif "Tensor" in type_name:
        return 3
    return 1


def handle_type_map(handle_type: type[Handle] | type) -> ValueType:
    """Map Handle type to ValueType."""
    # Handle Array types
    if is_array_type(handle_type):
        # For generic aliases like Vector[Bit], get element type from __args__
        element_type = None
        if hasattr(handle_type, "__args__") and handle_type.__args__:
            element_type = handle_type.__args__[0]
        else:
            element_type = getattr(handle_type, "element_type", None)

        if element_type:
            return handle_type_map(element_type)
        raise TypeError(f"Array type missing element_type: {handle_type}")

    # Ensure handle_type is a class before calling issubclass
    if not isinstance(handle_type, type):
        raise TypeError(f"Unsupported type annotation '{handle_type}'")

    if not issubclass(handle_type, Handle):
        if handle_type in TYPE_MAPPING:
            return TYPE_MAPPING[handle_type]()
        else:
            raise TypeError(f"Unsupported type annotation '{handle_type}'")

    if handle_type is UInt:
        return UIntType()
    elif handle_type is Float:
        return FloatType()
    elif handle_type is Bit:
        return BitType()
    elif handle_type is Qubit:
        return ir_types.QubitType()
    else:
        raise TypeError(f"Unsupported Handle type '{handle_type}'")


def create_dummy_handle(
    value_type: ValueType, name: str = "dummy", emit_init: bool = True
) -> Handle:
    """Create a dummy Handle instance based on ValueType.

    Args:
        value_type: The IR type for the value.
        name: Name for the value.
        emit_init: If True, emit QInitOperation for qubit types (requires active tracer).

    Used for creating input parameters during tracing.
    """
    if isinstance(value_type, ir_types.UIntType):
        # Mark as parameter so it can be bound at emit time
        return UInt(value=Value(type=value_type, name=name, params={"parameter": name}))
    elif isinstance(value_type, ir_types.FloatType):
        # Mark as parameter so it can be bound at emit time
        return Float(
            value=Value(type=value_type, name=name, params={"parameter": name})
        )
    elif isinstance(value_type, ir_types.BitType):
        return Bit(value=Value(type=value_type, name=name))
    elif isinstance(value_type, ir_types.QubitType):
        value = Value(type=value_type, name=name)
        if emit_init:
            qinit_op = QInitOperation(operands=[], results=[value])
            tracer = get_current_tracer()
            tracer.add_operation(qinit_op)
        return Qubit(value=value)
    else:
        raise TypeError(f"Unsupported ValueType '{value_type}'")


def create_dummy_input(
    param_type, name: str = "param", emit_init: bool = True
) -> Handle:
    """Create a dummy input based on parameter type annotation.

    Args:
        param_type: The type annotation for the parameter.
        name: Name for the value.
        emit_init: If True, emit QInitOperation for qubit arrays (default: True).
                   Set to False when creating BlockValue's internal dummy inputs.

    This creates input Handles for function parameters.
    """
    if is_array_type(param_type):
        # For Vector/Matrix/Tensor, create a placeholder instance with symbolic shape
        # For generic aliases like Vector[Qubit], get element type from __args__
        element_type = None
        if hasattr(param_type, "__args__") and param_type.__args__:
            element_type = param_type.__args__[0]
        else:
            element_type = getattr(param_type, "element_type", None)
        if element_type is None:
            raise TypeError(f"Array type missing element_type: {param_type}")

        element_ir_type = handle_type_map(element_type)

        # Determine number of dimensions (Vector=1, Matrix=2, Tensor=3)
        ndim = _get_ndim(param_type)

        # Create symbolic dimension Values (empty params = symbolic)
        shape_values = tuple(
            Value(type=UIntType(), name=f"{name}_dim{i}", params={})
            for i in range(ndim)
        )

        # Create ArrayValue with symbolic shape
        array_value = ArrayValue(
            type=element_ir_type,
            name=name,
            shape=shape_values,
        )

        # Create symbolic UInt handles for _shape
        shape_handles = tuple(UInt(value=dim_value) for dim_value in shape_values)

        # Create instance without calling __init__ (which requires size/shape)
        # For generic aliases like Vector[Qubit], use __origin__ to get the actual class
        actual_class = getattr(param_type, "__origin__", param_type)
        instance = object.__new__(actual_class)
        instance.value = array_value
        instance._shape = shape_handles  # Tuple of symbolic UInt
        instance._borrowed_indices = {}
        instance.parent = None
        instance.indices = ()
        instance.name = name
        instance.id = str(id(instance))
        instance._consumed = False
        instance.element_type = element_type  # Set element type for array access
        return instance

    # Scalar Handle types: map to ValueType first, then create dummy
    value_type = handle_type_map(param_type)
    return create_dummy_handle(value_type, name, emit_init)


def func_to_block(func: typing.Callable) -> BlockValue:
    """Convert a function to a BlockValue.

    Example:
        ```python
        def my_func(a: UInt, b: UInt) -> tuple[UInt]:
            c = a + b
            return (c, )

        block_value = func_to_block(my_func)
        ```
    """
    signature = inspect.signature(func)

    # Check type annotations ================

    # ========= Check Input Types =========
    input_types: dict[str, typing.Any] = {}
    for param in signature.parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' must have a type annotation")

        input_types[param.name] = handle_type_map(param.annotation)

    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError("Return type must have a type annotation")

    # ======== Check Output Types ========
    _output_type: ValueType | list[ValueType] | None = None
    if getattr(signature.return_annotation, "__origin__", None) is tuple:
        _output_type = []
        for ret_type in signature.return_annotation.__args__:
            _output_type.append(handle_type_map(ret_type))
    else:
        _output_type = handle_type_map(signature.return_annotation)
    # ======================= Check Input Types

    # Create dummy inputs from the original type annotations (preserving Array types)
    # Use emit_init=False to avoid emitting QInitOperation for BlockValue's internal inputs
    dummy_inputs = {
        name: create_dummy_input(param.annotation, name, emit_init=False)
        for name, param in signature.parameters.items()
    }

    # Trace the function execution to collect operations ========
    tracer = Tracer()
    with trace(tracer):
        result = func(**dummy_inputs)  # type: ignore

    operations = tracer.operations

    # Extract the input Values from the dummy Handles
    label_args = list(dummy_inputs.keys())
    input_values = [h.value for h in dummy_inputs.values()]

    # Extract return Values from result
    return_values: list[Value] = []
    if result is not None:
        if isinstance(result, tuple):
            for r in result:
                if hasattr(r, "value"):
                    return_values.append(r.value)
                else:
                    return_values.append(r)
        else:
            if hasattr(result, "value"):
                return_values.append(result.value)

    # Always emit ReturnOperation (even for void returns with empty operands)
    return_op = ReturnOperation(operands=return_values, results=[])
    tracer.add_operation(return_op)

    # Re-fetch operations after adding ReturnOperation
    operations = tracer.operations

    return BlockValue(
        name=func.__name__,
        label_args=label_args,
        input_values=input_values,
        return_values=return_values,
        operations=operations,
    )
