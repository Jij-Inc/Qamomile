import inspect
import typing

import qamomile.circuit.ir.types as ir_types
from qamomile.circuit.frontend.handle.containers import Dict, Tuple
from qamomile.circuit.frontend.handle.hamiltonian import Observable
from qamomile.circuit.frontend.handle.primitives import (
    Bit,
    Float,
    Handle,
    Qubit,
    UInt,
)
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    DictType,
    FloatType,
    TupleType,
    UIntType,
    ValueType,
)
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value

TYPE_MAPPING: dict[typing.Any, typing.Any] = {
    int: UIntType,
    float: FloatType,
    bool: BitType,
}


def is_array_type(t: typing.Any) -> bool:
    """Check if type is a Vector, Matrix, or Tensor subclass."""
    # Handle generic aliases like Vector[Qubit], Matrix[Bit], etc.
    actual_type = getattr(t, "__origin__", t)

    return hasattr(actual_type, "__mro__") and any(
        c.__name__ in ("Vector", "Matrix", "Tensor", "ArrayBase")
        for c in getattr(actual_type, "__mro__", [])
    )


def _get_ndim(param_type: typing.Any) -> int:
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


def _array_element_type(param_type: typing.Any) -> typing.Any:
    """Return the element handle type of an array annotation, or ``None``.

    Args:
        param_type (Any): A frontend type annotation such as
            ``Vector[Float]`` or ``Float``.

    Returns:
        Any | None: The element handle type for an array annotation
            (``Float`` for ``Vector[Float]``), or ``None`` if
            ``param_type`` is not an array annotation.
    """
    if hasattr(param_type, "__args__") and param_type.__args__:
        return param_type.__args__[0]
    return getattr(param_type, "element_type", None)


def build_param_slots(
    signature: inspect.Signature,
    input_types: dict[str, typing.Any],
    *,
    parameters: list[str] | None = None,
    kwargs: dict[str, typing.Any] | None = None,
    qubit_sizes: dict[str, int] | None = None,
    bind_defaults: bool,
) -> tuple[ParamSlot, ...]:
    """Build a ``ParamSlot`` tuple for the classical arguments of a kernel.

    Mirrors the argument-classification logic in
    ``QKernel._create_traced_block`` so the resulting slot list reflects
    the same decisions that drive symbolic-vs-bound input creation.
    Classical scalar / array arguments are always included; a ``Dict``
    argument is included only when it is a runtime parameter (its slot
    carries a ``DictType``). Pure-quantum arguments, ``Tuple`` arguments,
    and compile-time-bound ``Dict`` arguments are excluded and live in
    ``Block.input_values`` instead.

    Args:
        signature (inspect.Signature): The kernel function's signature.
        input_types (dict[str, Any]): Resolved frontend type annotations
            keyed by argument name (typically
            ``QKernel.input_types`` or the equivalent computed in
            ``func_to_block``).
        parameters (list[str] | None): Names explicitly requested as
            runtime parameters via ``parameters=[...]``. ``None`` is
            treated as an empty list.
        kwargs (dict[str, Any] | None): Concrete values supplied via
            ``bindings`` / direct kwargs. ``None`` is treated as an
            empty dict.
        qubit_sizes (dict[str, int] | None): Optional mapping from
            ``Vector[Qubit]`` parameter names to their integer sizes;
            these are quantum inputs and are not included in the slot
            list.
        bind_defaults (bool): When ``True``, Python signature defaults
            are treated as ``COMPILE_TIME_BOUND`` with
            ``bound_value=default``. When ``False`` (e.g., the
            ``func_to_block`` path that does not bake in defaults),
            defaulted arguments stay ``RUNTIME_PARAMETER`` and the
            default appears only in ``ParamSlot.default``.

    Returns:
        tuple[ParamSlot, ...]: One slot per classical argument, in the
            order they appear in ``signature.parameters``.
    """
    parameters_set = set(parameters or ())
    kwargs_map = kwargs or {}
    qubit_sizes_set = set((qubit_sizes or {}).keys())

    slots: list[ParamSlot] = []
    for name, param in signature.parameters.items():
        param_type = input_types.get(name, param.annotation)

        # Skip pure-quantum arguments. These do not participate in the
        # classical parameter contract.
        if param_type is Qubit:
            continue
        if name in qubit_sizes_set:
            continue
        if is_array_type(param_type):
            elem_handle_type = _array_element_type(param_type)
            if elem_handle_type is Qubit:
                continue
        # Tuple arguments are purely structural (multi-index keys) and are
        # never runtime parameters, so they stay out of the manifest.
        if is_tuple_type(param_type):
            continue
        # A Dict kept as a runtime parameter DOES participate in the
        # contract: its per-key values are rebound per call, which is
        # exactly what the manifest exists to describe. Emit a slot whose
        # ``type`` is the ``DictType`` (it already captures the key and
        # value types, so no extra ParamSlot fields are needed). A
        # compile-time-bound Dict has no rebind role and — like Tuple —
        # stays out; its ``DictValue`` lives in ``Block.input_values``.
        if is_dict_type(param_type):
            if name not in parameters_set:
                continue
            dict_default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            slots.append(
                ParamSlot(
                    name=name,
                    type=handle_type_map(param_type),
                    kind=ParamKind.RUNTIME_PARAMETER,
                    ndim=0,
                    default=dict_default,
                )
            )
            continue

        # Decide the slot's kind. ``Observable`` semantics mirror the
        # tracer in ``QKernel._create_traced_block`` (see
        # ``qamomile/circuit/frontend/qkernel.py``): a scalar
        # ``Observable`` and an *unbound* ``Vector[Observable]`` are
        # always RUNTIME_PARAMETER (the value is supplied at execute
        # time and ``partial_eval`` cannot fold it). A *bound*
        # ``Vector[Observable]`` — i.e., one that appears in
        # ``kwargs_map`` — falls through to the ``name in kwargs_map``
        # branch below and is recorded as COMPILE_TIME_BOUND with the
        # concrete observable list.
        is_scalar_observable = param_type is Observable
        is_unbound_observable_array = (
            is_array_type(param_type)
            and _array_element_type(param_type) is Observable
            and name not in kwargs_map
        )

        if is_scalar_observable or is_unbound_observable_array:
            kind = ParamKind.RUNTIME_PARAMETER
            bound_value: typing.Any = None
        elif name in parameters_set:
            kind = ParamKind.RUNTIME_PARAMETER
            bound_value = None
        elif name in kwargs_map:
            kind = ParamKind.COMPILE_TIME_BOUND
            bound_value = kwargs_map[name]
        elif bind_defaults and param.default is not inspect.Parameter.empty:
            kind = ParamKind.COMPILE_TIME_BOUND
            bound_value = param.default
        else:
            kind = ParamKind.RUNTIME_PARAMETER
            bound_value = None

        ndim = _get_ndim(param_type) if is_array_type(param_type) else 0
        default = (
            param.default if param.default is not inspect.Parameter.empty else None
        )

        slots.append(
            ParamSlot(
                name=name,
                type=handle_type_map(param_type),
                kind=kind,
                ndim=ndim,
                default=default,
                bound_value=bound_value,
            )
        )

    return tuple(slots)


def is_tuple_type(t: typing.Any) -> bool:
    """Check if type is a Tuple handle type."""
    actual_type = getattr(t, "__origin__", t)
    if actual_type is Tuple:
        return True
    if hasattr(actual_type, "__mro__"):
        return any(c.__name__ == "Tuple" for c in actual_type.__mro__)
    return False


def is_dict_type(t: typing.Any) -> bool:
    """Check if type is a Dict handle type."""
    actual_type = getattr(t, "__origin__", t)
    if actual_type is Dict:
        return True
    if hasattr(actual_type, "__mro__"):
        return any(c.__name__ == "Dict" for c in actual_type.__mro__)
    return False


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

    # Handle Tuple types (e.g., Tuple[UInt, UInt])
    if is_tuple_type(handle_type):
        if hasattr(handle_type, "__args__") and handle_type.__args__:
            element_types = tuple(handle_type_map(t) for t in handle_type.__args__)
            return TupleType(element_types=element_types)
        raise TypeError(f"Tuple type missing element types: {handle_type}")

    # Handle Dict types (e.g., Dict[Tuple[UInt, UInt], Float])
    if is_dict_type(handle_type):
        if hasattr(handle_type, "__args__") and len(handle_type.__args__) == 2:
            key_type = handle_type_map(handle_type.__args__[0])
            value_type = handle_type_map(handle_type.__args__[1])
            return DictType(key_type=key_type, value_type=value_type)
        raise TypeError(f"Dict type missing key/value types: {handle_type}")

    # Ensure handle_type is a class before calling issubclass
    if not isinstance(handle_type, type):
        raise TypeError(f"Unsupported type annotation '{handle_type}'")

    if not issubclass(handle_type, Handle):
        if handle_type in TYPE_MAPPING:
            return TYPE_MAPPING[handle_type]()
        raise TypeError(f"Unsupported type annotation '{handle_type}'")

    if handle_type is UInt:
        return UIntType()
    elif handle_type is Float:
        return FloatType()
    elif handle_type is Bit:
        return BitType()
    elif handle_type is Qubit:
        return ir_types.QubitType()
    elif handle_type is Observable:
        return ObservableType()
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
        return UInt(value=Value(type=value_type, name=name).with_parameter(name))
    elif isinstance(value_type, ir_types.FloatType):
        # Mark as parameter so it can be bound at emit time
        return Float(value=Value(type=value_type, name=name).with_parameter(name))
    elif isinstance(value_type, ir_types.BitType):
        return Bit(value=Value(type=value_type, name=name))
    elif isinstance(value_type, ir_types.QubitType):
        value = Value(type=value_type, name=name)
        if emit_init:
            qinit_op = QInitOperation(operands=[], results=[value])
            tracer = get_current_tracer()
            tracer.add_operation(qinit_op)
        return Qubit(value=value)
    elif isinstance(value_type, ObservableType):
        # Observable parameters are provided via bindings
        return Observable(value=Value(type=value_type, name=name).with_parameter(name))
    else:
        raise TypeError(f"Unsupported ValueType '{value_type}'")


def create_dummy_input(
    param_type: typing.Any,
    name: str = "param",
    emit_init: bool = True,
    *,
    shape: tuple[int, ...] | None = None,
) -> Handle:
    """Create a dummy input based on parameter type annotation.

    Args:
        param_type (Any): The type annotation for the parameter.
        name (str): Name for the value.
        emit_init (bool): If True, emit QInitOperation for qubit arrays
            (default: True). Set to False when creating a nested Block's
            internal dummy inputs, or when the dummy will receive its
            qubits from a caller's CallBlockOperation.
        shape (tuple[int, ...] | None): Optional concrete shape for array
            types. When provided, the dummy array's shape Values carry
            compile-time constants instead of symbolic placeholders.
            Used by call-time sub-kernel specialization so that
            shape-dependent stdlib helpers (qft / iqft / qpe) resolve
            ``get_size`` to a concrete integer and emit the correct gate
            sequence. Ignored for non-array types. Default: None
            (symbolic shape).

    Returns:
        Handle: A frontend Handle wrapping a dummy Value or ArrayValue
            suitable for use as a function-parameter input during
            tracing.

    Raises:
        TypeError: If ``param_type`` is not a supported parameter type,
            or if a Tuple/array annotation is missing its element
            type(s).
    """
    # Handle Tuple types (e.g., Tuple[UInt, UInt])
    if is_tuple_type(param_type):
        if hasattr(param_type, "__args__") and param_type.__args__:
            # Create symbolic element Values
            element_values = []
            element_handles = []
            for i, elem_type in enumerate(param_type.__args__):
                elem_handle = create_dummy_input(elem_type, f"{name}_{i}", emit_init)
                element_handles.append(elem_handle)
                element_values.append(elem_handle.value)

            tuple_value = TupleValue(
                name=name,
                elements=tuple(element_values),
            ).with_parameter(name)

            # Create Tuple handle
            tuple_handle = object.__new__(Tuple)
            tuple_handle.value = tuple_value
            tuple_handle._elements = tuple(element_handles)
            tuple_handle.parent = None
            tuple_handle.indices = ()
            tuple_handle.name = name
            tuple_handle.id = str(id(tuple_handle))
            tuple_handle._consumed = False
            return tuple_handle
        raise TypeError(f"Tuple type missing element types: {param_type}")

    # Handle Dict types (e.g., Dict[Tuple[UInt, UInt], Float])
    if is_dict_type(param_type):
        if hasattr(param_type, "__args__") and len(param_type.__args__) == 2:
            dict_value = DictValue(
                name=name,
                entries=(),  # Entries will be bound at transpile time
            ).with_parameter(name)

            # Create Dict handle
            dict_handle = object.__new__(Dict)
            dict_handle.value = dict_value
            dict_handle._entries = []
            dict_handle._size = None
            dict_handle.parent = None
            dict_handle.indices = ()
            dict_handle.name = name
            dict_handle.id = str(id(dict_handle))
            dict_handle._consumed = False
            dict_handle._key_type = param_type.__args__[0]
            dict_handle._value_type = param_type.__args__[1]
            return dict_handle
        raise TypeError(f"Dict type missing key/value types: {param_type}")

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

        if shape is not None:
            if len(shape) != ndim:
                raise TypeError(
                    f"Concrete shape {shape!r} has {len(shape)} dimensions "
                    f"but parameter '{name}' of type {param_type} expects "
                    f"{ndim} dimension(s)."
                )
            shape_values = tuple(
                Value(type=UIntType(), name=f"{name}_dim{i}").with_const(dim)
                for i, dim in enumerate(shape)
            )
        else:
            # Symbolic dimensions for the standalone trace path.
            shape_values = tuple(
                Value(type=UIntType(), name=f"{name}_dim{i}") for i in range(ndim)
            )

        # Create ArrayValue with the resolved shape
        array_value = ArrayValue(
            type=element_ir_type,
            name=name,
            shape=shape_values,
        )

        # Emit QInitOperation for qubit arrays (consistent with scalar Qubit handling)
        if emit_init and isinstance(element_ir_type, ir_types.QubitType):
            qinit_op = QInitOperation(operands=[], results=[array_value])
            tracer = get_current_tracer()
            tracer.add_operation(qinit_op)

        # Create symbolic UInt handles for _shape
        shape_handles = tuple(UInt(value=dim_value) for dim_value in shape_values)

        # Create instance without calling __init__ (which requires size/shape)
        # For generic aliases like Vector[Qubit], use __origin__ to get the actual class
        actual_class = getattr(param_type, "__origin__", param_type)
        instance = object.__new__(actual_class)
        instance.value = array_value.with_parameter(name)
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


def _validate_returned_arrays(result: typing.Any) -> None:
    """Call ``validate_all_returned`` on quantum arrays in the trace result.

    The frontend's ``ArrayBase.validate_all_returned`` is otherwise only
    triggered by ``consume`` paths (measure / expval / passing to another
    kernel).  A kernel that takes a borrow and then returns the parent
    without consuming it — e.g. ``qv = q[0]; return q`` — escapes the
    consume-driven check entirely.  Running the validator on every
    returned quantum array handle at trace end closes that hole; it
    reuses the existing error machinery (``UnreturnedBorrowError``) so
    users see the same diagnostic they would from a consume failure.

    Args:
        result: Whatever ``func(**dummy_inputs)`` returned — a Handle,
            a tuple of Handles, or ``None`` for void.  Non-quantum
            arrays and scalars are silently skipped.
    """
    from qamomile.circuit.frontend.handle.array import ArrayBase

    def _visit(obj: typing.Any) -> None:
        if obj is None:
            return
        if isinstance(obj, tuple):
            for item in obj:
                _visit(item)
            return
        if isinstance(obj, ArrayBase) and obj.value.type.is_quantum():
            obj.validate_all_returned()

    _visit(result)


def func_to_block(func: typing.Callable) -> Block:
    """Convert a function to a hierarchical Block.

    Example:
        ```python
        def my_func(a: UInt, b: UInt) -> tuple[UInt]:
            c = a + b
            return (c, )

        block = func_to_block(my_func)
        ```
    """
    signature = inspect.signature(func)

    # Resolve type hints to handle string annotations (from __future__ import annotations)
    try:
        func_globals = getattr(func, "__globals__", {})
        type_hints = typing.get_type_hints(func, globalns=func_globals, localns=None)
    except Exception:
        # Fallback to raw annotations if get_type_hints fails
        type_hints = {}
        for param in signature.parameters.values():
            if param.annotation is not inspect.Parameter.empty:
                type_hints[param.name] = param.annotation
        if signature.return_annotation is not inspect.Signature.empty:
            type_hints["return"] = signature.return_annotation

    # Check type annotations ================

    # ========= Check Input Types =========
    input_types: dict[str, typing.Any] = {}
    for param in signature.parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' must have a type annotation")

        # Use resolved type hint instead of raw annotation
        param_type = type_hints.get(param.name, param.annotation)
        input_types[param.name] = handle_type_map(param_type)

    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError("Return type must have a type annotation")

    # ======== Check Output Types ========
    # Use resolved return type hint instead of raw annotation
    return_type = type_hints.get("return", signature.return_annotation)
    _output_type: ValueType | list[ValueType] | None = None
    if getattr(return_type, "__origin__", None) is tuple:
        _output_type = []
        for ret_type in return_type.__args__:
            _output_type.append(handle_type_map(ret_type))
    else:
        _output_type = handle_type_map(return_type)
    # ======================= Check Input Types

    # Create dummy inputs from resolved type hints (preserving Array types)
    # Use emit_init=False to avoid emitting QInitOperation for nested Block inputs
    dummy_inputs = {
        name: create_dummy_input(
            type_hints.get(name, param.annotation), name, emit_init=False
        )
        for name, param in signature.parameters.items()
    }

    # Trace the function execution to collect operations ========
    tracer = Tracer()
    with trace(tracer):
        result = func(**dummy_inputs)  # type: ignore

    # Validate that returned / live quantum arrays have no unreturned
    # borrows.  The existing ``validate_all_returned`` is consume-driven
    # (fires on measure / expval / passing to another kernel), so a
    # kernel that does ``qv = q[0]; return q`` — taking a borrow and
    # returning the parent without consuming it — would silently pass.
    # Running the check explicitly at trace end closes that hole
    # without needing a dedicated consume operation.
    _validate_returned_arrays(result)

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

    # ``input_types`` above already maps each name to a converted
    # ``ValueType``; ``build_param_slots`` needs the raw handle-side type
    # annotations (e.g., ``Vector[Float]``) so it can introspect array
    # element types and dimensionality. Pass ``type_hints`` directly.
    param_slots = build_param_slots(
        signature=signature,
        input_types=type_hints,
        bind_defaults=False,
    )

    return Block(
        name=func.__name__,
        label_args=label_args,
        input_values=input_values,
        output_values=return_values,
        operations=operations,
        kind=BlockKind.HIERARCHICAL,
        param_slots=param_slots,
    )
