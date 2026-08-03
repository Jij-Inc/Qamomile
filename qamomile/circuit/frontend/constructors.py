import typing

import qamomile.circuit.ir.types as ir_type
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value

from .handle import (
    Bit,
    Float,
    Qubit,
    UInt,
    Vector,
)
from .tracer import get_current_tracer


@typing.overload
def uint(arg: int) -> UInt: ...
@typing.overload
def uint(arg: str) -> UInt: ...


def uint(arg: int | str) -> UInt:
    """Create a UInt handle from an integer literal or a named parameter.

    Args:
        arg (int | str): An integer literal to bake in as a compile-time
            constant, or a ``str`` naming a symbolic UInt parameter. A
            ``bool`` is rejected: ``True`` / ``False`` are not valid integer
            values here even though ``bool`` subclasses ``int``. (Sign is
            not validated here -- a negative literal is accepted and baked
            in as-is.)

    Returns:
        UInt: A constant-valued handle for an ``int`` argument, or a named
            symbolic handle for a ``str`` argument.

    Raises:
        TypeError: If ``arg`` is neither a plain ``int`` nor a ``str``
            (in particular, if it is a ``bool``).
    """
    # Reject bool first with an explicit guard (not is_plain_int): is_plain_int
    # alone would send a bool to the ``str`` (named-parameter) branch below and
    # build a malformed UInt(name=True) rather than raising.
    if isinstance(arg, bool):
        raise TypeError(f"uint() argument must be an int or str, got bool ({arg}).")
    if isinstance(arg, int):
        value = Value(type=ir_type.UIntType(), name="uint_const").with_const(arg)
        return UInt(value=value, init_value=arg)
    if isinstance(arg, str):
        value = Value(type=ir_type.UIntType(), name=arg)
        return UInt(name=arg, value=value)
    raise TypeError(f"uint() argument must be an int or str, got {type(arg).__name__}.")


@typing.overload
def float_(arg: float) -> Float: ...
@typing.overload
def float_(arg: str) -> Float: ...


def float_(arg: float | str) -> Float:
    """Create a Float handle from a float literal or declare a named Float parameter."""
    name = str(arg) if isinstance(arg, str) else "float_const"
    if isinstance(arg, float):
        value = Value(type=ir_type.FloatType(), name=name).with_const(arg)
        return Float(value=value, init_value=arg)
    elif isinstance(arg, str):
        value = Value(type=ir_type.FloatType(), name=name)
        return Float(name=arg, value=value)
    else:
        raise TypeError("Argument must be of type float or str")


@typing.overload
def bit(arg: bool) -> Bit: ...
@typing.overload
def bit(arg: str) -> Bit: ...
@typing.overload
def bit(arg: int) -> Bit: ...


def bit(arg: bool | str | int) -> Bit:
    """Create a Bit handle from a boolean/int literal or declare a named Bit parameter."""
    name = str(arg) if isinstance(arg, str) else "bit_const"
    if isinstance(arg, bool):
        value = Value(type=ir_type.BitType(), name=name).with_const(arg)
        return Bit(value=value, init_value=arg)
    elif isinstance(arg, str):
        value = Value(type=ir_type.BitType(), name=name)
        return Bit(name=arg, value=value)
    elif isinstance(arg, int):
        value = Value(type=ir_type.BitType(), name=name).with_const(bool(arg))
        return Bit(value=value, init_value=bool(arg))
    else:
        raise TypeError("Argument must be of type bool, str, or int")


@typing.overload
def bit_array(shape: int | UInt, name: str = "bits") -> Vector[Bit]: ...


@typing.overload
def bit_array(shape: tuple[int | UInt], name: str = "bits") -> Vector[Bit]: ...


def bit_array(
    shape: UInt | int | tuple[UInt | int, ...],
    name: str = "bits",
) -> Vector[Bit]:
    """Create a fixed-length classical bit vector initialized to zero.

    The vector is represented as an initialized IR constant array, so it can
    receive measured ``Bit`` values through ordinary element assignment and
    can be returned from a qkernel. Its length must be known while tracing;
    the emitted classical initializer materializes contents, not a dynamic
    array shape.

    Args:
        shape (UInt | int | tuple[UInt | int, ...]): Number of bits in the
            vector, given either as a scalar or a one-element tuple. A
            ``UInt`` must resolve to a compile-time constant.
        name (str): Display name for the underlying array value. Defaults to
            ``"bits"``.

    Returns:
        Vector[Bit]: A fixed-length vector whose elements are initialized to
            ``False``.

    Raises:
        TypeError: If ``shape`` or ``name`` has the wrong type.
        ValueError: If the shape is empty, negative, or not known at trace
            time.
        NotImplementedError: If a shape with more than one dimension is
            requested.

    Example:
        >>> import qamomile.circuit as qmc
        >>>
        >>> @qmc.qkernel
        ... def readout() -> qmc.Vector[qmc.Bit]:
        ...     qubits = qmc.qubit_array(2, "qubits")
        ...     measured = qmc.measure(qubits)
        ...     bits = qmc.bit_array(2)
        ...     bits[0] = measured[0]
        ...     bits[1] = measured[1]
        ...     return bits
    """
    raw_shape: typing.Any = shape
    raw_name: typing.Any = name
    if not isinstance(raw_name, str):
        raise TypeError(f"bit_array name must be a str, got {type(raw_name).__name__}.")
    if isinstance(raw_shape, bool) or not isinstance(raw_shape, (int, UInt, tuple)):
        raise TypeError(
            "bit_array shape must be an int, UInt, or a tuple containing "
            f"one such value, got {type(raw_shape).__name__}."
        )

    normalized_shape = raw_shape if isinstance(raw_shape, tuple) else (raw_shape,)
    ndim = len(normalized_shape)
    if ndim == 0:
        raise ValueError("bit_array shape must contain one dimension.")
    if ndim > 1:
        raise NotImplementedError(
            f"bit_array does not support rank-{ndim} shapes; use a 1-D "
            "Vector[Bit] and compute flat indices explicitly."
        )

    raw_size = normalized_shape[0]
    if isinstance(raw_size, bool) or not isinstance(raw_size, (int, UInt)):
        raise TypeError(
            "bit_array shape tuple must contain one int or UInt, got "
            f"{type(raw_size).__name__}."
        )
    dim = raw_size if isinstance(raw_size, UInt) else uint(raw_size)
    const_size = dim.value.get_const()
    if const_size is None:
        raise ValueError(
            "bit_array shape must be known at trace time; bind the UInt size "
            "through `bindings` instead of keeping it in `parameters`."
        )
    size = int(const_size)
    if size < 0:
        raise ValueError(f"bit_array shape must be non-negative, got {size}.")

    array_value = ArrayValue(
        type=ir_type.BitType(),
        name=name,
        shape=(dim.value,),
    ).with_array_runtime_metadata(const_array=(False,) * size)
    return Vector[Bit](value=array_value, _shape=(dim,))


def qubit(name: str) -> Qubit:
    """Create a new qubit and emit a QInitOperation."""
    value = Value(type=ir_type.QubitType(), name=name)

    # Emit QInitOperation to register qubit allocation
    qinit_op = QInitOperation(operands=[], results=[value])
    tracer = get_current_tracer()
    tracer.add_operation(qinit_op)

    return Qubit(value=value)


@typing.overload
def qubit_array(shape: int | UInt, name: str) -> Vector[Qubit]: ...


@typing.overload
def qubit_array(shape: tuple[int | UInt], name: str) -> Vector[Qubit]: ...


def qubit_array(
    shape: UInt | int | tuple[UInt | int, ...],
    name: str,
) -> Vector[Qubit]:
    """Create a new 1-D qubit register and emit its QInitOperation.

    Args:
        shape (UInt | int | tuple[UInt | int, ...]): Number of qubits in
            the register, given either as a scalar or as a 1-tuple.
            Tuples with more than one dimension are rejected (see
            Raises).
        name (str): Name for the underlying ArrayValue.

    Returns:
        Vector[Qubit]: A 1-D quantum register handle of the requested
            size.

    Raises:
        TypeError: If ``shape`` or ``name`` has the wrong type.
        ValueError: If ``shape`` is an empty tuple.
        NotImplementedError: If ``shape`` has more than one dimension.
            The quantum addressing path is rank-1, so a higher-rank
            register would silently alias distinct elements onto the
            same physical qubit. Allocate a 1-D ``Vector[Qubit]`` of
            the total size and compute flat indices explicitly instead
            (e.g. ``q[i * ncols + j]``).
    """

    raw_shape: typing.Any = shape
    raw_name: typing.Any = name
    if not isinstance(raw_name, str):
        if isinstance(raw_shape, str):
            raise TypeError(
                "qubit_array expects (shape, name), but received arguments "
                f"that look swapped: ({raw_shape!r}, {raw_name!r})."
            )
        raise TypeError(
            f"qubit_array name must be a str, got {type(raw_name).__name__}."
        )
    if isinstance(raw_shape, bool) or not isinstance(raw_shape, (int, UInt, tuple)):
        raise TypeError(
            "qubit_array shape must be an int, UInt, or a tuple containing "
            f"one such value, got {type(raw_shape).__name__}."
        )
    normalized_shape = raw_shape if isinstance(raw_shape, tuple) else (raw_shape,)

    # ``len()`` is read into a variable so zuban does not narrow the
    # variadic tuple to fixed-length forms that make ``shape[0]`` look
    # out of range below.
    ndim = len(normalized_shape)

    if ndim == 0:
        raise ValueError("Shape must have at least one dimension.")

    if ndim > 1:
        raise NotImplementedError(
            f"qubit_array does not support rank-{ndim} shapes: the "
            f"quantum addressing path is rank-1, so a higher-rank register "
            f"would silently alias distinct elements onto the same physical "
            f"qubit. Allocate a 1-D Vector[Qubit] of the total size and "
            f"compute flat indices explicitly (e.g. q[i * ncols + j])."
        )

    if isinstance(normalized_shape[0], bool) or not isinstance(
        normalized_shape[0], (int, UInt)
    ):
        raise TypeError(
            "qubit_array shape tuple must contain one int or UInt, got "
            f"{type(normalized_shape[0]).__name__}."
        )
    if isinstance(normalized_shape[0], int) and normalized_shape[0] < 0:
        raise ValueError(
            f"qubit_array shape must be non-negative, got {normalized_shape[0]}."
        )

    dim = (
        normalized_shape[0]
        if isinstance(normalized_shape[0], UInt)
        else uint(normalized_shape[0])
    )

    array_value = ArrayValue(type=ir_type.QubitType(), name=name, shape=(dim.value,))

    return Vector[Qubit](value=array_value, _shape=(dim,))
