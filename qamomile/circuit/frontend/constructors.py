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
    """Create a UInt handle from an integer literal or declare a named UInt parameter."""
    name = str(arg) if isinstance(arg, str) else "uint_const"
    if isinstance(arg, int):
        value = Value(type=ir_type.UIntType(), name=name).with_const(arg)
        return UInt(value=value, init_value=arg)
    else:
        value = Value(type=ir_type.UIntType(), name=name)
        return UInt(name=arg, value=value)


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
        ValueError: If ``shape`` is an empty tuple.
        NotImplementedError: If ``shape`` has more than one dimension.
            The quantum addressing path is rank-1, so a higher-rank
            register would silently alias distinct elements onto the
            same physical qubit. Allocate a 1-D ``Vector[Qubit]`` of
            the total size and compute flat indices explicitly instead
            (e.g. ``q[i * ncols + j]``).
    """

    if not isinstance(shape, tuple):
        shape = (shape,)

    # ``len()`` is read into a variable so zuban does not narrow the
    # variadic tuple to fixed-length forms that make ``shape[0]`` look
    # out of range below.
    ndim = len(shape)

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

    dim = shape[0] if isinstance(shape[0], UInt) else uint(shape[0])

    array_value = ArrayValue(type=ir_type.QubitType(), name=name, shape=(dim.value,))

    return Vector[Qubit](value=array_value, _shape=(dim,))
