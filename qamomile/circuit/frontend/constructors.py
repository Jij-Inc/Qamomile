import typing

import qamomile.circuit.ir.types as ir_type
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value

from .handle import (
    Bit,
    Float,
    Matrix,
    Qubit,
    Tensor,
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
def qubit_array(shape: tuple[int | UInt, int | UInt], name: str) -> Matrix[Qubit]: ...


@typing.overload
def qubit_array(shape: tuple[int | UInt, ...], name: str) -> Tensor[Qubit]: ...


def qubit_array(
    shape: UInt | int | tuple[UInt | int, ...],
    name: str,
) -> Vector[Qubit] | Matrix[Qubit] | Tensor[Qubit]:
    """Create a new qubit array (vector/matrix/tensor) and emit QInitOperations."""

    if not isinstance(shape, tuple):
        shape = (shape,)

    if len(shape) == 0:
        raise ValueError("Shape must have at least one dimension.")

    dims: list[UInt] = [dim if isinstance(dim, UInt) else uint(dim) for dim in shape]

    raw_shape_values = tuple(d.value for d in dims)

    array_value = ArrayValue(
        type=ir_type.QubitType(), name=name, shape=raw_shape_values
    )

    shape_tuple = tuple(dims)
    ndim = len(dims)

    if ndim == 1:
        return Vector[Qubit](value=array_value, _shape=(shape_tuple[0],))
    elif ndim == 2:
        return Matrix[Qubit](value=array_value, _shape=(shape_tuple[0], shape_tuple[1]))
    else:
        return Tensor[Qubit](value=array_value, _shape=shape_tuple)
