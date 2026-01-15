import typing

import qamomile.circuit.ir.types as ir_type
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import Value

from .handle import Bit, Float, Handle, Qubit, UInt
from .tracer import get_current_tracer


@typing.overload
def uint(arg: int) -> UInt: ...
@typing.overload
def uint(arg: str) -> UInt: ...


def uint(arg: int | str) -> UInt:
    name = str(arg) if isinstance(arg, str) else "uint_const"
    value = Value(type=ir_type.UIntType(), name=name)
    if isinstance(arg, int):
        return UInt(value=value, init_value=arg)
    else:
        return UInt(name=arg, value=value)


@typing.overload
def float_(arg: float) -> Float: ...
@typing.overload
def float_(arg: str) -> Float: ...


def float_(arg: float | str) -> Float:
    name = str(arg) if isinstance(arg, str) else "float_const"
    value = Value(type=ir_type.FloatType(), name=name)
    if isinstance(arg, float):
        return Float(value=value, init_value=arg)
    elif isinstance(arg, str):
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
    name = str(arg) if isinstance(arg, str) else "bit_const"
    value = Value(type=ir_type.BitType(), name=name)
    if isinstance(arg, bool):
        return Bit(value=value, init_value=arg)
    elif isinstance(arg, str):
        return Bit(name=arg, value=value)
    elif isinstance(arg, int):
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
