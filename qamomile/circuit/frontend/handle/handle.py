from __future__ import annotations

import abc
import copy
import dataclasses
import typing
import uuid

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.value import Value

if typing.TYPE_CHECKING:
    from .array import ArrayBase
    from .primitives import UInt


def _emit_binop(lhs: Value, rhs: Value, result: Value, kind: BinOpKind) -> None:
    """Emit a BinOp to the current tracer."""
    binop = BinOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(binop)


def _emit_compop(lhs: Value, rhs: Value, result: Value, kind: CompOpKind) -> None:
    """Emit a CompOp to the current tracer."""
    compop = CompOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(compop)


@dataclasses.dataclass
class Handle(abc.ABC):
    value: Value
    parent: "ArrayBase | None" = None
    indices: tuple["UInt", ...] = ()
    name: str | None = None
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    _consumed: bool = False

    def consume(self) -> "Handle":
        if self._consumed:
            raise RuntimeError(f"Handle {self} has already been consumed.")
        self._consumed = True

        # Use type(self) to preserve the actual subclass type (Qubit, UInt, etc.)
        cls = type(self)
        new_handle = object.__new__(cls)
        new_handle.value = self.value
        new_handle.parent = self.parent
        new_handle.indices = copy.copy(self.indices)
        new_handle.name = self.name
        new_handle.id = self.id
        new_handle._consumed = False
        return new_handle



class ArithmeticMixin:
    """Mixin providing arithmetic operations for numeric Handle types.

    Requires:
        - value: Value attribute
        - _make_result(): Method to create result Handle of same type
        - _coerce(): Method to convert Python literals to Handle
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = self.value # type: ignore

    def _coerce(self, other) -> "Handle":
        """Convert int/float to Handle if needed (to be implemented in subclass)."""
        raise NotImplementedError("_coerce must be implemented in subclass.")
    
    def _make_result(self) -> "Handle":
        """Create a result Handle for an operation (to be implemented in subclass)."""
        raise NotImplementedError("_make_result must be implemented in subclass.")

    def _make_float_result(self) -> "Handle":
        """Create a Float result for division operations (to be implemented in subclass)."""
        raise NotImplementedError("_make_float_result must be implemented in subclass.")

    def __add__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.ADD)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.SUB)
        return result

    def __rsub__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result.value, BinOpKind.SUB)
        return result

    def __mul__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.MUL)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        result = self._make_float_result()  # Division always returns Float
        _emit_binop(self.value, other.value, result.value, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other):
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(other.value, self.value, result.value, BinOpKind.DIV)
        return result


