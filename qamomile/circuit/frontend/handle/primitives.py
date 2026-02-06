from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind, CompOpKind
from qamomile.circuit.ir.types import QFixedType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import Value

from .handle import ArithmeticMixin, Handle, _emit_binop, _emit_compop


@dataclasses.dataclass
class Qubit(Handle):
    value: Value[QubitType]


@dataclasses.dataclass
class QFixed(Handle):
    value: Value[QFixedType]


@dataclasses.dataclass
class UInt(ArithmeticMixin, Handle):
    """Unsigned integer handle with arithmetic operations."""

    init_value: int = 0

    def _make_uint(self, val: int) -> "UInt":
        """Create a UInt from an integer constant."""
        return UInt(
            value=Value(type=UIntType(), name="uint_const", params={"const": val}),
            init_value=val,
        )

    def _make_result(self) -> "UInt":
        """Create a UInt result for an operation (required by ArithmeticMixin)."""
        return UInt(value=Value(type=UIntType(), name="uint_tmp"), init_value=0)

    def _make_float_result(self) -> "Float":
        """Create a Float result for division operations (required by ArithmeticMixin)."""
        return Float(value=Value(type=FloatType(), name="float_tmp"), init_value=0.0)

    def _make_bit(self) -> "Bit":
        """Create a Bit result for a comparison."""
        return Bit(value=Value(type=BitType(), name="bit_tmp"))

    def _coerce(self, other: int | float | UInt | Float) -> "UInt | Float":
        """Convert int/float to Handle if needed (required by ArithmeticMixin)."""
        if isinstance(other, int):
            return self._make_uint(other)
        if isinstance(other, float):
            return Float(
                value=Value(
                    type=FloatType(), name="float_const", params={"const": other}
                ),
                init_value=other,
            )
        return other

    def _result_for(self, other: "UInt | Float") -> "UInt | Float":
        """Create the appropriate result type based on the coerced operand."""
        if isinstance(other, Float):
            return self._make_float_result()
        return self._make_result()

    def __lt__(self, other) -> "Bit":
        if isinstance(other, int):
            other = self._make_uint(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.LT)
        return result

    def __gt__(self, other) -> "Bit":
        if isinstance(other, int):
            other = self._make_uint(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.GT)
        return result

    def __le__(self, other) -> "Bit":
        if isinstance(other, int):
            other = self._make_uint(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.LE)
        return result

    def __ge__(self, other) -> "Bit":
        if isinstance(other, int):
            other = self._make_uint(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.GE)
        return result

    def __add__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(self.value, other.value, result.value, BinOpKind.ADD)
        return result

    def __radd__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        return self.__add__(other)

    def __sub__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(self.value, other.value, result.value, BinOpKind.SUB)
        return result

    def __rsub__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(other.value, self.value, result.value, BinOpKind.SUB)
        return result

    def __mul__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(self.value, other.value, result.value, BinOpKind.MUL)
        return result

    def __rmul__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        return self.__mul__(other)

    def __truediv__(self, other: "int | float | UInt | Float") -> "Float":
        """UInt true division always returns Float."""
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other: "int | float | UInt | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(other.value, self.value, result.value, BinOpKind.DIV)
        return result

    def __floordiv__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(self.value, other.value, result.value, BinOpKind.FLOORDIV)
        return result

    def __rfloordiv__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(other.value, self.value, result.value, BinOpKind.FLOORDIV)
        return result

    def __pow__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(self.value, other.value, result.value, BinOpKind.POW)
        return result

    def __rpow__(self, other: "int | float | UInt | Float") -> "UInt | Float":
        other = self._coerce(other)
        result = self._result_for(other)
        _emit_binop(other.value, self.value, result.value, BinOpKind.POW)
        return result


@dataclasses.dataclass
class Float(ArithmeticMixin, Handle):
    """Floating-point handle with arithmetic operations."""

    init_value: float = 0.0

    def _make_float(self, val: float) -> "Float":
        """Create a Float from a constant."""
        return Float(
            value=Value(type=FloatType(), name="float_const", params={"const": val}),
            init_value=val,
        )

    def _make_result(self) -> "Float":
        """Create a Float result for an operation (required by ArithmeticMixin)."""
        return Float(value=Value(type=FloatType(), name="float_tmp"), init_value=0.0)

    def _make_float_result(self) -> "Float":
        """Create a Float result for division (same as _make_result for Float)."""
        return self._make_result()

    def _coerce(self, other: "int | float | Float") -> "Float":
        """Convert int or float to Float if needed (required by ArithmeticMixin)."""
        if isinstance(other, (int, float)):
            return self._make_float(float(other))
        return other

    # Override arithmetic operations with proper type hints for Float
    def __add__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.ADD)
        return result

    def __radd__(self, other: "int | float | Float") -> "Float":
        return self.__add__(other)

    def __sub__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.SUB)
        return result

    def __rsub__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result.value, BinOpKind.SUB)
        return result

    def __mul__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.MUL)
        return result

    def __rmul__(self, other: "int | float | Float") -> "Float":
        return self.__mul__(other)

    def __truediv__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(other.value, self.value, result.value, BinOpKind.DIV)
        return result


@dataclasses.dataclass
class Bit(Handle):
    init_value: bool = False
