from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOpKind,
    CompOpKind,
    CondOpKind,
)
from qamomile.circuit.ir.types import QFixedType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import Value

from .handle import (
    ArithmeticMixin,
    Handle,
    _emit_binop,
    _emit_compop,
    _emit_condop,
    _emit_notop,
)


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
            value=Value(type=UIntType(), name="").with_const(val),
            init_value=val,
        )

    def _make_result(self) -> "UInt":
        """Create a UInt result for an operation (required by ArithmeticMixin)."""
        return UInt(value=Value(type=UIntType(), name=""), init_value=0)

    def _make_float_result(self) -> "Float":
        """Create a Float result for division operations (required by ArithmeticMixin)."""
        return Float(value=Value(type=FloatType(), name=""), init_value=0.0)

    def _make_bit(self) -> "Bit":
        """Create a Bit result for a comparison."""
        return Bit(value=Value(type=BitType(), name=""))

    def _coerce(self, other: int | float | UInt | Float) -> "UInt | Float":
        """Convert int/float to Handle if needed (required by ArithmeticMixin)."""
        if isinstance(other, int):
            return self._make_uint(other)
        if isinstance(other, float):
            return Float(
                value=Value(type=FloatType(), name="").with_const(other),
                init_value=other,
            )
        return other

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

    def __eq__(self, other) -> "Bit":  # type: ignore[override]
        if isinstance(other, bool):
            other = self._make_uint(int(other))
        elif isinstance(other, int):
            other = self._make_uint(other)
        elif not isinstance(other, UInt):
            return NotImplemented  # type: ignore[return-value]
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.EQ)
        return result

    def __ne__(self, other) -> "Bit":  # type: ignore[override]
        if isinstance(other, bool):
            other = self._make_uint(int(other))
        elif isinstance(other, int):
            other = self._make_uint(other)
        elif not isinstance(other, UInt):
            return NotImplemented  # type: ignore[return-value]
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.NEQ)
        return result

    # Override __eq__ above returns Bit (for DSL semantics), so we have to
    # restore a hashable object identity or dataclass equality-based
    # hashing would break anything that puts UInt handles into dicts/sets.
    __hash__ = object.__hash__  # type: ignore[assignment]

    # Override arithmetic operations with proper type hints for UInt
    def __add__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result.value, BinOpKind.ADD)
        return result

    def __radd__(self, other: "int | UInt") -> "UInt":
        return self.__add__(other)

    def __sub__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result.value, BinOpKind.SUB)
        return result

    def __rsub__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result.value, BinOpKind.SUB)
        return result

    # UInt-specific multiplication (handles float case differently)
    def __mul__(self, other) -> "UInt | Float":
        if isinstance(other, float):
            other_value = Value(type=FloatType(), name="").with_const(other)
            result = self._make_float_result()
            _emit_binop(self.value, other_value, result.value, BinOpKind.MUL)
            return result

        # Use mixin's default implementation for int/UInt
        return super().__mul__(other)  # type: ignore

    def __rmul__(self, other) -> "UInt | Float":
        return self.__mul__(other)

    # UInt-specific true division (always returns Float)
    def __truediv__(self, other: "int | float | UInt | Float") -> "Float":
        coerced = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(self.value, coerced.value, result.value, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other: "int | float | UInt | Float") -> "Float":
        coerced = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(coerced.value, self.value, result.value, BinOpKind.DIV)
        return result

    # UInt-specific floor division
    def __floordiv__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result.value, BinOpKind.FLOORDIV)
        return result

    def __rfloordiv__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result.value, BinOpKind.FLOORDIV)
        return result

    # UInt-specific power operation
    def __pow__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result.value, BinOpKind.POW)
        return result

    def __rpow__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result.value, BinOpKind.POW)
        return result


@dataclasses.dataclass
class Float(ArithmeticMixin, Handle):
    """Floating-point handle with arithmetic operations."""

    init_value: float = 0.0

    def _make_float(self, val: float) -> "Float":
        """Create a Float from a constant."""
        return Float(
            value=Value(type=FloatType(), name="").with_const(val),
            init_value=val,
        )

    def _make_result(self) -> "Float":
        """Create a Float result for an operation (required by ArithmeticMixin)."""
        return Float(value=Value(type=FloatType(), name=""), init_value=0.0)

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

    def __pow__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result.value, BinOpKind.POW)
        return result

    def __rpow__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result.value, BinOpKind.POW)
        return result

    def _make_bit(self) -> "Bit":
        """Create a Bit result for a comparison."""
        return Bit(value=Value(type=BitType(), name=""))

    def __lt__(self, other) -> "Bit":
        other = self._coerce(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.LT)
        return result

    def __gt__(self, other) -> "Bit":
        other = self._coerce(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.GT)
        return result

    def __le__(self, other) -> "Bit":
        other = self._coerce(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.LE)
        return result

    def __ge__(self, other) -> "Bit":
        other = self._coerce(other)
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.GE)
        return result

    def __eq__(self, other) -> "Bit":  # type: ignore[override]
        if isinstance(other, (int, float, bool)) and not isinstance(other, Float):
            other = self._make_float(float(other))
        elif not isinstance(other, Float):
            return NotImplemented  # type: ignore[return-value]
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.EQ)
        return result

    def __ne__(self, other) -> "Bit":  # type: ignore[override]
        if isinstance(other, (int, float, bool)) and not isinstance(other, Float):
            other = self._make_float(float(other))
        elif not isinstance(other, Float):
            return NotImplemented  # type: ignore[return-value]
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.NEQ)
        return result

    __hash__ = object.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class Bit(Handle):
    init_value: bool = False

    def _make_bit(self) -> "Bit":
        """Create a fresh result Bit handle for an op."""
        return Bit(value=Value(type=BitType(), name=""))

    def _coerce(self, other: "bool | int | Bit") -> "Bit":
        """Promote ``bool`` / ``int`` (0 or 1) to a constant Bit handle.

        Args:
            other: A Python ``bool``, integer ``0``/``1``, or another Bit.

        Returns:
            A Bit handle. Constants are wrapped in a Value with the
            corresponding boolean constant baked in.

        Raises:
            TypeError: If ``other`` is not a Bit, bool, or 0/1 int.
        """
        if isinstance(other, Bit):
            return other
        if isinstance(other, bool):
            return Bit(
                value=Value(type=BitType(), name="").with_const(other),
                init_value=other,
            )
        if isinstance(other, int) and other in (0, 1):
            return Bit(
                value=Value(type=BitType(), name="").with_const(bool(other)),
                init_value=bool(other),
            )
        raise TypeError(
            f"Bit logical op operand must be Bit, bool, or 0/1 int; got "
            f"{type(other).__name__}"
        )

    def __and__(self, other: "bool | int | Bit") -> "Bit":
        other_bit = self._coerce(other)
        result = self._make_bit()
        _emit_condop(self.value, other_bit.value, result.value, CondOpKind.AND)
        return result

    def __rand__(self, other: "bool | int") -> "Bit":
        other_bit = self._coerce(other)
        result = self._make_bit()
        _emit_condop(other_bit.value, self.value, result.value, CondOpKind.AND)
        return result

    def __or__(self, other: "bool | int | Bit") -> "Bit":
        other_bit = self._coerce(other)
        result = self._make_bit()
        _emit_condop(self.value, other_bit.value, result.value, CondOpKind.OR)
        return result

    def __ror__(self, other: "bool | int") -> "Bit":
        other_bit = self._coerce(other)
        result = self._make_bit()
        _emit_condop(other_bit.value, self.value, result.value, CondOpKind.OR)
        return result

    def __invert__(self) -> "Bit":
        result = self._make_bit()
        _emit_notop(self.value, result.value)
        return result
