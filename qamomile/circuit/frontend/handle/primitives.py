from __future__ import annotations

import dataclasses
from typing import overload

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
        _emit_binop(self.value, coerced.value, result, BinOpKind.ADD)
        return result

    def __radd__(self, other: "int | UInt") -> "UInt":
        return self.__add__(other)

    def __sub__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result, BinOpKind.SUB)
        return result

    def __rsub__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result, BinOpKind.SUB)
        return result

    # UInt-specific multiplication (handles float case differently).
    # The result type is fully determined by the operand at runtime, so
    # @overload narrows it: integral operands keep the result a UInt
    # (valid as a Vector index), only a Python float widens it to Float.
    @overload
    def __mul__(self, other: "int | UInt") -> "UInt": ...

    @overload
    def __mul__(self, other: float) -> "Float": ...

    def __mul__(self, other) -> "UInt | Float":
        """Multiply this UInt by an integer-like or float operand.

        Args:
            other (int | UInt | float): The right-hand operand. An ``int``
                or ``UInt`` keeps the product unsigned-integral; a Python
                ``float`` promotes the product to a floating-point result.

        Returns:
            UInt | Float: A ``UInt`` when ``other`` is ``int``/``UInt``,
                or a ``Float`` when ``other`` is a Python ``float``. The
                two cases are distinguished statically via ``@overload``.
        """
        if isinstance(other, float):
            other_value = Value(type=FloatType(), name="").with_const(other)
            result = self._make_float_result()
            _emit_binop(self.value, other_value, result, BinOpKind.MUL)
            return result

        # Use mixin's default implementation for int/UInt
        return super().__mul__(other)  # type: ignore

    @overload
    def __rmul__(self, other: "int | UInt") -> "UInt": ...

    @overload
    def __rmul__(self, other: float) -> "Float": ...

    def __rmul__(self, other) -> "UInt | Float":
        """Multiply an integer-like or float operand by this UInt.

        Multiplication is commutative, so this delegates to ``__mul__``.

        Args:
            other (int | UInt | float): The left-hand operand.

        Returns:
            UInt | Float: A ``UInt`` for ``int``/``UInt`` operands, a
                ``Float`` for a Python ``float`` operand.
        """
        return self.__mul__(other)

    # UInt-specific true division (always returns Float)
    def __truediv__(self, other: "int | float | UInt | Float") -> "Float":
        coerced = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(self.value, coerced.value, result, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other: "int | float | UInt | Float") -> "Float":
        coerced = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(coerced.value, self.value, result, BinOpKind.DIV)
        return result

    # UInt-specific floor division
    def __floordiv__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result, BinOpKind.FLOORDIV)
        return result

    def __rfloordiv__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result, BinOpKind.FLOORDIV)
        return result

    # UInt-specific modulo
    def __mod__(self, other: "int | UInt") -> "UInt":
        """Take this UInt modulo an integer-like operand.

        Mirrors the other ``UInt`` arithmetic operators by emitting a
        ``BinOp(MOD)`` whose result is a fresh ``UInt`` handle. When both
        operands are compile-time constants the result is folded eagerly
        (``_emit_binop``), so ``i % k`` is a valid index/loop-bound
        expression and ``i % k == 0`` is a valid compile-time ``if``
        predicate once the enclosing loop is unrolled.

        Like the sibling integer operators (``__floordiv__`` / ``__pow__``),
        the ``int | UInt`` operand contract is enforced statically by the
        type checker; passing a non-integral operand (e.g. a Python
        ``float``) is a type error rather than a runtime branch here.

        Args:
            other (int | UInt): The divisor. A Python ``int`` is coerced
                to a constant ``UInt`` operand.

        Returns:
            UInt: A new ``UInt`` handle holding ``self % other``.

        Example:
            >>> import qamomile.circuit as qmc
            >>> @qmc.qkernel
            ... def circuit() -> qmc.Vector[qmc.Qubit]:
            ...     q = qmc.qubit_array(4, "q")
            ...     for i in qmc.range(4):
            ...         if i % 2 == 0:
            ...             q[i] = qmc.x(q[i])
            ...     return q
        """
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result, BinOpKind.MOD)
        return result

    def __rmod__(self, other: "int | UInt") -> "UInt":
        """Take an integer-like operand modulo this UInt.

        Args:
            other (int | UInt): The dividend. A Python ``int`` is coerced
                to a constant ``UInt`` operand.

        Returns:
            UInt: A new ``UInt`` handle holding ``other % self``.
        """
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result, BinOpKind.MOD)
        return result

    # UInt-specific power operation
    def __pow__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, coerced.value, result, BinOpKind.POW)
        return result

    def __rpow__(self, other: "int | UInt") -> "UInt":
        coerced = self._coerce(other)
        result = self._make_result()
        _emit_binop(coerced.value, self.value, result, BinOpKind.POW)
        return result

    def __index__(self) -> int:
        """Return the underlying Python ``int`` when the handle is constant.

        Implementing ``__index__`` makes ``UInt`` satisfy the
        :class:`typing.SupportsIndex` protocol so that constructs like
        ``q[0 : n - 1]`` (slice with a ``UInt`` bound) type-check — Python
        builds the ``slice`` object as ``slice(0, n - 1)`` and
        ``slice.__init__`` accepts only ``SupportsIndex`` for its
        arguments at the stub level. Without ``__index__`` static checkers
        flag every such slice as ``"Slice index must be an integer,
        SupportsIndex or None"``.

        Runtime semantics:

        - When ``self.value.is_constant()`` returns ``True``, this returns
          the underlying ``int`` const so callers that legitimately want
          the concrete value (``range(uint_const)``, list indexing with a
          constant ``UInt``, ...) see the expected behaviour.
        - When the handle is symbolic, this raises ``TypeError``. Symbolic
          ``UInt`` cannot be coerced to a concrete ``int`` at trace time;
          the kinds of operations that actually need an ``int`` here (raw
          Python list / ``range`` / etc.) cannot consume a symbolic value
          either, so raising surfaces the misuse loudly rather than
          letting it propagate. ``Vector`` slicing does *not* take this
          path because ``Vector.__getitem__(slice)`` reads
          ``slice.start`` / ``slice.stop`` directly and dispatches on
          ``int`` / ``UInt`` without going through ``__index__``.

        Returns:
            int: The underlying integer when this handle wraps a
                compile-time constant.

        Raises:
            TypeError: When this handle is symbolic (no compile-time
                constant attached). Symbolic ``UInt`` cannot be turned
                into a Python ``int``.
        """
        if self.value.is_constant():
            const = self.value.get_const()
            if isinstance(const, int):
                return const
        raise TypeError(
            "Symbolic UInt cannot be converted to a Python int via "
            "__index__(). This typically happens when a symbolic UInt is "
            "passed to a Python builtin that requires a concrete integer "
            "(e.g. range(), list indexing). For Vector slicing the "
            "symbolic bound is handled directly by Vector.__getitem__ "
            "without going through __index__."
        )


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
        _emit_binop(self.value, other.value, result, BinOpKind.ADD)
        return result

    def __radd__(self, other: "int | float | Float") -> "Float":
        return self.__add__(other)

    def __sub__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.SUB)
        return result

    def __rsub__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result, BinOpKind.SUB)
        return result

    def __neg__(self) -> "Float":
        """Negate this Float handle, returning the unary-minus result.

        Lets users write the natural ``-x`` inside a ``@qkernel`` instead
        of the awkward ``0 - x`` idiom (GitHub issue #329). The negation
        is lowered to the existing ``MUL`` IR op as ``self * -1.0`` —
        multiplication by ``-1`` is the direct expression of unary minus
        — so it carries no new IR node and every backend that already
        supports multiplication emits it unchanged. When ``self`` is a
        compile-time constant the result is folded eagerly by
        ``_emit_binop``.

        Returns:
            Float: A new Float handle holding the negated value.

        Example:
            >>> import qamomile.circuit as qmc
            >>> @qmc.qkernel
            ... def circuit(theta: qmc.Float) -> qmc.Qubit:
            ...     q = qmc.qubit("q")
            ...     return qmc.rz(q, -theta)
        """
        neg_one = self._make_float(-1.0)
        result = self._make_result()
        _emit_binop(self.value, neg_one.value, result, BinOpKind.MUL)
        return result

    def __mul__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.MUL)
        return result

    def __rmul__(self, other: "int | float | Float") -> "Float":
        return self.__mul__(other)

    def __truediv__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(self.value, other.value, result, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(other.value, self.value, result, BinOpKind.DIV)
        return result

    def __pow__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.POW)
        return result

    def __rpow__(self, other: "int | float | Float") -> "Float":
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result, BinOpKind.POW)
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
        if isinstance(other, (int, float)) and not isinstance(other, Float):
            other = self._make_float(float(other))
        elif not isinstance(other, Float):
            return NotImplemented  # type: ignore[return-value]
        result = self._make_bit()
        _emit_compop(self.value, other.value, result.value, CompOpKind.EQ)
        return result

    def __ne__(self, other) -> "Bit":  # type: ignore[override]
        if isinstance(other, (int, float)) and not isinstance(other, Float):
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
