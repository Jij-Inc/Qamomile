"""Tests for UInt arithmetic with float/Float operands."""

import pytest

from qamomile.circuit.frontend.handle.primitives import Float, UInt
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import Value


def _make_uint(val: int = 0) -> UInt:
    return UInt(value=Value(type=UIntType(), name="u"), init_value=val)


def _make_float(val: float = 0.0) -> Float:
    return Float(value=Value(type=FloatType(), name="f"), init_value=val)


class TestUIntWithIntOperand:
    """UInt op int should return UInt."""

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__floordiv__", "__pow__"])
    def test_returns_uint(self, op: str) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = getattr(_make_uint(3), op)(2)
        assert isinstance(result, UInt)

    @pytest.mark.parametrize("op", ["__radd__", "__rsub__", "__rmul__", "__rfloordiv__", "__rpow__"])
    def test_reflected_returns_uint(self, op: str) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = getattr(_make_uint(3), op)(2)
        assert isinstance(result, UInt)


class TestUIntWithFloatOperand:
    """UInt op float should return Float."""

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__floordiv__", "__pow__"])
    def test_returns_float(self, op: str) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = getattr(_make_uint(3), op)(1.5)
        assert isinstance(result, Float)

    @pytest.mark.parametrize("op", ["__radd__", "__rsub__", "__rmul__", "__rfloordiv__", "__rpow__"])
    def test_reflected_returns_float(self, op: str) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = getattr(_make_uint(3), op)(1.5)
        assert isinstance(result, Float)


class TestUIntWithFloatHandle:
    """UInt op Float (handle) should return Float."""

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__floordiv__", "__pow__"])
    def test_returns_float(self, op: str) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = getattr(_make_uint(3), op)(_make_float(1.5))
        assert isinstance(result, Float)


class TestUIntTruediv:
    """UInt truediv always returns Float regardless of operand type."""

    def test_truediv_int(self) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = _make_uint(6).__truediv__(2)
        assert isinstance(result, Float)

    def test_truediv_float(self) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = _make_uint(6).__truediv__(1.5)
        assert isinstance(result, Float)

    def test_rtruediv_int(self) -> None:
        tracer = Tracer()
        with trace(tracer):
            result = _make_uint(6).__rtruediv__(2)
        assert isinstance(result, Float)


class TestEmittedOperations:
    """Verify that BinOp operations are correctly emitted."""

    def test_add_float_emits_binop(self) -> None:
        tracer = Tracer()
        with trace(tracer):
            _make_uint(3).__add__(1.5)
        assert len(tracer.operations) == 1
        op = tracer.operations[0]
        assert isinstance(op, BinOp)
        assert op.kind == BinOpKind.ADD

    def test_mul_float_emits_binop(self) -> None:
        tracer = Tracer()
        with trace(tracer):
            _make_uint(3).__mul__(2.0)
        assert len(tracer.operations) == 1
        op = tracer.operations[0]
        assert isinstance(op, BinOp)
        assert op.kind == BinOpKind.MUL
