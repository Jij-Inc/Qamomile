"""Tests for emit pass zero-division error handling.

Verifies that EmitPass._pre_evaluate_classical() and
StandardEmitPass._evaluate_binop() raise EmitError on exact zero
division instead of silently returning 0.
"""

import pytest

from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.segments import ClassicalSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyEmit(EmitPass[object]):
    def _emit_quantum_segment(self, operations, bindings):
        return object(), {}, {}

    def _compile_expval(
        self, segment, quantum_segment_index, program, compiled_quantum
    ):
        raise RuntimeError("unused")


class DummyEmitter:
    def create_parameter(self, name):
        return name


def _make_binop(kind: BinOpKind, lhs_const: float, rhs_const: float) -> BinOp:
    lhs = Value(type=FloatType(), name="lhs", params={"const": lhs_const})
    rhs = Value(type=FloatType(), name="rhs", params={"const": rhs_const})
    out = Value(type=FloatType(), name="out")
    return BinOp(operands=[lhs, rhs], results=[out], kind=kind)


# ---------------------------------------------------------------------------
# EmitPass._pre_evaluate_classical
# ---------------------------------------------------------------------------


class TestEmitPassZeroDivision:
    """EmitPass._pre_evaluate_classical raises EmitError on exact zero division."""

    def test_div_by_zero_raises_emit_error(self):
        op = _make_binop(BinOpKind.DIV, 1.0, 0.0)
        seg = ClassicalSegment(operations=[op])
        emit = DummyEmit()
        with pytest.raises(EmitError, match="Division by zero"):
            emit._pre_evaluate_classical(seg)

    def test_floordiv_by_zero_raises_emit_error(self):
        op = _make_binop(BinOpKind.FLOORDIV, 1.0, 0.0)
        seg = ClassicalSegment(operations=[op])
        emit = DummyEmit()
        with pytest.raises(EmitError, match="Floor division by zero"):
            emit._pre_evaluate_classical(seg)

    def test_div_nonzero_succeeds(self):
        op = _make_binop(BinOpKind.DIV, 6.0, 3.0)
        seg = ClassicalSegment(operations=[op])
        emit = DummyEmit()
        emit._pre_evaluate_classical(seg)
        assert emit.bindings[op.results[0].uuid] == 2.0

    def test_floordiv_nonzero_succeeds(self):
        op = _make_binop(BinOpKind.FLOORDIV, 7.0, 2.0)
        seg = ClassicalSegment(operations=[op])
        emit = DummyEmit()
        emit._pre_evaluate_classical(seg)
        assert emit.bindings[op.results[0].uuid] == 3.0

    def test_near_zero_not_rejected(self):
        """Exact non-zero divisors must not be rejected."""
        op = _make_binop(BinOpKind.DIV, 1.0, 1e-320)
        seg = ClassicalSegment(operations=[op])
        emit = DummyEmit()
        emit._pre_evaluate_classical(seg)
        assert op.results[0].uuid in emit.bindings


# ---------------------------------------------------------------------------
# StandardEmitPass._evaluate_binop
# ---------------------------------------------------------------------------


class TestStandardEmitPassZeroDivision:
    """StandardEmitPass._evaluate_binop raises EmitError on exact zero division."""

    def test_div_by_zero_raises_emit_error(self):
        op = _make_binop(BinOpKind.DIV, 1.0, 0.0)
        std = StandardEmitPass(DummyEmitter())
        bindings: dict = {}
        with pytest.raises(EmitError, match="Division by zero"):
            std._evaluate_binop(op, bindings)

    def test_floordiv_by_zero_raises_emit_error(self):
        op = _make_binop(BinOpKind.FLOORDIV, 1.0, 0.0)
        std = StandardEmitPass(DummyEmitter())
        bindings: dict = {}
        with pytest.raises(EmitError, match="Floor division by zero"):
            std._evaluate_binop(op, bindings)

    def test_div_nonzero_succeeds(self):
        op = _make_binop(BinOpKind.DIV, 10.0, 4.0)
        std = StandardEmitPass(DummyEmitter())
        bindings: dict = {}
        std._evaluate_binop(op, bindings)
        assert bindings[op.results[0].uuid] == 2.5

    def test_floordiv_nonzero_succeeds(self):
        op = _make_binop(BinOpKind.FLOORDIV, 10.0, 3.0)
        std = StandardEmitPass(DummyEmitter())
        bindings: dict = {}
        std._evaluate_binop(op, bindings)
        assert bindings[op.results[0].uuid] == 3.0
