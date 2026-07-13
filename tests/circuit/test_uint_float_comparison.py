"""Tests for ``UInt`` ↔ ``Float`` comparison type promotion (FE-5).

Before these overloads, ``UInt == Float`` (and the reflected
``Float == UInt``) returned ``NotImplemented`` from both handles, so Python
fell back to identity comparison and the result silently collapsed to a
Python ``bool`` — exactly the same failure family as the missing ``Bit``
equality operators (P1-3). The comparison operators now promote a ``UInt``
and a ``Float`` to a common operand pair and emit a ``CompOp``, so a
mixed-type comparison stays in the IR (and folds when both operands are
compile-time constants).

The tests exercise two layers:

1. **Frontend**: every comparison operator (``==`` / ``!=`` / ``<`` / ``>``
   / ``<=`` / ``>=``) between a ``UInt`` and a ``Float`` handle emits a
   single ``CompOp`` and never returns a Python ``bool``.
2. **Compile-time fold**: a ``UInt`` loop variable compared against a bound
   ``Float`` folds per iteration, dropping the inactive branch's gates.

Note: Do NOT use ``from __future__ import annotations`` in this file. The
@qkernel AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import Value


def _make_uint_handle(name: str) -> qmc.UInt:
    """Build a non-constant ``UInt`` handle for tracing."""
    return qmc.UInt(value=Value(type=UIntType(), name=name))


def _make_float_handle(name: str) -> qmc.Float:
    """Build a non-constant ``Float`` handle for tracing."""
    return qmc.Float(value=Value(type=FloatType(), name=name))


# ---------------------------------------------------------------------------
# Layer 1: mixed UInt/Float comparisons emit a CompOp of the right kind
# ---------------------------------------------------------------------------

_OPS = [
    ("eq", lambda a, b: a == b, CompOpKind.EQ),
    ("ne", lambda a, b: a != b, CompOpKind.NEQ),
    ("lt", lambda a, b: a < b, CompOpKind.LT),
    ("gt", lambda a, b: a > b, CompOpKind.GT),
    ("le", lambda a, b: a <= b, CompOpKind.LE),
    ("ge", lambda a, b: a >= b, CompOpKind.GE),
]


class TestUIntFloatComparisonEmitsIR:
    """Each comparison between a ``UInt`` and a ``Float`` emits one CompOp."""

    @pytest.mark.parametrize("name,op,kind", _OPS, ids=[o[0] for o in _OPS])
    def test_uint_op_float(self, name, op, kind):
        tracer = Tracer()
        u = _make_uint_handle("u")
        f = _make_float_handle("f")
        with trace(tracer):
            result = op(u, f)
        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert isinstance(result, qmc.Bit)
        assert not isinstance(result, bool)

    @pytest.mark.parametrize("name,op,kind", _OPS, ids=[o[0] for o in _OPS])
    def test_float_op_uint(self, name, op, kind):
        tracer = Tracer()
        f = _make_float_handle("f")
        u = _make_uint_handle("u")
        with trace(tracer):
            result = op(f, u)
        assert len(tracer.operations) == 1
        emitted = tracer.operations[0]
        assert isinstance(emitted, CompOp)
        assert emitted.kind is kind
        assert isinstance(result, qmc.Bit)
        assert not isinstance(result, bool)

    def test_uint_compared_with_python_float(self):
        """``uint < 1.5`` promotes the Python ``float`` to a constant Float
        operand instead of raising / collapsing."""
        tracer = Tracer()
        u = _make_uint_handle("u")
        with trace(tracer):
            result = u < 1.5
        assert len(tracer.operations) == 1
        assert isinstance(tracer.operations[0], CompOp)
        assert isinstance(result, qmc.Bit)

    def test_eq_never_collapses_to_python_bool(self):
        """The FE-5 regression guard: a mixed comparison must be a Bit."""
        tracer = Tracer()
        u = _make_uint_handle("u")
        f = _make_float_handle("f")
        with trace(tracer):
            eq = u == f
            ne = f != u
        assert isinstance(eq, qmc.Bit)
        assert isinstance(ne, qmc.Bit)
        assert not isinstance(eq, bool)
        assert not isinstance(ne, bool)

    def test_same_type_comparisons_still_emit_compop(self):
        """The shared ``_coerce_comparison`` refactor keeps the
        homogeneous ``uint == uint`` / ``float < float`` paths emitting a
        CompOp (not a regression from the mixed-type work)."""
        tracer = Tracer()
        with trace(tracer):
            r_uu = _make_uint_handle("u1") == _make_uint_handle("u2")
            r_ff = _make_float_handle("f1") < _make_float_handle("f2")
        assert isinstance(r_uu, qmc.Bit)
        assert isinstance(r_ff, qmc.Bit)
        assert [type(op).__name__ for op in tracer.operations] == [
            "CompOp",
            "CompOp",
        ]

    def test_incomparable_operand_returns_notimplemented(self):
        """A UInt / Float compared with an unsupported operand yields
        ``NotImplemented`` so Python raises ``TypeError`` (ordering) or
        falls back to identity (equality) instead of a malformed op."""
        u = _make_uint_handle("u")
        f = _make_float_handle("f")
        assert u.__eq__("not a number") is NotImplemented
        assert u.__lt__(object()) is NotImplemented
        assert f.__ne__("not a number") is NotImplemented
        assert f.__ge__(object()) is NotImplemented


# ---------------------------------------------------------------------------
# Layer 2: compile-time fold of a UInt loop var against a bound Float
# ---------------------------------------------------------------------------


class TestUIntFloatCompileTimeFold:
    """``if i == threshold:`` folds per iteration with a bound Float."""

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_uint_eq_float_folds(self, transpiler):
        @qmc.qkernel
        def kernel(threshold: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            for i in qmc.range(3):
                if i == threshold:
                    q[i] = qmc.x(q[i])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel, bindings={"threshold": 1.0})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        # Only i == 1 (== 1.0) matches, so a single X survives, fully folded.
        assert gate_names.count("x") == 1
        assert all(name != "if_else" for name in gate_names)

    def test_uint_lt_float_folds(self, transpiler):
        @qmc.qkernel
        def kernel(threshold: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            for i in qmc.range(3):
                if i < threshold:
                    q[i] = qmc.x(q[i])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel, bindings={"threshold": 1.5})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        # i < 1.5 holds for i in {0, 1} → two X gates survive.
        assert gate_names.count("x") == 2
        assert all(name != "if_else" for name in gate_names)
