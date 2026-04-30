"""Tests for the anonymous-tmp-name invariant (Phase 1 of #7).

Auto-generated tmp values produced by the arithmetic / comparison /
logical-op overloads on ``UInt`` / ``Float`` / ``Bit`` carry an empty
string for ``Value.name``. The empty string is the **anonymous marker**:
it makes it structurally impossible for two unrelated tmp values to
collide on a shared name-keyed binding lookup.

These tests pin the invariant in two layers:

1. **White-box**: the helper constructors return values with ``name=""``.
2. **Regression**: chained predicates like ``(~a) & (~b) & c`` (where
   each ``~``/``&`` produces a tmp ``Bit``) emit and execute without
   short-circuiting the result of one tmp into another via name
   collisions.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import Value

# ---------------------------------------------------------------------------
# Layer 1: white-box — tmp values created by the handle helpers are anonymous
# ---------------------------------------------------------------------------


class TestTmpValuesHaveEmptyNames:
    """Auto-generated tmp values must carry ``name=""``."""

    def _u(self, name: str = "x") -> qmc.UInt:
        return qmc.UInt(value=Value(type=UIntType(), name=name))

    def _f(self, name: str = "x") -> qmc.Float:
        return qmc.Float(value=Value(type=FloatType(), name=name))

    def _b(self, name: str = "x") -> qmc.Bit:
        return qmc.Bit(value=Value(type=BitType(), name=name))

    def test_uint_arithmetic_result_is_anonymous(self):
        tracer = Tracer()
        a = self._u("a")
        b = self._u("b")
        with trace(tracer):
            c = a + b
        assert c.value.name == ""

    def test_uint_int_constant_is_anonymous(self):
        tracer = Tracer()
        a = self._u("a")
        with trace(tracer):
            c = a + 3
        assert c.value.name == ""
        op = tracer.operations[0]
        assert isinstance(op, BinOp)
        # The coerced int constant is the BinOp's rhs.
        assert op.rhs.name == ""

    def test_uint_float_promotion_is_anonymous(self):
        tracer = Tracer()
        a = self._u("a")
        with trace(tracer):
            c = a * 1.5
        assert isinstance(c, qmc.Float)
        assert c.value.name == ""

    def test_uint_comparison_bit_is_anonymous(self):
        tracer = Tracer()
        a = self._u("a")
        b = self._u("b")
        with trace(tracer):
            r = a < b
        assert r.value.name == ""
        op = tracer.operations[0]
        assert isinstance(op, CompOp)
        assert op.kind is CompOpKind.LT

    def test_float_arithmetic_result_is_anonymous(self):
        tracer = Tracer()
        a = self._f("a")
        b = self._f("b")
        with trace(tracer):
            c = a / b
        assert c.value.name == ""

    def test_float_constant_is_anonymous(self):
        tracer = Tracer()
        a = self._f("a")
        with trace(tracer):
            c = a + 0.5
        assert c.value.name == ""

    def test_float_comparison_bit_is_anonymous(self):
        tracer = Tracer()
        a = self._f("a")
        with trace(tracer):
            r = a >= 0.0
        assert r.value.name == ""

    def test_bit_and_result_is_anonymous(self):
        tracer = Tracer()
        a = self._b("a")
        b = self._b("b")
        with trace(tracer):
            r = a & b
        assert r.value.name == ""
        op = tracer.operations[0]
        assert isinstance(op, CondOp)
        assert op.kind is CondOpKind.AND

    def test_bit_or_result_is_anonymous(self):
        tracer = Tracer()
        a = self._b("a")
        b = self._b("b")
        with trace(tracer):
            r = a | b
        assert r.value.name == ""

    def test_bit_invert_result_is_anonymous(self):
        tracer = Tracer()
        a = self._b("a")
        with trace(tracer):
            r = ~a
        assert r.value.name == ""
        op = tracer.operations[0]
        assert isinstance(op, NotOp)

    def test_bit_constant_coercion_is_anonymous(self):
        tracer = Tracer()
        a = self._b("a")
        with trace(tracer):
            _ = a & True
        # The coerced True constant is the CondOp's rhs.
        op = tracer.operations[0]
        assert isinstance(op, CondOp)
        assert op.rhs.name == ""


# ---------------------------------------------------------------------------
# Layer 2: regression — chained predicates fold to the right branch
# ---------------------------------------------------------------------------


class TestChainedPredicateRegression:
    """``(~a) & (~b) & c`` must not fold to ``true & c`` via name collision.

    Before Phase 1, every chained ``Bit`` op produced a tmp ``Value`` named
    ``"bit_tmp"``. Since multiple tmp values shared the same name, some
    name-keyed bindings lookups would conflate them (e.g. find a constant
    bound for an earlier ``"bit_tmp"`` and reuse it as the value for a
    later one), folding three-input predicates incorrectly. The fix is
    to mark all tmp values anonymous (``name=""``) and require truthy
    guards on every name-based reader.
    """

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    @pytest.mark.parametrize(
        "a, b, c, expected_x_applied",
        [
            (False, False, True, True),  # the only "all clear" case
            (True, False, True, False),
            (False, True, True, False),
            (False, False, False, False),
            (True, True, True, False),
        ],
    )
    def test_three_way_chained_predicate_compile_time(
        self, transpiler, a, b, c, expected_x_applied
    ):
        """With Bit constants bound, the if-branch folds to the right value."""

        @qmc.qkernel
        def kernel(a: qmc.Bit, b: qmc.Bit, c: qmc.Bit) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            cond = (~a) & (~b) & c
            if cond:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel, bindings={"a": a, "b": b, "c": c})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        # An X gate appears in the circuit iff the predicate is true.
        assert ("x" in gate_names) is expected_x_applied
        # No surviving runtime if/else after compile-time fold.
        assert all(name != "if_else" for name in gate_names)
