"""Tests for Bit handle's logical operator overloads (``&`` / ``|`` / ``~``).

These overloads are the frontend producers of ``CondOp`` (AND / OR) and
``NotOp`` IR ops. The IR contract is that all three predicate kinds
(``CompOp`` / ``CondOp`` / ``NotOp``) flow through every transpiler pass
identically; before these overloads existed, ``CondOp`` and ``NotOp``
were defined in IR but had no producer — handlers in
``compile_time_if_lowering`` and ``classical_executor`` were dead code,
and ``emit`` had no handler at all.

Tests cover three layers:

1. **Frontend**: ``&`` / ``|`` / ``~`` on ``Bit`` handles emit the
   correct IR op kind and shape.
2. **Compile-time path**: with bound ``Bit`` parameters, the if-condition
   folds to its selected branch and the inactive gates are dropped.
3. **Runtime path**: with measurement-derived bits, Qiskit emission
   builds a ``qiskit.circuit.classical.expr.Expr`` and uses it as the
   ``if_test`` / ``while_loop`` condition; end-to-end execution returns
   the expected outcome.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.types.primitives import BitType

# ---------------------------------------------------------------------------
# Layer 1: frontend overloads emit the right IR ops
# ---------------------------------------------------------------------------


def _make_bit_handle(name: str) -> qmc.Bit:
    """Build a non-constant Bit handle for tracing."""
    from qamomile.circuit.ir.value import Value

    return qmc.Bit(value=Value(type=BitType(), name=name))


class TestBitOperatorOverloadsEmitIR:
    """Each operator overload emits one IR op of the expected kind."""

    def test_and_emits_condop_and(self):
        tracer = Tracer()
        a = _make_bit_handle("a")
        b = _make_bit_handle("b")
        with trace(tracer):
            _ = a & b
        assert len(tracer.operations) == 1
        op = tracer.operations[0]
        assert isinstance(op, CondOp)
        assert op.kind is CondOpKind.AND
        assert isinstance(op.results[0].type, BitType)

    def test_or_emits_condop_or(self):
        tracer = Tracer()
        a = _make_bit_handle("a")
        b = _make_bit_handle("b")
        with trace(tracer):
            _ = a | b
        assert len(tracer.operations) == 1
        assert isinstance(tracer.operations[0], CondOp)
        assert tracer.operations[0].kind is CondOpKind.OR

    def test_invert_emits_notop(self):
        tracer = Tracer()
        a = _make_bit_handle("a")
        with trace(tracer):
            _ = ~a
        assert len(tracer.operations) == 1
        assert isinstance(tracer.operations[0], NotOp)

    def test_and_with_python_bool_is_coerced(self):
        """``bit & True`` should still produce a CondOp (with bool coerced
        to a constant Bit)."""
        tracer = Tracer()
        a = _make_bit_handle("a")
        with trace(tracer):
            _ = a & True
        # Coercion of ``True`` may produce a CompositeOp or just a const
        # Bit value, but the final op is still a CondOp.
        condops = [op for op in tracer.operations if isinstance(op, CondOp)]
        assert len(condops) == 1
        assert condops[0].kind is CondOpKind.AND

    def test_reflected_and_with_python_bool(self):
        """``True & bit`` uses ``__rand__``."""
        tracer = Tracer()
        a = _make_bit_handle("a")
        with trace(tracer):
            _ = True & a
        condops = [op for op in tracer.operations if isinstance(op, CondOp)]
        assert len(condops) == 1
        assert condops[0].kind is CondOpKind.AND

    def test_invalid_operand_raises(self):
        """Non-bool/Bit/0/1 operands should raise TypeError, not silently
        produce a malformed op."""
        a = _make_bit_handle("a")
        with pytest.raises(TypeError, match="must be Bit"):
            _ = a & "not a bit"
        with pytest.raises(TypeError, match="must be Bit"):
            _ = a | 2  # not 0/1


# ---------------------------------------------------------------------------
# Layer 2: compile-time fold via bound Bit parameters
# ---------------------------------------------------------------------------


class TestCompileTimeFold:
    """``if a & b:`` etc. should fold when ``a``, ``b`` are bound bools.

    Verified by counting gates in the emitted Qiskit circuit: the inactive
    branch's gates must be dropped entirely.
    """

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    @pytest.mark.parametrize(
        "a,b,expected_x_count",
        # Per row: (a & b), (a | b), (~a) — count only the True ones.
        [
            (True, True, 2),  # True & True, True | True, ~True=False
            (True, False, 1),  # F, True, F
            (False, True, 2),  # F, True, ~False=True
            (False, False, 1),  # F, F, ~False=True
        ],
    )
    def test_and_or_not_fold(self, transpiler, a, b, expected_x_count):
        @qmc.qkernel
        def kernel(a: qmc.Bit, b: qmc.Bit) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            if a & b:
                q[0] = qmc.x(q[0])
            if a | b:
                q[1] = qmc.x(q[1])
            if ~a:
                q[2] = qmc.x(q[2])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel, bindings={"a": a, "b": b})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        assert gate_names.count("x") == expected_x_count
        # No surviving runtime if/else after compile-time fold
        assert all(name != "if_else" for name in gate_names)

    def test_predicate_inside_loop(self, transpiler):
        """``&`` / ``|`` / ``~`` on loop-iterated values fold per-iteration."""

        @qmc.qkernel
        def kernel(spec: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            for idx, val in qmc.items(spec):
                if (val == 1) & (idx < 2):
                    q[idx] = qmc.x(q[idx])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel, bindings={"spec": {0: 1, 1: 1, 2: 1}})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        # idx=0: val==1 ✓ AND idx<2 ✓ → X.  idx=1: same → X.  idx=2: idx<2 ✗ → no X.
        assert gate_names.count("x") == 2


# ---------------------------------------------------------------------------
# Layer 3: runtime path with measurement-derived bits
# ---------------------------------------------------------------------------


class TestRuntimePath:
    """``if a & b:`` with measurement bits emits via Qiskit ``expr``."""

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_runtime_and_executes(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a & b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # a=1, b=1 → AND=1 → q2 flipped → measurement = 1 every shot
        assert result.results == [(1, 100)]

    def test_runtime_or_executes(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)  # a will be 1
            a = qmc.measure(q0)
            b = qmc.measure(q1)  # b will be 0
            if a | b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # a|b = 1 → q2 flipped
        assert result.results == [(1, 100)]

    def test_runtime_not_executes(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            a = qmc.measure(q0)  # a will be 0
            if ~a:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # ~0 = 1 → q1 flipped
        assert result.results == [(1, 100)]

    def test_runtime_emits_classical_expr(self, transpiler):
        """The emitted Qiskit circuit should contain an ``if_else`` with a
        classical expression as condition (not a single ``(clbit, value)``
        tuple), which is how Qiskit 2.x represents compound conditions."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a & b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        from qiskit.circuit.classical import expr

        exe = transpiler.transpile(kernel)
        qc = exe.compiled_quantum[0].circuit
        # Look for an IfElseOp whose condition is an Expr (compound).
        from qiskit.circuit.controlflow import IfElseOp

        if_ops = [
            inst.operation for inst in qc.data if isinstance(inst.operation, IfElseOp)
        ]
        assert if_ops, "Expected at least one IfElseOp"
        # In Qiskit 2.x, IfElseOp.condition for a compound is an Expr.
        condition = if_ops[0].condition
        assert isinstance(condition, expr.Expr)


# ---------------------------------------------------------------------------
# ``==`` / ``!=`` on Bit handles (P1-3)
# ---------------------------------------------------------------------------
#
# Without ``Bit.__eq__`` / ``Bit.__ne__`` the dataclass-generated equality
# reduces ``m0 == m1`` (two measured bits) to a Python ``bool`` at trace
# time, so a runtime branch silently becomes a compile-time constant. The
# overloads emit a ``CompOp`` (EQ / NEQ) instead — the equality counterpart
# to the ``&`` / ``|`` / ``~`` overloads above.


class TestBitComparisonOverloadsEmitIR:
    """``==`` / ``!=`` on Bit handles emit a ``CompOp`` of the right kind."""

    def test_eq_emits_compop_eq(self):
        tracer = Tracer()
        a = _make_bit_handle("a")
        b = _make_bit_handle("b")
        with trace(tracer):
            result = a == b
        assert len(tracer.operations) == 1
        op = tracer.operations[0]
        assert isinstance(op, CompOp)
        assert op.kind is CompOpKind.EQ
        assert isinstance(op.results[0].type, BitType)
        assert isinstance(result, qmc.Bit)

    def test_ne_emits_compop_neq(self):
        tracer = Tracer()
        a = _make_bit_handle("a")
        b = _make_bit_handle("b")
        with trace(tracer):
            result = a != b
        assert len(tracer.operations) == 1
        op = tracer.operations[0]
        assert isinstance(op, CompOp)
        assert op.kind is CompOpKind.NEQ
        assert isinstance(result, qmc.Bit)

    def test_eq_with_python_bool_is_coerced(self):
        """``bit == True`` produces a ``CompOp`` with ``True`` coerced to a
        constant Bit operand."""
        tracer = Tracer()
        a = _make_bit_handle("a")
        with trace(tracer):
            _ = a == True  # noqa: E712 - exercising the bool operand path
        compops = [op for op in tracer.operations if isinstance(op, CompOp)]
        assert len(compops) == 1
        assert compops[0].kind is CompOpKind.EQ

    def test_eq_with_zero_one_int_is_coerced(self):
        """``bit != 1`` coerces the ``0`` / ``1`` int to a constant Bit."""
        tracer = Tracer()
        a = _make_bit_handle("a")
        with trace(tracer):
            _ = a != 1
        compops = [op for op in tracer.operations if isinstance(op, CompOp)]
        assert len(compops) == 1
        assert compops[0].kind is CompOpKind.NEQ

    def test_eq_never_collapses_to_python_bool(self):
        """The regression guard for P1-3: comparing two Bit handles must
        yield a Bit (an IR op), never a Python ``bool``."""
        tracer = Tracer()
        a = _make_bit_handle("a")
        b = _make_bit_handle("b")
        with trace(tracer):
            eq = a == b
            ne = a != b
        assert not isinstance(eq, bool)
        assert not isinstance(ne, bool)
        assert isinstance(eq, qmc.Bit)
        assert isinstance(ne, qmc.Bit)

    def test_bit_is_hashable(self):
        """Defining ``__eq__`` drops dataclass hashing to ``None``; the
        overload restores identity hashing so Bit stays a dict/set member."""
        a = _make_bit_handle("a")
        b = _make_bit_handle("b")
        assert hash(a) == hash(a)
        # Distinct handles occupy distinct set slots (identity hashing).
        assert len({a, b}) == 2

    def test_incomparable_operand_returns_notimplemented(self):
        """A non-Bit / bool / 0-1 operand yields ``NotImplemented`` so
        Python falls back to identity (never a malformed op)."""
        a = _make_bit_handle("a")
        assert a.__eq__("not a bit") is NotImplemented
        assert a.__ne__(2) is NotImplemented


class TestBitComparisonCompileTimeFold:
    """``if a == b:`` folds when ``a``, ``b`` are bound bools."""

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    @pytest.mark.parametrize(
        "a,b",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_eq_ne_fold(self, transpiler, a, b):
        @qmc.qkernel
        def kernel(a: qmc.Bit, b: qmc.Bit) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            if a == b:
                q[0] = qmc.x(q[0])
            if a != b:
                q[1] = qmc.x(q[1])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel, bindings={"a": a, "b": b})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        # Exactly one of ``==`` / ``!=`` holds, so exactly one X survives,
        # and a fully folded circuit keeps no runtime if/else.
        assert gate_names.count("x") == 1
        assert all(name != "if_else" for name in gate_names)
        # Pin the EQ-vs-NEQ semantics: the surviving X must sit on q[0]
        # when ``a == b`` (EQ branch) and on q[1] when ``a != b`` (NEQ
        # branch). Asserting only the count would pass even if EQ / NEQ
        # were swapped, since exactly one branch fires either way.
        x_qubits = [
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if inst.operation.name == "x"
        ]
        expected_qubit = 0 if a == b else 1
        assert x_qubits == [expected_qubit]


class TestBitComparisonRuntimePath:
    """``if a == b:`` with measured bits executes through Qiskit ``expr``."""

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_runtime_eq_executes(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)  # a = 1
            q1 = qmc.x(q1)  # b = 1
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a == b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # a == b (1 == 1) → q2 flipped → 1 every shot.
        assert result.results == [(1, 100)]

    def test_runtime_eq_false_leaves_untouched(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)  # a = 1
            a = qmc.measure(q0)
            b = qmc.measure(q1)  # b = 0
            if a == b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # a != b (1 == 0 is False) → q2 untouched → 0 every shot.
        assert result.results == [(0, 100)]

    def test_runtime_ne_executes(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)  # a = 1
            a = qmc.measure(q0)
            b = qmc.measure(q1)  # b = 0
            if a != b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # a != b (1 != 0) → q2 flipped → 1 every shot.
        assert result.results == [(1, 100)]

    def test_runtime_eq_emits_classical_expr(self, transpiler):
        """``if a == b:`` emits an ``IfElseOp`` whose condition is a Qiskit
        ``Expr`` (the runtime comparison), not a folded constant."""
        from qiskit.circuit.classical import expr
        from qiskit.circuit.controlflow import IfElseOp

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a == b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        qc = exe.compiled_quantum[0].circuit
        if_ops = [
            inst.operation for inst in qc.data if isinstance(inst.operation, IfElseOp)
        ]
        assert if_ops, "Expected at least one IfElseOp"
        assert isinstance(if_ops[0].condition, expr.Expr)
