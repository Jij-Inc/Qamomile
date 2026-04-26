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

        if_ops = [inst.operation for inst in qc.data if isinstance(inst.operation, IfElseOp)]
        assert if_ops, "Expected at least one IfElseOp"
        # In Qiskit 2.x, IfElseOp.condition for a compound is an Expr.
        condition = if_ops[0].condition
        assert isinstance(condition, expr.Expr)
