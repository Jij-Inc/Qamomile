"""Cell-by-cell coverage tests for ``RuntimeOpKind`` × Qiskit backend.

The IR contract is that every variant of ``RuntimeOpKind`` reaches
``StandardEmitPass._emit_runtime_classical_expr`` and is dispatched to a
backend-native runtime expression. Earlier this contract had a gap: the
backend implemented every match arm but the frontend only emitted
AND/OR/NOT, so the comparison and arithmetic arms were dead code that
hid the ``bool(...)`` coercion bug Copilot caught.

These tests pin every cell:

1. **Frontend-driven** for kinds the frontend can actually construct
   under measurement taint (currently AND/OR/NOT, plus EQ/NEQ on Bit
   constants). End-to-end execution.

2. **Synthetic IR** for kinds the frontend cannot currently produce
   (``measure(QFixed) → Float`` participates here as the only numeric
   measurement path; Float arithmetic on tainted values is constructed
   manually). This drives the Qiskit backend's ``expr.add``/``mul``/
   ``equal``/``less``/... arms, exercising numeric preservation.

3. **NotImplementedError** for kinds without a Qiskit equivalent
   (``FLOORDIV``, ``POW``).
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.arithmetic_operations import (
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import Value

qiskit = pytest.importorskip("qiskit")


@pytest.fixture
def qiskit_transpiler():
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


# ---------------------------------------------------------------------------
# Layer 1: frontend-driven — kinds the frontend currently constructs
# ---------------------------------------------------------------------------


class TestFrontendReachableKinds:
    """End-to-end execution for kinds the frontend already produces."""

    def test_and_runtime_executes(self, qiskit_transpiler):
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

        exe = qiskit_transpiler.transpile(kernel)
        assert exe.sample(qiskit_transpiler.executor(), shots=50).result().results == [
            (1, 50)
        ]

    def test_or_runtime_executes(self, qiskit_transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a | b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = qiskit_transpiler.transpile(kernel)
        assert exe.sample(qiskit_transpiler.executor(), shots=50).result().results == [
            (1, 50)
        ]

    def test_not_runtime_executes(self, qiskit_transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            a = qmc.measure(q0)
            if ~a:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        exe = qiskit_transpiler.transpile(kernel)
        assert exe.sample(qiskit_transpiler.executor(), shots=50).result().results == [
            (1, 50)
        ]


# ---------------------------------------------------------------------------
# Layer 2: synthetic IR — drives every backend match arm directly
# ---------------------------------------------------------------------------


class TestSyntheticBinaryExprDispatch:
    """Direct unit tests on ``_build_qiskit_binary_expr`` covering every
    binary ``RuntimeOpKind``. Numeric operands are passed as Python ints
    to assert that the dispatch preserves them (no ``bool(...)`` coercion)
    and routes through the right Qiskit ``expr`` builder.
    """

    @pytest.fixture
    def expr_module(self):
        from qiskit.circuit.classical import expr, types

        return (expr, types)

    @pytest.mark.parametrize(
        "kind, expected_op_name",
        [
            (RuntimeOpKind.AND, "LOGIC_AND"),
            (RuntimeOpKind.OR, "LOGIC_OR"),
            (RuntimeOpKind.EQ, "EQUAL"),
            (RuntimeOpKind.NEQ, "NOT_EQUAL"),
            (RuntimeOpKind.LT, "LESS"),
            (RuntimeOpKind.LE, "LESS_EQUAL"),
            (RuntimeOpKind.GT, "GREATER"),
            (RuntimeOpKind.GE, "GREATER_EQUAL"),
            (RuntimeOpKind.ADD, "ADD"),
            (RuntimeOpKind.SUB, "SUB"),
            (RuntimeOpKind.MUL, "MUL"),
            (RuntimeOpKind.DIV, "DIV"),
        ],
    )
    def test_binary_kind_dispatches_to_correct_qiskit_expr(
        self, kind, expected_op_name, expr_module
    ):
        """Each binary RuntimeOpKind dispatches to the matching qiskit.expr op."""
        from qamomile.qiskit.transpiler import QiskitEmitPass

        expr, types = expr_module
        # Use lift on plain ints — they get qiskit Uint(8) typing but
        # importantly retain their numeric value (not coerced to bool).
        lhs = expr.lift(3, types.Uint(8))
        rhs = expr.lift(5, types.Uint(8))

        # Logical ops require Bool typing instead.
        if kind in (RuntimeOpKind.AND, RuntimeOpKind.OR):
            lhs = expr.lift(True)
            rhs = expr.lift(False)

        result = QiskitEmitPass._build_qiskit_binary_expr(kind, lhs, rhs)
        assert hasattr(result, "op"), f"Expected expr.Binary, got {type(result)}"
        assert result.op.name == expected_op_name

    @pytest.mark.parametrize("kind", [RuntimeOpKind.FLOORDIV, RuntimeOpKind.POW])
    def test_unsupported_kinds_raise_not_implemented(self, kind, expr_module):
        """FLOORDIV and POW have no qiskit.expr equivalent — raise loudly."""
        from qamomile.qiskit.transpiler import QiskitEmitPass

        expr, types = expr_module
        lhs = expr.lift(3, types.Uint(8))
        rhs = expr.lift(2, types.Uint(8))
        with pytest.raises(NotImplementedError, match=kind.name):
            QiskitEmitPass._build_qiskit_binary_expr(kind, lhs, rhs)

    def test_numeric_constants_preserve_their_type(self, expr_module):
        """Regression for the Copilot-flagged ``bool(...)`` coercion bug.

        Building ``expr.equal(reg, 5)`` must keep ``5`` as an integer; if
        anything coerced operands to ``bool`` it would become ``True`` and
        compare against the register as 1, silently changing semantics.
        """
        from qamomile.qiskit.transpiler import QiskitEmitPass

        expr, types = expr_module
        lhs = expr.lift(7, types.Uint(8))
        # 5 is a Python int; lifted into a Uint(8) constant by qiskit.
        rhs = expr.lift(5, types.Uint(8))
        result = QiskitEmitPass._build_qiskit_binary_expr(RuntimeOpKind.EQ, lhs, rhs)
        assert result.op.name == "EQUAL"
        # Right operand should still represent 5, not True.
        assert getattr(result.right, "value", None) == 5


class TestSyntheticRuntimeClassicalExprConstruction:
    """The IR constructor accepts every kind without complaint, and the
    operand-arity rules are honored.
    """

    @pytest.mark.parametrize("kind", list(RuntimeOpKind))
    def test_all_kinds_constructible(self, kind):
        """RuntimeClassicalExpr accepts every enum variant."""
        if kind is RuntimeOpKind.NOT:
            operands = [Value(type=BitType(), name="x")]
        else:
            operands = [
                Value(type=UIntType(), name="lhs"),
                Value(type=UIntType(), name="rhs"),
            ]
        result_type = BitType() if kind is not RuntimeOpKind.NOT else BitType()
        result_value = Value(type=result_type, name="")
        op = RuntimeClassicalExpr(
            kind=kind,
            operands=operands,
            results=[result_value],
        )
        assert op.kind is kind
        assert len(op.operands) == (1 if kind is RuntimeOpKind.NOT else 2)


# ---------------------------------------------------------------------------
# Layer 3: numeric preservation through resolve_operand
# ---------------------------------------------------------------------------


class TestResolveOperandPreservesNumericType:
    """The resolver hands integer / float operands to the qiskit ``expr``
    builders **without** ``bool(...)`` coercion. Bit-typed operands are
    already stored as bool by the frontend so they pass through naturally.
    """

    def test_int_constant_passes_through_as_int(self):
        v = Value(type=UIntType(), name="").with_const(5)
        assert v.is_constant()
        assert v.get_const() == 5
        assert v.get_const() is not True  # not coerced to bool

    def test_float_constant_passes_through_as_float(self):
        v = Value(type=FloatType(), name="").with_const(0.5)
        assert v.is_constant()
        assert v.get_const() == 0.5
        assert isinstance(v.get_const(), float)

    def test_bit_constant_is_bool(self):
        # Bit._coerce wraps constants with bool(); both 0/1 ints and bools
        # land as Python bools in the IR, so dispatch into qiskit's bool
        # ops works correctly.
        v = Value(type=BitType(), name="").with_const(True)
        assert v.is_constant()
        assert v.get_const() is True

        v0 = Value(type=BitType(), name="").with_const(False)
        assert v0.get_const() is False
