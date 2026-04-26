"""Tests for ``ClassicalLoweringPass`` and ``RuntimeClassicalExpr``.

The pass identifies measurement-derived classical ops (CompOp / CondOp /
NotOp / BinOp) and rewrites them to ``RuntimeClassicalExpr`` so emit can
dispatch them through a dedicated backend hook instead of the legacy
fold-or-translate path. Non-measurement-derived classical ops are left
unchanged so the existing fold paths continue to handle them.

Tests cover three layers:

1. **IR rewriting** — synthetic kernels are run through ``transpile()``
   up to (and including) ``classical_lowering``; the resulting IR is
   inspected to check which ops became ``RuntimeClassicalExpr`` and
   which stayed as their original type.

2. **End-to-end** — the same kernels run on the qiskit simulator and
   produce correct results, exercising the full pipeline including the
   new ``_emit_runtime_classical_expr`` hook.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.arithmetic_operations import (
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps


def _walk_ops(operations):
    """Yield every operation, recursing through nested op lists."""
    for op in operations:
        yield op
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                yield from _walk_ops(body)


def _count_ops_of_type(operations, op_type) -> int:
    return sum(1 for op in _walk_ops(operations) if isinstance(op, op_type))


def _lower_to_classical_lowering(kernel, bindings=None):
    """Run pipeline up through classical_lowering and return the block."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    tp = QiskitTranspiler()
    block = tp.to_block(kernel, bindings=bindings)
    block = tp.substitute(block)
    block = tp.resolve_parameter_shapes(block, bindings)
    block = tp.inline(block)
    block = tp.affine_validate(block)
    block = tp.partial_eval(block, bindings)
    block = tp.analyze(block)
    return tp.classical_lowering(block)


# ---------------------------------------------------------------------------
# Layer 1: IR rewriting verification
# ---------------------------------------------------------------------------


class TestClassicalLoweringIR:
    """Inspect the IR after ``classical_lowering`` to confirm which ops
    were rewritten and which were left as their compile-time type."""

    def test_measurement_taint_propagates_to_condop(self):
        """``a & b`` where a, b are measurements becomes RuntimeClassicalExpr.

        The frontend emits CondOp(AND, a, b); after lowering this becomes
        RuntimeClassicalExpr(AND, [a, b]).
        """

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

        block = _lower_to_classical_lowering(kernel)
        cond_ops = _count_ops_of_type(block.operations, CondOp)
        runtime_ops = [
            op for op in _walk_ops(block.operations) if isinstance(op, RuntimeClassicalExpr)
        ]
        assert cond_ops == 0, "CondOp should be lowered when operands are measurements"
        assert any(op.kind is RuntimeOpKind.AND for op in runtime_ops)

    def test_measurement_taint_propagates_to_notop(self):
        """``~a`` where a is a measurement becomes RuntimeClassicalExpr(NOT)."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            a = qmc.measure(q0)
            if ~a:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        block = _lower_to_classical_lowering(kernel)
        not_ops = _count_ops_of_type(block.operations, NotOp)
        runtime_ops = [
            op for op in _walk_ops(block.operations) if isinstance(op, RuntimeClassicalExpr)
        ]
        assert not_ops == 0, "NotOp should be lowered when operand is a measurement"
        assert any(op.kind is RuntimeOpKind.NOT for op in runtime_ops)

    def test_chained_taint_lowers_all_levels(self):
        """``(~a) & (~b) & c`` over measurements: all 4 ops lower."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q3 = qmc.qubit("q3")
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            q2 = qmc.x(q2)
            c = qmc.measure(q2)
            if (~a) & (~b) & c:
                q3 = qmc.x(q3)
            return qmc.measure(q3)

        block = _lower_to_classical_lowering(kernel)
        cond_ops = _count_ops_of_type(block.operations, CondOp)
        not_ops = _count_ops_of_type(block.operations, NotOp)
        runtime_ops = [
            op for op in _walk_ops(block.operations) if isinstance(op, RuntimeClassicalExpr)
        ]
        # All compile-time classical ops should be lowered:
        # 2 NotOp + 2 CondOp(AND) → 4 RuntimeClassicalExpr
        assert cond_ops == 0
        assert not_ops == 0
        assert len(runtime_ops) >= 4
        kinds = [op.kind for op in runtime_ops]
        assert kinds.count(RuntimeOpKind.NOT) >= 2
        assert kinds.count(RuntimeOpKind.AND) >= 2

    def test_compile_time_op_not_lowered(self):
        """A CompOp folded to a constant by partial_eval doesn't survive
        to classical_lowering, but if it did (e.g. via parameter), it
        should remain a CompOp (not lowered)."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            # ``flag > 0`` resolves at partial_eval if flag is bound.
            # If not bound (parameter), it could survive — but it's still
            # not measurement-derived, so it should NOT lower.
            if flag > 0:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        # Flag bound: compile_time_if_lowering folds the CompOp away.
        block = _lower_to_classical_lowering(kernel, bindings={"flag": 1})
        runtime_ops = [
            op for op in _walk_ops(block.operations) if isinstance(op, RuntimeClassicalExpr)
        ]
        # No RuntimeClassicalExpr — the CompOp was folded by an earlier pass.
        assert len(runtime_ops) == 0

    def test_loop_var_op_not_lowered(self):
        """A CompOp on a loop variable is not measurement-derived; should
        stay as CompOp (foldable at emit time per iteration)."""

        @qmc.qkernel
        def kernel(target: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, name="q")
            for j in qmc.range(5):
                if j == target:
                    q[j] = qmc.x(q[j])
            return qmc.measure(q)

        block = _lower_to_classical_lowering(kernel, bindings={"target": 2})
        runtime_ops = [
            op for op in _walk_ops(block.operations) if isinstance(op, RuntimeClassicalExpr)
        ]
        # Loop-bound CompOp resolves at emit time, never measurement-derived.
        assert len(runtime_ops) == 0


# ---------------------------------------------------------------------------
# Layer 2: End-to-end execution
# ---------------------------------------------------------------------------


class TestClassicalLoweringEndToEnd:
    """The qiskit simulator must produce correct results for kernels whose
    classical predicates were lowered to ``RuntimeClassicalExpr``."""

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_runtime_and_executes(self, transpiler):
        """``if a & b:`` over measurements is now lowered to
        RuntimeClassicalExpr and emitted via the new backend hook."""

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

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=100)
            .result()
        )
        assert result.results == [(1, 100)]

    def test_runtime_or_executes(self, transpiler):
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

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=100)
            .result()
        )
        assert result.results == [(1, 100)]

    def test_runtime_not_executes(self, transpiler):
        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            a = qmc.measure(q0)
            if ~a:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=100)
            .result()
        )
        assert result.results == [(1, 100)]

    def test_chained_runtime_expr_emits_compound_classical_condition(
        self, transpiler
    ):
        """End-to-end equivalent of ``test_chained_taint_lowers_all_levels``:
        the resulting Qiskit circuit must carry a compound classical
        expression in its ``IfElseOp.condition``."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q3 = qmc.qubit("q3")
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            q2 = qmc.x(q2)
            c = qmc.measure(q2)
            if (~a) & (~b) & c:
                q3 = qmc.x(q3)
            return qmc.measure(q3)

        from qiskit.circuit.classical import expr
        from qiskit.circuit.controlflow import IfElseOp

        exe = transpiler.transpile(kernel)
        qc = exe.compiled_quantum[0].circuit
        if_ops = [
            i.operation for i in qc.data if isinstance(i.operation, IfElseOp)
        ]
        assert if_ops, "Expected a runtime IfElseOp in the circuit"
        condition = if_ops[0].condition
        assert isinstance(condition, expr.Expr)
        identifiers = list(expr.iter_identifiers(condition))
        # All three measurement clbits should appear in the compound condition.
        assert len(identifiers) >= 3


# ---------------------------------------------------------------------------
# Layer 3: segmentation rule (#6)
# ---------------------------------------------------------------------------


class TestSegmentationKeepsRuntimeExprInQuantumSegment:
    """The segmentation pass must keep ``RuntimeClassicalExpr`` ops inside
    the quantum segment that wraps measurement → runtime if/while.

    Pre-#6 the segmenter used a BitType-only operand+result heuristic
    (``_is_bit_only_predicate``). Post-#6 the rule is a single
    ``isinstance(op, RuntimeClassicalExpr)`` check — clearer and
    forward-compatible with future patterns that produce non-Bit
    intermediates between a measurement and a runtime if/while (e.g.
    the syndrome-as-integer decode pattern needed by surface codes).

    The end-to-end tests above already exercise the runtime path with
    chained ``& | ~`` operators. This white-box test additionally
    verifies that exactly one quantum segment is produced — i.e. the
    segmenter did not split the run because of the intervening
    classical predicate.
    """

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_runtime_compound_kept_in_single_quantum_segment(self, transpiler):
        """A measurement-bridged compound runtime predicate produces a
        single quantum segment (no ``MultipleQuantumSegmentsError``)."""
        from qamomile.circuit.transpiler.segments import QuantumSegment

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if (~a) & b:  # NotOp + CondOp(AND), both runtime
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = transpiler.transpile(kernel)
        # The full quantum execution should be a single segment.
        assert len(exe.compiled_quantum) == 1
        plan = exe.plan
        quantum_steps = [
            s for s in plan.steps if isinstance(s.segment, QuantumSegment)
        ]
        assert len(quantum_steps) == 1, (
            f"Expected exactly one quantum segment; got "
            f"{len(quantum_steps)}. The runtime classical predicate must be "
            f"kept in the surrounding quantum segment."
        )
