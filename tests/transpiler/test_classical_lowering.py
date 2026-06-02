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
    """Count operations of ``op_type`` across the op tree, recursing nested ops."""
    return sum(1 for op in _walk_ops(operations) if isinstance(op, op_type))


@qmc.qkernel
def _and_kernel() -> qmc.Bit:
    """``if a & b:`` over two true measurements; target qubit ends in |1>."""
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


@qmc.qkernel
def _or_kernel() -> qmc.Bit:
    """``if a | b:`` with ``a`` true and ``b`` false; target ends in |1>."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q2 = qmc.qubit("q2")
    q0 = qmc.x(q0)
    a = qmc.measure(q0)
    b = qmc.measure(q1)
    if a | b:
        q2 = qmc.x(q2)
    return qmc.measure(q2)


@qmc.qkernel
def _not_kernel() -> qmc.Bit:
    """``if ~a:`` with ``a`` measured from |0>; target ends in |1>."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    a = qmc.measure(q0)
    if ~a:
        q1 = qmc.x(q1)
    return qmc.measure(q1)


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
            op
            for op in _walk_ops(block.operations)
            if isinstance(op, RuntimeClassicalExpr)
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
            op
            for op in _walk_ops(block.operations)
            if isinstance(op, RuntimeClassicalExpr)
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
            op
            for op in _walk_ops(block.operations)
            if isinstance(op, RuntimeClassicalExpr)
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
            op
            for op in _walk_ops(block.operations)
            if isinstance(op, RuntimeClassicalExpr)
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
            op
            for op in _walk_ops(block.operations)
            if isinstance(op, RuntimeClassicalExpr)
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

    @pytest.mark.parametrize(
        "kernel",
        [
            pytest.param(_and_kernel, id="and"),
            pytest.param(_or_kernel, id="or"),
            pytest.param(_not_kernel, id="not"),
        ],
    )
    def test_runtime_logical_executes(self, transpiler, kernel):
        """Runtime ``&`` / ``|`` / ``~`` over measurements lower to
        RuntimeClassicalExpr and emit via the new backend hook.

        Each kernel sets the gating measurement(s) so the predicate is
        guaranteed true, then applies ``X`` to the target qubit and
        measures it. The path is fully deterministic, so all 100 shots
        must yield ``1``.
        """
        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=100)
            .result()
        )
        assert result.results == [(1, 100)]

    def test_chained_runtime_expr_emits_compound_classical_condition(self, transpiler):
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
        if_ops = [i.operation for i in qc.data if isinstance(i.operation, IfElseOp)]
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
        quantum_steps = [s for s in plan.steps if isinstance(s.segment, QuantumSegment)]
        assert len(quantum_steps) == 1, (
            f"Expected exactly one quantum segment; got "
            f"{len(quantum_steps)}. The runtime classical predicate must be "
            f"kept in the surrounding quantum segment."
        )


# ---------------------------------------------------------------------------
# Regression: ``Vector[Bit]`` element-access provenance (BACKLOG-8270)
# ---------------------------------------------------------------------------


class TestVectorBitElementProvenance:
    """``s = qmc.measure(register)`` then ``if s[i]:`` must lower the same
    way as ``bit = qmc.measure(register[i])`` then ``if bit:``.

    Previously, indexing a measured ``Vector[Bit]`` lost the
    measurement-derived marker because

    - ``find_measurement_results`` only seeded ``MeasureOperation``
      results, ignoring ``MeasureVectorOperation`` outputs, and
    - the emit-time clbit lookup used ``QubitAddress(value.uuid)`` for
      the if/while condition, but vector elements register under
      ``QubitAddress(parent_array.uuid, index)``.

    Flat ``if s[i]:`` raised ``EmitError`` and ``if s[i]:`` inside a
    ``qmc.range`` for loop raised ``MultipleQuantumSegmentsError``.
    """

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    @pytest.mark.parametrize(
        "flip_mask",
        [
            pytest.param((1, 0), id="anc=10"),
            pytest.param((0, 1), id="anc=01"),
            pytest.param((1, 1), id="anc=11"),
            pytest.param((0, 0), id="anc=00"),
        ],
    )
    def test_flat_if_on_measured_vector_element(self, transpiler, flip_mask):
        """``if s[i]:`` flips ``q[i]`` for each set ancilla bit.

        Setup: ``anc[i] = X^{flip_mask[i]} |0>``. The body runs
        ``if s[0]: x(q[0])`` and ``if s[1]: x(q[1])`` after measuring
        ``anc``, so ``q[i] == flip_mask[i]`` and ``q[2] == 0`` every
        shot. Parametrised over all four 2-bit ancilla configurations
        to exercise both true-condition and false-condition arms of
        each ``if``.
        """

        @qmc.qkernel
        def kernel(flip0: qmc.UInt, flip1: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            anc = qmc.qubit_array(2, name="anc")
            if flip0:
                anc[0] = qmc.x(anc[0])
            if flip1:
                anc[1] = qmc.x(anc[1])
            s = qmc.measure(anc)
            if s[0]:
                q[0] = qmc.x(q[0])
            if s[1]:
                q[1] = qmc.x(q[1])
            return qmc.measure(q)

        result = (
            transpiler.transpile(
                kernel, bindings={"flip0": flip_mask[0], "flip1": flip_mask[1]}
            )
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((flip_mask[0], flip_mask[1], 0), 64)]

    def test_compound_if_over_measured_vector_elements(self, transpiler):
        """``if s[0] & s[1]:`` lowers ``s[0]`` and ``s[1]`` as runtime
        clbit references inside a single quantum segment.

        Both ancillas set to ``|1>`` → predicate true → ``q[0]`` flips.
        Pre-fix this raised ``MultipleQuantumSegmentsError`` because the
        BinOp was not classified as measurement-derived.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            anc = qmc.qubit_array(2, name="anc")
            anc[0] = qmc.x(anc[0])
            anc[1] = qmc.x(anc[1])
            s = qmc.measure(anc)
            if s[0] & s[1]:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((1, 0), 64)]

    def test_if_on_measured_vector_inside_unrolled_for_loop(self, transpiler):
        """``for i in qmc.range(N): if s[i]: x(q[i])`` flips exactly the
        qubits whose ancilla measured ``1``.

        Setup: ``anc[0] = |1>``, ``anc[1] = |0>``, ``anc[2] = |1>`` →
        ``q = (1, 0, 1)``. Pre-fix this raised
        ``MultipleQuantumSegmentsError``.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            anc = qmc.qubit_array(3, name="anc")
            anc[0] = qmc.x(anc[0])
            anc[2] = qmc.x(anc[2])
            s = qmc.measure(anc)
            for i in qmc.range(3):
                if s[i]:
                    q[i] = qmc.x(q[i])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((1, 0, 1), 64)]

    def test_measurement_taint_seeded_from_measure_vector(self):
        """White-box: ``a & b`` over a measured ``Vector[Bit]`` lowers to
        ``RuntimeClassicalExpr`` even though the operands are
        ``parent_array`` element accesses, not direct measurement
        results.

        Without seeding ``MeasureVectorOperation`` results, the taint
        analysis cannot reach the BinOp and the predicate stays as a
        plain ``CondOp``, which downstream causes the segmentation pass
        to split the run into multiple quantum segments.
        """

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit_array(2, name="q")
            target = qmc.qubit("t")
            s = qmc.measure(q)
            if s[0] & s[1]:
                target = qmc.x(target)
            return qmc.measure(target)

        block = _lower_to_classical_lowering(kernel)
        runtime_ops = [
            op
            for op in _walk_ops(block.operations)
            if isinstance(op, RuntimeClassicalExpr)
        ]
        assert any(op.kind is RuntimeOpKind.AND for op in runtime_ops), (
            "BinOp(AND) over measured-vector elements must lower to "
            "RuntimeClassicalExpr(AND)."
        )

    def test_flat_if_on_measured_vector_element_cudaq(self):
        """The same ``if s[i]:`` pattern compiles on the CUDA-Q backend.

        CUDA-Q's emit pass goes through both the shared
        ``control_flow_emission.emit_if`` path (the inner condition
        lookup) and the CUDA-Q-specific ``_collect_loop_carried_clbits``
        pre-scan (which also walks ``parent_array``). Exercising this
        end-to-end on CUDA-Q ensures both call sites stay consistent with
        the Qiskit-side coverage. Skipped when CUDA-Q is not installed.
        """
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            anc = qmc.qubit_array(2, name="anc")
            anc[0] = qmc.x(anc[0])
            s = qmc.measure(anc)
            if s[0]:
                q[0] = qmc.x(q[0])
            if s[1]:
                q[1] = qmc.x(q[1])
            return qmc.measure(q)

        tp = CudaqTranspiler()
        result = tp.transpile(kernel).sample(tp.executor(), shots=64).result()
        # CUDA-Q's deterministic runtime if path: anc[0]=1 flips q[0],
        # anc[1]=0 leaves q[1] untouched. Every shot reads (1, 0).
        assert result.results == [((1, 0), 64)]

    def test_if_on_sliced_view_of_measured_vector(self, transpiler):
        """Sliced view of a measured ``Vector[Bit]`` resolves through
        the ``slice_of`` chain.

        Setup: ``anc = [|1>, |0>, |1>, |0>]`` → ``s = (1, 0, 1, 0)``,
        ``s_slice = s[0:4:2]`` resolves to root-space indices ``[0, 2]``
        → values ``(1, 1)``. ``if s_slice[0] & s_slice[1]:`` flips
        ``q[0]``. Pre-fix this raised ``EmitError`` for the simple form
        and ``MultipleQuantumSegmentsError`` for the compound form
        because the address-resolution helper walked ``parent_array``
        only one level and the taint-analysis dependency graph had no
        edge across the ``slice_of`` link (``StripSliceArrayOpsPass``
        removes the explicit ``SliceArrayOperation`` so there is no
        op-mediated edge).
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            anc = qmc.qubit_array(4, name="anc")
            anc[0] = qmc.x(anc[0])
            anc[2] = qmc.x(anc[2])
            s = qmc.measure(anc)
            s_slice = s[0:4:2]
            if s_slice[0] & s_slice[1]:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((1, 0), 64)]

    def test_while_on_measured_vector_element(self, transpiler):
        """A measured ``Vector[Bit]`` element is a valid ``while`` condition.

        The same provenance gap affected ``while s[i]:`` on two fronts:
        ``ValidateWhileContractPass`` rejected the condition because it
        only recognized scalar ``MeasureOperation`` results, and
        ``emit_while`` could not resolve the element's clbit address. This
        test measures ``anc = |00>`` so ``s[0] == 0`` and the loop body
        never runs (a measured ``Vector[Bit]`` element is fixed for the
        loop's lifetime, so a body-entering form cannot terminate). Even
        so the predicate is a runtime measurement value, so a backend
        ``while_loop`` op is genuinely emitted and its condition resolved
        — exercising both the validation and emit paths. ``q`` stays
        ``|0>`` every shot.
        """
        from qiskit.circuit.controlflow import WhileLoopOp

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            anc = qmc.qubit_array(2, name="anc")
            s = qmc.measure(anc)
            while s[0]:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel)
        qc = exe.compiled_quantum[0].circuit
        assert any(isinstance(i.operation, WhileLoopOp) for i in qc.data), (
            "emit_while must emit a runtime while_loop for a measured "
            "Vector[Bit] element condition."
        )
        result = exe.sample(transpiler.executor(), shots=64).result()
        assert result.results == [((0,), 64)]

    def test_while_on_sliced_view_of_measured_vector(self, transpiler):
        """A sliced view element resolves as a ``while`` condition too.

        ``s_slice = s[0:4:2]`` then ``while s_slice[0]:`` must trace the
        ``slice_of`` chain back to the root measured array in both the
        while-condition validation and emit. ``anc = |0000>`` makes the
        condition false so the loop never enters; the runtime
        ``while_loop`` op is still emitted. ``q`` stays ``|0>``.
        """
        from qiskit.circuit.controlflow import WhileLoopOp

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            anc = qmc.qubit_array(4, name="anc")
            s = qmc.measure(anc)
            s_slice = s[0:4:2]
            while s_slice[0]:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        exe = transpiler.transpile(kernel)
        qc = exe.compiled_quantum[0].circuit
        assert any(isinstance(i.operation, WhileLoopOp) for i in qc.data), (
            "emit_while must emit a runtime while_loop for a sliced "
            "measured Vector[Bit] element condition."
        )
        result = exe.sample(transpiler.executor(), shots=64).result()
        assert result.results == [((0,), 64)]

    def test_if_on_runtime_bound_slice_of_measured_vector(self, transpiler):
        """A slice whose bounds are loop variables resolves per iteration.

        ``s_view = s[j:j+1]`` inside ``for j in qmc.range(3)`` has a
        non-constant ``slice_start`` (the loop variable ``j``), so the
        view-element condition ``if s_view[0]:`` (== ``s[j]``) can only be
        resolved by folding the slice bound through the emit-time
        bindings. ``resolve_condition_address`` now resolves both the
        element index and each ``slice_start`` / ``slice_step`` the same
        way (constant, else via the resolver), composing the affine map to
        the root measured array. Setup: ``anc = (1,0,1,0,1,0)`` so
        ``s[j]`` flips ``q[j]`` for even ``j`` only, giving ``q ==
        (1, 0, 1)`` every shot. Before this, a non-constant slice bound
        fell back to the scalar UUID and raised ``EmitError``.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            anc = qmc.qubit_array(6, name="anc")
            anc[0] = qmc.x(anc[0])
            anc[2] = qmc.x(anc[2])
            anc[4] = qmc.x(anc[4])
            s = qmc.measure(anc)
            for j in qmc.range(3):
                s_view = s[j : j + 1]
                if s_view[0]:
                    q[j] = qmc.x(q[j])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((1, 0, 1), 64)]

    def test_if_on_runtime_strided_slice_of_measured_vector(self, transpiler):
        """A strided slice with a loop-variable start resolves too.

        ``s_view = s[j::2]`` then ``if s_view[0]:`` (== ``s[j]``) exercises
        the ``start + step * idx`` composition with a non-constant
        ``slice_start`` (the loop variable ``j``) and a constant
        ``slice_step`` of 2. Setup: ``anc = (0, 1, 0, 0)`` so only
        ``s[1] == 1`` flips ``q[1]``, giving ``q == (0, 1)`` every shot.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            anc = qmc.qubit_array(4, name="anc")
            anc[1] = qmc.x(anc[1])
            s = qmc.measure(anc)
            for j in qmc.range(2):
                s_view = s[j::2]
                if s_view[0]:
                    q[j] = qmc.x(q[j])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((0, 1), 64)]

    def test_for_loop_condition_only_loop_var(self, transpiler):
        """A loop variable that appears only in a runtime condition forces
        the for loop to unroll.

        ``for j in qmc.range(3): if s[j]: q[0] = x(q[0])`` reads the loop
        variable ``j`` only inside the ``if`` condition — the body indexes
        ``q[0]``, not ``q[j]``. A native backend for-loop would keep ``j``
        as a loop parameter that classical-register indexing cannot
        consume, so ``LoopAnalyzer`` must detect the condition's loop-var
        dependency and unroll. Setup: ``anc = (1, 0, 1)``, so ``q[0]`` is
        flipped at ``j = 0`` and ``j = 2`` (twice → back to 0). Without
        unrolling this raised ``EmitError``.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            anc = qmc.qubit_array(3, name="anc")
            anc[0] = qmc.x(anc[0])
            anc[2] = qmc.x(anc[2])
            s = qmc.measure(anc)
            for j in qmc.range(3):
                if s[j]:
                    q[0] = qmc.x(q[0])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((0,), 64)]

    def test_for_loop_condition_only_loop_var_in_slice_bound(self, transpiler):
        """The loop variable in a slice bound of a condition also forces
        unrolling.

        ``for j in qmc.range(3): if s[j:j+1][0]: ...`` carries the loop
        variable in the view's ``slice_start`` rather than the element
        index, so ``LoopAnalyzer`` must inspect the ``slice_of`` chain too.
        Setup: ``anc = (0, 1, 0)`` → only ``j = 1`` flips ``q[0]`` → 1.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            anc = qmc.qubit_array(3, name="anc")
            anc[1] = qmc.x(anc[1])
            s = qmc.measure(anc)
            for j in qmc.range(3):
                if s[j : j + 1][0]:
                    q[0] = qmc.x(q[0])
            return qmc.measure(q)

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=64)
            .result()
        )
        assert result.results == [((1,), 64)]

    def test_while_loop_carried_init_from_measured_vector_element(self, transpiler):
        """A ``while`` whose initial condition is a measured ``Vector[Bit]``
        element aliases its loop-carried clbit correctly.

        ``bit = s[0]; while bit: bit = qmc.measure(qz)`` reassigns the
        condition in the body (loop-carried). The resource allocator must
        recognise that the initial condition ``s[0]`` lives at the measured
        vector's ``(root, index)`` clbit so the body's re-measurement
        aliases onto the same physical clbit the ``while`` reads. A broken
        alias leaves the condition stuck on the stale initial value and the
        loop never terminates, so this is verified structurally (no
        execution): the emitted ``while_loop`` condition clbit must equal
        the clbit the body re-measures into. Regression for a loop-carried
        alias that ignored vector-element sources.
        """
        from qiskit.circuit.controlflow import WhileLoopOp

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            anc = qmc.qubit_array(1, name="anc")
            qz = qmc.qubit("qz")
            anc[0] = qmc.x(anc[0])
            s = qmc.measure(anc)
            bit = s[0]
            while bit:
                bit = qmc.measure(qz)
            return bit

        qc = transpiler.transpile(kernel).compiled_quantum[0].circuit
        clbit_index = {bit: i for i, bit in enumerate(qc.clbits)}
        while_ops = [
            inst for inst in qc.data if isinstance(inst.operation, WhileLoopOp)
        ]
        assert while_ops, "expected a runtime while_loop"
        cond_clbit = clbit_index[while_ops[0].operation.condition[0]]
        body = while_ops[0].operation.blocks[0]
        body_to_outer = {b: while_ops[0].clbits[j] for j, b in enumerate(body.clbits)}
        body_measure_clbits = [
            clbit_index[body_to_outer[binst.clbits[0]]]
            for binst in body.data
            if binst.operation.name == "measure"
        ]
        assert body_measure_clbits, "expected a re-measure inside the while body"
        assert all(c == cond_clbit for c in body_measure_clbits), (
            "loop-carried alias broken: the while condition reads a different "
            "clbit than the body re-measures into, which would loop forever."
        )

    def test_out_of_bounds_constant_element_index_rejected(self, transpiler):
        """A constant index that overflows a constant dimension is rejected
        at trace time rather than silently misresolving.

        ``empty = s[1:1]; empty[0]`` indexes a length-0 slice view; the
        element would otherwise compose to a valid-but-wrong root clbit
        (``s[1]``) and be read silently. A plain out-of-range index
        (``s[5]`` on a length-2 register) is rejected the same way. Both
        raise ``IndexError`` while a kernel is being traced.
        """

        @qmc.qkernel
        def empty_slice_kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            anc = qmc.qubit_array(3, name="anc")
            s = qmc.measure(anc)
            empty = s[1:1]
            if empty[0]:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        with pytest.raises(IndexError):
            transpiler.transpile(empty_slice_kernel)

        @qmc.qkernel
        def oob_kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            anc = qmc.qubit_array(2, name="anc")
            s = qmc.measure(anc)
            if s[5]:
                q[0] = qmc.x(q[0])
            return qmc.measure(q)

        with pytest.raises(IndexError):
            transpiler.transpile(oob_kernel)

    @pytest.mark.parametrize("msg_bit", [0, 1])
    def test_teleportation_with_vector_measured_corrections(self, transpiler, msg_bit):
        """End-to-end quantum teleportation built with a ``qubit_array``.

        The Bell measurement of the message qubit plus Alice's half of the
        Bell pair is taken as a single vector measurement
        ``m = qmc.measure(ab)``, and the X / Z corrections on Bob are
        conditioned on the elements ``m[1]`` / ``m[0]``. This is exactly the
        measured-``Vector[Bit]``-element-as-runtime-condition pattern that
        the unfixed pipeline rejected with ``EmitError``. The corrections
        make Bob deterministic, so teleporting ``|msg_bit>`` yields
        ``msg_bit`` on Bob every shot regardless of the random intermediate
        Bell-measurement outcomes (which split the result into several
        rows, all carrying the same Bob value).
        """

        @qmc.qkernel
        def teleport(prep: qmc.UInt) -> qmc.Bit:
            ab = qmc.qubit_array(2, name="ab")  # ab[0]=message, ab[1]=Alice
            bob = qmc.qubit("bob")
            if prep:
                ab[0] = qmc.x(ab[0])
            ab[1] = qmc.h(ab[1])
            ab[1], bob = qmc.cx(ab[1], bob)
            ab[0], ab[1] = qmc.cx(ab[0], ab[1])
            ab[0] = qmc.h(ab[0])
            m = qmc.measure(ab)
            if m[1]:
                bob = qmc.x(bob)
            if m[0]:
                bob = qmc.z(bob)
            return qmc.measure(bob)

        result = (
            transpiler.transpile(teleport, bindings={"prep": msg_bit})
            .sample(transpiler.executor(), shots=128)
            .result()
        )
        assert all(bob == msg_bit for bob, _ in result.results), (
            f"teleporting |{msg_bit}> must put Bob in |{msg_bit}>, got {result.results}"
        )
        assert sum(count for _, count in result.results) == 128
