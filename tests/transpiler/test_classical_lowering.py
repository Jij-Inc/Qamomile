"""Tests for ``ClassicalLoweringPass`` and ``RuntimeClassicalExpr``.

The pass identifies measurement-derived classical ops (CompOp / CondOp /
NotOp / BinOp) and rewrites them to ``RuntimeClassicalExpr`` so emit can
dispatch them through a dedicated backend hook instead of the legacy
fold-or-translate path. Non-measurement-derived classical ops are left
unchanged so the existing fold paths continue to handle them.

Tests cover four layers:

1. **IR rewriting** — synthetic kernels are run through ``transpile()``
   up to (and including) ``classical_lowering``; the resulting IR is
   inspected to check which ops became ``RuntimeClassicalExpr`` and
   which stayed as their original type.

2. **End-to-end** — the same kernels run on the qiskit simulator and
   produce correct results, exercising the full pipeline including the
   new ``_emit_runtime_classical_expr`` hook.

3. **Segmentation** — white-box checks that an expr bridging a
   measurement to a runtime if/while stays inside the single quantum
   segment.

4. **Host-side outputs** — an expr whose result is a block output (or
   feeds host-side post-processing) is instead routed to a post-quantum
   classical segment and evaluated per shot by ``ClassicalExecutor``.
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
from qamomile.circuit.ir.value import TupleValue
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor


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


class _CountsExecutor(QuantumExecutor[object]):
    """Return fixed raw counts while ignoring the backend circuit."""

    def __init__(self, counts: dict[str, int]) -> None:
        """Store fixed counts returned by ``execute``.

        Args:
            counts (dict[str, int]): Raw backend bitstring counts.
        """
        self._counts = counts

    def execute(self, circuit: object, shots: int) -> dict[str, int]:
        """Return the fixed raw counts.

        Args:
            circuit (object): Ignored backend circuit.
            shots (int): Ignored shot count.

        Returns:
            dict[str, int]: Fixed raw backend bitstring counts.
        """
        return self._counts

    def bind_parameters(
        self,
        circuit: object,
        bindings: dict[str, float],
        parameter_metadata: ParameterMetadata,
    ) -> object:
        """Return the circuit unchanged.

        Args:
            circuit (object): Backend circuit to bind.
            bindings (dict[str, float]): Runtime parameter bindings.
            parameter_metadata (ParameterMetadata): Parameter metadata for
                ``bindings``.

        Returns:
            object: The original circuit.
        """
        return circuit


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

    @pytest.mark.cudaq
    def test_flat_if_on_measured_vector_element_cudaq(self):
        """The same ``if s[i]:`` pattern compiles on the CUDA-Q backend.

        CUDA-Q's emit pass goes through both the shared
        ``control_flow_emission.emit_if`` path (the inner condition
        lookup) and the CUDA-Q-specific ``_collect_loop_carried_clbits``
        pre-scan (which also walks ``parent_array``). Exercising this
        end-to-end on CUDA-Q ensures both call sites stay consistent with
        the Qiskit-side coverage. Runs in ``-m cudaq`` sessions only:
        loading cudaq (and the torch/OpenMP runtimes it brings) into a
        default session that also executes qiskit-aer can segfault — see
        tests/_cudaq_isolation.py.
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

    @pytest.mark.parametrize("idx", [0, 1])
    def test_while_loop_carried_init_from_bound_param_vector_element(
        self, transpiler, idx
    ):
        """A loop-carried ``while`` whose initial condition indexes a measured
        ``Vector[Bit]`` with a transpile-time *bound parameter* aliases its
        loop-carried clbit correctly.

        Same loop-carried alias contract as
        ``test_while_loop_carried_init_from_measured_vector_element``, but the
        element index in ``bit = s[idx]`` comes from a kernel parameter
        resolved through ``bindings={"idx": idx}`` rather than a literal. The
        binding is folded to a constant before the resource allocator runs, so
        the allocator must still resolve ``s[idx]`` to the measured vector's
        ``(root, idx)`` clbit and alias the body re-measurement onto it.
        Verified structurally (no execution): a broken alias would leave the
        ``while`` reading a stale clbit and never terminate. Regression for the
        bound-parameter index path of the loop-carried alias.
        """
        from qiskit.circuit.controlflow import WhileLoopOp

        @qmc.qkernel
        def kernel(idx: qmc.UInt) -> qmc.Bit:
            anc = qmc.qubit_array(2, name="anc")
            qz = qmc.qubit("qz")
            anc[idx] = qmc.x(anc[idx])
            s = qmc.measure(anc)
            bit = s[idx]
            while bit:
                bit = qmc.measure(qz)
            return bit

        qc = (
            transpiler.transpile(kernel, bindings={"idx": idx})
            .compiled_quantum[0]
            .circuit
        )
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
            "loop-carried alias broken for a bound-parameter index: the while "
            "condition reads a different clbit than the body re-measures into, "
            "which would loop forever."
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


# ---------------------------------------------------------------------------
# Layer 4: measurement-derived classical outputs (host-side evaluation)
# ---------------------------------------------------------------------------


class TestMeasurementDerivedOutput:
    """A measurement-derived classical expression returned as a kernel
    output must be evaluated host-side and surfaced by the orchestrator.

    Previously the segmenter unconditionally absorbed every
    ``RuntimeClassicalExpr`` into the quantum segment, where nothing
    surfaces its value: a kernel returning ``bits[0] & bits[1]`` sampled
    ``[(None, shots)]``. The expr must instead land in a post-quantum
    classical segment executed by ``ClassicalExecutor`` per shot.
    """

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    @pytest.mark.parametrize(
        "flip_mask",
        [
            pytest.param((0, 0), id="00"),
            pytest.param((0, 1), id="01"),
            pytest.param((1, 0), id="10"),
            pytest.param((1, 1), id="11"),
        ],
    )
    def test_condop_and_output_sample(self, transpiler, flip_mask):
        """``return bits[0] & bits[1]`` samples the host-computed AND of
        the deterministic measurement outcomes (the original repro shape,
        reading elements of a measured ``Vector[Bit]``)."""

        @qmc.qkernel
        def kernel(flip0: qmc.UInt, flip1: qmc.UInt) -> qmc.Bit:
            qs = qmc.qubit_array(2, name="qs")
            if flip0:
                qs[0] = qmc.x(qs[0])
            if flip1:
                qs[1] = qmc.x(qs[1])
            bits = qmc.measure(qs)
            return bits[0] & bits[1]

        expected = flip_mask[0] & flip_mask[1]
        result = (
            transpiler.transpile(
                kernel, bindings={"flip0": flip_mask[0], "flip1": flip_mask[1]}
            )
            .sample(transpiler.executor(), shots=50)
            .result()
        )
        assert result.results == [(expected, 50)]

    def test_condop_output_sample_aggregates_postprocessed_values(self, transpiler):
        """``sample()`` aggregates counts by the postprocessed return value,
        not by the raw backend bitstring that produced that value."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qs = qmc.qubit_array(2, name="qs")
            qs = qmc.h(qs)
            bits = qmc.measure(qs)
            return bits[0] & bits[1]

        result = (
            transpiler.transpile(kernel)
            .sample(
                _CountsExecutor({"00": 1, "01": 2, "10": 3, "11": 4}),
                shots=10,
            )
            .result()
        )
        assert result.results == [(0, 6), (1, 4)]

    def test_tuple_output_resolves_expr_and_measured_vector_elements(self, transpiler):
        """Tuple outputs may mix host-computed expressions with direct
        measured ``Vector[Bit]`` elements."""

        @qmc.qkernel
        def kernel() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(2, name="qs")
            qs[0] = qmc.x(qs[0])
            bits = qmc.measure(qs)
            return bits[0] & bits[1], bits[0], bits[1]

        exe = transpiler.transpile(kernel)
        assert exe.sample(_CountsExecutor({"01": 5}), shots=5).result().results == [
            ((0, 1, 0), 5)
        ]
        assert exe.run(_CountsExecutor({"01": 1})).result() == (0, 1, 0)

    def test_vector_output_sample_still_reconstructs_measured_bits(self, transpiler):
        """Whole ``Vector[Bit]`` outputs still reconstruct from measured
        indexed context entries."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, name="qs")
            return qmc.measure(qs)

        result = (
            transpiler.transpile(kernel)
            .sample(_CountsExecutor({"01": 5}), shots=5)
            .result()
        )
        assert result.results == [((1, 0), 5)]

    def test_qfixed_output_sample_still_uses_decoded_values(self, transpiler):
        """QFixed measurement still samples post-classical decoded floats."""

        @qmc.qkernel
        def kernel() -> qmc.Float:
            qs = qmc.qubit_array(2, name="qs")
            qf = qmc.cast(qs, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        result = (
            transpiler.transpile(kernel)
            .sample(
                _CountsExecutor({"00": 2, "01": 3, "10": 5, "11": 7}),
                shots=17,
            )
            .result()
        )
        assert result.results == [(0.0, 2), (0.25, 3), (0.5, 5), (0.75, 7)]

    def test_slice_output_resolves_measured_vector_view(self, transpiler):
        """A whole sliced ``Vector[Bit]`` output reconstructs from the root
        measured vector entries."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, name="qs")
            bits = qmc.measure(qs)
            return bits[1:]

        exe = transpiler.transpile(kernel)
        assert exe.sample(_CountsExecutor({"110": 4}), shots=4).result().results == [
            ((1, 1), 4)
        ]
        assert exe.run(_CountsExecutor({"110": 1})).result() == (1, 1)

    def test_runtime_slice_output_resolves_measured_vector_view(self, transpiler):
        """A whole sliced output with a runtime-bound start resolves using
        execution bindings."""

        @qmc.qkernel
        def kernel(lo: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, name="qs")
            bits = qmc.measure(qs)
            return bits[lo:]

        exe = transpiler.transpile(kernel)
        assert exe.sample(
            _CountsExecutor({"110": 4}),
            shots=4,
            bindings={"lo": 1},
        ).result().results == [((1, 1), 4)]
        assert exe.run(_CountsExecutor({"110": 1}), bindings={"lo": 1}).result() == (
            1,
            1,
        )

    def test_runtime_slice_element_output_resolves_root_entry(self, transpiler):
        """A direct element output from a runtime-bound slice resolves to the
        corresponding root measured vector entry."""

        @qmc.qkernel
        def kernel(lo: qmc.UInt) -> qmc.Bit:
            qs = qmc.qubit_array(3, name="qs")
            bits = qmc.measure(qs)
            view = bits[lo:]
            return view[0]

        exe = transpiler.transpile(kernel)
        assert exe.sample(
            _CountsExecutor({"010": 3}),
            shots=3,
            bindings={"lo": 1},
        ).result().results == [(1, 3)]
        assert exe.run(_CountsExecutor({"010": 1}), bindings={"lo": 1}).result() == 1

    def test_abi_public_inputs_excludes_quantum_arguments(self):
        """Program ABI exposes runtime-bindable classical inputs only."""
        from qamomile.circuit.ir.block import Block, BlockKind
        from qamomile.circuit.ir.operation.gate import MeasureOperation
        from qamomile.circuit.ir.types.primitives import (
            BitType,
            FloatType,
            QubitType,
            UIntType,
        )
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.passes.separate import SegmentationPass

        q = Value(type=QubitType(), name="q")
        theta = Value(type=FloatType(), name="theta")
        pair = TupleValue(
            name="pair",
            elements=(Value(type=UIntType(), name="pair_0"),),
        )
        bit = Value(type=BitType(), name="bit")
        measure = MeasureOperation(operands=[q], results=[bit])
        block = Block(
            name="abi_inputs",
            label_args=["q", "theta", "pair"],
            input_values=[q, theta, pair],
            operations=[measure],
            output_values=[bit],
            kind=BlockKind.AFFINE,
        )

        plan = SegmentationPass().run(block)

        assert "q" not in plan.abi.public_inputs
        assert plan.abi.public_inputs["theta"].uuid == theta.uuid
        assert plan.abi.public_inputs["pair"].uuid == pair.uuid

    def test_expr_output_reads_sliced_element_in_classical_executor(self, transpiler):
        """Host-side runtime expressions can read elements of measured vector
        slice views."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qs = qmc.qubit_array(3, name="qs")
            bits = qmc.measure(qs)
            view = bits[1:]
            return view[1] & bits[0]

        exe = transpiler.transpile(kernel)
        assert exe.sample(_CountsExecutor({"101": 6}), shots=6).result().results == [
            (1, 6)
        ]
        assert exe.run(_CountsExecutor({"101": 1})).result() == 1

    def test_tuple_input_output_resolves_runtime_name_binding(self, transpiler):
        """A tuple-typed input returned as a structural output resolves from
        runtime bindings."""

        @qmc.qkernel
        def kernel(
            pair: qmc.Tuple[qmc.UInt, qmc.UInt],
        ) -> qmc.Tuple[qmc.UInt, qmc.UInt]:
            q = qmc.qubit("q")
            _ = qmc.measure(q)
            return pair

        exe = transpiler.transpile(kernel)
        assert exe.sample(
            transpiler.executor(), shots=5, bindings={"pair": (2, 3)}
        ).result().results == [((2, 3), 5)]
        assert exe.run(transpiler.executor(), bindings={"pair": (2, 3)}).result() == (
            2,
            3,
        )

    def test_tuple_input_element_output_resolves_runtime_binding(self, transpiler):
        """A scalar element of a tuple-typed input resolves from runtime
        bindings."""

        @qmc.qkernel
        def kernel(pair: qmc.Tuple[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            q = qmc.qubit("q")
            _ = qmc.measure(q)
            return pair[0]

        exe = transpiler.transpile(kernel)
        assert exe.plan is not None
        assert isinstance(exe.plan.abi.public_inputs["pair"], TupleValue)
        assert exe.sample(
            transpiler.executor(), shots=5, bindings={"pair": (2, 3)}
        ).result().results == [(2, 5)]
        assert exe.run(transpiler.executor(), bindings={"pair": (2, 3)}).result() == 2

    def test_tuple_input_element_expr_reads_runtime_binding(self, transpiler):
        """Post-classical expressions can read scalar elements of tuple-typed
        runtime bindings."""

        @qmc.qkernel
        def kernel(pair: qmc.Tuple[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            q = qmc.qubit("q")
            _ = qmc.measure(q)
            return pair[0] + pair[1]

        exe = transpiler.transpile(kernel)
        assert exe.sample(
            transpiler.executor(), shots=5, bindings={"pair": (2, 3)}
        ).result().results == [(5, 5)]
        assert exe.run(transpiler.executor(), bindings={"pair": (2, 3)}).result() == 5

    def test_subqkernel_tuple_output_can_be_indexed_by_caller(self, transpiler):
        """A tuple returned by a sub-qkernel keeps its element handles."""

        @qmc.qkernel
        def identity(
            pair: qmc.Tuple[qmc.UInt, qmc.UInt],
        ) -> qmc.Tuple[qmc.UInt, qmc.UInt]:
            return pair

        @qmc.qkernel
        def kernel(pair: qmc.Tuple[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            q = qmc.qubit("q")
            _ = qmc.measure(q)
            returned = identity(pair)
            return returned[0] + returned[1]

        exe = transpiler.transpile(kernel)
        assert exe.sample(
            transpiler.executor(), shots=5, bindings={"pair": (2, 3)}
        ).result().results == [(5, 5)]
        assert exe.run(transpiler.executor(), bindings={"pair": (2, 3)}).result() == 5

    def test_tuple_input_element_alias_does_not_override_public_input(self, transpiler):
        """Tuple element aliases must not overwrite a same-named public
        runtime input."""

        @qmc.qkernel
        def kernel(
            pair: qmc.Tuple[qmc.UInt, qmc.UInt],
            pair_0: qmc.UInt,
        ) -> qmc.UInt:
            q = qmc.qubit("q")
            _ = qmc.measure(q)
            return pair_0 + pair[0]

        exe = transpiler.transpile(kernel)
        assert exe.sample(
            transpiler.executor(),
            shots=5,
            bindings={"pair": (2, 3), "pair_0": 9},
        ).result().results == [(11, 5)]
        assert (
            exe.run(
                transpiler.executor(),
                bindings={"pair": (2, 3), "pair_0": 9},
            ).result()
            == 11
        )

    def test_vector_uint_element_output_keeps_parent_container_fallback(
        self, transpiler
    ):
        """A scalar element of a bound classical vector output still resolves
        through its parent container."""

        @qmc.qkernel
        def kernel(input: qmc.Vector[qmc.UInt]) -> qmc.UInt:
            q = qmc.qubit("q")
            _ = qmc.measure(q)
            return input[1]

        exe = transpiler.transpile(kernel, bindings={"input": [4, 5, 6]})
        assert exe.sample(transpiler.executor(), shots=3).result().results == [(5, 3)]
        assert exe.run(transpiler.executor()).result() == 5

    def test_dynamic_if_phi_expr_branch_output_uses_post_value(self, transpiler):
        """A quantum-effective if returning a Phi with an expression branch
        uses the host-computed Phi result, not a one-sided clbit alias."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qa = qmc.qubit("qa")
            target = qmc.qubit("target")
            qa = qmc.x(qa)
            a = qmc.measure(qa)
            out = a
            if a:
                target = qmc.x(target)
                out = ~a
            else:
                out = a
            return out

        result = transpiler.transpile(kernel).sample(transpiler.executor(), shots=20)
        assert result.result().results == [(0, 20)]

    def test_dynamic_if_phi_shadow_computes_runtime_condition(self, transpiler):
        """A post-classical shadow if also computes a runtime expression used
        as the branch condition."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qa = qmc.qubit("qa")
            qb = qmc.qubit("qb")
            target = qmc.qubit("target")
            qa = qmc.x(qa)
            qb = qmc.x(qb)
            a = qmc.measure(qa)
            b = qmc.measure(qb)
            out = a
            if a & b:
                target = qmc.x(target)
                out = ~a
            else:
                out = a
            return out

        result = transpiler.transpile(kernel).sample(transpiler.executor(), shots=20)
        assert result.result().results == [(0, 20)]

    def test_dynamic_if_phi_different_direct_clbits_use_selected_branch(
        self, transpiler
    ):
        """Different direct clbit Phi sources are selected host-side instead
        of being collapsed into a single alias."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qa = qmc.qubit("qa")
            qb = qmc.qubit("qb")
            selq = qmc.qubit("sel")
            target = qmc.qubit("target")
            qb = qmc.x(qb)
            a = qmc.measure(qa)
            b = qmc.measure(qb)
            sel = qmc.measure(selq)
            out = a
            if sel:
                target = qmc.x(target)
                out = a
            else:
                out = b
            return out

        result = transpiler.transpile(kernel).sample(transpiler.executor(), shots=20)
        assert result.result().results == [(1, 20)]

    def test_dynamic_if_phi_same_direct_clbit_still_resolves(self, transpiler):
        """The safe same-clbit Phi case remains supported."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qa = qmc.qubit("qa")
            target = qmc.qubit("target")
            qa = qmc.x(qa)
            a = qmc.measure(qa)
            out = a
            if a:
                target = qmc.x(target)
                out = a
            else:
                out = a
            return out

        result = transpiler.transpile(kernel).sample(transpiler.executor(), shots=20)
        assert result.result().results == [(1, 20)]

    def test_dynamic_if_phi_nested_under_for_uses_post_value(self, transpiler):
        """A host-needed Bit Phi inside a qmc.range loop is shadowed
        host-side instead of resolving to ``None``."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qa = qmc.qubit("qa")
            qb = qmc.qubit("qb")
            selq = qmc.qubit("sel")
            target = qmc.qubit("target")
            qb = qmc.x(qb)
            a = qmc.measure(qa)
            b = qmc.measure(qb)
            sel = qmc.measure(selq)
            out = a
            for _ in qmc.range(1):
                if sel:
                    target = qmc.h(target)
                    target = qmc.h(target)
                    out = a
                else:
                    target = qmc.h(target)
                    target = qmc.h(target)
                    out = b
            return out

        result = transpiler.transpile(kernel).sample(transpiler.executor(), shots=20)
        assert result.result().results == [(1, 20)]

    def test_nested_shadow_if_uses_outer_runtime_condition_expr(self, transpiler):
        """A loop-nested shadow if recomputes an outer runtime condition."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qa = qmc.qubit("qa")
            qb = qmc.qubit("qb")
            qc = qmc.qubit("qc")
            target = qmc.qubit("target")
            qb = qmc.x(qb)
            a = qmc.measure(qa)
            b = qmc.measure(qb)
            c = qmc.measure(qc)
            sel = a | c
            out = a
            for _ in qmc.range(1):
                if sel:
                    target = qmc.h(target)
                    target = qmc.h(target)
                    out = a
                else:
                    target = qmc.h(target)
                    target = qmc.h(target)
                    out = b
            return out

        result = transpiler.transpile(kernel).sample(transpiler.executor(), shots=20)
        assert result.result().results == [(1, 20)]

    def test_dynamic_if_phi_nested_under_for_items_uses_post_value(self, transpiler):
        """A host-needed Bit Phi inside qmc.items is shadowed host-side."""

        @qmc.qkernel
        def kernel(spec: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.Bit:
            qa = qmc.qubit("qa")
            qb = qmc.qubit("qb")
            selq = qmc.qubit("sel")
            target = qmc.qubit("target")
            qb = qmc.x(qb)
            a = qmc.measure(qa)
            b = qmc.measure(qb)
            sel = qmc.measure(selq)
            out = a
            for _key, _value in qmc.items(spec):
                if sel:
                    target = qmc.h(target)
                    target = qmc.h(target)
                    out = a
                else:
                    target = qmc.h(target)
                    target = qmc.h(target)
                    out = b
            return out

        exe = transpiler.transpile(kernel, bindings={"spec": {0: 1}})
        assert exe.sample(transpiler.executor(), shots=20).result().results == [(1, 20)]

    def test_while_loop_carried_external_measurement_is_rejected(self, transpiler):
        """A while condition cannot be updated from measurements taken before
        the loop body."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit_array(5, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            bit = qmc.measure(q[0])
            sel = qmc.measure(q[1])
            a = qmc.measure(q[3])
            b = qmc.measure(q[2])
            while bit:
                q[4] = qmc.x(q[4])
                if sel:
                    bit = b
                else:
                    bit = a
            return qmc.measure(q[4])

        with pytest.raises(EmitError, match="updated by measurements produced"):
            transpiler.transpile(kernel)

    def test_condop_or_output_sample(self, transpiler):
        """``return a | b`` with ``a=1``, ``b=0`` samples ``1``."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            return a | b

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=50)
            .result()
        )
        assert result.results == [(1, 50)]

    def test_notop_output_sample(self, transpiler):
        """``return ~a`` with ``a`` measured from ``|0>`` samples ``1``."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            a = qmc.measure(q0)
            return ~a

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=50)
            .result()
        )
        assert result.results == [(1, 50)]

    def test_condop_output_run(self, transpiler):
        """The single-shot ``run()`` path resolves the host-computed
        expression output the same way ``sample()`` does."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            return a & b

        exe = transpiler.transpile(kernel)
        assert exe.run(transpiler.executor()).result() == 1

    def test_chained_exprs_output(self, transpiler):
        """A chain of host-side expressions (``~(a & b)``) evaluates in
        source order inside the post-quantum classical segment."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            s = a & b  # 1 & 0 = 0
            return ~s

        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=50)
            .result()
        )
        assert result.results == [(1, 50)]

    @pytest.mark.parametrize(
        "flip_second, expected",
        [
            pytest.param(True, 1, id="and-true"),
            pytest.param(False, 0, id="and-false"),
        ],
    )
    def test_expr_as_condition_and_output(self, transpiler, flip_second, expected):
        """An expression consumed by an in-circuit ``if`` *and* returned is
        placed in both worlds: the runtime if fires on the backend
        expression while the returned value is recomputed host-side."""

        @qmc.qkernel
        def kernel(flip_b: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            if flip_b:
                q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            s = a & b
            if s:
                q2 = qmc.x(q2)
            return s, qmc.measure(q2)

        result = (
            transpiler.transpile(kernel, bindings={"flip_b": int(flip_second)})
            .sample(transpiler.executor(), shots=50)
            .result()
        )
        # q2 is flipped exactly when s is true, so both outputs agree.
        assert result.results == [((expected, expected), 50)]

    def test_expr_between_measurements_sinks_past_quantum_ops(self, transpiler):
        """A host-side expression written between measurements defers past
        the remaining quantum ops instead of splitting the quantum segment
        (no ``MultipleQuantumSegmentsError``) and still evaluates correctly."""
        from qamomile.circuit.transpiler.segments import QuantumSegment

        @qmc.qkernel
        def kernel() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            a = qmc.measure(q0)
            s = ~a
            q1 = qmc.x(q1)
            b = qmc.measure(q1)
            return s, b

        exe = transpiler.transpile(kernel)
        quantum_steps = [
            step for step in exe.plan.steps if isinstance(step.segment, QuantumSegment)
        ]
        assert len(quantum_steps) == 1
        result = exe.sample(transpiler.executor(), shots=50).result()
        assert result.results == [((1, 1), 50)]

    @pytest.mark.quri_parts
    def test_condop_output_sample_quri_parts(self):
        """Host-side evaluation makes measurement-derived outputs work on a
        backend without dynamic-circuit (runtime classical expression)
        support — previously the expr stranded in the quantum segment made
        this kernel unexecutable on QURI Parts."""
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qs = qmc.qubit_array(2, name="qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            bits = qmc.measure(qs)
            return bits[0] & bits[1]

        transpiler = QuriPartsTranspiler()
        result = (
            transpiler.transpile(kernel)
            .sample(transpiler.executor(), shots=50)
            .result()
        )
        assert result.results == [(1, 50)]


class TestMeasurementDerivedOutputSegmentation:
    """White-box checks of where segmentation places a
    ``RuntimeClassicalExpr`` relative to the quantum segment."""

    @pytest.fixture
    def transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_output_expr_lands_in_post_classical_segment(self, transpiler):
        """An expr consumed only by the block output is routed to a
        post-quantum classical segment, not the quantum segment."""
        from qamomile.circuit.transpiler.segments import (
            ClassicalSegment,
            QuantumSegment,
        )

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qs = qmc.qubit_array(2, name="qs")
            qs[0] = qmc.x(qs[0])
            bits = qmc.measure(qs)
            return bits[0] & bits[1]

        plan = transpiler.transpile(kernel).plan
        quantum_segments = [
            s.segment for s in plan.steps if isinstance(s.segment, QuantumSegment)
        ]
        classical_segments = [
            s.segment for s in plan.steps if isinstance(s.segment, ClassicalSegment)
        ]
        assert len(quantum_segments) == 1
        assert not any(
            isinstance(op, RuntimeClassicalExpr)
            for op in quantum_segments[0].operations
        ), "output-only expr must not stay in the quantum segment"
        assert any(
            isinstance(op, RuntimeClassicalExpr)
            for seg in classical_segments
            for op in seg.operations
        ), "output-only expr must be placed in a classical segment"

    def test_condition_and_output_expr_duplicated(self, transpiler):
        """An expr consumed by a runtime if *and* returned appears in both
        the quantum segment (for backend emission) and a classical segment
        (for host-side evaluation). Duplication is safe: the op is pure."""
        from qamomile.circuit.transpiler.segments import (
            ClassicalSegment,
            QuantumSegment,
        )

        @qmc.qkernel
        def kernel() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            s = a & b
            if s:
                q2 = qmc.x(q2)
            return s, qmc.measure(q2)

        plan = transpiler.transpile(kernel).plan
        quantum_ops = [
            op
            for s in plan.steps
            if isinstance(s.segment, QuantumSegment)
            for op in s.segment.operations
        ]
        classical_ops = [
            op
            for s in plan.steps
            if isinstance(s.segment, ClassicalSegment)
            for op in s.segment.operations
        ]
        assert any(isinstance(op, RuntimeClassicalExpr) for op in quantum_ops)
        assert any(isinstance(op, RuntimeClassicalExpr) for op in classical_ops)

    def test_condition_only_expr_stays_out_of_classical_segments(self, transpiler):
        """An expr consumed only by a runtime if stays exclusive to the
        quantum segment — no spurious host-side copy."""
        from qamomile.circuit.transpiler.segments import ClassicalSegment

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

        plan = transpiler.transpile(kernel).plan
        classical_ops = [
            op
            for s in plan.steps
            if isinstance(s.segment, ClassicalSegment)
            for op in s.segment.operations
        ]
        assert not any(isinstance(op, RuntimeClassicalExpr) for op in classical_ops)


class TestRuntimeExprHostDispatch:
    """Unit coverage of ``ClassicalExecutor``'s ``RuntimeClassicalExpr``
    dispatch: every ``RuntimeOpKind`` maps onto the shared ``eval_utils``
    helper of its family. Constructed as synthetic IR because the frontend
    cannot yet produce every kind under measurement taint (only ``& | ~``
    on Bit; arithmetic and comparisons need the QFixed float path)."""

    @staticmethod
    def _execute_single(kind, operand_values):
        """Run one synthetic ``RuntimeClassicalExpr`` and return its result.

        Args:
            kind (RuntimeOpKind): The expression kind to execute.
            operand_values (list[int | float]): Concrete operand constants.

        Returns:
            Any: The value the executor stored for the result UUID.
        """
        from qamomile.circuit.ir.types.primitives import FloatType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
        from qamomile.circuit.transpiler.execution_context import ExecutionContext
        from qamomile.circuit.transpiler.segments import ClassicalSegment

        operands = [
            Value(type=FloatType(), name=f"in{i}").with_const(v)
            for i, v in enumerate(operand_values)
        ]
        out = Value(type=FloatType(), name="out")
        op = RuntimeClassicalExpr(operands=operands, results=[out], kind=kind)
        segment = ClassicalSegment(operations=[op])
        results = ClassicalExecutor().execute(segment, ExecutionContext())
        return results[out.uuid]

    @pytest.mark.parametrize(
        "kind, lhs, rhs, expected",
        [
            pytest.param(RuntimeOpKind.ADD, 3, 4, 7, id="add"),
            pytest.param(RuntimeOpKind.SUB, 9, 4, 5, id="sub"),
            pytest.param(RuntimeOpKind.MUL, 3, 4, 12, id="mul"),
            pytest.param(RuntimeOpKind.DIV, 8, 2, 4.0, id="div"),
            pytest.param(RuntimeOpKind.FLOORDIV, 9, 4, 2, id="floordiv"),
            pytest.param(RuntimeOpKind.MOD, 9, 4, 1, id="mod"),
            pytest.param(RuntimeOpKind.POW, 2, 5, 32, id="pow"),
            pytest.param(RuntimeOpKind.EQ, 3, 3, 1, id="eq"),
            pytest.param(RuntimeOpKind.NEQ, 3, 4, 1, id="neq"),
            pytest.param(RuntimeOpKind.LT, 3, 4, 1, id="lt"),
            pytest.param(RuntimeOpKind.LE, 4, 4, 1, id="le"),
            pytest.param(RuntimeOpKind.GT, 5, 4, 1, id="gt"),
            pytest.param(RuntimeOpKind.GE, 4, 5, 0, id="ge"),
            pytest.param(RuntimeOpKind.AND, 1, 0, 0, id="and"),
            pytest.param(RuntimeOpKind.OR, 1, 0, 1, id="or"),
        ],
    )
    def test_binary_kind_evaluates(self, kind, lhs, rhs, expected):
        """Each binary ``RuntimeOpKind`` evaluates with the semantics of
        its per-family ``eval_utils`` helper; booleans surface as ints."""
        assert self._execute_single(kind, [lhs, rhs]) == expected

    @pytest.mark.parametrize(
        "operand, expected",
        [pytest.param(1, 0, id="not-1"), pytest.param(0, 1, id="not-0")],
    )
    def test_not_kind_evaluates(self, operand, expected):
        """The unary NOT kind negates its single operand."""
        assert self._execute_single(RuntimeOpKind.NOT, [operand]) == expected

    def test_division_by_zero_raises_execution_error(self):
        """A failing evaluation (division by zero) raises ExecutionError
        instead of silently storing ``None``."""
        from qamomile.circuit.transpiler.errors import ExecutionError

        with pytest.raises(ExecutionError, match="evaluation failed"):
            self._execute_single(RuntimeOpKind.DIV, [1, 0])
