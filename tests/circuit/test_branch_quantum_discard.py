"""Regression tests for branch-internal quantum silent-discard rejection.

The decoration-time rebind analyzer deliberately suppresses violations
recorded inside ``if`` / ``for`` / ``while`` bodies so that compile-time-if
dead-branch rebinds keep decorating. That used to leave a runtime hole:
``if cond: q = qmc.qubit("fresh")`` with a measurement-backed ``cond``
silently dropped the original state of ``q`` exactly when the branch was
taken, surfacing (at best) as the unhelpful emit-time "Quantum PhiOp merge
requires identical physical resources across branches" error.
``reject_branch_internal_quantum_discard`` now rejects the pattern at the
IR layer with a targeted ``ValidationError``, from both
``PartialEvaluationPass`` (pre-fold, with bindings) and ``AnalyzePass``
(safety net).

Covered here: the LIMITATIONS.md motivating example (adapted to entrypoint
constraints — qubits are allocated in-kernel and the condition is
measurement-backed), the symmetric else-branch case, fresh lineage through
gates, whole-register ``Vector[Qubit]`` rebinds (same phi shape as scalar
``Qubit``, covered by the same check), compile-time conditions (dead and
taken branches stay legal, including nested inside runtime branches), the
consume-then-reallocate pattern, and the conservative corners documented in
LIMITATIONS.md. Allowed patterns that are emittable are executed on Qiskit
(AerSimulator) with deterministic state preparation and their measured
values asserted — transpile-only success would not catch a miscompile;
allowed patterns stopped by the emit-level physical-resource guard pin that
emit failure so a future silent-pass regression is caught.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import (
    EmitError,
    QamomileCompileError,
    ValidationError,
)
from qamomile.circuit.transpiler.passes.analyze import (
    reject_branch_internal_quantum_discard,
)

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

DISCARD = "Branch-internal fresh allocation"


def _transpile(kernel, bindings=None):
    """Run the full Qiskit pipeline and return the executable.

    Args:
        kernel (QKernel): The qkernel to compile.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to None
            (empty bindings).

    Returns:
        ExecutableProgram: The compiled executable.
    """
    transpiler = QiskitTranspiler()
    return transpiler.transpile(kernel, bindings=bindings or {})


def _sample_single(kernel, bindings=None, shots=300):
    """Sample a kernel on Qiskit and return the single deterministic outcome.

    Allowed-pattern tests execute the compiled circuit instead of stopping
    at transpile success: a kernel that transpiles but miscompiles would
    slip through a transpile-only assertion.

    Args:
        kernel (QKernel): The qkernel to compile and run.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to
            None (empty bindings).
        shots (int): Number of shots. Defaults to 300.

    Returns:
        int: The unique value measured across all shots.
    """
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings=bindings or {})
    result = executable.sample(transpiler.executor(), shots=shots).result()
    assert len(result.results) == 1, f"expected deterministic outcome: {result}"
    return result.results[0][0]


def _run_through_analyze(kernel, bindings=None):
    """Run the pipeline through the analyze pass (both check hooks) only.

    Stops before plan/emit so patterns that pass the discard check but
    still hit emit-level physical-resource constraints can be asserted
    as accepted by the analysis stage.

    Args:
        kernel (QKernel): The qkernel to compile.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to None
            (empty bindings).

    Returns:
        Block: The ANALYZED block.
    """
    bindings = bindings or {}
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(kernel, bindings=bindings)
    block = transpiler.resolve_parameter_shapes(block, bindings)
    block = transpiler.inline(block)
    block = transpiler.affine_validate(block)
    block = transpiler.partial_eval(block, bindings)
    return transpiler.analyze(block)


def _inlined_block(kernel, bindings=None):
    """Build the kernel to an inlined (AFFINE) block.

    Args:
        kernel (QKernel): The qkernel to build.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to None
            (empty bindings).

    Returns:
        Block: The inlined AFFINE block, before partial evaluation.
    """
    bindings = bindings or {}
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(kernel, bindings=bindings)
    block = transpiler.resolve_parameter_shapes(block, bindings)
    return transpiler.inline(block)


# ---------------------------------------------------------------------------
# Rejected: runtime-branch fresh allocations that discard quantum state
# ---------------------------------------------------------------------------


class TestRejectedDiscards:
    """Runtime-branch fresh allocations fail with the targeted error."""

    def test_limitations_example_rejected(self):
        """The LIMITATIONS.md motivating example is rejected at transpile.

        Adapted to entrypoint constraints: qubits are allocated inside the
        kernel and the branch condition is a measurement-backed Bit.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_else_branch_fresh_rejected(self):
        """A fresh allocation in the else branch is rejected symmetrically."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            r = qmc.qubit("r")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                r = qmc.h(r)
            else:
                q = qmc.qubit("fresh")
            qmc.measure(r)
            return qmc.measure(q)

        with pytest.raises(ValidationError, match="false branch"):
            _transpile(kernel, bindings={"dummy": 0})

    def test_fresh_lineage_through_gates_rejected(self):
        """Gating the fresh qubit before the merge does not hide the discard."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
                q = qmc.h(q)
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_gate_rebind_in_other_branch_rejected(self):
        """Fresh in one branch discards even when the other branch gates ``q``.

        On the fresh path the other branch's gate never executes, so the
        pre-branch state is still silently dropped; the check traces the
        non-allocating side back to the pre-branch root through its gates.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
            else:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_vector_qubit_fresh_rejected(self):
        """Whole-register ``Vector[Qubit]`` rebinds merge through a single phi
        and are rejected exactly like scalar ``Qubit`` rebinds."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit_array(2, "fresh")
            bits = qmc.measure(q)
            return bits[0]

        with pytest.raises(ValidationError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_compile_time_taken_branch_nested_in_runtime_rejected(self):
        """A compile-time TAKEN fresh allocation nested inside a runtime
        branch is a genuine conditional discard and is rejected."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                if flag > 0:
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=DISCARD):
            _transpile(kernel, bindings={"flag": 1})

    def test_discard_error_preempts_emit_error(self):
        """The discard is diagnosed at the analysis stage, not at emit.

        Before this check the same kernel failed only at emit with the
        unhelpful "Quantum PhiOp merge requires identical physical
        resources across branches" ``EmitError``; the targeted
        ``ValidationError`` must now fire first.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QamomileCompileError) as excinfo:
            _transpile(kernel, bindings={"dummy": 0})
        assert isinstance(excinfo.value, ValidationError)
        assert not isinstance(excinfo.value, EmitError)
        assert DISCARD in str(excinfo.value)

    def test_module_level_check_rejects_inlined_block(self):
        """The module-level helper rejects the pattern on an inlined block."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        operations = _inlined_block(kernel, bindings={"dummy": 0}).operations
        with pytest.raises(ValidationError, match=DISCARD):
            reject_branch_internal_quantum_discard(operations, {"dummy": 0})

    def test_analyze_pass_safety_net_rejects_without_partial_eval(self):
        """AnalyzePass rejects the pattern even when partial_eval is skipped.

        The analyze hook runs without bindings; the measurement-backed
        condition keeps the if classified as runtime, so the safety net
        fires for pipelines that never ran ``PartialEvaluationPass``.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        block = _inlined_block(kernel, bindings={"dummy": 0})
        with pytest.raises(ValidationError, match=DISCARD):
            transpiler.analyze(block)


# ---------------------------------------------------------------------------
# Allowed: compile-time branches, in-branch consumption, gate rebinds
# ---------------------------------------------------------------------------


class TestAllowedPatterns:
    """Legal branch patterns keep compiling and executing correctly.

    Wherever the pattern is emittable, the test executes the circuit and
    asserts the deterministic measured value — transpile success alone
    would not catch a miscompile. Patterns that pass this check but are
    stopped by the emit-level cross-branch physical-resource guard are
    asserted at the analysis stage and their emit failure is pinned.
    """

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            pytest.param(0, 1, id="dead-branch-keeps-original"),
            pytest.param(1, 0, id="taken-branch-selects-fresh"),
        ],
    )
    def test_compile_time_branch_selection_executes(self, flag, expected):
        """Compile-time fresh rebinds stay legal and execute correctly.

        The original qubit is prepared to |1> so the measured value
        distinguishes the two lowerings: flag=0 eliminates the dead branch
        and keeps the original (measures 1); flag=1 selects the fresh |0>
        register — the documented branch-selection idiom, where discarding
        the original is unconditional and intended (measures 0).
        """

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            if flag > 0:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"flag": flag}) == expected

    def test_compile_time_dead_branch_nested_in_runtime_allowed(self):
        """A compile-time-DEAD fresh allocation nested inside a runtime
        branch is eliminated by if-lowering and must not be flagged.

        The end-to-end variant keeps an X gate in the runtime branch so the
        branch stays non-empty after the nested dead if is eliminated, and
        prepares the condition qubit to |1> so the branch is deterministically
        taken: measuring 1 proves the X executed on the original qubit and
        the dead fresh allocation was really eliminated. A runtime if whose
        branches both become empty hits a pre-existing segmentation
        limitation unrelated to this check, so that shape is asserted at the
        analysis stage only.
        """

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            p = qmc.x(p)
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
                if flag > 0:
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"flag": 0}) == 1

        @qmc.qkernel
        def kernel_empty_branch(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                if flag > 0:
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        analyzed = _run_through_analyze(kernel_empty_branch, bindings={"flag": 0})
        assert analyzed is not None

    def test_gate_rebind_runtime_branch_taken_executes(self):
        """The ordinary runtime branch pattern ``q = qmc.x(q)`` executes.

        The condition qubit is prepared to |1> so the branch is
        deterministically taken: measuring 1 proves the in-branch gate ran
        on the original qubit.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            p = qmc.x(p)
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"dummy": 0}) == 1

    def test_gate_rebind_runtime_branch_not_taken_executes(self):
        """The runtime branch is skipped when the condition measures 0.

        The condition qubit stays |0> so the branch is deterministically
        not taken: measuring 0 proves the in-branch X did not leak onto
        the pass-through path.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"dummy": 0}) == 0

    def test_gate_rebind_in_both_branches_executes(self):
        """Gating the same qubit in both branches selects the right branch.

        With the condition prepared to |1> the X branch runs (measures 1);
        with the condition at |0> the Z branch runs, leaving |0> unchanged
        (measures 0 — an incorrectly taken X branch would measure 1).
        """

        @qmc.qkernel
        def kernel_cond1(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            p = qmc.x(p)
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
            else:
                q = qmc.z(q)
            return qmc.measure(q)

        assert _sample_single(kernel_cond1, bindings={"dummy": 0}) == 1

        @qmc.qkernel
        def kernel_cond0(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
            else:
                q = qmc.z(q)
            return qmc.measure(q)

        assert _sample_single(kernel_cond0, bindings={"dummy": 0}) == 0

    def test_measure_then_fresh_passes_analysis(self):
        """Consuming the original before re-allocating is not a discard.

        The pattern passes the analysis stage (both check hooks). It still
        fails at emit on Qiskit because a quantum phi merge must reference
        identical physical qubits across branches — the discard check
        diagnoses the discard shape early, it does not make cross-branch
        physical merges compile (see LIMITATIONS.md).
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                qmc.measure(q)
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        analyzed = _run_through_analyze(kernel, bindings={"dummy": 0})
        assert analyzed is not None

        with pytest.raises(EmitError, match="identical physical resources"):
            _transpile(kernel, bindings={"dummy": 0})

    def test_element_read_counts_as_consumption(self):
        """One element read of the original register disqualifies the error.

        Conservative corner (documented in LIMITATIONS.md): any in-branch
        reference to the original — including a single element measure —
        counts as consumption, so the whole-register rebind passes the
        analysis stage. The cross-branch physical merge is still stopped at
        emit (pinned below), so no silent path exists today.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                qmc.measure(q[0])
                q = qmc.qubit_array(2, "fresh")
            bits = qmc.measure(q)
            return bits[0]

        analyzed = _run_through_analyze(kernel, bindings={"dummy": 0})
        assert analyzed is not None

        with pytest.raises(EmitError):
            _transpile(kernel, bindings={"dummy": 0})

    def test_both_branches_fresh_not_flagged_by_this_check(self):
        """Both branches allocating fresh is outside this check's evidence.

        The pre-branch value then no longer appears in the phi merge, so
        the discard check has nothing to pair (documented conservative
        corner in LIMITATIONS.md); the kernel passes the analysis stage.
        The cross-branch physical merge is still stopped at emit (pinned
        below), so no silent path exists today.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh_a")
            else:
                q = qmc.qubit("fresh_b")
            return qmc.measure(q)

        analyzed = _run_through_analyze(kernel, bindings={"dummy": 0})
        assert analyzed is not None

        with pytest.raises(EmitError, match="identical physical resources"):
            _transpile(kernel, bindings={"dummy": 0})

    def test_module_level_check_accepts_dead_branch_with_bindings(self):
        """The module-level helper resolves compile-time conditions from
        bindings and accepts the dead-branch fresh allocation."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag > 0:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        operations = _inlined_block(kernel, bindings={"flag": 0}).operations
        reject_branch_internal_quantum_discard(operations, {"flag": 0})
