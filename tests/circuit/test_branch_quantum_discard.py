"""Regression tests for control-flow-internal quantum silent-discard rejection.

The decoration-time rebind analyzer deliberately suppresses violations
recorded inside ``if`` / ``for`` / ``while`` bodies so that compile-time-if
dead-branch rebinds keep decorating. That used to leave a runtime hole:
rebinding a quantum variable inside a runtime branch — to a fresh
allocation or to another quantum value, in one branch or both — silently
dropped the variable's pre-branch state exactly when a rebinding branch
was taken, surfacing (at best) as the unhelpful emit-time "Quantum PhiOp
merge requires identical physical resources across branches" error, or
(for both-branch external rebinds) executing and silently returning the
wrong register's state. Loop bodies had the same hole with no error at
all: ``for _ in qmc.range(n): q = qmc.qubit("fresh")`` compiled and
executed, silently returning the fresh state instead of the prepared one.
The frontend now records every branch-internal quantum binding change on
the ``IfOperation`` (``BranchRebind``) and every loop-body quantum rebind
on the loop operation (``LoopCarriedRebind`` with a quantum ``before``),
and ``reject_control_flow_quantum_discard`` rejects a record whose
incoming value has no owner on a rebinding path (no in-scope consumer,
no phi carrying it out of that side, no reference outside the construct)
with a ``QubitRebindError`` (the same ``AffineTypeError`` the
decoration-time analyzer raises for a top-level rebind from a different
quantum source), from both ``PartialEvaluationPass`` (pre-fold, with
bindings) and ``AnalyzePass`` (safety net).

Covered here: the LIMITATIONS.md motivating example (adapted to entrypoint
constraints — qubits are allocated in-kernel and the condition is
measurement-backed), the symmetric else-branch case, fresh lineage through
gates, rebinds to external values (one branch, both branches, and gated —
the review counterexamples), expression-derived runtime conditions
(``~bit``, ``a & b``), whole-register ``Vector[Qubit]`` rebinds, compile-time
conditions (dead and taken branches stay legal, including nested inside
runtime branches), the consume-then-reallocate pattern, loop-body discards
(``for`` / ``for``-items / ``while``, at top level and nested in branches),
and the conservative corners documented in LIMITATIONS.md. Allowed
patterns that are emittable are executed on Qiskit (AerSimulator) with
deterministic state preparation and their measured values asserted —
transpile-only success would not catch a miscompile; allowed patterns
stopped by the emit-level physical-resource guard pin that emit failure so
a future silent-pass regression is caught.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import (
    AffineTypeError,
    EmitError,
    QamomileCompileError,
    QubitRebindError,
)
from qamomile.circuit.transpiler.passes.analyze import (
    reject_control_flow_quantum_discard,
)
from qamomile.circuit.transpiler.segments import MultipleQuantumSegmentsError

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

DISCARD = "Branch-internal quantum rebind"
LOOP_DISCARD = "Loop-body quantum rebind"


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
# Rejected: runtime-branch quantum rebinds that discard quantum state
# ---------------------------------------------------------------------------


class TestRejectedDiscards:
    """Runtime-branch quantum rebinds that discard fail with the targeted error."""

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

        with pytest.raises(QubitRebindError, match=DISCARD):
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

        with pytest.raises(QubitRebindError, match="false branch"):
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

        with pytest.raises(QubitRebindError, match=DISCARD):
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

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_rebind_to_external_value_in_one_branch_rejected(self):
        """Rebinding to a pre-existing external qubit discards like fresh.

        The replacement value comes from outside the if rather than an
        in-branch allocation; the pre-branch state of ``q`` is dropped
        all the same when the branch is taken.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            fresh = qmc.qubit("fresh")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = fresh
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_rebind_to_external_value_in_both_branches_rejected(self):
        """Rebinding to the same external qubit in both branches is caught.

        The pre-branch value of ``q`` then appears in no phi at all (the
        frontend even elides the no-op merge), so only the recorded
        pre-branch binding exposes the discard; before the records this
        shape transpiled, sampled, and silently returned the external
        register's state instead of ``q``'s.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            fresh = qmc.qubit("fresh")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = fresh
            else:
                q = fresh
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_gated_external_rebind_in_both_branches_rejected(self):
        """The review counterexample: both branches gate and swap in the
        same external register.

        ``q``'s pre-branch |1> state is unconditionally dropped while both
        phi sides carry the external register's lineage; before the
        records this transpiled and measured the external register (0)
        despite ``q`` being prepared to 1.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            fresh = qmc.qubit("fresh")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                fresh = qmc.x(fresh)
                q = fresh
            else:
                fresh = qmc.z(fresh)
                q = fresh
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_both_branches_fresh_rejected(self):
        """Fresh allocations in BOTH branches drop the original either way.

        The pre-branch value appears in no phi, so only the recorded
        pre-branch binding exposes the discard.
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

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_not_condition_fresh_rejected(self):
        """A ``~bit`` condition is runtime control flow and is checked.

        ``is_measurement_backed`` does not follow classical expressions,
        so condition classification uses measurement-taint analysis;
        expression-derived runtime conditions must not slip to the
        generic emit error.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if ~cond:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_compound_condition_fresh_rejected(self):
        """An ``a & b`` condition is runtime control flow and is checked."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            pa = qmc.qubit("pa")
            pb = qmc.qubit("pb")
            a = qmc.measure(pa)
            b = qmc.measure(pb)
            if a & b:
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_composite_hidden_fresh_rejected(self):
        """Routing the fresh register through a composite gate does not hide it.

        The carried-exemption traces lineage through unknown producers by
        over-approximating with their quantum inputs, so a fresh register
        passed through ``qmc.qft`` before the merge is still provably not
        carrying the pre-branch register.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit_array(2, "fresh")
                q = qmc.qft(q)
            bits = qmc.measure(q)
            return bits[0]

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_sibling_branch_ownership_rejected(self):
        """Ownership on the sibling branch of an enclosing if does not exempt.

        The outer else branch consumes ``q``, but that branch never
        executes together with the inner rebinding branch; the
        path-sensitive outside-ownership evidence excludes it, so the
        inner discard is rejected.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            a = qmc.qubit("a")
            b = qmc.qubit("b")
            outer = qmc.measure(a)
            inner = qmc.measure(b)
            if outer:
                if inner:
                    q = qmc.qubit("fresh")
            else:
                qmc.measure(q)
                q = qmc.qubit("q2")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_dead_after_fresh_rebind_rejected(self):
        """A rebind whose variable is never read after the if still rejects.

        The variable is dead-store-eliminated from the phi merge, so only
        the recorded pre-branch binding (probed from the branch bodies)
        exposes the rebind — matching the decoration-time policy, which
        rejects a top-level rebind regardless of later use.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            r = qmc.qubit("r")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
            r = qmc.x(r)
            return qmc.measure(r)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_dead_after_external_rebind_rejected(self):
        """A dead-after rebind to another register leaves no IR op at all;
        only the record exposes it."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            other = qmc.qubit("other")
            r = qmc.qubit("r")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = other
            r = qmc.x(r)
            qmc.measure(other)
            return qmc.measure(r)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_dead_rebind_in_nested_taken_branch_rejected(self):
        """A dead rebind inside a compile-time-TAKEN nested if is promoted.

        The nested if is not runtime control flow itself, but its taken
        side executes whenever the enclosing runtime branch runs, so its
        record is checked against the enclosing if. The pre-branch handle
        is resolved through the enclosing emit_if captures because the
        variable never enters the generated branch scopes.
        """

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            r = qmc.qubit("r")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                r = qmc.x(r)
                if flag > 0:
                    q = qmc.qubit("fresh")  # noqa: F841 — rebind under test
            return qmc.measure(r)

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"flag": 1})

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

        with pytest.raises(QubitRebindError, match=DISCARD):
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

        with pytest.raises(QubitRebindError, match=DISCARD):
            _transpile(kernel, bindings={"flag": 1})

    def test_discard_error_preempts_emit_error(self):
        """The discard is diagnosed at the analysis stage, not at emit.

        Before this check the same kernel failed only at emit with the
        unhelpful "Quantum PhiOp merge requires identical physical
        resources across branches" ``EmitError``; the targeted
        ``QubitRebindError`` must now fire first. It is an
        ``AffineTypeError`` — the same affine-violation family as the
        decoration-time top-level rebind check — not a generic
        ``ValidationError`` or an ``EmitError``.
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
        assert isinstance(excinfo.value, QubitRebindError)
        assert isinstance(excinfo.value, AffineTypeError)
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
        with pytest.raises(QubitRebindError, match=DISCARD):
            reject_control_flow_quantum_discard(operations, {"dummy": 0})

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
        with pytest.raises(QubitRebindError, match=DISCARD):
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

    def test_preconsumed_original_rebound_in_both_branches_allowed(self):
        """Rebinding a variable whose value was consumed pre-if is legal.

        The pre-branch value is measured before the if, so it is owned
        outside the if and the both-branch rebind to an external register
        is just variable reuse — the analysis stage accepts it. The
        surviving runtime if has empty branches, which hits the
        pre-existing segmentation limitation before emit, so this is
        asserted at the analysis stage only.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            fresh = qmc.qubit("fresh")
            qmc.measure(q)
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = fresh
            else:
                q = fresh
            return qmc.measure(q)

        analyzed = _run_through_analyze(kernel, bindings={"dummy": 0})
        assert analyzed is not None

    def test_handle_swap_in_branch_passes_analysis(self):
        """A pure handle exchange carries both pre-branch values through.

        ``q1, q2 = q2, q1`` rebinds both variables, but each pre-branch
        value is carried out by the other variable's phi on the same
        side, so nothing is discarded and the analysis stage accepts it.
        (The conditional physical relabeling is not emittable — the
        surviving empty-branch runtime if hits the pre-existing
        segmentation limitation — so this is asserted at the analysis
        stage only.)
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q1, q2 = q2, q1
            qmc.measure(q2)
            return qmc.measure(q1)

        analyzed = _run_through_analyze(kernel, bindings={"dummy": 0})
        assert analyzed is not None

    def test_expression_condition_gate_rebind_executes(self):
        """A gate rebind under an expression-derived condition executes.

        ``~cond`` with ``cond`` deterministically 0 takes the branch, so
        the X runs on the original qubit and 1 is measured — the
        measurement-taint condition classification must not disturb the
        supported runtime path.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if ~cond:
                q = qmc.x(q)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"dummy": 0}) == 1

    def test_composite_carry_in_both_branches_allowed(self):
        """Carrying the original register through composite gates is legal.

        The over-approximated lineage sees the pre-branch register flow
        into ``qmc.qft`` / ``qmc.iqft``, so it counts as carried. The
        measured outcome is not deterministic (QFT creates superposition),
        so this asserts compilation rather than a sampled value.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qft(q)
            else:
                q = qmc.iqft(q)
            bits = qmc.measure(q)
            return bits[0]

        executable = _transpile(kernel, bindings={"dummy": 0})
        assert executable is not None

    def test_same_side_preconsume_before_nested_if_passes_analysis(self):
        """Consuming the original on the same path before the nested if is
        ownership on that path; the analysis stage accepts it (the
        cross-branch physical merge still stops at emit, pinned below)."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            a = qmc.qubit("a")
            b = qmc.qubit("b")
            outer = qmc.measure(a)
            inner = qmc.measure(b)
            if outer:
                qmc.measure(q)
                if inner:
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        analyzed = _run_through_analyze(kernel, bindings={"dummy": 0})
        assert analyzed is not None

        with pytest.raises(EmitError):
            _transpile(kernel, bindings={"dummy": 0})

    def test_dead_self_update_allowed_and_executes(self):
        """A dead-after gate self-update keeps the same wire; no record.

        ``q = qmc.x(q)`` changes the SSA version but not the logical wire,
        so no rebind is recorded even though ``q`` is never read after the
        if; the kernel compiles and the unrelated measured qubit behaves.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            r = qmc.qubit("r")
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.x(q)
            r = qmc.x(r)
            return qmc.measure(r)

        assert _sample_single(kernel, bindings={"dummy": 0}) == 1

    def test_dead_rebind_in_nested_dead_branch_allowed_and_executes(self):
        """A dead rebind confined to a compile-time-DEAD nested branch never
        executes on any path and stays legal."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            r = qmc.qubit("r")
            p = qmc.qubit("p")
            p = qmc.x(p)
            cond = qmc.measure(p)
            if cond:
                r = qmc.x(r)
                if flag > 0:
                    q = qmc.qubit("fresh")  # noqa: F841 — rebind under test
            return qmc.measure(r)

        assert _sample_single(kernel, bindings={"flag": 0}) == 1

    def test_dead_rebind_with_alias_owner_allowed_and_executes(self):
        """A dead-after rebind whose pre-branch value an alias still owns is
        legal: the alias's read is outside-ownership on every path."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            alias = q
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                q = qmc.qubit("fresh")
            return qmc.measure(alias)

        assert _sample_single(kernel, bindings={"dummy": 0}) == 0

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
        reject_control_flow_quantum_discard(operations, {"flag": 0})


# ---------------------------------------------------------------------------
# Rejected: loop-body quantum discards (for / for-items / while)
# ---------------------------------------------------------------------------


class TestRejectedLoopDiscards:
    """Loop-body quantum rebinds that discard incoming state are rejected.

    Unlike the if case, a loop-body discard needs no runtime/compile-time
    condition classification: whenever the loop runs, every iteration
    rebinds the variable to a different quantum value without consuming
    the incoming state (the pre-loop value on the first iteration, the
    previous iteration's value afterwards). The check is trip-count-
    agnostic, matching the classical loop-carried rejection.
    """

    def test_for_fresh_rebind_rejected(self):
        """The review repro: a for body rebinding to a fresh allocation.

        Before the loop-side check, this transpiled and sampled 0 — the
        X-prepared state was silently dropped and the fresh register
        returned.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            for _ in qmc.range(n):
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QamomileCompileError) as excinfo:
            _transpile(kernel, bindings={"n": 1})
        assert isinstance(excinfo.value, QubitRebindError)
        assert isinstance(excinfo.value, AffineTypeError)
        assert not isinstance(excinfo.value, EmitError)
        assert LOOP_DISCARD in str(excinfo.value)

    def test_while_fresh_rebind_rejected(self):
        """The review repro's while variant: a measurement-conditioned while
        body rebinding to a fresh allocation is rejected."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            c = qmc.qubit("c")
            b = qmc.measure(c)
            while b:
                q = qmc.qubit("fresh")
                c2 = qmc.qubit("c2")
                b = qmc.measure(c2)
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"dummy": 0})

    def test_while_rebind_read_after_loop_rejected_despite_consumption(self):
        """A while-body rebind read after the loop is rejected even when the
        body consumes the incoming state: the zero-trip path (a while trip
        count is a runtime measurement outcome) must observe the pre-loop
        state, but the emitted circuit binds the post-loop read to the
        body's register unconditionally. Before this rule, this kernel
        transpiled and sampled 0 where Python semantics give 1.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            c = qmc.qubit("c")
            b = qmc.measure(c)
            while b:
                b = qmc.measure(q)
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match="read after the loop"):
            _transpile(kernel, bindings={"dummy": 0})

    def test_for_items_fresh_rebind_rejected(self):
        """A qmc.items body rebinding an outer qubit is rejected."""

        @qmc.qkernel
        def kernel(
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            for _key, _angle in qmc.items(angles):
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"angles": {0: 0.5}})

    def test_for_external_rebind_rejected(self):
        """Rebinding to another existing register in a for body discards the
        incoming state the same way a fresh allocation does."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            other = qmc.qubit("other")
            for _ in qmc.range(n):
                q = other
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"n": 1})

    def test_for_dead_after_rebind_rejected(self):
        """A loop-body rebind is a discard even when the variable is dead
        after the loop: the incoming state is dropped by every iteration."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            r = qmc.qubit("r")
            for _ in qmc.range(n):
                q = qmc.qubit("fresh")  # noqa: F841 — rebind under test
            return qmc.measure(r)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"n": 1})

    def test_vector_rebind_rejected(self):
        """Rebinding a whole Vector[Qubit] register in a for body is
        rejected like the scalar form.

        No element of ``qs`` is touched before the loop: an element
        read/write would count as consumption of the register under the
        documented element-granularity conservatism (LIMITATIONS.md) and
        exempt the rebind, exactly as in the if-branch vector case.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, name="qs")
            for _ in qmc.range(n):
                qs = qmc.qubit_array(2, name="fresh")
            return qmc.measure(qs)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"n": 1})

    def test_rebind_then_gate_still_rejected(self):
        """Gating the fresh value after the rebind does not consume the
        incoming state: the body reads only the post-rebind register."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            for _ in qmc.range(n):
                q = qmc.qubit("fresh")
                q = qmc.h(q)
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"n": 1})

    def test_loop_inside_runtime_branch_rejected(self):
        """A discarding loop nested in a runtime if is rejected (the branch
        record and the loop record both witness it; either error is the
        same QubitRebindError family)."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            p = qmc.qubit("p")
            cond = qmc.measure(p)
            if cond:
                for _ in qmc.range(n):
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match="quantum rebind"):
            _transpile(kernel, bindings={"n": 2})

    def test_loop_inside_compile_time_taken_branch_rejected(self):
        """A discarding loop inside a compile-time-TAKEN branch is rejected:
        the loop probe resolves the pre-branch handle through the if-branch
        pre-binding stack, so the record survives even though the branch
        function never binds ``q`` itself."""

        @qmc.qkernel
        def kernel(n: qmc.UInt, flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            if flag > 0:
                for _ in qmc.range(n):
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            _transpile(kernel, bindings={"n": 2, "flag": 1})

    def test_module_level_check_rejects_loop_discard(self):
        """The module-level helper rejects the loop pattern on an inlined
        block."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            for _ in qmc.range(n):
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        operations = _inlined_block(kernel, bindings={"n": 1}).operations
        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            reject_control_flow_quantum_discard(operations, {"n": 1})

    def test_analyze_pass_safety_net_rejects_loop_discard(self):
        """AnalyzePass rejects the loop pattern even when partial_eval is
        skipped: loop records survive on the loop operation (unlike
        promoted if records, which the lowering erases)."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            for _ in qmc.range(n):
                q = qmc.qubit("fresh")
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        block = _inlined_block(kernel, bindings={"n": 1})
        with pytest.raises(QubitRebindError, match=LOOP_DISCARD):
            transpiler.analyze(block)


# ---------------------------------------------------------------------------
# Allowed: loop patterns that consume, carry, or never run the rebind
# ---------------------------------------------------------------------------


class TestAllowedLoopPatterns:
    """Legal loop patterns keep compiling and executing correctly.

    Wherever the pattern is emittable, the test executes the circuit and
    asserts the deterministic measured value — transpile success alone
    would not catch a miscompile.
    """

    def test_measure_then_fresh_reset_executes(self):
        """Consuming the incoming state before rebinding is the legal
        reset idiom: the body measures the X-prepared register (reads it),
        then rebinds to a fresh one. The measured value is the prepared 1."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            b = qmc.bit(0)
            for _ in qmc.range(n):
                b = qmc.measure(q)
                q = qmc.qubit("fresh")
            return b

        assert _sample_single(kernel, bindings={"n": 1}) == 1

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            pytest.param(1, 1, id="one-flip"),
            pytest.param(2, 0, id="two-flips-cancel"),
        ],
    )
    def test_gate_chain_rebind_executes(self, n, expected):
        """Gate self-updates keep the wire identity: no record, no
        rejection, and the unrolled circuit applies the gate per
        iteration."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            for _ in qmc.range(n):
                q = qmc.x(q)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": n}) == expected

    def test_dead_branch_wrapped_loop_allowed_and_executes(self):
        """A discarding loop inside a compile-time-DEAD branch is pruned
        with the branch and stays legal — the branch-selection idiom."""

        @qmc.qkernel
        def kernel(n: qmc.UInt, flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            if flag > 0:
                for _ in qmc.range(n):
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": 2, "flag": 0}) == 1

    def test_compile_time_dead_if_inside_loop_accepted_at_analysis(self):
        """A compile-time-dead if inside the loop body passes the variable
        through its collapsed phi — that pass-through read is consumption
        evidence, so the loop record is exempt at the analysis stage."""

        @qmc.qkernel
        def kernel(n: qmc.UInt, flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            for _ in qmc.range(n):
                if flag > 0:
                    q = qmc.qubit("fresh")
            return qmc.measure(q)

        analyzed = _run_through_analyze(kernel, bindings={"n": 2, "flag": 0})
        assert analyzed is not None
        # Downstream, plan still splits this shape into multiple quantum
        # segments (pre-existing on the base branch, verified via git
        # stash); pin that so a future silent-pass regression is caught.
        with pytest.raises(MultipleQuantumSegmentsError):
            _transpile(kernel, bindings={"n": 2, "flag": 0})

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            pytest.param(1, 1, id="pi-rotation"),
            pytest.param(2, 0, id="two-pi-identity"),
        ],
    )
    def test_store_only_classical_reassignment_still_allowed(self, n, expected):
        """Store-only classical reassignments in a loop body stay legal:
        the widened probe candidates record quantum rebinds only, so a
        per-iteration recomputed angle produces no classical record. The
        rx(pi) per iteration flips the qubit deterministically."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            angle = 0.0
            for _ in qmc.range(n):
                angle = 3.141592653589793
                q = qmc.rx(q, angle)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": n}) == expected

    def test_while_rebind_unused_after_loop_allowed_and_executes(self):
        """The documented repeat-until-success idiom stays legal: the body
        rebinds to a fresh register, but nothing reads the register after
        the loop (only the measured condition escapes), and the pre-loop
        value is owned by the pre-loop measurement. The loop exits only
        when the measurement is 0, so the returned bit is deterministic.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        assert _sample_single(kernel, bindings={"dummy": 0}) == 0

    def test_alias_owned_loop_rebind_allowed(self):
        """A loop rebind whose pre-loop value an alias still owns passes the
        discard check: the alias's post-loop read is outside-ownership."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            saved = q
            for _ in qmc.range(n):
                q = qmc.qubit("fresh")  # noqa: F841 — rebind under test
            return qmc.measure(saved)

        assert _sample_single(kernel, bindings={"n": 1}) == 1

    def test_module_level_check_accepts_consumed_loop_rebind(self):
        """The module-level helper accepts the consume-then-reallocate loop
        body without raising."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            b = qmc.bit(0)
            for _ in qmc.range(n):
                b = qmc.measure(q)
                q = qmc.qubit("fresh")
            return b

        operations = _inlined_block(kernel, bindings={"n": 1}).operations
        reject_control_flow_quantum_discard(operations, {"n": 1})
