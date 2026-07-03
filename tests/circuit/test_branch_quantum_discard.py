"""Regression tests for branch-internal quantum silent-discard rejection.

The decoration-time rebind analyzer deliberately suppresses violations
recorded inside ``if`` / ``for`` / ``while`` bodies so that compile-time-if
dead-branch rebinds keep decorating. That used to leave a runtime hole:
rebinding a quantum variable inside a runtime branch — to a fresh
allocation or to another quantum value, in one branch or both — silently
dropped the variable's pre-branch state exactly when a rebinding branch
was taken, surfacing (at best) as the unhelpful emit-time "Quantum PhiOp
merge requires identical physical resources across branches" error, or
(for both-branch external rebinds) executing and silently returning the
wrong register's state. The frontend now records every branch-internal
quantum binding change on the ``IfOperation`` (``BranchRebind``), and
``reject_branch_internal_quantum_discard`` rejects a record whose
pre-branch value has no owner on a rebinding path (no in-branch consumer,
no phi carrying it out of that side, no reference outside the if) with a
``QubitRebindError`` (the same ``AffineTypeError`` the decoration-time
analyzer raises for a top-level rebind from a different quantum source),
from both ``PartialEvaluationPass`` (pre-fold, with bindings) and
``AnalyzePass`` (safety net).

Covered here: the LIMITATIONS.md motivating example (adapted to entrypoint
constraints — qubits are allocated in-kernel and the condition is
measurement-backed), the symmetric else-branch case, fresh lineage through
gates, rebinds to external values (one branch, both branches, and gated —
the review counterexamples), expression-derived runtime conditions
(``~bit``, ``a & b``), whole-register ``Vector[Qubit]`` rebinds, compile-time
conditions (dead and taken branches stay legal, including nested inside
runtime branches), the consume-then-reallocate pattern, and the
conservative corners documented in LIMITATIONS.md. Allowed patterns that
are emittable are executed on Qiskit (AerSimulator) with deterministic
state preparation and their measured values asserted — transpile-only
success would not catch a miscompile; allowed patterns stopped by the
emit-level physical-resource guard pin that emit failure so a future
silent-pass regression is caught.
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
    reject_branch_internal_quantum_discard,
)

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

DISCARD = "Branch-internal quantum rebind"


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
        reject_branch_internal_quantum_discard(operations, {"flag": 0})
