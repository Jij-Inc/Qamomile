"""Regression tests for loop-carried classical scalar rebind rejection.

A qkernel loop body is traced exactly once, so a Python-level reassignment
like ``total = total + i`` inside ``qmc.range`` / ``while`` / ``qmc.items``
produces IR whose right-hand side reads the fixed pre-loop value instead of
the previous iteration's value. Every executor (the classical segment
interpreter and emit-time unrolling) re-runs the same traced operations per
iteration, so the compiled program silently diverged from Python semantics
(e.g. ``sum(range(4))`` compiled to ``0 + 3 == 3`` instead of ``6``). The
frontend now records such rebinds on the loop operations and the transpiler
rejects them with a targeted ``ValidationError`` instead of miscompiling.

Legal patterns — read-only outer values, quantum rebinds, per-iteration
re-measurement of a Bit (the loop-carried while-condition machinery), and
rebinds confined to compile-time-dead branches — must keep compiling.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    WhileOperation,
)
from qamomile.circuit.transpiler.errors import ValidationError

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

LOOP_CARRIED = "Loop-carried"


def _transpile(kernel, bindings=None, parameters=None):
    """Transpile a kernel on Qiskit and return the executable."""
    transpiler = QiskitTranspiler()
    return transpiler.transpile(kernel, bindings=bindings or {}, parameters=parameters)


def _sample_single(kernel, bindings=None, shots=200):
    """Sample a kernel on Qiskit and return the single deterministic outcome."""
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings=bindings or {})
    result = executable.sample(transpiler.executor(), shots=shots).result()
    assert len(result.results) == 1, f"expected deterministic result: {result}"
    return result.results[0][0]


def _find_loops(ops):
    """Collect every loop operation reachable from ``ops``."""
    loops = []
    for op in ops:
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            loops.append(op)
        nested = getattr(op, "nested_op_lists", None)
        if nested is not None:
            for body in nested():
                loops.extend(_find_loops(body))
    return loops


# ---------------------------------------------------------------------------
# Rejected: loop-carried classical scalar rebinds
# ---------------------------------------------------------------------------


class TestRejectedRebinds:
    """Loop-carried classical rebinds fail with the targeted error."""

    def test_for_accumulation_rejected(self):
        """`total = total + i` in qmc.range is rejected, not miscompiled."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4})

    def test_for_accumulation_symbolic_bound_rejected(self):
        """The rejection also fires when the loop bound stays a parameter."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, parameters=["n"], bindings={})

    def test_builtin_range_accumulation_rejected(self):
        """builtin range() is transformed like qmc.range and is equally rejected."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in range(4):
                total = total + i
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"dummy": 0})

    def test_for_if_accumulation_rejected(self):
        """Accumulation guarded by a runtime if (merge-mediated) is rejected."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                if i > 1:
                    total = total + i
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4})

    def test_all_constant_accumulation_rejected(self):
        """`total = total + 1` folds at trace time; the pre-fold hook catches it."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for _i in qmc.range(n):
                total = total + 1
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 3})

    def test_plain_python_int_init_rejected(self):
        """A plain `total = 0` initialization (synthesized before) is rejected."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = 0
            for i in qmc.range(n):
                total = total + i
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4})

    def test_float_angle_accumulation_rejected(self):
        """A non-escaping Float accumulation driving a gate angle is rejected."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = qmc.float_(0.0)
            for _i in qmc.range(n):
                total = total + 1.0
            q = qmc.qubit("q")
            q = qmc.rx(q, total * 3.141592653589793)
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 2})

    def test_nested_loop_accumulation_rejected(self):
        """Accumulating an outer variable inside nested loops is rejected."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                for j in qmc.range(n):
                    if i == j:
                        total = total + 1
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 3})

    def test_while_counter_rejected(self):
        """A classical counter inside a measurement-conditioned while is rejected."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            count = qmc.uint(0)
            while bit:
                q2 = qmc.qubit("q2")
                q2 = qmc.h(q2)
                bit = qmc.measure(q2)
                count = count + 1
            return count

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"dummy": 0})

    def test_accumulation_as_range_bound_rejected(self):
        """Accumulation feeding a later qmc.range bound gets the targeted error.

        Previously this failed at emit with the confusing "Cannot unroll
        loop: bounds could not be resolved (stop='_merge_0')".
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = qmc.uint(0)
            for i in qmc.range(n):
                if i > 1:
                    total = total + i
            q = qmc.qubit("q")
            for _j in qmc.range(total):
                q = qmc.x(q)
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4})

    def test_subkernel_loop_accumulation_rejected(self):
        """Accumulation inside a called sub-kernel is detected after inlining.

        The rebind records ride the sub-kernel's ForOperation and are
        remapped by inline cloning (all_input_values / replace_values),
        so the transpile-time check still sees consistent UUIDs.
        """

        @qmc.qkernel
        def helper(n: qmc.UInt) -> qmc.UInt:
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            return helper(n)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4})

    def test_augassign_accumulation_rejected(self):
        """`total += i` (AugAssign) is detected like the explicit form."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total += i
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4})

    def test_while_condition_snapshot_saved_in_body_rejected(self):
        """Saving the while condition's entry value in the body is rejected.

        The allocator aliases the whole condition series onto one clbit,
        so the post-loop read of the saved snapshot would observe the
        final in-loop measurement (measured divergence: Python semantics
        1, emitted circuit 0).
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            out = bit
            while bit:
                out = bit
                q2 = qmc.qubit("f")
                bit = qmc.measure(q2)
            return out

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"dummy": 0})

    def test_while_condition_snapshot_saved_pre_loop_rejected(self):
        """A pre-loop snapshot of the while condition is rejected too.

        This variant carries no rebind record for the snapshot variable
        (it is never stored in the body), so the rejection comes from the
        stale-condition-read scan over the block outputs.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            out = bit
            while bit:
                q2 = qmc.qubit("f")
                bit = qmc.measure(q2)
            return out

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"dummy": 0})

    def test_while_condition_snapshot_through_pre_loop_merge_rejected(self):
        """A condition snapshot aliased through a pre-loop compile-time if
        and returned after the loop is rejected.

        The compile-time merge output selects the initial condition, so
        it shares the condition's classical bit; returning it after the
        loop observes the final in-loop measurement instead of the
        snapshot Python promises. The stale-condition scan follows the
        pruned merge's alias to catch the read.
        """

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            if flag == 1:
                saved = bit
            else:
                saved = qmc.bit(False)
            while bit:
                q2 = qmc.qubit("f")
                bit = qmc.measure(q2)
            return saved

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"flag": 1})

    def test_live_rebind_in_loop_inside_runtime_if_rejected(self):
        """A live accumulation in a loop under a runtime if stays rejected."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt, n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            total = qmc.uint(0)
            if bit:
                for i in qmc.range(n):
                    if flag == 1:
                        total = total + i
            return bit

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"flag": 1, "n": 3})

    def test_swap_rotation_rejected(self):
        """`a, b = b, a` in a loop leaves stale one-shot swaps and is rejected."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            a = qmc.uint(1)
            b = qmc.uint(2)
            for _i in qmc.range(n):
                a, b = b, a
            return a

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 3})


# ---------------------------------------------------------------------------
# Allowed: legal loop patterns keep compiling (and computing correctly)
# ---------------------------------------------------------------------------


class TestAllowedPatterns:
    """Legal loop patterns are not rejected and still execute correctly."""

    def test_loop_invariant_rebind_allowed(self):
        """`last = x + i` re-executes to the correct final value (8)."""

        @qmc.qkernel
        def kernel(n: qmc.UInt, x: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            last = qmc.uint(0)
            for i in qmc.range(n):
                last = x + i
            return last

        assert _sample_single(kernel, bindings={"n": 4, "x": 5}) == 8

    def test_quantum_rebind_allowed(self):
        """`q = qmc.h(q)` inside a loop is the fundamental legal idiom."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            for _i in qmc.range(n):
                q = qmc.x(q)
            return qmc.measure(q)

        # n=3: odd number of X gates flips |0> to |1>.
        assert _sample_single(kernel, bindings={"n": 3}) == 1

    def test_dead_branch_rebind_in_loop_inside_runtime_if_allowed(self):
        """A dead-branch-only rebind in a loop under a runtime if compiles.

        The pruning walk descends into runtime-if branches with the same
        per-branch state scoping as the lowering pass, so the
        compile-time-dead `total = total + i` canonicalizes away and the
        loop carries no live rebind.
        """

        @qmc.qkernel
        def kernel(flag: qmc.UInt, n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            total = qmc.uint(0)
            if bit:
                for i in qmc.range(n):
                    if flag == 1:
                        total = total + i
            return bit

        _transpile(kernel, bindings={"flag": 0, "n": 3})

    def test_unobserved_post_loop_merge_of_condition_snapshot_allowed(self):
        """A dead post-loop compile-time merge of the condition compiles.

        The pruned merge selects the pre-loop condition value, but its
        output is never read and does not escape through the outputs, so
        no operation can observe the shared clbit's stale value — there
        is no divergence to reject. Only the updated condition variable
        is returned (the documented legal pattern).
        """

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            snapshot = bit
            while bit:
                q2 = qmc.qubit("f")
                bit = qmc.measure(q2)
            if flag == 1:
                unused = snapshot  # noqa: F841 - dead merge is the point
            else:
                unused = qmc.bit(False)  # noqa: F841
            return bit

        _transpile(kernel, bindings={"flag": 1})

    def test_remeasured_bit_allowed(self):
        """Per-iteration `bit = qmc.measure(...)` (no stale read) compiles.

        Only transpilation is asserted: the executed value is currently
        wrong for an unrelated pre-existing reason (a fresh
        ``qmc.qubit(...)`` allocated inside a runtime loop is not reset
        per iteration, so gates accumulate on the same physical qubit).
        That gap is outside the loop-carried rebind check's scope.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            bit = qmc.bit(False)
            for _i in qmc.range(n):
                q = qmc.qubit("q")
                q = qmc.x(q)
                bit = qmc.measure(q)
            return bit

        _transpile(kernel, bindings={"n": 2})

    def test_for_loop_measurement_backed_bit_rejected(self):
        """A measured Bit read and re-measured inside a for loop is rejected.

        Only ``WhileOperation.operands[1]`` gets loop-carried clbit
        aliasing from the resource allocator; a for-loop has no such
        machinery, so the ``if state:`` condition keeps addressing the
        pre-loop clbit while the branch measurements write elsewhere
        (measured divergence: with fresh-per-iteration Python semantics
        ``n=2`` yields ``out == 1``, but the emitted ForLoopOp circuit
        returns ``0``). Rejecting is the honest behavior until real
        aliasing lands.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            init_q = qmc.qubit("init")
            state = qmc.measure(init_q)
            out = state
            for _i in qmc.range(n):
                out = state
                if state:
                    q_zero = qmc.qubit("zero")
                    state = qmc.measure(q_zero)
                else:
                    q_one = qmc.qubit("one")
                    q_one = qmc.x(q_one)
                    state = qmc.measure(q_one)
            return out

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 2})

    def test_repeat_until_zero_while_allowed(self):
        """The documented repeat-until-zero while pattern keeps compiling."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q2 = qmc.qubit("q2")
                q2 = qmc.h(q2)
                bit = qmc.measure(q2)
            return bit

        assert _sample_single(kernel, bindings={"dummy": 0}) == 0

    def test_dead_branch_accumulation_allowed_taken_rejected(self):
        """A rebind confined to a compile-time-dead branch is allowed.

        The same kernel with the branch taken is rejected.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt, flag: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                if flag > 0:
                    total = total + i
            return total

        _transpile(kernel, bindings={"n": 4, "flag": 0})

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 4, "flag": 1})

    def test_iteration_local_fresh_name_allowed(self):
        """A fresh per-iteration local feeding a gate angle compiles."""

        @qmc.qkernel
        def kernel(n: qmc.UInt, x: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            for i in qmc.range(n):
                angle = x * i
                q = qmc.rz(q, angle)
            return qmc.measure(q)

        _transpile(kernel, bindings={"n": 3, "x": 0.5})

    def test_items_loop_readonly_scalar_allowed(self):
        """qmc.items loops reading an outer scalar (gamma * Jij) compile."""

        @qmc.qkernel
        def kernel(
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for (i, j), coeff in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * coeff)
            return qmc.measure(q)

        _transpile(
            kernel,
            bindings={"ising": {(0, 1): 1.0}, "gamma": 0.3},
        )

    def test_build_does_not_raise(self):
        """kernel.build() only records; rejection happens at transpile."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        block = kernel.build(parameters=["n"])
        loops = _find_loops(block.operations)
        assert loops, "expected a ForOperation in the built block"
        assert any(loop.loop_carried_rebinds for loop in loops)

    def test_undecorated_helper_accumulation_allowed(self):
        """Accumulation inside an undecorated Python helper runs natively."""

        def native_total(n: int) -> int:
            total = 0
            for i in range(n):
                total = total + i
            return total

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            for _i in qmc.range(native_total(3)):
                q = qmc.x(q)
            return qmc.measure(q)

        # native_total(3) == 3: odd number of X gates flips to |1>.
        assert _sample_single(kernel, bindings={"dummy": 0}) == 1


# ---------------------------------------------------------------------------
# IR plumbing: records survive serialization
# ---------------------------------------------------------------------------


class TestRebindRecordSerialization:
    """LoopCarriedRebind records round-trip through both wire formats."""

    def _build_block_with_records(self):
        """Build a block whose ForOperation carries rebind records."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        return kernel.build(parameters=["n"])

    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_records_roundtrip(self, fmt):
        """Records survive a serialize/deserialize cycle."""
        from qamomile.circuit.ir.serialize import (
            dump_json,
            dump_msgpack,
            load_json,
            load_msgpack,
        )

        block = self._build_block_with_records()
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(block)

        if fmt == "json":
            restored = load_json(dump_json(affine))
        else:
            restored = load_msgpack(dump_msgpack(affine))

        original_loops = _find_loops(affine.operations)
        restored_loops = _find_loops(restored.operations)
        assert len(original_loops) == len(restored_loops)
        for orig, rest in zip(original_loops, restored_loops, strict=True):
            assert len(orig.loop_carried_rebinds) == len(rest.loop_carried_rebinds)
            for orig_rec, rest_rec in zip(
                orig.loop_carried_rebinds,
                rest.loop_carried_rebinds,
                strict=True,
            ):
                assert orig_rec.var_name == rest_rec.var_name
                assert orig_rec.before.uuid == rest_rec.before.uuid
                assert orig_rec.after.uuid == rest_rec.after.uuid
                assert orig_rec.before_synthesized == rest_rec.before_synthesized
