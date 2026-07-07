"""Tests for loop-carried classical scalar values (iter_args/yield semantics).

A qkernel loop body is traced exactly once. A Python-level reassignment
like ``total = total + i`` inside ``qmc.range`` / ``qmc.items`` is
promoted at trace time into a loop-carry slot (``LoopCarry``): the loop
enters with the initializer (``iter_arg``), each iteration reads the
previous iteration's value through a fresh body formal (``body_arg``),
the post-body value becomes the next iteration's input (``body_yield``),
and post-loop code reads the loop's carry ``result`` — matching Python
semantics on every executor (the classical segment interpreter and
emit-time unrolling).

Shapes no carry slot can express stay rejected with the targeted
``ValidationError``: while-loop carries (a runtime loop cannot thread a
classical value between iterations), measurement-backed Bit rebinds
(no clbit re-routing between iterations), and while-condition snapshots
(the clbit aliasing makes post-loop reads observe the final
measurement).
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
# Supported: loop-carried classical scalars compute Python semantics
# ---------------------------------------------------------------------------


class TestCarriedAccumulation:
    """Loop-carried classical scalar updates compile and match Python."""

    def test_for_accumulation(self):
        """`total = total + i` in qmc.range computes sum(range(4)) == 6."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    def test_builtin_range_accumulation(self):
        """builtin range() is transformed like qmc.range and carries equally."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in range(4):
                total = total + i
            return total

        assert _sample_single(kernel, bindings={"dummy": 0}) == 6

    def test_for_if_accumulation(self):
        """Accumulation guarded by a per-iteration if computes 2 + 3 == 5."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                if i > 1:
                    total = total + i
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 5

    def test_all_constant_accumulation(self):
        """`total = total + 1` carries despite the all-constant update.

        The initializer promotion wraps the constant handle in a
        symbolic one, so trace-time constant folding cannot collapse
        the update and erase the loop-carried dependency.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for _i in qmc.range(n):
                total = total + 1
            return total

        assert _sample_single(kernel, bindings={"n": 3}) == 3

    def test_plain_python_int_init(self):
        """A plain `total = 0` initialization is promoted and carried."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = 0
            for i in qmc.range(n):
                total = total + i
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    def test_float_angle_accumulation(self):
        """A Float accumulation driving a gate angle emits rx(2*pi).

        The classical carried loop is absorbed into the quantum segment
        and evaluated at emit time; rx(2*pi) leaves |0> invariant (up to
        global phase), so the measurement is deterministically 0.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = qmc.float_(0.0)
            for _i in qmc.range(n):
                total = total + 1.0
            q = qmc.qubit("q")
            q = qmc.rx(q, total * 3.141592653589793)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": 2}) == 0

    def test_nested_loop_accumulation(self):
        """Accumulating an outer variable inside nested loops counts i == j."""

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

        assert _sample_single(kernel, bindings={"n": 3}) == 3

    def test_accumulation_as_range_bound(self):
        """An accumulated total drives a later qmc.range bound: x^5 flips.

        The carried loop feeds the second loop's bound, so it is
        absorbed into the quantum segment and evaluated at emit time;
        total == 2 + 3 == 5 applies five X gates, flipping |0> to |1>.
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

        assert _sample_single(kernel, bindings={"n": 4}) == 1

    def test_subkernel_loop_accumulation(self):
        """Accumulation inside a called sub-kernel carries after inlining.

        The carry slots ride the sub-kernel's ForOperation and are
        remapped by inline cloning (all_input_values / replace_values),
        so the executors still see consistent UUIDs.
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

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    def test_augassign_accumulation(self):
        """`total += i` (AugAssign) carries like the explicit form."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total += i
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    def test_swap_rotation(self):
        """`a, b = b, a` in a loop swaps per iteration: three swaps land on (2, 1)."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
            q = qmc.qubit("q")
            qmc.measure(q)
            a = qmc.uint(1)
            b = qmc.uint(2)
            for _i in qmc.range(n):
                a, b = b, a
            return a, b

        assert _sample_single(kernel, bindings={"n": 3}) == (2, 1)

    def test_items_loop_accumulation(self):
        """A qmc.items loop accumulates dict coefficients into a Float carry."""

        @qmc.qkernel
        def kernel(
            coeffs: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            n: qmc.UInt,
        ) -> qmc.Bit:
            total = qmc.float_(0.0)
            for (_i, _j), coeff in qmc.items(coeffs):
                total = total + coeff
            q = qmc.qubit("q")
            q = qmc.rx(q, total)
            return qmc.measure(q)

        # total == 2*pi: rx(2*pi) leaves |0> invariant up to global phase.
        assert (
            _sample_single(
                kernel,
                bindings={
                    "coeffs": {(0, 1): 3.141592653589793, (1, 2): 3.141592653589793},
                    "n": 0,
                },
            )
            == 0
        )

    def test_zero_trip_static_loop_keeps_initializer(self):
        """A statically-zero-trip loop is never traced: rx(0.0) measures 0."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = 0.0
            for _i in qmc.range(n):
                total = total + 3.141592653589793
            q = qmc.qubit("q")
            q = qmc.rx(q, total)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": 0}) == 0

    def test_one_trip_loop_carries_once(self):
        """A single-iteration loop yields exactly one update: rx(pi) flips."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = 0.0
            for _i in qmc.range(n):
                total = total + 3.141592653589793
            q = qmc.qubit("q")
            q = qmc.rx(q, total)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": 1}) == 1

    def test_shared_constant_initializer_carries_split(self):
        """Two carries sharing one CONSTANT initializer split and both work.

        The per-candidate promotion wraps each variable's constant
        binding in its own symbolic handle, so `b = a` no longer shares
        an IR value with `a` inside the body and each carry gets its own
        back-edge: a = 1 + n, b = 1 + 10 n.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
            q = qmc.qubit("q")
            qmc.measure(q)
            a = qmc.uint(1)
            b = a
            for _i in qmc.range(n):
                a = a + 1
                b = b + 10
            return a, b

        assert _sample_single(kernel, bindings={"n": 3}) == (4, 31)

    def test_annotated_assign_accumulation(self):
        """`total: qmc.UInt = total + i` (AnnAssign) carries like plain Assign."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total: qmc.UInt = total + i  # noqa: F821 - traced read-before-write
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    def test_emit_zero_trip_carried_loop_passthrough(self):
        """A carried loop whose bound resolves to zero AT EMIT passes through.

        The first loop leaves ``total == 0`` for ``n=2`` (the guard
        never fires), so the second (carried, quantum-absorbed) loop
        unrolls to zero iterations at emit; its carry result must still
        resolve to the loop-entry value, giving rx(0.0) and a
        deterministic 0.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = qmc.uint(0)
            for i in qmc.range(n):
                if i > 1:
                    total = total + i
            angle = 0.0
            for _j in qmc.range(total):
                angle = angle + 3.141592653589793
            q = qmc.qubit("q")
            q = qmc.rx(q, angle)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": 2}) == 0

    def test_carry_slots_on_built_block(self):
        """build() promotes the classical rebind into a carry slot."""

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
        (loop,) = loops
        carries = list(loop.iter_carries())
        assert [c.var_name for c in carries] == ["total"]
        (carry,) = carries
        assert carry.iter_arg.get_const() == 0
        assert carry.result is loop.results[0]
        # The promoted classical rebind is gone; the carry slot is the
        # single representation of the back-edge.
        assert not loop.loop_carried_rebinds


# ---------------------------------------------------------------------------
# Rejected: shapes no carry slot can express
# ---------------------------------------------------------------------------


class TestRejectedRebinds:
    """Unsupported loop-carried shapes fail with the targeted error."""

    def test_while_counter_rejected(self):
        """A classical counter inside a measurement-conditioned while is rejected.

        A while loop's trip count is a runtime measurement outcome, so
        the loop cannot unroll, and no backend can thread a classical
        value between runtime-loop iterations.
        """

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

    def test_carried_loop_with_runtime_bound_rejected(self):
        """A carried loop whose bound stays a runtime parameter is rejected.

        The carry forces unrolling (a native runtime loop cannot thread
        classical values between iterations), and the pre-emit
        symbolic-bound diagnostic rejects unrollable loops whose bounds
        depend on runtime parameters.
        """
        from qamomile.circuit.transpiler.errors import QamomileCompileError

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        with pytest.raises(QamomileCompileError, match="depends on runtime parameter"):
            _transpile(kernel, parameters=["n"])

    def test_shared_symbolic_initializer_carries_rejected(self):
        """Two carried variables sharing one SYMBOLIC pre-loop value are rejected.

        A runtime-parameter initializer cannot be split by the constant
        promotion, so both records point at one value; the traced body
        holds a single read site per expression, and UUID-keyed
        substitution cannot give each variable its own back-edge —
        miscompiling would make one variable's update read the other's
        previous value.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt, x: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
            q = qmc.qubit("q")
            qmc.measure(q)
            a = x
            b = a
            for _i in qmc.range(n):
                a = a + 1
                b = b + 10
            return a, b

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 3}, parameters=["x"])

    def test_quantum_loop_carry_escaping_to_output_rejected(self):
        """A quantum-segment loop's carried result cannot be a block output.

        Emit-time unrolling computes the carried value while building
        the circuit, so it has no runtime representation the classical
        executor could surface; returning it would silently yield None.
        """
        from qamomile.circuit.transpiler.segments import (
            MultipleQuantumSegmentsError,
        )

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> tuple[qmc.Bit, qmc.UInt]:
            q = qmc.qubit("q")
            total = qmc.uint(0)
            for i in qmc.range(n):
                q = qmc.x(q)
                total = total + i
            return qmc.measure(q), total

        with pytest.raises(MultipleQuantumSegmentsError, match="carries classical"):
            _transpile(kernel, bindings={"n": 3})

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
        That gap is outside the loop-carried machinery's scope.
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
        machinery, and a Bit is not a promotable carry (a measurement
        result has no emit-time value to thread through unrolled
        iterations), so the ``if state:`` condition would keep
        addressing the pre-loop clbit while the branch measurements
        write elsewhere (measured divergence: with fresh-per-iteration
        Python semantics ``n=2`` yields ``out == 1``, but the emitted
        ForLoopOp circuit returns ``0``). Rejecting is the honest
        behavior until real aliasing lands.
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

    def test_dead_branch_accumulation_both_sides(self):
        """A rebind under a compile-time if carries only when taken.

        With the branch dead the loop keeps the initializer; with the
        branch taken the accumulation carries and matches Python.
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

        assert _sample_single(kernel, bindings={"n": 4, "flag": 0}) == 0
        assert _sample_single(kernel, bindings={"n": 4, "flag": 1}) == 6

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
# IR plumbing: carry slots and rebind records survive serialization
# ---------------------------------------------------------------------------


class TestCarrySerialization:
    """Carry slots and rebind records round-trip through both wire formats."""

    @staticmethod
    def _roundtrip(block, fmt):
        """Serialize and deserialize a block through the given format."""
        from qamomile.circuit.ir.serialize import (
            dump_json,
            dump_msgpack,
            load_json,
            load_msgpack,
        )

        if fmt == "json":
            return load_json(dump_json(block))
        return load_msgpack(dump_msgpack(block))

    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_carries_roundtrip(self, fmt):
        """Carry slots survive a serialize/deserialize cycle."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        block = kernel.build(parameters=["n"])
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(block)
        restored = self._roundtrip(affine, fmt)

        original_loops = _find_loops(affine.operations)
        restored_loops = _find_loops(restored.operations)
        assert len(original_loops) == len(restored_loops) == 1
        original_carries = list(original_loops[0].iter_carries())
        restored_carries = list(restored_loops[0].iter_carries())
        assert len(original_carries) == len(restored_carries) == 1
        for orig, rest in zip(original_carries, restored_carries, strict=True):
            assert orig.var_name == rest.var_name
            assert orig.iter_arg.uuid == rest.iter_arg.uuid
            assert orig.body_arg.uuid == rest.body_arg.uuid
            assert orig.body_yield.uuid == rest.body_yield.uuid
            assert orig.result.uuid == rest.result.uuid

    def test_mismatched_carry_slot_types_rejected_on_load(self):
        """A payload whose carry slots disagree in type fails to decode."""
        from qamomile.circuit.ir.serialize import from_dict, to_dict

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        block = kernel.build(parameters=["n"])
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(block)
        payload = to_dict(affine)

        loop_dict = next(
            op for op in payload["block"]["operations"] if op["$type"] == "ForOperation"
        )
        measure_dict = next(
            op
            for op in payload["block"]["operations"]
            if op["$type"] == "MeasureOperation"
        )
        # Point the carry's iter_arg at the measurement's Bit result: a
        # UInt carry fed by a Bit value must be rejected at load time.
        loop_dict["iter_arg_refs"] = [measure_dict["result_refs"][0]]

        with pytest.raises(ValueError, match="mismatched slot types"):
            from_dict(payload)

    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_records_roundtrip(self, fmt):
        """Residual rebind records survive a serialize/deserialize cycle.

        Bit rebinds are not promotable carries, so this kernel keeps
        genuine ``LoopCarriedRebind`` records on its loop.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            init_q = qmc.qubit("init")
            state = qmc.measure(init_q)
            for _i in qmc.range(n):
                if state:
                    q_zero = qmc.qubit("zero")
                    state = qmc.measure(q_zero)
                else:
                    q_one = qmc.qubit("one")
                    q_one = qmc.x(q_one)
                    state = qmc.measure(q_one)
            return state

        block = kernel.build(parameters=["n"])
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(block)
        restored = self._roundtrip(affine, fmt)

        original_loops = _find_loops(affine.operations)
        restored_loops = _find_loops(restored.operations)
        assert len(original_loops) == len(restored_loops) == 1
        assert original_loops[0].loop_carried_rebinds, "expected residual records"
        for orig_rec, rest_rec in zip(
            original_loops[0].loop_carried_rebinds,
            restored_loops[0].loop_carried_rebinds,
            strict=True,
        ):
            assert orig_rec.var_name == rest_rec.var_name
            assert orig_rec.before.uuid == rest_rec.before.uuid
            assert orig_rec.after.uuid == rest_rec.after.uuid
            assert orig_rec.before_synthesized == rest_rec.before_synthesized
