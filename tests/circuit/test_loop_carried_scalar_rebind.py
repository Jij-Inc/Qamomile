"""Tests for loop-carried classical scalars (region arguments).

A qkernel loop body is traced exactly once, so a Python-level
reassignment like ``total = total + i`` inside ``qmc.range`` /
``qmc.items`` used to produce IR whose right-hand side read the fixed
pre-loop value instead of the previous iteration's value, and the
transpiler rejected the shape with a targeted ``ValidationError``.

The control-flow region-argument redesign (R1) makes these carries
explicit: the frontend rebinds each read-before-write classical scalar
candidate to a fresh region argument at loop-body entry (MLIR
``iter_args`` style — see ``RegionArg`` in
``qamomile/circuit/ir/operation/control_flow.py``), the constant folder
evaluates fully static carried loops at compile time, and the classical
executor and emit-time unrolling thread ``init → block_arg → yielded →
result`` per iteration. ``sum(range(4))`` now compiles to ``6``; these
tests pin Python-exact execution for the previously rejected shapes.

``while`` loops keep the record-based rejection for non-condition
carries: a runtime while loop cannot be unrolled, so a per-iteration
classical carry (other than the aliased condition pair) is still not
representable. Measurement-backed ``Bit`` carries in ``for`` loops keep
their targeted rejection too.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    WhileOperation,
)
from qamomile.circuit.transpiler.errors import QamomileCompileError, ValidationError

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
# Supported: loop-carried classical scalars execute with Python semantics
# ---------------------------------------------------------------------------


class TestSupportedLoopCarriedScalars:
    """Region-argument carries compute exactly what Python computes."""

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
        """Accumulation guarded by a loop-var if (phi-mediated) carries: 2+3 == 5."""

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
        """`total = total + 1` counts iterations: n=3 -> 3."""

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
        """A plain `total = 0` initialization synthesizes a constant init."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = 0
            for i in qmc.range(n):
                total = total + i
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    def test_float_angle_accumulation_drives_gate(self):
        """A Float carry driving a gate angle emits the correct rotation.

        n=2 accumulates 2.0; rx(2.0 * pi) is a full rotation, so the
        measured qubit stays |0>.
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

    def test_carry_inside_quantum_loop_body(self):
        """A carried angle updated next to gates threads per unrolled iteration.

        Three rx rotations of pi/3, 2pi/3, pi sum to 2pi -> |0>.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = 0.0
            for _i in qmc.range(3):
                theta = theta + 1.0471975511965976  # pi / 3
                q = qmc.rx(q, theta)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"dummy": 0}) == 0

    def test_nested_loop_accumulation(self):
        """Accumulating an outer variable inside nested loops carries: 3 hits."""

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
        """A folded carry can drive a later qmc.range bound.

        n=4 accumulates 2+3 == 5; five X gates flip |0> to |1>.
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
        """A carry inside a called sub-kernel survives inline cloning.

        The region args ride the sub-kernel's ForOperation and are
        remapped by inline cloning (all_input_values / replace_values),
        so the executor threads consistent UUIDs.
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

    def test_swap_carries_simultaneously(self):
        """`a, b = b, a` carries both values with simultaneous semantics.

        Three swaps of (1, 2) leave a == 2 — a partially-updated
        (sequential) advance would compute a == b instead.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            a = qmc.uint(1)
            b = qmc.uint(2)
            for _i in qmc.range(n):
                a, b = b, a
            return a

        assert _sample_single(kernel, bindings={"n": 3}) == 2

    def test_items_loop_accumulation(self):
        """A carried counter inside qmc.items counts the dict entries."""

        @qmc.qkernel
        def kernel(
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            count = qmc.uint(0)
            for (_i, _j), _coeff in qmc.items(ising):
                count = count + 1
            return count

        assert (
            _sample_single(kernel, bindings={"ising": {(0, 1): 1.0, (1, 2): 0.5}}) == 2
        )

    def test_zero_trip_loop_passes_init_through(self):
        """A zero-trip loop's region result is the init value.

        The bound is symbolic at trace time (so the body IS traced and
        region args ARE created) and resolves to 0 at transpile time.
        With n=0 the carried angle stays 0.0 and ``rx(pi)`` flips the
        qubit; with n=1 the carry adds 1.0 and ``rx(2*pi)`` leaves it.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = qmc.float_(0.0)
            for _i in qmc.range(n):
                total = total + 1.0
            q = qmc.qubit("q")
            q = qmc.rx(q, (total + 1.0) * 3.141592653589793)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"n": 0}) == 1
        assert _sample_single(kernel, bindings={"n": 1}) == 0

    def test_dead_and_taken_compile_time_branch_accumulation(self):
        """Compile-time-selected branches carry (or skip) correctly.

        flag=0 leaves the dead branch out (total stays 0); flag=1 takes
        the branch and accumulates sum(range(4)) == 6.
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


# ---------------------------------------------------------------------------
# Rejected: shapes region arguments cannot (yet) represent
# ---------------------------------------------------------------------------


class TestRejectedRebinds:
    """Non-representable loop-carried shapes keep their targeted errors."""

    def test_symbolic_bound_accumulation_rejected_structurally(self):
        """A runtime-parameter loop bound is rejected by shape validation.

        Loop bounds determine the emitted gate count, so they must be
        compile-time bound regardless of carried values; the carry no
        longer produces its own rejection, so the structural check is
        what fires.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        with pytest.raises(QamomileCompileError, match="runtime parameter"):
            _transpile(kernel, parameters=["n"], bindings={})

    def test_while_counter_rejected(self):
        """A classical counter inside a measurement-conditioned while is rejected.

        A runtime while loop cannot be unrolled, so a non-condition
        classical carry has no emit-time threading; the record-based
        rejection stays.
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

        Measurement-backed ``Bit`` values are excluded from region
        binding (their per-iteration value is a runtime outcome, which
        unrolled threading cannot carry), so the record-based rejection
        stays: only ``WhileOperation.operands[1]`` gets loop-carried
        clbit aliasing from the resource allocator.
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

    def test_build_attaches_region_args(self):
        """kernel.build() represents the carry as region args, not records."""

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
        assert any(loop.region_args for loop in loops)
        assert not any(loop.loop_carried_rebinds for loop in loops)
        for loop in loops:
            for arg in loop.region_args:
                assert arg.result in loop.results

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
# Cross-backend execution: carried scalars drive identical circuits everywhere
# ---------------------------------------------------------------------------


def _make_transpiler(backend: str):
    """Build the requested backend transpiler, skipping if the SDK is absent.

    Args:
        backend (str): One of ``"qiskit"``, ``"quri_parts"``, ``"cudaq"``.

    Returns:
        The backend transpiler instance.
    """
    if backend == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler as T
    elif backend == "quri_parts":
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts import QuriPartsTranspiler as T
    else:
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler as T
    return T()


class TestCarriedScalarCrossBackend:
    """The carried-angle kernel samples identically on every backend."""

    @pytest.mark.parametrize("backend", ["qiskit", "quri_parts", "cudaq"])
    @pytest.mark.parametrize("n_expected", [(2, 0), (3, 1), (4, 0)])
    def test_carried_angle_rotation(self, backend, n_expected):
        """n accumulated pi-rotations flip the qubit iff n is odd."""
        n, expected = n_expected

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            total = qmc.float_(0.0)
            for _i in qmc.range(n):
                total = total + 1.0
            q = qmc.qubit("q")
            q = qmc.rx(q, total * 3.141592653589793)
            return qmc.measure(q)

        transpiler = _make_transpiler(backend)
        executable = transpiler.transpile(kernel, bindings={"n": n})
        result = executable.sample(transpiler.executor(), shots=100).result()
        assert len(result.results) == 1, f"expected deterministic result: {result}"
        assert result.results[0][0] == expected


# ---------------------------------------------------------------------------
# IR plumbing: region args and rebind records survive serialization
# ---------------------------------------------------------------------------


class TestRegionArgSerialization:
    """RegionArg records round-trip through both wire formats."""

    def _build_block_with_region_args(self):
        """Build a block whose ForOperation carries region args."""

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
    def test_region_args_roundtrip(self, fmt):
        """Region args survive a serialize/deserialize cycle."""
        from qamomile.circuit.ir.serialize import (
            dump_json,
            dump_msgpack,
            load_json,
            load_msgpack,
        )

        block = self._build_block_with_region_args()
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(block)

        if fmt == "json":
            restored = load_json(dump_json(affine))
        else:
            restored = load_msgpack(dump_msgpack(affine))

        original_loops = _find_loops(affine.operations)
        restored_loops = _find_loops(restored.operations)
        assert len(original_loops) == len(restored_loops)
        found_region_args = False
        for orig, rest in zip(original_loops, restored_loops, strict=True):
            assert len(orig.region_args) == len(rest.region_args)
            for orig_arg, rest_arg in zip(
                orig.region_args, rest.region_args, strict=True
            ):
                found_region_args = True
                assert orig_arg.var_name == rest_arg.var_name
                assert orig_arg.init.uuid == rest_arg.init.uuid
                assert orig_arg.block_arg.uuid == rest_arg.block_arg.uuid
                assert orig_arg.yielded.uuid == rest_arg.yielded.uuid
                assert orig_arg.result.uuid == rest_arg.result.uuid
                assert rest_arg.result in rest.results
        assert found_region_args, "fixture must produce at least one region arg"


class TestRebindRecordSerialization:
    """LoopCarriedRebind records still round-trip (while-loop carries)."""

    def _build_block_with_records(self):
        """Build a block whose WhileOperation carries rebind records."""

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

        return kernel.build(parameters=["dummy"])

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
        found_records = False
        for orig, rest in zip(original_loops, restored_loops, strict=True):
            assert len(orig.loop_carried_rebinds) == len(rest.loop_carried_rebinds)
            for orig_rec, rest_rec in zip(
                orig.loop_carried_rebinds,
                rest.loop_carried_rebinds,
                strict=True,
            ):
                found_records = True
                assert orig_rec.var_name == rest_rec.var_name
                assert orig_rec.before.uuid == rest_rec.before.uuid
                assert orig_rec.after.uuid == rest_rec.after.uuid
                assert orig_rec.before_synthesized == rest_rec.before_synthesized
        assert found_records, "fixture must produce at least one rebind record"
