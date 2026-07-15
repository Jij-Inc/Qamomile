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

import math

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.containers import Tuple
from qamomile.circuit.frontend.operation.control_flow import record_loop_rebinds
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.canonical import to_canonical_bytes
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    WhileOperation,
)
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import DictValue, TupleValue, Value
from qamomile.circuit.transpiler.errors import (
    QamomileCompileError,
    QubitRebindError,
    ValidationError,
)
from qamomile.circuit.transpiler.passes.analyze import (
    reject_loop_carried_classical_rebinds,
)

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

LOOP_CARRIED = "Loop-carried"


def _transpile(kernel, bindings=None, parameters=None):
    """Transpile a kernel on Qiskit and return the executable.

    Args:
        kernel (Any): Qkernel to transpile.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to
            None.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            None.

    Returns:
        Any: Qiskit executable program.
    """
    transpiler = QiskitTranspiler()
    return transpiler.transpile(kernel, bindings=bindings or {}, parameters=parameters)


def _sample_single(kernel, bindings=None, shots=200):
    """Sample a kernel and return its single deterministic outcome.

    Args:
        kernel (Any): Qkernel to transpile and execute.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to
            None.
        shots (int): Number of samples. Defaults to 200.

    Returns:
        Any: The sole sampled result value.

    Raises:
        AssertionError: If sampling produces more than one distinct outcome.
    """
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings=bindings or {})
    result = executable.sample(transpiler.executor(), shots=shots).result()
    assert len(result.results) == 1, f"expected deterministic result: {result}"
    return result.results[0][0]


def _find_loops(ops):
    """Collect every loop operation reachable from ``ops``.

    Args:
        ops (list[Any]): Operation list to walk recursively.

    Returns:
        list[Any]: Reachable for, items, and while operations.
    """
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

    def test_for_items_carry_observes_dict_insertion_order(self):
        """A non-commutative carry follows Python mapping iteration order."""

        @qmc.qkernel
        def kernel(items: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            total = qmc.uint(0)
            for key, _value in qmc.items(items):
                total = total * 10 + key
            return total

        assert _sample_single(kernel, bindings={"items": {1: 0, 2: 0}}) == 12
        assert _sample_single(kernel, bindings={"items": {2: 0, 1: 0}}) == 21

    def test_for_items_uint_values_keep_uint_carry_type(self):
        """Dict[UInt, UInt] item values accumulate through a UInt carry."""

        @qmc.qkernel
        def kernel(items: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            total = qmc.uint(0)
            for _key, value in qmc.items(items):
                total = total + value
            return total

        assert _sample_single(kernel, bindings={"items": {0: 2, 1: 3}}) == 5

    def test_for_items_bit_values_keep_bit_condition_type(self):
        """Dict[UInt, Bit] item values remain usable as static conditions."""

        @qmc.qkernel
        def kernel(items: qmc.Dict[qmc.UInt, qmc.Bit]) -> qmc.Bit:
            q = qmc.qubit("q")
            for _key, enabled in qmc.items(items):
                if enabled:
                    q = qmc.x(q)
            return qmc.measure(q)

        assert (
            _sample_single(
                kernel,
                bindings={"items": {0: True, 1: False}},
            )
            == 1
        )

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
        """Accumulation guarded by a loop-var if carries through merges: 2+3 == 5."""

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

    def test_runtime_if_equal_constants_eliminates_classical_merge(self):
        """Separately-created equal branch constants remain one SSA value."""

        @qmc.qkernel
        def kernel() -> tuple[qmc.Bit, qmc.UInt]:
            predicate_q = qmc.qubit("predicate")
            predicate = qmc.measure(predicate_q)
            target = qmc.qubit("target")
            value = qmc.uint(0)
            if predicate:
                target = qmc.x(target)
                value = qmc.uint(1)
            else:
                target = qmc.z(target)
                value = qmc.uint(1)
            return qmc.measure(target), value

        assert _sample_single(kernel) == (0, 1)

    @pytest.mark.parametrize(("flag", "expected"), [(0, 2), (1, 1)])
    def test_nonempty_store_only_loop_may_change_scalar_type(self, flag, expected):
        """A guaranteed loop overwrite needs no zero-trip type join."""

        @qmc.qkernel
        def kernel(bound_flag: qmc.UInt) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            value = qmc.bit(False)
            for _i in qmc.range(2):
                value = qmc.uint(1)
            if bound_flag == 1:
                out = value
            else:
                out = qmc.uint(2)
            return out

        assert _sample_single(kernel, bindings={"bound_flag": flag}) == expected

    @pytest.mark.parametrize("n", [0, 3])
    def test_dead_post_loop_read_drops_symbolic_mixed_type_record(self, n):
        """Pruning the only read makes an unsupported residual carry dead."""

        @qmc.qkernel
        def kernel(count: qmc.UInt, flag: qmc.UInt) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            value = qmc.bit(False)
            for _i in qmc.range(count):
                value = qmc.uint(1)
            if flag == 1:
                out = value
            else:
                out = qmc.uint(2)
            return out

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(
            kernel,
            bindings={"flag": 0},
            parameters=["count"],
        )
        result = executable.sample(
            transpiler.executor(),
            bindings={"count": n},
            shots=20,
        ).result()
        assert result.results == [(2, 20)]

    @pytest.mark.parametrize(
        ("n", "expected_sign"),
        [(0, 1.0), (1, -1.0), (2, -1.0)],
    )
    def test_constant_loop_overwrite_preserves_signed_zero(self, n, expected_sign):
        """Exact constant identity keeps negative zero distinct from positive."""

        @qmc.qkernel
        def kernel(count: qmc.UInt) -> qmc.Float:
            qmc.measure(qmc.qubit("segment_anchor"))
            value = qmc.float_(0.0)
            for _i in qmc.range(count):
                value = qmc.float_(-0.0)
            return value

        result = _sample_single(kernel, bindings={"count": n})
        assert result == 0.0
        assert math.copysign(1.0, result) == expected_sign

    def test_huge_static_range_raises_typed_compile_error(self):
        """A range larger than ssize_t never leaks a raw OverflowError."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            for _i in qmc.range(10**100):
                q = qmc.x(q)
            return qmc.measure(q)

        with pytest.raises(QamomileCompileError, match="backend-representable"):
            _transpile(kernel)

    def test_huge_static_region_range_raises_typed_compile_error(self):
        """A huge scalar-carry range also reaches the typed emit diagnostic."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            angle = qmc.float_(0.0)
            for _i in qmc.range(10**100):
                q = qmc.rx(q, angle)
                angle = angle + 1.0
            return qmc.measure(q)

        with pytest.raises(QamomileCompileError, match="backend-representable"):
            _transpile(kernel)

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

    def test_one_trip_measurement_yield_is_result_not_backedge(self):
        """A one-trip result may be runtime data without feeding another trip."""

        @qmc.qkernel
        def kernel() -> qmc.Float:
            target = qmc.qubit("target")
            total = qmc.float_(0.0)
            for _i in qmc.range(1):
                target = qmc.rx(target, total)
                measured = qmc.qubit_array(1, "measured")
                fixed = qmc.cast(measured, qmc.QFixed, int_bits=0)
                total = qmc.measure(fixed)
            return total

        assert _sample_single(kernel) == 0.0

    def test_one_trip_inner_bound_uses_initializer_not_final_yield(self):
        """A one-trip body structural read cannot depend on its final update."""

        @qmc.qkernel
        def kernel(x: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            total = qmc.uint(0)
            for _i in qmc.range(1):
                for _j in qmc.range(total):
                    q = qmc.x(q)
                total = x
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel, parameters=["x"])
        result = executable.sample(
            transpiler.executor(), shots=100, bindings={"x": 7}
        ).result()
        assert result.results == [(0, 100)]

    def test_one_entry_items_inner_bound_uses_initializer(self):
        """A one-entry items body does not back-propagate its final update."""

        @qmc.qkernel
        def kernel(
            data: qmc.Dict[qmc.UInt, qmc.Float],
            x: qmc.UInt,
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            total = qmc.uint(0)
            for _key, _value in qmc.items(data):
                for _j in qmc.range(total):
                    q = qmc.x(q)
                total = x
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(
            kernel,
            bindings={"data": {0: 1.0}},
            parameters=["x"],
        )
        result = executable.sample(
            transpiler.executor(), shots=100, bindings={"x": 7}
        ).result()
        assert result.results == [(0, 100)]

    @pytest.mark.parametrize(("initial", "expected"), [(0, 0), (1, 1)])
    def test_one_trip_runtime_if_yield_tracks_final_constant_fold(
        self, initial, expected
    ):
        """Flattened runtime-if yields follow folded branch results."""

        @qmc.qkernel
        def kernel(prepare: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            if prepare == 1:
                q = qmc.x(q)
            bit = qmc.measure(q)
            total = qmc.uint(0)
            for _i in qmc.range(1):
                if bit:
                    total = total + 1
            return total

        assert _sample_single(kernel, bindings={"prepare": initial}) == expected

    def test_static_multi_trip_overwrite_drops_runtime_dependency(self):
        """A guaranteed nonempty constant overwrite forgets its initializer."""

        @qmc.qkernel
        def kernel(x: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            total = x
            for _i in qmc.range(2):
                total = 0
            for _j in qmc.range(total):
                q = qmc.x(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel, parameters=["x"])
        result = executable.sample(
            transpiler.executor(), shots=100, bindings={"x": 7}
        ).result()
        assert result.results == [(0, 100)]

    def test_static_multi_trip_carry_resolves_qubit_array_size(self):
        """A statically accumulated carry can size a later qubit register."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            size = qmc.uint(0)
            for _i in qmc.range(n):
                size = size + 1
            register = qmc.qubit_array(size, "register")
            for i in qmc.range(size):
                register[i] = qmc.x(register[i])
            return qmc.measure(register)

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel, bindings={"n": 2})
        result = executable.sample(transpiler.executor(), shots=100).result()
        assert result.results == [((1, 1), 100)]

    def test_static_carry_evaluation_keeps_quantum_loop_body(self):
        """Static carry replay never expands or removes a quantum loop body."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            angle = qmc.float_(0.0)
            for _i in qmc.range(n):
                q = qmc.rx(q, angle)
                angle = angle + 1.0
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        block = kernel.build(n=2)
        affine = transpiler.inline(block)
        validated = transpiler.affine_validate(affine)
        lowered = transpiler.partial_eval(validated, bindings={"n": 2})
        loops = _find_loops(lowered.operations)
        assert len(loops) == 1
        assert isinstance(loops[0], ForOperation)

    def test_static_replay_keeps_incompletely_evaluated_classical_body(self):
        """A known carry cannot erase another unresolved classical operation."""

        @qmc.qkernel
        def kernel(x: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = x
            for _i in qmc.range(2):
                _unused = qmc.uint(1) // x
                total = qmc.uint(0)
            return total

        transpiler = QiskitTranspiler()
        block = kernel.build(parameters=["x"])
        affine = transpiler.inline(block)
        validated = transpiler.affine_validate(affine)
        lowered = transpiler.partial_eval(validated)

        assert len(_find_loops(lowered.operations)) == 1

    def test_static_replay_does_not_reuse_a_previous_iteration_result(self):
        """A failed later fold cannot inherit the prior trip's result UUID."""

        @qmc.qkernel
        def kernel() -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(2):
                total = qmc.uint(1) // (qmc.uint(1) - i)
            return total

        transpiler = QiskitTranspiler()
        affine = transpiler.inline(kernel.build())
        validated = transpiler.affine_validate(affine)
        lowered = transpiler.partial_eval(validated)

        assert len(_find_loops(lowered.operations)) == 1

    def test_nested_static_replay_uses_one_pass_wide_budget(self, monkeypatch):
        """Nested loop products stop replay when their shared budget is spent."""
        from qamomile.circuit.ir.operation.operation import QInitOperation
        from qamomile.circuit.transpiler.passes import compile_time_if_lowering

        monkeypatch.setattr(
            compile_time_if_lowering,
            "_MAX_STATIC_CARRY_ITERATIONS",
            8,
        )

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            size = qmc.uint(0)
            for _i in qmc.range(3):
                for _j in qmc.range(3):
                    size = size + 1
            register = qmc.qubit_array(size, "register")
            return qmc.measure(register)

        transpiler = QiskitTranspiler()
        affine = transpiler.inline(kernel.build())
        validated = transpiler.affine_validate(affine)
        lowered = transpiler.partial_eval(validated)

        register_init = next(
            op
            for op in lowered.operations
            if isinstance(op, QInitOperation)
            and op.results
            and op.results[0].name == "register"
        )
        assert not register_init.results[0].shape[0].is_constant()

    def test_static_replay_budget_resets_for_each_pass_run(self, monkeypatch):
        """Reusing one lowering pass gives each invocation a fresh budget."""
        from qamomile.circuit.transpiler.passes import compile_time_if_lowering
        from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
            CompileTimeIfLoweringPass,
        )

        monkeypatch.setattr(
            compile_time_if_lowering,
            "_MAX_STATIC_CARRY_ITERATIONS",
            3,
        )

        @qmc.qkernel
        def kernel() -> qmc.UInt:
            total = qmc.uint(0)
            for _i in qmc.range(3):
                total = total + 1
            qmc.measure(qmc.qubit("segment_anchor"))
            return total

        transpiler = QiskitTranspiler()
        affine = transpiler.inline(kernel.build())
        lowering = CompileTimeIfLoweringPass()

        first = lowering.run(affine)
        second = lowering.run(affine)

        assert not _find_loops(first.operations)
        assert not _find_loops(second.operations)
        assert first.output_values[0].get_const() == 3
        assert second.output_values[0].get_const() == 3
        assert first.output_values[0].uuid == second.output_values[0].uuid
        assert first.output_values[0].logical_id == second.output_values[0].logical_id

        assert to_canonical_bytes(first) == to_canonical_bytes(second)

    def test_static_for_items_replay_preserves_identity_across_runs(self):
        """Repeated items-loop replay produces identical canonical IR."""
        from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
            CompileTimeIfLoweringPass,
        )

        @qmc.qkernel
        def kernel(items: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            total = qmc.uint(0)
            for key, _value in qmc.items(items):
                total = total * 10 + key
            return total

        transpiler = QiskitTranspiler()
        affine = transpiler.inline(kernel.build(items={1: 0, 2: 0}))
        lowering = CompileTimeIfLoweringPass()

        first = lowering.run(affine)
        second = lowering.run(affine)

        assert first.output_values[0].get_const() == 12
        assert first.output_values[0].uuid == second.output_values[0].uuid
        assert first.output_values[0].logical_id == second.output_values[0].logical_id
        assert to_canonical_bytes(first) == to_canonical_bytes(second)

    def test_static_quantum_loop_carry_can_surface_as_constant(self):
        """A statically evaluated quantum-loop carry is a real block output."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> tuple[qmc.Bit, qmc.UInt]:
            q = qmc.qubit("q")
            total = qmc.uint(0)
            for i in qmc.range(n):
                q = qmc.x(q)
                total = total + i
            return qmc.measure(q), total

        assert _sample_single(kernel, bindings={"n": 3}) == (1, 3)

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

    def test_shared_runtime_initializer_carries_split(self):
        """Aliased runtime initializers receive independent body formals."""

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

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel, bindings={"n": 3}, parameters=["x"])
        result = executable.sample(
            transpiler.executor(), shots=100, bindings={"x": 1}
        ).result()
        assert result.results == [((4, 31), 100)]

    @pytest.mark.parametrize(
        ("n", "expected"),
        [(0, (1, 2)), (1, (2, 3)), (3, (4, 5))],
    )
    def test_store_only_live_out_depends_on_another_carry(self, n, expected):
        """A store-only live-out receives the last iteration's carried value."""

        @qmc.qkernel
        def kernel(bound: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
            q = qmc.qubit("q")
            qmc.measure(q)
            a = qmc.uint(1)
            b = qmc.uint(2)
            for _i in qmc.range(bound):
                a = b
                b = a + 1
            return a, b

        assert _sample_single(kernel, bindings={"bound": n}) == expected

    @pytest.mark.parametrize(("n", "expected"), [(0, 7), (1, 5), (3, 5)])
    def test_store_only_constant_live_out_preserves_zero_trip(self, n, expected):
        """A constant live-out yields the initializer only for zero trips."""

        @qmc.qkernel
        def kernel(bound: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            last = qmc.uint(7)
            for _i in qmc.range(bound):
                last = 5
            return last

        assert _sample_single(kernel, bindings={"bound": n}) == expected

    def test_nested_zero_trip_loop_preserves_outer_carry(self):
        """An empty inner loop keeps the preceding outer-iteration value."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            last = qmc.float_(3.141592653589793)
            for i in qmc.range(3):
                for _j in qmc.range(qmc.uint(2) - i):
                    last = qmc.float_(0.0)
                q = qmc.rx(q, last)
            return qmc.measure(q)

        assert _sample_single(kernel) == 0

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

    def test_named_expression_accumulation(self):
        """`total := total + i` follows RHS-first carry semantics."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                _ = (total := total + i)
            return total

        assert _sample_single(kernel, bindings={"n": 4}) == 6

    @pytest.mark.parametrize("n", [0, 1, 3])
    @pytest.mark.parametrize("handle_init", [False, True], ids=["plain", "handle"])
    def test_identity_update_keeps_promoted_initializer(self, n, handle_init):
        """A promoted `total = total` carry preserves its initializer."""

        @qmc.qkernel
        def kernel(bound: qmc.UInt) -> qmc.Bit:
            total = qmc.float_(3.141592653589793) if handle_init else 3.141592653589793
            for _i in qmc.range(bound):
                total = total
            q = qmc.qubit("q")
            q = qmc.rx(q, total)
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"bound": n}) == 1

    @pytest.mark.parametrize("bound", [0, 1, 3])
    def test_identity_plain_uint_preserves_python_index(self, bound):
        """An unchanged plain int stays usable for list indexing after a loop."""

        @qmc.qkernel
        def kernel(bound: qmc.UInt) -> qmc.Bit:
            angles = [0.0, 3.141592653589793]
            n = 1
            for _i in qmc.range(bound):
                n = n
            q = qmc.qubit("q")
            q = qmc.rx(q, angles[n])
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"bound": bound}) == 1

    @pytest.mark.parametrize(
        "items",
        [{}, {0: 0}, {0: 0, 1: 0}],
        ids=["empty", "one-entry", "two-entry"],
    )
    def test_for_items_identity_plain_uint_preserves_python_index(self, items):
        """ForItems uses the same identity fast path for plain Python values."""

        @qmc.qkernel
        def kernel(items: qmc.Dict[qmc.UInt, qmc.UInt]) -> qmc.Bit:
            angles = [0.0, 3.141592653589793]
            n = 1
            for _key, _value in qmc.items(items):
                n = n
            q = qmc.qubit("q")
            q = qmc.rx(q, angles[n])
            return qmc.measure(q)

        assert _sample_single(kernel, bindings={"items": items}) == 1

    def test_nested_identity_plain_uint_preserves_python_index(self):
        """Nested identity substitution reaches a real inner recurrence."""

        @qmc.qkernel
        def kernel() -> tuple[qmc.Bit, qmc.UInt]:
            angles = [0.0, 3.141592653589793]
            base = 1
            total = qmc.uint(0)
            for _i in qmc.range(2):
                for _j in qmc.range(2):
                    total = total + base
                base = base
            q = qmc.qubit("q")
            q = qmc.rx(q, angles[base])
            return qmc.measure(q), total

        assert _sample_single(kernel) == (1, 4)

    def test_identity_carry_restores_python_binding_but_recurrence_does_not(self):
        """Frontend publishes plain identities without hiding their valid IR."""

        @qmc.qkernel
        def identity() -> qmc.UInt:
            n = 1
            for _i in qmc.range(2):
                n = n
            return n

        @qmc.qkernel
        def divergent() -> qmc.UInt:
            n = qmc.uint(1)
            for _i in qmc.range(2):
                n = n + 1
            return n

        identity_block = identity.build()
        [identity_loop] = _find_loops(identity_block.operations)
        assert len(identity_loop.region_args) == 1
        assert identity_loop.region_args[0].yielded.uuid == (
            identity_loop.region_args[0].block_arg.uuid
        )

        divergent_block = divergent.build()
        [divergent_loop] = _find_loops(divergent_block.operations)
        assert len(divergent_loop.region_args) == 1
        assert divergent_block.output_values[0].uuid == (
            divergent_loop.region_args[0].result.uuid
        )

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

    def test_region_args_on_built_block(self):
        """Build a classical rebind as one explicit region argument."""

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
        assert [arg.var_name for arg in loop.region_args] == ["total"]
        (region_arg,) = loop.region_args
        assert region_arg.init.get_const() == 0
        assert region_arg.result is loop.results[0]
        # The promoted classical rebind is gone; the RegionArg is the
        # single representation of the back-edge.
        assert not loop.loop_carried_rebinds


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

    def test_while_condition_pre_loop_expression_rejected(self):
        """A lazy expression of the entry condition remains a stale snapshot."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            old_not = ~run
            out = qmc.qubit("out")
            while run:
                stop = qmc.qubit("stop")
                run = qmc.measure(stop)
            if old_not:
                out = qmc.x(out)
            return qmc.measure(out)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_while_condition_body_snapshot_after_update_rejected(self):
        """A body-local old expression is stale after remeasurement."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            out = qmc.qubit("out")
            while run:
                old_not = ~run
                run = qmc.measure(qmc.qubit("stop"))
                if old_not:
                    out = qmc.x(out)
            return qmc.measure(out)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_while_condition_snapshot_after_runtime_if_source_rejected(self):
        """A branch-local remeasurement invalidates later nested snapshot reads."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            old = run
            selector_q = qmc.qubit("selector")
            selector_q = qmc.x(selector_q)
            selector = qmc.measure(selector_q)
            out = qmc.qubit("out")
            while run:
                if selector:
                    run = qmc.measure(qmc.qubit("true_stop"))
                    if old:
                        out = qmc.x(out)
                else:
                    run = qmc.measure(qmc.qubit("false_stop"))
            return qmc.measure(out)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_while_condition_snapshot_before_runtime_if_source_allowed(self):
        """A branch snapshot read before its own remeasurement remains valid."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            old = run
            selector = qmc.measure(qmc.qubit("selector"))
            out = qmc.qubit("out")
            while run:
                if selector:
                    run = qmc.measure(qmc.qubit("true_stop"))
                else:
                    if old:
                        out = qmc.x(out)
                    run = qmc.measure(qmc.qubit("false_stop"))
            return qmc.measure(out)

        assert _sample_single(kernel) == 1

    def test_while_condition_snapshot_after_nested_merge_source_rejected(self):
        """Nested runtime merges expose every branch-local update producer."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            old = run
            outer_q = qmc.qubit("outer")
            outer_q = qmc.x(outer_q)
            outer = qmc.measure(outer_q)
            inner_q = qmc.qubit("inner")
            inner_q = qmc.x(inner_q)
            inner = qmc.measure(inner_q)
            out = qmc.qubit("out")
            while run:
                if outer:
                    if inner:
                        run = qmc.measure(qmc.qubit("inner_true_stop"))
                        if old:
                            out = qmc.x(out)
                    else:
                        run = qmc.measure(qmc.qubit("inner_false_stop"))
                else:
                    run = qmc.measure(qmc.qubit("outer_false_stop"))
            return qmc.measure(out)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_while_condition_snapshot_between_measure_and_slice_rejected(self):
        """A later slice producer cannot hide the earlier clbit measurement."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            old = run
            selector_q = qmc.qubit("selector")
            selector_q = qmc.x(selector_q)
            selector = qmc.measure(selector_q)
            out = qmc.qubit("out")
            while run:
                if selector:
                    measured = qmc.measure(qmc.qubit_array(1, "true_stop"))
                    if old:
                        out = qmc.x(out)
                    view = measured[:]
                    run = view[0]
                else:
                    run = qmc.measure(qmc.qubit("false_stop"))
            return qmc.measure(out)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_updated_while_condition_is_body_dependency_barrier(self):
        """A body expression built after remeasurement reads the new value."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            out = qmc.qubit("out")
            while run:
                run = qmc.measure(qmc.qubit("stop"))
                new_not = ~run
                if new_not:
                    out = qmc.x(out)
            return qmc.measure(out)

        assert _sample_single(kernel) == 1

    def test_updated_while_condition_is_dependency_barrier(self):
        """An expression built from the updated condition is not stale."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            out = qmc.qubit("out")
            while run:
                stop = qmc.qubit("stop")
                run = qmc.measure(stop)
            new_not = ~run
            if new_not:
                out = qmc.x(out)
            return qmc.measure(out)

        assert _sample_single(kernel) == 1

    def test_while_condition_vector_element_snapshot_rejected(self):
        """A repeated access to the aliased vector element is stale."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            flags_q = qmc.qubit_array(1, "flags")
            flags_q[0] = qmc.x(flags_q[0])
            flags = qmc.measure(flags_q)
            run = flags[0]
            out = qmc.qubit("out")
            while run:
                run = qmc.measure(qmc.qubit("stop"))
            if flags[0]:
                out = qmc.x(out)
            return qmc.measure(out)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_while_condition_vector_sibling_element_is_not_stale(self):
        """Updating element zero does not invalidate a sibling clbit."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            flags_q = qmc.qubit_array(2, "flags")
            flags_q[0] = qmc.x(flags_q[0])
            flags_q[1] = qmc.x(flags_q[1])
            flags = qmc.measure(flags_q)
            run = flags[0]
            out = qmc.qubit("out")
            while run:
                run = qmc.measure(qmc.qubit("stop"))
            if flags[1]:
                out = qmc.x(out)
            return qmc.measure(out)

        assert _sample_single(kernel) == 1

    def test_nested_while_ignores_mutually_exclusive_sibling_read(self):
        """A sibling runtime branch does not read after the nested while."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            run_q = qmc.qubit("run")
            run_q = qmc.x(run_q)
            run = qmc.measure(run_q)
            selector = qmc.measure(qmc.qubit("selector"))
            out = qmc.qubit("out")
            if selector:
                while run:
                    run = qmc.measure(qmc.qubit("stop"))
            else:
                if run:
                    out = qmc.x(out)
            return qmc.measure(out)

        assert _sample_single(kernel) == 1

    def test_dead_loop_result_ignores_mutually_exclusive_sibling_read(self):
        """A sibling branch cannot make a dead loop result externally live."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            value = qmc.measure(qmc.qubit("value"))
            selector = qmc.measure(qmc.qubit("selector"))
            replacement = qmc.measure(qmc.qubit("replacement"))
            out = qmc.qubit("out")
            if selector:
                for _i in qmc.range(2):
                    value = ~replacement
            else:
                if value:
                    out = qmc.x(out)
            return qmc.measure(out)

        assert _sample_single(kernel) == 0

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

    def test_store_only_array_rebind_in_for_rejected(self):
        """A live-out Array rebind must not leak its traced body value."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            left: qmc.Vector[qmc.UInt],
            right: qmc.Vector[qmc.UInt],
        ) -> qmc.UInt:
            selected = left
            for _i in qmc.range(n):
                selected = right
            return selected[0]

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(
                kernel,
                bindings={"n": 2, "left": [1], "right": [2]},
            )

    def test_array_rebind_in_nested_runtime_if_rejected(self):
        """A runtime-if Array merge nested in a for loop stays fail-closed."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            left: qmc.Vector[qmc.UInt],
            right: qmc.Vector[qmc.UInt],
        ) -> qmc.UInt:
            predicate_qubit = qmc.qubit("predicate")
            predicate_qubit = qmc.x(predicate_qubit)
            predicate = qmc.measure(predicate_qubit)
            selected = left
            for _i in qmc.range(n):
                if predicate:
                    selected = right
            return selected[0]

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(
                kernel,
                bindings={"n": 2, "left": [1], "right": [2]},
            )

    def test_store_only_array_rebind_in_while_rejected(self):
        """A while body cannot expose a conditionally-updated Array binding."""

        @qmc.qkernel
        def kernel(
            left: qmc.Vector[qmc.UInt],
            right: qmc.Vector[qmc.UInt],
        ) -> qmc.UInt:
            condition_qubit = qmc.qubit("condition")
            condition_qubit = qmc.x(condition_qubit)
            condition = qmc.measure(condition_qubit)
            selected = left
            while condition:
                selected = right
                stop = qmc.qubit("stop")
                condition = qmc.measure(stop)
            return selected[0]

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"left": [1], "right": [2]})

    def test_zero_trip_dict_rebind_is_rejected(self):
        """An empty range cannot leak a traced replacement dictionary."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            left: qmc.Dict[qmc.UInt, qmc.UInt],
            right: qmc.Dict[qmc.UInt, qmc.UInt],
        ) -> qmc.UInt:
            selected = left
            for _i in qmc.range(n):
                selected = right
            return selected[0]

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(
                kernel,
                bindings={"left": {0: 1}, "right": {0: 2}},
                parameters=["n"],
            )

    def test_identity_scalar_with_dict_rebind_builds_then_rejects(self):
        """Identity publication preserves unrelated container validation."""

        @qmc.qkernel
        def kernel(
            bound: qmc.UInt,
            left: qmc.Dict[qmc.UInt, qmc.UInt],
            right: qmc.Dict[qmc.UInt, qmc.UInt],
        ) -> qmc.UInt:
            index = 1
            selected = left
            for _i in qmc.range(bound):
                index = index
                selected = right
            return selected[index]

        block = kernel.build()
        [loop] = _find_loops(block.operations)
        assert len(loop.region_args) == 1
        assert len(loop.loop_carried_rebinds) == 1
        record = loop.loop_carried_rebinds[0]
        assert isinstance(record.before, DictValue)
        assert isinstance(record.after, DictValue)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(
                kernel,
                bindings={"left": {1: 1}, "right": {1: 2}},
                parameters=["bound"],
            )

    def test_identity_scalar_used_by_residual_rebind_keeps_region_arg(self):
        """A residual endpoint keeps the identity RegionArg that defines it."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            selector = qmc.measure(qmc.qubit("selector"))
            base = 1
            flag = qmc.bit(False)
            for _i in qmc.range(2):
                if selector:
                    flag = base
                else:
                    flag = base
                base = base
            return flag

        block = kernel.build()
        [loop] = _find_loops(block.operations)
        assert len(loop.region_args) == 1
        assert len(loop.loop_carried_rebinds) == 1
        assert loop.loop_carried_rebinds[0].after.uuid == (
            loop.region_args[0].block_arg.uuid
        )

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_tuple_rebind_record_is_captured_and_rejected(self):
        """Tuple handles participate in residual loop-rebind validation."""

        left = Tuple(value=TupleValue(name="left"))
        right = Tuple(value=TupleValue(name="right"))
        with trace() as tracer:
            record_loop_rebinds(
                {"selected": left},
                {"selected": right},
                ("selected",),
                ("selected",),
            )

        assert len(tracer.loop_carried_rebinds) == 1
        record = tracer.loop_carried_rebinds[0]
        assert isinstance(record.before, TupleValue)
        assert isinstance(record.after, TupleValue)

        loop_var = Value(type=UIntType(), name="i")
        loop = ForOperation(
            operands=[
                Value(type=UIntType(), name="start").with_const(0),
                Value(type=UIntType(), name="stop").with_const(0),
                Value(type=UIntType(), name="step").with_const(1),
            ],
            loop_var="i",
            loop_var_value=loop_var,
            operations=[],
            loop_carried_rebinds=tracer.loop_carried_rebinds,
        )
        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            reject_loop_carried_classical_rebinds([loop])

    def test_store_only_bit_rebind_in_static_for_is_allowed(self):
        """A non-empty unrolled loop may surface its last measurement."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            bit = qmc.bit(False)
            for _i in qmc.range(n):
                q = qmc.qubit("q")
                q = qmc.x(q)
                bit = qmc.measure(q)
            return bit

        assert _sample_single(kernel, bindings={"n": 2}) == 1

    def test_store_only_bit_if_merge_in_static_for_is_allowed(self):
        """A final branch-local measurement merge owns a physical clbit."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            selector = qmc.measure(qmc.qubit("selector"))
            bit = qmc.bit(False)
            for _i in qmc.range(2):
                if selector:
                    bit = qmc.measure(qmc.qubit("zero"))
                else:
                    one = qmc.qubit("one")
                    one = qmc.x(one)
                    bit = qmc.measure(one)
            return bit

        assert _sample_single(kernel) == 1

    def test_store_only_measured_vector_element_in_static_for_is_allowed(self):
        """A final measured-vector element resolves to its parent clbit."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            bit = qmc.bit(False)
            for _i in qmc.range(2):
                values = qmc.qubit_array(2, "values")
                values[1] = qmc.x(values[1])
                measured = qmc.measure(values)
                bit = measured[1]
            return bit

        assert _sample_single(kernel) == 1

    def test_store_only_bit_rebind_with_symbolic_bound_is_rejected(self):
        """A possibly empty runtime loop cannot route the Bit initializer."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            bit = qmc.bit(False)
            for _i in qmc.range(n):
                q = qmc.qubit("q")
                bit = qmc.measure(q)
            return bit

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, parameters=["n"])

    def test_store_only_qfixed_rebind_in_for_rejected(self):
        """A QFixed rebind cannot replace its carrier register per iteration."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Float:
            initial = qmc.qubit_array(1, "initial")
            fixed = qmc.cast(initial, qmc.QFixed, int_bits=0)
            for _i in qmc.range(n):
                replacement = qmc.qubit_array(1, "replacement")
                fixed = qmc.cast(replacement, qmc.QFixed, int_bits=0)
            return qmc.measure(fixed)

        with pytest.raises(QubitRebindError, match="Loop-body quantum rebind"):
            _transpile(kernel, bindings={"n": 2})

    def test_dead_assignment_only_qubit_rebind_in_for_rejected(self):
        """A dead quantum rebind record must survive empty-loop stripping."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            _carried = qmc.qubit("carried")
            for _i in qmc.range(n):
                _carried = qmc.qubit("replacement")
            return qmc.measure(qmc.qubit("output"))

        with pytest.raises(QubitRebindError, match="Loop-body quantum rebind"):
            _transpile(kernel, parameters=["n"])

    def test_multi_trip_shrinking_quantum_view_rejected(self):
        """A shared base cannot hide a changing carried view region."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qubits = qmc.qubit_array(3, "shrinking")
            view = qubits
            for _i in qmc.range(2):
                view = view[1:]
                view[0] = qmc.x(view[0])
            return qmc.measure(view)

        with pytest.raises(QubitRebindError, match="Loop-body quantum rebind"):
            _transpile(kernel)

    def test_multi_trip_strided_quantum_view_rejected(self):
        """Nested stride changes cannot reuse the trace-time first view."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qubits = qmc.qubit_array(4, "strided")
            view = qubits
            for _i in qmc.range(2):
                view = view[1::2]
                view[0] = qmc.x(view[0])
            return qmc.measure(view)

        with pytest.raises(QubitRebindError, match="Loop-body quantum rebind"):
            _transpile(kernel)

    def test_mixed_type_store_only_rebind_supported_when_nonempty(self):
        """A guaranteed overwrite needs no Bit-to-UInt zero-trip join."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            value = qmc.bit(False)
            for _i in qmc.range(n):
                value = qmc.uint(1)
            return value

        assert _sample_single(kernel, bindings={"n": 2}) == 1

    def test_mixed_type_loop_variable_overwrite_is_rejected(self):
        """A loop-local nonconstant cannot materialize after unrolling."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            value = qmc.bit(False)
            for i in qmc.range(n):
                value = i
            return value

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"n": 2})

    def test_nested_translation_invariant_nonempty_overwrite_allowed(self):
        """An inner ``range(i, i + 1)`` is one trip for every outer index."""

        @qmc.qkernel
        def kernel() -> qmc.UInt:
            qmc.measure(qmc.qubit("segment_anchor"))
            value = qmc.bit(False)
            for i in qmc.range(2):
                for _j in qmc.range(i, i + 1):
                    value = qmc.uint(1)
            return value

        assert _sample_single(kernel) == 1

    def test_nested_singleton_outer_index_proves_inner_nonempty(self):
        """A singleton outer range supplies its concrete index to the inner."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            bit = qmc.bit(False)
            for i in qmc.range(1, 2):
                for _j in qmc.range(i):
                    q = qmc.qubit("q")
                    q = qmc.x(q)
                    bit = qmc.measure(q)
            return bit

        assert _sample_single(kernel) == 1

    def test_nested_multi_outer_domain_proves_every_inner_nonempty(self):
        """Every index in a positive outer domain makes ``range(i)`` run."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            bit = qmc.bit(False)
            for i in qmc.range(1, 3):
                for _j in qmc.range(i):
                    q = qmc.qubit("q")
                    q = qmc.x(q)
                    bit = qmc.measure(q)
            return bit

        assert _sample_single(kernel) == 1

    def test_nested_outer_domain_with_zero_inner_trip_is_rejected(self):
        """One zero-trip inner iteration prevents a nonempty-loop proof."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            bit = qmc.bit(False)
            for i in qmc.range(0, 2):
                for _j in qmc.range(i):
                    q = qmc.qubit("q")
                    bit = qmc.measure(q)
            return bit

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_nested_inner_loop_under_empty_outer_is_unreachable(self):
        """An empty static outer loop makes its unsafe inner body unreachable."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            qmc.measure(qmc.qubit("anchor"))
            bit = qmc.bit(False)
            for i in qmc.range(0):
                for _j in qmc.range(i):
                    q = qmc.qubit("q")
                    bit = qmc.measure(q)
            return bit

        assert _sample_single(kernel) == 0

    def test_mixed_type_structural_body_read_is_rejected(self):
        """A carried scalar embedded in a loop-body index is a real read."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            index = qmc.bit(False)
            q = qmc.qubit_array(2, "q")
            for _i in qmc.range(2):
                q[index] = qmc.x(q[index])
                index = qmc.uint(1)
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel)

    def test_symbolic_mixed_type_rebind_used_as_array_index_is_rejected(self):
        """Structural index references keep a zero-trip type join live."""

        @qmc.qkernel
        def kernel(count: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            index = qmc.bit(False)
            for _i in qmc.range(count):
                index = qmc.uint(1)
            q = qmc.qubit_array(2, "q")
            q[index] = qmc.x(q[index])
            return qmc.measure(q)

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, parameters=["count"])

    def test_range_target_read_after_loop_rejected(self):
        """A range target cannot silently escape as a post-loop ``None``."""
        with pytest.raises(SyntaxError, match="read after the loop"):

            @qmc.qkernel
            def kernel(n: qmc.UInt) -> qmc.UInt:
                last = qmc.uint(9)
                for last in qmc.range(n):
                    _unused = last
                return last

    def test_items_target_read_after_loop_rejected(self):
        """For-items key/value targets cannot escape the loop scope."""
        with pytest.raises(SyntaxError, match="read after the loop"):

            @qmc.qkernel
            def kernel(
                coeffs: qmc.Dict[qmc.UInt, qmc.UInt],
            ) -> qmc.UInt:
                result = qmc.uint(0)
                for key, value in qmc.items(coeffs):
                    result = value  # noqa: F841 - loop body must be non-empty
                return key

    def test_range_body_local_read_after_loop_rejected(self):
        """A zero-trip range cannot leak a trace-time body local."""
        with pytest.raises(SyntaxError, match="first defined inside a for loop"):

            @qmc.qkernel
            def kernel(n: qmc.UInt) -> qmc.UInt:
                for _i in qmc.range(n):
                    result = qmc.uint(5)
                return result

    def test_one_trip_conditional_body_local_escape_rejected(self):
        """One trip does not define a local assigned on one runtime path."""
        with pytest.raises(SyntaxError, match="first defined inside a for loop"):

            @qmc.qkernel
            def kernel() -> qmc.UInt:
                selector = qmc.measure(qmc.qubit("selector"))
                for _i in qmc.range(1):
                    if selector:
                        result = qmc.uint(5)
                return result

    def test_one_trip_definite_body_local_escape_rejected(self):
        """Definite assignment still lacks a formal post-loop result."""
        with pytest.raises(SyntaxError, match="first defined inside a for loop"):

            @qmc.qkernel
            def kernel() -> qmc.UInt:
                selector = qmc.measure(qmc.qubit("selector"))
                for _i in qmc.range(1):
                    if selector:
                        result = qmc.uint(5)
                    else:
                        result = qmc.uint(7)
                return result

    def test_one_trip_loop_variable_alias_escape_rejected(self):
        """A loop-variable alias would lose its scoped emit binding."""
        with pytest.raises(SyntaxError, match="first defined inside a for loop"):

            @qmc.qkernel
            def kernel() -> qmc.UInt:
                for i in qmc.range(1):
                    result = i
                return result

    def test_items_body_local_read_after_loop_rejected(self):
        """An empty items loop cannot leak a trace-time body local."""
        with pytest.raises(SyntaxError, match="first defined inside a for loop"):

            @qmc.qkernel
            def kernel(
                coeffs: qmc.Dict[qmc.UInt, qmc.UInt],
            ) -> qmc.UInt:
                for _key, value in qmc.items(coeffs):
                    result = value
                return result

    def test_while_body_local_read_after_loop_rejected(self):
        """An initially-false while cannot leak a trace-time body local."""
        with pytest.raises(SyntaxError, match="first defined inside a while loop"):

            @qmc.qkernel
            def kernel(bit: qmc.Bit) -> qmc.UInt:
                while bit:
                    result = qmc.uint(5)
                    bit = qmc.bit(False)
                return result

    def test_live_compile_time_branch_carry_in_while_rejected(self):
        """A selected classical update in a runtime while remains unsupported."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            total = qmc.uint(0)
            while bit:
                if flag == 1:
                    total = total + 1
                stop = qmc.qubit("stop")
                bit = qmc.measure(stop)
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"flag": 1})

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
        def kernel(n: qmc.UInt, start: qmc.UInt) -> tuple[qmc.Bit, qmc.UInt]:
            q = qmc.qubit("q")
            total = start
            for i in qmc.range(n):
                q = qmc.x(q)
                total = total + i
            return qmc.measure(q), total

        with pytest.raises(MultipleQuantumSegmentsError, match="carries classical"):
            _transpile(kernel, bindings={"n": 3}, parameters=["start"])

    def test_quantum_loop_carry_escaping_through_runtime_merge_rejected(self):
        """A runtime-if merge cannot hide an emit-only classical value.

        The statically bounded loop may remain an explicit carry or fold to a
        constant before segmentation. In the latter case the runtime merge is
        itself the emit-only value; both diagnostics reject the same unsafe
        attempt to surface it through the classical executor.
        """
        from qamomile.circuit.transpiler.segments import (
            MultipleQuantumSegmentsError,
        )

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> tuple[qmc.Bit, qmc.UInt]:
            pred_q = qmc.qubit("pred")
            pred_q = qmc.x(pred_q)
            bit = qmc.measure(pred_q)
            q = qmc.qubit("data")
            total = qmc.uint(0)
            for i in qmc.range(n):
                q = qmc.x(q)
                total = total + i
            if bit:
                q = qmc.x(q)
                out = total
            else:
                q = qmc.z(q)
                out = qmc.uint(99)
            return qmc.measure(q), out

        with pytest.raises(
            MultipleQuantumSegmentsError,
            match="carries classical|classical merges",
        ):
            _transpile(kernel, bindings={"n": 3})

    def test_quantum_if_classical_merge_escaping_to_output_executes(self):
        """A host-side SELECT resolves a quantum-controlled UInt merge."""

        @qmc.qkernel
        def kernel() -> tuple[qmc.Bit, qmc.UInt]:
            predicate = qmc.qubit("predicate")
            predicate = qmc.x(predicate)
            bit = qmc.measure(predicate)
            q = qmc.qubit("q")
            value = qmc.uint(0)
            if bit:
                q = qmc.x(q)
                value = qmc.uint(1)
            return qmc.measure(q), value

        assert _sample_single(kernel) == (1, 1)

    def test_quantum_loop_carry_escaping_through_output_index_rejected(self):
        """An array element's carried index cannot hide an output escape."""
        from qamomile.circuit.transpiler.segments import (
            MultipleQuantumSegmentsError,
        )

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            start: qmc.UInt,
            vals: qmc.Vector[qmc.UInt],
        ) -> tuple[qmc.Bit, qmc.UInt]:
            q = qmc.qubit("q")
            idx = start
            for _i in qmc.range(n):
                q = qmc.x(q)
                idx = idx + 1
            return qmc.measure(q), vals[idx]

        with pytest.raises(MultipleQuantumSegmentsError, match="carries classical"):
            _transpile(
                kernel,
                bindings={"n": 2, "vals": [10, 20, 30]},
                parameters=["start"],
            )

    def test_quantum_loop_carry_escaping_to_expval_index_rejected(self):
        """An ExpvalSegment cannot consume an emit-only carried index."""
        import qamomile.observable as qm_o
        from qamomile.circuit.transpiler.segments import (
            MultipleQuantumSegmentsError,
        )

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            start: qmc.UInt,
            obs: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            idx = start
            for _i in qmc.range(n):
                q[0] = qmc.x(q[0])
                idx = idx + 1
            return qmc.expval(q[idx], obs)

        with pytest.raises(MultipleQuantumSegmentsError, match="carries classical"):
            _transpile(
                kernel,
                bindings={"n": 2, "obs": qm_o.Z(0)},
                parameters=["start"],
            )

    def test_while_store_only_same_type_carry_rejected(self):
        """A while body's store-only same-type scalar overwrite is rejected.

        While bodies get no RegionArg promotion, and the always-live
        zero-trip path means no trip-count proof can surface the traced
        body value. Accepting this shape compiles and then fails at
        sampling with an unbound backend parameter (the traced
        ``x + 1`` lives only in the per-iteration emit scope).
        """

        @qmc.qkernel
        def kernel(x: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            bit = qmc.measure(q)
            total = qmc.uint(0)
            while bit:
                q2 = qmc.qubit("f")
                bit = qmc.measure(q2)
                total = x + 1
            return total

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, parameters=["x"])

    def test_while_second_variable_aliasing_condition_rejected(self):
        """A second variable rebound to the condition measurement is rejected.

        ``saved = bit`` shares the condition-update's ``after`` UUID but
        not its ``before``; exempting it alongside the condition pair
        would bind the post-loop read of ``saved`` to the final in-loop
        measurement, silently diverging from the pre-loop snapshot
        Python guarantees on the always-live zero-trip path.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            s = qmc.qubit("s")
            s = qmc.x(s)
            saved = qmc.measure(s)
            q = qmc.qubit("q")
            bit = qmc.measure(q)
            while bit:
                q2 = qmc.qubit("f")
                bit = qmc.measure(q2)
                saved = bit
            return saved

        with pytest.raises(ValidationError, match=LOOP_CARRIED):
            _transpile(kernel, bindings={"dummy": 0})

    def test_while_plain_snapshot_aliasing_condition_rejected(self):
        """A plain-bool snapshot cannot alias the updated condition clbit.

        The initial condition measures zero, so the while body never runs and
        Python returns the ``True`` initializer. If ``before_synthesized`` is
        mistaken for proof that this is the condition variable itself, emit
        aliases ``saved`` to the measured-zero condition clbit and silently
        returns false instead.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Bit:
            condition_q = qmc.qubit("condition")
            bit = qmc.measure(condition_q)
            saved = True
            while bit:
                update_q = qmc.qubit("update")
                bit = qmc.measure(update_q)
                saved = bit
            return saved

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

    def test_overwritten_loop_target_is_not_treated_as_post_loop_read(self):
        """A fresh assignment after a loop kills the target's loop value."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            for i in qmc.range(n):
                _unused = i
            i = qmc.uint(7)
            return i

        assert _sample_single(kernel, bindings={"n": 0}) == 7

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

    def test_one_trip_region_changing_quantum_view_allowed(self):
        """A single static iteration has no backedge to misroute a view."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qubits = qmc.qubit_array(3, "one_trip_view")
            view = qubits
            for _i in qmc.range(1):
                view = view[1:]
                view[0] = qmc.x(view[0])
            return qmc.measure(view)

        assert _sample_single(kernel) == (1, 0)

    def test_multi_trip_identity_quantum_view_allowed(self):
        """A full re-slice preserves ordered physical coverage each trip."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qubits = qmc.qubit_array(3, "identity_view")
            view = qubits
            for _i in qmc.range(2):
                view = view[:]
                view[0] = qmc.x(view[0])
            return qmc.measure(view)

        assert _sample_single(kernel) == (0, 0, 0)

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

    def test_dead_compile_time_branch_removes_while_carry(self):
        """A folded-away while update leaves no unsupported carry slot."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            total = qmc.uint(0)
            while bit:
                if flag == 1:
                    total = total + 1
                stop = qmc.qubit("stop")
                bit = qmc.measure(stop)
            return total

        assert _sample_single(kernel, bindings={"flag": 0}) == 0

    def test_dead_while_carry_inside_runtime_if_merges_initializer(self):
        """Nested identity-carry removal preserves the runtime branch yield."""

        @qmc.qkernel
        def kernel(flag: qmc.UInt) -> qmc.UInt:
            predicate_qubit = qmc.qubit("predicate")
            predicate_qubit = qmc.x(predicate_qubit)
            predicate = qmc.measure(predicate_qubit)
            condition_qubit = qmc.qubit("condition")
            condition_qubit = qmc.x(condition_qubit)
            condition = qmc.measure(condition_qubit)
            total = qmc.uint(0)
            if predicate:
                while condition:
                    if flag == 1:
                        total = total + 1
                    stop = qmc.qubit("stop")
                    condition = qmc.measure(stop)
            return total

        assert _sample_single(kernel, bindings={"flag": 0}) == 0

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
            """Return the native Python sum below ``n``.

            Args:
                n (int): Exclusive upper bound.

            Returns:
                int: Sum of ``range(n)``.
            """
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

    def test_carried_loop_inside_runtime_if_merges_by_branch(self):
        """A carried loop under a runtime if merges per-branch totals.

        The loop's carry unrolls to a pure classical sum inside the
        taken branch, and the runtime merge muxes the branch total
        against the pre-if initializer on the measured bit. Both
        deterministic branch outcomes pin the mux: a measured |1>
        selects the accumulated 0 + 1 + 2 == 3, a measured |0> keeps
        the initializer 0.
        """

        @qmc.qkernel
        def taken(flag: qmc.UInt, n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            total = qmc.uint(0)
            if bit:
                for i in qmc.range(n):
                    if flag == 1:
                        total = total + i
            return total

        assert _sample_single(taken, bindings={"flag": 1, "n": 3}) == 3

        @qmc.qkernel
        def skipped(flag: qmc.UInt, n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            bit = qmc.measure(q)
            total = qmc.uint(0)
            if bit:
                for i in qmc.range(n):
                    if flag == 1:
                        total = total + i
            return total

        assert _sample_single(skipped, bindings={"flag": 1, "n": 3}) == 0

    def test_outer_value_yield_survives_losing_branch_pruning(self):
        """A losing merge source read only as a loop-invariant yield stays live.

        The compile-time-false branch marks its merge source (``c``)
        dead, and the same value is the loop's region yield (``x = c``,
        the loop-invariant overwrite shape) — a read that
        ``genuine_input_values`` deliberately hides from read-based
        analyses. Dead-op elimination must still keep the producer
        alive: deleting it compiles cleanly and then fails at every
        execution with "Value not found in context or results". The
        runtime parameter keeps ``c = a + 1`` from constant-folding
        away before the pruning runs.
        """

        @qmc.qkernel
        def kernel(
            n: qmc.UInt, a: qmc.UInt, flag: qmc.UInt
        ) -> tuple[qmc.Bit, qmc.UInt, qmc.UInt]:
            q = qmc.qubit("q")
            b = qmc.measure(q)
            c = a + 1
            if flag > 0:
                y = 2 + a
            else:
                y = c
            x = qmc.uint(0)
            for _i in qmc.range(n):
                _t = x + 1
                x = c
            return b, x, y

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(
            kernel, bindings={"flag": 1, "n": 2}, parameters=["a"]
        )
        result = executable.sample(
            transpiler.executor(), shots=100, bindings={"a": 5}
        ).result()
        assert result.results == [((0, 6, 7), 100)]


# ---------------------------------------------------------------------------
# Cross-backend execution: carried scalars drive identical circuits everywhere
# ---------------------------------------------------------------------------


def _make_transpiler(backend: str):
    """Build the requested backend transpiler, skipping if the SDK is absent.

    Args:
        backend (str): One of ``"qiskit"``, ``"quri_parts"``, ``"cudaq"``.

    Returns:
        Any: The backend transpiler instance.
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
    """RegionArg records round-trip through qkernel serialization."""

    def _build_block_with_region_args(self):
        """Build a block whose ForOperation carries region args.

        Returns:
            Block: Affine block built with a symbolic loop bound.
        """

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.UInt:
            q = qmc.qubit("q")
            qmc.measure(q)
            total = qmc.uint(0)
            for i in qmc.range(n):
                total = total + i
            return total

        return kernel

    def test_region_args_roundtrip(self):
        """Region args survive a serialize/deserialize cycle."""
        from qamomile.circuit.serialization import (
            deserialize,
            serialize,
        )

        kernel = self._build_block_with_region_args()
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(kernel.block)

        restored = transpiler.inline(deserialize(serialize(kernel)).block)

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
                original_values = (
                    orig_arg.init,
                    orig_arg.block_arg,
                    orig_arg.yielded,
                    orig_arg.result,
                )
                restored_values = (
                    rest_arg.init,
                    rest_arg.block_arg,
                    rest_arg.yielded,
                    rest_arg.result,
                )
                assert [value.type for value in original_values] == [
                    value.type for value in restored_values
                ]
                for left_index in range(len(original_values)):
                    for right_index in range(len(original_values)):
                        assert (
                            original_values[left_index].uuid
                            == original_values[right_index].uuid
                        ) == (
                            restored_values[left_index].uuid
                            == restored_values[right_index].uuid
                        )
                        assert (
                            original_values[left_index].logical_id
                            == original_values[right_index].logical_id
                        ) == (
                            restored_values[left_index].logical_id
                            == restored_values[right_index].logical_id
                        )
                assert rest_arg.result in rest.results
        assert found_region_args, "fixture must produce at least one region arg"


class TestRebindRecordSerialization:
    """LoopCarriedRebind records round-trip with their qkernel."""

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

        return kernel

    def test_records_roundtrip(self):
        """Records survive a serialize/deserialize cycle."""
        from qamomile.circuit.serialization import (
            deserialize,
            serialize,
        )

        kernel = self._build_block_with_records()
        transpiler = QiskitTranspiler()
        affine = transpiler.inline(kernel.block)

        restored = transpiler.inline(deserialize(serialize(kernel)).block)

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
                assert orig_rec.before.type == rest_rec.before.type
                assert orig_rec.after.type == rest_rec.after.type
                assert (orig_rec.before.uuid == orig_rec.after.uuid) == (
                    rest_rec.before.uuid == rest_rec.after.uuid
                )
                assert (orig_rec.before.logical_id == orig_rec.after.logical_id) == (
                    rest_rec.before.logical_id == rest_rec.after.logical_id
                )
                assert orig_rec.before_synthesized == rest_rec.before_synthesized
        assert found_records, "fixture must produce at least one rebind record"
