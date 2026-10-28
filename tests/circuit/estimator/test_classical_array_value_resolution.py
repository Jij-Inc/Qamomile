"""Resource-estimator tests for classical array element values."""

import pytest

import qamomile.circuit as qm


def test_fresh_bit_array_element_resolves_to_false() -> None:
    """A fresh Bit-array element keeps its false initialization."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Condition one gate on a fresh Bit-array element."""
        bits = qm.bit_array(3)
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.parameters == {}


def test_stored_bit_array_element_resolves_to_stored_input() -> None:
    """A Bit-array read follows the latest stored scalar value."""

    @qm.qkernel
    def circuit(value: qm.Bit) -> qm.Qubit:
        """Condition one gate on an input stored into a Bit array.

        Args:
            value (qm.Bit): Scalar value written into the array.

        Returns:
            qm.Qubit: Target qubit after the conditional gate.
        """
        bits = qm.bit_array(3)
        bits[0] = value
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    false_estimate = circuit.estimate_resources(inputs={"value": False})
    true_estimate = circuit.estimate_resources(inputs={"value": True})

    assert false_estimate.gates.total == 0
    assert false_estimate.parameters == {}
    assert true_estimate.gates.total == 1
    assert true_estimate.parameters == {}


def test_zero_trip_store_loop_preserves_pre_loop_bit_array_value() -> None:
    """An unreachable measured store cannot replace the pre-loop value."""

    @qm.qkernel
    def circuit(iterations: qm.UInt) -> qm.Qubit:
        """Store a measurement into a Bit array inside a possibly empty loop.

        Args:
            iterations (qm.UInt): Number of store iterations.

        Returns:
            qm.Qubit: Target qubit after the post-loop conditional gate.
        """
        bits = qm.bit_array(1)
        source = qm.qubit("source")
        for _index in qm.range(iterations):
            bits[0] = qm.measure(source)
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    zero_trip = circuit.estimate_resources(inputs={"iterations": 0})
    one_trip = circuit.estimate_resources(inputs={"iterations": 1})
    symbolic = circuit.estimate_resources()

    assert zero_trip.gates.total == 0
    assert zero_trip.measurements.total == 0
    assert zero_trip.parameters == {}
    assert one_trip.gates.total == 1
    assert one_trip.measurements.total == 1
    assert one_trip.parameters == {}
    assert set(symbolic.parameters) == {"iterations"}
    assert symbolic.substitute(iterations=0).gates.total == 0
    assert symbolic.substitute(iterations=1).gates.total == 1


@pytest.mark.parametrize(
    ("outer_iterations", "inner_iterations", "expected"),
    [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ],
)
def test_nested_zero_trip_store_loops_preserve_pre_loop_bit_array_value(
    outer_iterations: int,
    inner_iterations: int,
    expected: int,
) -> None:
    """Every enclosing loop must execute before its measured store is visible.

    Args:
        outer_iterations (int): Concrete outer-loop trip count.
        inner_iterations (int): Concrete inner-loop trip count.
        expected (int): Expected gate and measurement count.
    """

    @qm.qkernel
    def circuit(outer: qm.UInt, inner: qm.UInt) -> qm.Qubit:
        """Store a measurement inside two possibly empty loops.

        Args:
            outer (qm.UInt): Number of outer-loop iterations.
            inner (qm.UInt): Number of inner-loop iterations.

        Returns:
            qm.Qubit: Target qubit after the post-loop conditional gate.
        """
        bits = qm.bit_array(1)
        source = qm.qubit("source")
        for _outer_index in qm.range(outer):
            for _inner_index in qm.range(inner):
                bits[0] = qm.measure(source)
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    estimate = circuit.estimate_resources(
        inputs={"outer": outer_iterations, "inner": inner_iterations}
    )

    assert estimate.gates.total == expected
    assert estimate.measurements.total == expected
    assert estimate.parameters == {}


@pytest.mark.parametrize(
    ("iterations", "enabled", "expected"),
    [(0, False, 0), (0, True, 0), (1, False, 0), (1, True, 1)],
)
def test_loop_and_compile_if_guards_compose_for_bit_array_store(
    iterations: int,
    enabled: bool,
    expected: int,
) -> None:
    """A nested store is visible only when its loop and branch both execute.

    Args:
        iterations (int): Concrete enclosing-loop trip count.
        enabled (bool): Compile-time branch selector.
        expected (int): Expected gate and measurement count.
    """

    @qm.qkernel
    def circuit(count: qm.UInt, flag: qm.Bit) -> qm.Qubit:
        """Store a measurement behind a loop-nested ordinary conditional.

        Args:
            count (qm.UInt): Number of loop iterations.
            flag (qm.Bit): Compile-time store selector.

        Returns:
            qm.Qubit: Target qubit after the post-loop conditional gate.
        """
        bits = qm.bit_array(1)
        source = qm.qubit("source")
        for _index in qm.range(count):
            if flag:
                bits[0] = qm.measure(source)
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    estimate = circuit.estimate_resources(inputs={"count": iterations, "flag": enabled})

    assert estimate.gates.total == expected
    assert estimate.measurements.total == expected
    assert estimate.parameters == {}


def test_zero_trip_while_store_preserves_pre_loop_bit_array_value() -> None:
    """A zero-trip while cannot expose its body-local measured store."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Store a measurement in a runtime while body."""
        active = qm.measure(qm.qubit("control"))
        bits = qm.bit_array(1)
        source = qm.qubit("source")
        while active:
            bits[0] = qm.measure(source)
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    symbolic = circuit.estimate_resources()
    zero_trip = symbolic.substitute(**{"|while|": 0})
    one_trip = symbolic.substitute(**{"|while|": 1})

    assert set(symbolic.parameters) == {"|while|"}
    assert zero_trip.gates.total == 0
    assert zero_trip.measurements.total == 1
    assert one_trip.gates.total == 1
    assert one_trip.measurements.total == 2


def test_bit_array_state_resolves_through_views() -> None:
    """Fresh, stored, and nested sliced array states retain root contents."""

    @qm.qkernel
    def fresh_view() -> qm.Qubit:
        """Read a fresh false element through one view."""
        view = qm.bit_array(4)[1:3]
        target = qm.qubit("target")
        if view[0]:
            target = qm.x(target)
        return target

    @qm.qkernel
    def stored_view() -> qm.Qubit:
        """Read a stored true element through one view."""
        bits = qm.bit_array(4)
        bits[2] = True
        view = bits[1:4]
        target = qm.qubit("target")
        if view[1]:
            target = qm.x(target)
        return target

    @qm.qkernel
    def nested_view() -> qm.Qubit:
        """Read a stored true root element through nested strided views."""
        bits = qm.bit_array(6)
        bits[4] = True
        view = bits[0:6:2][1:3]
        target = qm.qubit("target")
        if view[1]:
            target = qm.x(target)
        return target

    assert fresh_view.estimate_resources().gates.total == 0
    assert stored_view.estimate_resources().gates.total == 1
    assert nested_view.estimate_resources().gates.total == 1


def test_bit_array_state_resolves_across_qkernel_call() -> None:
    """A callee element read uses the caller's initialized or stored state."""

    @qm.qkernel
    def inspect_first(bits: qm.Vector[qm.Bit]) -> qm.Qubit:
        """Condition one gate on the first array element.

        Args:
            bits (qm.Vector[qm.Bit]): Caller-provided classical array.

        Returns:
            qm.Qubit: Conditionally updated target qubit.
        """
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    @qm.qkernel
    def fresh_caller() -> qm.Qubit:
        """Pass a fresh false array through one call boundary."""
        return inspect_first(qm.bit_array(1))

    @qm.qkernel
    def stored_caller() -> qm.Qubit:
        """Pass a stored true array through one call boundary."""
        bits = qm.bit_array(1)
        bits[0] = True
        return inspect_first(bits)

    assert fresh_caller.estimate_resources().gates.total == 0
    assert stored_caller.estimate_resources().gates.total == 1


def test_qkernel_array_output_preserves_stored_state_per_invocation() -> None:
    """Each array-returning invocation freezes its own stored scalar value."""

    @qm.qkernel
    def set_first(
        bits: qm.Vector[qm.Bit],
        value: qm.Bit,
    ) -> qm.Vector[qm.Bit]:
        """Store a call-specific scalar and return the updated array.

        Args:
            bits (qm.Vector[qm.Bit]): Input array.
            value (qm.Bit): Scalar stored at index zero.

        Returns:
            qm.Vector[qm.Bit]: Updated array.
        """
        bits[0] = value
        return bits

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Read two results after both calls have completed."""
        true_result = set_first(qm.bit_array(1), qm.bit(True))
        false_result = set_first(qm.bit_array(1), qm.bit(False))
        true_target = qm.qubit("true_target")
        false_target = qm.qubit("false_target")
        if true_result[0]:
            true_target = qm.x(true_target)
        if false_result[0]:
            false_target = qm.x(false_target)
        return true_target, false_target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.parameters == {}


def test_qkernel_array_output_preserves_untouched_caller_slots() -> None:
    """A callee Store retains untouched slots from each caller snapshot."""

    @qm.qkernel
    def set_first(bits: qm.Vector[qm.Bit]) -> qm.Vector[qm.Bit]:
        """Set index zero and return the updated array.

        Args:
            bits (qm.Vector[qm.Bit]): Input array.

        Returns:
            qm.Vector[qm.Bit]: Updated array.
        """
        bits[0] = True
        return bits

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Read stored and untouched slots from two call results."""
        first_bits = qm.bit_array(2)
        first_bits[1] = True
        first_result = set_first(first_bits)
        second_result = set_first(qm.bit_array(2))

        first_zero = qm.qubit("first_zero")
        first_one = qm.qubit("first_one")
        second_zero = qm.qubit("second_zero")
        second_one = qm.qubit("second_one")
        if first_result[0]:
            first_zero = qm.x(first_zero)
        if first_result[1]:
            first_one = qm.x(first_one)
        if second_result[0]:
            second_zero = qm.x(second_zero)
        if second_result[1]:
            second_one = qm.x(second_one)
        return first_zero, first_one, second_zero, second_one

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.parameters == {}


def test_qkernel_array_output_preserves_untouched_slice_slots() -> None:
    """A callee Store retains untouched elements of a caller array view."""

    @qm.qkernel
    def set_first(bits: qm.Vector[qm.Bit]) -> qm.Vector[qm.Bit]:
        """Set the first element of a caller-provided view.

        Args:
            bits (qm.Vector[qm.Bit]): Input array view.

        Returns:
            qm.Vector[qm.Bit]: Updated view.
        """
        bits[0] = True
        return bits

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Read the stored and untouched elements after one helper call."""
        bits = qm.bit_array(5)
        view = set_first(bits[1:5:2])
        stored_target = qm.qubit("stored_target")
        untouched_target = qm.qubit("untouched_target")
        if view[0]:
            stored_target = qm.x(stored_target)
        if view[1]:
            untouched_target = qm.x(untouched_target)
        return stored_target, untouched_target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.parameters == {}
