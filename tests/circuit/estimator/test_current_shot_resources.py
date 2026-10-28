"""Quantum resource bounds for device-supplied shot context."""

import json

import pytest

import qamomile.circuit as qm
from qamomile.circuit.estimator._wire import (
    resource_estimate_from_wire,
    resource_estimate_to_wire,
)
from qamomile.circuit.frontend.composite_gate import configure_composite


@qm.qkernel
def _shot_value() -> qm.UInt:
    """Return the current repetition index."""
    return qm.get_current_shot()


configure_composite(_shot_value)


@qm.qkernel
def _same_shot() -> qm.Qubit:
    """Compare reads across a callable boundary."""
    a = qm.get_current_shot()
    b = _shot_value()
    c = _shot_value()
    q = qm.qubit("q")
    if a != b:
        q = qm.x(q)
    if b != c:
        q = qm.h(q)
    return q


@qm.qkernel
def _shot_branch() -> qm.Qubit:
    """Select between fixed quantum costs."""
    q = qm.qubit("q")
    if qm.get_current_shot() % 2 == 0:
        q = qm.x(q)
    else:
        q = qm.h(q)
        q = qm.h(q)
        q = qm.h(q)
    return q


@qm.qkernel
def _shot_modulo_loop() -> qm.Qubit:
    """Repeat a fixed body by a bounded remainder."""
    count = qm.get_current_shot() % 4 + 2
    q = qm.qubit("q")
    for _ in qm.range(count):
        q = qm.h(q)
    return q


@qm.qkernel
def _shot_table_loop(counts: qm.Vector[qm.UInt]) -> qm.Qubit:
    """Select a repetition count from a concrete input table."""
    count = counts[qm.get_current_shot() % 2]
    q = qm.qubit("q")
    for _ in qm.range(count):
        q = qm.h(q)
    return q


@qm.qkernel
def _unbounded_shot_loop() -> qm.Qubit:
    """Require a resource bound on the unrestricted device index."""
    q = qm.qubit("q")
    for _ in qm.range(qm.get_current_shot()):
        q = qm.h(q)
    return q


@qm.qkernel
def _shot_table_angle(angles: qm.Vector[qm.Float]) -> qm.Qubit:
    """Select an angle without changing the logical gate count."""
    q = qm.qubit("q")
    return qm.ry(q, angles[qm.get_current_shot()])


@qm.qkernel
def _correlated_shot_loop(counts: qm.Vector[qm.UInt]) -> qm.Qubit:
    """Keep the expensive short path when bounding a correlated loop."""
    residue = qm.get_current_shot() % 2
    count = counts[residue]
    q = qm.qubit("q")
    for _ in qm.range(count):
        if residue == 0:
            for _j in qm.range(10):
                q = qm.h(q)
        else:
            q = qm.x(q)
    return q


@qm.qkernel
def _nested_shot_loops() -> qm.Qubit:
    """Bound related inner and outer counts independently."""
    residue = qm.get_current_shot() % 4
    outer = residue + 1
    inner = 4 - residue
    q = qm.qubit("q")
    for _i in qm.range(outer):
        for _j in qm.range(inner):
            q = qm.h(q)
    return q


@qm.qkernel
def _ordinary_while_with_shot() -> qm.Qubit:
    """Keep the established symbolic while count beside a context read."""
    _unused = qm.get_current_shot()
    active = qm.measure(qm.qubit("predicate"))
    q = qm.qubit("q")
    while active:
        q = qm.h(q)
        active = qm.measure(qm.qubit("next_predicate"))
    return q


@qm.qkernel
def _shifted_angle(angles: qm.Vector[qm.Float], shift: qm.UInt) -> qm.Qubit:
    """Retain public validity requirements for a shifted shot index."""
    q = qm.qubit("q")
    return qm.ry(q, angles[qm.get_current_shot() + shift])


@qm.qkernel
def _negative_angle(angles: qm.Vector[qm.Float]) -> qm.Qubit:
    """Address a provably negative slot for every shot index."""
    q = qm.qubit("q")
    return qm.ry(q, angles[0 - qm.get_current_shot() - 1])


@qm.qkernel
def _mutated_counts(counts: qm.Vector[qm.UInt], index: qm.UInt) -> qm.Qubit:
    """Do not reuse the original input after an unresolved array store."""
    counts[index] = qm.uint(100)
    q = qm.qubit("q")
    for _ in qm.range(counts[qm.get_current_shot() % 2]):
        q = qm.h(q)
    return q


def test_shot_read_has_no_quantum_cost_or_public_parameter() -> None:
    """Reading context does not allocate a qubit or publish an input."""
    estimate = _shot_value.estimate_resources()
    assert estimate.gates.total == 0
    assert estimate.measurements.total == 0
    assert estimate.depth.depth == 0
    assert estimate.parameters == {}
    assert estimate.assumptions == ()


def test_repeated_reads_share_identity_across_calls() -> None:
    """A callee cannot observe a different index in the same repetition."""
    estimate = _same_shot.estimate_resources()
    assert estimate.gates.total == 0
    assert estimate.parameters == {}
    assert estimate.assumptions == ()


def test_preserved_identity_helper_does_not_assume_uint_arithmetic() -> None:
    """Passing a context value through a helper adds no arithmetic premise."""

    @qm.qkernel
    def identity(value: qm.UInt) -> qm.UInt:
        """Return the supplied value without arithmetic."""
        return value

    configure_composite(identity)

    @qm.qkernel
    def circuit() -> qm.UInt:
        """Pass the device index through a preserved helper."""
        return identity(qm.get_current_shot())

    estimate = circuit.estimate_resources()
    assert estimate.gates.total == 0
    assert estimate.measurements.total == 0
    assert estimate.depth.depth == 0
    assert estimate.parameters == {}
    assert estimate.assumptions == ()


def test_shot_uint_arithmetic_serializes_its_no_overflow_assumption() -> None:
    """A mathematical simplification discloses its native overflow premise."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a count that becomes zero under mathematical integer arithmetic."""
        count = (qm.get_current_shot() * 3) % 3
        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources(trace=True)
    assert estimate.gates.total == 0
    assert estimate.parameters == {}
    arithmetic_assumptions = [
        item for item in estimate.assumptions if item.source == "shot UInt arithmetic"
    ]
    assert len(arithmetic_assumptions) == 1
    assert "nonnegative mathematical integers" in arithmetic_assumptions[0].message
    assert "without backend integer overflow" in arithmetic_assumptions[0].message
    assert "without backend integer overflow" in json.dumps(estimate.to_dict())
    restored = resource_estimate_from_wire(resource_estimate_to_wire(estimate))
    assert restored.assumptions == estimate.assumptions
    assert restored.to_dict() == estimate.to_dict()


def test_shot_branch_uses_a_conservative_bound() -> None:
    """All shot-dependent fixed-cost branches contribute to the upper bound."""
    estimate = _shot_branch.estimate_resources()
    assert estimate.gates.total == 3
    assert estimate.measurements.total == 0
    assert estimate.depth.depth == 3
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_modulo_arithmetic_bounds_a_fixed_body_loop() -> None:
    """Simple arithmetic on a remainder retains a finite resource bound."""
    estimate = _shot_modulo_loop.estimate_resources()
    assert estimate.gates.total == 5
    assert estimate.depth.depth == 5
    assert estimate.parameters == {}


def test_finite_table_bounds_a_fixed_body_loop() -> None:
    """A bound input table limits a shot-selected repetition count."""
    estimate = _shot_table_loop.estimate_resources(inputs={"counts": [2, 5]})
    assert estimate.gates.total == 5
    assert estimate.parameters == {}


def test_unbounded_shot_resource_structure_remains_an_error() -> None:
    """The device index must never become a user-substitutable count."""
    with pytest.raises(NotImplementedError, match="shot|runtime-derived"):
        _unbounded_shot_loop.estimate_resources()


def test_shot_selected_angle_has_fixed_gate_resources() -> None:
    """Array validity does not turn an angle into a gate-count input."""
    estimate = _shot_table_angle.estimate_resources(inputs={"angles": [0.2, 0.4]})
    assert estimate.gates.total == 1
    assert estimate.parameters == {}
    assert any("array accesses" in item.message for item in estimate.assumptions)


def test_maximum_count_does_not_select_only_the_cheapest_body() -> None:
    """Bounding loop counts must retain the expensive correlated path."""
    estimate = _correlated_shot_loop.estimate_resources(inputs={"counts": [2, 5]})
    assert estimate.gates.total >= 20
    assert estimate.parameters == {}


def test_nested_finite_counts_return_a_safe_bound() -> None:
    """Independent finite bounds may overestimate but never undercount."""
    estimate = _nested_shot_loops.estimate_resources()
    assert estimate.gates.total >= 6
    assert estimate.parameters == {}


def test_ordinary_while_keeps_its_existing_semantic_parameter() -> None:
    """A context read does not change the ordinary while-loop contract."""
    estimate = _ordinary_while_with_shot.estimate_resources()
    assert set(estimate.parameters) == {"|while|"}
    assert estimate.substitute(**{"|while|": 3}).gates.total == 3


@pytest.mark.parametrize(
    "kernel", [_shot_branch, _shot_modulo_loop, _nested_shot_loops]
)
def test_trace_and_serialization_keep_shot_values_private(kernel) -> None:
    """Explanations cannot expose a device index as a resource parameter."""
    first = kernel.estimate_resources(trace=True)
    second = kernel.estimate_resources(trace=True)
    assert first.parameters == second.parameters == {}
    assert first.to_dict() == second.to_dict()
    payload = json.dumps(first.to_dict())
    assert "current_shot" not in payload
    assert "shot_remainder" not in payload
    restored = resource_estimate_from_wire(resource_estimate_to_wire(first))
    assert restored.to_dict() == first.to_dict()


def test_known_invalid_shot_array_access_is_not_hidden() -> None:
    """A validity assumption cannot make an impossible array access valid."""
    with pytest.raises(
        (ValueError, NotImplementedError), match="invalid|dimension|bounds|UInt"
    ):
        _negative_angle.estimate_resources(inputs={"angles": [0.2]})
    with pytest.raises(ValueError, match="dimension"):
        _shot_table_angle.estimate_resources(inputs={"angles": []})
    symbolic = _shot_table_angle.estimate_resources()
    with pytest.raises(ValueError, match="dimension"):
        symbolic.substitute(angles_dim0=0)


def test_public_index_shift_constraints_survive_substitution() -> None:
    """A later public input cannot invalidate an assumed context access."""
    symbolic = _shifted_angle.estimate_resources(inputs={"angles": [0.2, 0.4]})
    with pytest.raises(ValueError, match="margin"):
        symbolic.substitute(shift=2)


def test_unresolved_array_updates_never_reuse_stale_input_values() -> None:
    """An unresolved store may require refusal instead of a smaller bound."""
    with pytest.raises(NotImplementedError, match="shot|runtime-derived"):
        _mutated_counts.estimate_resources(inputs={"counts": [1, 1]})


def test_large_remainder_domain_is_explicitly_unsupported() -> None:
    """Finite candidate expansion has a fixed work limit."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use more remainder alternatives than the bounded analysis supports."""
        q = qm.qubit("q")
        for _ in qm.range(qm.get_current_shot() % 17):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot"):
        circuit.estimate_resources()


def test_shot_bounded_loop_cannot_publish_a_definite_array_exit() -> None:
    """A maximum trip count cannot erase a costly zero-trip continuation."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Keep a true bit when the shot-dependent loop executes zero times."""
        bits = qm.bit_array(1)
        bits[0] = qm.bit(True)
        for _ in qm.range(qm.get_current_shot() % 2):
            bits[0] = qm.bit(False)
        q = qm.qubit("q")
        if bits[0]:
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent loop"):
        circuit.estimate_resources()


def test_nonlinear_angle_shift_preserves_its_public_endpoint() -> None:
    """A provably monotone index retains necessary public validity bounds."""

    @qm.qkernel
    def circuit(angles: qm.Vector[qm.Float], shift: qm.UInt) -> qm.Qubit:
        """Use a nonnegative square plus a public offset as an array index."""
        shot = qm.get_current_shot()
        q = qm.qubit("q")
        return qm.ry(q, angles[shot * shot + shift])

    symbolic = circuit.estimate_resources(inputs={"angles": [0.2, 0.4]})
    with pytest.raises(ValueError, match="margin"):
        symbolic.substitute(shift=2)


def test_shot_bounded_loop_cannot_definitely_consume_a_captured_qubit() -> None:
    """The zero-trip path keeps captured qubits live beside later allocation."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Consume a captured array slot only on a nonempty shot loop."""
        controls = qm.qubit_array(3, "controls")
        for index in qm.range(qm.get_current_shot() % 2):
            qm.measure(controls[index])
        return qm.qubit_array(1, "later")

    with pytest.raises(NotImplementedError, match="shot-dependent loop"):
        circuit.estimate_resources()


def test_shot_bounded_loop_checks_selected_helper_consumption() -> None:
    """A preserved helper cannot hide a destructive body operation."""

    @qm.qkernel
    def consume(q: qm.Qubit) -> qm.Bit:
        """Consume the supplied qubit."""
        return qm.measure(q)

    configure_composite(consume)

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Call the consuming helper on a captured array element."""
        controls = qm.qubit_array(3, "controls")
        for index in qm.range(qm.get_current_shot() % 2):
            consume(controls[index])
        return qm.qubit_array(1, "later")

    with pytest.raises(NotImplementedError, match="shot-dependent loop"):
        circuit.estimate_resources()


def test_shot_reads_do_not_add_measurement_readiness_edges() -> None:
    """Two independent shot branches may overlap after an unrelated measurement."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Run independent quantum work selected by the same shot context."""
        source = qm.qubit("source")
        source = qm.h(source)
        qm.measure(source)
        left = qm.qubit("left")
        right = qm.qubit("right")
        if qm.get_current_shot() % 2 == 0:
            left = qm.h(left)
        if qm.get_current_shot() % 2 == 0:
            right = qm.h(right)
        return left, right

    estimate = circuit.estimate_resources()
    assert estimate.gates.total == 3
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1


def test_public_input_remains_symbolic_beside_a_finite_shot_value() -> None:
    """Ordinary inputs retain their substitution contract."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Add a public count to a bounded shot remainder."""
        q = qm.qubit("q")
        for _ in qm.range(n + qm.get_current_shot() % 4):
            q = qm.h(q)
        return q

    symbolic = circuit.estimate_resources()
    assert set(symbolic.parameters) == {"n"}
    assert symbolic.substitute(n=2).gates.total == 5
    assert circuit.estimate_resources(inputs={"n": 2}).gates.total == 5


def test_shot_loop_checks_stores_inside_a_preserved_helper() -> None:
    """Selected helper calls cannot hide persistent classical array writes."""

    @qm.qkernel
    def clear(bits: qm.Vector[qm.Bit]) -> qm.Vector[qm.Bit]:
        """Clear the supplied array slot."""
        bits[0] = qm.bit(False)
        return bits

    configure_composite(clear)

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Preserve the costly initial state when a context loop is empty."""
        bits = qm.bit_array(1)
        bits[0] = qm.bit(True)
        for _ in qm.range(qm.get_current_shot() % 2):
            bits = clear(bits)
        q = qm.qubit("q")
        if bits[0]:
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent loop"):
        circuit.estimate_resources()


def test_negative_shot_uint_count_is_not_clipped_to_zero() -> None:
    """Unsigned subtraction cannot be modeled as a negative Python range."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a count whose finite candidates include negative integers."""
        q = qm.qubit("q")
        for _ in qm.range(qm.get_current_shot() % 4 - 2):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources()


def test_shot_uint_underflow_is_rejected_before_later_arithmetic() -> None:
    """A positive final count cannot hide an unsupported negative operand."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Divide an unsafe intermediate value before adding a constant."""
        residue = qm.get_current_shot() % 2
        count = (residue - 2) // 2 + 100
        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources()


def test_simplified_shot_uint_underflow_remains_an_error() -> None:
    """Simplification cannot erase the origin of a negative UInt result."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Subtract a strictly larger value from the same residue."""
        residue = qm.get_current_shot() % 2
        count = residue - (residue + 1)
        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources()


def test_nonnegative_decreasing_shot_count_remains_supported() -> None:
    """A decreasing expression is safe when every candidate is nonnegative."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a finite count between two and five."""
        q = qm.qubit("q")
        for _ in qm.range(5 - qm.get_current_shot() % 4):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()
    assert estimate.gates.total == 5
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_supplied_input_proves_nonnegative_shot_candidates() -> None:
    """Public input values participate in finite UInt-domain validation."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Subtract a bounded remainder from a supplied nonnegative count."""
        q = qm.qubit("q")
        for _ in qm.range(n - qm.get_current_shot() % 4):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources(inputs={"n": 10})
    assert estimate.gates.total == 10
    assert estimate.parameters == {}
    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources(inputs={"n": 2})


def test_negative_shot_selected_table_count_is_rejected() -> None:
    """A finite input table cannot supply an unsupported negative UInt count."""
    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        _shot_table_loop.estimate_resources(inputs={"counts": [-1, 5]})


def test_unbounded_shot_subtraction_cannot_disappear_in_modulo() -> None:
    """Unknown UInt subtraction cannot become a false constant-zero count."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a multiple whose mathematical remainder erases underflow."""
        count = ((qm.get_current_shot() - 2) * 3) % 3
        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources()


def test_shot_domain_budget_cannot_hide_unsafe_subtraction() -> None:
    """Exceeding the candidate budget cannot bypass UInt-domain validation."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Combine more remainder alternatives than bounded analysis allows."""
        shot = qm.get_current_shot()
        left = shot % 4
        right = shot % 5
        count = ((left - right) * 3) % 3
        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources()


def test_shot_uint_underflow_after_a_constant_intermediate_is_rejected() -> None:
    """A constant intermediate cannot hide a subsequent negative UInt result."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Cancel a residue before subtracting from the resulting zero."""
        residue = qm.get_current_shot() % 2
        zero = residue - residue
        count = zero - 1
        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="shot-dependent UInt"):
        circuit.estimate_resources()
