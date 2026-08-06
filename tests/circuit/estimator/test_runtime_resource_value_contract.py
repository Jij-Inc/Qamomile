"""Contract tests for runtime-derived classical resource values."""

from __future__ import annotations

from typing import Any

import pytest
import sympy as sp

import qamomile.circuit as qm


@qm.qkernel
def _runtime_uint_merge_to_if() -> qm.Qubit:
    """Use a measurement-selected UInt in a later conditional."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)

    target = qm.qubit("target")
    if selected > 0:
        target = qm.x(target)
    else:
        target = qm.h(target)
        target = qm.z(target)
    return target


@qm.qkernel
def _runtime_uint_merge_to_range() -> qm.Qubit:
    """Use a measurement-selected UInt as a resource-sensitive range bound."""
    measured = qm.measure(qm.qubit("predicate"))
    repetitions = qm.uint(0)
    if measured:
        repetitions = qm.uint(2)

    target = qm.qubit("target")
    for _ in qm.range(repetitions):
        target = qm.h(target)
    return target


@qm.qkernel
def _runtime_identical_uint_merge_to_range() -> qm.Qubit:
    """Use an identical measurement-selected UInt as a range bound."""
    measured = qm.measure(qm.qubit("predicate"))
    repetitions = qm.uint(2)
    if measured:
        repetitions = qm.uint(2)

    target = qm.qubit("target")
    for _ in qm.range(repetitions):
        target = qm.h(target)
    return target


@qm.qkernel
def _runtime_index_with_input_alternative(
    index: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Use an input as one alternative of a measurement-selected index."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = index

    register = qm.qubit_array(2, "register")
    register[selected] = qm.h(register[selected])
    return register


@qm.qkernel
def _range_carried_runtime_array_element(
    n: qm.UInt,
    values: qm.Vector[qm.UInt],
) -> qm.Qubit:
    """Carry an element selected by a measured index through a range loop."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)
    state = qm.uint(0)
    for _ in qm.range(n):
        state = values[selected]
    target = qm.qubit("target")
    if state > 0:
        target = qm.h(target)
    return target


@qm.qkernel
def _choose_runtime_index(measured: qm.Bit) -> qm.UInt:
    """Return zero or one according to a runtime observation."""
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)
    return selected


@qm.qkernel
def _items_carried_runtime_array_element(
    data: qm.Dict[qm.UInt, qm.Float],
    values: qm.Vector[qm.UInt],
) -> qm.Qubit:
    """Carry an element selected by a measured index through an items loop."""
    selected = _choose_runtime_index(qm.measure(qm.qubit("predicate")))
    state = qm.uint(0)
    for _key, _value in qm.items(data):
        state = values[selected]
    target = qm.qubit("target")
    if state > 0:
        target = qm.h(target)
    return target


@qm.qkernel
def _runtime_indexed_range(counts: qm.Vector[qm.UInt]) -> qm.Qubit:
    """Use an array element selected by measurement as a loop bound."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)
    target = qm.qubit("target")
    for _ in qm.range(counts[selected]):
        target = qm.h(target)
    return target


@qm.qkernel
def _runtime_indexed_array_index(
    indices: qm.Vector[qm.UInt],
) -> qm.Vector[qm.Qubit]:
    """Use an array element selected by measurement as a quantum index."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)
    register = qm.qubit_array(2, "register")
    dynamic_index = indices[selected]
    register[dynamic_index] = qm.h(register[dynamic_index])
    return register


@qm.qkernel
def _runtime_indexed_allocation(
    widths: qm.Vector[qm.UInt],
) -> qm.Vector[qm.Qubit]:
    """Use an array element selected by measurement as an allocation width."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)
    return qm.qubit_array(widths[selected], "register")


@qm.qkernel
def _unsupported_symbolic_range_carry(repetitions: qm.UInt) -> qm.Qubit:
    """Use a nonlinear symbolic range carry as a later resource bound."""
    state = qm.uint(0)
    for _ in qm.range(repetitions):
        state = state * state + 1

    target = qm.qubit("target")
    for _ in qm.range(state):
        target = qm.h(target)
    return target


@qm.qkernel
def _unsupported_symbolic_items_carry(
    data: qm.Dict[qm.UInt, qm.Float],
) -> qm.Qubit:
    """Use a nonlinear symbolic items carry as a later resource bound."""
    state = qm.uint(0)
    for _key, _value in qm.items(data):
        state = state * state + 1

    target = qm.qubit("target")
    for _ in qm.range(state):
        target = qm.h(target)
    return target


@qm.qkernel
def _runtime_while_count() -> qm.Qubit:
    """Apply one gate per execution of a measurement-controlled while body."""
    active = qm.measure(qm.qubit("initial_predicate"))
    target = qm.qubit("target")
    while active:
        target = qm.h(target)
        active = qm.measure(qm.qubit("next_predicate"))
    return target


@qm.qkernel
def _mixed_runtime_and_compile_input(n: qm.UInt) -> qm.Qubit:
    """Keep an ordinary input public beside a runtime-derived expression."""
    measured = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if measured:
        selected = qm.uint(1)
    _unused = selected + n
    target = qm.qubit("target")
    for _ in qm.range(n):
        target = qm.h(target)
    return target


@qm.qkernel
def _trace_only_unresolved_carry(n: qm.UInt) -> qm.Qubit:
    """Use an unresolved carry only to choose equal-cost trace branches."""
    total = qm.float_(0.5)
    for _ in qm.range(n):
        total = total * total
    target = qm.qubit("target")
    if total > 0.0:
        target = qm.h(target)
    else:
        target = qm.x(target)
    return target


def test_runtime_uint_merge_if_uses_fieldwise_worst_case() -> None:
    """A runtime UInt condition takes field-wise maxima without parameters."""
    estimate = _runtime_uint_merge_to_if.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.gates.single_qubit == 2
    assert estimate.gates.clifford == 2
    assert estimate.depth.depth == 3
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_runtime_uint_merge_range_fails_without_fake_parameter() -> None:
    """A runtime range bound fails instead of becoming a public parameter."""
    with pytest.raises(
        NotImplementedError,
        match="runtime-derived or unresolved loop-carried value",
    ):
        _runtime_uint_merge_to_range.estimate_resources()


def test_identical_runtime_merge_specializes_without_fake_parameter() -> None:
    """Equal runtime alternatives publish their sole concrete value."""
    estimate = _runtime_identical_uint_merge_to_range.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_runtime_index_domain_is_specialized_before_constraint_validation() -> None:
    """A supplied valid alternative is consumed without becoming a parameter."""
    estimate = _runtime_index_with_input_alternative.estimate_resources(
        inputs={"index": 1}
    )

    assert estimate.gates.total == 1
    assert estimate.parameters == {}


def test_runtime_index_domain_rejects_supplied_out_of_bounds_alternative() -> None:
    """Every finite runtime alternative must satisfy array bounds."""
    with pytest.raises(ValueError, match="index"):
        _runtime_index_with_input_alternative.estimate_resources(inputs={"index": 5})


@pytest.mark.parametrize(
    ("kernel", "count_name"),
    (
        (_range_carried_runtime_array_element, "n"),
        (_items_carried_runtime_array_element, "|data|"),
    ),
    ids=("range", "items"),
)
def test_runtime_selected_array_element_never_becomes_public_loop_carry(
    kernel: Any,
    count_name: str,
) -> None:
    """A measured element selection retains its nonempty-loop guard.

    Args:
        kernel (Any): Kernel containing the carried runtime-selected element.
        count_name (str): Public loop-cardinality parameter name.
    """
    estimate = kernel.estimate_resources()
    count = estimate.parameters[count_name]

    assert estimate.gates.total == sp.Piecewise((1, count > 0), (0, True))
    assert estimate.substitute(**{count_name: 0}).gates.total == 0
    assert estimate.substitute(**{count_name: 1}).gates.total == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert all("values[" not in name for name in estimate.parameters)


@pytest.mark.parametrize(
    "kernel",
    (
        _runtime_indexed_range,
        _runtime_indexed_array_index,
        _runtime_indexed_allocation,
    ),
    ids=("range-bound", "quantum-index", "allocation-width"),
)
def test_runtime_selected_array_content_fails_closed(
    kernel: Any,
) -> None:
    """A runtime-selected unresolved array element is never a parameter."""
    with pytest.raises(NotImplementedError, match="runtime-derived"):
        kernel.estimate_resources()


@pytest.mark.parametrize(
    "kernel",
    (
        _unsupported_symbolic_range_carry,
        _unsupported_symbolic_items_carry,
    ),
    ids=("range", "items"),
)
def test_unsupported_symbolic_carry_fails_without_fake_parameter(
    kernel: Any,
) -> None:
    """An unresolved loop carry fails before public parameter discovery."""
    with pytest.raises(
        NotImplementedError,
        match="runtime-derived or unresolved loop-carried value",
    ):
        kernel.estimate_resources()


def test_while_trip_count_remains_a_semantic_parameter() -> None:
    """The semantic while trip count remains public and substitutable."""
    estimate = _runtime_while_count.estimate_resources()

    assert set(estimate.parameters) == {"|while|"}
    trip_count = estimate.parameters["|while|"]
    assert trip_count.is_integer is True
    assert trip_count.is_nonnegative is True

    specialized = estimate.substitute(**{"|while|": 3})
    assert specialized.gates.total == 3
    assert specialized.measurements.total == 4
    assert specialized.parameters == {}


def test_runtime_expression_does_not_internalize_compile_time_input() -> None:
    """A public input remains substitutable when mixed with runtime state."""
    symbolic = _mixed_runtime_and_compile_input.estimate_resources()
    direct = _mixed_runtime_and_compile_input.estimate_resources(inputs={"n": 3})

    assert set(symbolic.parameters) == {"n"}
    assert symbolic.gates.total == symbolic.parameters["n"]
    substituted = symbolic.substitute(n=3)
    assert substituted.gates.total == 3
    assert substituted.parameters == {}
    assert direct.gates.total == 3
    assert direct.parameters == {}


def test_internal_carry_cannot_escape_only_through_trace_guard() -> None:
    """An explanation guard cannot expose an unsubstitutable carry symbol."""
    without_trace = _trace_only_unresolved_carry.estimate_resources(trace=False)

    assert set(without_trace.parameters) == {"n"}
    with pytest.raises(NotImplementedError, match="unresolved loop-carried"):
        _trace_only_unresolved_carry.estimate_resources(trace=True)
