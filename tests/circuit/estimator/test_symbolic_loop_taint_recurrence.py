"""Tests for path-insensitive measurement taint across symbolic loops."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator.resource_estimator import ResourceEstimate


@qm.qkernel
def _range_may_taint(n: qm.UInt) -> qm.Qubit:
    """Introduce measurement taint into a symbolic range carry."""
    probe = qm.cast(qm.qubit_array(1, "probe"), qm.QFixed, int_bits=0)
    state = qm.float_(0.0)
    target = qm.qubit("target")
    for _ in qm.range(n):
        if state > 0.0:
            target = qm.h(target)
        state = qm.measure(probe)
    return target


@qm.qkernel
def _items_may_taint(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
    """Introduce measurement taint into a symbolic items carry."""
    probe = qm.cast(qm.qubit_array(1, "probe"), qm.QFixed, int_bits=0)
    state = qm.float_(0.0)
    target = qm.qubit("target")
    for _key, _value in qm.items(data):
        if state > 0.0:
            target = qm.h(target)
        state = qm.measure(probe)
    return target


@qm.qkernel
def _nested_range_may_taint(n: qm.UInt, m: qm.UInt) -> qm.Qubit:
    """Carry measurement taint through two nested symbolic ranges."""
    probe = qm.cast(qm.qubit_array(1, "probe"), qm.QFixed, int_bits=0)
    state = qm.float_(0.0)
    target = qm.qubit("target")
    for _ in qm.range(n):
        for _inner in qm.range(m):
            if state > 0.0:
                target = qm.h(target)
            state = qm.measure(probe)
    return target


@qm.qkernel
def _nested_items_range_may_taint(
    data: qm.Dict[qm.UInt, qm.Float],
    m: qm.UInt,
) -> qm.Qubit:
    """Carry measurement taint through symbolic items and range loops."""
    probe = qm.cast(qm.qubit_array(1, "probe"), qm.QFixed, int_bits=0)
    state = qm.float_(0.0)
    target = qm.qubit("target")
    for _key, _value in qm.items(data):
        for _inner in qm.range(m):
            if state > 0.0:
                target = qm.h(target)
            state = qm.measure(probe)
    return target


@qm.qkernel
def _post_range_may_taint(n: qm.UInt) -> qm.Qubit:
    """Use a potentially measured carry after a symbolic range."""
    probe = qm.cast(qm.qubit_array(1, "probe"), qm.QFixed, int_bits=0)
    state = qm.float_(0.0)
    for _ in qm.range(n):
        state = qm.measure(probe)
    target = qm.qubit("target")
    if state > 0.0:
        target = qm.h(target)
    return target


@qm.qkernel
def _post_items_may_taint(
    data: qm.Dict[qm.UInt, qm.Float],
) -> qm.Qubit:
    """Use a potentially measured carry after a symbolic items loop."""
    probe = qm.cast(qm.qubit_array(1, "probe"), qm.QFixed, int_bits=0)
    state = qm.float_(0.0)
    for _key, _value in qm.items(data):
        state = qm.measure(probe)
    target = qm.qubit("target")
    if state > 0.0:
        target = qm.h(target)
    return target


@qm.qkernel
def _unresolved_items_carry(
    data: qm.Dict[qm.UInt, qm.Float],
) -> qm.Qubit:
    """Use a nonlinear symbolic carry as a later range bound."""
    repetitions = qm.uint(0)
    for _key, _value in qm.items(data):
        repetitions = repetitions * repetitions + 1
    target = qm.qubit("target")
    for _ in qm.range(repetitions):
        target = qm.h(target)
    return target


@qm.qkernel
def _runtime_range_bound() -> qm.Qubit:
    """Use a measurement-derived value as a range bound."""
    predicate = qm.measure(qm.qubit("predicate"))
    repetitions = qm.uint(0)
    if predicate:
        repetitions = qm.uint(1)
    target = qm.qubit("target")
    for _ in qm.range(repetitions):
        target = qm.h(target)
    return target


@qm.qkernel
def _compile_input_with_unrelated_runtime_value(n: qm.UInt) -> qm.Qubit:
    """Keep a compile-time loop bound distinct from a runtime value."""
    predicate = qm.measure(qm.qubit("predicate"))
    selected = qm.uint(0)
    if predicate:
        selected = qm.uint(1)
    selected = selected + n
    target = qm.qubit("target")
    for _ in qm.range(n):
        target = qm.h(target)
    return target


def _assert_no_internal_parameters(
    estimate: ResourceEstimate,
    expected: set[str],
) -> None:
    """Assert that only user-substitutable parameters remain public.

    Args:
        estimate (ResourceEstimate): Estimate whose public registry is checked.
        expected (set[str]): Exact set of expected public parameter aliases.
    """
    assert set(estimate.parameters) == expected
    assert all("state_after" not in name for name in estimate.parameters)
    assert all("fallback" not in name for name in estimate.parameters)


def _assert_numeric_upper_bound(
    upper: ResourceEstimate,
    lower: ResourceEstimate,
) -> None:
    """Assert that every numeric resource field in ``upper`` is safe.

    Args:
        upper (ResourceEstimate): Specialized symbolic estimate.
        lower (ResourceEstimate): Estimate obtained by direct concrete replay.
    """
    for upper_group, lower_group in (
        (upper.gates, lower.gates),
        (upper.depth, lower.depth),
        (upper.width, lower.width),
        (upper.measurements, lower.measurements),
        (upper.resets, lower.resets),
    ):
        for field in fields(upper_group):
            upper_value = sp.sympify(getattr(upper_group, field.name))
            lower_value = sp.sympify(getattr(lower_group, field.name))
            assert upper_value.is_number
            assert lower_value.is_number
            assert bool(upper_value >= lower_value), (
                field.name,
                upper_value,
                lower_value,
            )


@pytest.mark.parametrize(
    ("kernel", "public_count", "concrete_input"),
    (
        (_range_may_taint, "n", lambda count: {"n": count}),
        (
            _items_may_taint,
            "|data|",
            lambda count: {"data": {index: float(index) for index in range(count)}},
        ),
    ),
)
def test_symbolic_loop_may_taint_is_a_conservative_upper_bound(
    kernel: Any,
    public_count: str,
    concrete_input: Any,
) -> None:
    """Range and items loops conservatively retain carried measurement taint.

    Args:
        kernel (Any): Kernel containing the symbolic loop under test.
        public_count (str): Public alias for the symbolic iteration count.
        concrete_input (Any): Factory for a concrete estimation input mapping.
    """
    symbolic = kernel.estimate_resources()

    _assert_no_internal_parameters(symbolic, {public_count})
    assert symbolic.quality is qm.EstimateQuality.CONSERVATIVE
    for count in (0, 1, 3):
        specialized = symbolic.substitute(**{public_count: count})
        direct = kernel.estimate_resources(inputs=concrete_input(count))
        _assert_numeric_upper_bound(specialized, direct)
        _assert_no_internal_parameters(specialized, set())


@pytest.mark.parametrize(
    ("kernel", "public_counts", "substitutions", "inputs"),
    (
        (
            _nested_range_may_taint,
            {"m", "n"},
            {"n": 2, "m": 3},
            {"n": 2, "m": 3},
        ),
        (
            _nested_items_range_may_taint,
            {"m", "|data|"},
            {"|data|": 2, "m": 3},
            {"data": {0: 0.0, 1: 1.0}, "m": 3},
        ),
    ),
)
def test_nested_symbolic_loops_keep_only_public_counts(
    kernel: Any,
    public_counts: set[str],
    substitutions: dict[str, int],
    inputs: dict[str, Any],
) -> None:
    """Nested may-taint remains bounded without exposing loop carry state.

    Args:
        kernel (Any): Kernel containing nested symbolic loops.
        public_counts (set[str]): Expected iteration-count aliases.
        substitutions (dict[str, int]): Values specializing the symbolic estimate.
        inputs (dict[str, Any]): Values used for direct concrete replay.
    """
    symbolic = kernel.estimate_resources()
    specialized = symbolic.substitute(**substitutions)
    direct = kernel.estimate_resources(inputs=inputs)

    _assert_no_internal_parameters(symbolic, public_counts)
    _assert_no_internal_parameters(specialized, set())
    assert symbolic.quality is qm.EstimateQuality.CONSERVATIVE
    _assert_numeric_upper_bound(specialized, direct)


@pytest.mark.parametrize(
    ("kernel", "public_count", "substitution", "inputs"),
    (
        (_post_range_may_taint, "n", {"n": 0}, {"n": 0}),
        (_post_items_may_taint, "|data|", {"|data|": 0}, {"data": {}}),
    ),
)
def test_direct_zero_trip_input_prunes_post_loop_runtime_branch(
    kernel: Any,
    public_count: str,
    substitution: dict[str, int],
    inputs: dict[str, Any],
) -> None:
    """Concrete zero-trip inputs prune work without requiring rescheduling.

    A symbolic estimate specialized later may remain conservative because
    ``substitute`` does not rerun taint analysis or scheduling.

    Args:
        kernel (Any): Kernel with a post-loop measurement-derived condition.
        public_count (str): Public alias for the loop trip count.
        substitution (dict[str, int]): Zero-trip symbolic specialization.
        inputs (dict[str, Any]): Zero-trip input used before interpretation.
    """
    symbolic = kernel.estimate_resources()
    specialized = symbolic.substitute(**substitution)
    direct = kernel.estimate_resources(inputs=inputs)

    _assert_no_internal_parameters(symbolic, {public_count})
    _assert_no_internal_parameters(specialized, set())
    _assert_no_internal_parameters(direct, set())
    _assert_numeric_upper_bound(specialized, direct)
    assert direct.gates.total == 0
    assert direct.measurements.total == 0
    assert direct.depth.depth == 0
    assert direct.quality is qm.EstimateQuality.EXACT


@pytest.mark.parametrize("kernel", (_unresolved_items_carry, _runtime_range_bound))
def test_runtime_or_unresolved_structural_values_fail_closed(kernel: Any) -> None:
    """Non-public runtime structure cannot escape as a user parameter.

    Args:
        kernel (Any): Kernel whose resource-sensitive structure is unresolved.
    """
    with pytest.raises(
        NotImplementedError,
        match="runtime-derived or unresolved loop-carried value",
    ):
        kernel.estimate_resources()


def test_runtime_value_does_not_hide_an_independent_compile_input() -> None:
    """An unrelated runtime expression cannot consume public loop input ``n``."""
    symbolic = _compile_input_with_unrelated_runtime_value.estimate_resources()
    concrete = _compile_input_with_unrelated_runtime_value.estimate_resources(
        inputs={"n": 3}
    )

    _assert_no_internal_parameters(symbolic, {"n"})
    _assert_no_internal_parameters(concrete, set())
    assert symbolic.gates.total == symbolic.parameters["n"]
    assert concrete.gates.total == 3
