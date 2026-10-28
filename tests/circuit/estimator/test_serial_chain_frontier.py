"""Regression tests for serial-chain synchronized-entry frontiers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp

import qamomile.circuit as qmc


@qmc.qkernel
def _ghz_like_frontier_chain(trips: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply an H gate to the first wire before an adjacent CX chain."""
    register = qmc.qubit_array(trips + 1, "register")
    register[0] = qmc.h(register[0])
    for index in qmc.range(trips):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _second_frontier_wire_chain(trips: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply an H gate to the second first-iteration operand."""
    register = qmc.qubit_array(trips + 2, "register")
    register[1] = qmc.h(register[1])
    for index in qmc.range(trips):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _late_wire_before_chain() -> qmc.Vector[qmc.Qubit]:
    """Touch the final wire before a two-gate adjacent chain."""
    register = qmc.qubit_array(3, "register")
    register[2] = qmc.h(register[2])
    for index in qmc.range(2):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _mixed_frontier_and_late_event() -> qmc.Vector[qmc.Qubit]:
    """Touch one frontier wire and one late wire in the same event."""
    register = qmc.qubit_array(3, "register")
    register[0], register[2] = qmc.cx(register[0], register[2])
    for index in qmc.range(2):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _partial_uniform_write_before_chain() -> qmc.Vector[qmc.Qubit]:
    """Leave a late wire outside a uniform two-wire write."""
    register = qmc.qubit_array(3, "register")
    register[2] = qmc.h(register[2])
    register[2] = qmc.h(register[2])
    register[0], register[1] = qmc.cx(register[0], register[1])
    for index in qmc.range(2):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _full_uniform_write_before_chain() -> qmc.Vector[qmc.Qubit]:
    """Resynchronize all chain wires with one uniform three-wire gate."""
    register = qmc.qubit_array(3, "register")
    register[2] = qmc.h(register[2])
    register[2] = qmc.h(register[2])
    register[0], register[1], register[2] = qmc.ccx(
        register[0],
        register[1],
        register[2],
    )
    for index in qmc.range(2):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _violated_parallel_write_before_chain() -> qmc.Vector[qmc.Qubit]:
    """Prevent a violated legacy certificate from resetting a later chain."""
    register = qmc.qubit_array(3, "register")
    register[2] = qmc.h(register[2])
    for index in qmc.range(3):
        register[index] = qmc.x(register[index])
    for index in qmc.range(2):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _adjacent_chain_helper(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply an adjacent CX chain to a caller-owned register or view."""
    size = register.shape[0]
    for index in qmc.range(1, size):
        register[index - 1], register[index] = qmc.cx(
            register[index - 1],
            register[index],
        )
    return register


@qmc.qkernel
def _frontier_view_through_helper() -> qmc.Vector[qmc.Qubit]:
    """Map a helper frontier through an exact nonunit-stride view."""
    register = qmc.qubit_array(7, "register")
    register[1] = qmc.h(register[1])
    _adjacent_chain_helper(register[1:7:2])
    return register


@qmc.qkernel
def _late_view_wire_through_helper() -> qmc.Vector[qmc.Qubit]:
    """Map a helper late wire through an exact nonunit-stride view."""
    register = qmc.qubit_array(7, "register")
    register[5] = qmc.h(register[5])
    _adjacent_chain_helper(register[1:7:2])
    return register


@qmc.qkernel
def _symbolic_view_mapping(start: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Move one prior event between late and frontier view positions."""
    register = qmc.qubit_array(5, "register")
    register[2] = qmc.h(register[2])
    _adjacent_chain_helper(register[start : start + 3])
    return register


@qmc.qkernel
def _symbolic_view_may_exclude_prior(start: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Move a symbolic chain view completely away from one prior event."""
    register = qmc.qubit_array(8, "register")
    register[6] = qmc.h(register[6])
    _adjacent_chain_helper(register[start : start + 3])
    return register


@qmc.qkernel
def _long_frontier_prefix(trips: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply a three-layer prefix to the first serial-chain frontier wire."""
    register = qmc.qubit_array(trips + 1, "register")
    register[0] = qmc.h(register[0])
    register[0] = qmc.x(register[0])
    register[0] = qmc.h(register[0])
    for index in qmc.range(trips):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _conditional_late_wire(
    flag: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Guard late-wire work independently of harmless frontier work."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    if flag:
        register[2] = qmc.x(register[2])
    for index in qmc.range(2):
        register[index], register[index + 1] = qmc.cx(
            register[index],
            register[index + 1],
        )
    return register


@qmc.qkernel
def _positive_strided_frontier_chain(
    trips: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a frontier gate before a positive stride-two chain."""
    register = qmc.qubit_array(2 * trips + 1, "register")
    register[0] = qmc.h(register[0])
    for index in qmc.range(0, 2 * trips, 2):
        register[index], register[index + 2] = qmc.cx(
            register[index],
            register[index + 2],
        )
    return register


@qmc.qkernel
def _negative_strided_frontier_chain(
    trips: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a frontier gate before a negative stride-two chain."""
    register = qmc.qubit_array(2 * trips + 1, "register")
    register[2 * trips] = qmc.h(register[2 * trips])
    for index in qmc.range(2 * trips, 0, -2):
        register[index], register[index - 2] = qmc.cx(
            register[index],
            register[index - 2],
        )
    return register


@qmc.qkernel
def _outer_repeat_clears_frontier() -> qmc.Vector[qmc.Qubit]:
    """Move a helper certificate through an outer repeat aggregation."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    for _index in qmc.range(1):
        register = _adjacent_chain_helper(register)
    return register


@qmc.qkernel
def _outer_sum_clears_frontier() -> qmc.Vector[qmc.Qubit]:
    """Move a helper certificate through a binder-dependent outer sum."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    for index in qmc.range(1):
        _scratch = qmc.qubit_array(index, "scratch")
        register = _adjacent_chain_helper(register)
    return register


@qmc.qkernel
def _outer_concrete_replay(
    outer: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Replay a carried outer loop containing an adjacent-chain helper."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    total = qmc.uint(0)
    for _index in qmc.range(outer):
        total = total + 1
        register = _adjacent_chain_helper(register)
    return register


@qmc.qkernel
def _cross_iteration_concrete_replay(
    outer: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Separate frontier work and a nested chain across outer iterations."""
    register = qmc.qubit_array(3, "register")
    total = qmc.uint(0)
    for index in qmc.range(outer):
        total = total + 1
        if index == 0:
            register[0] = qmc.h(register[0])
        if index == 1:
            register = _adjacent_chain_helper(register)
    return register


@qmc.qkernel
def _allocated_concrete_replay(
    outer: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Allocate local workspace while replaying a nested adjacent chain."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    total = qmc.uint(0)
    for _index in qmc.range(outer):
        total = total + 1
        _scratch = qmc.qubit("scratch")
        register = _adjacent_chain_helper(register)
    return register


@qmc.qkernel
def _allocated_cross_iteration_concrete_replay(
    outer: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Combine body-local allocation with a late-wire replay hazard."""
    register = qmc.qubit_array(3, "register")
    total = qmc.uint(0)
    for index in qmc.range(outer):
        total = total + 1
        _scratch = qmc.qubit("scratch")
        if index == 0:
            register[2] = qmc.h(register[2])
        if index == 1:
            register = _adjacent_chain_helper(register)
    return register


@qmc.qkernel
def _inverse_chain_after_forward_frontier() -> qmc.Vector[qmc.Qubit]:
    """Apply an inverted chain after work on its former frontier."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    return qmc.inverse(_adjacent_chain_helper)(register)


@qmc.qkernel
def _downstream_early_wire() -> qmc.Vector[qmc.Qubit]:
    """Use a chain's early-finishing first wire after aggregation."""
    register = qmc.qubit_array(3, "register")
    register = _adjacent_chain_helper(register)
    register[0] = qmc.h(register[0])
    return register


@qmc.qkernel
def _many_frontier_events_before_chain() -> qmc.Vector[qmc.Qubit]:
    """Place more than the exact-scan budget of harmless frontier events."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register[0] = qmc.h(register[0])
    register = _adjacent_chain_helper(register)
    return register


def _assert_exact_without_assumptions(estimate: qmc.ResourceEstimate) -> None:
    """Assert that an estimate retains exact, assumption-free metadata."""
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()


def _assert_dependency_conservative(estimate: qmc.ResourceEstimate) -> None:
    """Assert that dependency scheduling disclosed a conservative result."""
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source in {"dependency scheduler", "loop dependency scheduler"}
        for assumption in estimate.assumptions
    )


def _assert_symbolically_equal(actual: sp.Expr, expected: sp.Expr) -> None:
    """Assert symbolic equality after simplification."""
    assert sp.simplify(actual - expected) == 0


def test_ghz_like_frontier_is_exact_symbolically_and_after_binding() -> None:
    """H on the first chain operand is valid for every specialization route."""
    symbolic = _ghz_like_frontier_chain.estimate_resources()
    trips = symbolic.parameters["trips"]

    _assert_symbolically_equal(symbolic.gates.total, trips + 1)
    _assert_symbolically_equal(symbolic.depth.depth, trips + 1)
    _assert_exact_without_assumptions(symbolic)

    for concrete_trips in (0, 1, 2, 4):
        direct = _ghz_like_frontier_chain.estimate_resources(
            inputs={"trips": concrete_trips}
        )
        substituted = symbolic.substitute(trips=concrete_trips)

        assert direct.gates == substituted.gates
        assert direct.depth == substituted.depth
        assert direct.depth.depth == concrete_trips + 1
        _assert_exact_without_assumptions(direct)
        _assert_exact_without_assumptions(substituted)


def test_long_frontier_prefix_delays_the_chain_without_losing_exactness() -> None:
    """Several prior frontier layers remain an exact chain dependency."""
    symbolic = _long_frontier_prefix.estimate_resources()
    trips = symbolic.parameters["trips"]

    _assert_symbolically_equal(symbolic.depth.depth, trips + 3)
    _assert_exact_without_assumptions(symbolic)
    direct = _long_frontier_prefix.estimate_resources(inputs={"trips": 4})
    substituted = symbolic.substitute(trips=4)
    assert direct.depth.depth == substituted.depth.depth == 7
    _assert_exact_without_assumptions(direct)
    _assert_exact_without_assumptions(substituted)


def test_second_first_iteration_operand_is_also_frontier() -> None:
    """Different readiness on the second frontier wire remains exact."""
    symbolic = _second_frontier_wire_chain.estimate_resources()
    trips = symbolic.parameters["trips"]

    _assert_symbolically_equal(symbolic.depth.depth, trips + 1)
    _assert_exact_without_assumptions(symbolic)
    for concrete_trips in (0, 1, 2, 4):
        direct = _second_frontier_wire_chain.estimate_resources(
            inputs={"trips": concrete_trips}
        )
        substituted = symbolic.substitute(trips=concrete_trips)
        assert direct.depth == substituted.depth
        assert direct.depth.depth == concrete_trips + 1
        _assert_exact_without_assumptions(direct)
        _assert_exact_without_assumptions(substituted)


@pytest.mark.parametrize(
    "kernel",
    [_late_wire_before_chain, _mixed_frontier_and_late_event],
)
def test_late_or_mixed_prior_event_is_not_frontier_exempt(
    kernel: Callable[[], qmc.Vector[qmc.Qubit]],
) -> None:
    """Any definitely late overlap makes the serial aggregate conservative."""
    estimate = kernel.estimate_resources()

    assert estimate.depth.depth == 3
    _assert_dependency_conservative(estimate)


def test_partial_write_cannot_hide_an_older_late_wire() -> None:
    """A frontier-only write does not stop the scan before an older hazard."""
    estimate = _partial_uniform_write_before_chain.estimate_resources()

    assert estimate.depth.depth == 4
    _assert_dependency_conservative(estimate)


def test_full_uniform_write_resynchronizes_the_entire_chain_coverage() -> None:
    """One uniform writer covering full U validly resets the certificate."""
    estimate = _full_uniform_write_before_chain.estimate_resources()

    assert estimate.gates.total == 5
    assert estimate.depth.depth == 5
    _assert_exact_without_assumptions(estimate)


def test_violated_legacy_writer_cannot_reset_a_grouped_certificate() -> None:
    """A legacy entry violation poisons reset trust for a following chain."""
    estimate = _violated_parallel_write_before_chain.estimate_resources()

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 4
    _assert_dependency_conservative(estimate)


def test_exact_strided_view_mapping_preserves_frontier_membership() -> None:
    """A unique root-scalar view mapping keeps harmless work exact."""
    estimate = _frontier_view_through_helper.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    _assert_exact_without_assumptions(estimate)


def test_exact_strided_view_mapping_preserves_late_wire_hazards() -> None:
    """The same view mapping does not erase its final late wire."""
    estimate = _late_view_wire_through_helper.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    _assert_dependency_conservative(estimate)


def test_symbolic_view_overlap_specializes_without_widening_frontier() -> None:
    """A possible overlap is not promoted to frontier before it is proven."""
    symbolic = _symbolic_view_mapping.estimate_resources()
    direct_late = _symbolic_view_mapping.estimate_resources(inputs={"start": 0})
    direct_frontier = _symbolic_view_mapping.estimate_resources(inputs={"start": 1})
    substituted_late = symbolic.substitute(start=0)
    substituted_frontier = symbolic.substitute(start=1)

    _assert_dependency_conservative(symbolic)
    _assert_dependency_conservative(direct_late)
    _assert_dependency_conservative(substituted_late)
    _assert_exact_without_assumptions(direct_frontier)
    _assert_exact_without_assumptions(substituted_frontier)
    assert direct_late.depth.depth == substituted_late.depth.depth == 3
    assert direct_frontier.depth.depth == substituted_frontier.depth.depth == 3


def test_post_hoc_disjoint_view_specialization_remains_fail_closed() -> None:
    """Late substitution does not pretend to rerun dependency scheduling."""
    symbolic = _symbolic_view_may_exclude_prior.estimate_resources()
    direct = _symbolic_view_may_exclude_prior.estimate_resources(inputs={"start": 0})
    substituted = symbolic.substitute(start=0)

    assert direct.depth.depth == 2
    _assert_exact_without_assumptions(direct)
    assert substituted.depth.depth == 3
    _assert_dependency_conservative(substituted)


@pytest.mark.parametrize(
    ("flag", "expected_quality"),
    [
        (0, qmc.EstimateQuality.EXACT),
        (1, qmc.EstimateQuality.CONSERVATIVE),
    ],
)
def test_conditional_late_wire_guard_survives_substitution(
    flag: int,
    expected_quality: qmc.EstimateQuality,
) -> None:
    """Only the branch with active late-wire work violates the certificate."""
    symbolic = _conditional_late_wire.estimate_resources()
    direct = _conditional_late_wire.estimate_resources(inputs={"flag": flag})
    substituted = symbolic.substitute(flag=flag)

    assert direct.depth.depth == substituted.depth.depth == 3
    assert direct.quality is substituted.quality is expected_quality
    if flag:
        _assert_dependency_conservative(direct)
        _assert_dependency_conservative(substituted)
    else:
        _assert_exact_without_assumptions(direct)
        _assert_exact_without_assumptions(substituted)


@pytest.mark.parametrize(
    "kernel",
    [_positive_strided_frontier_chain, _negative_strided_frontier_chain],
)
def test_signed_nonunit_stride_frontier_is_exact(
    kernel: Callable[[qmc.UInt], qmc.Vector[qmc.Qubit]],
) -> None:
    """The physical first operands follow signed iteration order."""
    symbolic = kernel.estimate_resources()
    trips = symbolic.parameters["trips"]

    _assert_symbolically_equal(symbolic.depth.depth, trips + 1)
    _assert_exact_without_assumptions(symbolic)
    for concrete_trips in (0, 1, 2, 4):
        direct = kernel.estimate_resources(inputs={"trips": concrete_trips})
        substituted = symbolic.substitute(trips=concrete_trips)
        assert direct.depth == substituted.depth
        assert direct.depth.depth == concrete_trips + 1
        _assert_exact_without_assumptions(direct)
        _assert_exact_without_assumptions(substituted)


@pytest.mark.parametrize(
    "kernel",
    [_outer_repeat_clears_frontier, _outer_sum_clears_frontier],
)
def test_outer_aggregation_clears_nested_frontier_fail_closed(
    kernel: Callable[[], qmc.Vector[qmc.Qubit]],
) -> None:
    """A certificate crossing any outer binder loses frontier exemptions."""
    estimate = kernel.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert estimate.assumptions


def test_concrete_outer_replay_clears_nested_frontier_fail_closed() -> None:
    """Concrete replay cannot preserve a nested chain's directional frontier."""
    symbolic = _outer_concrete_replay.estimate_resources()
    direct = _outer_concrete_replay.estimate_resources(inputs={"outer": 1})
    substituted = symbolic.substitute(outer=1)

    assert direct.depth.depth == substituted.depth.depth == 3
    _assert_dependency_conservative(direct)
    _assert_dependency_conservative(substituted)


def test_concrete_replay_clears_frontier_before_cross_iteration_scheduling() -> None:
    """An earlier outer iteration cannot use a nested helper's frontier."""
    symbolic = _cross_iteration_concrete_replay.estimate_resources()
    direct = _cross_iteration_concrete_replay.estimate_resources(inputs={"outer": 2})
    substituted = symbolic.substitute(outer=2)

    assert direct.depth.depth == substituted.depth.depth == 3
    _assert_dependency_conservative(direct)
    assert substituted.quality is qmc.EstimateQuality.CONSERVATIVE
    assert substituted.assumptions


def test_allocated_concrete_replay_clears_frontier_on_early_return() -> None:
    """Body-local workspace cannot bypass an outer binder's frontier reset."""
    symbolic = _allocated_concrete_replay.estimate_resources()
    direct = _allocated_concrete_replay.estimate_resources(inputs={"outer": 1})
    substituted = symbolic.substitute(outer=1)

    assert direct.depth.depth == substituted.depth.depth == 3
    assert direct.quality is substituted.quality is qmc.EstimateQuality.CONSERVATIVE
    assert direct.assumptions
    assert substituted.assumptions


def test_allocated_concrete_replay_still_checks_internal_entry_hazards() -> None:
    """Local allocation disables parallelization, not entry-hazard checks."""
    symbolic = _allocated_cross_iteration_concrete_replay.estimate_resources()
    direct = _allocated_cross_iteration_concrete_replay.estimate_resources(
        inputs={"outer": 2}
    )
    substituted = symbolic.substitute(outer=2)

    assert direct.depth.depth == substituted.depth.depth == 3
    _assert_dependency_conservative(direct)
    assert substituted.quality is qmc.EstimateQuality.CONSERVATIVE
    assert substituted.assumptions


def test_inverse_clears_directional_frontier_fail_closed() -> None:
    """Inversion cannot reuse the forward chain's first-iteration frontier."""
    estimate = _inverse_chain_after_forward_frontier.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    _assert_dependency_conservative(estimate)


def test_downstream_early_wire_remains_conservative() -> None:
    """A scalar aggregate cannot invent an exact early-wire completion."""
    estimate = _downstream_early_wire.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        "aggregate latency" in assumption.message for assumption in estimate.assumptions
    )


def test_many_frontier_only_events_do_not_trigger_coarse_fallback() -> None:
    """Harmless frontier events do not consume the exact hazard-scan budget."""
    estimate = _many_frontier_events_before_chain.estimate_resources()

    assert estimate.gates.total == 36
    assert estimate.depth.depth == 36
    _assert_exact_without_assumptions(estimate)
