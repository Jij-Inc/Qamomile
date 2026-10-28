"""Regression tests for exact symbolic pair-loop depth scheduling."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.circuit.estimator._dependency_metadata import (
    _project_dependency_metadata_over_symbol,
)
from qamomile.circuit.estimator._interpreter_for_items_context import (
    _ForItemsContextInterpreter,
)
from qamomile.circuit.estimator._resource_types import ResourceTraceNode
from qamomile.circuit.ir.operation.control_flow import ForOperation


@qmc.qkernel
def _triangular_cx(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply one CX gate to every ordered pair in triangular loop order."""
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _shifted_triangular_cx(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply a triangular CX schedule to a register slice starting at two."""
    register = qmc.qubit_array(n + 2, "register")
    for left in qmc.range(2, n + 2):
        for right in qmc.range(left + 1, n + 2):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _triangular_rzz(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply one RZZ rotation to every ordered pair in triangular loop order."""
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.rzz(
                register[left],
                register[right],
                0.125,
            )
    return register


@qmc.qkernel
def _shared_anchor_cx(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply CX gates from one fixed anchor to every later register slot."""
    register = qmc.qubit_array(n, "register")
    for target in qmc.range(1, n):
        register[0], register[target] = qmc.cx(register[0], register[target])
    return register


@qmc.qkernel
def _triangular_with_extra_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Add quantum work that makes the triangular single-gate proof inapplicable."""
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
            register[left] = qmc.h(register[left])
    return register


@qmc.qkernel
def _triangular_with_control(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use controlled CX work that the plain two-qubit-gate proof must reject."""
    control = qmc.qubit("control")
    register = qmc.qubit_array(n, "register")
    controlled_cx = qmc.control(qmc.cx)
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            control, register[left], register[right] = controlled_cx(
                control,
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _nontriangular_inner_start(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Skip the adjacent pair so the triangular schedule proof cannot apply."""
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 2, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _nontriangular_inner_step(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use a strided inner loop that the triangular proof must reject."""
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n, 2):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _triangular_after_last_wire_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Run a triangular schedule after work on its last register wire."""
    register = qmc.qubit_array(n, "register")
    register[n - 1] = qmc.h(register[n - 1])
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _triangular_after_disjoint_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Run a triangular schedule beside earlier work on a disjoint wire."""
    unrelated = qmc.h(qmc.qubit("unrelated"))
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    unrelated = qmc.x(unrelated)
    return register


@qmc.qkernel
def _triangular_then_early_wire_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use an early-finishing register wire after a triangular schedule."""
    register = qmc.qubit_array(n, "register")
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    register[0] = qmc.h(register[0])
    return register


@qmc.qkernel
def _triangular_input_helper(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply a triangular schedule to a caller-owned input register."""
    n = register.shape[0]
    for left in qmc.range(n):
        for right in qmc.range(left + 1, n):
            register[left], register[right] = qmc.cx(
                register[left],
                register[right],
            )
    return register


@qmc.qkernel
def _triangular_through_helper(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Pass a symbolic register through the triangular helper call boundary."""
    return _triangular_input_helper(qmc.qubit_array(n, "register"))


@qmc.qkernel
def _triangular_helper_after_last_wire_gate(
    n: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Call the triangular helper after work on its last input wire."""
    register = qmc.qubit_array(n, "register")
    register[n - 1] = qmc.h(register[n - 1])
    return _triangular_input_helper(register)


@qmc.qkernel
def _unrolled_four_qubit_triangle() -> qmc.Vector[qmc.Qubit]:
    """Apply the six four-qubit triangular CX pairs without range loops."""
    register = qmc.qubit_array(4, "register")
    register[0], register[1] = qmc.cx(register[0], register[1])
    register[0], register[2] = qmc.cx(register[0], register[2])
    register[0], register[3] = qmc.cx(register[0], register[3])
    register[1], register[2] = qmc.cx(register[1], register[2])
    register[1], register[3] = qmc.cx(register[1], register[3])
    register[2], register[3] = qmc.cx(register[2], register[3])
    return register


@qmc.qkernel
def _conditional_triangular_helper_after_last_wire_gate(
    n: qmc.UInt,
    flag: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Conditionally call the helper after work on its last input wire."""
    register = qmc.qubit_array(n, "register")
    register[n - 1] = qmc.h(register[n - 1])
    if flag:
        register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _repeat_triangular_helper(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Call the triangular helper twice on the same caller-owned register."""
    register = qmc.qubit_array(n, "register")
    for _ in qmc.range(2):
        register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _sum_triangular_helper_with_unused_allocation() -> qmc.Vector[qmc.Qubit]:
    """Exercise the general range-sum path around a triangular helper."""
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(2):
        _scratch = qmc.qubit_array(index, "scratch")
        register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _conditional_inner_repeat_inside_outer_sum(
    flag: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Exercise distinct inner- and outer-loop synchronization guards."""
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(2):
        _scratch = qmc.qubit_array(index, "scratch")
        if flag:
            for _ in qmc.range(2):
                register = _triangular_input_helper(register)
        else:
            register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _single_active_triangle_inside_outer_sum() -> qmc.Vector[qmc.Qubit]:
    """Run a triangular helper in only one of two outer iterations."""
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(2):
        _scratch = qmc.qubit_array(index, "scratch")
        if index == 1:
            register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _prior_iteration_work_before_triangle() -> qmc.Vector[qmc.Qubit]:
    """Advance one input wire before a later triangular helper iteration."""
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(2):
        if index == 0:
            register[3] = qmc.h(register[3])
        if index == 1:
            register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _disjoint_iteration_work_before_triangle() -> qmc.Vector[qmc.Qubit]:
    """Run unrelated work before a later triangular helper iteration."""
    unrelated = qmc.qubit("unrelated")
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(2):
        if index == 0:
            unrelated = qmc.h(unrelated)
        if index == 1:
            register = _triangular_input_helper(register)
    return register


@qmc.qkernel
def _different_wires_across_iterations() -> qmc.Vector[qmc.Qubit]:
    """Use different fixed wires in separate conditional iterations."""
    register = qmc.qubit_array(2, "register")
    for index in qmc.range(2):
        if index == 0:
            register[0] = qmc.h(register[0])
        if index == 1:
            register[1] = qmc.x(register[1])
    return register


@qmc.qkernel
def _fixed_nonuniform_body_once() -> qmc.Vector[qmc.Qubit]:
    """Apply one two-wire body whose exit layers are nonuniform."""
    register = qmc.qubit_array(2, "register")
    register[0] = qmc.h(register[0])
    register[0], register[1] = qmc.cx(register[0], register[1])
    register[1] = qmc.h(register[1])
    return register


@qmc.qkernel
def _fixed_nonuniform_body_repeated() -> qmc.Vector[qmc.Qubit]:
    """Repeat a fixed two-wire body whose exit layers are nonuniform."""
    register = qmc.qubit_array(2, "register")
    for _ in qmc.range(2):
        register[0] = qmc.h(register[0])
        register[0], register[1] = qmc.cx(register[0], register[1])
        register[1] = qmc.h(register[1])
    return register


@qmc.qkernel
def _if_else_uses_different_wires() -> qmc.Vector[qmc.Qubit]:
    """Select different fixed wires across three concrete iterations."""
    register = qmc.qubit_array(2, "register")
    for index in qmc.range(3):
        if index < 2:
            register[0] = qmc.h(register[0])
        else:
            register[1] = qmc.x(register[1])
    return register


@qmc.qkernel
def _single_wire_across_iterations() -> qmc.Vector[qmc.Qubit]:
    """Serialize conditional iteration work on one precise physical wire."""
    register = qmc.qubit_array(1, "register")
    for index in qmc.range(2):
        if index == 0:
            register[0] = qmc.h(register[0])
        else:
            register[0] = qmc.x(register[0])
    return register


@qmc.qkernel
def _repeated_owner_wide_pauli_evolution(
    hamiltonian: qmc.Observable,
) -> qmc.Vector[qmc.Qubit]:
    """Repeat a nonuniform whole-register operation through a range loop."""
    register = qmc.qubit_array(2, "register")
    for _ in qmc.range(2):
        register = qmc.pauli_evolve(register, hamiltonian, qmc.float_(0.5))
    return register


@qmc.qkernel
def _explicit_owner_wide_pauli_evolution(
    hamiltonian: qmc.Observable,
) -> qmc.Vector[qmc.Qubit]:
    """Repeat a nonuniform whole-register operation without a range loop."""
    register = qmc.qubit_array(2, "register")
    register = qmc.pauli_evolve(register, hamiltonian, qmc.float_(0.5))
    register = qmc.pauli_evolve(register, hamiltonian, qmc.float_(0.5))
    return register


def _assert_symbolically_equal(actual: sp.Expr, expected: sp.Expr) -> None:
    """Assert that two symbolic resource expressions are equivalent.

    Args:
        actual (sp.Expr): Expression produced by resource estimation.
        expected (sp.Expr): Independently derived reference expression.

    Raises:
        AssertionError: If simplification does not prove equality.
    """
    assert sp.simplify(actual - expected) == 0


def _has_loop_fallback_assumption(estimate: qmc.ResourceEstimate) -> bool:
    """Return whether an estimate records the symbolic-loop fallback.

    Args:
        estimate (qmc.ResourceEstimate): Estimate whose assumptions to inspect.

    Returns:
        bool: Whether a loop fallback assumption is active.
    """
    return any(assumption.source == "for" for assumption in estimate.assumptions)


def test_concrete_projection_keeps_iteration_specific_entry_guards() -> None:
    """A concrete loop guard is attached only to its corresponding wire."""
    loop_symbol = sp.Dummy("i", integer=True)
    estimate = dataclasses.replace(
        qmc.ResourceEstimate.zero(),
        _dependency_synchronized_entry_conditions={
            ("owner", loop_symbol): sp.Eq(loop_symbol, 1)
        },
    )

    projected = _project_dependency_metadata_over_symbol(
        estimate,
        loop_symbol,
        start=0,
        step=1,
        iterations=3,
    )

    assert projected._dependency_synchronized_entry_conditions == {
        ("owner", 1): sp.true
    }


def test_loop_sum_does_not_treat_entry_guard_binder_as_public() -> None:
    """A scheduler-only loop guard takes the range path and binds its index."""
    loop_symbol = sp.Dummy("i", integer=True)
    estimate = dataclasses.replace(
        qmc.ResourceEstimate.zero(),
        _dependency_keys=frozenset({("owner", 0)}),
        _dependency_completion={("owner", 0): sp.Integer(1)},
        _dependency_completion_uniform=False,
        _dependency_synchronized_entry_conditions={("owner", 0): sp.Eq(loop_symbol, 1)},
    )

    summed = estimate.sum_over(
        loop_symbol,
        sp.Integer(0),
        sp.Integer(2),
        sp.Integer(1),
    )

    assert summed.parameters == {}
    assert summed.quality is qmc.EstimateQuality.EXACT
    assert not any(
        "different layers" in assumption.message for assumption in summed.assumptions
    )


def test_loop_sum_binds_completion_only_induction_symbols() -> None:
    """A binder used only by completion metadata cannot escape the range."""
    loop_symbol = sp.Dummy("i", integer=True)
    completion = sp.Piecewise(
        (sp.Integer(1), sp.Eq(loop_symbol, 1)),
        (sp.Integer(0), True),
    )
    estimate = dataclasses.replace(
        qmc.ResourceEstimate.zero(),
        depth=qmc.DepthResources(
            depth=sp.Integer(1),
            gate_depth=sp.Integer(1),
        ),
        _dependency_keys=frozenset({("owner", 0)}),
        _dependency_reads=frozenset({("owner", 0)}),
        _dependency_writes=frozenset({("owner", 0)}),
        _dependency_completion={("owner", 0): completion},
        _dependency_completion_uniform=False,
    )

    summed = estimate.sum_over(
        loop_symbol,
        sp.Integer(0),
        sp.Integer(2),
        sp.Integer(1),
    )

    assert summed.parameters == {}
    assert summed._dependency_completion == {("owner", 0): 2}
    assert not any(
        loop_symbol in value.free_symbols
        for value in (summed._dependency_completion or {}).values()
    )


def test_symbolic_triangular_cx_uses_exact_critical_path() -> None:
    """A complete pair loop has C(n, 2) gates but critical path 2n - 3."""
    estimate = _triangular_cx.estimate_resources()
    n = estimate.parameters["n"]

    _assert_symbolically_equal(estimate.gates.total, n * (n - 1) / 2)
    _assert_symbolically_equal(estimate.gates.two_qubit, n * (n - 1) / 2)
    assert estimate.depth.depth == sp.Max(0, 2 * n - 3)
    assert estimate.depth.gate_depth == sp.Max(0, 2 * n - 3)
    assert estimate.depth.clifford_depth == sp.Max(0, 2 * n - 3)
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_loop_fallback_assumption(estimate)


def test_symbolic_triangle_matches_an_independently_unrolled_schedule() -> None:
    """The symbolic proof agrees with an explicit four-qubit gate schedule."""
    symbolic = _triangular_cx.estimate_resources(inputs={"n": 4})
    unrolled = _unrolled_four_qubit_triangle.estimate_resources()

    assert symbolic.gates == unrolled.gates
    assert symbolic.depth == unrolled.depth
    assert symbolic.depth.depth == 5
    assert symbolic.quality is qmc.EstimateQuality.EXACT


def test_legacy_inner_loop_without_explicit_step_uses_default_one() -> None:
    """A two-operand inner ForOperation keeps the triangular proof robust."""
    block = _triangular_cx.build()
    outer = block.operations[1]
    assert isinstance(outer, ForOperation)
    inner = next(
        operation
        for operation in outer.operations
        if isinstance(operation, ForOperation)
    )
    legacy_inner = dataclasses.replace(inner, operands=inner.operands[:2])
    legacy_outer = dataclasses.replace(
        outer,
        operations=[
            legacy_inner if operation is inner else operation
            for operation in outer.operations
        ],
    )
    legacy_block = dataclasses.replace(
        block,
        operations=[
            legacy_outer if operation is outer else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(legacy_block, inputs={"n": 4})

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 5
    assert estimate.quality is qmc.EstimateQuality.EXACT


@pytest.mark.parametrize(
    ("n", "expected_depth"),
    [(0, 0), (1, 0), (2, 1), (4, 5), (5, 7)],
)
def test_triangular_cx_specialization_matches_direct_inputs(
    n: int,
    expected_depth: int,
) -> None:
    """Direct input and late substitution preserve the exact pair-loop result."""
    symbolic = _triangular_cx.estimate_resources()
    direct = _triangular_cx.estimate_resources(inputs={"n": n})
    substituted = symbolic.substitute(n=n)

    assert direct.gates == substituted.gates
    assert direct.depth == substituted.depth
    assert direct.depth.depth == expected_depth
    assert direct.gates.total == n * (n - 1) // 2
    assert direct.quality is qmc.EstimateQuality.EXACT
    assert substituted.quality is qmc.EstimateQuality.EXACT
    assert not _has_loop_fallback_assumption(direct)
    assert not _has_loop_fallback_assumption(substituted)


def test_flat_shared_wire_loop_is_exact() -> None:
    """A proven loop-invariant anchor serializes every iteration exactly."""
    estimate = _shared_anchor_cx.estimate_resources()
    n = estimate.parameters["n"]
    expected = sp.Max(0, n - 1)

    assert estimate.gates.total == expected
    assert estimate.depth.depth == expected
    assert estimate.depth.gate_depth == expected
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_loop_fallback_assumption(estimate)


def test_shifted_triangular_bounds_keep_the_same_exact_formula() -> None:
    """A constant register offset does not change pair-loop scheduling."""
    estimate = _shifted_triangular_cx.estimate_resources()
    n = estimate.parameters["n"]

    _assert_symbolically_equal(estimate.gates.total, n * (n - 1) / 2)
    assert estimate.depth.depth == sp.Max(0, 2 * n - 3)
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_loop_fallback_assumption(estimate)


def test_triangular_rzz_applies_the_path_to_rotation_depths() -> None:
    """The exact pair path is applied to every active RZZ depth category."""
    estimate = _triangular_rzz.estimate_resources()
    n = estimate.parameters["n"]
    expected = sp.Max(0, 2 * n - 3)

    _assert_symbolically_equal(estimate.gates.total, n * (n - 1) / 2)
    assert estimate.depth.depth == expected
    assert estimate.depth.gate_depth == expected
    assert estimate.depth.rotation_depth == expected
    assert estimate.depth.non_clifford_depth == expected
    assert estimate.depth.clifford_depth == 0
    assert estimate.depth.t_depth == 0
    assert estimate.quality is qmc.EstimateQuality.EXACT


@pytest.mark.parametrize(
    "kernel",
    [
        _triangular_with_extra_gate,
        _triangular_with_control,
        _nontriangular_inner_start,
        _nontriangular_inner_step,
    ],
)
def test_nonmatching_pair_loops_keep_the_general_fallback(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """Near-miss loop bodies do not receive the narrow triangular proof."""
    estimate = kernel.estimate_resources()

    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert _has_loop_fallback_assumption(estimate)


def test_prior_overlapping_work_keeps_a_safe_conservative_completion() -> None:
    """Earlier work on a late wire uses the safe aggregate loop completion."""
    estimate = _triangular_after_last_wire_gate.estimate_resources()
    specialized = estimate.substitute(n=4)

    assert specialized.depth.depth == 6
    assert specialized.quality is qmc.EstimateQuality.CONSERVATIVE
    assert specialized.depth.depth >= 5


def test_prior_disjoint_work_does_not_degrade_pair_loop_exactness() -> None:
    """Earlier work on another owner remains parallel with the pair schedule."""
    estimate = _triangular_after_disjoint_gate.estimate_resources()
    specialized = estimate.substitute(n=4)

    assert specialized.depth.depth == 5
    assert specialized.quality is qmc.EstimateQuality.EXACT


def test_later_early_wire_use_discloses_aggregate_completion_loss() -> None:
    """Using an early-finishing wire after the loop marks its bound conservative."""
    estimate = _triangular_then_early_wire_gate.estimate_resources()
    specialized = estimate.substitute(n=4)

    assert specialized.depth.depth == 6
    assert specialized.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        "aggregate latency" in assumption.message
        for assumption in specialized.assumptions
    )


def test_input_register_helper_preserves_pair_loop_owner_mapping() -> None:
    """A helper call maps its triangular footprint to the caller register."""
    estimate = _triangular_through_helper.estimate_resources()
    n = estimate.parameters["n"]

    _assert_symbolically_equal(estimate.gates.total, n * (n - 1) / 2)
    assert estimate.depth.depth == sp.Max(0, 2 * n - 3)
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_loop_fallback_assumption(estimate)


def test_helper_call_maps_synchronized_entry_requirement_to_caller() -> None:
    """A helper maps its synchronized-entry requirement to caller inputs."""
    estimate = _triangular_helper_after_last_wire_gate.estimate_resources()
    specialized = estimate.substitute(n=4)

    assert specialized.depth.depth == 6
    assert specialized.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source == "dependency scheduler"
        and "synchronized input wires" in assumption.message
        for assumption in specialized.assumptions
    )


@pytest.mark.parametrize(
    ("flag", "expected_depth", "expected_quality"),
    [
        (0, 1, qmc.EstimateQuality.EXACT),
        (1, 6, qmc.EstimateQuality.CONSERVATIVE),
    ],
)
def test_conditional_helper_guards_synchronized_entry_requirement(
    flag: int,
    expected_depth: int,
    expected_quality: qmc.EstimateQuality,
) -> None:
    """An inactive helper branch does not retain its entry requirement."""
    symbolic = _conditional_triangular_helper_after_last_wire_gate.estimate_resources()
    direct = _conditional_triangular_helper_after_last_wire_gate.estimate_resources(
        inputs={"n": 4, "flag": flag}
    )
    substituted = symbolic.substitute(n=4, flag=flag)

    assert direct.depth.depth == expected_depth
    assert substituted.depth.depth == expected_depth
    assert direct.quality is expected_quality
    assert substituted.quality is expected_quality


def test_repeated_helper_discloses_internal_entry_desynchronization() -> None:
    """A second helper call safely bounds the first call's nonuniform exit."""
    estimate = _repeat_triangular_helper.estimate_resources(inputs={"n": 4})

    assert estimate.gates.total == 12
    assert estimate.depth.depth == 10
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source == "repeat" and "different layers" in assumption.message
        for assumption in estimate.assumptions
    )


def test_general_range_sum_discloses_internal_entry_desynchronization() -> None:
    """A binder-dependent zero-depth allocation cannot bypass repeat quality."""
    estimate = _sum_triangular_helper_with_unused_allocation.estimate_resources()

    assert estimate.gates.total == 12
    assert estimate.depth.depth == 10
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source == "for" and "over-serialize" in assumption.message
        for assumption in estimate.assumptions
    )


def test_outer_loop_keeps_its_guard_when_inner_repeat_is_inactive() -> None:
    """An inactive inner-repeat fact cannot suppress the outer-loop bound."""
    symbolic = _conditional_inner_repeat_inside_outer_sum.estimate_resources()
    specialized = symbolic.substitute(flag=0)

    assert specialized.gates.total == 12
    assert specialized.depth.depth == 10
    assert specialized.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source == "for" and "over-serialize" in assumption.message
        for assumption in specialized.assumptions
    )


def test_single_active_iteration_needs_no_repeat_bound() -> None:
    """One active pair schedule does not leave state for another iteration."""
    estimate = _single_active_triangle_inside_outer_sum.estimate_resources()

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 5
    assert estimate.parameters == {}
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not any(
        assumption.source == "for" and "over-serialize" in assumption.message
        for assumption in estimate.assumptions
    )


def test_prior_iteration_work_makes_sequential_triangle_bound_conservative() -> None:
    """Earlier work cannot leave a sequential aggregate labeled exact."""
    estimate = _prior_iteration_work_before_triangle.estimate_resources()

    assert estimate.gates.total == 7
    assert estimate.depth.depth == 6
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in estimate.assumptions)


def test_disjoint_iteration_work_still_exposes_scalar_sum_loss() -> None:
    """Disjoint earlier work still makes the scalar sequential sum an upper bound."""
    estimate = _disjoint_iteration_work_before_triangle.estimate_resources()

    assert estimate.gates.total == 7
    assert estimate.depth.depth == 6
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in estimate.assumptions)


def test_varying_nonuniform_loop_completion_is_not_labeled_exact() -> None:
    """Different per-iteration wires retain a safe sequential upper bound."""
    estimate = _different_wires_across_iterations.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in estimate.assumptions)


def test_fixed_nonuniform_body_repetition_is_not_labeled_exact() -> None:
    """A binder-independent nonuniform body still exposes its scalar bound."""
    estimate = _fixed_nonuniform_body_repeated.estimate_resources()

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 6
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in estimate.assumptions)


def test_public_repeat_discloses_nonuniform_scalar_depth_bound() -> None:
    """Public repeat does not label a nonuniform scalar depth product exact."""
    once = _fixed_nonuniform_body_once.estimate_resources()
    repeated = once.repeat(2)

    assert once.depth.depth == 3
    assert repeated.depth.depth == 6
    assert repeated.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "repeat" for assumption in repeated.assumptions)


def test_public_sum_discloses_nonuniform_scalar_depth_bound() -> None:
    """Public range sum does not label a nonuniform scalar depth sum exact."""
    loop_symbol = sp.Dummy("iteration", integer=True, nonnegative=True)
    once = _fixed_nonuniform_body_once.estimate_resources()
    summed = once.sum_over(loop_symbol, 0, 2)

    assert summed.depth.depth == 6
    assert summed.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "sum_over" for assumption in summed.assumptions)


def test_public_repetition_counts_category_only_depth_as_active() -> None:
    """Category-only depth cannot bypass nonuniform repetition disclosure."""
    loop_symbol = sp.Dummy("iteration", integer=True, nonnegative=True)
    category_only = dataclasses.replace(
        qmc.ResourceEstimate.zero(),
        depth=qmc.DepthResources(gate_depth=2),
        _dependency_keys=frozenset({("owner", 0), ("owner", 1)}),
        _dependency_reads=frozenset({("owner", 0), ("owner", 1)}),
        _dependency_writes=frozenset({("owner", 0), ("owner", 1)}),
        _dependency_completion={
            ("owner", 0): sp.Integer(1),
            ("owner", 1): sp.Integer(2),
        },
        _dependency_completion_uniform=False,
    )

    repeated = category_only.repeat(2)
    summed = category_only.sum_over(loop_symbol, 0, 2)

    for estimate in (repeated, summed):
        assert estimate.depth.gate_depth == 4
        assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
        assert estimate.assumptions


def test_public_sum_discloses_iteration_dependent_quantum_wires() -> None:
    """A public sum cannot label a scalar depth over disjoint wires exact."""
    loop_symbol = sp.Dummy("iteration", integer=True, nonnegative=True)
    for completion_uniform in (True, False):
        indexed = dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            depth=qmc.DepthResources(depth=1, gate_depth=1),
            _dependency_keys=frozenset({("owner", loop_symbol)}),
            _dependency_reads=frozenset({("owner", loop_symbol)}),
            _dependency_writes=frozenset({("owner", loop_symbol)}),
            _dependency_completion={("owner", loop_symbol): 1},
            _dependency_completion_uniform=completion_uniform,
        )

        summed = indexed.sum_over(loop_symbol, 0, 2)

        assert summed.depth.depth == 2
        assert summed.quality is qmc.EstimateQuality.CONSERVATIVE
        assert any(assumption.source == "sum_over" for assumption in summed.assumptions)


def test_if_else_iteration_footprints_are_not_labeled_exact() -> None:
    """Exclusive branches on different wires retain a conservative sum."""
    estimate = _if_else_uses_different_wires.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in estimate.assumptions)


def test_single_precise_dependency_proves_sequential_depth_exact() -> None:
    """One shared physical wire makes the sequential scalar sum exact."""
    estimate = _single_wire_across_iterations.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not any(assumption.source == "for" for assumption in estimate.assumptions)


def test_owner_wide_vector_dependency_is_not_treated_as_one_qubit() -> None:
    """A whole-vector wildcard cannot use the one-qubit exactness proof."""
    inputs = {"hamiltonian": qmo.X(0) + qmo.X(1)}
    loop_estimate = _repeated_owner_wide_pauli_evolution.estimate_resources(
        inputs=inputs
    )
    explicit_estimate = _explicit_owner_wide_pauli_evolution.estimate_resources(
        inputs=inputs
    )

    assert loop_estimate.gates.total == explicit_estimate.gates.total
    assert loop_estimate.depth.depth == explicit_estimate.depth.depth
    assert loop_estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in loop_estimate.assumptions)


def test_owner_wide_width_one_dependency_proves_public_repeat_exact() -> None:
    """An owner-wide key is scalar only when its captured width proves one."""
    scalar = dataclasses.replace(
        qmc.ResourceEstimate.zero(),
        depth=qmc.DepthResources(depth=1, gate_depth=1),
        _input_sizes={"owner": sp.Integer(1)},
        _dependency_keys=frozenset({("owner", None)}),
        _dependency_reads=frozenset({("owner", None)}),
        _dependency_writes=frozenset({("owner", None)}),
        _dependency_completion={("owner", None): sp.Integer(1)},
        _dependency_completion_uniform=False,
    )

    repeated = scalar.repeat(2)

    assert repeated.depth.depth == 2
    assert repeated.quality is qmc.EstimateQuality.EXACT
    assert repeated.assumptions == ()


def test_for_items_rejects_item_dependent_private_metadata() -> None:
    """A symbolic items-loop binder cannot escape through private metadata."""
    item = sp.Dummy("item", integer=True)
    condition = sp.Eq(item, 1)
    estimates = (
        dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            _dependency_synchronized_entry_conditions={("owner", 0): condition},
        ),
        dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            _dependency_completion={("owner", 0): item},
        ),
        dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            _dependency_keys=frozenset({("owner", item)}),
        ),
        dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            _measurement_taint_conditions={"bit": condition},
        ),
        dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            _allocation_sites={"site": item},
        ),
        dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            trace=ResourceTraceNode(
                "item-dependent",
                "test",
                active_when=condition,
            ),
        ),
    )
    for estimate in estimates:
        with pytest.raises(NotImplementedError, match="current item key or value"):
            _ForItemsContextInterpreter._ensure_for_items_resource_independent(
                estimate,
                frozenset({item}),
            )
