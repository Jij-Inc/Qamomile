"""Regression tests for exact symbolic affine-loop depth scheduling."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator._dependency_indices import (
    _UNKNOWN_WIRE_INDEX,
    _WIRE_RANGE_OFFSET,
    _wire_index_covers,
    _wire_index_relation,
    _WireRangeIndex,
    _WireRelation,
)
from qamomile.circuit.estimator._dependency_synchronization import (
    _SynchronizedEntryCertificate,
)
from qamomile.circuit.estimator._scheduling import (
    _SYNCHRONIZED_ENTRY_EXACT_EVENT_LIMIT,
    _synchronized_entry_overlap_condition,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.operation.operation import Operation

_GENERIC_SYMBOLIC_LOOP_FALLBACK = (
    "symbolic loop depth is sequential because disjoint iteration footprints "
    "could not be proven"
)


@qmc.qkernel
def _adjacent_cx(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply a CX chain to adjacent elements of one symbolic register."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(1, n):
        register[index - 1], register[index] = qmc.cx(
            register[index - 1],
            register[index],
        )
    return register


@qmc.qkernel
def _positive_strided_cx(iterations: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply a consecutive CX chain whose range stride is positive two."""
    register = qmc.qubit_array(2 * iterations + 1, "register")
    for index in qmc.range(0, 2 * iterations, 2):
        register[index], register[index + 2] = qmc.cx(
            register[index],
            register[index + 2],
        )
    return register


@qmc.qkernel
def _negative_strided_cx(iterations: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply a consecutive CX chain whose range stride is negative two."""
    register = qmc.qubit_array(2 * iterations + 1, "register")
    for index in qmc.range(2 * iterations, 0, -2):
        register[index], register[index - 2] = qmc.cx(
            register[index],
            register[index - 2],
        )
    return register


@qmc.qkernel
def _adjacent_rzz(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply an RZZ chain to adjacent elements of one symbolic register."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(1, n):
        register[index - 1], register[index] = qmc.rzz(
            register[index - 1],
            register[index],
            0.125,
        )
    return register


@qmc.qkernel
def _adjacent_helper(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply an adjacent CX chain to a caller-owned register."""
    n = register.shape[0]
    for index in qmc.range(1, n):
        register[index - 1], register[index] = qmc.cx(
            register[index - 1],
            register[index],
        )
    return register


@qmc.qkernel
def _adjacent_through_helper(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Pass a fresh symbolic register through an adjacent-chain helper."""
    return _adjacent_helper(qmc.qubit_array(n, "register"))


@qmc.qkernel
def _adjacent_after_first_wire_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Run an adjacent chain after prior work on its first input wire."""
    register = qmc.qubit_array(n, "register")
    register[0] = qmc.h(register[0])
    return _adjacent_helper(register)


@qmc.qkernel
def _adjacent_after_disjoint_owner(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Run an adjacent chain beside prior work on an unrelated owner."""
    unrelated = qmc.h(qmc.qubit("unrelated"))
    register = _adjacent_helper(qmc.qubit_array(n, "register"))
    unrelated = qmc.x(unrelated)
    return register


@qmc.qkernel
def _adjacent_then_first_wire_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use an early-finishing register wire after an adjacent chain."""
    register = _adjacent_helper(qmc.qubit_array(n, "register"))
    register[0] = qmc.h(register[0])
    return register


@qmc.qkernel
def _conditional_adjacent_after_first_wire_gate(
    n: qmc.UInt,
    flag: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Conditionally run an adjacent chain after prior overlapping work."""
    register = qmc.qubit_array(n, "register")
    register[0] = qmc.h(register[0])
    if flag:
        register = _adjacent_helper(register)
    return register


@qmc.qkernel
def _repeat_adjacent_helper(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply an adjacent-chain helper twice to the same register."""
    register = qmc.qubit_array(n, "register")
    for _ in qmc.range(2):
        register = _adjacent_helper(register)
    return register


@qmc.qkernel
def _sum_adjacent_helper_with_unused_allocation() -> qmc.Vector[qmc.Qubit]:
    """Exercise a general range sum around an adjacent-chain helper."""
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(2):
        _scratch = qmc.qubit_array(index, "scratch")
        register = _adjacent_helper(register)
    return register


@qmc.qkernel
def _piecewise_injective_near_miss(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use equal-slope Piecewise branches that collide at zero and one."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(n):
        target = index
        if index > 0:
            target = index - 1
        register[target] = qmc.h(register[target])
    return register


@qmc.qkernel
def _adjacent_with_extra_gate(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Add a second gate that excludes the one-gate affine certificate."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(1, n):
        register[index - 1], register[index] = qmc.cx(
            register[index - 1],
            register[index],
        )
        register[index] = qmc.h(register[index])
    return register


@qmc.qkernel
def _controlled_adjacent_cx(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use an extra coherent control that excludes the two-wire proof."""
    control = qmc.qubit("control")
    register = qmc.qubit_array(n, "register")
    controlled_cx = qmc.control(qmc.cx)
    for index in qmc.range(1, n):
        control, register[index - 1], register[index] = controlled_cx(
            control,
            register[index - 1],
            register[index],
        )
    return register


@qmc.qkernel
def _nonconsecutive_pair(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use a wire delta different from the loop stride."""
    register = qmc.qubit_array(n + 1, "register")
    for index in qmc.range(0, n - 1):
        register[index], register[index + 2] = qmc.cx(
            register[index],
            register[index + 2],
        )
    return register


@qmc.qkernel
def _parallel_loop_between_endpoint_work(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Surround an injective parallel loop with work on opposite endpoints."""
    register = qmc.qubit_array(n, "register")
    register[0] = qmc.h(register[0])
    for index in qmc.range(n):
        register[index] = qmc.x(register[index])
    register[n - 1] = qmc.h(register[n - 1])
    return register


@qmc.qkernel
def _full_parallel_write_then_adjacent(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Synchronize a register before applying one adjacent CX chain."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(n):
        register[index] = qmc.x(register[index])
    return _adjacent_helper(register)


@qmc.qkernel
def _reverse_full_parallel_write_then_adjacent(
    n: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Synchronize a register in reverse order before an adjacent CX chain."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(n - 1, -1, -1):
        register[index] = qmc.x(register[index])
    return _adjacent_helper(register)


@qmc.qkernel
def _partial_parallel_write_then_full(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Leave one register endpoint unsynchronized before a full write."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(n - 1):
        register[index] = qmc.x(register[index])
    for index in qmc.range(n):
        register[index] = qmc.h(register[index])
    return register


@qmc.qkernel
def _full_parallel_write_then_strided_adjacent(
    iterations: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Synchronize a dense register before a stride-two CX chain."""
    register = qmc.qubit_array(2 * iterations + 1, "register")
    for index in qmc.range(2 * iterations + 1):
        register[index] = qmc.x(register[index])
    for index in qmc.range(0, 2 * iterations, 2):
        register[index], register[index + 2] = qmc.cx(
            register[index],
            register[index + 2],
        )
    return register


@qmc.qkernel
def _symbolic_stride_write_then_chain(
    iterations: qmc.UInt,
    stride: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply one sparse full write and chain with an input-dependent stride."""
    step = stride + 1
    register = qmc.qubit_array(step * iterations + 1, "register")
    for index in qmc.range(iterations + 1):
        register[step * index] = qmc.x(register[step * index])
    for index in qmc.range(iterations):
        register[step * index], register[step * (index + 1)] = qmc.cx(
            register[step * index],
            register[step * (index + 1)],
        )
    return register


def _has_for_assumption(estimate: qmc.ResourceEstimate) -> bool:
    """Return whether an estimate retains a range-loop assumption.

    Args:
        estimate (qmc.ResourceEstimate): Estimate whose assumptions to inspect.

    Returns:
        bool: Whether any active assumption has ``for`` as its source.
    """
    return any(assumption.source == "for" for assumption in estimate.assumptions)


def _has_generic_symbolic_loop_fallback(estimate: qmc.ResourceEstimate) -> bool:
    """Return whether an estimate fell back before affine classification.

    Args:
        estimate (qmc.ResourceEstimate): Estimate whose assumptions to inspect.

    Returns:
        bool: Whether the generic unresolved-symbolic-footprint fallback is
        present.
    """
    return any(
        _GENERIC_SYMBOLIC_LOOP_FALLBACK in assumption.message
        for assumption in estimate.assumptions
    )


def _assert_symbolically_equal(actual: sp.Expr, expected: sp.Expr) -> None:
    """Assert symbolic equality after simplification.

    Args:
        actual (sp.Expr): Expression produced by resource estimation.
        expected (sp.Expr): Independently derived reference expression.

    Raises:
        AssertionError: If simplification does not prove equality.
    """
    assert sp.simplify(actual - expected) == 0


def test_adjacent_cx_is_exact_for_symbolic_inputs_and_substitution() -> None:
    """All public specialization routes retain the exact adjacent-chain proof."""
    symbolic = _adjacent_cx.estimate_resources()
    n = symbolic.parameters["n"]

    _assert_symbolically_equal(symbolic.gates.total, sp.Max(0, n - 1))
    assert symbolic.depth.depth == sp.Max(0, n - 1)
    assert symbolic.depth.gate_depth == sp.Max(0, n - 1)
    assert symbolic.depth.clifford_depth == sp.Max(0, n - 1)
    assert symbolic.quality is qmc.EstimateQuality.EXACT
    assert not _has_for_assumption(symbolic)

    for concrete_n in (0, 1, 2, 4):
        direct = _adjacent_cx.estimate_resources(inputs={"n": concrete_n})
        substituted = symbolic.substitute(n=concrete_n)
        expected = max(0, concrete_n - 1)

        assert direct.gates == substituted.gates
        assert direct.depth == substituted.depth
        assert direct.depth.depth == expected
        assert direct.quality is qmc.EstimateQuality.EXACT
        assert substituted.quality is qmc.EstimateQuality.EXACT
        assert not _has_for_assumption(direct)
        assert not _has_for_assumption(substituted)


@pytest.mark.parametrize(
    "kernel",
    [_positive_strided_cx, _negative_strided_cx],
)
def test_signed_strided_chain_uses_the_actual_range_step(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """Signed non-unit strides prove consecutiveness in iteration order."""
    estimate = kernel.estimate_resources()
    iterations = estimate.parameters["iterations"]

    _assert_symbolically_equal(estimate.gates.total, iterations)
    assert estimate.depth.depth == iterations
    assert estimate.depth.gate_depth == iterations
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_for_assumption(estimate)


def test_adjacent_rzz_projects_every_active_depth_category() -> None:
    """The affine path length applies to every RZZ depth category."""
    estimate = _adjacent_rzz.estimate_resources()
    n = estimate.parameters["n"]
    expected = sp.Max(0, n - 1)

    _assert_symbolically_equal(estimate.gates.total, expected)
    assert estimate.depth.depth == expected
    assert estimate.depth.gate_depth == expected
    assert estimate.depth.rotation_depth == expected
    assert estimate.depth.non_clifford_depth == expected
    assert estimate.depth.clifford_depth == 0
    assert estimate.depth.t_depth == 0
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_helper_call_preserves_affine_owner_mapping() -> None:
    """A helper maps its exact adjacent footprint to the caller register."""
    estimate = _adjacent_through_helper.estimate_resources()
    n = estimate.parameters["n"]

    assert estimate.depth.depth == sp.Max(0, n - 1)
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_for_assumption(estimate)


def test_first_frontier_work_keeps_serial_chain_exact() -> None:
    """Prior work confined to the first gate frontier remains exact."""
    estimate = _adjacent_after_first_wire_gate.estimate_resources(inputs={"n": 4})

    assert estimate.depth.depth == 4
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()
    assert not _has_generic_symbolic_loop_fallback(estimate)


def test_prior_disjoint_owner_keeps_affine_loop_exact() -> None:
    """Work on another allocation owner remains parallel and exact."""
    estimate = _adjacent_after_disjoint_owner.estimate_resources(inputs={"n": 4})

    assert estimate.depth.depth == 3
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not _has_for_assumption(estimate)


def test_downstream_early_wire_reuse_discloses_completion_loss() -> None:
    """Aggregate affine-loop completion cannot make early-wire reuse exact."""
    estimate = _adjacent_then_first_wire_gate.estimate_resources(inputs={"n": 4})

    assert estimate.depth.depth == 4
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source == "dependency scheduler"
        for assumption in estimate.assumptions
    )
    assert not _has_generic_symbolic_loop_fallback(estimate)


@pytest.mark.parametrize(
    ("flag", "expected_depth", "expected_quality"),
    [
        (0, 1, qmc.EstimateQuality.EXACT),
        (1, 4, qmc.EstimateQuality.EXACT),
    ],
)
def test_conditional_affine_entry_requirement_is_guarded(
    flag: int,
    expected_depth: int,
    expected_quality: qmc.EstimateQuality,
) -> None:
    """An inactive affine-loop branch drops its synchronized-entry requirement."""
    symbolic = _conditional_adjacent_after_first_wire_gate.estimate_resources()
    direct = _conditional_adjacent_after_first_wire_gate.estimate_resources(
        inputs={"n": 4, "flag": flag}
    )
    substituted = symbolic.substitute(n=4, flag=flag)

    assert direct.depth.depth == expected_depth
    assert substituted.depth.depth == expected_depth
    assert direct.quality is expected_quality
    assert substituted.quality is expected_quality
    for estimate in (direct, substituted):
        assert not _has_generic_symbolic_loop_fallback(estimate)
        assert estimate.assumptions == ()


def test_repeated_affine_helper_discloses_nonuniform_exit_layers() -> None:
    """A second chain does not treat the first chain's exits as synchronized."""
    estimate = _repeat_adjacent_helper.estimate_resources(inputs={"n": 4})

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 6
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert {assumption.source for assumption in estimate.assumptions} >= {
        "repeat",
        "for",
    }
    assert not _has_generic_symbolic_loop_fallback(estimate)


def test_affine_helper_metadata_survives_general_range_sum() -> None:
    """A general range sum retains the adjacent chain's safe exit bound."""
    estimate = _sum_adjacent_helper_with_unused_allocation.estimate_resources()

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 6
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(assumption.source == "for" for assumption in estimate.assumptions)
    assert not _has_generic_symbolic_loop_fallback(estimate)


@pytest.mark.parametrize(
    "kernel",
    [
        _piecewise_injective_near_miss,
        _adjacent_with_extra_gate,
        _controlled_adjacent_cx,
        _nonconsecutive_pair,
    ],
)
def test_affine_near_misses_fail_closed(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """Unsupported lookalikes retain the safe general-loop fallback."""
    estimate = kernel.estimate_resources()

    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert _has_for_assumption(estimate)


def test_gate_result_order_near_miss_fails_closed() -> None:
    """A malformed result permutation cannot receive the serial certificate."""
    block = _adjacent_cx.build()
    loop = next(
        operation
        for operation in block.operations
        if isinstance(operation, ForOperation)
    )
    gate = next(
        operation
        for operation in loop.operations
        if isinstance(operation, GateOperation)
    )
    swapped_gate = dataclasses.replace(gate, results=list(reversed(gate.results)))
    swapped_loop = dataclasses.replace(
        loop,
        operations=[
            swapped_gate if operation is gate else operation
            for operation in loop.operations
        ],
    )
    malformed = dataclasses.replace(
        block,
        operations=[
            swapped_loop if operation is loop else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(malformed)

    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert _has_for_assumption(estimate)


def test_projected_parallel_loop_discloses_unsynchronized_entry_and_exit() -> None:
    """Parallel projection cannot claim exactness across uneven endpoints."""
    estimate = _parallel_loop_between_endpoint_work.estimate_resources(inputs={"n": 4})

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 3
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert estimate.assumptions


@pytest.mark.parametrize(
    "kernel",
    (
        _full_parallel_write_then_adjacent,
        _reverse_full_parallel_write_then_adjacent,
    ),
)
def test_full_parallel_write_resynchronizes_adjacent_chain(
    kernel: Callable[..., object],
) -> None:
    """A full affine write removes stale entry assumptions in either order."""
    symbolic = kernel.estimate_resources()

    assert symbolic.quality is qmc.EstimateQuality.EXACT
    assert symbolic.assumptions == ()
    for concrete_n in (0, 1, 4):
        direct = kernel.estimate_resources(inputs={"n": concrete_n})
        substituted = symbolic.substitute(n=concrete_n)
        assert direct.depth.depth == substituted.depth.depth
        assert direct.quality is qmc.EstimateQuality.EXACT
        assert substituted.quality is qmc.EstimateQuality.EXACT
        assert direct.assumptions == substituted.assumptions == ()


def test_partial_parallel_write_does_not_resynchronize_full_register() -> None:
    """A strict-subset write cannot clear a full-register entry premise."""
    symbolic = _partial_parallel_write_then_full.estimate_resources()
    direct = _partial_parallel_write_then_full.estimate_resources(inputs={"n": 4})
    substituted = symbolic.substitute(n=4)

    for estimate in (symbolic, direct, substituted):
        assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
        assert any(
            assumption.source == "dependency scheduler"
            and "synchronized input wires" in assumption.message
            for assumption in estimate.assumptions
        )


def test_dense_parallel_write_resynchronizes_sparse_affine_chain() -> None:
    """A dense full write covers every wire of a stride-two affine chain."""
    symbolic = _full_parallel_write_then_strided_adjacent.estimate_resources()

    assert symbolic.quality is qmc.EstimateQuality.EXACT
    assert symbolic.assumptions == ()
    for iterations in (0, 1, 4):
        direct = _full_parallel_write_then_strided_adjacent.estimate_resources(
            inputs={"iterations": iterations}
        )
        substituted = symbolic.substitute(iterations=iterations)
        assert direct.depth.depth == substituted.depth.depth
        assert direct.quality is qmc.EstimateQuality.EXACT
        assert substituted.quality is qmc.EstimateQuality.EXACT
        assert direct.assumptions == substituted.assumptions == ()


def test_matching_input_dependent_affine_slopes_preserve_exact_coverage() -> None:
    """Equal symbolic address strides prove the same coverage as direct input."""
    symbolic = _symbolic_stride_write_then_chain.estimate_resources()
    direct = _symbolic_stride_write_then_chain.estimate_resources(
        inputs={"iterations": 4, "stride": 1}
    )
    substituted = symbolic.substitute(iterations=4, stride=1)

    for estimate in (symbolic, direct, substituted):
        assert estimate.quality is qmc.EstimateQuality.EXACT
        assert estimate.assumptions == ()


def test_wire_range_coverage_is_directional_and_fail_closed() -> None:
    """Range containment handles orientation without widening overlap facts."""
    n = sp.Symbol("n", integer=True, nonnegative=True)
    full = _WireRangeIndex(_WIRE_RANGE_OFFSET, n)
    reverse_full = _WireRangeIndex(n - 1 - _WIRE_RANGE_OFFSET, n)
    prefix = _WireRangeIndex(_WIRE_RANGE_OFFSET, sp.Max(0, n - 1))
    suffix = _WireRangeIndex(_WIRE_RANGE_OFFSET + 1, sp.Max(0, n - 1))
    outside = _WireRangeIndex(n + _WIRE_RANGE_OFFSET, sp.Max(0, n - 1))
    dense = _WireRangeIndex(_WIRE_RANGE_OFFSET, 2 * n + 1)
    sparse = _WireRangeIndex(2 * _WIRE_RANGE_OFFSET, n + 1)
    symbolic_stride = sp.Symbol("stride", integer=True, positive=True)
    symbolic_cover = _WireRangeIndex(symbolic_stride * _WIRE_RANGE_OFFSET, n)
    symbolic_dense = _WireRangeIndex(
        symbolic_stride * _WIRE_RANGE_OFFSET,
        2 * n + 1,
    )
    symbolic_sparse = _WireRangeIndex(
        2 * symbolic_stride * _WIRE_RANGE_OFFSET,
        n + 1,
    )
    symbolic_mismatch = _WireRangeIndex(
        (symbolic_stride + 1) * _WIRE_RANGE_OFFSET,
        n,
    )

    assert _wire_index_covers(full, prefix)
    assert _wire_index_covers(full, suffix)
    assert _wire_index_covers(reverse_full, prefix)
    assert _wire_index_covers(reverse_full, suffix)
    assert not _wire_index_covers(prefix, full)
    assert not _wire_index_covers(full, outside)
    assert _wire_index_covers(dense, sparse)
    assert not _wire_index_covers(sparse, dense)
    assert _wire_index_covers(symbolic_dense, symbolic_sparse)
    assert not _wire_index_covers(symbolic_cover, symbolic_mismatch)
    assert not _wire_index_covers(None, prefix)
    assert not _wire_index_covers(None, None)
    assert not _wire_index_covers(_UNKNOWN_WIRE_INDEX, _UNKNOWN_WIRE_INDEX)
    assert _wire_index_relation(full, prefix) is _WireRelation.POSSIBLE_OVERLAP


@pytest.mark.parametrize("include_grouped_certificate", [False, True])
def test_dense_synchronized_entry_history_uses_compact_safe_guard(
    include_grouped_certificate: bool,
) -> None:
    """Dense conditional overlap avoids pairwise Boolean expansion."""
    size = _SYNCHRONIZED_ENTRY_EXACT_EVENT_LIMIT + 2
    key = ("owner", 0)
    scheduled: list[tuple[Operation, qmc.ResourceEstimate]] = []
    footprints = []
    activity_conditions = []
    symbols = []
    for index in range(size):
        symbol = sp.Symbol(f"active_{index}", integer=True)
        active = sp.Eq(symbol, 1)
        estimate = dataclasses.replace(
            qmc.ResourceEstimate.zero(),
            depth=qmc.DepthResources(depth=1, gate_depth=1),
            _dependency_keys=frozenset({key}),
            _dependency_completion={key: sp.Integer(1)},
            _dependency_completion_uniform=False,
            _dependency_synchronized_entry_conditions={key: active},
        )
        if include_grouped_certificate and index == size - 1:
            estimate = dataclasses.replace(
                estimate,
                _dependency_synchronized_entry_certificates=(
                    _SynchronizedEntryCertificate(
                        coverage=frozenset({key}),
                        frontier=frozenset({key}),
                        active_when=active,
                    ),
                ),
            )
        scheduled.append((cast(Operation, object()), estimate))
        footprints.append((frozenset({key}), frozenset({key})))
        activity_conditions.append(active)
        symbols.append(symbol)

    condition = _synchronized_entry_overlap_condition(
        scheduled,
        footprints,
        activity_conditions=activity_conditions,
    )
    one_active = {symbol: 0 for symbol in symbols}
    one_active[symbols[0]] = 1
    two_active = dict(one_active)
    two_active[symbols[1]] = 1

    assert condition.subs(one_active) is sp.false
    assert condition.subs(two_active) is sp.true
    assert len(str(condition)) < 5_000
