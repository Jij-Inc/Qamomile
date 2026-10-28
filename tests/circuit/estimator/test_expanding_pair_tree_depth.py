"""Regression tests for exact rooted expanding-pair tree scheduling."""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation


@qmc.qkernel
def _expanding_cx_tree(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Prepare a power-of-two GHZ state by doubling its active prefix.

    Args:
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared power-of-two register.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control],
                register[target],
            )
    return register


@qmc.qkernel
def _expanding_rzz_tree(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use T as the seed and RZZ as each expanding pair primitive.

    Args:
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the expanding RZZ schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.t(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.rzz(
                register[control],
                register[target],
                0.125,
            )
    return register


@qmc.qkernel
def _expanding_tree_with_scalar_binop(
    stages: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Keep a harmless scalar expression between the seed and outer loop.

    Args:
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared power-of-two register.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    unused = stages + 1
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control],
                register[target],
            )
    _unused = unused + 1
    return register


@qmc.qkernel
def _expanding_tree_input_helper(
    register: qmc.Vector[qmc.Qubit],
    stages: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply an expanding tree to a caller-owned register or exact view.

    Args:
        register (qmc.Vector[qmc.Qubit]): Caller-owned target register.
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated caller-owned register.
    """
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control],
                register[target],
            )
    return register


@qmc.qkernel
def _expanding_tree_through_helper(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Map an exact expanding-tree schedule through a helper call.

    Args:
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared power-of-two register.
    """
    register = qmc.qubit_array(2**stages, "register")
    return _expanding_tree_input_helper(register, stages)


@qmc.qkernel
def _expanding_tree_through_strided_view() -> qmc.Vector[qmc.Qubit]:
    """Map an expanding tree through a fixed positive-stride register view.

    Returns:
        qmc.Vector[qmc.Qubit]: Register containing the transformed strided view.
    """
    register = qmc.qubit_array(17, "register")
    register[0] = qmc.x(register[0])
    _expanding_tree_input_helper(register[1:17:2], 3)
    return register


@qmc.qkernel
def _unrolled_three_stage_tree() -> qmc.Vector[qmc.Qubit]:
    """Prepare the same eight-qubit state with an explicit gate schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: Explicitly prepared eight-qubit register.
    """
    register = qmc.qubit_array(8, "register")
    register[0] = qmc.h(register[0])
    register[0], register[1] = qmc.cx(register[0], register[1])
    register[0], register[2] = qmc.cx(register[0], register[2])
    register[1], register[3] = qmc.cx(register[1], register[3])
    register[0], register[4] = qmc.cx(register[0], register[4])
    register[1], register[5] = qmc.cx(register[1], register[5])
    register[2], register[6] = qmc.cx(register[2], register[6])
    register[3], register[7] = qmc.cx(register[3], register[7])
    return register


@qmc.qkernel
def _all_depth_fields_after_tree(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Exercise every depth category after a uniform expanding-tree exit.

    Args:
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after all depth-category sentinels.
    """
    tree_width = 2**stages
    register = _expanding_tree_input_helper(
        qmc.qubit_array(tree_width + 2, "register"),
        stages,
    )
    register[0] = qmc.t(register[0])
    register[0] = qmc.rz(register[0], 0.125)
    register[0], register[tree_width], register[tree_width + 1] = qmc.ccx(
        register[0],
        register[tree_width],
        register[tree_width + 1],
    )
    _measured = qmc.measure(register[tree_width])
    register[tree_width + 1] = qmc.reset(register[tree_width + 1])
    return register


@qmc.qkernel
def _prior_seed_wire_work() -> qmc.Vector[qmc.Qubit]:
    """Advance the safe seed frontier before an expanding tree.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after prior frontier work and the tree.
    """
    register = qmc.qubit_array(8, "register")
    register[0] = qmc.x(register[0])
    return _expanding_tree_input_helper(register, 3)


@qmc.qkernel
def _prior_second_wire_work() -> qmc.Vector[qmc.Qubit]:
    """Advance a nonfrontier tree wire before an expanding tree.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after prior second-wire work and the tree.
    """
    register = qmc.qubit_array(8, "register")
    register[1] = qmc.x(register[1])
    return _expanding_tree_input_helper(register, 3)


@qmc.qkernel
def _prior_late_wire_work() -> qmc.Vector[qmc.Qubit]:
    """Advance the final tree wire before an expanding tree.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after prior late-wire work and the tree.
    """
    register = qmc.qubit_array(8, "register")
    register[7] = qmc.x(register[7])
    return _expanding_tree_input_helper(register, 3)


@qmc.qkernel
def _prior_mixed_frontier_and_late_work() -> qmc.Vector[qmc.Qubit]:
    """Touch the seed frontier and a late wire in one preceding event.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after mixed prior work and the tree.
    """
    register = qmc.qubit_array(8, "register")
    register[0], register[7] = qmc.cx(register[0], register[7])
    return _expanding_tree_input_helper(register, 3)


@qmc.qkernel
def _prior_disjoint_work() -> qmc.Vector[qmc.Qubit]:
    """Run unrelated work beside an expanding tree.

    Returns:
        qmc.Vector[qmc.Qubit]: Tree register after disjoint surrounding work.
    """
    unrelated = qmc.h(qmc.qubit("unrelated"))
    register = _expanding_tree_input_helper(qmc.qubit_array(8, "register"), 3)
    unrelated = qmc.x(unrelated)
    return register


@qmc.qkernel
def _prior_full_uniform_write() -> qmc.Vector[qmc.Qubit]:
    """Resynchronize every tree wire before the expanding schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after two consecutive uniform trees.
    """
    register = qmc.qubit_array(8, "register")
    register = _expanding_tree_input_helper(register, 3)
    return _expanding_tree_input_helper(register, 3)


@qmc.qkernel
def _downstream_representative_wires() -> qmc.Vector[qmc.Qubit]:
    """Use early, middle, and late wires after the uniform tree exit.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after representative downstream gates.
    """
    register = _expanding_tree_input_helper(qmc.qubit_array(8, "register"), 3)
    register[0] = qmc.x(register[0])
    register[3] = qmc.x(register[3])
    register[7] = qmc.x(register[7])
    return register


@qmc.qkernel
def _tree_on_prefix_of_larger_register() -> qmc.Vector[qmc.Qubit]:
    """Leave wires beyond the power-of-two tree prefix untouched.

    Returns:
        qmc.Vector[qmc.Qubit]: Larger register with a transformed prefix.
    """
    register = qmc.qubit_array(10, "register")
    register[9] = qmc.h(register[9])
    register = _expanding_tree_input_helper(register, 3)
    register[9] = qmc.x(register[9])
    return register


@qmc.qkernel
def _symbolic_tree_after_second_wire_work(
    stages: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Move prior second-wire work into or out of the symbolic tree prefix.

    Args:
        stages (qmc.UInt): Number of prefix-doubling stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Larger register after prior work and the tree.
    """
    register = qmc.qubit_array(2**stages + 1, "register")
    register[1] = qmc.x(register[1])
    return _expanding_tree_input_helper(register, stages)


@qmc.qkernel
def _repeat_expanding_tree_helper() -> qmc.Vector[qmc.Qubit]:
    """Call the same expanding-tree helper twice.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after two helper applications.
    """
    register = qmc.qubit_array(8, "register")
    for _iteration in qmc.range(2):
        register = _expanding_tree_input_helper(register, 3)
    return register


@qmc.qkernel
def _tree_without_seed(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Omit the unary root event required by the whole-span proof.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the unseeded pair schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_binary_seed(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use a binary gate where the structural theorem requires a unary seed.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the binary-seeded schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0], register[1] = qmc.cx(register[0], register[1])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_wrong_initial_width(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Start with two pairs instead of the unique rooted pair.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the wrong-width schedule.
    """
    register = qmc.qubit_array(2 ** (stages + 1) + 2, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage + 1
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_wrong_width_recurrence(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Grow the pair width by three instead of doubling each stage.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the non-doubling schedule.
    """
    register = qmc.qubit_array(2 * 3**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 3**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_shifted_outer_bounds(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Shift the stage range away from the required zero origin.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the shifted-stage schedule.
    """
    register = qmc.qubit_array(2 ** (stages + 1), "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(1, stages + 1):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_strided_outer_loop(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Skip expansion stages with a nonunit outer range step.

    Args:
        stages (qmc.UInt): Outer range stop.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the strided-stage schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(0, stages, 2):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_shifted_inner_loop(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Skip each stage's first pair by shifting the inner range start.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the shifted-pair schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(1, width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_strided_inner_loop(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Enumerate only every second pair within each expansion stage.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the strided-pair schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(0, width, 2):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_colliding_targets(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Reuse one target across every nominal pair in a stage.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the colliding-target schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            register[control], register[width] = qmc.cx(
                register[control], register[width]
            )
    return register


@qmc.qkernel
def _tree_with_extra_seed_gate(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Place another quantum event between the seed and outer loop.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the interrupted tree schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    register[1] = qmc.x(register[1])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_extra_outer_gate(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Add extra quantum work directly inside each outer stage.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the augmented outer schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        register[0] = qmc.x(register[0])
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_extra_inner_gate(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Add a second quantum event to every inner pair iteration.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the augmented inner schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
            register[control] = qmc.x(register[control])
    return register


@qmc.qkernel
def _tree_with_outer_workspace(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate stage-local workspace beside the pair expansion.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after the allocating schedule.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        _scratch = qmc.qubit("scratch")
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_different_target_owner(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Draw targets from a different allocation owner.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Control-owner register after cross-owner pairs.
    """
    controls = qmc.qubit_array(2**stages, "controls")
    targets = qmc.qubit_array(2**stages, "targets")
    controls[0] = qmc.h(controls[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            controls[control], targets[target] = qmc.cx(
                controls[control], targets[target]
            )
    return controls


@qmc.qkernel
def _tree_with_coherent_control(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Share one coherent carrier across all nominally disjoint pairs.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after coherently controlled pairs.
    """
    carrier = qmc.qubit("carrier")
    register = qmc.qubit_array(2**stages, "register")
    controlled_cx = qmc.control(qmc.cx)
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            carrier, register[control], register[target] = controlled_cx(
                carrier,
                register[control],
                register[target],
            )
    return register


@qmc.qkernel
def _tree_with_conditional_outer_work(
    stages: qmc.UInt,
    flag: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Put a classical branch beside the nested pair loop.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.
        flag (qmc.UInt): Compile-time branch selector.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after conditional stage work.
    """
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        if flag:
            register[0] = qmc.x(register[0])
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _tree_with_runtime_classical_work(stages: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Put measurement-backed runtime control beside the nested pair loop.

    Args:
        stages (qmc.UInt): Number of nominal expansion stages.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after runtime-controlled stage work.
    """
    predicate_source = qmc.qubit("predicate_source")
    predicate = qmc.measure(predicate_source)
    register = qmc.qubit_array(2**stages, "register")
    register[0] = qmc.h(register[0])
    for stage in qmc.range(stages):
        if predicate:
            register[0] = qmc.x(register[0])
        width = 2**stage
        for control in qmc.range(width):
            target = width + control
            register[control], register[target] = qmc.cx(
                register[control], register[target]
            )
    return register


@qmc.qkernel
def _inverse_expanding_tree_helper() -> qmc.Vector[qmc.Qubit]:
    """Request the currently unsupported symbolic inverse tree lowering.

    Returns:
        qmc.Vector[qmc.Qubit]: Inverted helper output if lowering succeeds.
    """
    register = qmc.qubit_array(8, "register")
    return qmc.inverse(_expanding_tree_input_helper)(register, 3)


@qmc.qkernel
def _controlled_expanding_tree_helper() -> qmc.Vector[qmc.Qubit]:
    """Apply coherent control around the whole expanding-tree helper.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after coherently controlled tree work.
    """
    carrier = qmc.qubit("carrier")
    register = qmc.qubit_array(8, "register")
    carrier, register = qmc.control(_expanding_tree_input_helper)(
        carrier,
        register,
        3,
    )
    return register


def _ready_vector_tree_depth(stages: int) -> tuple[int, list[int]]:
    """Compute tree depth with an independent per-wire readiness oracle.

    Args:
        stages (int): Number of power-of-two expansion stages.

    Returns:
        tuple[int, list[int]]: Aggregate depth and final per-wire readiness.
    """
    ready = [0] * (2**stages)
    ready[0] += 1
    for stage in range(stages):
        width = 2**stage
        next_ready = ready.copy()
        for control in range(width):
            target = width + control
            layer = max(ready[control], ready[target]) + 1
            next_ready[control] = layer
            next_ready[target] = layer
        ready = next_ready
    return max(ready), ready


def _assert_exact_tree_estimate(
    estimate: qmc.ResourceEstimate,
    stages: int,
) -> None:
    """Assert all public resources for one specialized CX tree.

    Args:
        estimate (qmc.ResourceEstimate): Specialized tree estimate.
        stages (int): Expected expansion-stage count.

    Returns:
        None: This helper succeeds by completing all assertions.
    """
    expected_depth, ready = _ready_vector_tree_depth(stages)
    assert ready == [stages + 1] * (2**stages)
    assert estimate.width.allocated_qubits == 2**stages
    assert estimate.width.peak_qubits == 2**stages
    assert estimate.gates == qmc.GateResources(
        total=2**stages,
        single_qubit=1,
        two_qubit=2**stages - 1,
        clifford=2**stages,
    )
    assert estimate.depth == qmc.DepthResources(
        depth=expected_depth,
        clifford_depth=expected_depth,
        gate_depth=expected_depth,
    )
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()


def _has_tree_fallback(estimate: qmc.ResourceEstimate) -> bool:
    """Return whether unresolved nested-tree scheduling remains visible.

    Args:
        estimate (qmc.ResourceEstimate): Estimate whose assumptions to inspect.

    Returns:
        bool: Whether an unresolved-index or loop scheduling fallback is active.
    """
    return estimate.quality is not qmc.EstimateQuality.EXACT and any(
        assumption.source in {"GateOperation", "for", "dependency scheduler"}
        for assumption in estimate.assumptions
    )


def test_symbolic_direct_and_substituted_tree_match_ready_vector_oracle() -> None:
    """All specialization routes agree with an independent scheduler oracle."""
    symbolic = _expanding_cx_tree.estimate_resources()
    stages_symbol = symbolic.parameters["stages"]

    assert symbolic.width.allocated_qubits == 2**stages_symbol
    assert symbolic.gates.total == 2**stages_symbol
    assert symbolic.gates.two_qubit == 2**stages_symbol - 1
    assert symbolic.depth.depth == stages_symbol + 1
    assert symbolic.depth.clifford_depth == stages_symbol + 1
    assert symbolic.depth.gate_depth == stages_symbol + 1
    assert symbolic.quality is qmc.EstimateQuality.EXACT
    assert symbolic.assumptions == ()

    for stages in range(9):
        direct = _expanding_cx_tree.estimate_resources(inputs={"stages": stages})
        substituted = symbolic.substitute(stages=stages)
        _assert_exact_tree_estimate(direct, stages)
        _assert_exact_tree_estimate(substituted, stages)
        assert direct.width == substituted.width
        assert direct.gates == substituted.gates
        assert direct.depth == substituted.depth


def test_tree_certificate_is_compact_binder_free_and_uniform() -> None:
    """The structural proof exports its exact grouped scheduler contract."""
    estimate = _expanding_tree_input_helper.estimate_resources(
        inputs={"register": 8, "stages": 3}
    )

    assert estimate._dependency_keys is not None
    assert len(estimate._dependency_keys) == 1
    assert estimate._dependency_reads == estimate._dependency_keys
    assert estimate._dependency_writes == estimate._dependency_keys
    assert estimate._dependency_completion is not None
    assert set(estimate._dependency_completion) == set(estimate._dependency_keys)
    assert set(estimate._dependency_completion.values()) == {4}
    assert estimate._dependency_completion_uniform is True
    (certificate,) = estimate._dependency_synchronized_entry_certificates
    assert certificate.coverage == estimate._dependency_keys
    assert len(certificate.frontier) == 1
    (coverage_key,) = certificate.coverage
    (frontier_key,) = certificate.frontier
    assert frontier_key[0] == coverage_key[0]
    assert frontier_key[1] == 0
    assert certificate.active_when is sp.true
    assert not any(
        symbol.name.startswith(("i", "control", "stage"))
        for key in estimate._dependency_keys
        for symbol in (
            key[1].free_symbols if isinstance(key[1], sp.Expr) else frozenset()
        )
    )


def test_large_symbolic_tree_does_not_expand_physical_wires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large concrete stage count retains a compact symbolic footprint."""
    import qamomile.circuit.estimator._interpreter_for_region as for_region

    def reject_concrete_replay(*args: object, **kwargs: object) -> None:
        """Fail if an exponentially large tree reaches concrete replay.

        Args:
            *args (object): Positional replay arguments, intentionally ignored.
            **kwargs (object): Keyword replay arguments, intentionally ignored.

        Returns:
            None: This callback never returns normally.

        Raises:
            AssertionError: Always, because concrete replay is forbidden here.
        """
        raise AssertionError("expanding tree used concrete region replay")

    monkeypatch.setattr(
        for_region._ForRegionInterpreter,
        "_eval_concrete_region_for",
        reject_concrete_replay,
    )
    estimate = _expanding_cx_tree.estimate_resources(inputs={"stages": 64})

    assert estimate.gates.total == 2**64
    assert estimate.depth.depth == 65
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()
    assert estimate._dependency_keys is not None
    assert len(estimate._dependency_keys) < 20


def test_three_stage_tree_matches_an_explicit_unrolling() -> None:
    """The compact proof agrees with explicit ready-vector scheduling."""
    compact = _expanding_cx_tree.estimate_resources(inputs={"stages": 3})
    unrolled = _unrolled_three_stage_tree.estimate_resources()

    assert compact.width == unrolled.width
    assert compact.gates == unrolled.gates
    assert compact.depth == unrolled.depth
    assert compact.depth.depth == 4


def test_seed_and_pair_profiles_are_composed_field_by_field() -> None:
    """Distinct seed and pair gate families retain exact category depths."""
    symbolic = _expanding_rzz_tree.estimate_resources()
    direct = _expanding_rzz_tree.estimate_resources(inputs={"stages": 3})
    substituted = symbolic.substitute(stages=3)

    stages = symbolic.parameters["stages"]
    assert symbolic.depth.depth == stages + 1
    assert symbolic.depth.rotation_depth == stages
    assert symbolic.depth.t_depth == 1
    assert symbolic.depth.non_clifford_depth == stages + 1
    for estimate in (direct, substituted):
        assert estimate.gates == qmc.GateResources(
            total=8,
            single_qubit=1,
            two_qubit=7,
            rotation=7,
            t=1,
            non_clifford=8,
        )
        assert estimate.depth == qmc.DepthResources(
            depth=4,
            rotation_depth=3,
            t_depth=1,
            non_clifford_depth=4,
            gate_depth=4,
        )
        assert estimate.quality is qmc.EstimateQuality.EXACT
        assert estimate.assumptions == ()


def test_zero_depth_binop_between_seed_and_loop_is_supported() -> None:
    """Harmless scalar IR does not break adjacency of the structural span."""
    estimate = _expanding_tree_with_scalar_binop.estimate_resources(
        inputs={"stages": 3}
    )

    _assert_exact_tree_estimate(estimate, 3)


def test_uniform_tree_completion_schedules_every_suffix_depth_field() -> None:
    """All nine depth fields compose exactly after the tree's uniform exit."""
    estimate = _all_depth_fields_after_tree.estimate_resources(inputs={"stages": 3})

    assert estimate.depth == qmc.DepthResources(
        depth=8,
        clifford_depth=4,
        rotation_depth=1,
        t_depth=1,
        toffoli_depth=1,
        non_clifford_depth=3,
        measurement_depth=1,
        gate_depth=7,
        reset_depth=1,
    )
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_helper_allocation_and_fixed_strided_view_preserve_exact_schedule() -> None:
    """Exact owner mappings retain the expanding-tree structural proof."""
    direct = _expanding_cx_tree.estimate_resources(inputs={"stages": 3})
    helper = _expanding_tree_through_helper.estimate_resources(inputs={"stages": 3})
    view = _expanding_tree_through_strided_view.estimate_resources()

    assert helper.gates == direct.gates
    assert helper.depth == direct.depth
    assert helper.quality is qmc.EstimateQuality.EXACT
    assert view.gates.total == 9
    assert view.depth.depth == 4
    assert view.quality is qmc.EstimateQuality.EXACT


@pytest.mark.parametrize(
    ("kernel", "expected_depth", "expected_quality"),
    [
        (_prior_seed_wire_work, 5, qmc.EstimateQuality.EXACT),
        (_prior_second_wire_work, 5, qmc.EstimateQuality.CONSERVATIVE),
        (_prior_late_wire_work, 5, qmc.EstimateQuality.CONSERVATIVE),
        (_prior_mixed_frontier_and_late_work, 5, qmc.EstimateQuality.CONSERVATIVE),
        (_prior_disjoint_work, 4, qmc.EstimateQuality.EXACT),
        (_prior_full_uniform_write, 8, qmc.EstimateQuality.EXACT),
    ],
)
def test_tree_entry_frontier_is_checked_against_prior_readiness(
    kernel: qmc.QKernel,
    expected_depth: int,
    expected_quality: qmc.EstimateQuality,
) -> None:
    """Only safe frontier or fully synchronized prior work remains exact."""
    estimate = kernel.estimate_resources()

    assert estimate.depth.depth == expected_depth
    assert estimate.quality is expected_quality


def test_uniform_completion_supports_downstream_wires_exactly() -> None:
    """Every tree wire exposes the same exact completion layer."""
    estimate = _downstream_representative_wires.estimate_resources()

    assert estimate.gates.total == 11
    assert estimate.depth.depth == 5
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_tree_coverage_excludes_larger_register_suffix() -> None:
    """The proof covers exactly the used prefix of a larger allocation."""
    estimate = _tree_on_prefix_of_larger_register.estimate_resources()

    assert estimate.width.allocated_qubits == 10
    assert estimate.gates.total == 10
    assert estimate.depth.depth == 4
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_symbolic_second_wire_overlap_specializes_at_zero_stages() -> None:
    """Late specialization stays fail-closed while direct zero stages are exact."""
    symbolic = _symbolic_tree_after_second_wire_work.estimate_resources()

    assert symbolic.depth.depth == symbolic.parameters["stages"] + 2
    assert symbolic.quality is qmc.EstimateQuality.CONSERVATIVE
    for stages in range(4):
        direct = _symbolic_tree_after_second_wire_work.estimate_resources(
            inputs={"stages": stages}
        )
        substituted = symbolic.substitute(stages=stages)
        if stages == 0:
            assert direct.depth.depth == 1
            assert direct.quality is qmc.EstimateQuality.EXACT
            assert direct.assumptions == ()
            assert substituted.depth.depth == 2
            assert substituted.quality is qmc.EstimateQuality.CONSERVATIVE
            assert substituted.assumptions
        else:
            for estimate in (direct, substituted):
                assert estimate.depth.depth == stages + 2
                assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE


def test_repeat_sum_inverse_and_control_fail_closed_or_preserve_exactness() -> None:
    """Public transforms retain exactness only with valid temporal metadata."""
    once = _expanding_tree_input_helper.estimate_resources(
        inputs={"register": 8, "stages": 3}
    )
    iteration = sp.Dummy("iteration", integer=True, nonnegative=True)

    assert any(
        certificate.frontier
        for certificate in once._dependency_synchronized_entry_certificates
    )

    for aggregate in (once.repeat(2), once.sum_over(iteration, 0, 2)):
        assert aggregate.gates.total == 16
        assert aggregate.depth.depth == 8
        assert aggregate.quality is qmc.EstimateQuality.EXACT

    inverted = once.inverse()
    assert inverted._dependency_completion is None
    assert inverted._dependency_completion_uniform is None
    assert all(
        not certificate.frontier
        for certificate in inverted._dependency_synchronized_entry_certificates
    )
    repeated_inverse = inverted.repeat(2)
    assert repeated_inverse.depth.depth == 8
    assert repeated_inverse.quality is qmc.EstimateQuality.CONSERVATIVE

    controlled = once.controlled(1)
    assert controlled.gates.total >= once.gates.total
    assert controlled.quality is not qmc.EstimateQuality.EXACT


def test_repeated_helper_uses_uniform_completion_exactly() -> None:
    """A second expanding tree starts from a uniform first-tree exit."""
    estimate = _repeat_expanding_tree_helper.estimate_resources()

    assert estimate.gates.total == 16
    assert estimate.depth.depth == 8
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()


@pytest.mark.parametrize(
    "kernel",
    [
        _tree_without_seed,
        _tree_with_binary_seed,
        _tree_with_wrong_initial_width,
        _tree_with_wrong_width_recurrence,
        _tree_with_shifted_outer_bounds,
        _tree_with_strided_outer_loop,
        _tree_with_shifted_inner_loop,
        _tree_with_strided_inner_loop,
        _tree_with_colliding_targets,
        _tree_with_extra_seed_gate,
        _tree_with_extra_outer_gate,
        _tree_with_extra_inner_gate,
        _tree_with_outer_workspace,
        _tree_with_different_target_owner,
        _tree_with_coherent_control,
    ],
)
def test_near_miss_structures_fail_closed(kernel: qmc.QKernel) -> None:
    """Unsupported lookalikes retain explicit conservative fallback metadata."""
    estimate = kernel.estimate_resources(inputs={"stages": 3})

    assert _has_tree_fallback(estimate)


def test_conditional_and_runtime_classical_near_misses_fail_closed() -> None:
    """Classical or measurement-backed stage work blocks the structural proof."""
    conditional = _tree_with_conditional_outer_work.estimate_resources(
        inputs={"stages": 3, "flag": 1}
    )
    runtime = _tree_with_runtime_classical_work.estimate_resources(inputs={"stages": 3})

    assert _has_tree_fallback(conditional)
    assert _has_tree_fallback(runtime)


def test_qkernel_inverse_and_control_do_not_reuse_forward_tree_proof() -> None:
    """Temporal transforms fail closed instead of reusing forward metadata."""
    with pytest.raises(NotImplementedError):
        _inverse_expanding_tree_helper.estimate_resources()

    controlled = _controlled_expanding_tree_helper.estimate_resources()
    assert controlled.depth.depth >= 4
    assert controlled.quality is not qmc.EstimateQuality.EXACT


def test_malformed_seed_result_mapping_does_not_receive_tree_proof() -> None:
    """A raw seed result returned to another index blocks the tree proof."""
    block = _expanding_cx_tree.build()
    seed = next(
        operation
        for operation in block.operations
        if isinstance(operation, GateOperation)
    )
    result = seed.results[0]
    index = result.element_indices[0]
    scalar = dataclasses.replace(index.metadata.scalar, const_value=1)
    metadata = dataclasses.replace(index.metadata, scalar=scalar)
    wrong_index = dataclasses.replace(index, metadata=metadata)
    wrong_result = dataclasses.replace(result, element_indices=(wrong_index,))
    malformed_seed = dataclasses.replace(seed, results=[wrong_result])
    malformed_block = dataclasses.replace(
        block,
        operations=[
            malformed_seed if operation is seed else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(
        malformed_block,
        inputs={"stages": 3},
    )

    assert _has_tree_fallback(estimate)
    assert estimate._dependency_synchronized_entry_certificates == ()


def test_seed_declared_as_cx_with_unary_arity_does_not_receive_tree_proof() -> None:
    """A unary seed mislabeled as CX cannot satisfy the structural theorem."""
    block = _expanding_cx_tree.build()
    seed = next(
        operation
        for operation in block.operations
        if isinstance(operation, GateOperation)
    )
    outer = next(
        operation
        for operation in block.operations
        if isinstance(operation, ForOperation)
    )
    inner = next(
        operation
        for operation in outer.operations
        if isinstance(operation, ForOperation)
    )
    pair = next(
        operation
        for operation in inner.operations
        if isinstance(operation, GateOperation)
    )
    mismatched_seed = dataclasses.replace(seed, gate_type=pair.gate_type)
    mismatched_block = dataclasses.replace(
        block,
        operations=[
            mismatched_seed if operation is seed else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(
        mismatched_block,
        inputs={"stages": 3},
    )

    assert _has_tree_fallback(estimate)
    assert estimate._dependency_synchronized_entry_certificates == ()


def test_pair_declared_as_h_with_binary_arity_does_not_receive_tree_proof() -> None:
    """A binary pair mislabeled as H cannot satisfy the structural theorem."""
    block = _expanding_cx_tree.build()
    seed = next(
        operation
        for operation in block.operations
        if isinstance(operation, GateOperation)
    )
    outer = next(
        operation
        for operation in block.operations
        if isinstance(operation, ForOperation)
    )
    inner = next(
        operation
        for operation in outer.operations
        if isinstance(operation, ForOperation)
    )
    pair = next(
        operation
        for operation in inner.operations
        if isinstance(operation, GateOperation)
    )
    mismatched_pair = dataclasses.replace(pair, gate_type=seed.gate_type)
    mismatched_inner = dataclasses.replace(
        inner,
        operations=[
            mismatched_pair if operation is pair else operation
            for operation in inner.operations
        ],
    )
    mismatched_outer = dataclasses.replace(
        outer,
        operations=[
            mismatched_inner if operation is inner else operation
            for operation in outer.operations
        ],
    )
    mismatched_block = dataclasses.replace(
        block,
        operations=[
            mismatched_outer if operation is outer else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(
        mismatched_block,
        inputs={"stages": 3},
    )

    assert _has_tree_fallback(estimate)
    assert estimate._dependency_synchronized_entry_certificates == ()


def test_malformed_pair_result_mapping_does_not_receive_tree_proof() -> None:
    """Malformed raw IR cannot acquire the exact expanding-tree certificate."""
    block = _expanding_cx_tree.build()
    outer = next(
        operation
        for operation in block.operations
        if isinstance(operation, ForOperation)
    )
    inner = next(
        operation
        for operation in outer.operations
        if isinstance(operation, ForOperation)
    )
    pair = next(
        operation
        for operation in inner.operations
        if isinstance(operation, GateOperation)
    )
    malformed_pair = dataclasses.replace(pair, results=pair.results[:1])
    malformed_inner = dataclasses.replace(
        inner,
        operations=[
            malformed_pair if operation is pair else operation
            for operation in inner.operations
        ],
    )
    malformed_outer = dataclasses.replace(
        outer,
        operations=[
            malformed_inner if operation is inner else operation
            for operation in outer.operations
        ],
    )
    malformed_block = dataclasses.replace(
        block,
        operations=[
            malformed_outer if operation is outer else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(
        malformed_block,
        inputs={"stages": 3},
    )

    assert _has_tree_fallback(estimate)
    assert estimate._dependency_synchronized_entry_certificates == ()


def test_swapped_pair_result_mapping_does_not_receive_tree_proof() -> None:
    """Reversing raw pair results blocks address-preservation reasoning."""
    block = _expanding_cx_tree.build()
    outer = next(
        operation
        for operation in block.operations
        if isinstance(operation, ForOperation)
    )
    inner = next(
        operation
        for operation in outer.operations
        if isinstance(operation, ForOperation)
    )
    pair = next(
        operation
        for operation in inner.operations
        if isinstance(operation, GateOperation)
    )
    swapped_pair = dataclasses.replace(pair, results=list(reversed(pair.results)))
    swapped_inner = dataclasses.replace(
        inner,
        operations=[
            swapped_pair if operation is pair else operation
            for operation in inner.operations
        ],
    )
    swapped_outer = dataclasses.replace(
        outer,
        operations=[
            swapped_inner if operation is inner else operation
            for operation in outer.operations
        ],
    )
    swapped_block = dataclasses.replace(
        block,
        operations=[
            swapped_outer if operation is outer else operation
            for operation in block.operations
        ],
    )

    estimate = qmc.ResourceEstimator().estimate(
        swapped_block,
        inputs={"stages": 3},
    )

    assert _has_tree_fallback(estimate)
    assert estimate._dependency_synchronized_entry_certificates == ()


def test_tree_matcher_is_required_for_exact_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling only the structural matcher makes the valid tree conservative."""
    import qamomile.circuit.estimator._interpreter_region_analysis as region_analysis

    monkeypatch.setattr(
        region_analysis, "_compound_affine_region_schedules", lambda *args, **kwargs: ()
    )

    estimate = _expanding_cx_tree.estimate_resources(inputs={"stages": 3})

    assert estimate.depth.depth == 8
    assert _has_tree_fallback(estimate)
