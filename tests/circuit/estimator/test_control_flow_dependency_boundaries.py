"""Regression tests for quantum dependencies across conditional boundaries."""

from __future__ import annotations

import qamomile.circuit as qm


@qm.qkernel
def _branch_local_scalar(flag: qm.UInt) -> qm.Qubit:
    """Return a gate-touched qubit allocated by the selected branch."""
    if flag:
        target = qm.h(qm.qubit("true_target"))
    else:
        target = qm.x(qm.qubit("false_target"))
    return target


@qm.qkernel
def _branch_local_array(flag: qm.UInt) -> qm.Vector[qm.Qubit]:
    """Return an array whose first element is touched by either branch."""
    if flag:
        register = qm.qubit_array(2, "true_register")
        register[0] = qm.h(register[0])
    else:
        register = qm.qubit_array(2, "false_register")
        register[0] = qm.x(register[0])
    return register


@qm.qkernel
def _apply_h(target: qm.Qubit) -> qm.Qubit:
    """Apply one Hadamard gate through a nested call boundary."""
    return qm.h(target)


def test_branch_local_scalar_merge_keeps_post_if_dependency() -> None:
    """A gate after a scalar merge waits for either branch producer."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Qubit:
        """Gate a qubit returned from one of two branch-local allocations."""
        if flag:
            target = qm.h(qm.qubit("true_target"))
        else:
            target = qm.x(qm.qubit("false_target"))
        return qm.z(target)

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    assert symbolic.depth.depth == 2
    assert symbolic.depth.gate_depth == 2
    assert symbolic.substitute(flag=0).depth.depth == 2
    assert symbolic.substitute(flag=1).depth.depth == 2
    assert false_branch.depth.depth == 2
    assert true_branch.depth.depth == 2


def test_preallocated_scalar_merge_keeps_post_if_dependency() -> None:
    """A gate after a scalar merge waits for either in-place branch gate."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Qubit:
        """Gate one existing qubit before and after a conditional merge."""
        target = qm.qubit("target")
        if flag:
            target = qm.h(target)
        else:
            target = qm.x(target)
        return qm.z(target)

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    for estimate in (false_branch, true_branch):
        assert estimate.depth.depth == 2
        assert estimate.quality is qm.EstimateQuality.EXACT
    assert symbolic.depth.depth == 2
    assert symbolic.substitute(flag=0).depth.depth == 2
    assert symbolic.substitute(flag=1).depth.depth == 2


def test_runtime_if_merge_keeps_feedforward_output_dependency() -> None:
    """A post-if gate waits for measurement and the selected branch gate."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Select a branch-local qubit using a measured runtime bit."""
        predicate = qm.measure(qm.qubit("predicate"))
        if predicate:
            target = qm.h(qm.qubit("true_target"))
        else:
            target = qm.x(qm.qubit("false_target"))
        return qm.z(target)

    estimate = circuit.estimate_resources()

    assert estimate.depth.depth == 3
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1


def test_branch_local_array_merge_maps_same_element_dependency() -> None:
    """The same array element cannot run beside its branch producer."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Gate element zero before and after selecting a local array."""
        if flag:
            register = qm.qubit_array(2, "true_register")
            register[0] = qm.h(register[0])
        else:
            register = qm.qubit_array(2, "false_register")
            register[0] = qm.x(register[0])
        register[0] = qm.z(register[0])
        return register

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    assert symbolic.depth.depth == 2
    assert false_branch.depth.depth == 2
    assert true_branch.depth.depth == 2


def test_branch_local_array_merge_preserves_disjoint_element_parallelism() -> None:
    """A different merged-array element remains parallel with branch work."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Gate different elements before and after selecting a local array."""
        if flag:
            register = qm.qubit_array(2, "true_register")
            register[0] = qm.h(register[0])
        else:
            register = qm.qubit_array(2, "false_register")
            register[0] = qm.x(register[0])
        register[1] = qm.z(register[1])
        return register

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    assert symbolic.depth.depth == 1
    assert false_branch.depth.depth == 1
    assert true_branch.depth.depth == 1


def test_zero_depth_array_merge_tracks_selected_owner_alias() -> None:
    """A zero-cost array selection still aliases the selected allocation."""

    @qm.qkernel
    def circuit(
        flag: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Use a selected array and its possible true-branch source."""
        left = qm.qubit_array(2, "left")
        right = qm.qubit_array(2, "right")
        if flag:
            selected = left
        else:
            selected = right
        selected[0] = qm.h(selected[0])
        left[0] = qm.x(left[0])
        return left, right

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    assert symbolic.depth.depth == 2
    assert false_branch.depth.depth == 1
    assert true_branch.depth.depth == 2


def test_zero_depth_merge_alias_expands_nested_call_dependencies() -> None:
    """A nested call on a selected owner conflicts with its possible source."""

    @qm.qkernel
    def circuit(
        flag: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Use the selected owner through a helper before its source."""
        left = qm.qubit_array(2, "left")
        right = qm.qubit_array(2, "right")
        if flag:
            selected = left
        else:
            selected = right
        selected[0] = _apply_h(selected[0])
        left[0] = qm.x(left[0])
        return left, right

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    assert symbolic.depth.depth == 2
    assert false_branch.depth.depth == 1
    assert true_branch.depth.depth == 2


def test_nested_qkernel_maps_conditional_output_completion() -> None:
    """A caller gate waits for a branch-local producer inside its callee."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Qubit:
        """Gate the scalar output of a conditional helper."""
        target = _branch_local_scalar(flag)
        return qm.z(target)

    symbolic = circuit.estimate_resources()
    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    assert symbolic.depth.depth == 2
    assert false_branch.depth.depth == 2
    assert true_branch.depth.depth == 2


def test_nested_array_output_keeps_selected_completion_exact() -> None:
    """A selected array completion remains exact across a qkernel call."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Gate the touched element of a conditional array result."""
        register = _branch_local_array(flag)
        register[0] = qm.z(register[0])
        return register

    false_branch = circuit.estimate_resources(inputs={"flag": 0})
    true_branch = circuit.estimate_resources(inputs={"flag": 1})

    for estimate in (false_branch, true_branch):
        assert estimate.depth.depth == 2
        assert estimate.quality is qm.EstimateQuality.EXACT
        assert not any(
            "aggregate latency" in assumption.message
            for assumption in estimate.assumptions
        )
