"""Regression tests for liveness across control-flow boundaries."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    RegionArg,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.types import QubitType, UIntType
from qamomile.circuit.ir.value import Value


@qm.qkernel
def _runtime_if_retains_local_owner() -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave a true-branch allocation live before a later allocation."""
    retained = qm.qubit("retained")
    predicate = qm.measure(qm.qubit("predicate"))
    if predicate:
        work = qm.qubit_array(2, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _compile_time_if_retains_local_owner(
    flag: qm.UInt,
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Conditionally leave a branch allocation live before later work."""
    retained = qm.qubit("retained")
    if flag:
        work = qm.qubit_array(2, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _range_retains_local_owner(
    iterations: qm.UInt,
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave a range-body allocation live when the loop is nonempty."""
    retained = qm.qubit("retained")
    for _index in qm.range(iterations):
        work = qm.qubit_array(2, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _range_retains_index_sized_local_owner(
    iterations: qm.UInt,
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave the final iteration's index-sized allocation live."""
    retained = qm.qubit("retained")
    for index in qm.range(iterations):
        work = qm.qubit_array(index + 1, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _descending_range_retains_maximum_local_owner() -> tuple[
    qm.Qubit, qm.Vector[qm.Qubit]
]:
    """Leave a wider early range allocation live across a narrower visit."""
    retained = qm.qubit("retained")
    for size in qm.range(3, 0, -1):
        work = qm.qubit_array(size, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _symbolic_descending_range_retains_maximum_local_owner(
    iterations: qm.UInt,
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave the first symbolic range allocation live after narrower visits."""
    retained = qm.qubit("retained")
    for size in qm.range(iterations, 0, -1):
        work = qm.qubit_array(size, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _items_retains_local_owner(
    data: qm.Dict[qm.UInt, qm.Float],
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave an items-body allocation live when the input is nonempty."""
    retained = qm.qubit("retained")
    for _key, _value in qm.items(data):
        work = qm.qubit_array(2, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _items_retains_key_sized_local_owner(
    data: qm.Dict[qm.UInt, qm.Float],
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave the widest key-sized items allocation live after the loop."""
    retained = qm.qubit("retained")
    for size, _value in qm.items(data):
        work = qm.qubit_array(size, "work")
        work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _items_retains_carry_sized_local_owner(
    data: qm.Dict[qm.UInt, qm.Float],
) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave the final item's carry-sized allocation live."""
    retained = qm.qubit("retained")
    size = qm.uint(1)
    for _key, _value in qm.items(data):
        work = qm.qubit_array(size, "work")
        work[0] = qm.h(work[0])
        size = size + 1
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _while_retains_unmeasured_sibling() -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Leave one while-local array sibling live after each modeled trip."""
    retained = qm.qubit("retained")
    predicate = qm.measure(qm.qubit("predicate"))
    while predicate:
        work = qm.qubit_array(2, "work")
        work[0] = qm.h(work[0])
        predicate = qm.measure(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _runtime_if_consumes_local_owner() -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Consume the complete branch-local allocation before later work."""
    retained = qm.qubit("retained")
    predicate = qm.measure(qm.qubit("predicate"))
    if predicate:
        work = qm.qubit_array(2, "work")
        qm.measure(work)
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _runtime_if_returns_one_local_element() -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Return one branch-local element while its allocation sibling stays live."""
    predicate = qm.measure(qm.qubit("predicate"))
    if predicate:
        pair = qm.qubit_array(2, "pair")
        selected = pair[0]
    else:
        selected = qm.qubit("fallback")
    later = qm.qubit_array(2, "later")
    return selected, later


@qm.qkernel
def _if_containing_range_retains_local_owner() -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Propagate a range-local owner through an enclosing runtime branch."""
    retained = qm.qubit("retained")
    predicate = qm.measure(qm.qubit("predicate"))
    if predicate:
        for _index in qm.range(1):
            work = qm.qubit_array(2, "work")
            work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


@qm.qkernel
def _range_containing_if_retains_local_owner() -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
    """Propagate a branch-local owner through an enclosing range loop."""
    retained = qm.qubit("retained")
    predicate = qm.measure(qm.qubit("predicate"))
    for _index in qm.range(1):
        if predicate:
            work = qm.qubit_array(2, "work")
            work[0] = qm.h(work[0])
    later = qm.qubit_array(2, "later")
    return retained, later


def _quantum_region_arg() -> RegionArg:
    """Build one unsupported manually constructed quantum loop carry."""
    return RegionArg(
        var_name="target",
        init=Value(type=QubitType(), name="init"),
        block_arg=Value(type=QubitType(), name="block_arg"),
        yielded=Value(type=QubitType(), name="yielded"),
        result=Value(type=QubitType(), name="result"),
    )


def test_runtime_if_retains_unreturned_local_owner_after_boundary() -> None:
    """A possible live branch allocation overlaps later caller allocations."""
    estimate = _runtime_if_retains_local_owner.estimate_resources()

    assert estimate.width.allocated_qubits == 6
    assert estimate.width.peak_qubits == 5


def test_compile_time_if_retains_only_the_selected_local_owner() -> None:
    """Symbolic branch substitution matches direct liveness specialization."""
    symbolic = _compile_time_if_retains_local_owner.estimate_resources()

    for flag, expected_width in ((0, 3), (1, 5)):
        direct = _compile_time_if_retains_local_owner.estimate_resources(
            inputs={"flag": flag}
        )
        substituted = symbolic.substitute(flag=flag)

        assert direct.width == substituted.width
        assert direct.width.allocated_qubits == expected_width
        assert direct.width.peak_qubits == expected_width


@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_range_retains_local_owner_after_nonempty_loop(iterations: int) -> None:
    """A range allocation survives once while one static site is reused."""
    estimate = _range_retains_local_owner.estimate_resources(
        inputs={"iterations": iterations}
    )
    expected_width = 3 if iterations == 0 else 5

    assert estimate.width.allocated_qubits == expected_width
    assert estimate.width.peak_qubits == expected_width


@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_symbolic_range_substitution_preserves_boundary_liveness(
    iterations: int,
) -> None:
    """Post-hoc range specialization matches estimation with bound inputs."""
    symbolic = _range_retains_local_owner.estimate_resources()
    substituted = symbolic.substitute(iterations=iterations)
    direct = _range_retains_local_owner.estimate_resources(
        inputs={"iterations": iterations}
    )

    assert substituted.width == direct.width


@pytest.mark.parametrize(("iterations", "expected_width"), [(1, 4), (3, 6)])
def test_range_projects_live_width_to_final_iteration(
    iterations: int,
    expected_width: int,
) -> None:
    """A loop-local output size does not leak its induction symbol."""
    symbolic = _range_retains_index_sized_local_owner.estimate_resources()
    substituted = symbolic.substitute(iterations=iterations)
    direct = _range_retains_index_sized_local_owner.estimate_resources(
        inputs={"iterations": iterations}
    )

    assert direct.parameters == {}
    assert direct.width.peak_qubits == expected_width
    assert substituted.width == direct.width
    assert direct.quality is qm.EstimateQuality.EXACT
    assert substituted.quality is qm.EstimateQuality.EXACT
    assert direct.quality is qm.EstimateQuality.EXACT
    assert substituted.quality is qm.EstimateQuality.EXACT


def test_range_retains_widest_owner_across_all_iterations() -> None:
    """A narrower final range visit cannot release an earlier wider owner."""
    estimate = _descending_range_retains_maximum_local_owner.estimate_resources()

    assert estimate.width.allocated_qubits == 6
    assert estimate.width.peak_qubits == 6
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize(("iterations", "expected_width"), [(0, 3), (1, 4), (3, 6)])
def test_symbolic_descending_range_retains_widest_owner(
    iterations: int,
    expected_width: int,
) -> None:
    """Direct and post-hoc specialization retain the symbolic range maximum."""
    symbolic = (
        _symbolic_descending_range_retains_maximum_local_owner.estimate_resources()
    )
    substituted = symbolic.substitute(iterations=iterations)
    direct = _symbolic_descending_range_retains_maximum_local_owner.estimate_resources(
        inputs={"iterations": iterations}
    )

    assert direct.width == substituted.width
    assert direct.width.peak_qubits == expected_width
    expected_quality = (
        qm.EstimateQuality.CONSERVATIVE if iterations > 1 else qm.EstimateQuality.EXACT
    )
    assert direct.quality is expected_quality
    assert substituted.quality is expected_quality


@pytest.mark.parametrize(
    ("data", "expected_width"),
    [({}, 3), ({0: 0.1}, 5), ({0: 0.1, 1: 0.2, 2: 0.3}, 5)],
)
def test_items_retains_local_owner_after_nonempty_loop(
    data: dict[int, float],
    expected_width: int,
) -> None:
    """Items iterations reuse one site while preserving its final owner."""
    estimate = _items_retains_local_owner.estimate_resources(inputs={"data": data})

    assert estimate.width.allocated_qubits == expected_width
    assert estimate.width.peak_qubits == expected_width


@pytest.mark.parametrize(
    "data",
    [
        {3: 0.1, 1: 0.2},
        {1: 0.1, 3: 0.2},
    ],
)
def test_items_retained_width_is_independent_of_entry_order(
    data: dict[int, float],
) -> None:
    """Every insertion order retains the widest items-loop allocation."""
    estimate = _items_retains_key_sized_local_owner.estimate_resources(
        inputs={"data": data}
    )

    assert estimate.width.allocated_qubits == 6
    assert estimate.width.peak_qubits == 6


def test_items_marks_earlier_wider_owner_as_conservative() -> None:
    """Owner-wise maxima report when they may exceed final-state liveness."""
    estimate = _items_retains_key_sized_local_owner.estimate_resources(
        inputs={"data": {3: 0.1, 1: 0.2}}
    )

    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize(
    ("data", "expected_width"),
    [({}, 3), ({0: 0.1}, 4), ({0: 0.1, 1: 0.2, 2: 0.3}, 6)],
)
def test_items_projects_live_width_to_final_carry(
    data: dict[int, float],
    expected_width: int,
) -> None:
    """An items-local output width does not leak its carry symbol."""
    symbolic = _items_retains_carry_sized_local_owner.estimate_resources()
    substituted = symbolic.substitute(**{"|data|": len(data)})
    direct = _items_retains_carry_sized_local_owner.estimate_resources(
        inputs={"data": data}
    )

    assert direct.parameters == {}
    assert direct.width.peak_qubits == expected_width
    assert substituted.width == direct.width


@pytest.mark.parametrize(("trip_count", "expected_peak"), [(0, 3), (1, 4)])
def test_while_retains_unmeasured_local_sibling(
    trip_count: int,
    expected_peak: int,
) -> None:
    """A nonempty while leaves its unmeasured local array sibling live."""
    estimate = _while_retains_unmeasured_sibling.estimate_resources().substitute(
        **{"|while|": trip_count}
    )

    assert estimate.width.peak_qubits == expected_peak


def test_fully_consumed_branch_owner_does_not_cross_boundary() -> None:
    """A complete branch-local measurement permits later width reuse."""
    estimate = _runtime_if_consumes_local_owner.estimate_resources()

    assert estimate.width.allocated_qubits == 6
    assert estimate.width.peak_qubits == 3


def test_partial_branch_return_keeps_unreturned_array_sibling_live() -> None:
    """A merged element and its inaccessible sibling count one pair in total."""
    estimate = _runtime_if_returns_one_local_element.estimate_resources()

    assert estimate.width.allocated_qubits == 6
    assert estimate.width.peak_qubits == 4


@pytest.mark.parametrize(
    "kernel",
    [
        _if_containing_range_retains_local_owner,
        _range_containing_if_retains_local_owner,
    ],
)
def test_nested_control_flow_propagates_local_owner_to_caller(kernel: object) -> None:
    """Nested branch and loop boundaries retain inner live allocations."""
    estimate = kernel.estimate_resources()

    assert estimate.width.allocated_qubits == 6
    assert estimate.width.peak_qubits == 5


@pytest.mark.parametrize(
    "operation",
    [
        ForOperation(
            operands=[
                Value(type=UIntType(), name="start").with_const(0),
                Value(type=UIntType(), name="stop").with_const(1),
                Value(type=UIntType(), name="step").with_const(1),
            ],
            loop_var="index",
            loop_var_value=Value(type=UIntType(), name="index"),
            region_args=(_quantum_region_arg(),),
        ),
        ForItemsOperation(region_args=(_quantum_region_arg(),)),
    ],
)
def test_quantum_loop_region_args_fail_closed(operation: Operation) -> None:
    """Unsupported quantum carries cannot silently distort boundary width."""
    with pytest.raises(NotImplementedError, match="quantum .*region arguments"):
        qm.ResourceEstimator().estimate([operation])
