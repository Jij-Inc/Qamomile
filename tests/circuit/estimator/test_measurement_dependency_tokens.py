"""Regression tests for measurement-derived dependency-token scheduling."""

from __future__ import annotations

import pytest

import qamomile.circuit as qm
import qamomile.observable as qm_o


@qm.qkernel
def _gate_on_negated_measurement(
    measured: qm.Bit,
    target: qm.Qubit,
) -> qm.Qubit:
    """Conditionally gate a target using a negated measured formal."""
    derived = ~measured
    if derived:
        target = qm.h(target)
    return target


def test_classical_negation_forwards_measurement_token() -> None:
    """A negated measurement delays only its runtime-controlled target."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Measure one wire and apply two gates under its negation."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        derived = ~measured
        if derived:
            target = qm.h(target)
            target = qm.x(target)
        return target

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 3
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_disjoint_ifs_reading_same_bit_overlap() -> None:
    """Two disjoint runtime branches sharing one token are not serialized."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Use one measured bit to gate two independent targets."""
        measured = qm.measure(qm.qubit("source"))
        left = qm.qubit("left")
        right = qm.qubit("right")
        if measured:
            left = qm.h(left)
        if measured:
            right = qm.x(right)
        return left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_post_if_disjoint_work_does_not_wait_for_branch() -> None:
    """Work outside a runtime branch overlaps when its wire is untouched."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Gate an independent wire after a measurement-controlled branch."""
        measured = qm.measure(qm.qubit("source"))
        branch_target = qm.qubit("branch_target")
        independent = qm.qubit("independent")
        if measured:
            branch_target = qm.h(branch_target)
        independent = qm.x(independent)
        return branch_target, independent

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1


def test_post_if_target_work_waits_for_runtime_branch() -> None:
    """Work on a runtime branch target waits for its possible producer."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Gate one target inside and immediately after a runtime branch."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if measured:
            target = qm.h(target)
        return qm.z(target)

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 3
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1


def test_inactive_compile_branch_drops_nested_measurement_read() -> None:
    """A dead outer branch does not fence an independent live branch."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> tuple[qm.Bit, qm.Qubit]:
        """Read a measurement only below the true compile-time branch."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if flag:
            if measured:
                target = qm.z(target)
        else:
            target = qm.h(target)
            target = qm.x(target)
        return measured, target

    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    projected = symbolic.substitute(flag=0)
    direct = circuit.estimate_resources(
        inputs={"flag": 0},
        basis=qm.GateBasis.LOGICAL,
    )

    for estimate in (projected, direct):
        assert estimate.gates.total == 2
        assert estimate.measurements.total == 1
        assert estimate.depth.depth == 2
        assert estimate.depth.gate_depth == 2
        assert estimate.depth.measurement_depth == 1
        assert estimate.quality is qm.EstimateQuality.EXACT


def test_range_preserves_nested_compile_branch_read_guard() -> None:
    """A range body retains the guard on a nested measurement read."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> tuple[qm.Bit, qm.Qubit]:
        """Put the guarded measurement read inside one loop iteration."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        for _ in qm.range(1):
            if flag:
                if measured:
                    target = qm.z(target)
            else:
                target = qm.h(target)
                target = qm.x(target)
        return measured, target

    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    projected = symbolic.substitute(flag=0)
    direct = circuit.estimate_resources(
        inputs={"flag": 0},
        basis=qm.GateBasis.LOGICAL,
    )

    for estimate in (projected, direct):
        assert estimate.gates.total == 2
        assert estimate.measurements.total == 1
        assert estimate.depth.depth == 2
        assert estimate.depth.gate_depth == 2
        assert estimate.depth.measurement_depth == 1
        assert estimate.quality is qm.EstimateQuality.EXACT


def test_zero_trip_range_drops_body_measurement_read() -> None:
    """A zero-trip loop does not retain a body-only measurement edge."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> tuple[qm.Bit, qm.Qubit]:
        """Read a measurement only from the optionally executed body."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        for _ in qm.range(repetitions):
            if measured:
                target = qm.z(target)
        target = qm.h(target)
        target = qm.x(target)
        return measured, target

    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    projected = symbolic.substitute(repetitions=0)
    direct = circuit.estimate_resources(
        inputs={"repetitions": 0},
        basis=qm.GateBasis.LOGICAL,
    )

    for estimate in (projected, direct):
        assert estimate.gates.total == 2
        assert estimate.measurements.total == 1
        assert estimate.depth.depth == 2
        assert estimate.depth.gate_depth == 2
        assert estimate.depth.measurement_depth == 1


def test_range_projects_dead_induction_branch_measurement_read() -> None:
    """A branch false at every iteration creates no measurement edge."""

    @qm.qkernel
    def circuit(
        repetitions: qm.UInt,
        cutoff: qm.UInt,
    ) -> tuple[qm.Bit, qm.Qubit]:
        """Read a measurement only below an induction-variable predicate."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        for index in qm.range(repetitions):
            if index < cutoff:
                if measured:
                    target = qm.z(target)
            else:
                target = qm.h(target)
        return measured, target

    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    projected = symbolic.substitute(repetitions=2, cutoff=0)
    direct = circuit.estimate_resources(
        inputs={"repetitions": 2, "cutoff": 0},
        basis=qm.GateBasis.LOGICAL,
    )

    for estimate in (projected, direct):
        assert estimate.gates.total == 2
        assert estimate.measurements.total == 1
        assert estimate.depth.depth == 2
        assert estimate.depth.gate_depth == 2
        assert estimate.depth.measurement_depth == 1
        assert estimate.quality is qm.EstimateQuality.EXACT
        assert estimate.parameters == {}


def test_nested_zero_range_does_not_leak_induction_symbol() -> None:
    """A nested empty range removes its body edge and private loop symbol."""

    @qm.qkernel
    def circuit() -> tuple[qm.Bit, qm.Qubit]:
        """Use the outer induction value as an empty inner-loop bound."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        for outer in qm.range(1):
            for _ in qm.range(outer):
                if measured:
                    target = qm.z(target)
        target = qm.h(target)
        return measured, target

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 1
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 1
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT
    assert estimate.parameters == {}


def test_items_preserves_nested_compile_branch_read_guard() -> None:
    """An items body retains the guard on a nested measurement read."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
        flag: qm.UInt,
    ) -> tuple[qm.Bit, qm.Qubit]:
        """Put the guarded measurement read inside one items iteration."""
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        for _key, _value in qm.items(data):
            if flag:
                if measured:
                    target = qm.z(target)
            else:
                target = qm.h(target)
                target = qm.x(target)
        return measured, target

    estimate = circuit.estimate_resources(
        inputs={"data": {0: 1.0}, "flag": 0},
        basis=qm.GateBasis.LOGICAL,
    )

    assert estimate.gates.total == 2
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_measurement_token_crosses_helper_call() -> None:
    """A measured actual reaches a helper formal without fencing other wires."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Pass a measurement to a helper and then gate a disjoint target."""
        measured = qm.measure(qm.qubit("source"))
        branch_target = _gate_on_negated_measurement(
            measured,
            qm.qubit("branch_target"),
        )
        independent = qm.x(qm.qubit("independent"))
        return branch_target, independent

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_measurement_array_element_token_crosses_helper_call() -> None:
    """A measured array element taints the corresponding helper formal."""

    @qm.qkernel
    def helper(measured: qm.Bit, target: qm.Qubit) -> qm.Qubit:
        """Apply a two-layer branch selected by a measured formal."""
        if measured:
            target = qm.h(target)
            target = qm.x(target)
        return target

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Pass one vector-measurement element through a call boundary."""
        measured = qm.measure(qm.qubit_array(2, "source"))
        return helper(measured[0], qm.qubit("target"))

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 3
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_inline_helper_preserves_measurement_input_readiness() -> None:
    """Helper extraction does not round a measured input up to call depth."""

    @qm.qkernel
    def helper(
        measured: qm.Bit,
        dependent: qm.Qubit,
        independent: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Gate dependent and independent paths with different readiness."""
        if measured:
            dependent = qm.z(dependent)
        independent = qm.h(independent)
        independent = qm.x(independent)
        return dependent, independent

    @qm.qkernel
    def nested() -> tuple[qm.Qubit, qm.Qubit]:
        """Invoke the helper after producing its measured condition."""
        measured = qm.measure(qm.qubit("source"))
        return helper(
            measured,
            qm.qubit("dependent"),
            qm.qubit("independent"),
        )

    @qm.qkernel
    def inline() -> tuple[qm.Qubit, qm.Qubit]:
        """Write the helper operations directly in the root qkernel."""
        measured = qm.measure(qm.qubit("source"))
        dependent = qm.qubit("dependent")
        independent = qm.qubit("independent")
        if measured:
            dependent = qm.z(dependent)
        independent = qm.h(independent)
        independent = qm.x(independent)
        return dependent, independent

    nested_estimate = nested.estimate_resources(basis=qm.GateBasis.LOGICAL)
    inline_estimate = inline.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert nested_estimate.gates == inline_estimate.gates
    assert nested_estimate.measurements == inline_estimate.measurements
    assert nested_estimate.width == inline_estimate.width
    assert nested_estimate.depth == inline_estimate.depth
    assert nested_estimate.depth.depth == 2
    assert nested_estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_valid_width_contract_does_not_restore_call_boundary_depth() -> None:
    """A valid width contract is checked without boxing the helper body."""

    @qm.qkernel
    def helper(
        measured: qm.Bit,
        dependent: qm.Qubit,
        independent: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Use one measured path beside an independent two-layer path."""
        if measured:
            dependent = qm.z(dependent)
        independent = qm.h(independent)
        independent = qm.x(independent)
        return dependent, independent

    contracted = helper._clone_with_callable_attrs(
        {
            "resource_contract": {
                "quantum_operand_widths": [
                    {"index": 0, "name": "dependent", "width": 1},
                    {"index": 1, "name": "independent", "width": 1},
                ]
            }
        }
    )

    @qm.qkernel
    def nested() -> tuple[qm.Qubit, qm.Qubit]:
        """Invoke a width-contracted helper after one measurement."""
        measured = qm.measure(qm.qubit("source"))
        return contracted(
            measured,
            qm.qubit("dependent"),
            qm.qubit("independent"),
        )

    @qm.qkernel
    def inline() -> tuple[qm.Qubit, qm.Qubit]:
        """Write the contracted helper's operations directly."""
        measured = qm.measure(qm.qubit("source"))
        dependent = qm.qubit("dependent")
        independent = qm.qubit("independent")
        if measured:
            dependent = qm.z(dependent)
        independent = qm.h(independent)
        independent = qm.x(independent)
        return dependent, independent

    nested_estimate = nested.estimate_resources(basis=qm.GateBasis.LOGICAL)
    sequence_estimate = qm.estimate_resources(
        nested.block.operations,
        basis=qm.GateBasis.LOGICAL,
    )
    inline_estimate = inline.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert nested_estimate.gates == inline_estimate.gates
    assert nested_estimate.measurements == inline_estimate.measurements
    assert nested_estimate.width == inline_estimate.width
    assert nested_estimate.depth == inline_estimate.depth
    assert sequence_estimate.depth == inline_estimate.depth
    assert nested_estimate.depth.depth == 2
    assert nested_estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_inline_helper_preserves_measurement_output_readiness() -> None:
    """A helper result is ready when measured, not at the call's final layer."""

    @qm.qkernel
    def helper(
        source: qm.Qubit,
        independent: qm.Qubit,
    ) -> tuple[qm.Bit, qm.Qubit]:
        """Measure one path before completing an independent two-layer path."""
        measured = qm.measure(source)
        independent = qm.h(independent)
        independent = qm.x(independent)
        return measured, independent

    @qm.qkernel
    def nested() -> tuple[qm.Qubit, qm.Qubit]:
        """Use an early helper measurement beside its independent path."""
        measured, independent = helper(
            qm.qubit("source"),
            qm.qubit("independent"),
        )
        dependent = qm.qubit("dependent")
        if measured:
            dependent = qm.z(dependent)
        return dependent, independent

    @qm.qkernel
    def inline() -> tuple[qm.Qubit, qm.Qubit]:
        """Write the early measurement and independent path directly."""
        measured = qm.measure(qm.qubit("source"))
        independent = qm.h(qm.qubit("independent"))
        independent = qm.x(independent)
        dependent = qm.qubit("dependent")
        if measured:
            dependent = qm.z(dependent)
        return dependent, independent

    nested_estimate = nested.estimate_resources(basis=qm.GateBasis.LOGICAL)
    inline_estimate = inline.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert nested_estimate.gates == inline_estimate.gates
    assert nested_estimate.measurements == inline_estimate.measurements
    assert nested_estimate.width == inline_estimate.width
    assert nested_estimate.depth == inline_estimate.depth
    assert nested_estimate.depth.depth == 2
    assert nested_estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_inline_helper_preserves_reusable_workspace_liveness() -> None:
    """Sequential helper calls and direct code reuse the same peak workspace."""

    @qm.qkernel
    def helper(target: qm.Qubit) -> qm.Qubit:
        """Use and consume one temporary qubit before returning."""
        workspace = qm.qubit("workspace")
        target, workspace = qm.cx(target, workspace)
        qm.measure(workspace)
        return target

    @qm.qkernel
    def nested() -> qm.Qubit:
        """Invoke the workspace helper twice in sequence."""
        target = qm.qubit("target")
        target = helper(target)
        return helper(target)

    @qm.qkernel
    def inline() -> qm.Qubit:
        """Write the two temporary-workspace lifetimes directly."""
        target = qm.qubit("target")
        first_workspace = qm.qubit("first_workspace")
        target, first_workspace = qm.cx(target, first_workspace)
        qm.measure(first_workspace)
        second_workspace = qm.qubit("second_workspace")
        target, second_workspace = qm.cx(target, second_workspace)
        qm.measure(second_workspace)
        return target

    nested_estimate = nested.estimate_resources(basis=qm.GateBasis.LOGICAL)
    inline_estimate = inline.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert nested_estimate.gates == inline_estimate.gates
    assert nested_estimate.measurements == inline_estimate.measurements
    assert nested_estimate.width == inline_estimate.width
    assert nested_estimate.width.peak_qubits == 2
    assert nested_estimate.depth == inline_estimate.depth


def test_inline_helper_retains_quantum_width_contract_boundary() -> None:
    """Inlining never removes a body-backed callable's width validation."""

    @qm.qkernel
    def body(register: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
        """Return one contracted register unchanged."""
        return register

    contracted = body._clone_with_callable_attrs(
        {
            "resource_contract": {
                "quantum_operand_widths": [{"index": 0, "name": "register", "width": 2}]
            }
        }
    )

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Pass a three-qubit register to the contracted helper."""
        return contracted(qm.qubit_array(3, "register"))

    with pytest.raises(ValueError, match="register width must equal 2"):
        circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)


def test_tuple_expval_dynamic_index_waits_for_whole_root_owner() -> None:
    """A dynamic tuple carrier cannot bypass its root preparation depth."""

    @qm.qkernel
    def circuit(index: qm.UInt, observable: qm.Observable) -> qm.Float:
        """Prepare one root lane before observing a dynamically chosen lane."""
        register = qm.qubit_array(2, "register")
        register[0] = qm.h(register[0])
        register[0] = qm.x(register[0])
        return qm.expval((register[index],), observable)

    observable = qm_o.Z(0)
    symbolic = circuit.estimate_resources(inputs={"observable": observable})
    selected_zero = circuit.estimate_resources(
        inputs={"index": 0, "observable": observable}
    )

    assert symbolic.depth.depth == 3
    assert selected_zero.depth.depth == 3
    assert symbolic.depth.measurement_depth == 1
    assert selected_zero.depth.measurement_depth == 1
    assert any(
        "unresolved quantum index" in assumption.message
        for assumption in symbolic.assumptions
    )


def test_range_publishes_measurement_array_readiness_after_loop() -> None:
    """A stored measurement token remains visible after a range boundary."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Measure array elements in a loop before reading the first result."""
        controls = qm.qubit_array(3, "controls")
        bits = qm.bit_array(3, "bits")
        for index in qm.range(3):
            bits[index] = qm.measure(controls[index])
        target = qm.qubit("target")
        if bits[0]:
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.depth.gate_depth == 1
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_items_publishes_measurement_array_readiness_after_loop() -> None:
    """A concrete items loop publishes the measured element's token."""

    @qm.qkernel
    def circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Store selected measurements before reading the first result."""
        controls = qm.qubit_array(2, "controls")
        bits = qm.bit_array(2, "bits")
        for index, _value in qm.items(data):
            bits[index] = qm.measure(controls[index])
        target = qm.qubit("target")
        if bits[0]:
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources(
        inputs={"data": {0: 0.5}},
        basis=qm.GateBasis.LOGICAL,
    )

    assert estimate.depth.depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.depth.gate_depth == 1
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize("selector", [0, 1])
def test_compile_time_merge_preserves_measurement_element_token(
    selector: int,
) -> None:
    """A selected measured-array element remains a runtime condition."""

    @qm.qkernel
    def circuit(selected_source: qm.UInt) -> qm.Qubit:
        """Choose one measured element before a later runtime branch."""
        measured = qm.measure(qm.qubit_array(2, "sources"))
        if selected_source:
            selected = measured[0]
        else:
            selected = measured[1]
        target = qm.qubit("target")
        if selected:
            target = qm.h(target)
            target = qm.x(target)
        return qm.z(target)

    estimate = circuit.estimate_resources(
        inputs={"selected_source": selector},
        basis=qm.GateBasis.LOGICAL,
    )

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 4
    assert estimate.depth.gate_depth == 3
    assert estimate.depth.clifford_depth == 3
    assert estimate.depth.measurement_depth == 1
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_compile_time_merge_does_not_delay_independent_branch_work() -> None:
    """Merge-only measured sources do not fence unrelated branch gates."""

    @qm.qkernel
    def circuit(selector: qm.UInt) -> tuple[qm.Qubit, qm.Qubit]:
        """Forward one measured bit beside an independent selected branch."""
        left_source = qm.measure(qm.qubit("left_source"))
        right_source = qm.measure(qm.qubit("right_source"))
        branch_target = qm.qubit("branch_target")
        if selector:
            branch_target = qm.h(branch_target)
            branch_target = qm.x(branch_target)
            selected = left_source
        else:
            selected = right_source
        feed_forward_target = qm.qubit("feed_forward_target")
        if selected:
            feed_forward_target = qm.t(feed_forward_target)
        return branch_target, feed_forward_target

    estimate = circuit.estimate_resources(
        inputs={"selector": 1},
        basis=qm.GateBasis.LOGICAL,
    )

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_measurement_derived_index_waits_for_token_and_whole_owner() -> None:
    """A runtime index waits for its token and every possible owner element."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Index a register using a UInt selected by a measured branch."""
        measured = qm.measure(qm.qubit("source"))
        index = qm.uint(0)
        if measured:
            index = qm.uint(1)

        register = qm.qubit_array(2, "register")
        register[1] = qm.h(register[1])
        register[1] = qm.x(register[1])
        register[index] = qm.z(register[index])
        return register

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    assert estimate.depth.gate_depth == 3
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_opaque_depth_occupies_only_actual_operand() -> None:
    """Opaque aggregate depth blocks its target but not a disjoint wire."""
    oracle = qm.opaque(
        "operand_local_depth",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(total=2),
            depth=qm.DepthResources(depth=2, gate_depth=2),
        ),
    )

    @qm.qkernel
    def same_operand() -> qm.Qubit:
        """Gate an opaque target again after its declared aggregate depth."""
        target = qm.qubit("target")
        (target,) = oracle(target)
        return qm.h(target)

    @qm.qkernel
    def disjoint_operand() -> tuple[qm.Qubit, qm.Qubit]:
        """Run a longer independent path beside one opaque invocation."""
        opaque_target = qm.qubit("opaque_target")
        independent = qm.qubit("independent")
        (opaque_target,) = oracle(opaque_target)
        independent = qm.h(independent)
        independent = qm.x(independent)
        independent = qm.z(independent)
        return opaque_target, independent

    same_estimate = same_operand.estimate_resources(basis=qm.GateBasis.LOGICAL)
    disjoint_estimate = disjoint_operand.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert same_estimate.depth.depth == 3
    assert same_estimate.depth.gate_depth == 3
    assert disjoint_estimate.depth.depth == 3
    assert disjoint_estimate.depth.gate_depth == 3


def test_zero_operand_opaque_with_nonzero_depth_is_rejected() -> None:
    """A nonzero opaque duration without a schedulable operand fails closed."""
    oracle = qm.opaque(
        "zero_operand_depth",
        num_qubits=0,
        cost=qm.ResourceEstimate(
            depth=qm.DepthResources(depth=2, gate_depth=2),
        ),
    )

    @qm.qkernel
    def circuit() -> None:
        """Invoke a zero-arity opaque callable with declared depth."""
        oracle()

    with pytest.raises(ValueError, match=r"nonzero-depth opaque.*operand"):
        circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)


def test_zero_operand_opaque_workspace_cannot_replace_external_endpoint() -> None:
    """Private workspace alone cannot anchor opaque latency in caller DAG."""
    oracle = qm.opaque(
        "zero_operand_workspace_depth",
        num_qubits=0,
        cost=qm.ResourceEstimate(
            width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
            depth=qm.DepthResources(depth=2, gate_depth=2),
        ),
    )

    @qm.qkernel
    def circuit() -> None:
        """Invoke a zero-arity opaque callable with private workspace."""
        oracle()

    with pytest.raises(ValueError, match=r"nonzero-depth opaque.*operand"):
        circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)


def test_concrete_loops_preserve_cross_iteration_observation_dependencies() -> None:
    """A carried observation delays dependent work in the next iteration."""

    @qm.qkernel
    def range_circuit(repetitions: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Carry QFixed measurements across disjoint range targets."""
        source0 = qm.cast(qm.qubit_array(1, "range_source0"), qm.QFixed, int_bits=0)
        source1 = qm.cast(qm.qubit_array(1, "range_source1"), qm.QFixed, int_bits=0)
        source2 = qm.cast(qm.qubit_array(1, "range_source2"), qm.QFixed, int_bits=0)
        targets = qm.qubit_array(3, "range_targets")
        state = qm.float_(0.0)
        for index in qm.range(repetitions):
            if state > 0.0:
                targets[index] = qm.h(targets[index])
            if index == 0:
                state = qm.measure(source0)
            elif index == 1:
                state = qm.measure(source1)
            else:
                state = qm.measure(source2)
        return targets

    @qm.qkernel
    def items_circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Vector[qm.Qubit]:
        """Carry QFixed measurements across disjoint item-keyed targets."""
        source0 = qm.cast(qm.qubit_array(1, "items_source0"), qm.QFixed, int_bits=0)
        source1 = qm.cast(qm.qubit_array(1, "items_source1"), qm.QFixed, int_bits=0)
        source2 = qm.cast(qm.qubit_array(1, "items_source2"), qm.QFixed, int_bits=0)
        targets = qm.qubit_array(3, "items_targets")
        state = qm.float_(0.0)
        for index, _value in qm.items(data):
            if state > 0.0:
                targets[index] = qm.h(targets[index])
            if index == 0:
                state = qm.measure(source0)
            elif index == 1:
                state = qm.measure(source1)
            else:
                state = qm.measure(source2)
        return targets

    range_estimate = range_circuit.estimate_resources(
        inputs={"repetitions": 3},
        basis=qm.GateBasis.LOGICAL,
    )
    items_estimate = items_circuit.estimate_resources(
        inputs={"data": {0: 0.0, 1: 1.0, 2: 2.0}},
        basis=qm.GateBasis.LOGICAL,
    )

    for estimate in (range_estimate, items_estimate):
        assert estimate.gates.total == 2
        assert estimate.measurements.total == 3
        assert estimate.depth.depth >= 2
        assert estimate.parameters == {}
        assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert range_estimate.depth == items_estimate.depth


def test_symbolic_loop_preserves_carried_measurement_dependency() -> None:
    """A symbolic scalar carry cannot parallelize across observations."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Use each measurement to control work in the next iteration."""
        sources = qm.qubit_array(3, "sources")
        targets = qm.qubit_array(3, "targets")
        state = qm.uint(0)
        for index in qm.range(repetitions):
            if state > 0:
                targets[index] = qm.h(targets[index])
            observed = qm.measure(sources[index])
            state = qm.uint(0)
            if observed:
                state = qm.uint(1)
        return targets

    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    projected = symbolic.substitute(repetitions=3)
    direct = circuit.estimate_resources(
        inputs={"repetitions": 3},
        basis=qm.GateBasis.LOGICAL,
    )

    assert projected.gates.total >= direct.gates.total
    assert projected.measurements.total == direct.measurements.total == 3
    assert projected.depth.depth >= direct.depth.depth
    assert symbolic.quality is qm.EstimateQuality.CONSERVATIVE
    assert projected.parameters == {}
