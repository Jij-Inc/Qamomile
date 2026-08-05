"""Regression tests for element-wise classical-array observation provenance."""

import pytest

import qamomile.circuit as qm


def test_measurement_store_does_not_taint_sibling_element() -> None:
    """A measured store leaves a fresh sibling element compile-time false."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Read a fresh sibling after storing one measurement."""
        bits = qm.bit_array(2)
        bits[0] = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if bits[1]:
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 1
    assert estimate.depth.gate_depth == 0
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT
    assert estimate.parameters == {}


def test_measurement_store_taints_only_updated_element() -> None:
    """Reading the measured slot retains its readiness dependency."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Gate one target from the measured array element."""
        bits = qm.bit_array(2)
        bits[0] = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if bits[0]:
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_clean_overwrite_kills_element_observation_dependency() -> None:
    """A definite clean overwrite removes the previous measurement token."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Overwrite a measured element before reading it."""
        bits = qm.bit_array(1)
        bits[0] = qm.measure(qm.qubit("source"))
        bits[0] = False
        target = qm.qubit("target")
        if bits[0]:
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT
    assert estimate.parameters == {}


def test_conditional_array_overwrite_specializes_observation_dependency() -> None:
    """A proven array overwrite removes only the inactive measurement edge."""

    @qm.qkernel
    def circuit(selector: qm.UInt) -> qm.Qubit:
        """Conditionally replace a measured array element with true."""
        bits = qm.bit_array(1)
        bits[0] = qm.measure(qm.qubit("source"))
        if selector:
            bits[0] = True
        target = qm.qubit("target")
        if bits[0]:
            target = qm.h(target)
        return target

    symbolic = circuit.estimate_resources()
    projected = [symbolic.substitute(selector=value) for value in (0, 1)]
    direct = [
        circuit.estimate_resources(inputs={"selector": value}) for value in (0, 1)
    ]

    assert [estimate.depth.depth for estimate in projected] == [2, 1]
    assert [estimate.depth.depth for estimate in direct] == [2, 1]
    assert [estimate.quality for estimate in projected] == [
        qm.EstimateQuality.CONSERVATIVE,
        qm.EstimateQuality.EXACT,
    ]
    assert [estimate.quality for estimate in direct] == [
        qm.EstimateQuality.CONSERVATIVE,
        qm.EstimateQuality.EXACT,
    ]


def test_conditional_scalar_overwrite_specializes_observation_dependency() -> None:
    """A proven scalar overwrite removes only the inactive measurement edge."""

    @qm.qkernel
    def circuit(selector: qm.UInt) -> qm.Qubit:
        """Conditionally replace one measured scalar with true."""
        value = qm.measure(qm.qubit("source"))
        if selector:
            value = qm.bit(True)
        target = qm.qubit("target")
        if value:
            target = qm.h(target)
        return target

    symbolic = circuit.estimate_resources()
    projected = [symbolic.substitute(selector=value) for value in (0, 1)]
    direct = [
        circuit.estimate_resources(inputs={"selector": value}) for value in (0, 1)
    ]

    assert [estimate.depth.depth for estimate in projected] == [2, 1]
    assert [estimate.depth.depth for estimate in direct] == [2, 1]
    assert [estimate.quality for estimate in projected] == [
        qm.EstimateQuality.CONSERVATIVE,
        qm.EstimateQuality.EXACT,
    ]
    assert [estimate.quality for estimate in direct] == [
        qm.EstimateQuality.CONSERVATIVE,
        qm.EstimateQuality.EXACT,
    ]


def test_absorbing_boolean_operations_drop_observation_dependency() -> None:
    """Constant AND and OR results do not retain an unused measurement edge."""

    @qm.qkernel
    def absorbed_and() -> qm.Qubit:
        """Conjoin a measured bit with false."""
        observed = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if observed & qm.bit(False):
            target = qm.h(target)
        return target

    @qm.qkernel
    def absorbed_or() -> qm.Qubit:
        """Disjoin a measured bit with true."""
        observed = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if observed | qm.bit(True):
            target = qm.h(target)
        return target

    false_result = absorbed_and.estimate_resources()
    true_result = absorbed_or.estimate_resources()

    assert false_result.gates.total == 0
    assert true_result.gates.total == 1
    assert false_result.depth.depth == true_result.depth.depth == 1
    assert false_result.quality is qm.EstimateQuality.EXACT
    assert true_result.quality is qm.EstimateQuality.EXACT


def test_clean_sibling_gate_remains_parallel_with_measurement() -> None:
    """A known sibling value does not inherit the measured slot's barrier."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Gate from a clean true sibling beside an unrelated measurement."""
        bits = qm.bit_array(2)
        bits[1] = True
        bits[0] = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if bits[1]:
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 1
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_measurement_store_provenance_is_separated_by_element() -> None:
    """Independent measured elements retain independent readiness roots."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Measure two sources and gate two disjoint targets."""
        bits = qm.bit_array(2)
        bits[0] = qm.measure(qm.qubit("left_source"))
        bits[1] = qm.measure(qm.qubit("right_source"))
        left = qm.qubit("left")
        right = qm.qubit("right")
        if bits[0]:
            left = qm.h(left)
        if bits[1]:
            right = qm.x(right)
        return left, right

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.measurements.total == 2
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.depth.measurement_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_slice_projects_measurement_provenance_to_root_element() -> None:
    """A view inherits provenance only from its corresponding root slot."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Read affected and unaffected elements through one slice."""
        bits = qm.bit_array(4)
        bits[2] = qm.measure(qm.qubit("source"))
        view = bits[1:4]
        affected = qm.qubit("affected")
        unaffected = qm.qubit("unaffected")
        if view[1]:
            affected = qm.h(affected)
        if view[0]:
            unaffected = qm.x(unaffected)
        return affected, unaffected

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_helper_slice_store_preserves_untouched_caller_provenance() -> None:
    """A helper Store snapshots the caller view's untouched elements."""

    @qm.qkernel
    def measure_first(
        bits: qm.Vector[qm.Bit],
        source: qm.Qubit,
    ) -> qm.Vector[qm.Bit]:
        """Measure into the first view element and return the updated view.

        Args:
            bits (qm.Vector[qm.Bit]): Caller-provided array view.
            source (qm.Qubit): Qubit measured into the first element.

        Returns:
            qm.Vector[qm.Bit]: Updated array view.
        """
        bits[0] = qm.measure(source)
        return bits

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Read the updated and untouched view elements after one call."""
        returned = measure_first(
            qm.bit_array(5)[1:5:2],
            qm.qubit("source"),
        )
        affected = qm.qubit("affected")
        unaffected = qm.qubit("unaffected")
        if returned[0]:
            affected = qm.h(affected)
        if returned[1]:
            unaffected = qm.x(unaffected)
        return affected, unaffected

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.gate_depth == 1
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_symbolic_store_alias_guards_element_provenance() -> None:
    """Symbolic store/load aliasing remains guarded by index equality."""

    @qm.qkernel
    def circuit(store: qm.UInt, load: qm.UInt) -> qm.Qubit:
        """Store one measurement and read a potentially different slot.

        Args:
            store (qm.UInt): Store index.
            load (qm.UInt): Load index.

        Returns:
            qm.Qubit: Conditionally gated target.
        """
        bits = qm.bit_array(2)
        bits[store] = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if bits[load]:
            target = qm.h(target)
        return target

    same = circuit.estimate_resources(inputs={"store": 0, "load": 0})
    disjoint = circuit.estimate_resources(inputs={"store": 0, "load": 1})
    symbolic = circuit.estimate_resources()

    assert same.gates.total == 1
    assert same.depth.depth == 2
    assert same.quality is qm.EstimateQuality.CONSERVATIVE
    assert disjoint.gates.total == 0
    assert disjoint.depth.depth == 1
    assert disjoint.quality is qm.EstimateQuality.EXACT
    assert set(symbolic.parameters) == {"store", "load"}
    assert symbolic.substitute(store=0, load=0).gates.total == 1
    assert symbolic.substitute(store=0, load=1).gates.total == 0


def test_runtime_branches_start_from_independent_array_snapshots() -> None:
    """A true-branch store cannot leak into evaluation of the false branch."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Write an array only in the sibling of a compile-time-clean read."""
        bits = qm.bit_array(1)
        measured = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if measured:
            bits[0] = True
        else:
            if bits[0]:
                target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == 1
    assert estimate.depth.gate_depth == 0
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert estimate.parameters == {}


def test_zero_trip_array_store_does_not_publish_probe_provenance() -> None:
    """A zero-trip concrete loop leaves its pre-loop array state unchanged."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Measure into one slot only when the loop executes."""
        bits = qm.bit_array(1)
        for _ in qm.range(repetitions):
            bits[0] = qm.measure(qm.qubit("source"))
        target = qm.qubit("target")
        if bits[0]:
            target = qm.h(target)
        return target

    empty = circuit.estimate_resources(inputs={"repetitions": 0})
    one = circuit.estimate_resources(inputs={"repetitions": 1})

    assert empty.gates.total == 0
    assert empty.measurements.total == 0
    assert empty.depth.depth == 0
    assert empty.quality is qm.EstimateQuality.EXACT
    assert one.gates.total == 1
    assert one.measurements.total == 1
    assert one.depth.depth == 2
    assert one.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize("loop_kind", ["range", "items"])
def test_concrete_loop_threads_array_state_across_pass_through_if(
    loop_kind: str,
) -> None:
    """A pass-through branch cannot reset an earlier iteration's Store.

    Args:
        loop_kind (str): Concrete loop form to exercise.
    """

    @qm.qkernel
    def range_circuit() -> qm.Qubit:
        """Measure two range-selected slots around a scalar carry branch."""
        controls = qm.qubit_array(2, "controls")
        bits = qm.bit_array(2)
        counter = qm.uint(0)
        for index in qm.range(2):
            if index:
                counter = counter + 1
            bits[index] = qm.measure(controls[index])
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    @qm.qkernel
    def items_circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Measure two item-keyed slots around a scalar carry branch."""
        controls = qm.qubit_array(2, "controls")
        bits = qm.bit_array(2)
        counter = qm.uint(0)
        for index, _value in qm.items(data):
            if index:
                counter = counter + 1
            bits[index] = qm.measure(controls[index])
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    estimate = (
        range_circuit.estimate_resources()
        if loop_kind == "range"
        else items_circuit.estimate_resources(inputs={"data": {0: 0.0, 1: 1.0}})
    )

    assert estimate.gates.total == 1
    assert estimate.gates.t == 1
    assert estimate.measurements.total == 2
    assert estimate.depth.depth == 2
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_large_region_loop_publishes_array_observation_without_private_symbol() -> None:
    """A loop beyond exact replay keeps post-loop element provenance internal."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Measure every slot while carrying an unrelated scalar value."""
        controls = qm.qubit_array(65, "controls")
        bits = qm.bit_array(65)
        counter = qm.uint(0)
        for index in qm.range(65):
            counter = counter + 1
            bits[index] = qm.measure(controls[index])
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.gates.t == 1
    assert estimate.measurements.total == 65
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_symbolic_region_loop_keeps_post_loop_array_dependency_guarded() -> None:
    """A symbolic trip count publishes only public parameters after the loop."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Measure one slot repeatedly while carrying an unrelated scalar."""
        source = qm.qubit("source")
        bits = qm.bit_array(1)
        counter = qm.uint(0)
        for _ in qm.range(repetitions):
            counter = counter + 1
            bits[0] = qm.measure(source)
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    symbolic = circuit.estimate_resources()
    empty = symbolic.substitute(repetitions=0)
    one = symbolic.substitute(repetitions=1)
    three = symbolic.substitute(repetitions=3)

    assert set(symbolic.parameters) == {"repetitions"}
    assert empty.parameters == {}
    assert empty.gates.total == 0
    assert empty.measurements.total == 0
    assert empty.depth.depth == 0
    assert one.parameters == {}
    assert one.gates.total == 1
    assert one.measurements.total == 1
    assert one.depth.depth == 2
    assert three.parameters == {}
    assert three.gates.total == 1
    assert three.measurements.total == 3
    assert three.depth.depth == 4
    assert three.quality is qm.EstimateQuality.CONSERVATIVE


def test_negative_step_loop_threads_array_state_in_execution_order() -> None:
    """Concrete descending ranges retain every element update."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Measure descending slots and branch on the first visited slot."""
        sources = qm.qubit_array(3, "sources")
        bits = qm.bit_array(3)
        for index in qm.range(2, -1, -1):
            bits[index] = qm.measure(sources[index])
        target = qm.qubit("target")
        if bits[2]:
            target = qm.t(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.gates.t == 1
    assert estimate.measurements.total == 3
    assert estimate.depth.depth == 2
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_symbolic_indexed_range_does_not_leak_bound_symbol() -> None:
    """A range-quantified Store retains only the public trip-count input."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Measure a symbolic prefix and branch on its first element."""
        sources = qm.qubit_array(3, "sources")
        bits = qm.bit_array(3)
        for index in qm.range(repetitions):
            bits[index] = qm.measure(sources[index])
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    symbolic = circuit.estimate_resources()
    empty = symbolic.substitute(repetitions=0)
    one = symbolic.substitute(repetitions=1)
    two = symbolic.substitute(repetitions=2)

    assert set(symbolic.parameters) == {"repetitions"}
    assert empty.parameters == {}
    assert empty.gates.total == 0
    assert empty.measurements.total == 0
    assert one.parameters == {}
    assert one.gates.total == 1
    assert one.measurements.total == 1
    assert one.depth.depth == 2
    assert two.parameters == {}
    assert two.gates.total == 1
    assert two.measurements.total == 2
    assert two.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize("loop_kind", ["range", "items"])
def test_symbolic_array_self_recurrence_uses_safe_unknown_exit(
    loop_kind: str,
) -> None:
    """An unresolved repeated self-update cannot select one traced value.

    Args:
        loop_kind (str): Symbolic loop form to exercise.
    """

    @qm.qkernel
    def range_circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Toggle one initially true bit for an unresolved trip count."""
        bits = qm.bit_array(1)
        bits[0] = True
        for _ in qm.range(repetitions):
            bits[0] = ~bits[0]
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    @qm.qkernel
    def items_circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Toggle one initially true bit for each unresolved dictionary item."""
        bits = qm.bit_array(1)
        bits[0] = True
        for _key, _value in qm.items(data):
            bits[0] = ~bits[0]
        target = qm.qubit("target")
        if bits[0]:
            target = qm.t(target)
        return target

    if loop_kind == "range":
        symbolic = range_circuit.estimate_resources()
        two = symbolic.substitute(repetitions=2)
        direct = range_circuit.estimate_resources(inputs={"repetitions": 2})
        assert set(symbolic.parameters) == {"repetitions"}
    else:
        symbolic = items_circuit.estimate_resources()
        two = symbolic.substitute(**{"|data|": 2})
        direct = items_circuit.estimate_resources(inputs={"data": {0: 0.0, 1: 1.0}})
        assert set(symbolic.parameters) == {"|data|"}

    assert direct.gates.total == 1
    assert two.parameters == {}
    assert two.gates.total >= direct.gates.total
    assert two.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize("loop_kind", ["range", "items"])
def test_symbolic_array_recurrence_is_safe_inside_loop_body(
    loop_kind: str,
) -> None:
    """Carried array state cannot reuse only the first iteration's resources.

    Args:
        loop_kind (str): Symbolic loop form to exercise.
    """

    @qm.qkernel
    def range_circuit(repetitions: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Enable gates after the first range iteration via a carried bit."""
        targets = qm.qubit_array(65, "targets")
        bits = qm.bit_array(1)
        for index in qm.range(repetitions):
            if bits[0]:
                targets[index] = qm.h(targets[index])
            bits[0] = True
        return targets

    @qm.qkernel
    def items_circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Enable a repeated gate after the first dictionary entry."""
        target = qm.qubit("target")
        bits = qm.bit_array(1)
        for _key, _value in qm.items(data):
            if bits[0]:
                target = qm.h(target)
            bits[0] = True
        return target

    if loop_kind == "range":
        symbolic = range_circuit.estimate_resources()
        assert set(symbolic.parameters) == {"repetitions"}
        counts = (0, 1, 2, 3, 65)
        for count in counts:
            projected = symbolic.substitute(repetitions=count)
            direct = range_circuit.estimate_resources(inputs={"repetitions": count})
            assert projected.gates.total >= direct.gates.total
            assert projected.depth.depth >= direct.depth.depth
    else:
        symbolic = items_circuit.estimate_resources()
        assert set(symbolic.parameters) == {"|data|"}
        for count in (0, 1, 2, 3):
            data = {index: float(index) for index in range(count)}
            projected = symbolic.substitute(**{"|data|": count})
            direct = items_circuit.estimate_resources(inputs={"data": data})
            assert projected.gates.total >= direct.gates.total
            assert projected.depth.depth >= direct.depth.depth

    assert symbolic.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize("loop_kind", ["range", "items"])
def test_region_loop_array_recurrence_is_safe_inside_body(
    loop_kind: str,
) -> None:
    """An unrelated scalar carry does not bypass carried-array safeguards.

    Args:
        loop_kind (str): Symbolic loop form to exercise.
    """

    @qm.qkernel
    def range_circuit(
        repetitions: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.UInt]:
        """Carry a counter beside a bit-controlled disjoint range."""
        targets = qm.qubit_array(65, "targets")
        bits = qm.bit_array(1)
        counter = qm.uint(0)
        for index in qm.range(repetitions):
            if bits[0]:
                targets[index] = qm.h(targets[index])
            bits[0] = True
            counter = counter + 1
        return targets, counter

    @qm.qkernel
    def items_circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> tuple[qm.Qubit, qm.UInt]:
        """Carry a counter beside a bit-controlled items loop."""
        target = qm.qubit("target")
        bits = qm.bit_array(1)
        counter = qm.uint(0)
        for _key, _value in qm.items(data):
            if bits[0]:
                target = qm.h(target)
            bits[0] = True
            counter = counter + 1
        return target, counter

    if loop_kind == "range":
        symbolic = range_circuit.estimate_resources()
        for count in (2, 3, 65):
            projected = symbolic.substitute(repetitions=count)
            direct = range_circuit.estimate_resources(inputs={"repetitions": count})
            assert projected.gates.total >= direct.gates.total
            assert projected.depth.depth >= direct.depth.depth
    else:
        symbolic = items_circuit.estimate_resources()
        for count in (2, 3):
            data = {index: float(index) for index in range(count)}
            projected = symbolic.substitute(**{"|data|": count})
            direct = items_circuit.estimate_resources(inputs={"data": data})
            assert projected.gates.total >= direct.gates.total
            assert projected.depth.depth >= direct.depth.depth

    assert symbolic.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize("loop_kind", ["range", "items"])
def test_inlined_helper_preserves_array_loop_lineage(loop_kind: str) -> None:
    """Inlining a helper cannot disconnect its loop-updated array result.

    Args:
        loop_kind (str): Concrete loop form inside the helper.
    """

    @qm.qkernel
    def range_set(
        bits: qm.Vector[qm.Bit],
        repetitions: qm.UInt,
        value: qm.Bit,
    ) -> qm.Vector[qm.Bit]:
        """Set one bit during every range iteration."""
        for _ in qm.range(repetitions):
            bits[0] = value
        return bits

    @qm.qkernel
    def items_set(
        bits: qm.Vector[qm.Bit],
        data: qm.Dict[qm.UInt, qm.Float],
        value: qm.Bit,
    ) -> qm.Vector[qm.Bit]:
        """Set one bit for every dictionary entry."""
        for _key, _value in qm.items(data):
            bits[0] = value
        return bits

    @qm.qkernel
    def range_caller(repetitions: qm.UInt) -> qm.Qubit:
        """Branch on a bit updated by an inlined range helper."""
        result = range_set(qm.bit_array(1), repetitions, qm.bit(True))
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    @qm.qkernel
    def items_caller(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Branch on a bit updated by an inlined items helper."""
        result = items_set(qm.bit_array(1), data, qm.bit(True))
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    if loop_kind == "range":
        estimates = [
            range_caller.estimate_resources(inputs={"repetitions": count})
            for count in (0, 1, 2)
        ]
    else:
        estimates = [
            items_caller.estimate_resources(
                inputs={"data": {index: float(index) for index in range(count)}}
            )
            for count in (0, 1, 2)
        ]

    assert [estimate.gates.total for estimate in estimates] == [0, 1, 1]
    assert [estimate.depth.depth for estimate in estimates] == [0, 1, 1]


@pytest.mark.parametrize("loop_kind", ["range", "items"])
def test_inlined_helper_publishes_array_measurement_dependency(
    loop_kind: str,
) -> None:
    """A helper's loop measurement delays its caller's dependent gate.

    Args:
        loop_kind (str): Concrete loop form inside the helper.
    """

    @qm.qkernel
    def range_measure(
        bits: qm.Vector[qm.Bit],
        repetitions: qm.UInt,
    ) -> qm.Vector[qm.Bit]:
        """Measure distinct sources into one carried bit."""
        sources = qm.qubit_array(2, "range_sources")
        for index in qm.range(repetitions):
            bits[0] = qm.measure(sources[index])
        return bits

    @qm.qkernel
    def items_measure(
        bits: qm.Vector[qm.Bit],
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Vector[qm.Bit]:
        """Measure item-selected sources into one carried bit."""
        sources = qm.qubit_array(2, "items_sources")
        for index, _value in qm.items(data):
            bits[0] = qm.measure(sources[index])
        return bits

    @qm.qkernel
    def range_caller(repetitions: qm.UInt) -> qm.Qubit:
        """Use a measurement returned by an inlined range helper."""
        result = range_measure(qm.bit_array(1), repetitions)
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    @qm.qkernel
    def items_caller(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Use a measurement returned by an inlined items helper."""
        result = items_measure(qm.bit_array(1), data)
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    if loop_kind == "range":
        empty = range_caller.estimate_resources(inputs={"repetitions": 0})
        active = range_caller.estimate_resources(inputs={"repetitions": 2})
    else:
        empty = items_caller.estimate_resources(inputs={"data": {}})
        active = items_caller.estimate_resources(inputs={"data": {0: 0.0, 1: 1.0}})

    assert empty.gates.total == 0
    assert empty.measurements.total == 0
    assert active.gates.total == 1
    assert active.measurements.total == 2
    assert active.depth.depth >= 2
    assert active.quality is qm.EstimateQuality.CONSERVATIVE


def test_inlined_loop_helper_snapshots_caller_pre_store() -> None:
    """A callee loop starts from the caller's latest stored array state."""

    @qm.qkernel
    def toggle(
        bits: qm.Vector[qm.Bit],
        repetitions: qm.UInt,
    ) -> qm.Vector[qm.Bit]:
        """Toggle the first bit once per iteration."""
        for _ in qm.range(repetitions):
            bits[0] = ~bits[0]
        return bits

    @qm.qkernel
    def caller(repetitions: qm.UInt) -> qm.Qubit:
        """Store true before calling the toggling helper."""
        bits = qm.bit_array(1)
        bits[0] = True
        result = toggle(bits, repetitions)
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    estimates = [
        caller.estimate_resources(inputs={"repetitions": count}) for count in (0, 1, 2)
    ]

    assert [estimate.gates.total for estimate in estimates] == [1, 0, 1]
    assert [estimate.depth.depth for estimate in estimates] == [1, 0, 1]
    assert all(estimate.parameters == {} for estimate in estimates)


def test_inlined_loop_helper_preserves_slice_actual_state() -> None:
    """A slice actual keeps its zero-trip value and symbolic loop state."""

    @qm.qkernel
    def set_first(
        bits: qm.Vector[qm.Bit],
        repetitions: qm.UInt,
    ) -> qm.Vector[qm.Bit]:
        """Set the first view element in every iteration."""
        for _ in qm.range(repetitions):
            bits[0] = True
        return bits

    @qm.qkernel
    def caller(repetitions: qm.UInt) -> qm.Qubit:
        """Pass a fresh array slice and branch on the helper result."""
        bits = qm.bit_array(3)
        result = set_first(bits[1:3], repetitions)
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    symbolic = caller.estimate_resources()
    direct = [
        caller.estimate_resources(inputs={"repetitions": count}) for count in (0, 1, 2)
    ]
    projected = [symbolic.substitute(repetitions=count) for count in (0, 1, 2)]

    assert set(symbolic.parameters) == {"repetitions"}
    assert [estimate.gates.total for estimate in direct] == [0, 1, 1]
    assert [estimate.gates.total for estimate in projected] == [0, 1, 1]
    assert [estimate.depth.depth for estimate in direct] == [0, 1, 1]
    assert [estimate.depth.depth for estimate in projected] == [0, 1, 1]
    assert all(estimate.parameters == {} for estimate in (*direct, *projected))


def test_inlined_loop_helper_preserves_choice_actual_on_zero_trip() -> None:
    """A zero-trip helper returns the selected caller array unchanged."""

    @qm.qkernel
    def set_first(
        bits: qm.Vector[qm.Bit],
        repetitions: qm.UInt,
    ) -> qm.Vector[qm.Bit]:
        """Set the first element only when the loop executes."""
        for _ in qm.range(repetitions):
            bits[0] = True
        return bits

    @qm.qkernel
    def caller(selector: qm.UInt, repetitions: qm.UInt) -> qm.Qubit:
        """Select a false or true array before calling the helper."""
        left = qm.bit_array(1)
        right = qm.bit_array(1)
        right[0] = True
        selected = left
        if selector:
            selected = right
        result = set_first(selected, repetitions)
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    estimates = [
        caller.estimate_resources(inputs={"selector": selector, "repetitions": 0})
        for selector in (0, 1)
    ]

    assert [estimate.gates.total for estimate in estimates] == [0, 1]
    assert [estimate.depth.depth for estimate in estimates] == [0, 1]
    assert all(estimate.parameters == {} for estimate in estimates)


def test_inlined_loop_helper_preserves_measurement_slice_readiness() -> None:
    """A measured slice result delays its caller's dependent gate."""

    @qm.qkernel
    def measure_first(
        bits: qm.Vector[qm.Bit],
        repetitions: qm.UInt,
    ) -> qm.Vector[qm.Bit]:
        """Measure distinct sources into the first view element."""
        sources = qm.qubit_array(2, "sources")
        for index in qm.range(repetitions):
            bits[0] = qm.measure(sources[index])
        return bits

    @qm.qkernel
    def caller(repetitions: qm.UInt) -> qm.Qubit:
        """Pass a slice and consume its measurement after the call."""
        bits = qm.bit_array(3)
        result = measure_first(bits[1:3], repetitions)
        target = qm.qubit("target")
        if result[0]:
            target = qm.x(target)
        return target

    symbolic = caller.estimate_resources()
    empty = symbolic.substitute(repetitions=0)
    active = caller.estimate_resources(inputs={"repetitions": 2})
    projected = symbolic.substitute(repetitions=2)

    assert set(symbolic.parameters) == {"repetitions"}
    assert empty.gates.total == 0
    assert empty.measurements.total == 0
    assert empty.depth.depth == 0
    for estimate in (active, projected):
        assert estimate.parameters == {}
        assert estimate.gates.total == 1
        assert estimate.measurements.total == 2
        assert estimate.depth.depth == 2
        assert estimate.depth.measurement_depth == 1
        assert estimate.depth.gate_depth == 1


def test_inlined_array_helper_rebinds_inside_symbolic_loop() -> None:
    """A call inside a loop publishes its updated array each iteration."""

    @qm.qkernel
    def set_first(
        bits: qm.Vector[qm.Bit],
        value: qm.Bit,
    ) -> qm.Vector[qm.Bit]:
        """Set and return the first array element."""
        bits[0] = value
        return bits

    @qm.qkernel
    def caller(repetitions: qm.UInt) -> qm.Qubit:
        """Invoke an array-updating helper from the repeated body."""
        bits = qm.bit_array(1)
        for _ in qm.range(repetitions):
            bits = set_first(bits, qm.bit(True))
        target = qm.qubit("target")
        if bits[0]:
            target = qm.x(target)
        return target

    symbolic = caller.estimate_resources()
    direct = [
        caller.estimate_resources(inputs={"repetitions": count}) for count in (0, 1, 2)
    ]
    projected = [symbolic.substitute(repetitions=count) for count in (0, 1, 2)]

    assert set(symbolic.parameters) == {"repetitions"}
    assert [estimate.gates.total for estimate in direct] == [0, 1, 1]
    assert [estimate.gates.total for estimate in projected] == [0, 1, 1]
    assert [estimate.depth.depth for estimate in direct] == [0, 1, 1]
    assert [estimate.depth.depth for estimate in projected] == [0, 1, 1]
    assert all(estimate.parameters == {} for estimate in (*direct, *projected))
