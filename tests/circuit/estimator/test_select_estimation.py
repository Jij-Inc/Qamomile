"""Resource-estimator coverage for the SELECT operation."""

from __future__ import annotations

import qamomile.circuit as qm


@qm.qkernel
def _identity_case(target: qm.Qubit) -> qm.Qubit:
    """Return a SELECT target unchanged."""
    return target


@qm.qkernel
def _x_case(target: qm.Qubit) -> qm.Qubit:
    """Apply X to a SELECT target."""
    return qm.x(target)


@qm.qkernel
def _repeated_x_case(target: qm.Qubit, repetitions: qm.UInt) -> qm.Qubit:
    """Apply X ``repetitions`` times to a SELECT target."""
    for _index in qm.range(repetitions):
        target = qm.x(target)
    return target


@qm.qkernel
def _repeated_z_case(target: qm.Qubit, repetitions: qm.UInt) -> qm.Qubit:
    """Apply Z ``repetitions`` times to a SELECT target."""
    for _index in qm.range(repetitions):
        target = qm.z(target)
    return target


@qm.qkernel
def _select_x(index: qm.Qubit, target: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
    """Apply an identity-or-X SELECT for outer-control coverage."""
    index, target = qm.select([_identity_case, _x_case])(index, target)
    return index, target


def test_select_estimates_every_controlled_case_body() -> None:
    """SELECT sums nonempty cases under every index qubit control."""

    @qm.qkernel
    def circuit() -> qm.Bit:
        """Apply four SELECT cases to one target and measure it."""
        index = qm.qubit_array(2, name="index")
        target = qm.qubit("target")
        index, target = qm.select([_identity_case, _x_case, _identity_case, _x_case])(
            index, target
        )
        return qm.measure(target)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.gates.multi_qubit == 2
    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 3


def test_select_estimator_broadcasts_scalar_case_over_vector_target() -> None:
    """A scalar case contributes once per vector target element."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Bit]:
        """Apply a scalar identity-or-X SELECT to three target qubits."""
        index = qm.qubit("index")
        targets = qm.qubit_array(3, name="targets")
        index, targets = qm.select([_identity_case, _x_case])(index, targets)
        return qm.measure(targets)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.two_qubit == 3
    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 4


def test_select_estimator_maps_each_case_parameter_scope() -> None:
    """Each case resolves its own formal parameter from the SELECT call."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Bit:
        """Apply two independently scoped repeated-gate SELECT cases."""
        index = qm.qubit("index")
        target = qm.qubit("target")
        index, target = qm.select([_repeated_x_case, _repeated_z_case])(
            index,
            target,
            repetitions=repetitions,
        )
        return qm.measure(target)

    symbolic = circuit.estimate_resources()
    concrete = circuit.estimate_resources(inputs={"repetitions": 3})

    assert str(symbolic.gates.total) == "2*repetitions"
    assert concrete.gates.total == 6
    assert concrete.gates.two_qubit == 6


def test_select_estimator_accumulates_outer_controls() -> None:
    """An outer control is added to the SELECT index control count."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply a one-index-qubit SELECT under one outer control."""
        outer = qm.qubit("outer")
        index = qm.qubit("index")
        target = qm.qubit("target")
        return qm.control(_select_x)(outer, index, target)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.gates.multi_qubit == 1
    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 3
