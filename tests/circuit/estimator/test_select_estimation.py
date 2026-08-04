"""Resource-estimator coverage for the SELECT operation."""

from __future__ import annotations

import math

import pytest
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.serialization import deserialize, serialize


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


@qm.qkernel
def _phased_identity_case(target: qm.Qubit) -> qm.Qubit:
    """Apply a relative quarter-turn phase to an identity case."""
    return qm.global_phase(_identity_case, math.pi / 2)(target)


@qm.qkernel
def _phase_select(
    index: qm.Qubit,
    target: qm.Qubit,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Select between an identity and a globally phased identity."""
    return qm.select([_identity_case, _phased_identity_case])(index, target)


@qm.qkernel
def _inverse_phase_select(
    index: qm.Qubit,
    target: qm.Qubit,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Apply the inverse of the relative-phase SELECT fixture."""
    return qm.inverse(_phase_select)(index, target)


@qm.qkernel
def _outer_controlled_phase_select() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
    """Apply the relative-phase SELECT under one coherent control."""
    outer = qm.qubit("outer")
    index = qm.qubit("index")
    target = qm.qubit("target")
    return qm.control(_phase_select)(outer, index, target)


@qm.qkernel
def _controlled_inverse_phase_select() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
    """Control the inverse relative-phase SELECT fixture."""
    outer = qm.qubit("outer")
    index = qm.qubit("index")
    target = qm.qubit("target")
    return qm.control(_inverse_phase_select)(outer, index, target)


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

    assert estimate.gates.total == 4
    assert estimate.gates.single_qubit == 2
    assert estimate.gates.multi_qubit == 2
    assert estimate.gates.toffoli == 2
    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 3


def test_select_phase_survives_inverse_and_outer_control() -> None:
    """A case-global phase remains observable under SELECT transformations."""
    direct = _phase_select.estimate_resources()
    inverse = _inverse_phase_select.estimate_resources()
    outer = _outer_controlled_phase_select.estimate_resources()
    outer_inverse = _controlled_inverse_phase_select.estimate_resources()

    for estimate in (direct, inverse):
        assert estimate.gates.total == 1
        assert estimate.gates.single_qubit == 1
        assert estimate.gates.rotation == 1
        assert estimate.width.input_qubits == 2
        assert estimate.width.peak_qubits == 2
        assert estimate.depth.depth == 1
        assert estimate.calls == qm.CallResources()
        assert estimate.quality is qm.EstimateQuality.EXACT
    for estimate in (outer, outer_inverse):
        assert estimate.gates.total == 1
        assert estimate.gates.two_qubit == 1
        assert estimate.gates.rotation == 1
        assert estimate.width.allocated_qubits == 3
        assert estimate.width.peak_qubits == 3
        assert estimate.depth.depth == 1
        assert estimate.calls == qm.CallResources()
        assert estimate.quality is qm.EstimateQuality.EXACT

    assert inverse.width == direct.width
    assert inverse.depth == direct.depth
    assert inverse.calls == direct.calls
    assert outer_inverse.width == outer.width
    assert outer_inverse.depth == outer.depth
    assert outer_inverse.calls == outer.calls

    for kernel, expected in (
        (_phase_select, direct),
        (_inverse_phase_select, inverse),
        (_outer_controlled_phase_select, outer),
        (_controlled_inverse_phase_select, outer_inverse),
    ):
        restored = qm.estimate_resources(deserialize(serialize(kernel)))
        assert restored.width == expected.width
        assert restored.gates == expected.gates
        assert restored.depth == expected.depth
        assert restored.calls == expected.calls
        assert restored.derivation is expected.derivation
        assert restored.quality is expected.quality


def test_select_clifford_t_rejects_undefined_controlled_hadamard() -> None:
    """SELECT rejects a Clifford+T control lowering that is not defined."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Select an identity or Hadamard on one target qubit."""
        index = qm.qubit("index")
        target = qm.qubit("target")
        return qm.select([_identity_case, qm.h])(index, target)

    with pytest.raises(ValueError, match="controlled gate 'h'"):
        circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)


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


def test_symbolic_select_broadcast_matches_direct_outer_control_cost() -> None:
    """Symbolic SELECT broadcast keeps the shared outer-control decomposition."""

    @qm.qkernel
    def select_body(
        index: qm.Qubit,
        targets: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
        """Select identity or X independently for every target."""
        return qm.select([_identity_case, _x_case])(index, targets)

    @qm.qkernel
    def circuit(
        width: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Qubit, qm.Vector[qm.Qubit]]:
        """Apply the symbolic SELECT body under three controls."""
        controls = qm.qubit_array(3, "controls")
        index = qm.qubit("index")
        targets = qm.qubit_array(width, "targets")
        *_, index, targets = qm.control(select_body, num_controls=3)(
            controls,
            index,
            targets,
        )
        return controls, index, targets

    symbolic = circuit.estimate_resources()
    specialized = symbolic.substitute(width=2)
    direct = circuit.estimate_resources(inputs={"width": 2})

    assert specialized.gates == direct.gates
    assert specialized.depth == direct.depth
    assert specialized.width == direct.width
    assert direct.gates.total == 6
    assert direct.width.clean_ancilla_qubits == 2


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

    repetitions = symbolic.parameters["repetitions"]
    expected = sp.Piecewise(
        (2 * repetitions + 2, repetitions > 0),
        (2 * repetitions, True),
    )
    assert sp.simplify(symbolic.gates.total - expected) == 0
    assert concrete.gates.total == 8
    assert concrete.gates.single_qubit == 2
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


def test_select_with_two_outer_controls_shares_the_outer_ladder() -> None:
    """One active case uses its index control under one shared outer carrier."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply an identity-or-X SELECT under two surrounding controls."""
        outer = qm.qubit_array(2, "outer")
        index = qm.qubit("index")
        target = qm.qubit("target")
        *_, index, target = qm.control(_select_x, num_controls=2)(
            outer,
            index,
            target,
        )
        return index, target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.multi_qubit == 3
    assert estimate.gates.toffoli == 3
    assert estimate.depth.depth == 3
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.peak_qubits == 5


def test_empty_select_does_not_create_an_outer_ladder() -> None:
    """An all-identity SELECT remains empty under surrounding controls."""

    @qm.qkernel
    def identity_select(
        index: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Select between two identity cases."""
        return qm.select([_identity_case, _identity_case])(index, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Control the empty SELECT with two surrounding qubits."""
        outer = qm.qubit_array(2, "outer")
        index = qm.qubit("index")
        target = qm.qubit("target")
        *_, index, target = qm.control(identity_select, num_controls=2)(
            outer,
            index,
            target,
        )
        return index, target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.depth.depth == 0
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.width.peak_qubits == 4


def test_select_estimator_resolves_symbolic_index_width() -> None:
    """A bound UInt controls SELECT gate arity and allocated register width."""

    @qm.qkernel
    def circuit(width: qm.UInt) -> qm.Bit:
        """Apply identity-or-X SELECT over a symbolic-size index array."""
        index = qm.qubit_array(width, name="index")
        target = qm.qubit("target")
        index, target = qm.select(
            [_identity_case, _x_case],
            num_index_qubits=width,
        )(index, target)
        return qm.measure(target)

    symbolic = circuit.estimate_resources()
    estimate = symbolic.substitute(width=3)

    assert estimate.gates.total == 9
    assert estimate.gates.single_qubit == 4
    assert estimate.gates.two_qubit == 1
    assert estimate.gates.multi_qubit == 4
    assert estimate.gates.toffoli == 4
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 6
    assert estimate.width.circuit_qubits == 6

    with pytest.raises(ValueError, match="at least 1 qubit"):
        symbolic.substitute(width=0)
    with pytest.raises(ValueError, match="at least 1 qubit"):
        circuit.estimate_resources(inputs={"width": 0})


def test_select_estimator_rejects_empty_target_register() -> None:
    """SELECT requires at least one physical target qubit."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Apply scalar SELECT cases to an empty target array."""
        index = qm.qubit("index")
        targets = qm.qubit_array(0, name="targets")
        index, targets = qm.select([_identity_case, _x_case])(index, targets)
        return targets

    with pytest.raises(
        ValueError,
        match="SELECT target operand width must be at least 1 qubit",
    ):
        circuit.estimate_resources()


def test_select_estimator_retains_symbolic_target_width_constraint() -> None:
    """A symbolic SELECT target width remains constrained after estimation."""

    @qm.qkernel
    def circuit(width: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Broadcast scalar SELECT cases over a symbolic target array."""
        index = qm.qubit("index")
        targets = qm.qubit_array(width, name="targets")
        index, targets = qm.select([_identity_case, _x_case])(index, targets)
        return targets

    symbolic = circuit.estimate_resources()
    concrete = symbolic.substitute(width=2)

    assert concrete.gates.total == 2
    assert concrete.gates.two_qubit == 2
    assert concrete.width.allocated_qubits == 3

    with pytest.raises(
        ValueError,
        match="SELECT target operand width must be at least 1 qubit",
    ):
        symbolic.substitute(width=0)
    with pytest.raises(
        ValueError,
        match="SELECT target operand width must be at least 1 qubit",
    ):
        circuit.estimate_resources(inputs={"width": 0})


def test_select_symbolic_width_must_match_index_operands() -> None:
    """SELECT rejects a declared width different from its flattened index."""

    @qm.qkernel
    def circuit(width: qm.UInt) -> qm.Bit:
        """Use a fixed two-qubit index with a symbolic width declaration."""
        index = qm.qubit_array(2, name="index")
        target = qm.qubit("target")
        index, target = qm.select(
            [_identity_case, _x_case],
            num_index_qubits=width,
        )(index, target)
        return qm.measure(target)

    symbolic = circuit.estimate_resources()
    assert symbolic.substitute(width=2).gates.total == 3
    with pytest.raises(ValueError, match="index operand width must equal 3"):
        symbolic.substitute(width=3)
    with pytest.raises(ValueError, match="index operand width must equal 3"):
        circuit.estimate_resources(inputs={"width": 3})
