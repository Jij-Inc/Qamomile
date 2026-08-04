"""Tests for the symbolic ResourceEstimator compiler service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator import (
    resource_estimator as resource_estimator_module,
)


@qm.qkernel
def _basis_probe(theta: qm.Float) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
    """Apply one exact Toffoli and one approximate axial rotation."""
    left = qm.qubit("left")
    right = qm.qubit("right")
    target = qm.qubit("target")
    left, right, target = qm.ccx(left, right, target)
    target = qm.rx(target, theta)
    return left, right, target


@qm.qkernel
def _allocation_helper(q: qm.Qubit) -> qm.Qubit:
    """Allocate and measure one helper-local qubit."""
    fresh = qm.qubit("fresh")
    qm.measure(fresh)
    return q


@qm.qkernel
def _distinct_helper_callsites() -> tuple[qm.Bit, qm.UInt]:
    """Select two branch-local helper call sites across range iterations."""
    q = qm.qubit("q")
    total = qm.uint(0)
    for index in qm.range(2):
        if index == 0:
            q = _allocation_helper(q)
        else:
            q = _allocation_helper(q)
        total = total + 1
    return qm.measure(q), total


@qm.qkernel
def _repeated_helper_callsite() -> tuple[qm.Bit, qm.UInt]:
    """Replay one helper call site across two range iterations."""
    q = qm.qubit("q")
    total = qm.uint(0)
    for _index in qm.range(2):
        q = _allocation_helper(q)
        total = total + 1
    return qm.measure(q), total


def test_clifford_t_basis_lowers_body_gates_and_reports_metadata() -> None:
    """Basis lowering reports aggregate Clifford+T counts and approximation."""
    logical = _basis_probe.estimate_resources()
    lowered = _basis_probe.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1 / 8,
    )

    assert logical.gates.total == 2
    assert logical.quality is qm.EstimateQuality.EXACT
    assert lowered.gates.total == 26
    assert lowered.gates.single_qubit == 20
    assert lowered.gates.two_qubit == 6
    assert lowered.gates.t == 16
    assert lowered.gates.rotation == 0
    assert lowered.quality is qm.EstimateQuality.UNKNOWN
    assert lowered.approximation is qm.ApproximationStatus.APPROXIMATE


def test_estimation_derives_public_symbol_metadata_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intermediate resource algebra defers full symbol-registry derivation."""

    @qm.qkernel
    def repeated_body() -> qm.Qubit:
        """Apply two gates in each of sixteen static loop iterations."""
        target = qm.qubit("target")
        for _index in qm.range(16):
            target = qm.h(target)
            target = qm.x(target)
        return target

    registry_spy = Mock(wraps=resource_estimator_module._serialization_registry)
    monkeypatch.setattr(
        resource_estimator_module,
        "_serialization_registry",
        registry_spy,
    )

    estimate = repeated_body.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert registry_spy.call_count == 1
    assert estimate.parameters == {}
    assert registry_spy.call_count == 1
    assert estimate.width.peak_qubits == 1
    assert estimate.gates.total == 32
    assert estimate.gates.single_qubit == 32
    assert estimate.depth.depth == 32

    payload = estimate.to_dict()

    assert registry_spy.call_count == 2
    assert payload["parameters"] == {}


def test_deferred_symbol_metadata_resets_after_estimation_error() -> None:
    """A failed estimate cannot defer later manual ResourceEstimate metadata."""

    @qm.qkernel
    def circuit(iterations: qm.UInt) -> qm.Qubit:
        """Apply one gate per symbolic iteration."""
        target = qm.qubit("target")
        for _index in qm.range(iterations):
            target = qm.h(target)
        return target

    with pytest.raises(ValueError, match="neither free symbols"):
        circuit.estimate_resources(inputs={"unknown": 2})

    manual_work = sp.Symbol(
        "manual_work",
        integer=True,
        nonnegative=True,
    )
    stale = sp.Symbol("stale")
    manual = qm.ResourceEstimate(
        gates=qm.GateResources(total=manual_work),
        parameters={"stale": stale},
    )

    assert manual.parameters == {"manual_work": manual_work}


def test_concurrent_estimates_keep_symbol_metadata_scoped() -> None:
    """Concurrent estimators each finalize their own parameter map."""

    @qm.qkernel
    def symbolic_body(iterations: qm.UInt) -> qm.Qubit:
        """Apply one gate per symbolic iteration."""
        target = qm.qubit("target")
        for _index in qm.range(iterations):
            target = qm.h(target)
        return target

    block = symbolic_body.build()
    with ThreadPoolExecutor(max_workers=2) as pool:
        estimates = list(
            pool.map(
                lambda _index: qm.ResourceEstimator(
                    basis=qm.GateBasis.LOGICAL
                ).estimate(block),
                range(2),
            )
        )

    assert all(set(estimate.parameters) == {"iterations"} for estimate in estimates)
    assert all(
        estimate.gates.total == estimate.parameters["iterations"]
        for estimate in estimates
    )


def test_pauli_evolve_depth_tracks_gadget_structure() -> None:
    """Pauli evolution depth grows with term support and basis changes."""
    import qamomile.observable as qm_o

    @qm.qkernel
    def circuit(hamiltonian: qm.Observable) -> qm.Vector[qm.Qubit]:
        """Apply one bound two-term Pauli evolution."""
        qubits = qm.qubit_array(2, "qubits")
        return qm.pauli_evolve(qubits, hamiltonian, qm.float_(0.5))

    estimate = circuit.estimate_resources(
        inputs={"hamiltonian": qm_o.X(0) + qm_o.Z(0) * qm_o.Z(1)}
    )

    assert estimate.gates.total == 6
    assert estimate.depth.depth == 6
    assert estimate.depth.clifford_depth == 4
    assert estimate.depth.rotation_depth == 2


def test_pauli_evolve_parallelizes_basis_changes_within_a_term() -> None:
    """Pauli evolution schedules same-stage basis changes in parallel."""
    import qamomile.observable as qm_o

    @qm.qkernel
    def circuit(hamiltonian: qm.Observable) -> qm.Vector[qm.Qubit]:
        """Apply one bound three-qubit Pauli evolution."""
        qubits = qm.qubit_array(3, "qubits")
        return qm.pauli_evolve(qubits, hamiltonian, qm.float_(0.5))

    estimate = circuit.estimate_resources(
        inputs={"hamiltonian": qm_o.X(0) * qm_o.X(1) * qm_o.X(2)}
    )

    assert estimate.gates.total == 11
    assert estimate.gates.single_qubit == 7
    assert estimate.gates.two_qubit == 4
    assert estimate.depth.depth == 7
    assert estimate.depth.clifford_depth == 6
    assert estimate.depth.rotation_depth == 1


def test_gate_basis_accepts_string_values() -> None:
    """The public basis option accepts notebook-friendly strings safely."""
    estimate = _basis_probe.estimate_resources(basis="logical")
    assert estimate.gates.total == 2


def test_qkernel_wrapper_defers_precision_default_to_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The qkernel facade does not pin a second synthesis-precision default."""
    estimate_resources = Mock(return_value=qm.ResourceEstimate())
    monkeypatch.setattr(
        resource_estimator_module,
        "estimate_resources",
        estimate_resources,
    )

    _basis_probe.estimate_resources()

    assert "precision" not in estimate_resources.call_args.kwargs


def test_unknown_resource_policy_accepts_string_values() -> None:
    """All public estimator entry points normalize resource-policy strings."""
    oracle = qm.opaque("string_policy_oracle", num_qubits=1)

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Invoke one bodyless Oracle whose resources are unknown."""
        (target,) = oracle(qm.qubit("target"))
        return target

    estimator = qm.ResourceEstimator(unknown_policy="opaque_call")
    estimates = (
        estimator.estimate(circuit),
        qm.estimate_resources(circuit, unknown_policy="opaque_call"),
        circuit.estimate_resources(unknown_policy="opaque_call"),
    )

    assert estimator.config.unknown_policy is qm.UnknownResourcePolicy.OPAQUE_CALL
    assert all(
        estimate.calls.calls_by_name == {"string_policy_oracle": 1}
        for estimate in estimates
    )


def test_control_decomposition_accepts_string_values() -> None:
    """The control model accepts stable string identifiers."""
    estimate = _basis_probe.estimate_resources(
        control_decomposition="abstract",
    )

    assert estimate.control_decomposition is qm.ControlDecomposition.ABSTRACT


@pytest.mark.parametrize("policy", ["ignore", ""])
def test_unknown_resource_policy_rejects_unknown_strings(policy: str) -> None:
    """Unknown policy identifiers report every supported stable value."""
    with pytest.raises(
        ValueError,
        match="expected one of: error, opaque_call, zero_with_warning",
    ):
        _basis_probe.estimate_resources(unknown_policy=policy)


@pytest.mark.parametrize("basis", ["unknown", "surface_code", ""])
def test_gate_basis_rejects_unknown_string_values(basis: str) -> None:
    """Unknown and empty strings are not replaced by the default basis."""
    with pytest.raises(
        ValueError,
        match="expected one of: logical, clifford_t",
    ):
        _basis_probe.estimate_resources(basis=basis)


@pytest.mark.parametrize("decomposition", ["ancilla_free", ""])
def test_control_decomposition_rejects_unknown_strings(
    decomposition: str,
) -> None:
    """Unknown control recipes are rejected instead of using the default."""
    with pytest.raises(
        ValueError,
        match="expected one of: abstract, clean_ancilla_toffoli",
    ):
        _basis_probe.estimate_resources(
            control_decomposition=decomposition,
        )


def test_clifford_t_basis_lowers_controlled_toffoli_with_clean_ancilla() -> None:
    """An added control uses the declared clean-ancilla Toffoli ladder."""

    @qm.composite_gate
    def toffoli(
        left: qm.Qubit,
        right: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply one Toffoli gate."""
        return qm.ccx(left, right, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply a Toffoli under one additional control."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        target = qm.qubit("target")
        controlled_toffoli = qm.control(toffoli)
        return controlled_toffoli(control, left, right, target)

    estimate = circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)

    assert estimate.gates.t == 28
    assert estimate.gates.total == 61
    assert estimate.depth.depth == 61
    assert estimate.depth.clifford_depth == 33
    assert estimate.depth.t_depth == 12
    assert estimate.depth.non_clifford_depth == 12
    assert estimate.depth.gate_depth == 61
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.qubits == 6
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize(
    ("name", "controls", "expected"),
    [
        pytest.param(
            "z",
            2,
            (17, 7, 17, 10, 3, 0, qm.EstimateQuality.EXACT),
            id="ccz-exact-boundary",
        ),
        pytest.param(
            "z",
            3,
            (63, 28, 63, 35, 12, 2, qm.EstimateQuality.CONSERVATIVE),
            id="controlled-z-upper-bound",
        ),
        pytest.param(
            "y",
            2,
            (17, 7, 17, 10, 3, 0, qm.EstimateQuality.EXACT),
            id="ccy-exact-boundary",
        ),
        pytest.param(
            "y",
            3,
            (63, 28, 63, 35, 12, 2, qm.EstimateQuality.CONSERVATIVE),
            id="controlled-y-upper-bound",
        ),
        pytest.param(
            "cz",
            1,
            (17, 7, 17, 10, 3, 0, qm.EstimateQuality.EXACT),
            id="ccz-from-cz-exact-boundary",
        ),
        pytest.param(
            "cz",
            2,
            (63, 28, 63, 35, 12, 2, qm.EstimateQuality.CONSERVATIVE),
            id="controlled-cz-upper-bound",
        ),
        pytest.param(
            "swap",
            1,
            (17, 7, 17, 10, 3, 0, qm.EstimateQuality.EXACT),
            id="fredkin-exact-boundary",
        ),
        pytest.param(
            "swap",
            2,
            (63, 28, 63, 35, 12, 2, qm.EstimateQuality.CONSERVATIVE),
            id="controlled-swap-upper-bound",
        ),
        pytest.param(
            "p",
            0,
            (9, 9, 9, 0, 9, 0, qm.EstimateQuality.UNKNOWN),
            id="phase-rotation",
        ),
        pytest.param(
            "p",
            1,
            (29, 27, 20, 2, 18, 0, qm.EstimateQuality.UNKNOWN),
            id="controlled-phase-parallel-rotations",
        ),
        pytest.param(
            "p",
            2,
            (59, 41, 50, 18, 24, 1, qm.EstimateQuality.UNKNOWN),
            id="multi-controlled-phase",
        ),
        pytest.param(
            "cp",
            0,
            (29, 27, 20, 2, 18, 0, qm.EstimateQuality.UNKNOWN),
            id="cp-parallel-rotations",
        ),
        pytest.param(
            "cp",
            1,
            (59, 41, 50, 18, 24, 1, qm.EstimateQuality.UNKNOWN),
            id="controlled-cp",
        ),
        pytest.param(
            "s",
            1,
            (5, 3, 4, 2, 2, 0, qm.EstimateQuality.EXACT),
            id="controlled-s",
        ),
        pytest.param(
            "s",
            2,
            (35, 17, 34, 18, 8, 1, qm.EstimateQuality.CONSERVATIVE),
            id="multi-controlled-s",
        ),
        pytest.param(
            "sdg",
            1,
            (5, 3, 4, 2, 2, 0, qm.EstimateQuality.EXACT),
            id="controlled-sdg",
        ),
        pytest.param(
            "sdg",
            2,
            (35, 17, 34, 18, 8, 1, qm.EstimateQuality.CONSERVATIVE),
            id="multi-controlled-sdg",
        ),
        pytest.param(
            "t",
            1,
            (31, 15, 31, 16, 7, 1, qm.EstimateQuality.EXACT),
            id="controlled-t",
        ),
        pytest.param(
            "t",
            2,
            (61, 29, 61, 32, 13, 2, qm.EstimateQuality.EXACT),
            id="multi-controlled-t",
        ),
        pytest.param(
            "tdg",
            1,
            (31, 15, 31, 16, 7, 1, qm.EstimateQuality.EXACT),
            id="controlled-tdg",
        ),
        pytest.param(
            "tdg",
            2,
            (61, 29, 61, 32, 13, 2, qm.EstimateQuality.EXACT),
            id="multi-controlled-tdg",
        ),
    ],
)
def test_controlled_clifford_t_depth_tracks_canonical_schedule(
    name: str,
    controls: int,
    expected: tuple[int, int, int, int, int, int, qm.EstimateQuality],
) -> None:
    """Named Clifford+T recipes expose their schedule and quality."""
    estimate = resource_estimator_module._estimate_named_gate_in_basis(
        name,
        sp.Integer(controls),
        basis=qm.GateBasis.CLIFFORD_T,
        control_decomposition=qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
        precision=1 / 8,
    )
    total, t_count, depth, clifford_depth, t_depth, ancillas, quality = expected

    assert estimate.gates.total == total
    assert estimate.gates.t == t_count
    assert estimate.depth.depth == depth
    assert estimate.depth.clifford_depth == clifford_depth
    assert estimate.depth.t_depth == t_depth
    assert estimate.depth.non_clifford_depth == t_depth
    assert estimate.depth.gate_depth == depth
    assert estimate.width.clean_ancilla_qubits == ancillas
    assert estimate.quality is quality
    expected_approximation = (
        qm.ApproximationStatus.APPROXIMATE
        if name in {"p", "cp"}
        else qm.ApproximationStatus.EXACT
    )
    assert estimate.approximation is expected_approximation


@pytest.mark.parametrize("name", ["z", "p", "s"])
def test_symbolic_clifford_t_named_depth_matches_direct_specialization(
    name: str,
) -> None:
    """Symbolic control guards retain every recipe and quality branch."""
    controls = sp.Symbol("controls", integer=True, nonnegative=True)
    symbolic = resource_estimator_module._estimate_named_gate_in_basis(
        name,
        controls,
        basis=qm.GateBasis.CLIFFORD_T,
        control_decomposition=qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
        precision=1 / 8,
    )

    for value in range(4):
        specialized = symbolic.substitute(controls=value)
        direct = resource_estimator_module._estimate_named_gate_in_basis(
            name,
            sp.Integer(value),
            basis=qm.GateBasis.CLIFFORD_T,
            control_decomposition=qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
            precision=1 / 8,
        )

        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width
        assert specialized.derivation is direct.derivation
        assert specialized.quality is direct.quality
        assert specialized.approximation is direct.approximation
        assert specialized.parameters == {}


@pytest.mark.parametrize("name", ["y", "z", "s", "sdg"])
def test_uncontrolled_fixed_gate_keeps_single_layer_depth(name: str) -> None:
    """Controlled-depth special cases preserve their uncontrolled branch."""
    gates = resource_estimator_module._classify_clifford_t_gate(
        name,
        sp.Integer(0),
        1 / 8,
    )

    depth = resource_estimator_module._named_clifford_t_depth(
        name,
        sp.Integer(0),
        gates,
        1 / 8,
    )

    assert depth.depth == 1
    assert depth.gate_depth == 1


def test_clifford_t_controlled_s_uses_exact_three_t_phase_polynomial() -> None:
    """Controlled S and Sdg use their ancilla-free exact 3T circuits."""

    @qm.composite_gate
    def one_s(target: qm.Qubit) -> qm.Qubit:
        """Apply one S gate."""
        return qm.s(target)

    @qm.composite_gate
    def one_sdg(target: qm.Qubit) -> qm.Qubit:
        """Apply one Sdg gate."""
        return qm.sdg(target)

    @qm.qkernel
    def s_circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Control one S gate with one qubit."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return qm.control(one_s)(control, target)

    @qm.qkernel
    def sdg_circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Control one Sdg gate with one qubit."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return qm.control(one_sdg)(control, target)

    for circuit in (s_circuit, sdg_circuit):
        estimate = circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)

        assert estimate.gates.t == 3
        assert estimate.gates.total == 5
        assert estimate.depth.t_depth == 2
        assert estimate.width.clean_ancilla_qubits == 0
        assert estimate.quality is qm.EstimateQuality.EXACT


def test_clean_ancilla_mcx_recipe_is_consistent_across_gate_bases() -> None:
    """Logical and Clifford+T MCX use the same four-control ladder."""

    @qm.composite_gate
    def one_x(target: qm.Qubit) -> qm.Qubit:
        """Apply one Pauli-X gate."""
        return qm.x(target)

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply one X gate under four coherent controls."""
        controls = qm.qubit_array(4, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(one_x, num_controls=4)(controls, target)
        return target

    logical = circuit.estimate_resources()
    clifford_t = circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)

    assert logical.gates.total == 7
    assert logical.gates.toffoli == 6
    assert logical.width.clean_ancilla_qubits == 3
    assert logical.quality is qm.EstimateQuality.CONSERVATIVE
    assert clifford_t.gates.total == 91
    assert clifford_t.gates.t == 42
    assert clifford_t.depth.t_depth == 18
    assert clifford_t.width.clean_ancilla_qubits == 3
    assert clifford_t.quality is qm.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize(
    ("basis", "control_decomposition", "should_fail"),
    [
        pytest.param(
            qm.GateBasis.LOGICAL,
            qm.ControlDecomposition.ABSTRACT,
            False,
            id="logical-abstract",
        ),
        pytest.param(
            qm.GateBasis.LOGICAL,
            qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
            False,
            id="logical-clean-ancilla",
        ),
        pytest.param(
            qm.GateBasis.CLIFFORD_T,
            qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
            False,
            id="clifford-t-clean-ancilla",
        ),
        pytest.param(
            qm.GateBasis.CLIFFORD_T,
            qm.ControlDecomposition.ABSTRACT,
            True,
            id="clifford-t-abstract",
        ),
    ],
)
def test_gate_basis_and_control_decomposition_cross_product(
    basis: qm.GateBasis,
    control_decomposition: qm.ControlDecomposition,
    should_fail: bool,
) -> None:
    """Basis and control axes compose except for an undecomposable abstract control."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one controlled X."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return qm.control(qm.x)(control, target)

    if should_fail:
        with pytest.raises(ValueError, match="cannot preserve a controlled primitive"):
            circuit.estimate_resources(
                basis=basis,
                control_decomposition=control_decomposition,
            )
        return

    estimate = circuit.estimate_resources(
        basis=basis,
        control_decomposition=control_decomposition,
    )

    assert estimate.basis is basis
    assert estimate.control_decomposition is control_decomposition
    assert estimate.gates.total == 1
    assert estimate.gates.two_qubit == 1
    assert estimate.gates.clifford == 1
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_clifford_t_basis_rejects_missing_controlled_gate_lowering() -> None:
    """Unsupported controlled gates fail instead of reporting guessed counts."""

    @qm.composite_gate
    def hadamard(target: qm.Qubit) -> qm.Qubit:
        """Apply one Hadamard gate."""
        return qm.h(target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply a controlled Hadamard through a body-backed callable."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        controlled_hadamard = qm.control(hadamard)
        return controlled_hadamard(control, target)

    with pytest.raises(ValueError, match="controlled gate 'h'"):
        circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)


def test_clifford_t_basis_rejects_controlled_rotation_lowering() -> None:
    """Controlled rotations fail when no decomposition contract is defined."""

    @qm.composite_gate
    def rotate(target: qm.Qubit, theta: qm.Float) -> qm.Qubit:
        """Apply one arbitrary Y rotation."""
        return qm.ry(target, theta)

    @qm.qkernel
    def circuit(theta: qm.Float) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply an arbitrary rotation under one control."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        controlled_rotate = qm.control(rotate)
        return controlled_rotate(control, target, theta)

    with pytest.raises(ValueError, match="controlled gate 'ry'"):
        circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)


@qm.qkernel
def _dependency_depth_probe() -> tuple[qm.Qubit, qm.Qubit]:
    """Apply two independent first-layer gates and one dependent gate."""
    left = qm.qubit("left")
    right = qm.qubit("right")
    left = qm.h(left)
    right = qm.x(right)
    left = qm.z(left)
    return left, right


def test_depth_uses_quantum_wire_dependencies() -> None:
    """Independent gates share a layer while dependent gates serialize."""
    estimate = _dependency_depth_probe.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 2
    assert estimate.depth.clifford_depth == 2


@qm.qkernel
def _released_qubit_probe() -> qm.Qubit:
    """Measure one allocation before creating a replacement qubit."""
    first = qm.qubit("first")
    _bit = qm.measure(first)
    second = qm.qubit("second")
    return second


def test_width_reuses_affinely_released_qubits() -> None:
    """Measurement ends a wire lifetime before a later allocation."""
    estimate = _released_qubit_probe.estimate_resources()

    assert estimate.width.allocated_qubits == 2
    assert estimate.qubits == 1
    assert estimate.circuit_qubits == 2


def test_loop_body_release_retains_unconsumed_local_allocation() -> None:
    """A loop releases a capture but retains its unconsumed local allocation."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Measure one capture before loop-local and later allocations."""
        target = qm.qubit("target")
        for _index in qm.range(1):
            qm.measure(target)
            qm.qubit("fresh")
        return qm.qubit("later")

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 2
    assert estimate.width.circuit_qubits == 3


def test_control_flow_counts_body_local_array_owners_used_through_elements() -> None:
    """Element views keep their body-local root allocations in every width."""

    @qm.qkernel
    def loop_circuit() -> qm.Qubit:
        """Use one element of a loop-local array while retaining one qubit."""
        retained = qm.qubit("retained")
        for _index in qm.range(1):
            work = qm.qubit_array(2, "work")
            work[0] = qm.h(work[0])
        return retained

    @qm.qkernel
    def sliced_loop_circuit() -> qm.Qubit:
        """Use a local array element through a nested slice view."""
        retained = qm.qubit("retained")
        for _index in qm.range(1):
            work = qm.qubit_array(2, "work")
            view = work[0:1]
            view[0] = qm.h(view[0])
        return retained

    @qm.qkernel
    def items_circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Use one element of an items-loop-local array through a slice."""
        retained = qm.qubit("retained")
        for _index, _value in qm.items(data):
            work = qm.qubit_array(2, "work")
            view = work[0:1]
            view[0] = qm.h(view[0])
        return retained

    @qm.qkernel
    def compile_time_if_circuit(flag: qm.UInt) -> qm.Qubit:
        """Use one element of a compile-time branch-local array."""
        retained = qm.qubit("retained")
        if flag:
            work = qm.qubit_array(2, "work")
            work[0] = qm.h(work[0])
        return retained

    @qm.qkernel
    def runtime_if_circuit() -> qm.Qubit:
        """Use one element of a runtime branch-local array."""
        retained = qm.qubit("retained")
        predicate = qm.measure(qm.qubit("predicate"))
        if predicate:
            work = qm.qubit_array(2, "work")
            work[0] = qm.h(work[0])
        return retained

    @qm.qkernel
    def while_circuit() -> qm.Qubit:
        """Use one element of a while-local array for one modeled trip."""
        retained = qm.qubit("retained")
        predicate = qm.measure(qm.qubit("predicate"))
        while predicate:
            work = qm.qubit_array(2, "work")
            work[0] = qm.h(work[0])
            predicate = qm.measure(work[0])
        return retained

    loop = loop_circuit.estimate_resources()
    sliced_loop = sliced_loop_circuit.estimate_resources()
    items_loop = items_circuit.estimate_resources(inputs={"data": {0: 0.1}})
    compile_time_if = compile_time_if_circuit.estimate_resources(inputs={"flag": 1})
    runtime_if = runtime_if_circuit.estimate_resources()
    modeled_while = while_circuit.estimate_resources().substitute(**{"|while|": 1})

    assert loop.width.allocated_qubits == 3
    assert loop.width.peak_qubits == 3
    assert loop.width.circuit_qubits == 3
    assert sliced_loop.width.allocated_qubits == 3
    assert sliced_loop.width.peak_qubits == 3
    assert sliced_loop.width.circuit_qubits == 3
    assert items_loop.width.allocated_qubits == 3
    assert items_loop.width.peak_qubits == 3
    assert items_loop.width.circuit_qubits == 3
    assert compile_time_if.width.allocated_qubits == 3
    assert compile_time_if.width.peak_qubits == 3
    assert compile_time_if.width.circuit_qubits == 3
    assert runtime_if.width.allocated_qubits == 4
    assert runtime_if.width.peak_qubits == 3
    assert runtime_if.width.circuit_qubits == 4
    assert modeled_while.width.allocated_qubits == 4
    assert modeled_while.width.peak_qubits == 3
    assert modeled_while.width.circuit_qubits == 4


def test_invoke_fresh_outputs_extend_caller_liveness() -> None:
    """Fresh outputs from nested qkernels remain live in the caller."""

    @qm.qkernel
    def allocate() -> qm.Qubit:
        """Return one newly allocated qubit."""
        return qm.qubit("fresh")

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Keep fresh results from two distinct call sites live together."""
        left = allocate()
        right = allocate()
        return left, right

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 2
    assert estimate.width.peak_qubits == 2
    assert estimate.width.circuit_qubits == 2


def test_invoke_fresh_array_elements_retain_their_combined_width() -> None:
    """Separate returned elements keep their shared fresh array live."""

    @qm.qkernel
    def allocate_pair() -> tuple[qm.Qubit, qm.Qubit]:
        """Return both elements of one fresh two-qubit array."""
        pair = qm.qubit_array(2, "pair")
        return pair[0], pair[1]

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Keep a returned pair live while allocating one more qubit."""
        left, right = allocate_pair()
        extra = qm.qubit("extra")
        return left, right, extra

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 3
    assert estimate.width.circuit_qubits == 3


def test_invoke_replacement_output_releases_consumed_input() -> None:
    """A nested qkernel can replace a consumed input without inflating peak."""

    @qm.qkernel
    def replace(target: qm.Qubit) -> qm.Qubit:
        """Measure the input before returning one new qubit."""
        qm.measure(target)
        return qm.qubit("fresh")

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Replace one caller allocation through a nested invocation."""
        initial = qm.qubit("initial")
        return replace(initial)

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 2
    assert estimate.width.peak_qubits == 1
    assert estimate.width.circuit_qubits == 2


def test_invoke_partial_array_replacement_keeps_sibling_live() -> None:
    """Consuming one array element does not release its live sibling."""

    @qm.qkernel
    def replace(target: qm.Qubit) -> qm.Qubit:
        """Measure one input element and return a fresh scalar."""
        qm.measure(target)
        return qm.qubit("fresh")

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Keep a sibling and replacement live with a later allocation."""
        pair = qm.qubit_array(2, "pair")
        sibling = pair[1]
        fresh = replace(pair[0])
        extra = qm.qubit("extra")
        return sibling, fresh, extra

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 3
    assert estimate.width.circuit_qubits == 4


def test_runtime_branches_union_static_allocations_but_reuse_peak() -> None:
    """Runtime branch-local results reserve sites without overlapping live."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Choose one fresh result after measuring the branch predicate."""
        predicate = qm.measure(qm.qubit("predicate"))
        if predicate:
            result = qm.qubit("true_result")
        else:
            result = qm.qubit("false_result")
        return result

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 1
    assert estimate.width.circuit_qubits == 3


def test_compile_time_branch_substitution_prunes_dead_allocations() -> None:
    """Post-hoc UInt substitution matches direct branch specialization."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Qubit:
        """Allocate differently sized results in two compile-time branches."""
        if flag:
            result = qm.qubit("true_result")
        else:
            result = qm.qubit_array(2, "false_result")[0]
        return result

    symbolic = circuit.estimate_resources()
    for flag in (0, 1):
        substituted = symbolic.substitute(flag=flag)
        direct = circuit.estimate_resources(inputs={"flag": flag})
        assert substituted.width == direct.width


def test_branch_merge_retains_the_selected_quantum_array_width() -> None:
    """Merged arrays use the selected branch width in outer liveness."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Bit:
        """Allocate, merge, and consume differently sized branch arrays."""
        result = qm.qubit("result")
        if flag:
            work = qm.qubit_array(10, "large")
        else:
            work = qm.qubit_array(2, "small")
        qm.measure(work)
        return qm.measure(result)

    symbolic = circuit.estimate_resources()
    expected_widths = {0: 3, 1: 11}
    for flag, width in expected_widths.items():
        substituted = symbolic.substitute(flag=flag)
        direct = circuit.estimate_resources(inputs={"flag": flag})
        assert substituted.width == direct.width
        assert direct.width.allocated_qubits == width
        assert direct.width.peak_qubits == width
        assert direct.width.circuit_qubits == width


def test_invoke_retains_branch_merged_quantum_array_width() -> None:
    """A body-backed call publishes its selected merged result width."""

    @qm.qkernel
    def allocate(flag: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Return a differently sized array from each symbolic branch."""
        if flag:
            result = qm.qubit_array(10, "large")
        else:
            result = qm.qubit_array(2, "small")
        return result

    @qm.qkernel
    def circuit(flag: qm.UInt) -> qm.Bit:
        """Consume the returned array while one independent qubit stays live."""
        result = qm.qubit("result")
        work = allocate(flag)
        qm.measure(work)
        return qm.measure(result)

    symbolic = circuit.estimate_resources()
    for flag, width in {0: 3, 1: 11}.items():
        substituted = symbolic.substitute(flag=flag)
        direct = circuit.estimate_resources(inputs={"flag": flag})
        assert substituted.width == direct.width
        assert direct.width.allocated_qubits == width
        assert direct.width.peak_qubits == width
        assert direct.width.circuit_qubits == width


def test_if_without_quantum_outputs_releases_captured_inputs() -> None:
    """An empty branch output summary still releases consumed outer wires."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Consume a captured wire in either branch before later allocation."""
        target = qm.qubit("target")
        predicate = qm.measure(qm.qubit("predicate"))
        if predicate:
            qm.measure(target)
            qm.measure(qm.qubit("true_fresh"))
        else:
            qm.measure(target)
            qm.measure(qm.qubit("false_fresh"))
        return qm.qubit("later")

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 5
    assert estimate.width.peak_qubits == 2


def test_symbolic_loop_maximizes_control_ancillas_without_leaking_index() -> None:
    """Loop-dependent control width stays reusable and externally symbolic."""

    @qm.qkernel
    def controlled_leaf(target: qm.Qubit) -> qm.Qubit:
        """Apply one X gate to the controlled target."""
        return qm.x(target)

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Vector[qm.Bit]:
        """Increase control arity across a symbolic range."""
        qubits = qm.qubit_array(n, "qubits")
        for target_index in qm.range(1, n):
            controls, target = qm.control(
                controlled_leaf,
                num_controls=target_index,
            )(
                qubits[0:target_index],
                qubits[target_index],
            )
            qubits[0:target_index] = controls
            qubits[target_index] = target
        return qm.measure(qubits)

    symbolic = circuit.estimate_resources()
    metric_expressions = [
        value
        for resources in (symbolic.width, symbolic.gates, symbolic.depth)
        for value in vars(resources).values()
    ]
    metric_expressions.extend(symbolic.calls.calls_by_name.values())
    metric_expressions.extend(symbolic.calls.queries_by_name.values())

    assert set(symbolic.parameters) == {"n"}
    assert all(
        symbol.name == "n"
        for expression in metric_expressions
        for symbol in expression.free_symbols
    )
    for requirement in symbolic.to_dict()["requirements"]:
        if "target_index" in requirement["expression"]:
            assert any(
                loop_range["symbol"] == "target_index"
                for loop_range in requirement["ranges"]
            )

    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (2, 6),
        5: (3, 8),
    }
    for n, (clean_ancillas, peak_qubits) in expected.items():
        concrete = symbolic.substitute(n=n)
        assert concrete.width.clean_ancilla_qubits == clean_ancillas
        assert concrete.width.peak_qubits == peak_qubits
        assert all(
            not expression.free_symbols
            for resources in (concrete.width, concrete.gates, concrete.depth)
            for expression in vars(resources).values()
        )
    assert symbolic.substitute(n=4).depth.depth == 8


def test_symbolic_loop_width_finds_piecewise_condition_boundaries() -> None:
    """A downward Piecewise jump is maximized at its reachable boundary."""
    index = sp.Symbol("index", integer=True, nonnegative=True)
    iterations = sp.Symbol("iterations", integer=True, nonnegative=True)
    per_iteration = sp.Piecewise(
        (index + 10, index < 3),
        (index, True),
    )
    body = qm.ResourceEstimate(
        width=qm.WidthResources(
            clean_ancilla_qubits=per_iteration,
            peak_qubits=per_iteration,
        )
    )

    symbolic = body.sum_over(index, sp.Integer(0), iterations)
    concrete = symbolic.substitute(iterations=5)

    assert index not in symbolic.width.peak_qubits.free_symbols
    assert concrete.width.clean_ancilla_qubits == 12
    assert concrete.width.peak_qubits == 12
    assert concrete.quality is qm.EstimateQuality.EXACT


def test_sum_over_reduces_loop_guarded_metadata_by_reachable_iterations() -> None:
    """Loop metadata remains active only when some iteration reaches its guard."""
    index = sp.Symbol("index", integer=True, nonnegative=True)
    assumption = qm.ResourceAssumption("guarded loop body")
    active_body = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        assumptions=(assumption,),
        derivation=qm.EstimateDerivation.MODELED,
        approximation=qm.ApproximationStatus.APPROXIMATE,
    )
    empty = qm.ResourceEstimate.zero()

    never = active_body.conditional(empty, sp.Gt(index, 10)).sum_over(
        index,
        sp.Integer(0),
        sp.Integer(2),
    )
    sometimes = active_body.conditional(empty, sp.Eq(index, 1)).sum_over(
        index,
        sp.Integer(0),
        sp.Integer(2),
    )
    iterations = sp.Symbol("iterations", integer=True, nonnegative=True)
    symbolic = active_body.conditional(empty, sp.Eq(index, 1)).sum_over(
        index,
        sp.Integer(0),
        iterations,
    )

    assert never.gates.total == 0
    assert never.assumptions == ()
    assert never.quality is qm.EstimateQuality.EXACT
    assert never.approximation is qm.ApproximationStatus.EXACT

    assert sometimes.gates.total == 1
    assert sometimes.assumptions == (assumption,)
    assert sometimes.derivation is qm.EstimateDerivation.MODELED
    assert sometimes.approximation is qm.ApproximationStatus.APPROXIMATE

    before_guard = symbolic.substitute(iterations=1)
    after_guard = symbolic.substitute(iterations=2)
    assert before_guard.assumptions == ()
    assert before_guard.quality is qm.EstimateQuality.EXACT
    assert before_guard.approximation is qm.ApproximationStatus.EXACT
    assert after_guard.assumptions == (assumption,)
    assert after_guard.derivation is qm.EstimateDerivation.MODELED
    assert after_guard.approximation is qm.ApproximationStatus.APPROXIMATE


def test_trace_is_opt_in_and_honors_the_flag() -> None:
    """Default estimates stay compact while trace=True retains explanations."""

    @qm.qkernel
    def circuit() -> qm.Bit:
        """Apply and measure one H gate."""
        return qm.measure(qm.h(qm.qubit("q")))

    assert circuit.estimate_resources().trace is None
    assert circuit.estimate_resources(trace=False).trace is None
    traced = circuit.estimate_resources(trace=True)
    assert traced.trace is not None
    assert "h" in traced.trace.render()


def test_composite_resources_follow_the_executable_body() -> None:
    """ResourceEstimator derives a named callable's cost from its body."""

    @qm.composite_gate
    @qm.qkernel
    def body(q: qm.Qubit) -> qm.Qubit:
        """Apply two gates."""
        return qm.x(qm.h(q))

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Call the body-backed composite."""
        q = qm.qubit("q")
        q = body(q)
        return q

    estimate = qm.ResourceEstimator().estimate(circuit)
    assert estimate.gates.total == 2


def test_independent_body_backed_calls_share_a_depth_layer() -> None:
    """Known unitary qkernel calls schedule by their caller wire footprint."""

    @qm.qkernel
    def leaf(target: qm.Qubit) -> qm.Qubit:
        """Apply one Hadamard gate."""
        return qm.h(target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Invoke the same body on two independent qubits."""
        left = leaf(qm.qubit("left"))
        right = leaf(qm.qubit("right"))
        return left, right

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 1
    assert estimate.depth.clifford_depth == 1


def test_zero_depth_body_call_does_not_serialize_independent_gates() -> None:
    """An identity call on one wire does not delay work on another wire."""

    @qm.qkernel
    def identity(target: qm.Qubit) -> qm.Qubit:
        """Return a target unchanged."""
        return target

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Place an identity call between independent Hadamard gates."""
        left = qm.h(qm.qubit("left"))
        left = identity(left)
        right = qm.h(qm.qubit("right"))
        return left, right

    assert circuit.estimate_resources().depth.depth == 1


def test_opaque_fixed_cost_counts_calls_and_gates() -> None:
    """Opaque callables may carry an explicit fixed cost."""
    oracle_estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=7, t=7, non_clifford=7),
        calls=qm.CallResources(
            calls_by_name={"phase_oracle": 1},
            queries_by_name={"phase_oracle": 1},
        ),
    )
    oracle = qm.opaque(
        "phase_oracle",
        num_qubits=1,
        cost=oracle_estimate,
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Call an opaque oracle once."""
        q = qm.qubit("q")
        (q,) = oracle(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 7
    assert estimate.gates.t == 7
    assert estimate.calls.calls_by_name["phase_oracle"] == 1
    assert estimate.calls.queries_by_name["phase_oracle"] == 1


def test_symbolic_loop_with_loop_dependent_cost_is_summed() -> None:
    """Loop-dependent resource expressions are summed symbolically."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Apply exponentially many gates through explicit loop structure."""
        q = qm.qubit("q")
        for i in qm.range(n):
            for _ in qm.range(2**i):
                q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()
    n = estimate.parameters["n"]

    assert sp.simplify(estimate.gates.total - (2**n - 1)) == 0


def test_zero_trip_offset_loop_has_zero_body_resources() -> None:
    """A UInt value of zero preserves Python's empty-range semantics."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Allocate work qubits only while an offset range executes."""
        q = qm.qubit("q")
        for _ in qm.range(1, n):
            _work = qm.qubit_array(3, "work")
            q = qm.h(q)
        return q

    symbolic = circuit.estimate_resources()
    n = symbolic.parameters["n"]

    assert n.is_nonnegative is True
    assert n.is_positive is None
    assert symbolic.gates.total == sp.Max(0, n - 1)
    assert symbolic.substitute(n=0).gates.total == 0
    assert symbolic.substitute(n=0).qubits == 1
    assert circuit.estimate_resources(inputs={"n": 0}).gates.total == 0
    assert circuit.estimate_resources(inputs={"n": 1}).qubits == 1
    with pytest.raises(ValueError, match="negative"):
        symbolic.substitute(n=-1)
    with pytest.raises(ValueError, match="negative"):
        circuit.estimate_resources(inputs={"n": -1})


def test_direct_zero_trip_loop_disables_reusable_width() -> None:
    """A direct range(n) body contributes no workspace when UInt n is zero."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Allocate temporary workspace only when the loop executes."""
        q = qm.qubit("q")
        for _ in qm.range(n):
            _work = qm.qubit_array(3, "work")
            q = qm.h(q)
        return q

    symbolic = circuit.estimate_resources()
    n = symbolic.parameters["n"]

    assert n.is_nonnegative is True
    assert symbolic.substitute(n=0).width.allocated_qubits == 1
    assert symbolic.substitute(n=0).qubits == 1
    assert circuit.estimate_resources(inputs={"n": 0}).qubits == 1


def test_uint_zero_branch_remains_symbolically_reachable() -> None:
    """A UInt symbol does not simplify its equality-to-zero branch away."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Use different gate counts on the zero and nonzero branches."""
        q = qm.qubit("q")
        if n == 0:
            q = qm.x(q)
        else:
            q = qm.h(q)
            q = qm.z(q)
        return q

    symbolic = circuit.estimate_resources()
    n = symbolic.parameters["n"]

    assert symbolic.gates.total == sp.Piecewise((1, sp.Eq(n, 0)), (2, True))
    assert symbolic.substitute(n=0).gates.total == 1
    assert symbolic.substitute(n=1).gates.total == 2


def test_float_parameter_remains_real_in_symbolic_branches() -> None:
    """A negative Float input can select a branch after symbolic estimation."""

    @qm.qkernel
    def circuit(theta: qm.Float) -> qm.Qubit:
        """Use a real-valued parameter as a compile-time predicate."""
        q = qm.qubit("q")
        if theta > 0:
            q = qm.x(q)
        else:
            q = qm.h(q)
            q = qm.z(q)
        return q

    symbolic = circuit.estimate_resources()

    assert symbolic.parameters["theta"].is_real is True
    assert symbolic.parameters["theta"].is_integer is None
    assert circuit.estimate_resources(inputs={"theta": -0.5}).gates.total == 2


def test_float_region_carry_fallback_remains_real() -> None:
    """An unsupported Float recurrence never becomes an integer condition."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Branch on a nonlinear Float carry after a symbolic loop."""
        total = qm.float_(0.5)
        for _ in qm.range(n):
            total = total * total
        q = qm.qubit("q")
        if total == 0.25:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()

    # The nonlinear recurrence is intentionally modeled symbolically. Its
    # Float-typed result must keep the equality undecidable, so the estimator
    # preserves the branch instead of simplifying a non-integer equality
    # against an incorrectly integer-typed symbol.
    fallback = estimate.parameters["total_after_loop"]
    assert fallback.is_real is True
    assert fallback.is_integer is None
    assert estimate.gates.total == sp.Piecewise(
        (
            3,
            sp.And(
                sp.Ne(estimate.parameters["n"], 0),
                sp.Eq(fallback, sp.Float(0.25)),
            ),
        ),
        (1, True),
    )
    assert estimate.substitute(n=1, total_after_loop=0.25).gates.total == 3
    zero_trip = estimate.substitute(n=0)
    assert zero_trip.gates.total == 1
    assert not any(
        "recurrence could not be reduced" in assumption.message
        for assumption in zero_trip.assumptions
    )
    assert any(
        "recurrence could not be reduced" in a.message for a in estimate.assumptions
    )


def test_large_affine_multiplicative_carry_stays_compact() -> None:
    """A geometric carry may drive a range beyond Python's ssize limit."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> tuple[qm.Qubit, qm.UInt]:
        """Double a carry, then use its final value as a gate-loop bound."""
        total = qm.uint(1)
        target = qm.qubit("target")
        for _index in qm.range(repetitions):
            total = total * 2
        for _index in qm.range(total):
            target = qm.x(target)
        return target, total

    symbolic = circuit.estimate_resources()
    repetitions = symbolic.parameters["repetitions"]

    assert sp.simplify(symbolic.gates.total - 2**repetitions) == 0
    assert symbolic.substitute(repetitions=63).gates.total == 2**63
    assert circuit.estimate_resources(inputs={"repetitions": 63}).gates.total == 2**63


def test_large_coupled_region_carry_replays_exactly() -> None:
    """A concrete unsupported carry retains exact values beyond the old limit."""

    @qm.qkernel
    def circuit(
        repetitions: qm.UInt,
    ) -> tuple[qm.Qubit, qm.UInt, qm.UInt]:
        """Apply one gate beside two coupled loop-carried counters."""
        first = qm.uint(0)
        second = qm.uint(1)
        target = qm.qubit("target")
        for _index in qm.range(repetitions):
            target = qm.x(target)
            first = first + second
            second = second + 1
        return target, first, second

    estimate = circuit.estimate_resources(inputs={"repetitions": 65})

    assert estimate.gates.total == 65
    assert estimate.parameters == {}


def test_large_nonlinear_carry_used_after_loop_replays_exactly() -> None:
    """A later controlled power sees the exact final concrete carry."""

    @qm.qkernel
    def one_h(target: qm.Qubit) -> qm.Qubit:
        """Apply one Hadamard gate."""
        return qm.h(target)

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Bit:
        """Use a nonlinear loop carry as a controlled-call power."""
        count = qm.uint(2)
        for index in qm.range(repetitions):
            count = count * index + 1
        control = qm.qubit("control")
        target = qm.qubit("target")
        control, target = qm.control(one_h)(
            control,
            target,
            power=count,
        )
        return qm.measure(target)

    estimate = circuit.estimate_resources(inputs={"repetitions": 65})
    expected_power = 2
    for index in range(65):
        expected_power = expected_power * index + 1

    assert estimate.gates.total == expected_power
    assert estimate.parameters == {}


def test_float_runtime_if_merge_keeps_reachable_resource_branch() -> None:
    """An undecidable Float merge never collapses a reachable comparison."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Branch on a Float selected by a measurement-backed conditional."""
        predicate = qm.qubit("predicate")
        measured = qm.measure(predicate)
        value = qm.float_(0.5)
        if measured:
            value = qm.float_(0.25)

        q = qm.qubit("q")
        if value == 0.25:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3


def test_same_named_runtime_if_merges_remain_independent() -> None:
    """Independent measurement merges never collapse by display name."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Compare two independently selected but same-named UInt merges."""
        left_predicate = qm.measure(qm.qubit("left_predicate"))
        right_predicate = qm.measure(qm.qubit("right_predicate"))

        left = qm.uint(0)
        if left_predicate:
            left = qm.uint(1)
        right = qm.uint(0)
        if right_predicate:
            right = qm.uint(1)

        q = qm.qubit("q")
        if left != right:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3


def test_runtime_if_uint_merge_symbol_keeps_ir_domain() -> None:
    """An undecidable UInt merge remains a nonnegative integer."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a runtime-selected UInt as a later resource loop bound."""
        predicate = qm.qubit("predicate")
        measured = qm.measure(predicate)

        count = qm.uint(0)
        if measured:
            count = qm.uint(2)

        q = qm.qubit("q")
        for _ in qm.range(count):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()
    (count_symbol,) = estimate.parameters.values()

    assert set(estimate.parameters) == {"uint_const_merge_0"}
    assert count_symbol.is_integer is True
    assert count_symbol.is_nonnegative is True
    assert estimate.substitute(uint_const_merge_0=2).gates.total == 2


def test_measurement_provenance_crosses_qkernel_call_boundary() -> None:
    """A helper's runtime merge stays fresh instead of nesting Piecewise."""

    @qm.qkernel
    def select_count(measured: qm.Bit) -> qm.UInt:
        """Select a loop count from a measurement-backed condition."""
        count = qm.uint(0)
        if measured:
            count = qm.uint(2)
        return count

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a helper-selected runtime value as a resource loop bound."""
        predicate = qm.measure(qm.qubit("predicate"))
        count = select_count(predicate)
        target = qm.qubit("target")
        for _ in qm.range(count):
            target = qm.h(target)
        return target

    estimate = circuit.estimate_resources()
    (count_symbol,) = estimate.parameters.values()

    assert not estimate.gates.total.has(sp.Piecewise)
    assert estimate.gates.total == count_symbol
    assert count_symbol.is_integer is True
    assert count_symbol.is_nonnegative is True


def test_runtime_if_bit_merge_symbol_keeps_ir_domain() -> None:
    """An undecidable Bit merge remains a nonnegative integer."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceInterpreter,
        _ResourceEstimatorConfig,
        build_if_scopes,
    )
    from qamomile.circuit.ir.operation.control_flow import IfOperation
    from qamomile.circuit.ir.types.primitives import BitType
    from qamomile.circuit.ir.value import Value

    condition = Value(type=BitType(), name="condition")
    true_value = Value(type=BitType(), name="true").with_const(True)
    false_value = Value(type=BitType(), name="false").with_const(False)
    result = Value(type=BitType(), name="merged")
    operation = IfOperation(
        operands=[condition],
        true_operations=[],
        false_operations=[],
    )
    operation.add_merge(true_value, false_value, result)

    resolver = ExprResolver()
    true_resolver, false_resolver = build_if_scopes(operation, resolver)
    interpreter = ResourceInterpreter(
        config=_ResourceEstimatorConfig(),
        bindings={},
    )
    interpreter._publish_if_results(
        operation,
        resolver,
        true_resolver,
        false_resolver,
        taken=None,
    )
    merged_symbol = resolver.resolve(result)

    assert merged_symbol.is_integer is True
    assert merged_symbol.is_nonnegative is True


def test_condition_values_never_capture_identity_fresh_fallbacks() -> None:
    """Public condition inputs specialize symbols but not same-named dummies."""
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceInterpreter,
        _ResourceEstimatorConfig,
    )

    public = sp.Symbol("flag", integer=True, nonnegative=True)
    internal = sp.Dummy("flag", integer=True, nonnegative=True)
    interpreter = ResourceInterpreter(
        config=_ResourceEstimatorConfig(),
        bindings={},
        condition_values={"flag": sp.Integer(1)},
    )

    specialized = interpreter._apply_condition_values(public + internal)
    decision, _note = interpreter._decide_branch(sp.Eq(internal, 1))

    assert public not in specialized.free_symbols
    assert internal in specialized.free_symbols
    assert decision is None


def test_loop_index_domain_preserves_zero_and_negative_branches() -> None:
    """Loop indices are integers rather than assumed-positive parameters."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Take the false branch for indices minus one and zero."""
        q = qm.qubit("q")
        for index in qm.range(-1, 1):
            if index > 0:
                q = qm.x(q)
            else:
                q = qm.h(q)
                q = qm.z(q)
        return q

    assert circuit.estimate_resources().gates.total == 4


def test_nested_shadowed_loop_names_keep_value_identity() -> None:
    """A saved outer index remains distinct from a shadowing inner index."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Qubit:
        """Apply outer-index times inner-index gates per index pair."""
        q = qm.qubit("q")
        for index in qm.range(n):
            outer_index = index
            for index in qm.range(n):
                for _ in qm.range(outer_index * index):
                    q = qm.h(q)
        return q

    # (0 + 1 + 2) * (0 + 1 + 2) = 9.
    assert circuit.estimate_resources(inputs={"n": 3}).gates.total == 9


def test_for_items_uses_concrete_entries_and_rejects_symbolic_dependency() -> None:
    """Entry-dependent resources are exact when bound and fail closed otherwise."""

    @qm.qkernel
    def circuit(sizes: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Apply one gate for every unit encoded by each dictionary key."""
        q = qm.qubit("q")
        for size, _ in qm.items(sizes):
            for _ in qm.range(size):
                q = qm.h(q)
        return q

    concrete = circuit.estimate_resources(inputs={"sizes": {1: 0.1, 3: 0.2}})

    assert concrete.gates.total == 4
    assert concrete.parameters == {}
    with pytest.raises(NotImplementedError, match="current item key or value"):
        circuit.estimate_resources()


def test_concrete_for_items_schedules_disjoint_entries_in_parallel() -> None:
    """Concrete dictionary entries on distinct wires share one depth layer."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Vector[qm.Qubit]:
        """Rotate the register slot selected by each dictionary key."""
        register = qm.qubit_array(3, "register")
        for index, _value in qm.items(data):
            register[index] = qm.rz(register[index], qm.float_(0.25))
        return register

    estimate = circuit.estimate_resources(inputs={"data": {0: 0.1, 1: 0.2, 2: 0.3}})

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 1
    assert estimate.depth.rotation_depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_concrete_for_items_serializes_overlapping_entries() -> None:
    """Concrete dictionary entries on one wire retain sequential depth."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Vector[qm.Qubit]:
        """Rotate one fixed register slot for every dictionary entry."""
        register = qm.qubit_array(3, "register")
        for _index, _value in qm.items(data):
            register[0] = qm.rz(register[0], qm.float_(0.25))
        return register

    estimate = circuit.estimate_resources(inputs={"data": {0: 0.1, 1: 0.2, 2: 0.3}})

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 3
    assert estimate.depth.rotation_depth == 3
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_concrete_for_items_propagates_each_wire_completion() -> None:
    """A gate after an items loop waits only for its own entry dependency."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Vector[qm.Qubit]:
        """Rotate disjoint entries, then gate one previously used slot."""
        register = qm.qubit_array(3, "register")
        for index, _value in qm.items(data):
            register[index] = qm.rz(register[index], qm.float_(0.25))
        register[1] = qm.x(register[1])
        return register

    estimate = circuit.estimate_resources(inputs={"data": {0: 0.1, 1: 0.2, 2: 0.3}})

    assert estimate.gates.total == 4
    assert estimate.depth.depth == 2
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_for_items_rejects_item_dependent_structural_constraints() -> None:
    """An unbound item key cannot leak through an array-index requirement."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Vector[qm.Qubit]:
        """Use each dictionary key as a quantum-register index."""
        register = qm.qubit_array(2, "register")
        for index, _value in qm.items(data):
            register[index] = qm.h(register[index])
        return register

    with pytest.raises(NotImplementedError, match="current item key or value"):
        circuit.estimate_resources()

    concrete = circuit.estimate_resources(inputs={"data": {0: 0.1, 1: 0.2}})
    assert concrete.gates.total == 2
    assert concrete.parameters == {}

    with pytest.raises(ValueError, match="element axis 0 in-bounds margin"):
        circuit.estimate_resources(inputs={"data": {2: 0.1}})


def test_for_items_default_entries_are_evaluated_exactly() -> None:
    """A concrete dictionary default does not leak cardinality or item symbols."""

    @qm.qkernel
    def circuit(
        sizes: qm.Dict[qm.UInt, qm.Float] = {1: 0.1, 3: 0.2},
    ) -> qm.Qubit:
        """Apply a key-sized gate sequence for the default dictionary."""
        q = qm.qubit("q")
        for size, _ in qm.items(sizes):
            for _ in qm.range(size):
                q = qm.h(q)
        return q

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 4
    assert estimate.parameters == {}


def test_for_items_vector_key_elements_are_bound_per_concrete_entry() -> None:
    """Concrete Vector-key elements drive exact per-entry resources."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.Vector[qm.UInt], qm.Float],
    ) -> qm.Qubit:
        """Apply a key-sized gate sequence for every dictionary entry."""
        q = qm.qubit("q")
        for key, _value in qm.items(data):
            for _ in qm.range(key[0]):
                q = qm.h(q)
        return q

    estimate = circuit.estimate_resources(inputs={"data": {(1, 2): 0.1, (3, 4): 0.2}})

    assert estimate.gates.total == 4
    assert estimate.parameters == {}


def test_for_items_tuple_key_binds_every_declared_component() -> None:
    """A concrete tuple key binds each component used by the loop body."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.Tuple[qm.UInt, qm.UInt], qm.Float],
    ) -> qm.Qubit:
        """Apply one gate per unit represented by both tuple components."""
        q = qm.qubit("q")
        for (left, right), _value in qm.items(data):
            for _ in qm.range(left):
                q = qm.h(q)
            for _ in qm.range(right):
                q = qm.x(q)
        return q

    estimate = circuit.estimate_resources(inputs={"data": {(1, 2): 0.1}})

    assert estimate.gates.total == 3
    assert estimate.parameters == {}


@pytest.mark.parametrize(
    "malformed_key",
    [
        pytest.param((3,), id="missing-component"),
        pytest.param("ab", id="string-is-not-a-tuple-key"),
        pytest.param((1, 2, 4), id="extra-component"),
    ],
)
def test_for_items_tuple_key_rejects_wrong_shape(malformed_key: object) -> None:
    """Tuple-key bindings require the declared arity exactly."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.Tuple[qm.UInt, qm.UInt], qm.Float],
    ) -> qm.Qubit:
        """Use both tuple components as resource-sensitive loop bounds."""
        q = qm.qubit("q")
        for (left, right), _value in qm.items(data):
            for _ in qm.range(left + right):
                q = qm.h(q)
        return q

    with pytest.raises(ValueError, match="must contain exactly 2 element"):
        circuit.estimate_resources(inputs={"data": {malformed_key: 0.1}})


def test_for_items_dynamic_vector_key_index_fails_closed() -> None:
    """Concrete Vector keys never degrade to an unbound element symbol."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.Vector[qm.UInt], qm.Float],
    ) -> qm.Qubit:
        """Use every dynamically indexed key element as a loop bound."""
        q = qm.qubit("q")
        for key, _value in qm.items(data):
            for index in qm.range(key.shape[0]):
                for _ in qm.range(key[index]):
                    q = qm.h(q)
        return q

    with pytest.raises(NotImplementedError, match="dynamically indexed"):
        circuit.estimate_resources(inputs={"data": {(1, 2): 0.1}})


def test_zero_cardinality_disables_reusable_loop_width() -> None:
    """A symbolic empty dictionary does not allocate its loop-body workspace."""

    @qm.qkernel
    def circuit(coeffs: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Allocate fixed workspace once per dictionary entry."""
        q = qm.qubit("q")
        for _, _ in qm.items(coeffs):
            _work = qm.qubit_array(3, "work")
        return q

    symbolic = circuit.estimate_resources()
    cardinality = symbolic.parameters["|coeffs|"]

    assert cardinality.is_nonnegative is True
    assert symbolic.substitute(**{"|coeffs|": 0}).qubits == 1


def test_for_items_carry_uses_concrete_entry_order() -> None:
    """A carried accumulator receives every concrete dictionary key in order."""

    @qm.qkernel
    def circuit(sizes: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Use the accumulated dictionary keys as a later loop bound."""
        total = qm.uint(0)
        for size, _ in qm.items(sizes):
            total = total + size
        q = qm.qubit("q")
        for _ in qm.range(total):
            q = qm.h(q)
        return q

    estimate = circuit.estimate_resources(inputs={"sizes": {1: 0.1, 3: 0.2}})

    assert estimate.gates.total == 4


def test_symbolic_for_items_carry_sums_each_iteration() -> None:
    """A carried items-loop bound is summed over the symbolic item index."""

    @qm.qkernel
    def circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Qubit:
        """Apply zero, one, two, and so on gates across dictionary entries."""
        total = qm.uint(0)
        q = qm.qubit("q")
        for _key, _value in qm.items(data):
            for _ in qm.range(total):
                q = qm.h(q)
            total = total + 1
        return q

    estimate = circuit.estimate_resources()
    cardinality = estimate.parameters["|data|"]
    expected = cardinality * (cardinality - 1) / 2

    assert sp.simplify(estimate.gates.total - expected) == 0
    assert set(estimate.parameters) == {"|data|"}
    for size, gate_count in ((0, 0), (1, 0), (2, 1), (3, 3), (4, 6)):
        assert estimate.substitute(**{"|data|": size}).gates.total == gate_count


def test_while_trip_count_is_nonnegative_and_zero_disables_width() -> None:
    """A while loop may execute zero times without allocating body workspace."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Allocate and measure body-local work under a measured condition."""
        trigger = qm.qubit("trigger")
        bit = qm.measure(trigger)
        while bit:
            work = qm.qubit_array(3, "work")
            work[0] = qm.h(work[0])
            bit = qm.measure(work[0])
        return qm.qubit("result")

    symbolic = circuit.estimate_resources()
    trip_count = symbolic.parameters["|while|"]
    zero = symbolic.substitute(**{"|while|": 0})

    assert trip_count.is_nonnegative is True
    assert zero.gates.total == 0
    assert zero.qubits == 1


def test_independent_while_loops_expose_distinct_trip_counts() -> None:
    """Each while call site can be specialized without changing another loop."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one- and two-gate bodies in independent measured loops."""
        first = qm.qubit("first")
        first_bit = qm.measure(qm.qubit("first_trigger"))
        while first_bit:
            first = qm.h(first)
            first_bit = qm.measure(qm.qubit("first_next"))

        second = qm.qubit("second")
        second_bit = qm.measure(qm.qubit("second_trigger"))
        while second_bit:
            second = qm.x(second)
            second = qm.z(second)
            second_bit = qm.measure(qm.qubit("second_next"))
        return first, second

    estimate = circuit.estimate_resources()

    assert set(estimate.parameters) == {"|while|", "|while[2]|"}
    specialized = estimate.substitute(**{"|while|": 3, "|while[2]|": 4})
    assert specialized.gates.total == 11


def test_skipped_for_body_preserves_later_while_name() -> None:
    """Input specialization cannot renumber a while after an empty loop."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Bit:
        """Place one measured loop inside a range before a second loop."""
        for _index in qm.range(repetitions):
            first = qm.measure(qm.qubit("first_trigger"))
            while first:
                first = qm.measure(qm.qubit("first_next"))

        second = qm.measure(qm.qubit("second_trigger"))
        while second:
            second = qm.measure(qm.qubit("second_next"))
        return second

    symbolic = circuit.estimate_resources()
    specialized = symbolic.substitute(repetitions=0)
    direct = circuit.estimate_resources(inputs={"repetitions": 0})

    assert set(specialized.parameters) == {"|while[2]|"}
    assert set(direct.parameters) == {"|while[2]|"}
    assert direct.gates == specialized.gates
    assert direct.depth == specialized.depth
    assert direct.width == specialized.width


def test_symbolic_region_probe_reuses_while_name() -> None:
    """A region-loop probe and its final evaluation share one while symbol."""

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> tuple[qm.Bit, qm.UInt]:
        """Carry a counter around one measured loop in a symbolic range."""
        total = qm.uint(0)
        output = qm.qubit("output")
        for _index in qm.range(repetitions):
            active = qm.measure(qm.qubit("trigger"))
            while active:
                output = qm.x(output)
                active = qm.measure(qm.qubit("next"))
            total = total + 1
        return qm.measure(output), total

    estimate = circuit.estimate_resources()

    assert set(estimate.parameters) == {"repetitions", "|while|"}


def test_while_names_distinguish_reused_qkernel_call_sites() -> None:
    """Two calls to one while-bearing definition retain independent counts."""

    @qm.qkernel
    def measured_loop(target: qm.Qubit) -> qm.Bit:
        """Measure one target repeatedly while its result remains true."""
        active = qm.measure(target)
        while active:
            active = qm.measure(qm.qubit("next"))
        return active

    @qm.qkernel
    def circuit() -> tuple[qm.Bit, qm.Bit]:
        """Invoke the same measured-loop definition at two call sites."""
        first = measured_loop(qm.qubit("first"))
        second = measured_loop(qm.qubit("second"))
        return first, second

    estimate = circuit.estimate_resources()

    assert set(estimate.parameters) == {"|while|", "|while[2]|"}


def test_positive_symbolic_while_releases_consumed_capture() -> None:
    """Substitution updates peak liveness when a loop consumes an outer qubit."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Measure one captured qubit before allocating later workspace."""
        retained = qm.qubit("retained")
        trigger = qm.measure(qm.qubit("trigger"))
        while trigger:
            qm.measure(retained)
            trigger = qm.measure(qm.qubit("next"))
        return qm.qubit_array(3, "later")

    estimate = circuit.estimate_resources()

    assert estimate.substitute(**{"|while|": 0}).width.peak_qubits == 4
    assert estimate.substitute(**{"|while|": 1}).width.peak_qubits == 3


def test_while_classical_recurrence_fails_closed() -> None:
    """A carried classical while value is not modeled from one traced body."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Use a while-carried counter as a later loop bound."""
        trigger = qm.qubit("trigger")
        bit = qm.measure(trigger)
        total = qm.uint(0)
        while bit:
            total = total + 1
            next_trigger = qm.qubit("next_trigger")
            bit = qm.measure(next_trigger)
        q = qm.qubit("q")
        for _ in qm.range(total):
            q = qm.x(q)
        return q

    with pytest.raises(NotImplementedError, match="WhileOperation"):
        circuit.estimate_resources()


def test_resource_substitute_uses_public_alias_symbol_identity() -> None:
    """Same-named symbols are substituted independently by public alias."""
    positive_n = sp.Symbol("n", integer=True, positive=True)
    nonnegative_n = sp.Symbol("n", integer=True, nonnegative=True)
    expression = positive_n + 10 * nonnegative_n
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=expression),
    )

    assert list(estimate.parameters) == ["n", "n__2"]
    first = estimate.parameters["n"]
    second = estimate.parameters["n__2"]

    partially_bound = estimate.substitute(n=2)
    assert partially_bound.gates.total == expression.xreplace({first: sp.Integer(2)})
    assert partially_bound.parameters == {"n__2": second}

    expected = expression.xreplace(
        {
            first: sp.Integer(2),
            second: sp.Integer(5),
        }
    )
    assert estimate.substitute(n=2, n__2=5).gates.total == expected
    assert estimate.substitute(n__2=5, n=2).gates.total == expected


def test_resource_algebra_preserves_partial_public_symbol_aliases() -> None:
    """Every resource-algebra path retains a surviving suffixed alias."""
    positive_n = sp.Symbol("n", integer=True, positive=True)
    nonnegative_n = sp.Symbol("n", integer=True, nonnegative=True)
    original = qm.ResourceEstimate(
        gates=qm.GateResources(total=positive_n + nonnegative_n),
    )
    remaining = original.parameters["n__2"]
    partial = original.substitute(n=1)
    zero = qm.ResourceEstimate.zero()
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    loop_symbol = sp.Dummy("index", integer=True, nonnegative=True)
    dependent = qm.ResourceEstimate(
        gates=qm.GateResources(total=partial.gates.total + loop_symbol),
        _symbol_aliases=partial._symbol_aliases,
    )

    results = {
        "seq-left-zero": zero.seq(partial),
        "seq-right-zero": partial.seq(zero),
        "seq-all": qm.ResourceEstimate.seq_all((zero, partial)),
        "parallel": partial.parallel(zero),
        "choice": partial.choice(zero),
        "conditional": partial.conditional(zero, sp.Gt(flag, 0)),
        "repeat": partial.repeat(2),
        "inverse": partial.inverse(),
        "sum-independent": partial.sum_over(loop_symbol, 0, 2),
        "sum-dependent": dependent.sum_over(loop_symbol, 0, 2),
        "controlled": partial.controlled(1),
        "simplify": partial.simplify(),
    }

    for name, result in results.items():
        assert result.parameters["n__2"] == remaining, name


def test_binary_resource_algebra_resolves_alias_claims_left_first() -> None:
    """The left operand keeps a colliding alias regardless of expression order."""
    left_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    right_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    left = qm.ResourceEstimate(gates=qm.GateResources(total=left_symbol))
    right = qm.ResourceEstimate(gates=qm.GateResources(total=right_symbol))

    left_first = left.seq(right)
    right_first = right.seq(left)

    assert left_first.parameters == {"n": left_symbol, "n__2": right_symbol}
    assert right_first.parameters == {"n": right_symbol, "n__2": left_symbol}


def test_resource_substitute_rejects_unknown_parameter_names() -> None:
    """A typo cannot silently leave a symbolic estimate unchanged."""
    n = sp.Symbol("n", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(gates=qm.GateResources(total=n))

    with pytest.raises(ValueError, match="Unknown resource parameter 'nn'"):
        estimate.substitute(nn=3)


def test_input_vector_shape_symbol_is_nonnegative() -> None:
    """A public array dimension permits an empty input vector."""

    @qm.qkernel
    def circuit(values: qm.Vector[qm.Float]) -> qm.Vector[qm.Qubit]:
        """Allocate one qubit per input vector element."""
        return qm.qubit_array(values.shape[0], "q")

    estimate = circuit.estimate_resources()
    (dimension,) = estimate.parameters.values()

    assert dimension.is_nonnegative is True


def test_unresolved_uint_and_bit_fallbacks_are_nonnegative() -> None:
    """Identity-qualified UInt and Bit symbols preserve their IR domains."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.ir.types.primitives import BitType, UIntType
    from qamomile.circuit.ir.value import Value

    resolver = ExprResolver()
    uint_symbol = resolver.resolve(Value(type=UIntType(), name="value"))
    bit_symbol = resolver.resolve(Value(type=BitType(), name="value"))

    assert uint_symbol.is_integer is True
    assert uint_symbol.is_nonnegative is True
    assert bit_symbol.is_integer is True
    assert bit_symbol.is_nonnegative is True


def test_expr_resolver_indexes_every_operation_result() -> None:
    """Producer lookup uses UUIDs and includes non-leading results."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
    from qamomile.circuit.ir.types.primitives import UIntType
    from qamomile.circuit.ir.value import Value

    left = Value(type=UIntType(), name="left").with_const(2)
    right = Value(type=UIntType(), name="right").with_const(3)
    first = Value(type=UIntType(), name="first")
    second = Value(type=UIntType(), name="second")
    operation = BinOp(
        operands=[left, right],
        results=[first, second],
        kind=BinOpKind.ADD,
    )
    resolver = ExprResolver(Block(name="multiple_results", operations=[operation]))

    assert resolver.resolve(second) == 5


def test_expr_resolver_caches_input_shape_aliases_across_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One resolver tree indexes each block's input-shape aliases once."""
    from qamomile.circuit.estimator import _resolver as resolver_module
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.types.primitives import FloatType, UIntType
    from qamomile.circuit.ir.value import ArrayValue, Value

    dimension = Value(type=UIntType(), name="values_dim0")
    values = ArrayValue(
        type=FloatType(),
        name="values",
        shape=(dimension,),
    )
    root = Block(
        name="root",
        label_args=["values"],
        input_values=[values],
    )
    child = Block(name="child")
    alias_spy = Mock(wraps=resolver_module.input_shape_dimension_aliases)
    monkeypatch.setattr(
        resolver_module,
        "input_shape_dimension_aliases",
        alias_spy,
    )
    resolver = ExprResolver(root)
    expected = sp.Symbol("values_dim0", integer=True, nonnegative=True)

    assert resolver.resolve(dimension) == expected
    assert resolver.resolve(dimension) == expected
    assert resolver.child_scope(child).resolve(dimension) == expected
    assert resolver.isolated_scope(root).resolve(dimension) == expected
    assert sum(call.args[0] is root for call in alias_spy.call_args_list) == 1
    assert sum(call.args[0] is child for call in alias_spy.call_args_list) == 1

    assert ExprResolver(root).resolve(dimension) == expected
    assert sum(call.args[0] is root for call in alias_spy.call_args_list) == 2


def test_controlled_composite_body_counts_own_control() -> None:
    """A controlled body-backed composite reclassifies its primitives as controlled.

    A body-backed composite (one H gate), invoked with
    an explicit control, must have its exact-body estimate count the H as a
    two-qubit (controlled) gate — the body path honors the invocation's own
    control just like the model path and ``eval_controlled_u`` do.
    """

    @qm.composite_gate(name="one_h")
    def one_h(t: qm.Qubit) -> qm.Qubit:
        """Apply a single H gate to the target."""
        return qm.h(t)

    @qm.composite_gate(name="plain_one_h")
    def plain_one_h(t: qm.Qubit) -> qm.Qubit:
        """Apply a single H gate to the target (no controls)."""
        return qm.h(t)

    @qm.qkernel
    def controlled() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one_h through the normal higher-order control operator."""
        c = qm.qubit("c")
        t = qm.qubit("t")
        controlled_one_h = qm.control(one_h)
        return controlled_one_h(c, t)

    @qm.qkernel
    def uncontrolled() -> qm.Qubit:
        """Apply plain_one_h directly."""
        t = qm.qubit("t")
        return plain_one_h(t)

    ctrl = controlled.estimate_resources()
    plain = uncontrolled.estimate_resources()

    assert plain.gates.single_qubit == 1
    assert plain.gates.two_qubit == 0
    # The control turns the single-qubit H into a two-qubit controlled gate.
    assert ctrl.gates.single_qubit == 0
    assert ctrl.gates.two_qubit == 1


def test_controlled_swap_uses_clean_ancilla_fredkin_decomposition() -> None:
    """A controlled SWAP expands to two CNOTs and one Toffoli."""

    @qm.composite_gate(name="one_swap")
    def one_swap(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Swap two target qubits."""
        return qm.swap(left, right)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply SWAP under one surrounding control."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        return qm.control(one_swap)(control, left, right)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.two_qubit == 2
    assert estimate.gates.multi_qubit == 1
    assert estimate.gates.toffoli == 1

    clifford_t = circuit.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)

    assert clifford_t.gates.total == 17
    assert clifford_t.gates.t == 7
    assert clifford_t.depth.depth == 17
    assert clifford_t.depth.clifford_depth == 10
    assert clifford_t.depth.t_depth == 3
    assert clifford_t.depth.non_clifford_depth == 3
    assert clifford_t.depth.gate_depth == 17


def test_for_items_width_reuses_wires_across_entries() -> None:
    """A bound items loop's width is the per-entry max, not the entry sum.

    Each entry's body allocates one fresh qubit, but the emitted circuit
    resets and reuses dead wires between iterations — exactly like
    ``repeat``. Sequential composition alone would report
    1 (outs) + 2 (one fresh per entry) == 3 qubits; the loop must
    report 2, matching the emitted circuit.
    """

    @qm.qkernel
    def circuit(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Vector[qm.Bit]:
        """Rotate one fresh qubit per entry against a persistent register."""
        outs = qm.qubit_array(1, "outs")
        for _key, val in qm.items(data):
            fresh = qm.qubit("fresh")
            fresh = qm.rx(fresh, val)
            outs[0], fresh = qm.cx(outs[0], fresh)
            qm.measure(fresh)
        return qm.measure(outs)

    data = {1: 0.5, 2: 1.5}
    estimate = circuit.estimate_resources(inputs={"data": data})

    assert estimate.qubits == 2
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(
        circuit,
        bindings={"data": data},
    )
    assert estimate.width.allocated_qubits == 2
    assert emitted.quantum_circuit.num_qubits == estimate.qubits
    # Gates still accumulate sequentially across entries: one rx + one
    # cx per entry.
    assert estimate.gates.total >= 4


def test_concrete_region_for_width_reuses_wires_across_iterations() -> None:
    """A concrete carried range loop matches emitted reusable wire width."""

    @qm.qkernel
    def circuit() -> tuple[qm.Bit, qm.UInt]:
        """Allocate one temporary qubit in each carried loop iteration."""
        output = qm.qubit("output")
        total = qm.uint(0)
        for _index in qm.range(2):
            fresh = qm.qubit("fresh")
            qm.measure(fresh)
            total = total + 1
        return qm.measure(output), total

    estimate = circuit.estimate_resources()

    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(circuit)
    assert estimate.qubits == 2
    assert emitted.quantum_circuit.num_qubits == estimate.qubits


def test_concrete_region_for_items_width_reuses_wires_across_entries() -> None:
    """A concrete carried items loop matches emitted reusable wire width."""

    @qm.qkernel
    def circuit(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> tuple[qm.Bit, qm.UInt]:
        """Allocate one temporary qubit in each carried items entry."""
        output = qm.qubit("output")
        total = qm.uint(0)
        for key, _value in qm.items(data):
            fresh = qm.qubit("fresh")
            qm.measure(fresh)
            total = total + key
        return qm.measure(output), total

    data = {1: 0.5, 2: 1.5}
    estimate = circuit.estimate_resources(inputs={"data": data})

    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(
        circuit,
        bindings={"data": data},
    )
    assert estimate.qubits == 2
    assert emitted.quantum_circuit.num_qubits == estimate.qubits


def test_concrete_loops_union_distinct_qinit_sites_by_identity() -> None:
    """Concrete replays deduplicate one QInit site but retain distinct sites.

    Each loop selects allocation site ``a`` on its first iteration/entry and
    site ``b`` on its second. The sites are never simultaneously live, so the
    logical peak remains one, but both are static allocations in the emitted
    circuit. A third site ``out`` after the loop makes the identity-aware
    allocation total three for RegionArg range, RegionArg ForItems, and
    carry-less ForItems alike.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qm.qkernel
    def region_range() -> tuple[qm.Bit, qm.UInt]:
        """Select two allocation sites across carried range iterations."""
        total = qm.uint(0)
        for index in qm.range(2):
            if index == 0:
                a = qm.qubit("a")
                qm.measure(a)
            else:
                b = qm.qubit("b")
                qm.measure(b)
            total = total + 1
        out = qm.qubit("out")
        return qm.measure(out), total

    @qm.qkernel
    def region_items(
        data: qm.Dict[qm.UInt, qm.Float],
    ) -> tuple[qm.Bit, qm.UInt]:
        """Select two allocation sites across carried dictionary entries."""
        total = qm.uint(0)
        for key, _value in qm.items(data):
            if key == 0:
                a = qm.qubit("a")
                qm.measure(a)
            else:
                b = qm.qubit("b")
                qm.measure(b)
            total = total + 1
        out = qm.qubit("out")
        return qm.measure(out), total

    @qm.qkernel
    def plain_items(data: qm.Dict[qm.UInt, qm.Float]) -> qm.Bit:
        """Select two allocation sites across carry-less dictionary entries."""
        for key, _value in qm.items(data):
            if key == 0:
                a = qm.qubit("a")
                qm.measure(a)
            else:
                b = qm.qubit("b")
                qm.measure(b)
        out = qm.qubit("out")
        return qm.measure(out)

    data = {0: 0.5, 1: 1.5}
    cases = (
        (region_range, {}),
        (region_items, {"data": data}),
        (plain_items, {"data": data}),
    )
    for kernel, bindings in cases:
        estimate = kernel.estimate_resources(inputs=bindings)
        emitted = QiskitTranspiler().transpile(
            kernel,
            bindings=bindings,
        )

        assert estimate.width.allocated_qubits == 3
        assert estimate.width.peak_qubits == 1
        assert emitted.quantum_circuit.num_qubits == (estimate.width.allocated_qubits)


def test_nested_qinit_identity_is_namespaced_by_helper_callsite() -> None:
    """Distinct helper calls allocate separately while one replayed call reuses."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    distinct = _distinct_helper_callsites.estimate_resources()
    repeated = _repeated_helper_callsite.estimate_resources()
    distinct_emitted = (
        QiskitTranspiler().transpile(_distinct_helper_callsites).quantum_circuit
    )
    repeated_emitted = (
        QiskitTranspiler().transpile(_repeated_helper_callsite).quantum_circuit
    )

    assert distinct.width.allocated_qubits == 3
    assert distinct.width.peak_qubits == 2
    assert distinct_emitted.num_qubits == distinct.width.allocated_qubits
    assert repeated.width.allocated_qubits == 2
    assert repeated.width.peak_qubits == 2
    assert repeated_emitted.num_qubits == repeated.width.allocated_qubits


def test_same_named_range_fallback_symbols_remain_independent() -> None:
    """Separate unsupported range recurrences never collapse by source name."""

    @qm.qkernel
    def circuit(n: qm.UInt, m: qm.UInt) -> qm.Bit:
        """Compare two nonlinear carries created under the same variable name."""
        total = qm.float_(0.5)
        for _index in qm.range(n):
            total = total * total
        left = total

        total = qm.float_(0.25)
        for _index in qm.range(m):
            total = total * total
        right = total

        q = qm.qubit("q")
        if left != right:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return qm.measure(q)

    estimate = circuit.estimate_resources()
    final_symbols = [
        symbol
        for symbol in estimate.gates.total.free_symbols
        if symbol.name == "total_after_loop"
    ]

    assert len(final_symbols) == 2
    assert final_symbols[0] != final_symbols[1]
    assert estimate.gates.total != 1

    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    emitted = QiskitTranspiler().transpile(
        circuit,
        bindings={"n": 0, "m": 0},
    )
    assert emitted.quantum_circuit.count_ops()["x"] == 1
    assert emitted.quantum_circuit.count_ops()["h"] == 1
    assert emitted.quantum_circuit.count_ops()["z"] == 1


def test_same_named_for_items_fallback_symbols_remain_independent() -> None:
    """Separate unsupported items recurrences never collapse by source name."""

    @qm.qkernel
    def circuit(
        left_data: qm.Dict[qm.UInt, qm.Float],
        right_data: qm.Dict[qm.UInt, qm.Float],
    ) -> qm.Qubit:
        """Compare two nonlinear item carries with one reused variable name."""
        total = qm.float_(0.5)
        for _key, _value in qm.items(left_data):
            total = total * total
        left = total

        total = qm.float_(0.25)
        for _key, _value in qm.items(right_data):
            total = total * total
        right = total

        q = qm.qubit("q")
        if left != right:
            q = qm.x(q)
            q = qm.h(q)
            q = qm.z(q)
        else:
            q = qm.x(q)
        return q

    estimate = circuit.estimate_resources()
    final_symbols = [
        symbol
        for symbol in estimate.gates.total.free_symbols
        if symbol.name == "total_after_items"
    ]

    assert len(final_symbols) == 2
    assert final_symbols[0] != final_symbols[1]
    assert estimate.gates.total != 1


def test_allocation_site_identity_is_internal_to_resource_estimates() -> None:
    """Random QInit UUIDs do not affect equality or public serialization."""
    left = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"build-random-left": sp.Integer(1)},
    )
    right = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"build-random-right": sp.Integer(1)},
    )

    assert left == right
    assert "_allocation_sites" not in left.to_dict()


def test_branch_allocations_specialize_or_union_by_condition_kind() -> None:
    """Compile-time sites specialize while runtime choices reserve both."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    left = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"left": sp.Integer(1)},
    )
    right = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=2, peak_qubits=2),
        _allocation_sites={"right": sp.Integer(2)},
    )

    conditional = left.conditional(right, sp.Eq(flag, 1))
    choice = left.choice(right)

    assert set(conditional._allocation_sites) == {"left", "right"}
    assert conditional.substitute(flag=1).width.allocated_qubits == 1
    assert conditional.substitute(flag=0).width.allocated_qubits == 2
    assert conditional.substitute(flag=1).width.peak_qubits == 1
    assert conditional.substitute(flag=0).width.peak_qubits == 2
    assert choice._allocation_sites == {"left": 1, "right": 2}
    assert choice.width.allocated_qubits == 3
    assert choice.width.peak_qubits == 2


def test_constant_conditional_ignores_unreachable_provenance() -> None:
    """A decided condition never merges the unreachable branch metadata."""
    logical = qm.ResourceEstimate(
        gates=qm.GateResources(total=1),
        basis=qm.GateBasis.LOGICAL,
    )
    clifford_t = qm.ResourceEstimate(
        gates=qm.GateResources(total=100),
        assumptions=(qm.ResourceAssumption("unreachable"),),
        derivation=qm.EstimateDerivation.MODELED,
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-3,
    )

    assert logical.conditional(clifford_t, sp.true) is logical
    assert logical.conditional(clifford_t, sp.false) is clifford_t


def test_quantum_block_input_width_requires_an_integer() -> None:
    """Public Block estimates retain formal quantum-array shape domains."""
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.types import QubitType, UIntType
    from qamomile.circuit.ir.value import ArrayValue, Value

    dimension = Value(type=UIntType(), name="n")
    register = ArrayValue(type=QubitType(), name="register", shape=(dimension,))
    block = Block(
        name="input_only",
        label_args=["register"],
        input_values=[register],
        output_values=[register],
        output_names=["register"],
        operations=[],
    )

    symbolic = qm.estimate_resources(block)
    with pytest.raises(ValueError, match="non-integer value"):
        symbolic.substitute(n=1.5)
    with pytest.raises(ValueError, match="non-integer value"):
        qm.estimate_resources(block, inputs={"n": 1.5})


def test_zero_trip_repeat_disables_internal_allocation_sites() -> None:
    """A zero-trip repeat contributes neither width nor an active QInit site."""
    body = qm.ResourceEstimate(
        width=qm.WidthResources(allocated_qubits=1, peak_qubits=1),
        _allocation_sites={"body": sp.Integer(1)},
    )

    repeated = body.repeat(0)

    assert repeated.width.allocated_qubits == 0
    assert repeated.width.peak_qubits == 0
    assert repeated._allocation_sites == {"body": sp.Integer(0)}


def test_zero_trip_repeat_prunes_metadata_and_assumptions() -> None:
    """Literal and specialized zero repeats contribute no modeled metadata."""
    assumption = qm.ResourceAssumption("modeled body")
    body = qm.ResourceEstimate(
        gates=qm.GateResources(total=1),
        assumptions=(assumption,),
        derivation=qm.EstimateDerivation.MODELED,
        approximation=qm.ApproximationStatus.APPROXIMATE,
    )
    repetitions = sp.Symbol("repetitions", integer=True, nonnegative=True)

    literal_zero = body.repeat(0)
    symbolic_zero = body.repeat(repetitions).substitute(repetitions=0)

    for estimate in (literal_zero, symbolic_zero):
        assert estimate.gates.total == 0
        assert estimate.assumptions == ()
        assert estimate.quality is qm.EstimateQuality.EXACT
        assert estimate.approximation is qm.ApproximationStatus.EXACT

    active = body.repeat(repetitions).substitute(repetitions=2)
    assert active.gates.total == 2
    assert active.assumptions == (assumption,)
    assert active.derivation is qm.EstimateDerivation.MODELED
    assert active.approximation is qm.ApproximationStatus.APPROXIMATE
