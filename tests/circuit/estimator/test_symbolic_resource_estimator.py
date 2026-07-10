"""Tests for the symbolic ResourceEstimator compiler service."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.circuit as qm


@qm.qkernel
def _basis_probe(theta: qm.Float) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
    """Apply one exact Toffoli and one approximate axial rotation."""
    left = qm.qubit("left")
    right = qm.qubit("right")
    target = qm.qubit("target")
    left, right, target = qm.ccx(left, right, target)
    target = qm.rx(target, theta)
    return left, right, target


def test_clifford_t_basis_lowers_body_gates_and_reports_quality() -> None:
    """Basis lowering reports aggregate Clifford+T counts and approximation."""
    logical = _basis_probe.estimate_resources()
    lowered = _basis_probe.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1 / 8,
    )

    assert logical.gates.total == 2
    assert logical.quality is qm.EstimateQuality.EXACT
    assert lowered.gates.total == 26
    assert lowered.gates.t == 16
    assert lowered.gates.rotation == 0
    assert lowered.quality is qm.EstimateQuality.UPPER_BOUND


def test_gate_basis_accepts_strings_and_rejects_unknown_values() -> None:
    """The public basis option accepts notebook-friendly strings safely."""
    estimate = _basis_probe.estimate_resources(basis="logical")
    assert estimate.gates.total == 2

    with pytest.raises(ValueError, match="expected one of: logical, clifford_t"):
        _basis_probe.estimate_resources(basis="surface_code")


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

    assert estimate.gates.t == 21
    assert estimate.gates.total == 45
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.qubits == 5


def test_clifford_t_basis_rejects_missing_controlled_gate_lowering() -> None:
    """Unsupported controlled gates fail instead of reporting logical counts."""

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
    n = sp.Symbol("n", integer=True, positive=True)

    assert sp.simplify(estimate.gates.total - (2**n - 1)) == 0


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
