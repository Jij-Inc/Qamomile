"""Tests for the symbolic ResourceEstimator compiler service."""

from __future__ import annotations

import sympy as sp

import qamomile.circuit as qm


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


def test_composite_resource_model_is_policy_selected() -> None:
    """ResourceEstimator selects callable models unless exact body is requested."""

    @qm.composite_gate
    @qm.qkernel
    def modeled_h(q: qm.Qubit) -> qm.Qubit:
        """Apply one H gate."""
        return qm.h(q)

    @modeled_h.resource_model
    def modeled_h_resources(ctx: qm.ResourceContext) -> qm.ResourceEstimate:
        """Return an intentionally different model for policy testing."""
        return qm.ResourceEstimate(gates=qm.GateResources(total=5, clifford=5))

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Call the modeled composite."""
        q = qm.qubit("q")
        q = modeled_h(q)
        return q

    modeled = qm.ResourceEstimator().estimate(circuit)
    exact = qm.ResourceEstimator(policy=qm.ResourcePolicy.EXACT_BODY).estimate(circuit)

    assert modeled.gates.total == 5
    assert exact.gates.total == 1


def test_opaque_fixed_resource_model_counts_calls_and_gates() -> None:
    """Opaque callables use explicit resource models instead of metadata."""
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
        resource_model=qm.FixedResourceModel(oracle_estimate),
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

    A body-backed composite (one H gate) with no resource model, invoked with
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

    ctrl = controlled.estimate_resources(policy=qm.ResourcePolicy.EXACT_BODY)
    plain = uncontrolled.estimate_resources(policy=qm.ResourcePolicy.EXACT_BODY)

    assert plain.gates.single_qubit == 1
    assert plain.gates.two_qubit == 0
    # The control turns the single-qubit H into a two-qubit controlled gate.
    assert ctrl.gates.single_qubit == 0
    assert ctrl.gates.two_qubit == 1
