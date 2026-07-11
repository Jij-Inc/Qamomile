"""Tests for the single-source body-derived callable resource contract."""

from __future__ import annotations

import qamomile.circuit as qm


def test_different_bodies_produce_different_estimates() -> None:
    """Changing executable operations changes resources without extra metadata."""

    @qm.composite_gate
    def one_gate(q: qm.Qubit) -> qm.Qubit:
        """Apply one gate."""
        return qm.h(q)

    @qm.composite_gate
    def three_gates(q: qm.Qubit) -> qm.Qubit:
        """Apply three gates."""
        return qm.z(qm.x(qm.h(q)))

    @qm.qkernel
    def first() -> qm.Qubit:
        """Invoke the one-gate body."""
        return one_gate(qm.qubit("q"))

    @qm.qkernel
    def second() -> qm.Qubit:
        """Invoke the three-gate body."""
        return three_gates(qm.qubit("q"))

    assert first.estimate_resources().gates.total == 1
    assert second.estimate_resources().gates.total == 3


def test_opaque_cost_remains_explicit_when_no_body_exists() -> None:
    """A bodyless opaque callable may still declare its otherwise unknown cost."""
    oracle = qm.opaque(
        "oracle",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(total=5),
            calls=qm.CallResources(queries_by_name={"oracle": 1}),
        ),
    )

    @qm.qkernel
    def algorithm() -> qm.Qubit:
        """Invoke one opaque oracle."""
        (q,) = oracle(qm.qubit("q"))
        return q

    estimate = algorithm.estimate_resources()
    assert estimate.gates.total == 5
    assert estimate.calls.queries_by_name == {"oracle": 1}


def test_inputs_specialize_a_symbolic_body_estimate() -> None:
    """Concrete qkernel inputs evaluate a symbolic estimate through one API."""

    @qm.qkernel
    def repeated_h(n: qm.UInt) -> qm.Qubit:
        """Apply a symbolic number of gates."""
        q = qm.qubit("q")
        for _ in qm.range(n):
            q = qm.h(q)
        return q

    symbolic = repeated_h.estimate_resources()
    concrete = repeated_h.estimate_resources(inputs={"n": 8})

    assert str(symbolic.gates.total) == "n"
    assert concrete.gates.total == 8
