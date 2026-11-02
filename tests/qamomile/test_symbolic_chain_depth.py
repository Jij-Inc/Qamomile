"""Exact adjacent-chain depth proofs stay compact before specialization."""

import pytest

import qamomile as qm

SOURCE = """
import qamomile as qm
@qm.qkernel
def ghz(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.h(qm.qubits(n), target=0)
    for i in range(n - 1):
        state = qm.cx(state, control=i, target=i + 1)
    return qm.measure(state)
"""


@pytest.mark.parametrize("n", [1, 2, 4, 9, 100_000])
def test_symbolic_ghz_depth_is_exact_without_expansion(n):
    """The lane-zero prefix joins the entire adjacent dependence chain."""
    family = qm.build_source(SOURCE)["ghz"]
    study = qm.resources.logical(family)
    restored = qm.resources.ResourceStudy.from_bytes(study.to_bytes())
    for metric in (
        qm.resources.LOGICAL_GATE_DEPTH,
        qm.resources.LOGICAL_CLIFFORD_DEPTH,
    ):
        value = restored.metric(metric)
        assert value.relation is qm.resources.Relation.EQUALITY
        assert value.evaluate({"n": n}).numeric_enclosure() == (n, n)
        if n <= 9:
            assert qm.resources.logical(qm.specialize(family, n=n)).metric(
                metric
            ).numeric_enclosure() == (n, n)


@pytest.mark.parametrize("target", [0, 1, 2])
def test_chain_retains_per_lane_completion_times(target):
    """A following head-lane gate may overlap the remaining chain tail."""
    source = SOURCE.replace(
        "    return qm.measure(state)",
        f"    state = qm.h(state, target={target})\n    return qm.measure(state)",
    )
    family = qm.build_source(source)["ghz"]
    value = qm.resources.logical(family).metric(qm.resources.LOGICAL_GATE_DEPTH)
    assert value.relation is qm.resources.Relation.EQUALITY
    for n in (4, 8):
        expected = (
            qm.resources.logical(qm.specialize(family, n=n))
            .metric(qm.resources.LOGICAL_GATE_DEPTH)
            .numeric_enclosure()
        )
        assert value.evaluate({"n": n}).numeric_enclosure() == expected


def test_incomplete_or_nonadjacent_loop_remains_conservative():
    """Only a proved complete adjacent chain receives an equality claim."""
    for bound, target in (("n - 2", "i + 1"), ("n - 2", "i + 2")):
        source = SOURCE.replace("range(n - 1)", f"range({bound})").replace(
            "target=i + 1", f"target={target}"
        )
        value = qm.resources.logical(qm.build_source(source)["ghz"]).metric(
            qm.resources.LOGICAL_GATE_DEPTH
        )
        assert value.relation is not qm.resources.Relation.EQUALITY


def test_unproved_later_lane_prefix_does_not_get_chain_equality():
    """A different selected prefix requires a richer incoming recurrence proof."""
    source = SOURCE.replace("target=0", "target=1")
    value = qm.resources.logical(qm.build_source(source)["ghz"]).metric(
        qm.resources.LOGICAL_GATE_DEPTH
    )
    assert value.relation is not qm.resources.Relation.EQUALITY


def test_two_chains_require_a_new_incoming_frontier_proof():
    """An existing chain profile is not incorrectly serialized with the next loop."""
    source = SOURCE.replace(
        "    return qm.measure(state)",
        "    for j in range(n - 1):\n        state = qm.cx(state, control=j, target=j + 1)\n    return qm.measure(state)",
    )
    family = qm.build_source(source)["ghz"]
    value = qm.resources.logical(family).metric(qm.resources.LOGICAL_GATE_DEPTH)
    assert value.relation is not qm.resources.Relation.EQUALITY
    exact = (
        qm.resources.logical(qm.specialize(family, n=4))
        .metric(qm.resources.LOGICAL_GATE_DEPTH)
        .numeric_enclosure()[0]
    )
    lower, upper = value.evaluate({"n": 4}).numeric_enclosure()
    assert lower <= exact <= upper
