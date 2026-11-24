"""Native seed absence, explicit zero, and child-stream projection contracts."""

import os

import pytest

import qamomile as qm
from qamomile._native import compiler as native

SOURCE = """
import qamomile as qm
@qm.qkernel
def coin() -> qm.Shot[qm.Bit]:
    return qm.measure(qm.h(qm.prepare(False)))
@qm.qkernel
def coins() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    b = qm.prepare(False)
    if a:
        b = qm.x(b)
    return (a, qm.measure(b))
"""


@pytest.mark.parametrize(
    "planner_name",
    [
        "plan_independent_pauli_expectation",
        "plan_qwc_pauli_expectation",
        "plan_commuting_pauli_expectation",
    ],
)
def test_native_measurement_plans_distinguish_zero_from_absence(planner_name):
    """Every grouping preserves None and derives distinct streams for zero."""
    observable = (qm.X(0) + qm.Z(0))._native_literal
    planner = getattr(native, planner_name)
    assert [row[-1] for row in planner(observable, "4", None)] == [None, None]
    seeded = planner(observable, "4", 0)
    seeds = [row[-1] for row in seeded]
    assert len(set(seeds)) == 2
    assert all(type(seed) is int and 0 <= seed < 1 << 32 for seed in seeds)
    assert seeded == planner(observable, "4", 0)


@pytest.mark.parametrize("operation", ["sample", "value", "execute"])
def test_native_reference_uses_entropy_only_for_missing_seed(monkeypatch, operation):
    """Reference delegates omitted seeds to OS entropy without treating zero specially."""
    payload = qm.build_source(SOURCE).specialize("coin").to_bytes()
    entropy_calls = []

    def entropy(size):
        entropy_calls.append(size)
        return (7).to_bytes(size, "little")

    monkeypatch.setattr(os, "urandom", entropy)

    def execute(seed):
        if operation == "sample":
            return native.sample_static_circuit_at(payload, {}, 64, seed)
        if operation == "value":
            return native.evaluate_reference_value_at(payload, {}, seed)
        result = native.execute_reference_at(payload, {}, 64, seed)
        return result.kind, result.counts

    assert execute(None) == execute(7)
    assert execute(0) == execute(0)
    assert entropy_calls == [8]


def test_native_reference_batch_preserves_missing_seed_per_invocation(monkeypatch):
    """Each unseeded batch row obtains entropy, while seeded rows advance from zero."""
    payload = qm.build_source(SOURCE).specialize("coin").to_bytes()
    entropy_calls = []

    def entropy(size):
        value = 7 + len(entropy_calls)
        entropy_calls.append(size)
        return value.to_bytes(size, "little")

    monkeypatch.setattr(os, "urandom", entropy)
    unseeded = native.execute_reference_batch_at(payload, [{}, {}], 64, None)
    assert [row.counts for row in unseeded] == [
        native.execute_reference_at(payload, {}, 64, seed).counts for seed in [7, 8]
    ]
    seeded = native.execute_reference_batch_at(payload, [{}, {}], 64, 0)
    assert [row.counts for row in seeded] == [
        native.execute_reference_at(payload, {}, 64, seed).counts for seed in [0, 1]
    ]
    assert entropy_calls == [8, 8]
    with pytest.raises(ValueError, match="Q8145"):
        native.execute_reference_batch_at(payload, [{}, {}], 1, (1 << 64) - 1)


@pytest.mark.parametrize("seed", [None, 0])
def test_native_shot_children_preserve_seed_policy(seed):
    """An omitted continuation seed remains absent; zero starts an explicit stream."""
    payload = qm.build_source(SOURCE).specialize("coins").to_bytes()
    state = native.ShotExecutionHandle(payload, {}, 4, seed)
    first = state.next_request()
    assert first.seed == seed
    state.accept([([0], 2), ([1], 2)])
    child_seeds = []
    while (request := state.next_request()) is not None:
        child_seeds.append(request.seed)
        state.accept([([0], request.shots)])
    assert len(child_seeds) == 2
    if seed is None:
        assert child_seeds == [None, None]
    else:
        assert all(type(value) is int for value in child_seeds)
        assert len({seed, *child_seeds}) == 3
    assert sum(count for _, count in state.finish()) == 4
