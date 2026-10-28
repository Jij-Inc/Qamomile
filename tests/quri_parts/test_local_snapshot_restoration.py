"""Preserve local QURI Parts SDK values through public job restoration."""

from __future__ import annotations

import json
import math
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

pytest.importorskip("quri_parts.circuit")
pytest.importorskip("quri_parts.qulacs")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.quri_parts import QuriPartsExecutor, QuriPartsTranspiler  # noqa: E402


@qmc.qkernel
def _sample_pair(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare a rotated bit followed by an independent zero bit.

    Args:
        theta (qmc.Float): Rotation angle in radians.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Rotated bit and zero bit measurement outcomes.
    """
    first = qmc.qubit("first")
    second = qmc.qubit("second")
    first = qmc.ry(first, theta)
    return qmc.measure(first), qmc.measure(second)


@qmc.qkernel
def _expectation(theta: qmc.Float, observable: qmc.Observable) -> qmc.Float:
    """Estimate an observable on one rotated qubit.

    Args:
        theta (qmc.Float): Rotation angle in radians.
        observable (qmc.Observable): Observable evaluated on the rotated state.

    Returns:
        qmc.Float: Expectation value of the supplied observable.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.ry(qubit, theta)
    return qmc.expval(qubit, observable)


@qmc.qkernel
def _sample_superposition() -> qmc.Bit:
    """Measure one qubit with equal zero and one probabilities.

    Returns:
        qmc.Bit: One measurement outcome from the superposition.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _sample_one() -> qmc.Bit:
    """Measure a qubit prepared in the one state.

    Returns:
        qmc.Bit: A deterministic one measurement outcome.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.x(qubit)
    return qmc.measure(qubit)


@pytest.mark.parametrize("theta,expected_bit", [(0.0, 0), (math.pi, 1)])
def test_default_multinomial_sample_snapshot_restores_without_resubmission(
    theta: float,
    expected_bit: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ry(0)|0> and Ry(pi)|0> produce bits 0 and 1 for all 2048 shots."""
    program = QuriPartsTranspiler().transpile(_sample_pair, parameters=["theta"])
    executor = QuriPartsExecutor()
    bindings = {"theta": theta}
    # More than 1024 shots selects the SDK's NumPy multinomial sampling path.
    original = program.sample(executor, shots=2048, bindings=bindings)
    snapshot = original.snapshot()
    assert snapshot.executions == ()
    assert snapshot.execution is not None
    assert type(snapshot.execution.value) is dict
    assert all(type(count) is int for count in snapshot.execution.value.values())
    payload = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
    submit = Mock(side_effect=AssertionError("Restoration must not sample again"))
    lookup = Mock(side_effect=AssertionError("Local values need no provider lookup"))
    monkeypatch.setattr(executor, "submit_sample", submit)
    monkeypatch.setattr(executor, "restore", lookup)

    restored = program.restore(executor, qmc.JobSnapshot.from_dict(payload), bindings)

    assert type(restored) is qmc.SampleJob
    assert restored.result() == original.result()
    assert restored.result().results == [((expected_bit, 0), 2048)]
    assert restored.result().shots == 2048
    values, count = restored.result().results[0]
    assert type(values) is tuple
    assert all(type(value) is int for value in values)
    assert type(count) is int
    submit.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.parametrize("theta,expected", [(0.0, 1.0), (math.pi, -1.0)])
def test_sampling_estimator_snapshot_restores_native_float_without_resubmission(
    theta: float,
    expected: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ry(theta)|0> has Z expectation cos(theta), hence +1/-1 at 0/pi."""
    from quri_parts.core.estimator.sampling import create_sampling_estimator
    from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
    from quri_parts.core.sampling.shots_allocator import (
        create_equipartition_shots_allocator,
    )
    from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler

    estimator = create_sampling_estimator(
        total_shots=2048,
        sampler=create_qulacs_vector_concurrent_sampler(),
        measurement_factory=bitwise_commuting_pauli_measurement,
        shots_allocator=create_equipartition_shots_allocator(),
    )
    executor = QuriPartsExecutor(bound_estimator=estimator)
    program = QuriPartsTranspiler().transpile(
        _expectation, bindings={"observable": qm_o.Z(0)}, parameters=["theta"]
    )
    bindings = {"theta": theta}
    original = program.run(executor, bindings=bindings)
    snapshot = original.snapshot()
    assert snapshot.executions == ()
    assert snapshot.execution is not None
    assert type(snapshot.execution.children[0].value) is float
    payload = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
    submit = Mock(side_effect=AssertionError("Restoration must not estimate again"))
    lookup = Mock(side_effect=AssertionError("Local values need no provider lookup"))
    monkeypatch.setattr(executor, "submit_estimate", submit)
    monkeypatch.setattr(executor, "submit_estimates", submit)
    monkeypatch.setattr(executor, "restore", lookup)

    restored = program.restore(executor, qmc.JobSnapshot.from_dict(payload), bindings)

    assert type(restored) is type(original)
    assert type(restored.result()) is float
    np.testing.assert_allclose(
        [original.result(), restored.result()],
        [expected, expected],
        atol=1e-12,
        rtol=0.0,
    )
    submit.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.parametrize("shots", [None, 3])
def test_ideal_sampler_fractional_weights_reject_before_creating_job(
    shots: int | None,
) -> None:
    """Ideal H-state weights cannot become observed counts for sample or run."""
    from quri_parts.qulacs.sampler import create_qulacs_vector_ideal_sampler

    program = QuriPartsTranspiler().transpile(_sample_superposition)
    executor = QuriPartsExecutor(sampler=create_qulacs_vector_ideal_sampler())

    with pytest.raises(ValueError, match="whole-number.*create_qulacs_vector_sampler"):
        if shots is None:
            program.run(executor)
        else:
            program.sample(executor, shots=shots)


@pytest.mark.parametrize("shots", [None, 3])
def test_ideal_sampler_whole_counts_snapshot_restores_without_resubmission(
    shots: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ideal X-state counts restore as integers while zero-weight keys disappear."""
    from quri_parts.qulacs.sampler import create_qulacs_vector_ideal_sampler

    program = QuriPartsTranspiler().transpile(_sample_one)
    executor = QuriPartsExecutor(sampler=create_qulacs_vector_ideal_sampler())
    original = (
        program.run(executor)
        if shots is None
        else program.sample(executor, shots=shots)
    )
    snapshot = original.snapshot()
    assert snapshot.execution is not None
    assert snapshot.execution.value == {"1": shots or 1}
    assert all(type(count) is int for count in snapshot.execution.value.values())
    payload = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
    submit = Mock(side_effect=AssertionError("Restoration must not sample again"))
    lookup = Mock(side_effect=AssertionError("Local values need no provider lookup"))
    monkeypatch.setattr(executor, "submit_sample", submit)
    monkeypatch.setattr(executor, "restore", lookup)

    restored = program.restore(executor, qmc.JobSnapshot.from_dict(payload))

    assert type(restored) is type(original)
    assert restored.result() == original.result()
    if shots is None:
        assert type(restored.result()) is int
        assert restored.result() == 1
    else:
        assert restored.result().results == [(1, shots)]
        assert restored.result().shots == shots
    submit.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "count,expected",
    [
        pytest.param(3, 3, id="python-int"),
        pytest.param(np.int64(3), 3, id="numpy-int"),
        pytest.param(np.uint64(3), 3, id="numpy-unsigned-int"),
        pytest.param(3.0, 3, id="python-whole-float"),
        pytest.param(np.float32(3.0), 3, id="numpy-whole-float32"),
        pytest.param(np.float64(3.0), 3, id="numpy-whole-float64"),
        pytest.param(10**1000, 10**1000, id="large-int-without-float-conversion"),
    ],
)
def test_sampler_whole_count_scalars_preserve_exact_value(
    count: object, expected: int
) -> None:
    """Whole SDK scalars become portable integers without magnitude loss."""
    from quri_parts.circuit import QuantumCircuit

    sampler = Mock(return_value={0: count, 1: np.float64(0.0)})
    executor = QuriPartsExecutor(sampler=sampler)

    counts = executor.execute(QuantumCircuit(1), shots=expected)

    assert counts == {"0": expected}
    assert type(counts["0"]) is int


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(1.5, id="fractional-python-float"),
        pytest.param(np.float32(1.5), id="fractional-numpy-float"),
        pytest.param(np.nextafter(2.0, 0.0), id="roundoff-is-not-rounded"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(-1, id="negative-int"),
        pytest.param(np.int64(-1), id="negative-numpy-int"),
        pytest.param(-1.0, id="negative-whole-float"),
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("3", id="numeric-string"),
        pytest.param(3 + 0j, id="complex-scalar"),
    ],
)
def test_sampler_invalid_count_scalars_reject_early(count: object) -> None:
    """Invalid sampler values fail before an unusable completed job can exist."""
    from quri_parts.circuit import QuantumCircuit

    executor = QuriPartsExecutor(sampler=Mock(return_value={0: count}))

    with pytest.raises(ValueError, match="whole-number.*create_qulacs_vector_sampler"):
        executor.execute(QuantumCircuit(1), shots=3)
