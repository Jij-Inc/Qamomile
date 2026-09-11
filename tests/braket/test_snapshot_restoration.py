"""Restore cached Braket simulator results through public execution jobs."""

from __future__ import annotations

import json
import math
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.braket
pytest.importorskip("braket.circuits")

from braket.devices import LocalSimulator  # noqa: E402

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.braket import BraketExecutor, BraketTranspiler  # noqa: E402
from qamomile.braket.execution import BraketExecutionHandle  # noqa: E402
from qamomile.circuit.transpiler import (  # noqa: E402
    ExecutionReference,
    ExecutionSnapshotKind,
    JobSnapshot,
)
from qamomile.circuit.transpiler.execution_request import (  # noqa: E402
    Exact,
    ShotBased,
)


@qmc.qkernel
def _sample_pair(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Measure a rotated qubit followed by an independent zero qubit.

    Args:
        theta (qmc.Float): Rotation angle in radians for the first qubit.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Rotated bit followed by the zero bit.
    """
    first = qmc.qubit("first")
    second = qmc.qubit("second")
    first = qmc.ry(first, theta)
    return qmc.measure(first), qmc.measure(second)


@qmc.qkernel
def _expectation(theta: qmc.Float, observable: qmc.Observable) -> qmc.Float:
    """Evaluate an observable after rotating a zero qubit.

    Args:
        theta (qmc.Float): Rotation angle in radians.
        observable (qmc.Observable): One-qubit observable to evaluate.

    Returns:
        qmc.Float: Expectation value of the observable.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.ry(qubit, theta)
    return qmc.expval(qubit, observable)


@qmc.qkernel
def _expectation_pair(
    theta: qmc.Float,
    first_observable: qmc.Observable,
    second_observable: qmc.Observable,
) -> tuple[qmc.Float, qmc.Float]:
    """Return expectations for two separately rotated registers in order.

    Args:
        theta (qmc.Float): Rotation angle in radians for both registers.
        first_observable (qmc.Observable): First observable to evaluate.
        second_observable (qmc.Observable): Second observable to evaluate.

    Returns:
        tuple[qmc.Float, qmc.Float]: First expectation followed by the second.
    """
    first = qmc.qubit_array(1, "first")
    first[0] = qmc.ry(first[0], theta)
    second = qmc.qubit_array(1, "second")
    second[0] = qmc.ry(second[0], theta)
    first_value = qmc.expval(first, first_observable)
    second_value = qmc.expval(second, second_observable)
    return first_value, second_value


def _forbid_submission_and_lookup(
    executor: BraketExecutor, monkeypatch: pytest.MonkeyPatch
) -> Mock:
    """Reject provider submission and lookup after the original local run.

    Args:
        executor (BraketExecutor): Adapter whose submission methods to guard.
        monkeypatch (pytest.MonkeyPatch): Fixture managing temporary guards.

    Returns:
        Mock: Shared guard that raises on every forbidden call.
    """
    forbidden = Mock(side_effect=AssertionError("Snapshot restore must stay local"))
    for name in ("submit_sample", "submit_estimate", "submit_estimates", "restore"):
        monkeypatch.setattr(executor, name, forbidden)
    for name in ("run", "run_batch"):
        monkeypatch.setattr(executor.device, name, forbidden)
    return forbidden


@pytest.mark.parametrize("theta,expected_bit", [(0.0, 0), (math.pi, 1)])
def test_local_sample_requires_cached_result_and_restores_typed_values(
    theta: float, expected_bit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real simulator counts restore typed tuples only after explicit retrieval.

    Ry(0)|0> = |0> and Ry(pi)|0> = |1>, so every shot has the same measured bit.
    """
    executable = BraketTranspiler().transpile(_sample_pair, parameters=["theta"])
    executor = BraketExecutor(LocalSimulator())
    bindings = {"theta": theta}
    job = executable.sample(executor, shots=17, bindings=bindings)
    forbidden = _forbid_submission_and_lookup(executor, monkeypatch)
    native_task = job.native
    assert native_task is not None

    with monkeypatch.context() as before:
        before.setattr(native_task, "state", forbidden)
        before.setattr(native_task, "result", forbidden)
        with pytest.raises(ValueError, match=r"call result\(\) successfully"):
            job.snapshot()
    original_result = job.result()
    monkeypatch.setattr(native_task, "state", forbidden)
    monkeypatch.setattr(native_task, "result", forbidden)
    monkeypatch.setattr(BraketExecutionHandle, "result", forbidden)

    saved = JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))
    restored = executable.restore(executor, saved, bindings)

    assert saved.executions == ()
    assert saved.execution is not None
    assert saved.execution.kind is ExecutionSnapshotKind.LOCAL
    assert saved.execution.value == {f"0{expected_bit}": 17}
    assert type(restored) is type(job)
    assert restored.result() == original_result
    assert restored.result().results == [((expected_bit, 0), 17)]
    assert restored.result().shots == 17
    values, count = restored.result().results[0]
    assert type(values) is tuple
    assert all(type(value) is int for value in values)
    assert type(count) is int
    forbidden.assert_not_called()


@pytest.mark.parametrize("theta,expected", [(0.0, 1.0), (math.pi, -1.0)])
@pytest.mark.parametrize("estimation", [Exact(), ShotBased(17)], ids=["exact", "shot"])
def test_local_nonconstant_expectation_restores_cached_scalar(
    theta: float,
    expected: float,
    estimation: Exact | ShotBased,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real single-task and native-batch results restore as Python floats.

    The Z expectation of Ry(theta)|0> is cos(theta): +1 at zero and -1 at pi.
    Both states are Z eigenstates, so shot estimates are deterministic too.
    """
    executable = BraketTranspiler().transpile(
        _expectation,
        bindings={"observable": qm_o.Z(0)},
        parameters=["theta"],
    )
    executor = BraketExecutor(LocalSimulator())
    bindings = {"theta": theta}
    job = executable.run(executor, bindings=bindings, estimation=estimation)
    forbidden = _forbid_submission_and_lookup(executor, monkeypatch)
    with monkeypatch.context() as before:
        before.setattr(BraketExecutionHandle, "result", forbidden)
        with pytest.raises(ValueError, match=r"call result\(\) successfully"):
            job.snapshot()
    original_result = job.result()
    monkeypatch.setattr(BraketExecutionHandle, "result", forbidden)

    saved = JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))
    restored = executable.restore(executor, saved, bindings)

    assert saved.executions == ()
    assert saved.execution is not None
    assert saved.execution.children[0].kind is ExecutionSnapshotKind.LOCAL
    assert type(saved.execution.children[0].value) is float
    assert type(restored) is type(job)
    assert type(restored.result()) is float
    assert restored.result() == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert restored.result() == pytest.approx(original_result, rel=0.0, abs=0.0)
    forbidden.assert_not_called()


@pytest.mark.parametrize("constant_first", [False, True])
def test_cached_local_expectation_and_constant_preserve_return_order(
    constant_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulator result and a constant shortcut retain their group boundaries.

    The rotated register has Z expectation cos(pi) = -1; 0.25 I gives 0.25
    independently of the state.
    """
    observables = [qm_o.Z(0), qm_o.Hamiltonian.identity(0.25)]
    expected = (-1.0, 0.25)
    if constant_first:
        observables.reverse()
        expected = expected[::-1]
    executable = BraketTranspiler().transpile(
        _expectation_pair,
        bindings=dict(zip(("first_observable", "second_observable"), observables)),
        parameters=["theta"],
    )
    executor = BraketExecutor(LocalSimulator())
    bindings = {"theta": math.pi}
    job = executable.run(executor, bindings=bindings)
    original_result = job.result()
    forbidden = _forbid_submission_and_lookup(executor, monkeypatch)
    monkeypatch.setattr(BraketExecutionHandle, "result", forbidden)

    saved = JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))
    restored = executable.restore(executor, saved, bindings)

    assert saved.executions == ()
    assert saved.execution is not None
    assert saved.execution.kind is ExecutionSnapshotKind.COMPOSITE
    assert len(saved.execution.children) == 2
    assert all(
        child.kind is ExecutionSnapshotKind.LOCAL for child in saved.execution.children
    )
    assert tuple(child.value for child in saved.execution.children) == pytest.approx(
        expected, rel=0.0, abs=1e-12
    )
    assert type(restored) is type(job)
    assert type(restored.result()) is tuple
    assert all(type(value) is float for value in restored.result())
    assert restored.result() == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert restored.result() == pytest.approx(original_result, rel=0.0, abs=0.0)
    forbidden.assert_not_called()


def test_uncached_snapshot_does_not_poll_load_or_wait_for_result_lock() -> None:
    """An uncached local handle rejects snapshots even while another caller waits."""
    forbidden = Mock(side_effect=AssertionError("Snapshot must not wait or load"))
    task = Mock(state=forbidden)
    handle = BraketExecutionHandle(
        tasks=(task,),
        result_loader=forbidden,
        decoder=forbidden,
        reference=None,
        native=task,
    )
    handle._result_lock = Mock(acquire=forbidden, __enter__=forbidden)

    with pytest.raises(ValueError, match=r"call result\(\) successfully"):
        handle.snapshot()

    forbidden.assert_not_called()


def test_remote_snapshot_keeps_reference_after_result_retrieval() -> None:
    """Retrieving an AWS result never switches a remote snapshot to local data."""
    task = Mock()
    task.state.return_value = "COMPLETED"
    loader = Mock(return_value=(object(),))
    reference = ExecutionReference(
        provider="amazon_braket",
        job_ids=("arn:aws:braket:us-east-1:123:quantum-task/saved",),
        context={"kind": "sample", "width": "1"},
    )
    handle = BraketExecutionHandle(
        tasks=(task,),
        result_loader=loader,
        decoder=lambda _: {"1": 17},
        reference=reference,
        native=task,
    )
    before = handle.snapshot()
    loader.assert_not_called()
    task.state.assert_not_called()
    assert handle.result() == {"1": 17}

    after = handle.snapshot()

    assert before == after
    assert after.kind is ExecutionSnapshotKind.REMOTE
    assert after.reference == reference
    assert after.value is None
    loader.assert_called_once()
