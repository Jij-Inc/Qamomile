"""Tests for native Amazon Braket task lifecycle integration."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.braket
pytest.importorskip("braket.circuits")

from braket.circuits import Circuit, FreeParameter  # noqa: E402

import qamomile.observable as qm_o  # noqa: E402
from qamomile.braket import (  # noqa: E402
    BraketExecutionOptions,
    BraketExecutor,
)
from qamomile.circuit.transpiler.errors import ExecutionError  # noqa: E402
from qamomile.circuit.transpiler.execution_handle import (  # noqa: E402
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_request import (  # noqa: E402
    CircuitInvocation,
    EstimateRequest,
    SampleRequest,
    ShotBased,
)
from qamomile.circuit.transpiler.parameter_binding import (  # noqa: E402
    ParameterInfo,
    ParameterMetadata,
)


class _Result:
    """Provide the small gate-model result surface used by the executor."""

    def __init__(
        self,
        *,
        counts: dict[str, int] | None = None,
        values: list[float] | None = None,
    ) -> None:
        """Initialize a fake result.

        Args:
            counts (dict[str, int] | None): Measurement counts.
            values (list[float] | None): Expectation result values.
        """
        self.measurement_counts = counts
        self.measured_qubits = [0]
        self.values = values


class _Task:
    """Record retrieval and cancellation of one fake Braket task."""

    def __init__(self, task_id: str, result: _Result, state: str = "COMPLETED") -> None:
        """Initialize a fake task.

        Args:
            task_id (str): Provider task identifier.
            result (_Result): Raw task result.
            state (str): Initial provider state.
        """
        self.id = task_id
        self._result = result
        self._state = state
        self.result_calls = 0
        self.cancel_calls = 0

    def state(self) -> str:
        """Return the current fake state.

        Returns:
            str: Braket-like state string.
        """
        return self._state

    def result(self) -> _Result:
        """Return and record raw-result retrieval.

        Returns:
            _Result: Stored raw result.
        """
        self.result_calls += 1
        return self._result

    def cancel(self) -> None:
        """Record cancellation and transition to cancelled."""
        self.cancel_calls += 1
        self._state = "CANCELLED"


class _Device:
    """Record task and batch submissions made by the Braket executor."""

    arn = "arn:aws:braket:us-east-1::device/quantum-simulator/test"

    def __init__(self, tasks: list[_Task]) -> None:
        """Initialize a fake device.

        Args:
            tasks (list[_Task]): Tasks returned in submission order.
        """
        self.tasks = tasks
        self.run_calls: list[tuple[Any, int, dict[str, Any]]] = []
        self.batch_calls: list[tuple[Any, int, dict[str, Any]]] = []
        self.batch: _Batch | None = None

    def run(self, circuit: Any, shots: int, **kwargs: Any) -> _Task:
        """Record and return one task.

        Args:
            circuit (Any): Submitted task specification.
            shots (int): Submitted shots.
            **kwargs (Any): Native Braket options.

        Returns:
            _Task: Next configured task.
        """
        self.run_calls.append((circuit, shots, kwargs))
        return self.tasks[len(self.run_calls) - 1]

    def run_batch(self, circuits: Any, shots: int, **kwargs: Any) -> "_Batch":
        """Record and return a task batch.

        Args:
            circuits (Any): Submitted task specifications.
            shots (int): Submitted shots.
            **kwargs (Any): Native batch options.

        Returns:
            _Batch: Fake native batch.
        """
        self.batch_calls.append((circuits, shots, kwargs))
        self.batch = _Batch(self.tasks)
        return self.batch


class _Batch:
    """Expose child tasks and record SDK batch-result retry arguments."""

    id = "batch-1"

    def __init__(self, tasks: list[_Task]) -> None:
        """Initialize a fake batch.

        Args:
            tasks (list[_Task]): Ordered child tasks.
        """
        self.tasks = tasks
        self.results_calls: list[dict[str, Any]] = []

    def results(self, **kwargs: Any) -> list[_Result]:
        """Record batch retrieval and return child results.

        Args:
            **kwargs (Any): SDK retry arguments.

        Returns:
            list[_Result]: Ordered child results.
        """
        self.results_calls.append(kwargs)
        return [task._result for task in self.tasks]


def test_sample_submission_is_lazy_and_uses_native_inputs() -> None:
    """Braket free parameters remain unbound and travel through ``inputs``."""
    theta = FreeParameter("theta")
    task = _Task(
        "arn:aws:braket:us-east-1:123:quantum-task/sample",
        _Result(counts={"1": 4}),
    )
    device = _Device([task])
    metadata = ParameterMetadata(
        parameters=[ParameterInfo("theta", "theta", None, theta)]
    )
    request = SampleRequest(
        CircuitInvocation(Circuit().rx(0, theta), {"theta": 0.5}, metadata),
        shots=4,
    )

    handle = BraketExecutor(device).submit_sample(request)

    assert task.result_calls == 0
    assert device.run_calls[0][2]["inputs"] == {"theta": 0.5}
    assert handle.status() is JobStatus.COMPLETED
    assert handle.references()[0].job_ids == (task.id,)
    assert handle.native is task
    assert handle.result() == {"1": 4}
    assert task.result_calls == 1


def test_sampling_handle_delegates_cancellation() -> None:
    """Cancelling a handle requests native task cancellation."""
    task = _Task("local-task", _Result(counts={"0": 1}), state="RUNNING")
    handle = BraketExecutor(_Device([task])).submit_sample(
        SampleRequest(CircuitInvocation(Circuit().i(0), {}, ParameterMetadata()), 1)
    )

    handle.cancel()

    assert task.cancel_calls == 1
    assert handle.status() is JobStatus.CANCELLED
    assert handle.references() == ()


def test_failed_task_error_is_cached_without_resubmission() -> None:
    """Repeated result access cannot trigger another provider recovery path."""
    task = _Task("failed-task", _Result(counts=None), state="FAILED")
    handle = BraketExecutor(_Device([task])).submit_sample(
        SampleRequest(CircuitInvocation(Circuit().i(0), {}, ParameterMetadata()), 1)
    )

    with pytest.raises(ExecutionError) as first:
        handle.result()
    with pytest.raises(ExecutionError) as second:
        handle.result()

    assert second.value is first.value
    assert task.result_calls == 0


def test_batch_result_retrieval_has_no_implicit_retry() -> None:
    """Default shot estimation retrieves child tasks without batch retry."""
    tasks = [
        _Task("local-1", _Result(values=[1.0])),
        _Task("local-2", _Result(values=[-0.5])),
    ]
    device = _Device(tasks)
    request = EstimateRequest(
        CircuitInvocation(Circuit().h(0), {}, ParameterMetadata()),
        qm_o.X(0) + 2.0 * qm_o.Z(0),
        ShotBased(100),
    )

    handle = BraketExecutor(device).submit_estimate(request)

    assert all(task.result_calls == 0 for task in tasks)
    assert handle.result() == pytest.approx(0.0)
    assert all(task.result_calls == 1 for task in tasks)
    assert device.batch is not None
    assert device.batch.results_calls == []


def test_batch_retry_is_explicit_and_bounded() -> None:
    """Opted-in batch retries pass a concrete maximum to the SDK."""
    tasks = [
        _Task("local-1", _Result(values=[1.0]), state="FAILED"),
        _Task("local-2", _Result(values=[1.0])),
    ]
    device = _Device(tasks)
    executor = BraketExecutor(
        device,
        options=BraketExecutionOptions(batch_max_retries=2),
    )
    request = EstimateRequest(
        CircuitInvocation(Circuit().h(0), {}, ParameterMetadata()),
        qm_o.X(0) + qm_o.Z(0),
        ShotBased(100),
    )

    assert executor.submit_estimate(request).result() == pytest.approx(2.0)
    assert device.batch is not None
    assert device.batch.results_calls == [{"fail_unsuccessful": True, "max_retries": 2}]
    assert all(task.result_calls == 0 for task in tasks)


def test_restore_reconstructs_sampling_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A secret-free AWS reference restores its task and result decoder."""
    import braket.aws

    task_id = "arn:aws:braket:us-east-1:123:quantum-task/restored"
    task = _Task(task_id, _Result(counts={"1": 2}))
    device = _Device([])
    device.aws_session = object()
    constructor_calls: list[tuple[str, dict[str, Any]]] = []

    def task_factory(arn: str, **kwargs: Any) -> _Task:
        """Record task restoration arguments.

        Args:
            arn (str): Restored task ARN.
            **kwargs (Any): AWS task constructor options.

        Returns:
            _Task: Restored fake task.
        """
        constructor_calls.append((arn, kwargs))
        return task

    monkeypatch.setattr(braket.aws, "AwsQuantumTask", task_factory)
    reference = ExecutionReference(
        provider="amazon_braket",
        job_ids=(task_id,),
        context={"kind": "sample", "width": "1"},
    )

    handle = BraketExecutor(device).restore(reference)

    assert constructor_calls == [(task_id, {"aws_session": device.aws_session})]
    assert handle.result() == {"1": 2}
    assert handle.references() == (reference,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_max_retries": -1},
        {"batch_max_retries": True},
        {"batch_max_retries": 1.5},
        {"max_parallel": 0},
        {"max_parallel": 1.5},
        {"poll_interval_seconds": float("nan")},
        {"s3_destination_folder": ("", "prefix")},
        {"task_options": {"shots": 100}},
        {"batch_options": {"inputs": []}},
    ],
)
def test_execution_options_reject_ambiguous_values(kwargs: dict[str, Any]) -> None:
    """Executor-owned and invalid options fail before submission."""
    with pytest.raises(ValueError):
        BraketExecutionOptions(**kwargs)


def test_configured_poll_timeout_bounds_default_result_wait() -> None:
    """The execution option bounds result waits without an explicit timeout."""
    task = _Task("running-task", _Result(counts={"0": 1}), state="RUNNING")
    options = BraketExecutionOptions(
        poll_timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )
    handle = BraketExecutor(_Device([task]), options=options).submit_sample(
        SampleRequest(CircuitInvocation(Circuit().i(0), {}, ParameterMetadata()), 1)
    )

    with pytest.raises(TimeoutError, match="timed out"):
        handle.result()
