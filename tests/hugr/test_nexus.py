"""Verify Nexus SDK submission and execution lifecycle without remote jobs."""

from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr.qsystem.result import QsysResult

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_handle import ExecutionReference, JobStatus
from qamomile.hugr._nexus import NexusExecutionOptions, NexusTransport

pytestmark = pytest.mark.hugr


def _package():
    """Represent a serialized HUGR package without compiling a test fixture.

    Returns:
        SimpleNamespace: Package-shaped object exposing deterministic bytes.
    """
    return SimpleNamespace(to_bytes=lambda: b"test-hugr-package")


def _client(records=None, status="COMPLETED"):
    """Build a SDK-shaped client that never creates network connections.

    Args:
        records (list | None): Tagged shot records, or one default Boolean shot.
        status (str): Initial provider status, defaulting to completed.

    Returns:
        SimpleNamespace: Nexus-shaped client whose service calls are mocks.
    """
    native = SimpleNamespace(id="test-job-id", job_type="execute")
    downloaded = QsysResult(records if records is not None else [[("result", True)]])
    return SimpleNamespace(
        HeliosConfig=Mock(
            side_effect=lambda **kwargs: SimpleNamespace(type="HeliosConfig", **kwargs)
        ),
        hugr=SimpleNamespace(upload=Mock(return_value="uploaded-hugr")),
        start_execute_job=Mock(return_value=native),
        jobs=SimpleNamespace(
            get=Mock(return_value=native),
            cancel=Mock(),
            status=Mock(
                return_value=SimpleNamespace(status=status, queue_position=4, cost=1.25)
            ),
            results=Mock(
                return_value=[
                    SimpleNamespace(download_result=Mock(return_value=downloaded))
                ]
            ),
        ),
    )


def test_submission_is_nonblocking_and_forwards_native_settings():
    """Submit one tagged program without waiting, losing config, or double execution."""
    client = _client()
    project = object()
    config = SimpleNamespace(type="HeliosConfig", system_name="Helios-1SC")
    options = NexusExecutionOptions(
        project=project,
        backend_config=config,
        max_cost=2.5,
        n_qubits=4,
        credential_name="existing-credential",
        user_group="research",
        target_region="us",
    )
    package = _package()
    handle = NexusTransport(options, client=client).submit(package, 12)
    upload = client.hugr.upload.call_args.kwargs
    assert upload["hugr_package"] is package
    assert upload["project"] is project
    submitted = client.start_execute_job.call_args.kwargs
    assert submitted == {
        "programs": ["uploaded-hugr"],
        "n_shots": [12],
        "backend_config": config,
        "name": upload["name"],
        "project": project,
        "valid_check": True,
        "max_cost": 2.5,
        "n_qubits": 4,
        "credential_name": "existing-credential",
        "user_group": "research",
        "target_region": "us",
    }
    client.jobs.status.assert_not_called()
    client.jobs.results.assert_not_called()
    assert handle.native is client.start_execute_job.return_value
    reference = handle.references()[0]
    assert reference.provider == "quantinuum-nexus"
    assert reference.target == "Helios-1SC"
    assert reference.context == {
        "shots": "12",
        "package_sha256": hashlib.sha256(package.to_bytes()).hexdigest(),
    }
    assert "credential" not in repr(reference.to_dict())


def test_default_config_and_unique_submission_names():
    """Construct the documented Helios config and avoid Nexus name collisions."""
    client = _client()
    transport = NexusTransport(client=client)
    transport.submit(_package(), 1)
    transport.submit(_package(), 1)
    assert client.HeliosConfig.call_args.kwargs == {"system_name": "Helios-1"}
    names = [call.kwargs["name"] for call in client.start_execute_job.call_args_list]
    assert names[0] != names[1]


def test_typed_results_preserve_arrays_scalars_and_shot_order():
    """Decode QsysResult without interpreting values as bitstrings or losing types."""
    records = [
        [("bits", [True, False]), ("integer", 12), ("fraction", 0.375)],
        [("bits", [False, True]), ("integer", 3), ("fraction", -0.125)],
    ]
    client = _client(records)
    handle = NexusTransport(client=client).submit(_package(), 2)
    assert handle.result() == [dict(shot) for shot in records]
    assert handle.result() == [dict(shot) for shot in records]
    client.jobs.results.assert_called_once_with(handle.native, allow_incomplete=False)
    assert handle.status() == JobStatus.COMPLETED
    assert handle.metadata()["cost"] == 1.25


@pytest.mark.parametrize(
    ("provider_status", "status"),
    [
        ("SUBMITTED", JobStatus.PENDING),
        ("RETRYING", JobStatus.PENDING),
        ("QUEUED", JobStatus.QUEUED),
        ("RUNNING", JobStatus.RUNNING),
        ("COMPLETED", JobStatus.COMPLETED),
        ("CANCELLING", JobStatus.CANCELLING),
        ("CANCELLED", JobStatus.CANCELLED),
        ("ERROR", JobStatus.FAILED),
        ("TERMINATED", JobStatus.FAILED),
        ("DEPLETED", JobStatus.FAILED),
        ("FUTURE_STATE", JobStatus.UNKNOWN),
    ],
)
def test_provider_status_mapping(provider_status, status):
    """Preserve lifecycle distinctions for every documented Nexus state."""
    client = _client(status=provider_status)
    handle = NexusTransport(client=client).submit(_package(), 1)
    assert handle.status() == status
    assert handle.raw_status().status == provider_status


def test_timeout_leaves_remote_job_active_and_can_be_retried():
    """A local timeout neither cancels nor resubmits the existing execution."""
    client = _client(status="RUNNING")
    handle = NexusTransport(client=client).submit(_package(), 1)
    with pytest.raises(TimeoutError):
        handle.result(timeout=0)
    client.jobs.cancel.assert_not_called()
    client.jobs.results.assert_not_called()
    client.jobs.status.return_value = SimpleNamespace(status="COMPLETED")
    assert handle.result(timeout=0) == [{"result": True}]
    client.start_execute_job.assert_called_once()


def test_configured_default_timeout_is_used():
    """Respect a configured zero wait limit while preserving explicit overrides."""
    client = _client(status="QUEUED")
    handle = NexusTransport(
        NexusExecutionOptions(timeout_seconds=0), client=client
    ).submit(_package(), 1)
    with pytest.raises(TimeoutError):
        handle.result()


@pytest.mark.parametrize(
    "provider_status", ["ERROR", "CANCELLED", "DEPLETED", "TERMINATED"]
)
def test_unsuccessful_jobs_do_not_return_partial_results_or_retry(provider_status):
    """Do not report a completed sample from failed or cancelled job items."""
    client = _client(status=provider_status)
    handle = NexusTransport(client=client).submit(_package(), 1)
    with pytest.raises(ExecutionError, match=provider_status):
        handle.result()
    with pytest.raises(ExecutionError, match=provider_status):
        handle.result()
    client.jobs.results.assert_not_called()
    client.start_execute_job.assert_called_once()
    expected = (
        JobStatus.CANCELLED if provider_status == "CANCELLED" else JobStatus.FAILED
    )
    assert handle.status() == expected


def test_cancel_uses_provider_request_without_claiming_immediate_completion():
    """Observe asynchronous cancellation through a later status call."""
    client = _client(status="RUNNING")
    handle = NexusTransport(client=client).submit(_package(), 1)
    handle.cancel()
    client.jobs.cancel.assert_called_once_with(handle.native)
    assert handle.status() == JobStatus.RUNNING
    client.jobs.status.return_value = SimpleNamespace(status="CANCELLING")
    assert handle.status() == JobStatus.CANCELLING


def test_roundtrip_reference_restores_job_without_submission():
    """Restore saved job identifiers and fetch their original output."""
    client = _client()
    original = NexusTransport(client=client).submit(_package(), 1)
    reference = ExecutionReference.from_dict(original.references()[0].to_dict())
    restored = NexusTransport(client=client).retrieve(reference)
    client.jobs.get.assert_called_once_with(id="test-job-id")
    assert restored.result() == [{"result": True}]
    client.hugr.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


@pytest.mark.parametrize(
    "reference",
    [
        ExecutionReference("other", ("id",), context={"shots": "1"}),
        ExecutionReference("quantinuum-nexus", ("id", "id2"), context={"shots": "1"}),
        ExecutionReference("quantinuum-nexus", ("id",)),
        ExecutionReference("quantinuum-nexus", ("id",), context={"shots": "no"}),
        ExecutionReference("quantinuum-nexus", ("id",), context={"shots": "0"}),
    ],
)
def test_invalid_reference_rejected_before_provider_lookup(reference):
    """Reject foreign or incomplete restoration references locally."""
    client = _client()
    with pytest.raises(ValueError):
        NexusTransport(client=client).retrieve(reference)
    client.jobs.get.assert_not_called()


def test_restore_rejects_compilation_job():
    """A Nexus compilation reference cannot masquerade as an execution."""
    client = _client()
    client.jobs.get.return_value = SimpleNamespace(job_type="compile")
    reference = ExecutionReference("quantinuum-nexus", ("id",), context={"shots": "1"})
    with pytest.raises(ValueError, match="execute job"):
        NexusTransport(client=client).retrieve(reference)


@pytest.mark.parametrize(
    ("records", "shots", "message"),
    [
        ([], 1, "returned 0 shots"),
        ([[("result", 1)]], 2, "expected 2"),
        ([[("result", 1)], [("result", 0)]], 1, "expected 1"),
        ([[("result", 1), ("result", 0)]], 1, "exactly one value"),
    ],
)
def test_incomplete_or_ambiguous_result_records_fail(records, shots, message):
    """Reject missing shots and duplicate tags instead of silently corrupting output."""
    client = _client(records)
    handle = NexusTransport(client=client).submit(_package(), shots)
    with pytest.raises(ExecutionError, match=message):
        handle.result()
    assert handle.status() == JobStatus.FAILED


@pytest.mark.parametrize("count", [0, 2])
def test_result_count_must_match_one_program(count):
    """Reject absent or extra result references from a one-program submission."""
    client = _client()
    client.jobs.results.return_value *= count
    handle = NexusTransport(client=client).submit(_package(), 1)
    with pytest.raises(ExecutionError, match="exactly one result"):
        handle.result()


@pytest.mark.parametrize("raw", [None, object(), SimpleNamespace(results="invalid")])
def test_non_hugr_results_are_rejected(raw):
    """Do not reinterpret unrelated SDK result formats as HUGR output."""
    client = _client()
    client.jobs.results.return_value[0].download_result.return_value = raw
    handle = NexusTransport(client=client).submit(_package(), 1)
    with pytest.raises(ExecutionError, match="QsysResult"):
        handle.result()


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_invalid_shots_do_not_upload(shots):
    """Validate requested shots before remote state can change."""
    client = _client()
    with pytest.raises(ValueError, match="shots"):
        NexusTransport(client=client).submit(_package(), shots)
    client.hugr.upload.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"poll_interval_seconds": 0},
        {"poll_interval_seconds": float("nan")},
        {"timeout_seconds": -1},
        {"timeout_seconds": True},
        {"max_cost": float("inf")},
        {"n_qubits": 0},
        {"n_qubits": True},
        {"target_region": "invalid"},
        {"system_name": ""},
        {"name": ""},
        {"backend_config": object()},
    ],
)
def test_invalid_options_fail_locally(changes):
    """Reject invalid limits and targets during options construction."""
    with pytest.raises(ValueError):
        NexusExecutionOptions(**changes)


def test_nexus_import_is_optional(monkeypatch):
    """Only constructing the remote transport requires the optional SDK."""

    def missing(name):
        """Simulate an unavailable optional SDK.

        Args:
            name (str): Requested module name.

        Raises:
            ImportError: Always, to represent the missing dependency.
        """
        raise ImportError(name)

    monkeypatch.setattr("qamomile.hugr._nexus.importlib.import_module", missing)
    NexusExecutionOptions()
    with pytest.raises(ImportError, match=r"qamomile\[hugr\]"):
        NexusTransport()


@pytest.mark.ci_smoke
def test_installed_sdk_contract_without_network():
    """Autospec the supported SDK so argument names and result conversion stay real."""
    qnx = pytest.importorskip("qnexus")
    from qnexus.models.job_status import JobStatus as NexusStatus, JobStatusEnum
    from qnexus.models.references import ExecuteJobRef, ExecutionResultRef

    client = _client()
    client.HeliosConfig = qnx.HeliosConfig
    client.hugr.upload = create_autospec(qnx.hugr.upload, return_value="program")
    native = Mock(spec=ExecuteJobRef)
    native.id = "sdk-spec-job"
    native.job_type = "execute"
    native.last_status_detail = None
    client.start_execute_job = create_autospec(
        qnx.start_execute_job, return_value=native
    )
    client.jobs.status = create_autospec(
        qnx.jobs.status, return_value=NexusStatus(JobStatusEnum.COMPLETED)
    )
    ref = create_autospec(ExecutionResultRef, instance=True)
    ref.download_result.return_value = QsysResult(
        [[("number", 5), ("bits", [True, False])]]
    )
    client.jobs.results = create_autospec(qnx.jobs.results, return_value=[ref])
    client.jobs.cancel = create_autospec(qnx.jobs.cancel)
    client.jobs.get = create_autospec(qnx.jobs.get, return_value=native)
    transport = NexusTransport(client=client)
    handle = transport.submit(_package(), 1)
    assert handle.result() == [{"number": 5, "bits": [True, False]}]
    assert transport.retrieve(handle.references()[0]).native is native
    handle.cancel()
    parameters = inspect.signature(qnx.start_execute_job).parameters
    assert "inputs" not in parameters
    assert "arguments" not in parameters


def test_polling_progresses_to_completion_without_resubmission(monkeypatch):
    """Poll one job through queued and running states before loading its result."""
    client = _client()
    client.jobs.status.side_effect = [
        SimpleNamespace(status="QUEUED"),
        SimpleNamespace(status="RUNNING"),
        SimpleNamespace(status="COMPLETED"),
    ]
    pauses = []
    monkeypatch.setattr("qamomile.hugr._nexus.time.sleep", pauses.append)
    handle = NexusTransport(client=client).submit(_package(), 1)
    assert handle.result() == [{"result": True}]
    assert len(pauses) == 2
    client.start_execute_job.assert_called_once()


def test_uncertain_submission_failure_never_retries():
    """Do not duplicate billable work if the initial execute response is lost."""
    client = _client()
    client.start_execute_job.side_effect = ConnectionError("lost response")
    with pytest.raises(ConnectionError, match="lost response"):
        NexusTransport(client=client).submit(_package(), 1)
    client.start_execute_job.assert_called_once()
    client.jobs.results.assert_not_called()


def test_transient_retrieval_failure_can_retry_existing_job():
    """Retry result retrieval, preserving a remote job after a network failure."""
    client = _client()
    client.jobs.status.side_effect = [
        ConnectionError("temporarily unavailable"),
        SimpleNamespace(status="COMPLETED"),
    ]
    handle = NexusTransport(client=client).submit(_package(), 1)
    with pytest.raises(ConnectionError):
        handle.result()
    assert handle.result() == [{"result": True}]
    client.start_execute_job.assert_called_once()


def test_result_lock_timeout_does_not_cancel_remote_execution():
    """Bound concurrent callers' local waits without touching the active job."""
    client = _client()
    handle = NexusTransport(client=client).submit(_package(), 1)
    handle._lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            handle.result(timeout=0)
    finally:
        handle._lock.release()
    client.jobs.cancel.assert_not_called()
    assert handle.result() == [{"result": True}]


def test_raw_user_tags_match_service_normalized_output():
    """Handle SDK results both before and after service prefix normalization."""
    client = _client(
        [[("USER:BOOL:bit", True), ("USER:FLOAT:angle", 0.25), ("USER:UINT:count", 2)]]
    )
    result = NexusTransport(client=client).submit(_package(), 1).result()
    assert result == [{"bit": True, "angle": 0.25, "count": 2}]


def test_prefix_normalization_cannot_overwrite_an_existing_tag():
    """Reject a raw tag colliding with a service-normalized output."""
    client = _client([[("USER:BOOL:bit", True), ("bit", False)]])
    handle = NexusTransport(client=client).submit(_package(), 1)
    with pytest.raises(ExecutionError, match="ambiguous"):
        handle.result()


@pytest.mark.parametrize("capacity", [1, np.int64(2), np.uint32(4)])
def test_integral_capacity_reaches_the_real_sdk_http_boundary(monkeypatch, capacity):
    """Normalize accepted integer scalars before the real SDK serializes a job."""
    qnx = pytest.importorskip("qnexus")
    client = _client()
    client.HeliosConfig = qnx.HeliosConfig
    client.hugr.upload.return_value = SimpleNamespace(id="uploaded-program")
    client.start_execute_job = qnx.start_execute_job
    http = Mock()
    http.post.side_effect = ConnectionError("mock execution boundary")
    monkeypatch.setattr("qnexus.client.jobs._execute.get_nexus_client", lambda: http)
    options = NexusExecutionOptions(
        project=SimpleNamespace(id="project"), n_qubits=capacity
    )

    with pytest.raises(ConnectionError, match="mock execution boundary"):
        NexusTransport(options, client=client).submit(_package(), 1)

    assert type(options.n_qubits) is int
    assert options.n_qubits == int(capacity)
    client.hugr.upload.assert_called_once()
    http.post.assert_called_once()
    assert http.post.call_args.args == ("/api/jobs/v1beta3",)
    definition = http.post.call_args.kwargs["json"]["data"]["attributes"]["definition"]
    assert definition["items"] == [
        {"program_id": "uploaded-program", "n_shots": 1, "n_qubits": int(capacity)}
    ]
    assert type(definition["items"][0]["n_qubits"]) is int
