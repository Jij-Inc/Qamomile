"""Keep provider validation reports distinct from source execution results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

import qamomile as qm
from qamomile.compilation import CompletedJob, ExecutionResult

from .conftest import BELL_SOURCE


class _ResultEndpoint:
    """Return a controlled result through the ordinary endpoint contract."""

    name = "test.validation"
    target = qm.Reference()
    capabilities = target.capabilities

    def __init__(self, result: ExecutionResult, *, validation_only: bool) -> None:
        self.returned_result = result
        self.validation_only = validation_only
        self.invocations: list[qm.Invocation] = []

    def launch(self, artifact: qm.CompiledProgram, invocation: qm.Invocation) -> qm.Job:
        self.invocations.append(invocation)
        return CompletedJob(f"test-job-{len(self.invocations)}", self.returned_result)


class _BatchResultEndpoint(_ResultEndpoint):
    """Exercise the direct batch endpoint path as well as sequential launches."""

    def launch_batch(
        self, artifact: qm.CompiledProgram, invocations: tuple[qm.Invocation, ...]
    ) -> tuple[qm.Job, ...]:
        return tuple(self.launch(artifact, invocation) for invocation in invocations)


def _report(status: qm.JobStatus = qm.JobStatus.COMPLETED) -> qm.ValidationResult:
    """Construct a terminal provider report without measurement data."""
    return qm.ValidationResult.from_metadata(
        backend="test.validation",
        status=status,
        metadata={
            "provider_status": status.value.upper(),
            "estimated_quantum_seconds": None,
        },
    )


@pytest.mark.parametrize(
    "status", [qm.JobStatus.COMPLETED, qm.JobStatus.FAILED, qm.JobStatus.CANCELLED]
)
def test_validation_result_retains_terminal_status_without_fabricated_values(status):
    report = _report(status)

    assert report.status is status
    assert report.metadata()["status"] == status.value
    assert report.metadata()["estimated_quantum_seconds"] is None
    assert not hasattr(report, "counts")
    assert not hasattr(report, "value")
    with pytest.raises(FrozenInstanceError):
        report.backend = "changed"


@pytest.mark.parametrize(
    "status", [qm.JobStatus.COMPLETED, qm.JobStatus.FAILED, qm.JobStatus.CANCELLED]
)
def test_completed_validation_job_preserves_its_reported_status(status):
    """Public job state and saved result retain a terminal validation outcome."""
    report = _report(status)
    endpoint = _ResultEndpoint(report, validation_only=True)
    artifact = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"), to=endpoint.target
    )
    job = qm.submit(artifact, on=endpoint, shots=3)

    assert job.status is status
    assert job.result() is report
    restored = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes()).result
    assert isinstance(restored, qm.ValidationResult)
    assert restored.status is status


@pytest.mark.parametrize("status", [qm.JobStatus.SUBMITTED, qm.JobStatus.RUNNING])
def test_validation_result_rejects_a_pending_provider_report(status):
    with pytest.raises((qm.ExecutionError, ValueError)):
        _report(status)


def test_validation_result_can_be_constructed_without_provider_diagnostics():
    report = qm.ValidationResult("test.validation", qm.JobStatus.COMPLETED)

    assert report.metadata() == {"status": "completed"}


@pytest.mark.parametrize(
    "metadata", [None, [], {1: "invalid key"}, {"opaque": object()}]
)
def test_validation_result_rejects_nonportable_provider_metadata(metadata):
    with pytest.raises(TypeError):
        qm.ValidationResult.from_metadata(
            backend="test.validation", status=qm.JobStatus.COMPLETED, metadata=metadata
        )


def test_validation_result_owns_metadata_and_uses_authoritative_status():
    provider = {
        "status": "RUNNING",
        "provider_status": "COMPLETED",
        "estimated_quantum_seconds": 0.0,
        "details": {"messages": ["checked"]},
    }
    report = qm.ValidationResult.from_metadata(
        backend="test.validation", status=qm.JobStatus.COMPLETED, metadata=provider
    )
    provider["provider_status"] = "ERROR"
    provider["details"]["messages"].append("changed")

    first = report.metadata()
    assert first["status"] == "completed"
    assert first["provider_status"] == "COMPLETED"
    assert first["estimated_quantum_seconds"] == 0.0
    assert first["details"]["messages"] == ("checked",)
    assert isinstance(first, dict)
    first["provider_status"] = "CANCELLED"
    assert report.metadata()["provider_status"] == "COMPLETED"


@pytest.mark.parametrize(
    "route", ["run_source", "run_artifact", "execute", "execute_batch", "native_batch"]
)
def test_public_lifecycle_returns_a_validation_report_for_a_validation_endpoint(route):
    report = _report()
    endpoint_type = _BatchResultEndpoint if route == "native_batch" else _ResultEndpoint
    endpoint = endpoint_type(report, validation_only=True)
    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=endpoint.target)

    if route.startswith("run"):
        record = qm.run(
            program if route == "run_source" else artifact, on=endpoint, shots=3
        )
        results = (record.result,)
        assert record.invocation.shots == 3
        assert (
            record.resources.to_bytes() == record.artifact.report.resources.to_bytes()
        )
        with pytest.raises(KeyError):
            record.resources.metric("qamomile.observed_execution.shots")
    elif route == "execute":
        results = (qm.execute(artifact, on=endpoint, shots=3),)
    else:
        results = qm.execute_batch(artifact, on=endpoint, parameters=[{}, {}], shots=3)

    assert all(isinstance(result, qm.ValidationResult) for result in results)
    assert all(result == report for result in results)
    assert len(endpoint.invocations) == len(results)
    assert all(invocation.shots == 3 for invocation in endpoint.invocations)


@pytest.mark.parametrize("sampled", [True, False])
def test_execution_endpoint_cannot_replace_its_typed_result_with_a_validation_report(
    sampled,
):
    endpoint = _ResultEndpoint(_report(), validation_only=False)
    source = (
        BELL_SOURCE
        if sampled
        else """
import qamomile as qm

@qm.qkernel(kind="classical")
def answer() -> int:
    return 42
"""
    )
    program = qm.build_source(source).specialize("bell" if sampled else "answer")
    artifact = qm.compile(program, to=endpoint.target)

    with pytest.raises(qm.ExecutionError):
        qm.execute(artifact, on=endpoint, shots=3)


@pytest.mark.parametrize("kind", ["sample", "value"])
def test_validation_endpoint_cannot_return_a_source_program_result(kind):
    returned = (
        qm.SampleResult.from_counts(backend="test", counts={(0, 0): 3})
        if kind == "sample"
        else qm.ValueResult("test", 42)
    )
    endpoint = _ResultEndpoint(returned, validation_only=True)
    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=endpoint.target)

    with pytest.raises(qm.ExecutionError):
        qm.execute(artifact, on=endpoint, shots=3)


def test_batch_cannot_change_its_expected_result_kind_after_submission():
    """A batch response cannot retroactively authorize a validation result."""

    class ChangingBatchEndpoint(_BatchResultEndpoint):
        def launch_batch(self, artifact, invocations):
            self.validation_only = True
            return super().launch_batch(artifact, invocations)

    endpoint = ChangingBatchEndpoint(_report(), validation_only=False)
    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=endpoint.target)

    with pytest.raises(qm.ExecutionError, match="Q8151"):
        qm.execute_batch(artifact, on=endpoint, parameters=[{}], shots=3)

    assert len(endpoint.invocations) == 1
    assert endpoint.validation_only is True


@pytest.mark.parametrize("batched", [False, True], ids=["sequential", "native-batch"])
@pytest.mark.parametrize(
    "validation_only", [False, True], ids=["execution", "validation"]
)
@pytest.mark.parametrize("phase", ["preflight", "between_rows"])
def test_batch_preserves_its_initial_result_kind_across_endpoint_changes(
    batched, validation_only, phase
):
    """Preflight and earlier rows cannot change the remaining result contract."""
    sample = qm.SampleResult.from_counts(backend="test", counts={(0, 0): 3})
    report = _report()
    initial_result = report if validation_only else sample
    changed_result = sample if validation_only else report
    endpoint_base = _BatchResultEndpoint if batched else _ResultEndpoint

    class ChangingEndpoint(endpoint_base):
        def validate_invocation(self, invocation):
            if phase == "preflight":
                self.validation_only = not validation_only

        def launch(self, artifact, invocation):
            job = super().launch(artifact, invocation)
            if phase == "between_rows" and len(self.invocations) == 1:
                self.validation_only = not validation_only
                self.returned_result = changed_result
            return job

    endpoint = ChangingEndpoint(
        changed_result if phase == "preflight" else initial_result,
        validation_only=validation_only,
    )
    artifact = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"), to=endpoint.target
    )
    parameters = [{}] if phase == "preflight" else [{}, {}]

    with pytest.raises(qm.ExecutionError, match="Q8151"):
        qm.execute_batch(artifact, on=endpoint, parameters=parameters, shots=3)

    assert len(endpoint.invocations) == len(parameters)


def test_optimization_run_cannot_pair_a_validation_report_with_prior_samples():
    from qamomile.optimization import BinaryModel, OptimizationRun, QAOAExperiment

    experiment = QAOAExperiment(BinaryModel.from_hubo({(0,): 1.0}), layers=0)
    prior = experiment.run(on=qm.Reference(), gammas=(), betas=(), shots=2)
    validation_record = replace(prior.execution, result=_report())

    with pytest.raises(TypeError, match="(?i)sampl"):
        OptimizationRun(prior.identity, validation_record, prior.samples)
