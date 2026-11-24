"""Exercise IBM dry-run submission and reports without provider access."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

import qamomile as qm

_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(angle: qm.Param[qm.Real]) -> qm.Shot[qm.Bit]:
    return qm.measure(qm.ry(qm.prepare(False), angle))
"""

_INDEXED_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(angle: qm.Param[qm.Real]) -> qm.Shot[qm.Bool]:
    state = qm.ry(qm.prepare(False), angle)
    if qm.get_current_shot() % 2 == 0:
        state = qm.x(state)
    measured = qm.measure(state)
    return measured ^ (qm.get_current_shot() == 0)
"""

_TEMPLATE_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(angle: qm.Param[qm.Real]) -> qm.Shot[qm.Bit]:
    rotation = angle + float(qm.get_current_shot() % 2)
    return qm.measure(qm.ry(qm.prepare(False), rotation))
"""


@pytest.fixture
def runtime_harness(monkeypatch):
    """Provide isolated Runtime transport and immutable endpoint instances."""
    return _runtime_harness(monkeypatch)


def _runtime_harness(monkeypatch):
    """Keep real circuit preparation but replace all IBM service operations."""
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit import IBMRuntime
    from qamomile.qiskit._integration import _CAPABILITIES

    submissions = []
    samplers = []
    restores = []
    jobs = {}
    context = SimpleNamespace(mode=None)

    class IBMBackend(BasicSimulator):
        """Identify the test backend as a provider backend without credentials."""

    class Session:
        """Represent an explicit caller-owned Runtime session."""

        def __init__(self, backend):
            self._backend = backend

        def backend(self):
            return self._backend

    class Batch(Session):
        """Represent the supported batch mode, which subclasses Session."""

    class RuntimeJobFailureError(Exception):
        """Represent a terminal provider failure independently of transport."""

    class RuntimeInvalidStateError(Exception):
        """Represent retrieval from a cancelled provider job."""

    class NativeJob:
        """Expose status and usage while making mock measurement data unusable."""

        def __init__(self, identity, dry_run):
            self.identity = identity
            self.inputs = {"dry_run": dry_run}
            self.state = "QUEUED"
            self.result_calls = 0
            self.error = None
            self.result_gate = None
            self.result_started = Event()
            self.usage_state = "complete"
            self.seconds = 1.5
            self.resource_units = 0.025

        def job_id(self):
            return self.identity

        def backend(self):
            return backend

        def status(self):
            return self.state

        def result(self):
            self.result_calls += 1
            self.result_started.set()
            if self.result_gate is not None:
                assert self.result_gate.wait(5), "test did not release result retrieval"
            if self.error is not None:
                raise self.error
            if self.state == "CANCELLED":
                raise RuntimeInvalidStateError("Job was cancelled")
            self.state = "DONE"
            return object()

        def cancel(self):
            self.state = "CANCELLED"

        def metrics(self):
            return {"usage": {"status": self.usage_state}}

        def usage(self, *, resource_units=False):
            return self.resource_units if resource_units else self.seconds

        def error_message(self):
            return str(self.error) if self.error else None

    class Sampler:
        """Capture exact circuits, parameter rows, shots, and dry-run selection."""

        def __init__(self, *, mode, options):
            samplers.append(self)
            self.mode = mode if isinstance(mode, Session) else None
            self._backend = mode.backend() if self.mode is not None else mode

        def backend(self):
            return self._backend

        def run(self, pubs, *, shots, dry_run=False):
            native = NativeJob(f"dry-run-job-{len(submissions)}", dry_run)
            submissions.append((pubs, shots, dry_run))
            jobs[native.job_id()] = native
            return native

    def restore(identity):
        restores.append(identity)
        return jobs[identity]

    backend = IBMBackend()
    module = SimpleNamespace(
        __version__="0.51.0",
        IBMBackend=IBMBackend,
        SamplerV2=Sampler,
        Session=Session,
        Batch=Batch,
        RuntimeJobFailureError=RuntimeJobFailureError,
        RuntimeInvalidStateError=RuntimeInvalidStateError,
    )
    module.exceptions = SimpleNamespace(
        RuntimeJobFailureError=RuntimeJobFailureError,
        RuntimeInvalidStateError=RuntimeInvalidStateError,
    )
    monkeypatch.setattr(runtime, "_runtime_module", lambda: module)
    monkeypatch.setattr(runtime, "_runtime_context_mode", lambda: context.mode)

    def endpoint(*, dry_run=True, **kwargs):
        return IBMRuntime(
            backend,
            dry_run=dry_run,
            service=SimpleNamespace(job=restore),
            requirements=_CAPABILITIES,
            **kwargs,
        )

    return SimpleNamespace(
        backend=backend,
        module=module,
        endpoint=endpoint,
        submissions=submissions,
        samplers=samplers,
        restores=restores,
        jobs=jobs,
        context=context,
    )


def _compile(endpoint, source=_SOURCE):
    return qm.compile(qm.build_source(source).specialize("sample"), to=endpoint.target)


@pytest.mark.parametrize("route", ["ordinary", "template", "indexed"])
def test_dry_run_forwards_the_same_prepared_workload(runtime_harness, route):
    """Dry-run changes submission intent without changing circuits or shot demand."""
    import numpy as np

    harness = runtime_harness
    dry = harness.endpoint()
    real = harness.endpoint(dry_run=False)
    source = {
        "ordinary": _SOURCE,
        "template": _TEMPLATE_SOURCE,
        "indexed": _INDEXED_SOURCE,
    }[route]
    artifact = _compile(dry, source)
    assert dry.target.fingerprint == real.target.fingerprint
    dry_job = qm.submit(artifact, on=dry, parameters={"angle": 0.25}, shots=3)
    qm.submit(artifact, on=real, parameters={"angle": 0.25}, shots=3)

    dry_pubs, dry_shots, dry_flag = harness.submissions[0]
    real_pubs, real_shots, real_flag = harness.submissions[1]
    assert dry_flag is True and real_flag is False
    assert dry_shots == real_shots == (3 if route == "ordinary" else 1)
    assert len(dry_pubs) == len(real_pubs) == (3 if route == "indexed" else 1)
    if route == "template":
        assert dry_pubs[0][0] == real_pubs[0][0]
        np.testing.assert_array_equal(
            dry_pubs[0][1].as_array(), real_pubs[0][1].as_array()
        )
    else:
        assert dry_pubs == real_pubs
    native = harness.jobs[dry_job.job_id]
    assert native.result_calls == 0
    report = dry_job.result()
    assert isinstance(report, qm.ValidationResult)
    assert report.status is qm.JobStatus.COMPLETED
    assert report.metadata()["shots"] == 3
    assert report.metadata()["dry_run"] is True
    assert not hasattr(report, "counts") and not hasattr(report, "value")
    assert native.result_calls == 1


@pytest.mark.parametrize("route", ["run_source", "run_artifact", "execute", "submit"])
def test_dry_run_public_lifecycle_returns_provider_report(runtime_harness, route):
    harness = runtime_harness
    endpoint = harness.endpoint()
    program = qm.build_source(_SOURCE).specialize("sample")
    artifact = qm.compile(program, to=endpoint.target)
    arguments = {"on": endpoint, "parameters": {"angle": 0.5}, "shots": 2}
    if route.startswith("run"):
        record = qm.run(program if route == "run_source" else artifact, **arguments)
        report = record.result
        assert record.invocation.shots == 2
    elif route == "execute":
        report = qm.execute(artifact, **arguments)
    else:
        job = qm.submit(artifact, **arguments)
        assert job.status is qm.JobStatus.SUBMITTED
        assert harness.jobs[job.job_id].result_calls == 0
        report = job.result()
    assert isinstance(report, qm.ValidationResult)
    assert report.metadata()["usage"] == {
        "estimated_quantum_seconds": 1.5,
        "estimated_resource_units": 0.025,
    }
    assert len(harness.submissions) == 1


def test_dry_run_usage_distinguishes_pending_from_final_zero(runtime_harness):
    harness = runtime_harness
    endpoint = harness.endpoint()
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    native = harness.jobs[job.job_id]
    native.state = "DONE"
    native.usage_state = "pending"
    native.seconds = native.resource_units = 0.0
    assert job.metadata()["usage"] == {
        "estimated_quantum_seconds": None,
        "estimated_resource_units": None,
    }
    assert native.result_calls == 0
    native.usage_state = "complete"
    report = job.result()
    assert report.metadata()["usage"] == {
        "estimated_quantum_seconds": 0.0,
        "estimated_resource_units": 0.0,
    }


def test_dry_run_usage_can_finish_after_the_job_result(runtime_harness):
    endpoint = runtime_harness.endpoint()
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    native = runtime_harness.jobs[job.job_id]
    native.usage_state = "pending"
    report = job.result()
    assert report.metadata()["usage"]["estimated_quantum_seconds"] is None
    native.usage_state = "complete"
    assert job.metadata()["usage"]["estimated_quantum_seconds"] == 1.5
    assert report.metadata()["usage"]["estimated_quantum_seconds"] is None
    assert native.result_calls == 1


def test_dry_run_preserves_seconds_when_sdk_has_no_resource_unit_option(
    runtime_harness,
):
    endpoint = runtime_harness.endpoint()
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    runtime_harness.jobs[job.job_id].usage = lambda: 2.5
    assert job.result().metadata()["usage"] == {
        "estimated_quantum_seconds": 2.5,
        "estimated_resource_units": None,
    }


@pytest.mark.parametrize("usage", [-1.0, float("nan"), float("inf"), True])
def test_dry_run_rejects_invalid_provider_usage(runtime_harness, usage):
    endpoint = runtime_harness.endpoint()
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    runtime_harness.jobs[job.job_id].seconds = usage
    with pytest.raises(qm.ExecutionError, match="usage"):
        job.result()


@pytest.mark.parametrize("route", ["ordinary", "template", "indexed"])
def test_dry_run_restores_remote_and_completed_reports(runtime_harness, route):
    harness = runtime_harness
    dry = harness.endpoint()
    real = harness.endpoint(dry_run=False)
    source = {
        "ordinary": _SOURCE,
        "template": _TEMPLATE_SOURCE,
        "indexed": _INDEXED_SOURCE,
    }[route]
    artifact = _compile(dry, source)
    parameters = {"angle": 0.25}
    job = qm.submit(artifact, on=dry, parameters=parameters, shots=2)
    remote = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    assert remote.reference.context_values["dry_run"] == "true"
    with pytest.raises(ValueError, match="dry.run|validation"):
        real.restore(remote, artifact=artifact, parameters=parameters)
    assert not harness.restores
    restored = dry.restore(remote, artifact=artifact, parameters=parameters)
    report = restored.result()
    assert isinstance(report, qm.ValidationResult)
    local = qm.ExecutionSnapshot.from_bytes(restored.snapshot().to_bytes())
    assert local.result == report
    assert dry.restore(local, artifact=artifact).result() == report
    with pytest.raises(ValueError):
        real.restore(local, artifact=artifact)
    assert harness.restores == [job.job_id]
    assert len(harness.submissions) == 1


def test_dry_run_cannot_restore_a_real_job(runtime_harness):
    harness = runtime_harness
    dry = harness.endpoint()
    real = harness.endpoint(dry_run=False)
    artifact = _compile(real)
    job = qm.submit(artifact, on=real, parameters={"angle": 0.0}, shots=2)
    with pytest.raises(ValueError, match="dry.run|validation"):
        dry.restore(job.snapshot(), artifact=artifact, parameters={"angle": 0.0})
    assert not harness.restores


@pytest.mark.parametrize("marker", ["false", "True", "", "unknown"])
def test_dry_run_rejects_malformed_snapshot_mode_before_restoration(
    runtime_harness, marker
):
    harness = runtime_harness
    endpoint = harness.endpoint()
    artifact = _compile(endpoint)
    job = qm.submit(artifact, on=endpoint, parameters={"angle": 0.0}, shots=2)
    reference = job.snapshot().reference
    altered = qm.ExecutionReference.create(
        provider=reference.provider,
        job_ids=reference.job_ids,
        target=reference.target,
        context={**reference.context_values, "dry_run": marker},
    )
    with pytest.raises(ValueError, match="dry.run"):
        endpoint.restore(altered, artifact=artifact, parameters={"angle": 0.0})
    assert not harness.restores


@pytest.mark.parametrize("selected", ["explicit", "implicit"])
def test_dry_run_rejects_session_before_submission(
    runtime_harness, monkeypatch, selected
):
    harness = runtime_harness
    session = harness.module.Session(harness.backend)
    if selected == "implicit":
        sampler = harness.module.SamplerV2

        class ImplicitSessionSampler(sampler):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.mode = session

        monkeypatch.setattr(harness.module, "SamplerV2", ImplicitSessionSampler)
    with pytest.raises((TypeError, ValueError, qm.ExecutionError), match="[Ss]ession"):
        harness.endpoint(mode=session if selected == "explicit" else None)
    assert not harness.submissions


def test_dry_run_accepts_batch_mode(runtime_harness):
    harness = runtime_harness
    endpoint = harness.endpoint(mode=harness.module.Batch(harness.backend))
    report = qm.execute(
        _compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2
    )
    assert isinstance(report, qm.ValidationResult)
    assert len(harness.submissions) == 1


@pytest.mark.parametrize("route", ["ordinary", "template", "indexed"])
@pytest.mark.parametrize("mode_name", ["Session", "Batch"])
def test_dry_run_checks_context_selected_after_endpoint_creation(
    runtime_harness, route, mode_name
):
    """A new Runtime Executor must not pick up an unsupported late Session."""
    harness = runtime_harness
    endpoint = harness.endpoint()
    source = {
        "ordinary": _SOURCE,
        "template": _TEMPLATE_SOURCE,
        "indexed": _INDEXED_SOURCE,
    }[route]
    artifact = _compile(endpoint, source)
    assert endpoint._sampler.mode is None
    harness.context.mode = getattr(harness.module, mode_name)(harness.backend)
    if mode_name == "Session":
        with pytest.raises(ValueError, match="Session"):
            qm.submit(artifact, on=endpoint, parameters={"angle": 0.0}, shots=2)
        assert not harness.submissions
        return
    report = qm.execute(artifact, on=endpoint, parameters={"angle": 0.0}, shots=2)
    assert isinstance(report, qm.ValidationResult)
    assert len(harness.submissions) == 1
    assert harness.submissions[0][2] is True


def test_dry_run_keeps_explicit_batch_inside_unrelated_session_context(runtime_harness):
    """An explicit Batch remains authoritative when ambient context changes."""
    harness = runtime_harness
    batch = harness.module.Batch(harness.backend)
    endpoint = harness.endpoint(mode=batch)
    artifact = _compile(endpoint)
    harness.context.mode = harness.module.Session(harness.backend)
    assert endpoint._sampler.mode is batch
    report = qm.execute(artifact, on=endpoint, parameters={"angle": 0.0}, shots=2)
    assert isinstance(report, qm.ValidationResult)
    assert len(harness.submissions) == 1


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("seed", [0, 7])
def test_runtime_batch_seed_policy_fails_before_any_submission(
    runtime_harness, dry_run, seed
):
    """Every explicit seed is rejected before any batch job is submitted."""
    endpoint = runtime_harness.endpoint(dry_run=dry_run)
    with pytest.raises(qm.ExecutionError, match="seed"):
        qm.execute_batch(
            _compile(endpoint),
            on=endpoint,
            parameters=[{"angle": 0.0}, {"angle": 0.5}],
            shots=2,
            seed=seed,
        )
    assert not runtime_harness.submissions


def test_dry_run_batch_accepts_unspecified_seed(runtime_harness):
    """An unseeded batch returns one validation report per parameter row."""
    endpoint = runtime_harness.endpoint()
    reports = qm.execute_batch(
        _compile(endpoint),
        on=endpoint,
        parameters=[{"angle": 0.0}, {"angle": 0.5}],
        shots=2,
    )
    assert len(reports) == 2
    assert all(isinstance(report, qm.ValidationResult) for report in reports)
    assert all(report.status is qm.JobStatus.COMPLETED for report in reports)
    assert len(runtime_harness.submissions) == 2
    assert all(dry_run for _, _, dry_run in runtime_harness.submissions)


@pytest.mark.parametrize("dry_run", [False, True])
def test_runtime_provider_timeout_is_not_a_local_wait_timeout(runtime_harness, dry_run):
    """A provider retrieval failure remains terminal instead of a local wait expiry."""
    harness = runtime_harness
    endpoint = harness.endpoint(dry_run=dry_run)
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    native = harness.jobs[job.job_id]
    native.state = "DONE"
    native.error = TimeoutError("provider result retrieval timed out")
    for _ in range(2):
        with pytest.raises(qm.ExecutionError, match="Q8227.*provider result retrieval"):
            job.result()
    assert job.status is qm.JobStatus.FAILED
    assert native.result_calls == 1
    assert len(harness.submissions) == 1


def test_dry_run_rejects_unsupported_sampler_without_submission(
    runtime_harness, monkeypatch
):
    harness = runtime_harness

    class OlderSampler(harness.module.SamplerV2):
        def run(self, pubs, *, shots):
            raise AssertionError("unsupported SDK must not submit")

    monkeypatch.setattr(harness.module, "SamplerV2", OlderSampler)
    with pytest.raises(
        (TypeError, ValueError, qm.ExecutionError), match="dry.run|0.51"
    ):
        harness.endpoint()
    assert not harness.submissions


def test_dry_run_rejects_local_backend(runtime_harness):
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import IBMRuntime

    with pytest.raises((TypeError, ValueError, qm.ExecutionError), match="IBM|backend"):
        IBMRuntime(BasicSimulator(), dry_run=True)
    assert not runtime_harness.submissions


@pytest.mark.parametrize("dry_run", [False, True])
def test_cloud_mock_backend_requires_explicit_dry_run(
    runtime_harness, monkeypatch, dry_run
):
    """Cloud mock output cannot become ordinary program measurements."""
    harness = runtime_harness
    monkeypatch.setattr(harness.backend, "is_mock", True, raising=False)
    if not dry_run:
        with pytest.raises(ValueError, match="dry_run=True"):
            harness.endpoint(dry_run=False)
        assert not harness.samplers
        assert not harness.submissions
        return
    endpoint = harness.endpoint(dry_run=True)
    report = qm.execute(
        _compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2
    )
    assert isinstance(report, qm.ValidationResult)
    assert report.metadata()["dry_run"] is True
    assert len(harness.submissions) == 1
    assert harness.submissions[0][2] is True


@pytest.mark.parametrize("dry_run", [None, 0, 1, "true"])
def test_dry_run_requires_a_boolean(runtime_harness, dry_run):
    with pytest.raises(TypeError, match="dry_run"):
        runtime_harness.endpoint(dry_run=dry_run)
    assert not runtime_harness.submissions


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "transport"])
def test_dry_run_distinguishes_terminal_reports_from_transport_failure(
    runtime_harness, outcome
):
    harness = runtime_harness
    endpoint = harness.endpoint()
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    native = harness.jobs[job.job_id]
    if outcome == "cancelled":
        job.cancel()
        expected = qm.JobStatus.CANCELLED
    elif outcome == "failure":
        native.state = "ERROR"
        native.error = harness.module.RuntimeJobFailureError("Invalid instruction")
        expected = qm.JobStatus.FAILED
    else:
        native.error = OSError("Service unavailable")
        with pytest.raises(qm.ExecutionError, match="Service unavailable"):
            job.result()
        return
    report = job.result()
    assert isinstance(report, qm.ValidationResult)
    assert report.status is expected
    assert job.status is expected


def test_dry_run_timeout_preserves_the_original_submission(runtime_harness):
    harness = runtime_harness
    endpoint = harness.endpoint()
    job = qm.submit(_compile(endpoint), on=endpoint, parameters={"angle": 0.0}, shots=2)
    native = harness.jobs[job.job_id]
    gate = native.result_gate = Event()
    try:
        with pytest.raises(TimeoutError):
            job.result(timeout=0.01)
        assert native.result_started.wait(1)
        assert job.snapshot().kind is qm.ExecutionSnapshotKind.REMOTE
    finally:
        gate.set()
    assert isinstance(job.result(timeout=2), qm.ValidationResult)
    assert native.result_calls == len(harness.submissions) == 1


def test_validation_endpoint_rejects_expectation_before_submission(runtime_harness):
    endpoint = runtime_harness.endpoint()
    source = qm.build_source("""
import qamomile as qm

@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    return qm.qubits(1)
""").specialize("state")
    with pytest.raises(qm.ExecutionError, match="validation|dry.run"):
        qm.expectation(source, qm.Z(0), on=endpoint, shots=2)
    assert not runtime_harness.submissions


def test_validation_endpoint_rejects_measurement_driven_plan_before_submission(
    runtime_harness,
):
    from qamomile._shot_execution import ShotExecutable
    from qamomile.effects import Requirement

    from .test_shot_execution import program

    endpoint = runtime_harness.endpoint()
    endpoint = replace(
        endpoint, requirements=endpoint.requirements - {Requirement.DYNAMIC_BRANCH}
    )
    artifact = qm.compile(program("independent"), to=endpoint.target)
    assert isinstance(artifact.payload, ShotExecutable)
    with pytest.raises(qm.ExecutionError, match="validation|dry.run"):
        qm.submit(artifact, on=endpoint, shots=2)
    assert not runtime_harness.submissions


def test_dry_run_rejects_a_template_without_provider_work(runtime_harness):
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = runtime_harness.endpoint()
    artifact = _compile(
        endpoint,
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() == 0
""",
    )
    assert artifact.execution_payload(QiskitExecutable).shot_template is not None
    with pytest.raises(qm.ExecutionError, match="dry.run|quantum|provider"):
        qm.submit(artifact, on=endpoint, shots=2)
    assert not runtime_harness.submissions
