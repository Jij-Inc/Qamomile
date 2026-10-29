"""Verify lazy Nexus authentication and project selection without network calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import Mock

import pytest

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from qamomile.circuit.transpiler import ExecutionReference
from qamomile.hugr import HugrExecutor, NexusExecutionOptions
from qamomile.hugr._nexus import NexusTransport
from tests.hugr.test_nexus import _client, _package

pytestmark = pytest.mark.hugr


def _assert_project(client, project):
    """Check that the resolved native project reaches both provider operations."""
    assert client.hugr.upload.call_args.kwargs["project"] is project
    assert client.start_execute_job.call_args.kwargs["project"] is project


def _assert_context_unchanged(client):
    """Ensure explicit selection does not rewrite the SDK's active context."""
    client.context.set_active_project.assert_not_called()
    client.context.set_active_project_token.assert_not_called()


def test_public_executor_defers_session_setup_until_submission(monkeypatch):
    """Construct the public executor without authenticating or resolving projects."""
    client = _client()
    load = Mock(return_value=client)
    monkeypatch.setattr("qamomile.hugr._nexus.importlib.import_module", load)

    executor = HugrExecutor(
        "nexus", options=NexusExecutionOptions(max_cost=1.0, project="example")
    )

    load.assert_called_once_with("qnexus")
    client.login.assert_not_called()
    client.context.get_active_project.assert_not_called()
    client.projects.get_or_create.assert_not_called()
    client.hugr.upload.assert_not_called()
    executor.submit(_package(), 1)
    client.login.assert_called_once_with()
    client.projects.get_or_create.assert_called_once_with(name="example")


@pytest.mark.parametrize("project", ["", " ", "\t\n"])
def test_blank_project_names_fail_before_client_construction(monkeypatch, project):
    """Empty project names cannot trigger SDK loading or provider side effects."""
    load = Mock(side_effect=AssertionError("SDK must not be loaded"))
    monkeypatch.setattr("qamomile.hugr._nexus.importlib.import_module", load)
    with pytest.raises(ValueError, match="project"):
        NexusExecutionOptions(max_cost=1.0, project=project)
    load.assert_not_called()


def test_missing_active_project_creates_default_once_per_transport():
    """Use a named default project while checking the active context on every submit."""
    client = _client()
    project = client.projects.get_or_create.return_value
    transport = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client)

    transport.submit(_package(), 1)
    transport.submit(_package(), 1)

    client.login.assert_called_once_with()
    client.context.get_active_project.assert_any_call(project_required=False)
    assert client.context.get_active_project.call_count == 2
    client.projects.get_or_create.assert_called_once_with(name="qamomile")
    _assert_project(client, project)
    _assert_context_unchanged(client)


def test_active_project_changes_override_cached_default_project():
    """Respect context switches and reuse the fallback only when no project is active."""
    client = _client()
    first, second = object(), object()
    fallback = client.projects.get_or_create.return_value
    client.context.get_active_project.side_effect = [None, first, second, None]
    transport = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client)

    for expected in (fallback, first, second, fallback):
        transport.submit(_package(), 1)
        _assert_project(client, expected)

    client.login.assert_called_once_with()
    assert client.context.get_active_project.call_count == 4
    client.projects.get_or_create.assert_called_once_with(name="qamomile")
    _assert_context_unchanged(client)


def test_named_project_overrides_active_context_and_is_cached():
    """Resolve an explicit project name once without changing global SDK state."""
    client = _client()
    client.context.get_active_project.return_value = object()
    transport = NexusTransport(
        NexusExecutionOptions(max_cost=1.0, project="example"), client=client
    )

    transport.submit(_package(), 1)
    transport.submit(_package(), 1)

    client.login.assert_called_once_with()
    client.projects.get_or_create.assert_called_once_with(name="example")
    client.context.get_active_project.assert_not_called()
    _assert_project(client, client.projects.get_or_create.return_value)
    _assert_context_unchanged(client)


def test_native_project_reference_is_forwarded_without_resolution():
    """Preserve existing native project references and avoid unrelated project calls."""
    client = _client()
    project = object()
    transport = NexusTransport(
        NexusExecutionOptions(max_cost=1.0, project=project), client=client
    )

    transport.submit(_package(), 1)

    client.login.assert_called_once_with()
    client.context.get_active_project.assert_not_called()
    client.projects.get_or_create.assert_not_called()
    _assert_project(client, project)
    _assert_context_unchanged(client)


def test_execution_region_is_not_used_as_authentication_region():
    """Keep device scheduling region separate from the account login region."""
    client = _client()
    options = NexusExecutionOptions(max_cost=1.0, target_region="sg")

    NexusTransport(options, client=client).submit(_package(), 1)

    client.login.assert_called_once_with()
    assert client.start_execute_job.call_args.kwargs["target_region"] == "sg"


def test_login_project_upload_and_execution_happen_in_order():
    """Finish authentication and project selection before creating remote programs."""
    client = _client()
    events = Mock()
    events.attach_mock(client.login, "login")
    events.attach_mock(client.projects.get_or_create, "project")
    events.attach_mock(client.hugr.upload, "upload")
    events.attach_mock(client.start_execute_job, "execute")

    NexusTransport(
        NexusExecutionOptions(max_cost=1.0, project="example"), client=client
    ).submit(_package(), 1)

    assert [call[0] for call in events.mock_calls] == [
        "login",
        "project",
        "upload",
        "execute",
    ]


def test_failed_login_can_be_retried_without_partial_submission():
    """Cache successful authentication only and leave failed submissions untouched."""
    client = _client()
    client.login.side_effect = [ConnectionError("login unavailable"), None]
    transport = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client)

    with pytest.raises(ConnectionError, match="login unavailable"):
        transport.submit(_package(), 1)
    client.context.get_active_project.assert_not_called()
    client.projects.get_or_create.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()

    transport.submit(_package(), 1)
    assert client.login.call_count == 2
    client.projects.get_or_create.assert_called_once_with(name="qamomile")
    client.hugr.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


def test_failed_project_resolution_retries_without_reauthenticating():
    """Keep a successful login while retrying an unresolved project on the next call."""
    client = _client()
    project = object()
    client.projects.get_or_create.side_effect = [
        ConnectionError("project unavailable"),
        project,
    ]
    transport = NexusTransport(
        NexusExecutionOptions(max_cost=1.0, project="example"), client=client
    )

    with pytest.raises(ConnectionError, match="project unavailable"):
        transport.submit(_package(), 1)
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()

    transport.submit(_package(), 1)
    client.login.assert_called_once_with()
    assert client.projects.get_or_create.call_count == 2
    _assert_project(client, project)


def test_active_project_lookup_failure_does_not_create_a_fallback():
    """Do not mask a context lookup failure by silently choosing another project."""
    client = _client()
    client.context.get_active_project.side_effect = RuntimeError("context unavailable")

    with pytest.raises(RuntimeError, match="context unavailable"):
        NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client).submit(
            _package(), 1
        )

    client.projects.get_or_create.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_restoration_logs_in_without_resolving_or_creating_projects():
    """Restore existing jobs even when a configured project name cannot be resolved."""
    client = _client()
    client.projects.get_or_create.side_effect = AssertionError(
        "Must not create project"
    )
    client.context.get_active_project.side_effect = AssertionError(
        "Must not read project"
    )
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), context={"shots": "1"}
    )
    transport = NexusTransport(
        NexusExecutionOptions(max_cost=1.0, project="example"), client=client
    )

    restored = transport.retrieve(reference)
    transport.retrieve(reference)

    assert restored.native is client.jobs.get.return_value
    client.login.assert_called_once_with()
    assert client.jobs.get.call_count == 2
    client.context.get_active_project.assert_not_called()
    client.projects.get_or_create.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("program_format", ["hugr", "qir"])
def test_restored_job_requests_raw_only_for_helios(monkeypatch, program_format):
    """The stored format selects result version without rerunning an old job."""
    import builtins

    from qnexus.models.references import ResultVersions

    client = _client()
    context = {"shots": "1"}
    if program_format == "qir":
        context.update(program_format="qir", qir_outputs='{"result":["bool",1]}')
        monkeypatch.setattr(
            "qamomile.hugr._nexus.decode_qir_results",
            Mock(return_value=[{"result": True}]),
        )
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), context=context
    )
    original_import = builtins.__import__
    sdk_imports = []

    def observe_import(name, *args, **kwargs):
        if name.startswith("qnexus"):
            sdk_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe_import)
    transport = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client)
    restored = transport.retrieve(reference)
    assert sdk_imports == []
    assert restored.result() == [{"result": True}]
    assert restored.references() == (reference,)
    download = client.jobs.results.return_value[0].download_result
    if program_format == "hugr":
        download.assert_called_once_with(version=ResultVersions.RAW)
        assert sdk_imports == ["qnexus.models.references"]
    else:
        download.assert_called_once_with()
        assert sdk_imports == []
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_raw_download_failure_does_not_retry_default_format():
    """A RAW retrieval failure cannot silently erase namespace information."""
    from qnexus.models.references import ResultVersions

    client = _client()
    download = client.jobs.results.return_value[0].download_result
    download.side_effect = ConnectionError("RAW unavailable")
    handle = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client).submit(
        _package(), 1
    )
    with pytest.raises(ConnectionError, match="RAW unavailable"):
        handle.result()
    download.assert_called_once_with(version=ResultVersions.RAW)
    client.start_execute_job.assert_called_once()


def test_failed_restore_login_can_retry_without_project_side_effects():
    """A restoration login failure cannot create a new project or execution."""
    client = _client()
    client.login.side_effect = [ConnectionError("login unavailable"), None]
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), context={"shots": "1"}
    )
    transport = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client)

    with pytest.raises(ConnectionError, match="login unavailable"):
        transport.retrieve(reference)
    client.jobs.get.assert_not_called()
    transport.retrieve(reference)

    assert client.login.call_count == 2
    client.jobs.get.assert_called_once_with(id="existing-job")
    client.projects.get_or_create.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_submit_after_restore_reuses_login_and_resolves_project_on_demand():
    """Share authentication across lifecycle operations without eagerly making projects."""
    client = _client()
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), context={"shots": "1"}
    )
    transport = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client)
    transport.retrieve(reference)
    client.projects.get_or_create.assert_not_called()

    transport.submit(_package(), 1)

    client.login.assert_called_once_with()
    client.projects.get_or_create.assert_called_once_with(name="qamomile")


@pytest.mark.parametrize("project_name", [None, "example"])
def test_concurrent_submissions_share_successful_session_setup(project_name):
    """Concurrent callers do not duplicate login or named project creation."""
    client = _client()
    project = object()
    start = Barrier(4)
    pause = Event()
    client.login.side_effect = lambda: pause.wait(timeout=0.02)

    def resolve(**kwargs):
        """Expose overlapping setup attempts using a bounded local service delay."""
        pause.wait(timeout=0.02)
        return project

    def submit():
        """Start simultaneous submissions against one transport."""
        start.wait(timeout=5)
        return transport.submit(_package(), 1)

    client.projects.get_or_create.side_effect = resolve
    transport = NexusTransport(
        NexusExecutionOptions(max_cost=1.0, project=project_name), client=client
    )
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(submit) for _ in range(4)]
        assert len([future.result(timeout=5) for future in futures]) == 4

    client.login.assert_called_once_with()
    client.projects.get_or_create.assert_called_once_with(
        name=project_name or "qamomile"
    )
    assert client.hugr.upload.call_count == 4
    assert client.start_execute_job.call_count == 4
    assert all(
        call.kwargs["project"] is project for call in client.hugr.upload.call_args_list
    )
    _assert_context_unchanged(client)
