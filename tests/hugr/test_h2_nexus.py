"""Verify H2 routing and restoration through the shared Nexus executor."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr.qsystem.result import QsysResult

from qamomile.circuit.transpiler import ExecutionError, ExecutionReference, JobStatus
from qamomile.hugr import HugrExecutor, NexusExecutionOptions
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._shot_range import SHOT_SUCCESS_TAG

pytestmark = pytest.mark.hugr

_BITCODE = b"BC\xc0\xde-test-qir"
_OUTPUTS = {"left": ("bool", 1), "right": ("bool", 1)}


def _package():
    """Provide deterministic original HUGR bytes without compiler setup."""
    return SimpleNamespace(modules=(), to_bytes=lambda: b"original-hugr-package")


def _client():
    """Provide mock SDK operations that cannot connect to a remote service."""

    def download_result(version=None):
        """Keep H2 fixtures default while serving prefixed RAW Helios records."""
        if version is None:
            return QsysResult([[("left", True)]])
        from qnexus.models.references import ResultVersions

        assert version is ResultVersions.RAW
        return QsysResult([[("USER:BOOL:left", True)]])

    native = SimpleNamespace(id="h2-job-id", job_type="execute")
    return SimpleNamespace(
        login=Mock(),
        context=SimpleNamespace(
            get_active_project=Mock(return_value=None),
            set_active_project=Mock(),
            set_active_project_token=Mock(),
        ),
        projects=SimpleNamespace(
            get_or_create=Mock(return_value=SimpleNamespace(id="default-project"))
        ),
        HeliosConfig=Mock(
            side_effect=lambda **kwargs: SimpleNamespace(type="HeliosConfig", **kwargs)
        ),
        QuantinuumConfig=Mock(
            side_effect=lambda **kwargs: SimpleNamespace(
                type="QuantinuumConfig", **kwargs
            )
        ),
        hugr=SimpleNamespace(upload=Mock(return_value="uploaded-hugr")),
        qir=SimpleNamespace(upload=Mock(return_value="uploaded-qir")),
        start_execute_job=Mock(return_value=native),
        jobs=SimpleNamespace(
            get=Mock(return_value=native),
            cancel=Mock(),
            status=Mock(return_value=SimpleNamespace(status="COMPLETED")),
            results=Mock(
                return_value=[
                    SimpleNamespace(download_result=Mock(wraps=download_result))
                ]
            ),
        ),
    )


@pytest.fixture
def converter(monkeypatch):
    """Isolate SDK routing tests from the separately tested conversion tool."""
    convert = Mock(return_value=(_BITCODE, _OUTPUTS))
    monkeypatch.setattr("qamomile.hugr._nexus.to_h2_qir", convert)
    monkeypatch.setattr(
        "qamomile.hugr._nexus.normalize_h2_shots",
        Mock(side_effect=lambda package, shots, origin: (package, False)),
    )
    monkeypatch.setattr(
        "qamomile.hugr._nexus.prepare_h2_shot_range",
        Mock(side_effect=lambda package, shots, origin: (package, False)),
    )
    return convert


@pytest.mark.parametrize("target", ["nexus", "helios"])
@pytest.mark.parametrize("system_name", ["Helios-1E", "H2-1E"])
def test_same_executor_selects_format_from_device(
    monkeypatch, converter, target, system_name
):
    """Keep both public destination names while selecting the actual device format."""
    client = _client()
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor(
        target, options=NexusExecutionOptions(system_name=system_name, max_cost=1.0)
    )
    package = _package()
    handle = executor.submit(package, shots=2)
    assert executor.capabilities.supports_restoration
    assert handle.references()[0].target == system_name
    if system_name.startswith("H2-"):
        client.QuantinuumConfig.assert_called_once_with(device_name=system_name)
        client.qir.upload.assert_called_once()
        client.hugr.upload.assert_not_called()
        client.HeliosConfig.assert_not_called()
        converter.assert_called_once_with(package)
    else:
        client.HeliosConfig.assert_called_once_with(system_name=system_name)
        client.hugr.upload.assert_called_once()
        client.qir.upload.assert_not_called()
        client.QuantinuumConfig.assert_not_called()
        converter.assert_not_called()
    client.start_execute_job.assert_called_once()
    client.jobs.status.assert_not_called()


@pytest.mark.parametrize(
    ("config", "system_name", "program_format"),
    [
        (
            SimpleNamespace(type="QuantinuumConfig", device_name="H2-1SC"),
            "Helios-1E",
            "qir",
        ),
        (
            SimpleNamespace(type="HeliosConfig", system_name="Helios-1SC"),
            "H2-1E",
            "hugr",
        ),
    ],
)
def test_native_config_overrides_device_name(
    converter, config, system_name, program_format
):
    """Select both device and program format from an explicitly supplied config."""
    client = _client()
    options = NexusExecutionOptions(
        system_name=system_name, backend_config=config, max_cost=1.0
    )
    handle = NexusTransport(options, client=client).submit(_package(), 1)
    assert client.start_execute_job.call_args.kwargs["backend_config"] is config
    assert handle.references()[0].target == (
        config.device_name if program_format == "qir" else config.system_name
    )
    assert client.qir.upload.call_count == int(program_format == "qir")
    assert client.hugr.upload.call_count == int(program_format == "hugr")
    assert converter.call_count == int(program_format == "qir")
    client.HeliosConfig.assert_not_called()
    client.QuantinuumConfig.assert_not_called()


def test_h2_submission_preserves_settings_names_and_original_fingerprint(converter):
    """Upload bitcode once per job and retain enough original metadata for restoration."""
    client = _client()
    project = object()
    package = _package()
    original_bytes = package.to_bytes()
    options = NexusExecutionOptions(
        project=project,
        system_name="H2-1E",
        name="h2-sample",
        max_cost=2.5,
        n_qubits=4,
        credential_name="existing-credential",
        user_group="research",
        target_region="us",
    )
    transport = NexusTransport(options, client=client)
    handles = [transport.submit(package, 3), transport.submit(package, 3)]
    names = []
    for upload_call, execute_call in zip(
        client.qir.upload.call_args_list,
        client.start_execute_job.call_args_list,
        strict=True,
    ):
        upload = upload_call.kwargs
        submitted = execute_call.kwargs
        names.append(upload["name"])
        assert upload == {
            "qir": _BITCODE,
            "name": submitted["name"],
            "project": project,
        }
        assert submitted == {
            "programs": ["uploaded-qir"],
            "n_shots": [3],
            "backend_config": submitted["backend_config"],
            "name": upload["name"],
            "project": project,
            "valid_check": True,
            "max_cost": 2.5,
            "n_qubits": 4,
            "credential_name": "existing-credential",
            "user_group": "research",
            "target_region": "us",
        }
        assert submitted["backend_config"].type == "QuantinuumConfig"
        assert submitted["backend_config"].device_name == "H2-1E"
    assert names[0] != names[1]
    assert all(name.startswith("h2-sample-") for name in names)
    assert package.to_bytes() == original_bytes
    for handle in handles:
        reference = handle.references()[0]
        assert reference.context["program_format"] == "qir"
        assert reference.context["shots"] == "3"
        assert (
            reference.context["package_sha256"]
            == hashlib.sha256(original_bytes).hexdigest()
        )
        assert json.loads(reference.context["qir_outputs"]) == {
            "left": ["bool", 1],
            "right": ["bool", 1],
        }
        assert "credential" not in repr(reference.to_dict())
    client.hugr.upload.assert_not_called()
    client.jobs.status.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [ImportError("converter unavailable"), ExecutionError("unsupported program")],
)
def test_conversion_failure_has_no_remote_submission(converter, error):
    """Fail before either upload operation when H2 conversion cannot succeed."""
    client = _client()
    converter.side_effect = error
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    with pytest.raises(type(error), match=str(error)):
        transport.submit(_package(), 1)
    client.login.assert_not_called()
    client.context.get_active_project.assert_not_called()
    client.projects.get_or_create.assert_not_called()
    client.qir.upload.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_h2_conversion_precedes_authentication_and_project_resolution(converter):
    """Complete local conversion before authentication or any remote state changes."""
    client = _client()
    events = Mock()
    events.attach_mock(converter, "convert")
    events.attach_mock(client.login, "login")
    events.attach_mock(client.projects.get_or_create, "project")
    events.attach_mock(client.qir.upload, "upload")
    events.attach_mock(client.start_execute_job, "execute")
    options = NexusExecutionOptions(project="example", system_name="H2-1E")

    NexusTransport(options, client=client).submit(_package(), 1)

    assert [call[0] for call in events.mock_calls] == [
        "convert",
        "login",
        "project",
        "upload",
        "execute",
    ]
    project = client.projects.get_or_create.return_value
    assert client.qir.upload.call_args.kwargs["project"] is project
    assert client.start_execute_job.call_args.kwargs["project"] is project


@pytest.mark.parametrize(
    "context",
    [
        {"program_format": "other"},
        {"program_format": "qir"},
        {"program_format": "qir", "qir_outputs": "not-json"},
        {"program_format": "qir", "qir_outputs": "[]"},
        {"program_format": "qir", "qir_outputs": '{"bit":["bool",64]}'},
        {"program_format": "qir", "qir_outputs": '{"count":["uint",0]}'},
        {"program_format": "qir", "qir_outputs": '{"value":["float",64]}'},
    ],
)
def test_invalid_format_or_manifest_is_rejected_before_lookup(context):
    """Do not contact Nexus for an unusable saved QIR execution reference."""
    client = _client()
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), context={"shots": "1", **context}
    )
    with pytest.raises(ValueError):
        NexusTransport(client=client).retrieve(reference)
    client.login.assert_not_called()
    client.projects.get_or_create.assert_not_called()
    client.jobs.get.assert_not_called()
    client.qir.upload.assert_not_called()
    client.hugr.upload.assert_not_called()


def test_h2_reference_restores_without_conversion_or_resubmission(
    monkeypatch, converter
):
    """Restore the stored result format independently of the transport's current device."""
    client = _client()
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    original = transport.submit(_package(), 1)
    reference = ExecutionReference.from_dict(original.references()[0].to_dict())
    converter.reset_mock()
    raw = object()
    client.jobs.results.return_value[0].download_result.return_value = raw
    decode = Mock(return_value=[{"left": True, "right": False}])
    monkeypatch.setattr("qamomile.hugr._nexus.decode_qir_results", decode)
    restored = NexusTransport(client=client).retrieve(reference)
    assert restored.result() == [{"left": True, "right": False}]
    decode.assert_called_once_with(raw, 1, _OUTPUTS)
    restored.cancel()
    client.jobs.get.assert_called_once_with(id="h2-job-id")
    client.jobs.cancel.assert_called_once_with(restored.native)
    converter.assert_not_called()
    client.qir.upload.assert_called_once()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_called_once()


def test_helios_never_imports_optional_converter(monkeypatch):
    """Keep direct HUGR execution usable without installing the H2 converter."""
    client = _client()
    imports = Mock(side_effect=AssertionError("unexpected optional dependency import"))
    monkeypatch.setattr(
        "qamomile.hugr._qir.importlib", SimpleNamespace(import_module=imports)
    )
    handle = NexusTransport(NexusExecutionOptions(max_cost=1.0), client=client).submit(
        _package(), 1
    )
    assert handle.result() == [{"left": True}]
    imports.assert_not_called()
    client.hugr.upload.assert_called_once()
    client.qir.upload.assert_not_called()


@pytest.mark.ci_smoke
def test_installed_sdk_h2_upload_contract_without_network(converter):
    """Bind QIR submission to real SDK argument names and the actual H2 config."""
    qnx = pytest.importorskip("qnexus")
    from qnexus.models.references import ExecuteJobRef

    client = _client()
    client.login = create_autospec(qnx.login)
    client.context.get_active_project = create_autospec(
        qnx.context.get_active_project, return_value=None
    )
    client.projects.get_or_create = create_autospec(
        qnx.projects.get_or_create, return_value=SimpleNamespace(id="sdk-project")
    )
    client.QuantinuumConfig = qnx.QuantinuumConfig
    client.qir.upload = create_autospec(qnx.qir.upload, return_value="sdk-qir")
    native = Mock(spec=ExecuteJobRef)
    native.id = "sdk-h2-job"
    native.job_type = "execute"
    native.last_status_detail = None
    client.start_execute_job = create_autospec(
        qnx.start_execute_job, return_value=native
    )
    client.jobs.get = create_autospec(qnx.jobs.get, return_value=native)
    options = NexusExecutionOptions(system_name="H2-1E")
    transport = NexusTransport(options, client=client)
    handle = transport.submit(_package(), 2)
    assert (
        client.start_execute_job.call_args.kwargs["backend_config"].device_name
        == "H2-1E"
    )
    assert client.qir.upload.call_args.kwargs["qir"] == _BITCODE
    assert transport.retrieve(handle.references()[0]).native is native
    client.login.assert_called_once_with()
    client.projects.get_or_create.assert_called_once_with(name="qamomile")
    client.hugr.upload.assert_not_called()


def _guarded_submission(monkeypatch, converter, client, *, shots=2):
    from qamomile.hugr._qir import _H2AbortError

    package = _package()
    guarded = SimpleNamespace(to_bytes=lambda: b"guarded-hugr-package")
    prepare = Mock(return_value=(guarded, True))
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    converter.side_effect = [
        _H2AbortError("converter rejected abort"),
        (_BITCODE, {**_OUTPUTS, SHOT_SUCCESS_TAG: ("bool", 1)}),
    ]
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    handle = transport.submit(package, shots)
    prepare.assert_called_once_with(package, shots, 1)
    assert converter.call_count == 2
    assert converter.call_args.args == (guarded,)
    return handle, guarded


def _guarded_readouts(client, success, *, tag=SHOT_SUCCESS_TAG):
    bits = [
        SimpleNamespace(reg_name=label, index=[0]) for label in ("left", tag, "right")
    ]
    raw = SimpleNamespace(
        get_bitlist=Mock(return_value=bits),
        get_shots=Mock(return_value=[[1, value, 0] for value in success]),
    )
    client.jobs.results.return_value[0].download_result.return_value = raw
    return raw


def _saved_guard_reference(*, shot_origin=1, tag=SHOT_SUCCESS_TAG):
    """Serialize the required success marker and saved shot origin."""
    context = {
        "shots": "2",
        "program_format": "qir",
        "qir_outputs": json.dumps({**_OUTPUTS, tag: ("bool", 1)}),
    }
    context.update(shot_success_tag=tag, shot_origin=str(shot_origin))
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), target="H2-1E", context=context
    )
    return ExecutionReference.from_dict(json.loads(json.dumps(reference.to_dict())))


def test_guarded_submission_records_both_artifact_identity_and_guard_contract(
    monkeypatch, converter
):
    client = _client()
    handle, guarded = _guarded_submission(monkeypatch, converter, client)
    context = handle.references()[0].context
    assert (
        context["package_sha256"] == hashlib.sha256(_package().to_bytes()).hexdigest()
    )
    assert (
        context["h2_package_sha256"] == hashlib.sha256(guarded.to_bytes()).hexdigest()
    )
    assert context["shot_success_tag"] == SHOT_SUCCESS_TAG
    assert context["shot_origin"] == "1"
    assert "h2_shot_guard" not in context
    assert "h2_shot_origin" not in context
    assert json.loads(context["qir_outputs"])[SHOT_SUCCESS_TAG] == ["bool", 1]


def test_program_without_shot_reads_keeps_ordinary_conversion(monkeypatch, converter):
    prepare = Mock(
        side_effect=AssertionError("valid original QIR must remain unchanged")
    )
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    client = _client()
    package = _package()
    handle = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    ).submit(package, 2)
    converter.assert_called_once_with(package)
    prepare.assert_not_called()
    assert "h2_shot_guard" not in handle.references()[0].context
    assert "shot_success_tag" not in handle.references()[0].context
    assert "shot_origin" not in handle.references()[0].context


def test_non_abort_conversion_failure_is_not_retried(monkeypatch, converter):
    converter.side_effect = ExecutionError("unsupported runtime Float")
    prepare = Mock(
        side_effect=AssertionError("unrelated failures cannot enable rewriting")
    )
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    client = _client()
    with pytest.raises(ExecutionError, match="unsupported runtime Float"):
        NexusTransport(
            NexusExecutionOptions(system_name="H2-1E"), client=client
        ).submit(_package(), 2)
    converter.assert_called_once()
    prepare.assert_not_called()
    client.login.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_abort_without_supported_table_keeps_the_original_failure(converter):
    from qamomile.hugr._qir import _H2AbortError

    failure = _H2AbortError("unrelated abort")
    converter.side_effect = failure
    client = _client()
    with pytest.raises(_H2AbortError) as error:
        NexusTransport(
            NexusExecutionOptions(system_name="H2-1E"), client=client
        ).submit(_package(), 2)
    assert error.value is failure
    converter.assert_called_once()
    client.login.assert_not_called()


@pytest.mark.parametrize("shot_origin", [0, 1])
def test_saved_guard_origin_is_restored_without_using_current_destination(
    monkeypatch, converter, shot_origin
):
    converter.side_effect = AssertionError("restoration cannot convert a program")
    prepare = Mock(
        side_effect=AssertionError("restoration cannot specialize a program")
    )
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    client = _client()
    _guarded_readouts(client, [1, 1])
    reference = _saved_guard_reference(shot_origin=shot_origin)
    handle = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    ).retrieve(reference)
    assert handle.result() == [{"left": True, "right": False}] * 2
    assert handle.references() == (reference,)
    converter.assert_not_called()
    prepare.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_restored_guard_requires_success_on_every_shot(monkeypatch, converter, missing):
    """Reject missing or failed markers without returning or resubmitting results."""
    converter.side_effect = AssertionError("restoration cannot convert a program")
    prepare = Mock(side_effect=AssertionError("restoration cannot specialize"))
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    client = _client()
    raw = _guarded_readouts(client, [1, 0])
    if missing:
        raw.get_bitlist.return_value = [
            SimpleNamespace(reg_name=tag, index=[0]) for tag in _OUTPUTS
        ]
        raw.get_shots.return_value = [[1, 0], [1, 0]]
    reference = _saved_guard_reference()
    handle = NexusTransport(client=client).retrieve(reference)
    message = "Missing QIR result bits" if missing else "shot range check failed"
    for _ in range(2):
        with pytest.raises(ExecutionError, match=message):
            handle.result()
    assert handle.status() == JobStatus.FAILED
    assert handle.references() == (reference,)
    client.jobs.results.assert_called_once()
    converter.assert_not_called()
    prepare.assert_not_called()
    client.qir.upload.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_restore_uses_the_saved_success_tag_without_exposing_it(converter, success):
    """Use the saved output name rather than a version or hard-coded marker."""
    converter.side_effect = AssertionError("restoration cannot convert a program")
    tag = "private_success"
    client = _client()
    _guarded_readouts(client, [1, int(success)], tag=tag)
    reference = _saved_guard_reference(tag=tag)
    handle = NexusTransport(client=client).retrieve(reference)
    if success:
        assert handle.result() == [{"left": True, "right": False}] * 2
    else:
        with pytest.raises(ExecutionError, match="shot range check failed"):
            handle.result()
    converter.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("restored", [False, True])
def test_successful_guarded_results_hide_private_state(
    monkeypatch, converter, restored
):
    client = _client()
    handle, _ = _guarded_submission(monkeypatch, converter, client)
    raw = _guarded_readouts(client, [1, 1])
    if restored:
        reference = ExecutionReference.from_dict(handle.references()[0].to_dict())
        handle = NexusTransport(client=client).retrieve(reference)
    assert handle.result() == [
        {"left": True, "right": False},
        {"left": True, "right": False},
    ]
    raw.get_shots.assert_called_once()


@pytest.mark.parametrize("restored", [False, True])
def test_failed_guard_caches_failure_without_publishing_partial_results(
    monkeypatch, converter, restored
):
    client = _client()
    handle, _ = _guarded_submission(monkeypatch, converter, client)
    raw = _guarded_readouts(client, [1, 0])
    if restored:
        handle = NexusTransport(client=client).retrieve(handle.references()[0])
    for _ in range(2):
        with pytest.raises(ExecutionError, match="shot range check failed"):
            handle.result()
    assert handle.status() == JobStatus.FAILED
    raw.get_shots.assert_called_once()
    client.jobs.results.assert_called_once()
    client.start_execute_job.assert_called_once()


def test_preflight_failure_after_rejected_qir_precedes_authentication(
    monkeypatch, converter
):
    from qamomile.hugr._qir import _H2AbortError

    client = _client()
    prepare = Mock(side_effect=ExecutionError("table is too short"))
    converter.side_effect = _H2AbortError("converter rejected abort")
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    with pytest.raises(ExecutionError, match="table is too short"):
        transport.submit(_package(), 2)
    converter.assert_called_once()
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("guarded", [False, True])
def test_inconsistent_generated_marker_contract_is_rejected_before_upload(
    monkeypatch, converter, guarded
):
    from qamomile.hugr._qir import _H2AbortError

    client = _client()
    monkeypatch.setattr(
        "qamomile.hugr._nexus.prepare_h2_shot_range",
        Mock(side_effect=lambda package, shots, origin: (package, guarded)),
    )
    result = (
        _BITCODE,
        _OUTPUTS if guarded else {**_OUTPUTS, SHOT_SUCCESS_TAG: ("bool", 1)},
    )
    converter.side_effect = (
        [_H2AbortError("converter rejected abort"), result] if guarded else [result]
    )
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    with pytest.raises(ExecutionError, match="guard output contract"):
        transport.submit(_package(), 2)
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"shot_success_tag": None},
        {"shot_success_tag": ""},
        {"shot_success_tag": "undeclared"},
        {"shot_origin": None},
        {"shot_origin": "-1"},
        {"shot_origin": "1.0"},
        {"shot_origin": "01"},
        {"shot_origin": str(2**64)},
        {"shot_origin": str(2**64 - 1)},
        {"qir_outputs": json.dumps(_OUTPUTS)},
        {"program_format": "hugr"},
    ],
)
def test_invalid_saved_success_marker_is_rejected_before_lookup(changes):
    """Require an unambiguous named output and valid saved origin before lookup."""
    context = dict(_saved_guard_reference().context)
    for name, value in changes.items():
        if value is None:
            context.pop(name, None)
        else:
            context[name] = value
    reference = ExecutionReference(
        "quantinuum-nexus", ("existing-job",), context=context
    )
    client = _client()
    with pytest.raises(ValueError):
        NexusTransport(client=client).retrieve(reference)
    client.login.assert_not_called()
    client.jobs.get.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_abort_table_preparation_receives_original_instead_of_normalized_copy(
    monkeypatch, converter
):
    from qamomile.hugr._qir import _H2AbortError

    original = _package()
    normalized = SimpleNamespace(to_bytes=lambda: b"normalized-package")
    proven = SimpleNamespace(to_bytes=lambda: b"normalized-proven-package")
    monkeypatch.setattr(
        "qamomile.hugr._nexus.normalize_h2_shots",
        Mock(return_value=(normalized, True)),
    )
    prepare = Mock(return_value=(proven, True))
    monkeypatch.setattr("qamomile.hugr._nexus.prepare_h2_shot_range", prepare)
    converter.side_effect = [
        _H2AbortError("table bounds panic"),
        (_BITCODE, {**_OUTPUTS, SHOT_SUCCESS_TAG: ("bool", 1)}),
    ]
    client = _client()
    NexusTransport(NexusExecutionOptions(system_name="H2-1E"), client=client).submit(
        original, 4
    )
    prepare.assert_called_once_with(original, 4, 1)
    assert converter.call_args_list[0].args == (normalized,)
    assert converter.call_args_list[1].args == (proven,)
    client.qir.upload.assert_called_once()
