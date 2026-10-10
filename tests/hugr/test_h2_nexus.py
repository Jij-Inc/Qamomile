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

from qamomile.circuit.transpiler import ExecutionError, ExecutionReference
from qamomile.hugr import HugrExecutor, NexusExecutionOptions
from qamomile.hugr._nexus import NexusTransport

pytestmark = pytest.mark.hugr

_BITCODE = b"BC\xc0\xde-test-qir"
_OUTPUTS = {"left": ("bool", 1), "right": ("bool", 1)}


def _package():
    """Provide deterministic original HUGR bytes without compiler setup."""
    return SimpleNamespace(to_bytes=lambda: b"original-hugr-package")


def _client():
    """Provide mock SDK operations that cannot connect to a remote service."""
    native = SimpleNamespace(id="h2-job-id", job_type="execute")
    return SimpleNamespace(
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
                    SimpleNamespace(
                        download_result=Mock(
                            return_value=QsysResult([[("left", True)]])
                        )
                    )
                ]
            ),
        ),
    )


@pytest.fixture
def converter(monkeypatch):
    """Isolate SDK routing tests from the separately tested conversion tool."""
    convert = Mock(return_value=(_BITCODE, _OUTPUTS))
    monkeypatch.setattr("qamomile.hugr._nexus.to_h2_qir", convert)
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
        target, options=NexusExecutionOptions(system_name=system_name)
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
    options = NexusExecutionOptions(system_name=system_name, backend_config=config)
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
    client.qir.upload.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


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
    monkeypatch.setattr("qamomile.hugr._qir.importlib.import_module", imports)
    handle = NexusTransport(client=client).submit(_package(), 1)
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
    client.hugr.upload.assert_not_called()
