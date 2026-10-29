"""Verify portable H2 preparation and exact submission without remote jobs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qamomile.circuit.transpiler import ExecutionError, ExecutionReference
from qamomile.hugr import (
    HugrExecutor,
    HugrTranspiler,
    NexusExecutionOptions,
    PreparedH2Submission,
    prepare_h2_submission,
)
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._qir import _H2AbortError
from qamomile.hugr._runtime import prepare_execution
from qamomile.hugr._shot_range import SHOT_SUCCESS_TAG
from tests.hugr.test_h2_nexus import (
    _BITCODE,
    _OUTPUTS,
    _client,
    _guarded_readouts,
    _package,
)

pytestmark = pytest.mark.hugr


def _context(*, guarded=False, origin=1):
    """Build complete detached context for one fake final QIR payload."""
    outputs = dict(_OUTPUTS)
    context = {
        "shots": "2",
        "package_sha256": hashlib.sha256(b"original-hugr-package").hexdigest(),
        "program_format": "qir",
    }
    if guarded:
        outputs[SHOT_SUCCESS_TAG] = ("bool", 1)
        context.update(
            shot_success_tag=SHOT_SUCCESS_TAG,
            shot_origin=str(origin),
            h2_package_sha256=hashlib.sha256(b"guarded-hugr-package").hexdigest(),
        )
    context["qir_outputs"] = json.dumps(outputs, sort_keys=True, separators=(",", ":"))
    return context


@pytest.fixture
def converted(monkeypatch):
    """Provide final bytes while making normalization visible to callers."""
    convert = Mock(return_value=(_BITCODE, _OUTPUTS))
    monkeypatch.setattr("qamomile.hugr._nexus.to_h2_qir", convert)
    monkeypatch.setattr(
        "qamomile.hugr._nexus.normalize_h2_shots",
        Mock(side_effect=lambda package, shots, origin: (package, False)),
    )
    return convert


def _install_executor(monkeypatch, client, **options):
    """Create a public executor backed exclusively by simulated provider calls."""
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda settings: NexusTransport(settings, client=client),
    )
    return HugrExecutor("nexus", options=NexusExecutionOptions(**options))


def test_public_preparation_never_creates_or_imports_a_nexus_client(
    monkeypatch, converted
):
    """Prepare before executor construction without authentication or SDK imports."""
    forbidden = Mock(side_effect=AssertionError("preparation must remain local"))
    monkeypatch.setattr("qamomile.hugr._nexus.NexusTransport.__init__", forbidden)
    monkeypatch.setattr(
        "qamomile.hugr._nexus.importlib", SimpleNamespace(import_module=forbidden)
    )
    package = _package()
    before = package.to_bytes()
    prepared = prepare_h2_submission(package, shots=2)
    assert prepared.qir == _BITCODE
    assert dict(prepared.context) == _context()
    assert prepared.shots == 2
    assert prepared.qir_sha256 == hashlib.sha256(_BITCODE).hexdigest()
    assert package.to_bytes() == before
    forbidden.assert_not_called()
    converted.assert_called_once_with(package)


def test_prepared_payload_is_immutable_and_detaches_all_metadata():
    """Source mappings, metadata exports, and references cannot mutate the payload."""
    context = _context(guarded=True)
    context["caller_extension"] = "preserved"
    prepared = PreparedH2Submission(_BITCODE, context)
    context["shots"] = "3"
    assert prepared.shots == 2
    with pytest.raises(TypeError):
        prepared.context["shots"] = "4"
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.qir = b"changed"
    exported = prepared.to_dict()
    exported["context"]["shots"] = "5"
    reference = prepared.reference(job_id="found-id", target="H2-1")
    reference.context["shots"] = "6"
    assert prepared.shots == 2
    assert prepared.context["caller_extension"] == "preserved"


def test_saved_json_and_separate_qir_roundtrip(tmp_path):
    """Reload complete guard metadata from portable files before any submission."""
    prepared = PreparedH2Submission(_BITCODE, _context(guarded=True))
    metadata_file = tmp_path / "prepared.json"
    qir_file = tmp_path / "program.bc"
    metadata_file.write_text(json.dumps(prepared.to_dict()))
    qir_file.write_bytes(prepared.qir)
    loaded = PreparedH2Submission.from_dict(
        json.loads(metadata_file.read_text()), qir=qir_file.read_bytes()
    )
    assert loaded == prepared
    assert loaded.to_dict() == prepared.to_dict()
    assert "qir" not in loaded.to_dict()


@pytest.mark.parametrize("qir", [None, "BC", bytearray(_BITCODE), b"", b"invalid"])
def test_non_bitcode_payload_is_rejected(qir):
    """Reject non-byte or non-LLVM payloads before a client is needed."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    with pytest.raises(ValueError, match="bitcode"):
        PreparedH2Submission.from_dict(metadata, qir=qir)


@pytest.mark.parametrize("version", [None, True, "1", 0, 2])
def test_metadata_rejects_unsupported_or_untyped_versions(version):
    """Version one is an integer and does not silently accept future schemas."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    metadata["version"] = version
    with pytest.raises(ValueError, match="version"):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE)


@pytest.mark.parametrize(
    "field", ["version", "qir_sha256", "context", "prepared_sha256"]
)
def test_metadata_requires_every_top_level_field(field):
    """Reject incomplete records rather than silently rebuilding their checksums."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    del metadata[field]
    with pytest.raises(ValueError, match="schema"):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE)


def test_metadata_rejects_extra_top_level_fields():
    """Keep per-request destination data outside the portable payload schema."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    metadata["target"] = "H2-1"
    with pytest.raises(ValueError, match="schema"):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE)


@pytest.mark.parametrize("context", [None, [], {"shots": 2}, {1: "value"}])
def test_loaded_metadata_rejects_non_string_context(context):
    """Treat serialized context as external data before accepting checksums."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    metadata["context"] = context
    with pytest.raises(ValueError, match="context"):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE)


@pytest.mark.parametrize("field", ["qir_sha256", "prepared_sha256"])
@pytest.mark.parametrize("value", [None, "invalid", "A" * 64, "0" * 64])
def test_metadata_validates_both_checksums(field, value):
    """Reject malformed digests and valid-looking digests that do not match."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    metadata[field] = value
    with pytest.raises(ValueError):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE)


def test_qir_bytes_are_covered_by_the_saved_digest():
    """Detect changed binary bytes even when their LLVM magic is intact."""
    metadata = PreparedH2Submission(_BITCODE, _context()).to_dict()
    with pytest.raises(ValueError, match="QIR checksum"):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE + b"changed")


@pytest.mark.parametrize(
    "key,value",
    [
        ("shots", "3"),
        ("package_sha256", "a" * 64),
        ("qir_outputs", '{"different":["bool",1]}'),
        ("h2_package_sha256", "b" * 64),
        ("shot_origin", "0"),
        ("caller_extension", "changed"),
    ],
)
def test_checksum_covers_valid_looking_context_changes(key, value):
    """Detect metadata corruption that still passes the context schema."""
    metadata = PreparedH2Submission(_BITCODE, _context(guarded=True)).to_dict()
    if key == "qir_outputs":
        value = json.dumps({"different": ["bool", 1], SHOT_SUCCESS_TAG: ["bool", 1]})
    metadata["context"][key] = value
    with pytest.raises(ValueError, match="metadata checksum"):
        PreparedH2Submission.from_dict(metadata, qir=_BITCODE)


@pytest.mark.parametrize(
    "key,value",
    [
        ("shots", "0"),
        ("shots", "-1"),
        ("shots", "01"),
        ("shots", "1.5"),
        ("shots", True),
        ("shots", ""),
        ("package_sha256", "invalid"),
        ("package_sha256", "A" * 64),
        ("program_format", "hugr"),
        ("qir_outputs", "not-json"),
        ("qir_outputs", "[]"),
        ("qir_outputs", '{"left":["bool",64]}'),
        ("qir_outputs", '{"left":["bool",1],"left":["bool",1]}'),
    ],
)
def test_context_rejects_invalid_counts_hashes_formats_and_manifests(key, value):
    """Reject malformed recovery data independently of its corruption checksum."""
    context = _context()
    context[key] = value
    with pytest.raises(ValueError):
        PreparedH2Submission(_BITCODE, context)


@pytest.mark.parametrize(
    "missing", ["shots", "package_sha256", "program_format", "qir_outputs"]
)
def test_context_requires_complete_base_recovery_metadata(missing):
    """Every prepared payload keeps its original identity and output contract."""
    context = _context()
    del context[missing]
    with pytest.raises(ValueError):
        PreparedH2Submission(_BITCODE, context)


@pytest.mark.parametrize(
    "key,value",
    [
        ("shot_success_tag", None),
        ("shot_success_tag", "missing"),
        ("shot_origin", None),
        ("shot_origin", "-1"),
        ("shot_origin", "01"),
        ("shot_origin", str(2**64 - 2)),
        ("h2_package_sha256", None),
        ("h2_package_sha256", "invalid"),
        ("qir_outputs", json.dumps(_OUTPUTS)),
    ],
)
def test_context_rejects_incomplete_or_inconsistent_guards(key, value):
    """A mandatory private success marker requires its full guard metadata."""
    context = _context(guarded=True)
    if value is None:
        del context[key]
    else:
        context[key] = value
    with pytest.raises(ValueError):
        PreparedH2Submission(_BITCODE, context)


def test_unguarded_context_rejects_orphaned_normalized_package_hash():
    """Do not accept partial guard metadata on an unguarded payload."""
    context = _context()
    context["h2_package_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        PreparedH2Submission(_BITCODE, context)


@pytest.mark.parametrize("target", [None, "selene", "Helios-1", "H2-0", "H2-1E "])
def test_reference_requires_the_actual_supported_h2_target(target):
    """A found job must be associated with an H2 device explicitly."""
    prepared = PreparedH2Submission(_BITCODE, _context())
    with pytest.raises(ValueError, match="target"):
        prepared.reference(job_id="found-id", target=target)


@pytest.mark.parametrize("job_id", [None, "", " \t"])
def test_reference_rejects_empty_job_identifiers(job_id):
    """Reject an unusable caller-discovered ID before job lookup."""
    prepared = PreparedH2Submission(_BITCODE, _context())
    with pytest.raises(ValueError, match="job_id"):
        prepared.reference(job_id=job_id, target="H2-1")


def test_same_saved_bytes_submit_to_syntax_checker_and_hardware_without_conversion(
    monkeypatch,
):
    """Unguarded bytes are reusable with independently chosen exact names and devices."""
    prepared = PreparedH2Submission(_BITCODE, _context())
    forbidden = Mock(side_effect=AssertionError("saved bytes must not be regenerated"))
    for operation in ("to_h2_qir", "normalize_h2_shots", "prepare_h2_shot_range"):
        monkeypatch.setattr(f"qamomile.hugr._nexus.{operation}", forbidden)
    for target, name in (("H2-1SC", "caller-check-001"), ("H2-1", "caller-run-001")):
        client = _client()
        executor = _install_executor(monkeypatch, client, system_name=target)
        handle = executor.submit_prepared(prepared, name=name)
        client.qir.upload.assert_called_once_with(
            qir=prepared.qir,
            name=name,
            project=client.projects.get_or_create.return_value,
        )
        assert client.qir.upload.call_args.kwargs["qir"] is prepared.qir
        submitted = client.start_execute_job.call_args.kwargs
        assert submitted["name"] == name
        assert submitted["n_shots"] == [prepared.shots]
        assert submitted["valid_check"] is True
        assert submitted["backend_config"].device_name == target
        assert handle.references()[0].target == target
        assert handle.references()[0].context == prepared.context
        client.start_execute_job.assert_called_once()
        client.hugr.upload.assert_not_called()
    forbidden.assert_not_called()


def test_prepared_submit_preserves_all_executor_settings_and_native_target(monkeypatch):
    """Keep provider settings outside saved bytes and resolve the native config device."""
    prepared = PreparedH2Submission(_BITCODE, _context())
    client = _client()
    project = object()
    config = SimpleNamespace(type="QuantinuumConfig", device_name="H2-2", max_cost=2.5)
    executor = _install_executor(
        monkeypatch,
        client,
        system_name="Helios-1",
        backend_config=config,
        project=project,
        max_cost=2.5,
        n_qubits=4,
        credential_name="credential",
        user_group="research",
        target_region="sg",
    )
    handle = executor.submit_prepared(prepared, name="exact-native-target-name")
    assert client.start_execute_job.call_args.kwargs == {
        "programs": ["uploaded-qir"],
        "n_shots": [2],
        "backend_config": config,
        "name": "exact-native-target-name",
        "project": project,
        "valid_check": True,
        "max_cost": 2.5,
        "n_qubits": 4,
        "credential_name": "credential",
        "user_group": "research",
        "target_region": "sg",
    }
    assert handle.references()[0].target == "H2-2"
    client.QuantinuumConfig.assert_not_called()
    client.HeliosConfig.assert_not_called()


def test_prepared_submit_rejects_an_untyped_payload_before_authentication():
    """Saved plain dictionaries must pass through the validated payload loader."""
    client = _client()
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    with pytest.raises(TypeError, match="PreparedH2Submission"):
        transport.submit_prepared({}, name="caller-001")
    client.login.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("name", [None, "", " \t\n", 3])
def test_prepared_submit_rejects_invalid_names_before_authentication(name):
    """The complete caller-owned name must contain a non-whitespace character."""
    client = _client()
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    with pytest.raises(ValueError, match="name"):
        transport.submit_prepared(PreparedH2Submission(_BITCODE, _context()), name=name)
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_prepared_submit_requires_an_explicit_name():
    """Omission cannot silently generate a name after the caller saves its payload."""
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=_client()
    )
    with pytest.raises(TypeError, match="name"):
        transport.submit_prepared(PreparedH2Submission(_BITCODE, _context()))


def test_prepared_submit_rejects_local_and_helios_destinations(monkeypatch):
    """Only Nexus H2 can consume prepared QIR before any authentication."""
    prepared = PreparedH2Submission(_BITCODE, _context())
    with pytest.raises(ValueError, match="H2 destination"):
        HugrExecutor().submit_prepared(prepared, name="caller-001")
    client = _client()
    executor = _install_executor(
        monkeypatch, client, system_name="Helios-1", max_cost=1
    )
    with pytest.raises(ValueError, match="H2 destination"):
        executor.submit_prepared(prepared, name="caller-001")
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()
    client.hugr.upload.assert_not_called()


def test_prepared_submit_rejects_incompatible_guard_origin_before_authentication():
    """A saved guard with another origin cannot silently change its shot meaning."""
    prepared = PreparedH2Submission(_BITCODE, _context(guarded=True, origin=0))
    client = _client()
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1E"), client=client
    )
    with pytest.raises(ValueError, match="origin.*incompatible"):
        transport.submit_prepared(prepared, name="caller-001")
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()


@pytest.mark.parametrize("fallback", [False, True])
def test_final_prepared_guard_preserves_syntax_checker_submission(
    monkeypatch, converted, fallback
):
    """Send normalized and fallback guards while keeping syntax-check results unavailable."""
    guarded = SimpleNamespace(to_bytes=lambda: b"guarded-hugr-package")
    monkeypatch.setattr(
        "qamomile.hugr._nexus.normalize_h2_shots",
        Mock(return_value=(guarded, not fallback)),
    )
    monkeypatch.setattr(
        "qamomile.hugr._nexus.prepare_h2_shot_range", Mock(return_value=(guarded, True))
    )
    result = (_BITCODE, {**_OUTPUTS, SHOT_SUCCESS_TAG: ("bool", 1)})
    converted.side_effect = (
        [_H2AbortError("bounds abort"), result] if fallback else [result]
    )
    prepared = prepare_h2_submission(_package(), shots=2)
    assert prepared.context["shot_success_tag"] == SHOT_SUCCESS_TAG
    client = _client()
    transport = NexusTransport(
        NexusExecutionOptions(system_name="H2-1SC"), client=client
    )
    handle = transport.submit_prepared(prepared, name="caller-check-001")
    assert client.qir.upload.call_args.kwargs["qir"] == prepared.qir
    assert handle.references()[0].context == prepared.context
    with pytest.raises(ExecutionError, match="syntax-check.*measurement"):
        handle.result()
    client.login.assert_called_once()
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()
    client.jobs.results.assert_not_called()


@pytest.mark.parametrize("prepared_call", [False, True])
def test_cost_conflicts_fail_before_conversion_or_authentication(
    monkeypatch, prepared_call
):
    """Retain ordinary validation precedence and apply it to prepared submission."""
    client = _client()
    config = SimpleNamespace(type="QuantinuumConfig", device_name="H2-1E", max_cost=2)
    transport = NexusTransport(
        NexusExecutionOptions(backend_config=config, max_cost=1), client=client
    )
    convert = Mock(side_effect=AssertionError("invalid settings must fail first"))
    monkeypatch.setattr("qamomile.hugr._nexus.to_h2_qir", convert)
    with pytest.raises(ValueError, match="max_cost.*conflict"):
        if prepared_call:
            transport.submit_prepared(
                PreparedH2Submission(_BITCODE, _context()), name="caller-001"
            )
        else:
            transport.submit(_package(), 2)
    convert.assert_not_called()
    client.login.assert_not_called()
    client.QuantinuumConfig.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_lost_submission_response_recovers_with_saved_context_and_found_id(
    monkeypatch, tmp_path
):
    """Restore complete guarded rows after caller lookup without converting or submitting again."""
    prepared = PreparedH2Submission(_BITCODE, _context(guarded=True))
    (tmp_path / "program.bc").write_bytes(prepared.qir)
    (tmp_path / "prepared.json").write_text(json.dumps(prepared.to_dict()))
    client = _client()
    executor = _install_executor(monkeypatch, client, system_name="H2-1E")
    client.start_execute_job.side_effect = TimeoutError(
        "response lost after remote acceptance"
    )
    with pytest.raises(TimeoutError, match="response lost"):
        executor.submit_prepared(prepared, name="caller-recover-001")
    loaded = PreparedH2Submission.from_dict(
        json.loads((tmp_path / "prepared.json").read_text()),
        qir=(tmp_path / "program.bc").read_bytes(),
    )
    reference = ExecutionReference.from_dict(
        loaded.reference(job_id="caller-found-id", target="H2-1E").to_dict()
    )
    _guarded_readouts(client, [1, 1])
    forbidden = Mock(
        side_effect=AssertionError("recovery must not regenerate the program")
    )
    for operation in ("to_h2_qir", "normalize_h2_shots", "prepare_h2_shot_range"):
        monkeypatch.setattr(f"qamomile.hugr._nexus.{operation}", forbidden)
    restored = executor.retrieve(reference)
    assert restored.result() == [{"left": True, "right": False}] * 2
    assert restored.references()[0].context == prepared.context
    client.jobs.get.assert_called_once_with(id="caller-found-id")
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()
    forbidden.assert_not_called()


@pytest.mark.parametrize("kind", ["ordinary", "shot", "fallback"])
def test_real_converter_public_preparation_matches_normal_uploaded_bytes(
    monkeypatch, kind
):
    """Use real normalized and fallback QIR to prove both routes save identical context."""
    pytest.importorskip("hugr_qir.hugr_to_qir")
    from qamomile.hugr import _nexus
    from tests.hugr.test_h2_shot_range import _bit_table, _no_shot, _shot_only

    kernel = {"ordinary": _no_shot, "shot": _shot_only, "fallback": _bit_table}[kind]
    bindings = {"values": [False, True]} if kind == "fallback" else {}
    package = prepare_execution(
        HugrTranspiler().compile(kernel, bindings=bindings)
    ).package
    before = package.to_bytes()
    convert = Mock(wraps=_nexus.to_h2_qir)
    fallback = Mock(wraps=_nexus.prepare_h2_shot_range)
    monkeypatch.setattr(_nexus, "to_h2_qir", convert)
    monkeypatch.setattr(_nexus, "prepare_h2_shot_range", fallback)
    prepared = prepare_h2_submission(package, shots=2)
    client = _client()
    executor = _install_executor(monkeypatch, client, system_name="H2-1E")
    handle = executor.submit(package, shots=2)
    assert client.qir.upload.call_args.kwargs["qir"] == prepared.qir
    assert handle.references()[0].context == prepared.context
    assert package.to_bytes() == before
    assert convert.call_count == (4 if kind == "fallback" else 2)
    assert fallback.call_count == (2 if kind == "fallback" else 0)
    if kind != "ordinary":
        from tests.hugr.test_h2_shot_execution import _QirExecution

        execution = _QirExecution(prepared.qir)
        for raw_shot in (0, 1, 2, 3):
            code, trace = execution.run(raw_shot)
            assert code == (0 if raw_shot in (1, 2) else 1)
            outputs = dict(trace.outputs)
            assert outputs[SHOT_SUCCESS_TAG] is (raw_shot in (1, 2))
            assert outputs["qamomile.output.0.0"] is (raw_shot == 2)
