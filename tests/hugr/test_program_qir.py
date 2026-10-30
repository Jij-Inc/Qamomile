"""Verify ordinary typed calls retain exact H2 payloads before remote effects."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("hugr")
pytest.importorskip("hugr_qir.hugr_to_qir")

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler import (
    CompletedExecutionHandle,
    ExecutionError,
    ExecutionReference,
    JobKind,
    JobSnapshot,
    JobStatus,
    ShotBased,
)
from qamomile.hugr import HugrExecutor, HugrTranspiler, NexusExecutionOptions, _nexus
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._shot_range import SHOT_SUCCESS_TAG
from qamomile.hugr.executable import _qir_bindings_fingerprint
from tests.hugr.test_expval import _rotated
from tests.hugr.test_expval_workflows import _structured
from tests.hugr.test_h2_nexus import _client
from tests.hugr.test_h2_shot_range import _bit_table, _no_shot, _shot_only

pytestmark = pytest.mark.hugr


@pytest.mark.parametrize(
    "original,changed",
    [
        ({}, {"scale": 1.0}),
        ({"scale": 1.0}, {"scale": 2.0}),
        ({"target": 1}, {"target": 2}),
        ({"enabled": False}, {"enabled": True}),
        ({"values": ({1: (0.5, False)},)}, {"values": ({2: (0.5, False)},)}),
        ({"values": ({1: (0.5, False)},)}, {"values": ({1: (0.5, True)},)}),
        ({"values": ((1, 2),)}, {"values": ((2, 1),)}),
        (
            {"values": np.empty((0, 2), dtype=np.bool_)},
            {"values": np.empty((0, 3), dtype=np.bool_)},
        ),
        (
            {"values": np.empty((0, 2), dtype=np.bool_)},
            {"values": np.empty((0, 2), dtype=np.uint64)},
        ),
    ],
)
def test_reuse_identity_preserves_all_normalized_runtime_values(original, changed):
    """Reuse identities distinguish values, dictionary keys, and empty array schemas."""
    identity = _qir_bindings_fingerprint(original)
    assert identity == _qir_bindings_fingerprint(deepcopy(original))
    assert identity != _qir_bindings_fingerprint(changed)


@qmc.qkernel
def _chosen_shot(target: qmc.UInt) -> qmc.Bit:
    """Return True only at the requested zero-based logical shot."""
    return qmc.get_current_shot() == target


def _executor(monkeypatch, client, **options):
    """Install a fake provider while keeping the public executor and converter."""
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda settings: NexusTransport(settings, client=client),
    )
    return HugrExecutor(
        "nexus", options=NexusExecutionOptions(**{"system_name": "H2-1E", **options})
    )


def _program():
    """Create a fresh runtime-bound program that needs H2 shot normalization."""
    return HugrTranspiler().transpile(_chosen_shot, parameters=["target"])


def _snapshot(info, ids):
    """Build existing restoration objects from caller-saved JSON and found IDs."""
    return JobSnapshot(
        kind=JobKind(info["operation"]),
        shots=info["shots"],
        executions=tuple(
            ExecutionReference(
                provider=item["provider"],
                target=item["target"],
                job_ids=(job_id,),
                context=item["context"],
            )
            for item, job_id in zip(info["submissions"], ids, strict=True)
        ),
    )


def _readouts(client, info, rows):
    """Serve Boolean register rows according to the publicly saved output labels."""
    outputs = json.loads(info["submissions"][0]["context"]["qir_outputs"])
    tags = list(outputs)
    raw = SimpleNamespace(
        get_bitlist=Mock(
            return_value=[SimpleNamespace(reg_name=tag, index=[0]) for tag in tags]
        ),
        get_shots=Mock(
            return_value=[
                [1 if tag == SHOT_SUCCESS_TAG else int(value) for tag in tags]
                for value in rows
            ]
        ),
    )
    client.jobs.results.return_value = [
        SimpleNamespace(download_result=Mock(return_value=raw))
    ]


def _forbid_conversion(monkeypatch):
    """Make conversion and normalization fail if reuse or restoration rebuilds QIR."""
    forbidden = Mock(side_effect=AssertionError("QIR must not be regenerated"))
    for name in ("to_h2_qir", "normalize_h2_shots", "prepare_h2_shot_range"):
        monkeypatch.setattr(_nexus, name, forbidden)
    return forbidden


@pytest.mark.parametrize("kind", ["ordinary", "shot", "fallback"])
def test_normal_sample_publishes_real_final_qir_before_authentication(
    monkeypatch, kind
):
    """The observable payload matches uploaded bytes, including fallback guards."""
    kernel = {"ordinary": _no_shot, "shot": _shot_only, "fallback": _bit_table}[kind]
    bindings = {"values": [False, True]} if kind == "fallback" else {}
    program = HugrTranspiler().transpile(kernel, bindings=bindings)
    original = program.artifact.to_bytes()
    assert program.qir is None
    assert program.qir_info is None
    client = _client()
    executor = _executor(monkeypatch, client, name="normal-call")
    observed = []

    def observe():
        """Capture what callers can inspect when authentication first begins."""
        observed.append((program.qir, program.qir_info))

    client.login.side_effect = observe
    job = program.sample(executor, shots=2)
    qir, info = observed[0]
    assert info == program.qir_info
    assert qir == program.qir == client.qir.upload.call_args.kwargs["qir"]
    assert isinstance(qir, bytes) and qir.startswith(b"BC\xc0\xde")
    assert set(info) == {"version", "operation", "shots", "submissions"}
    assert (info["version"], info["operation"], info["shots"]) == (1, "sample", 2)
    [item] = info["submissions"]
    assert set(item) == {"provider", "name", "target", "qir_sha256", "context"}
    assert item["qir_sha256"] == hashlib.sha256(qir).hexdigest()
    assert item["name"].startswith("normal-call-")
    assert item["name"] == client.start_execute_job.call_args.kwargs["name"]
    assert item["target"] == "H2-1E"
    assert item["context"] == dict(job.snapshot().executions[0].context)
    assert all(isinstance(value, str) for value in item["context"].values())
    assert {
        "shots",
        "program_format",
        "package_sha256",
        "qir_outputs",
        "abi_sha256",
    } <= item["context"].keys()
    assert program.artifact.to_bytes() == original
    assert json.loads(json.dumps(info)) == info
    if kind != "ordinary":
        from tests.hugr.test_h2_shot_execution import _QirExecution

        assert item["context"]["shot_success_tag"] == SHOT_SUCCESS_TAG
        assert item["context"]["shot_origin"] == "1"
        assert item["context"]["h2_package_sha256"] != item["context"]["package_sha256"]
        execution = _QirExecution(qir)
        for raw_shot in (0, 1, 2, 3):
            code, trace = execution.run(raw_shot)
            assert code == (0 if raw_shot in (1, 2) else 1)
            outputs = dict(trace.outputs)
            assert outputs[SHOT_SUCCESS_TAG] is (raw_shot in (1, 2))
            assert outputs["qamomile.output.0.0"] is (raw_shot == 2)


@pytest.mark.parametrize("operation", ["sample", "run"])
@pytest.mark.parametrize("guarded", [False, True])
def test_sc_then_hardware_reuses_qir_and_refreshes_request(
    monkeypatch, operation, guarded
):
    """SC and hardware share bytes while each request uses its own names and settings."""
    program = _program() if guarded else HugrTranspiler().transpile(_no_shot)
    bindings = {"target": 0} if guarded else {}
    client = _client()
    sc = _executor(monkeypatch, client, system_name="H2-1SC", name="syntax", max_cost=1)
    first = getattr(program, operation)(sc, bindings=bindings)
    qir, info = program.qir, program.qir_info
    assert first.status() is JobStatus.COMPLETED
    with pytest.raises(ExecutionError, match="syntax-check"):
        first.result()
    forbidden = _forbid_conversion(monkeypatch)
    hardware = _executor(
        monkeypatch,
        client,
        system_name="Helios-1E",
        name="hardware",
        max_cost=3,
        backend_config=SimpleNamespace(type="QuantinuumConfig", device_name="H2-1"),
        n_qubits=2,
        credential_name="credential",
        user_group="research",
        target_region="sg",
    )
    second = getattr(program, operation)(hardware, bindings=bindings)
    assert program.qir is qir
    assert program.qir_info["submissions"][0]["target"] == "H2-1"
    name = program.qir_info["submissions"][0]["name"]
    assert name.startswith("hardware-") and name != info["submissions"][0]["name"]
    sent = client.start_execute_job.call_args.kwargs
    assert sent["name"] == name
    assert sent["backend_config"].device_name == "H2-1"
    assert sent["max_cost"] == 3 and sent["n_qubits"] == 2
    assert sent["credential_name"] == "credential" and sent["user_group"] == "research"
    assert sent["target_region"] == "sg"
    assert second.snapshot().executions[0].target == "H2-1"
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 2
    assert all(call.kwargs["qir"] is qir for call in client.qir.upload.call_args_list)
    assert program.qir_info["shots"] == (1024 if operation == "sample" else None)
    forbidden.assert_not_called()


def test_properties_are_read_only_and_all_metadata_copies_are_detached(monkeypatch):
    """Caller edits and later generations cannot mutate saved bytes or dictionaries."""
    program = _program()
    executor = _executor(monkeypatch, _client())
    first = program.sample(executor, shots=2, bindings={"target": 0})
    qir, info = program.qir, program.qir_info
    saved = json.loads(json.dumps(info))
    info["submissions"][0]["context"]["shots"] = "999"
    info["submissions"].append({})
    info["operation"] = "run"
    first.snapshot().executions[0].context["shots"] = "17"
    assert program.qir_info == saved
    for name in ("qir", "qir_info"):
        with pytest.raises(AttributeError):
            setattr(program, name, None)
    program.sample(executor, shots=3, bindings={"target": 1})
    assert program.qir != qir
    assert saved["shots"] == 2
    assert saved["submissions"][0]["context"]["shots"] == "2"


@pytest.mark.parametrize("change", ["bindings", "shots", "operation", "origin"])
def test_changed_conversion_conditions_do_not_reuse_old_qir(monkeypatch, change):
    """A local regeneration failure retains the last complete operation."""
    program = _program()
    client = _client()
    executor = _executor(monkeypatch, client)
    program.sample(executor, shots=1, bindings={"target": 0})
    qir, info = program.qir, program.qir_info
    forbidden = _forbid_conversion(monkeypatch)
    if change == "origin":
        monkeypatch.setattr(_nexus, "_H2_SHOT_ORIGIN", 0)
    with pytest.raises(AssertionError, match="regenerated"):
        getattr(program, "run" if change == "operation" else "sample")(
            executor,
            shots=2 if change == "shots" else 1,
            bindings={"target": 1 if change == "bindings" else 0},
        )
    assert program.qir is qir and program.qir_info == info
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 1
    forbidden.assert_called_once()


def test_only_immediately_retained_generation_can_be_reused(monkeypatch):
    """Returning to older bindings regenerates instead of consulting a keyed cache."""
    program = _program()
    executor = _executor(monkeypatch, _client())
    program.sample(executor, shots=2, bindings={"target": 0})
    program.sample(executor, shots=2, bindings={"target": 1})
    info = program.qir_info
    _forbid_conversion(monkeypatch)
    with pytest.raises(AssertionError, match="regenerated"):
        program.sample(executor, shots=2, bindings={"target": 0})
    assert program.qir_info == info


@pytest.mark.parametrize("stage", ["login", "project", "upload", "start"])
def test_remote_failure_retains_new_record_at_every_boundary(monkeypatch, stage):
    """Authentication, project, upload, and response failures retain complete context."""
    program = _program()
    client = _client()
    executor = _executor(monkeypatch, client)
    failed = {
        "login": client.login,
        "project": client.projects.get_or_create,
        "upload": client.qir.upload,
        "start": client.start_execute_job,
    }[stage]

    def fail(*args, **kwargs):
        """Check metadata visibility at the failing external boundary."""
        assert program.qir_info["submissions"][0]["context"]["abi_sha256"]
        assert program.qir is not None
        raise TimeoutError(stage)

    failed.side_effect = fail
    with pytest.raises(TimeoutError, match=stage):
        program.sample(executor, shots=3, bindings={"target": 1})
    assert program.qir_info["shots"] == 3
    assert program.qir_info["submissions"][0]["target"] == "H2-1E"
    assert len(program.qir_info["submissions"]) == 1


def test_lost_response_saved_context_restores_original_typed_program(
    monkeypatch, tmp_path
):
    """Caller-found IDs recover typed Boolean counts without conversion or resubmission."""
    program = _program()
    client = _client()
    executor = _executor(monkeypatch, client)
    client.start_execute_job.side_effect = TimeoutError(
        "response lost after acceptance"
    )
    with pytest.raises(TimeoutError):
        program.sample(executor, shots=3, bindings={"target": 1})
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(program.qir_info))
    info = json.loads(path.read_text())
    _readouts(client, info, [False, True, False])
    snapshot = _snapshot(info, ["caller-found-id"])
    _forbid_conversion(monkeypatch)
    with pytest.raises(ValueError, match="bindings"):
        program.restore(executor, snapshot, bindings={"target": 0})
    client.jobs.get.assert_not_called()
    restored = program.restore(executor, snapshot, bindings={"target": 1})
    assert dict(restored.result().results) == {False: 2, True: 1}
    assert (
        restored.snapshot().executions[0].context == info["submissions"][0]["context"]
    )
    client.jobs.get.assert_called_once_with(id="caller-found-id")
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 1


def test_non_h2_execution_preserves_latest_h2_generation(monkeypatch):
    """Helios and Selene leave either an initial or populated H2 record unchanged."""
    program = _program()
    client = _client()
    helios = _executor(monkeypatch, client, system_name="Helios-1E", max_cost=1)
    program.sample(helios, shots=2, bindings={"target": 0})
    assert program.qir is None and program.qir_info is None
    h2 = _executor(monkeypatch, client)
    program.sample(h2, shots=2, bindings={"target": 0})
    qir, info = program.qir, program.qir_info
    _forbid_conversion(monkeypatch)
    program.sample(helios, shots=3, bindings={"target": 1})
    selene = HugrExecutor()
    monkeypatch.setattr(
        selene, "submit", Mock(return_value=CompletedExecutionHandle([]))
    )
    program.sample(selene, shots=4, bindings={"target": 2})
    assert program.qir is qir and program.qir_info == info
    assert client.hugr.upload.call_count == 2
    selene.submit.assert_called_once()


def _expectation():
    """Create two ordered measurement terms with distinguishable coefficients."""
    return HugrTranspiler().transpile(
        _rotated,
        bindings={"observable": qm_o.Z(0) + 2 * qm_o.X(0)},
        parameters=["theta"],
    )


def test_all_expectation_terms_are_recorded_before_partial_submission_failure(
    monkeypatch,
):
    """The complete generation survives a lost second response and first-job cleanup."""
    program = _expectation()
    client = _client()
    executor = _executor(monkeypatch, client)
    observed = []
    client.login.side_effect = lambda: observed.append(program.qir_info)
    client.start_execute_job.side_effect = [
        SimpleNamespace(id="first-job", job_type="execute"),
        TimeoutError("second response lost"),
    ]
    with pytest.raises(TimeoutError):
        program.run(executor, {"theta": 0.3}, estimation=ShotBased(3))
    assert observed == [program.qir_info]
    assert isinstance(program.qir, tuple) and len(program.qir) == 2
    assert program.qir_info["operation"] == "run" and program.qir_info["shots"] is None
    for index, item in enumerate(program.qir_info["submissions"]):
        assert item["qir_sha256"] == hashlib.sha256(program.qir[index]).hexdigest()
        assert item["name"] == client.qir.upload.call_args_list[index].kwargs["name"]
        assert item["context"]["shots"] == "3"
        assert item["context"]["abi_sha256"] and item["context"]["expectation_sha256"]
    assert program.qir[0] != program.qir[1]
    client.jobs.cancel.assert_called_once()


def test_expectation_preparation_failure_retains_whole_previous_generation(monkeypatch):
    """Failing the second conversion cannot replace the record or submit the first term."""
    program = _expectation()
    client = _client()
    executor = _executor(monkeypatch, client)
    program.run(executor, {"theta": 0.3}, shots=2)
    qir, info = program.qir, program.qir_info
    real_convert = _nexus.to_h2_qir
    converted = []

    def convert(package):
        """Allow one new term, then fail before the full generation is ready."""
        if converted:
            raise ExecutionError("second conversion failed")
        converted.append(package)
        return real_convert(package)

    monkeypatch.setattr(_nexus, "to_h2_qir", convert)
    with pytest.raises(ExecutionError, match="second conversion"):
        program.run(executor, {"theta": 0.4}, shots=2)
    assert program.qir == qir and program.qir_info == info
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 2


def test_expectation_reuses_all_terms_and_restores_typed_result_in_order(monkeypatch):
    """A saved multi-term run reconstructs its weighted value without another submission."""
    program = _expectation()
    client = _client()
    sc = _executor(monkeypatch, client, system_name="H2-1SC")
    program.run(sc, {"theta": 0.3}, shots=2)
    qir = program.qir
    _forbid_conversion(monkeypatch)
    hardware = _executor(monkeypatch, client, system_name="H2-1")
    job = program.run(hardware, {"theta": 0.3}, shots=2)
    assert all(a is b for a, b in zip(qir, program.qir, strict=True))
    info = json.loads(json.dumps(program.qir_info))
    assert [item["context"] for item in info["submissions"]] == [
        dict(ref.context) for ref in job.snapshot().executions
    ]
    _readouts(client, info, [False, False])
    positive = client.jobs.results.return_value
    _readouts(client, info, [True, True])
    negative = client.jobs.results.return_value
    client.jobs.results.side_effect = [positive, negative]
    snapshot = _snapshot(info, ["z-job", "x-job"])
    restored = program.restore(hardware, snapshot, {"theta": 0.3})
    assert restored.result() == pytest.approx(-1.0)
    assert [call.kwargs["id"] for call in client.jobs.get.call_args_list] == [
        "z-job",
        "x-job",
    ]
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 4


def test_constant_expectation_creates_no_h2_record_or_remote_call(monkeypatch):
    """An identity-only expectation remains a local typed result."""
    observable = qm_o.Hamiltonian()
    observable.constant = 0.25
    program = HugrTranspiler().transpile(
        _rotated,
        bindings={"observable": observable},
        parameters=["theta"],
    )
    client = _client()
    executor = _executor(monkeypatch, client)
    _forbid_conversion(monkeypatch)
    job = program.run(executor, {"theta": 0.3})
    assert job.result() == pytest.approx(0.25)
    assert program.qir is None and program.qir_info is None
    client.login.assert_not_called()
    client.qir.upload.assert_not_called()


def test_inflight_request_does_not_read_back_later_generation(monkeypatch):
    """Another call can replace latest QIR without changing an active call's payload."""
    program = _program()
    outer_client, inner_client = _client(), _client()
    outer = _executor(monkeypatch, outer_client, system_name="H2-1E", name="outer")
    inner = _executor(monkeypatch, inner_client, system_name="H2-1SC", name="inner")
    initial = []

    def interleave():
        """Generate a second logical operation after the first has become visible."""
        initial.append((program.qir, program.qir_info))
        program.sample(inner, shots=3, bindings={"target": 1})

    outer_client.login.side_effect = interleave
    job = program.sample(outer, shots=2, bindings={"target": 0})
    qir, info = initial[0]
    assert outer_client.qir.upload.call_args.kwargs["qir"] is qir
    assert job.snapshot().executions[0].context == info["submissions"][0]["context"]
    assert job.snapshot().executions[0].target == "H2-1E"
    assert program.qir_info["submissions"][0]["target"] == "H2-1SC"
    assert program.qir_info["shots"] == 3 and program.qir != qir


def test_changed_classical_expectation_binding_regenerates(monkeypatch):
    """A runtime value used only in result reconstruction still invalidates reuse."""
    program = HugrTranspiler().transpile(
        _structured,
        bindings={"observable": qm_o.Z(0)},
        parameters=["theta", "scale"],
    )
    client = _client()
    executor = _executor(monkeypatch, client)
    program.run(executor, bindings={"theta": 0.3, "scale": 1.0}, shots=2)
    info = program.qir_info
    _forbid_conversion(monkeypatch)
    with pytest.raises(AssertionError, match="regenerated"):
        program.run(executor, bindings={"theta": 0.3, "scale": 2.0}, shots=2)
    assert program.qir_info == info
    assert client.qir.upload.call_count == 2


@pytest.mark.parametrize("failure", ["bindings", "shots", "settings"])
def test_invalid_local_arguments_preserve_prior_record(monkeypatch, failure):
    """Validation failure cannot replace QIR even if a provider was used earlier."""
    program = _program()
    client = _client()
    executor = _executor(monkeypatch, client)
    program.sample(executor, shots=2, bindings={"target": 1})
    qir, info = program.qir, program.qir_info
    if failure == "settings":
        executor = _executor(
            monkeypatch,
            client,
            max_cost=1,
            backend_config=SimpleNamespace(
                type="QuantinuumConfig", device_name="H2-1", max_cost=2
            ),
        )
    _forbid_conversion(monkeypatch)
    with pytest.raises(ValueError):
        program.sample(
            executor,
            shots=0 if failure == "shots" else 2,
            bindings={"target": -1 if failure == "bindings" else 1},
        )
    assert program.qir is qir and program.qir_info == info
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 1


def test_run_metadata_restores_scalar_without_sample_count(monkeypatch):
    """Ordinary run keeps one-shot context and reconstructs its scalar return type."""
    program = _program()
    client = _client()
    executor = _executor(monkeypatch, client)
    job = program.run(executor, bindings={"target": 0})
    info = program.qir_info
    assert info["shots"] is None and info["operation"] == "run"
    assert info["submissions"][0]["context"]["shots"] == "1"
    _readouts(client, info, [True])
    assert job.result() is True
    _forbid_conversion(monkeypatch)
    restored = program.restore(executor, _snapshot(info, ["run-id"]), {"target": 0})
    assert restored.result() is True
    assert client.qir.upload.call_count == client.start_execute_job.call_count == 1
