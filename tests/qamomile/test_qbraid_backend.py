"""Provider-boundary tests for the Qamomile qBraid endpoint."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

import qamomile as qm
from qamomile.compilation import ExecutionContract, Invocation, JobStatus
from qamomile.errors import ExecutionError
from qamomile.persistence import serialize
from qamomile.qbraid import QBraidDevice

from .conftest import TYPED_MAP_SOURCE

_SOURCE = """
import qamomile as qm

@qm.qkernel
def asymmetric(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.ry(state, theta, target=0)
    return qm.measure(state)
"""

_UINT_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(value: qm.Param[qm.UInt[3]]) -> qm.Shot[qm.UInt[3]]:
    return qm.measure(qm.prepare(value))
"""


def _program() -> qm.Program:
    """Build the smallest runtime-parametric asymmetric sampling program."""

    return qm.build_source(_SOURCE, module_name="qbraid_endpoint").specialize(
        "asymmetric"
    )


def _device(counts: Mapping[str, object] | None = None) -> tuple[MagicMock, MagicMock]:
    """Return a qBraid-shaped mock device and job."""

    device = MagicMock()
    device.id = "qbraid:test:device"
    job = MagicMock()
    job.id = "job-1"
    job.status.return_value = "QUEUED"
    result = MagicMock()
    result.data.get_counts.return_value = counts or {"01": 4}
    job.result.return_value = result
    device.run.return_value = job
    return device, job


def test_qbraid_executes_qpy_with_typed_source_order() -> None:
    """One QPY artifact binds parameters and decodes asymmetric qBraid counts."""

    device, native_job = _device()
    endpoint = QBraidDevice(
        device,
        poll_interval_seconds=0.25,
        wait_timeout_seconds=12.0,
        run_options={"name": "qamomile-test"},
    )
    program = _program()
    artifact = qm.compile(program, to=endpoint.target)

    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"theta": math.pi},
        shots=4,
        seed=0,
    )

    assert record.artifact is artifact
    assert serialize(artifact) == serialize(qm.compile(program, to=endpoint.target))
    assert record.result.counts == {(1, 0): 4}
    assert record.result.backend == "qbraid.device.qbraid:test:device"
    submitted_circuit = device.run.call_args.args[0]
    assert not submitted_circuit.parameters
    assert device.run.call_args.kwargs == {"shots": 4, "name": "qamomile-test"}
    native_job.wait_for_final_state.assert_called_once_with(
        timeout=12.0,
        poll_interval=0.25,
    )


def test_qbraid_measures_unsigned_integer_registers() -> None:
    """qBraid preserves integer launch and Qiskit-native result ordering."""

    device, _ = _device({"011": 4})
    endpoint = QBraidDevice(device)
    program = qm.build_source(
        _UINT_SOURCE,
        module_name="qbraid_uint",
    ).specialize("sample")

    result = qm.run(
        program,
        on=endpoint,
        parameters={"value": 6},
        shots=4,
        seed=0,
    ).result

    assert result.counts == {(1, 1, 0): 4}
    assert result.values == {6: 4}
    assert result.most_likely_value == 6


def test_qbraid_executes_specialized_immutable_mapping() -> None:
    """qBraid receives QPY only after typed Map iteration and lookup resolve."""

    device, _ = _device({"01": 4})
    endpoint = QBraidDevice(device)
    program = qm.build_source(TYPED_MAP_SOURCE).specialize(
        "typed_map",
        **{
            "switches": {0: True, 1: False},
            "enabled": {0: True, 1: True},
        },
    )

    result = qm.run(program, on=endpoint, shots=4, seed=0).result

    assert result.counts == {(1, 0): 4}


@pytest.mark.parametrize(
    ("native", "portable"),
    [
        ("PENDING", JobStatus.SUBMITTED),
        ("RUNNING", JobStatus.RUNNING),
        ("COMPLETED", JobStatus.COMPLETED),
        ("CANCELLED", JobStatus.CANCELLED),
        ("FAILED", JobStatus.FAILED),
    ],
)
def test_qbraid_maps_provider_status(native: str, portable: JobStatus) -> None:
    """qBraid lifecycle values map to the closed portable job states."""

    device, native_job = _device()
    native_job.status.return_value = native
    endpoint = QBraidDevice(device)
    artifact = qm.compile(_program(), to=endpoint.target)
    job = endpoint.launch(
        artifact,
        Invocation.create(parameters={"theta": 0.0}, shots=1),
    )

    assert job.job_id == "job-1"
    assert job.status is portable


def test_qbraid_rejects_nonportable_seed_before_submission() -> None:
    """Provider seed ambiguity fails before a remote side effect."""

    device, _ = _device()
    endpoint = QBraidDevice(device)
    artifact = qm.compile(_program(), to=endpoint.target)

    with pytest.raises(ExecutionError, match="Q8803"):
        qm.run(
            artifact,
            on=endpoint,
            parameters={"theta": 0.0},
            shots=1,
            seed=1,
        )

    device.run.assert_not_called()


def test_qbraid_rejects_wrong_artifact_format() -> None:
    """The endpoint never interprets a non-QPY payload."""

    device, _ = _device()
    endpoint = QBraidDevice(device)
    artifact = qm.compile(_program(), to=endpoint.target)
    incompatible = replace(
        artifact,
        execution_contract=ExecutionContract(
            "not.qiskit",
            artifact.execution_contract.requirements,
        ),
    )

    with pytest.raises(ExecutionError, match="Q8801"):
        endpoint.launch(
            incompatible,
            Invocation.create(parameters={"theta": 0.0}, shots=1),
        )

    device.run.assert_not_called()


@pytest.mark.parametrize("counts", [{"02": 1}, {"01": 1.5}])
def test_qbraid_rejects_malformed_counts(counts: dict[str, object]) -> None:
    """Non-binary keys and non-integral counts fail at the provider boundary."""

    device, _ = _device(counts)
    endpoint = QBraidDevice(device)
    artifact = qm.compile(_program(), to=endpoint.target)
    job = endpoint.launch(
        artifact,
        Invocation.create(parameters={"theta": 0.0}, shots=1),
    )

    with pytest.raises(ExecutionError, match="Q8808"):
        job.result()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"poll_interval_seconds": 0.0},
        {"wait_timeout_seconds": 0.0},
        {"run_options": {"shots": 3}},
    ],
)
def test_qbraid_rejects_invalid_endpoint_policy(kwargs: dict[str, object]) -> None:
    """Invalid local policy fails before a provider call."""

    device, _ = _device()

    with pytest.raises(ValueError):
        QBraidDevice(device, **kwargs)


def test_qbraid_resolves_a_device_without_retaining_provider() -> None:
    """The convenience constructor stores only the resolved live device."""

    provider = MagicMock()
    device, _ = _device()
    provider.get_device.return_value = device

    endpoint = QBraidDevice.from_device_id("qbraid:test:device", provider=provider)

    provider.get_device.assert_called_once_with("qbraid:test:device")
    assert endpoint.device is device


def test_qbraid_missing_optional_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing qBraid dependencies produce actionable install guidance."""

    monkeypatch.setitem(sys.modules, "qbraid", None)

    with pytest.raises(ImportError, match=r"qamomile\[qbraid\]"):
        QBraidDevice.from_device_id("qbraid:test:device")
