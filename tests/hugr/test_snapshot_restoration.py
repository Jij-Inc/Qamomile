"""Preserve HUGR local rows, remote references, and reconstruction identity."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

import qamomile.observable as qm_o
from qamomile.circuit.transpiler import (
    CompletedExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    Job,
    JobSnapshot,
    JobStatus,
    RunJob,
)
from qamomile.circuit.transpiler.execution_snapshot import ExecutionSnapshot
from qamomile.hugr import HugrExecutor, HugrTranspiler
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._runtime import prepare_execution
from qamomile.hugr.executable import _DecodedExecutionHandle
from tests.hugr.test_execution import _complex, _remote_executor
from tests.hugr.test_expval import _mock_helios_client, _rotated
from tests.hugr.test_expval_workflows import _array_postprocessing

pytestmark = pytest.mark.hugr


def _round_trip(job: Job[Any]) -> JobSnapshot:
    """Exercise strict JSON serialization independently of in-memory values.

    Args:
        job (Job[Any]): Public job whose restoration metadata is serialized.

    Returns:
        JobSnapshot: Independently decoded restoration metadata.

    Raises:
        TypeError: If a saved result has an unsupported type.
        ValueError: If a snapshot or its JSON representation is invalid.
    """
    return JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))


def test_decoded_handle_annotates_snapshot_only_remote_source():
    """A custom source can supply its reference through the structured API only."""
    reference = ExecutionReference("custom", ("job-1",), context={"shots": "1"})

    class SnapshotOnlyHandle(ExecutionHandle):
        """Expose remote restoration without the legacy inventory method."""

        def result(self, timeout: float | None = None) -> Any:
            """Reject retrieval while testing snapshot construction.

            Args:
                timeout (float | None): Ignored wait timeout.

            Raises:
                AssertionError: Always, because snapshotting must not retrieve.
            """
            raise AssertionError("Snapshot must not fetch a remote result")

        def status(self) -> JobStatus:
            """Report an unfinished execution.

            Returns:
                JobStatus: Running state without accessing provider results.
            """
            return JobStatus.RUNNING

        def snapshot(self) -> ExecutionSnapshot:
            """Expose a structured reference without fetching a result.

            Returns:
                ExecutionSnapshot: One remote leaf for the fixture reference.
            """
            return ExecutionSnapshot("remote", reference=reference)

    compiled = HugrTranspiler().compile(_complex, parameters=["theta", "label"])
    prepared = prepare_execution(compiled, {"theta": 1.0, "label": 8})
    snapshot = _DecodedExecutionHandle(SnapshotOnlyHandle(), prepared, 1).snapshot()
    assert snapshot.reference.context == {
        **reference.context,
        "abi_sha256": prepared.abi_sha256,
    }
    assert snapshot.reference.job_ids == reference.job_ids


@pytest.mark.parametrize("operation", ["run", "sample"])
def test_local_tagged_rows_restore_typed_results_without_executor_calls(operation):
    """Local raw rows retain shape and types while restore rebuilds decoding."""
    program = HugrTranspiler().transpile(_complex, parameters=["theta", "label"])
    rows = [
        {
            "qamomile.output.0.0": True,
            "qamomile.output.0.1": False,
            "qamomile.output.0.2": True,
            "qamomile.output.1.0": 0.5,
            "qamomile.output.2.0": 9,
        }
    ]
    shots = 1 if operation == "run" else 3
    executor = Mock(spec=HugrExecutor)
    executor.submit.return_value = CompletedExecutionHandle(rows * shots)
    bindings = {"theta": 1.0, "label": 8}
    job = (
        program.run(executor, bindings)
        if operation == "run"
        else program.sample(executor, shots, bindings)
    )
    saved = _round_trip(job)
    assert saved.executions == ()
    assert saved.execution.kind == "local"
    restored_executor = Mock(spec=HugrExecutor)
    restored = program.restore(restored_executor, saved, bindings)
    result = restored.result()
    original_result = job.result()
    value = result if operation == "run" else result.results[0][0]
    original_value = (
        original_result if operation == "run" else original_result.results[0][0]
    )
    assert value[0] == original_value[0] == (True, False, True)
    # The fixture emits theta / 2 = .5 and label + 1 = 9.
    np.testing.assert_allclose(
        [value[1], original_value[1]], [0.5, 0.5], atol=1e-12, rtol=1e-12
    )
    assert value[2] == original_value[2] == 9
    assert type(restored) is type(job)
    if operation == "sample":
        assert result.shots == original_result.shots == shots
        assert [count for _, count in result.results] == [shots]
        assert [count for _, count in original_result.results] == [shots]
    assert type(value) is tuple
    assert type(value[0]) is tuple
    assert type(value[0][0]) is bool
    assert type(value[1]) is float
    assert type(value[2]) is int
    assert restored_executor.mock_calls == []
    assert _round_trip(restored).to_dict() == saved.to_dict()
    with pytest.raises(ValueError, match="program and runtime bindings"):
        program.restore(restored_executor, saved, {"theta": 2.0, "label": 8})


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("operation", ["run", "sample"])
def test_remote_snapshots_keep_fingerprints_and_legacy_restore(
    monkeypatch, legacy, operation
):
    """JSON snapshots preserve remote ABI context and old flat references."""
    program = HugrTranspiler().transpile(_complex, parameters=["theta", "label"])
    record = [
        ("qamomile.output.0.0", True),
        ("qamomile.output.0.1", False),
        ("qamomile.output.0.2", True),
        ("qamomile.output.1.0", 0.5),
        ("qamomile.output.2.0", 9),
    ]
    shots = 1 if operation == "run" else 3
    executor, client = _remote_executor(monkeypatch, [record] * shots)
    bindings = {"theta": 1.0, "label": 8}
    job = (
        program.run(executor, bindings)
        if operation == "run"
        else program.sample(executor, shots, bindings)
    )
    saved = _round_trip(job)
    if legacy:
        saved = replace(saved, execution=None)
    assert saved.executions[0].context["abi_sha256"]
    assert saved.executions[0].context["package_sha256"]
    client.jobs.results.assert_not_called()
    restored = program.restore(executor, saved, bindings)
    client.jobs.results.assert_not_called()
    result = restored.result()
    original_result = job.result()
    value = result if operation == "run" else result.results[0][0]
    original_value = (
        original_result if operation == "run" else original_result.results[0][0]
    )
    assert type(restored) is type(job)
    assert type(value) is tuple
    assert type(value[0]) is tuple
    assert all(type(bit) is bool for bit in value[0])
    assert type(value[1]) is float
    assert type(value[2]) is int
    assert value[0] == original_value[0] == (True, False, True)
    np.testing.assert_allclose(
        [value[1], original_value[1]], [0.5, 0.5], atol=1e-12, rtol=1e-12
    )
    assert value[2] == original_value[2] == 9
    if operation == "sample":
        assert result.shots == original_result.shots == shots
        assert [count for _, count in result.results] == [shots]
        assert [count for _, count in original_result.results] == [shots]
    client.start_execute_job.assert_called_once()


@pytest.mark.parametrize("local_first", [False, True])
def test_mixed_term_snapshots_keep_order_without_waiting_or_resubmitting(
    monkeypatch, local_first
):
    """A local Pauli result and remote term retain their own weighted position."""
    program = HugrTranspiler().transpile(
        _rotated,
        bindings={"observable": 2.0 * qm_o.X(0) + qm_o.Z(0) + 0.25},
        parameters=["theta"],
    )
    client = _mock_helios_client([[(0,)]])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor("helios")
    submit = executor.submit
    calls = []

    def mixed_submit(package: Any, shots: int) -> ExecutionHandle[Any]:
        """Alternate one completed local term with one deferred remote term.

        Args:
            package (Any): Prepared HUGR package submitted for one Pauli term.
            shots (int): Number of requested term measurements.

        Returns:
            ExecutionHandle[Any]: Local tagged rows or deferred provider handle.

        Raises:
            Exception: If the mock remote executor rejects the package.
        """
        calls.append(package)
        if (len(calls) == 1) == local_first:
            return CompletedExecutionHandle([{"qamomile.output.0.0": True}])
        return submit(package, shots)

    monkeypatch.setattr(executor, "submit", mixed_submit)
    job = program.run(executor, {"theta": 0.0}, shots=1)
    saved = _round_trip(job)
    assert len(saved.executions) == 1
    assert [node.kind for node in saved.execution.children] == (
        ["local", "remote"] if local_first else ["remote", "local"]
    )
    client.jobs.results.assert_not_called()
    restored = program.restore(executor, saved, {"theta": 0.0})
    client.jobs.results.assert_not_called()
    # Tagged True has eigenvalue -1; remote zero has eigenvalue +1.
    # Terms are 2X then Z, so the two orderings give -2 + 1 + .25 or 2 - 1 + .25.
    expected = -0.75 if local_first else 1.25
    np.testing.assert_allclose(
        [restored.result(), job.result()], [expected, expected], atol=1e-12, rtol=1e-12
    )
    assert type(restored) is type(job)
    assert type(restored.result()) is float
    assert len(calls) == 2
    client.start_execute_job.assert_called_once()


@pytest.mark.parametrize("structured", [False, True])
def test_constant_only_expectations_restore_without_execution(structured):
    """Constant optimizations preserve scalar and classically processed results."""
    kernel = _array_postprocessing if structured else _rotated
    parameters = ["values"] if structured else ["theta"]
    arguments = {"parameter_shapes": {"values": (2,)}} if structured else {}
    observable = qm_o.Hamiltonian()
    observable.constant = 0.25
    program = HugrTranspiler().transpile(
        kernel,
        bindings={"observable": observable},
        parameters=parameters,
        **arguments,
    )
    bindings = {"values": [0.3, 2.0]} if structured else {"theta": 0.3}
    executor = Mock(spec=HugrExecutor)
    job = program.run(executor, bindings, shots=1)
    saved = _round_trip(job)
    restored = program.restore(executor, saved, bindings)
    # The identity yields .25; postprocessing adds 2 and retains the .3 angle.
    expected = (2.25, 0.25, 0.3) if structured else 0.25
    np.testing.assert_allclose(restored.result(), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(job.result(), expected, atol=1e-12, rtol=1e-12)
    assert type(restored) is type(job)
    assert saved.executions == ()
    assert executor.mock_calls == []
    assert "values" not in saved.execution.value
    assert "theta" not in saved.execution.value
    if structured:
        assert isinstance(restored, RunJob)
        assert type(restored.result()) is tuple
        equivalent = {"values[0]": 0.3, "values[1]": 2.0}
        np.testing.assert_allclose(
            program.restore(executor, saved, equivalent).result(),
            expected,
            atol=1e-12,
            rtol=1e-12,
        )
        assert all(type(value) is float for value in restored.result())
    else:
        assert type(restored.result()) is float
    changed = {"values": [0.5, 2.0]} if structured else {"theta": 0.5}
    with pytest.raises(ValueError, match="runtime bindings"):
        program.restore(executor, saved, changed)
    corrupt = replace(saved.execution, value={**saved.execution.value, "value": 9.0})
    with pytest.raises(ValueError, match="value does not match"):
        program.restore(executor, replace(saved, execution=corrupt), bindings)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("hugr_kind", "unknown", "tagged rows"),
        ("rows", [], "expected 1"),
        ("rows", [{"wrong": True}], "Missing HUGR output tags"),
        ("shots", "2", "shot count"),
        ("abi_sha256", "bad", "public ABI"),
        ("package_sha256", "bad", "runtime bindings"),
        ("unexpected", "bad", "context fields"),
    ],
)
def test_corrupt_local_envelopes_fail_without_provider_calls(field, value, message):
    """Malformed raw rows and restoration context cannot produce partial results."""
    program = HugrTranspiler().transpile(_complex, parameters=["theta", "label"])
    executor = Mock(spec=HugrExecutor)
    executor.submit.return_value = CompletedExecutionHandle([])
    bindings = {"theta": 1.0, "label": 8}
    saved = _round_trip(program.run(executor, bindings))
    corrupt = replace(saved.execution, value={**saved.execution.value, field: value})
    restored_executor = Mock(spec=HugrExecutor)
    with pytest.raises((ValueError, RuntimeError), match=message):
        program.restore(restored_executor, replace(saved, execution=corrupt), bindings)
    assert restored_executor.mock_calls == []


def test_hugr_rejects_nested_execution_groups_without_retrieval(monkeypatch):
    """HUGR's known term layout rejects nested groups rather than flattening them."""
    program = HugrTranspiler().transpile(
        _rotated, bindings={"observable": qm_o.Z(0)}, parameters=["theta"]
    )
    client = _mock_helios_client([[(0,)]])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor("helios")
    saved = _round_trip(program.run(executor, {"theta": 0.0}, shots=1))
    nested = ExecutionSnapshot("composite", children=(saved.execution,))
    with pytest.raises(ValueError, match="execution layout"):
        program.restore(executor, replace(saved, execution=nested), {"theta": 0.0})
    client.jobs.get.assert_not_called()


def test_restore_revalidates_mutated_local_snapshot_values():
    """Mutable containers cannot bypass the supported-value boundary on restore."""
    program = HugrTranspiler().transpile(_complex, parameters=["theta", "label"])
    executor = Mock(spec=HugrExecutor)
    executor.submit.return_value = CompletedExecutionHandle([])
    bindings = {"theta": 1.0, "label": 8}
    saved = _round_trip(program.run(executor, bindings))
    saved.execution.value["rows"] = [{"qamomile.output.0.0": object()}]
    restored_executor = Mock(spec=HugrExecutor)
    with pytest.raises(TypeError, match="unsupported local result type"):
        program.restore(restored_executor, saved, bindings)
    assert restored_executor.mock_calls == []
