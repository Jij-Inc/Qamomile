"""Verify shared execution policies and typed sample semantics for HUGR."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import qamomile.observable as qm_o
from qamomile.circuit.transpiler import (
    CompletedExecutionHandle,
    Exact,
    JobSnapshot,
    ShotBased,
    TargetPrecision,
)
from qamomile.circuit.transpiler.job import SampleJob
from qamomile.hugr import HugrExecutor, HugrTranspiler
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr.executable import _HugrSampleJob
from tests.hugr.test_expval import _measurement, _mock_helios_client, _rotated

pytestmark = pytest.mark.hugr


@pytest.fixture(scope="module")
def energy_program():
    """Compile a scalar expectation with one runtime rotation parameter.

    Returns:
        HugrExecutable: Executable estimating Z in the rotated state.
    """
    pytest.importorskip("hugr")
    return HugrTranspiler().transpile(
        _rotated, bindings={"observable": qm_o.Z(0)}, parameters=["theta"]
    )


@pytest.mark.parametrize(
    "arguments,shots",
    [({}, 1024), ({"shots": 7}, 7), ({"estimation": ShotBased(5)}, 5)],
)
def test_shared_shot_policy_submits_and_restores(
    monkeypatch, energy_program, arguments, shots
):
    """Default, legacy, and shared policies preserve the shot count on restore."""
    client = _mock_helios_client([[(0,)] * shots])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor("helios")
    job = energy_program.run(executor, bindings={"theta": 0.0}, **arguments)
    assert client.start_execute_job.call_args.kwargs["n_shots"] == [shots]
    assert job.result() == pytest.approx(1.0, abs=1e-12)
    saved = JobSnapshot.from_dict(job.snapshot().to_dict())
    assert energy_program.restore(
        executor, saved, {"theta": 0.0}
    ).result() == pytest.approx(1.0, abs=1e-12)
    client.start_execute_job.assert_called_once()


@pytest.mark.parametrize(
    "arguments,error,message",
    [
        ({"estimation": Exact()}, NotImplementedError, "Exact"),
        ({"estimation": TargetPrecision(0.1)}, NotImplementedError, "TargetPrecision"),
        ({"estimation": "exact"}, TypeError, "EstimationAccuracy"),
        ({"estimation": ShotBased(3), "shots": 3}, ValueError, "either"),
        ({"shots": 0}, ValueError, "positive"),
        ({"shots": True}, ValueError, "positive"),
    ],
)
def test_invalid_estimation_has_no_provider_effects(
    energy_program, arguments, error, message
):
    """Unsupported and ambiguous policies fail before uploads or submissions."""
    executor = Mock(spec=HugrExecutor)
    with pytest.raises(error, match=message):
        energy_program.run(executor, bindings={"theta": 0.0}, **arguments)
    assert executor.mock_calls == []


@pytest.mark.parametrize("target", ["selene", "helios"])
def test_capabilities_declare_supported_accuracy(target):
    """Both destinations advertise only their implemented shot estimator."""
    capabilities = HugrExecutor(target).capabilities
    assert capabilities.supports_estimation
    assert capabilities.estimation_accuracy == frozenset({ShotBased})


def test_ordinary_run_ignores_unused_estimation(monkeypatch):
    """Shared accuracy policies do not change ordinary one-shot execution."""
    client = _mock_helios_client([[(0,)]])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    program = HugrTranspiler().transpile(_measurement)
    assert program.run(HugrExecutor("helios"), estimation=Exact()).result() is False
    assert client.start_execute_job.call_args.kwargs["n_shots"] == [1]
    assert not program.has_parameters
    assert program.parameter_names == []


def test_native_parameter_names_are_defensive(energy_program):
    """Expose root ABI parameter names without allowing metadata mutation."""
    assert energy_program.has_parameters
    names = energy_program.parameter_names
    assert names == ["theta"]
    names.clear()
    assert energy_program.parameter_names == ["theta"]


def test_cancellation_errors_preserve_submission_failure(monkeypatch):
    """Cancellation is attempted for every term and errors annotate the cause."""
    client = _mock_helios_client([[(0,)], [(0,)], [(0,)]])
    failure = RuntimeError("third term submission failed")
    first = SimpleNamespace(id="term-0", job_type="execute")
    second = SimpleNamespace(id="term-1", job_type="execute")
    client.start_execute_job.side_effect = [first, second, failure]
    client.jobs.cancel.side_effect = [
        TimeoutError("cancel timed out"),
        ValueError("cancel rejected"),
    ]
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    program = HugrTranspiler().transpile(
        _rotated,
        bindings={"observable": qm_o.X(0) + qm_o.Y(0) + qm_o.Z(0)},
        parameters=["theta"],
    )
    with pytest.raises(RuntimeError, match="third term submission failed") as caught:
        program.run(HugrExecutor("helios"), {"theta": 0.0}, estimation=ShotBased(1))
    assert caught.value is failure
    assert [call.args[0] for call in client.jobs.cancel.call_args_list] == [
        second,
        first,
    ]
    assert len(failure.__notes__) == 2
    assert "cancel timed out" in failure.__notes__[0]
    assert "cancel rejected" in failure.__notes__[1]


@pytest.mark.parametrize(
    "shots", [pytest.param(..., id="missing"), "", "bad", "True", "0", "-1", "1.5"]
)
@pytest.mark.parametrize("legacy", [False, True])
def test_restore_rejects_malformed_shots_before_retrieval(
    monkeypatch, energy_program, shots, legacy
):
    """Malformed snapshot shot metadata raises a chained public ValueError."""
    client = _mock_helios_client([[(0,)]])
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor("helios")
    snapshot = energy_program.run(executor, {"theta": 0.0}, shots=1).snapshot()
    reference = snapshot.executions[0]
    context = dict(reference.context)
    if shots is ...:
        del context["shots"]
    else:
        context["shots"] = shots
    reference = replace(reference, context=context)
    execution = (
        None
        if legacy
        else replace(
            snapshot.execution,
            children=(replace(snapshot.execution.children[0], reference=reference),),
        )
    )
    malformed = replace(snapshot, executions=(reference,), execution=execution)
    with pytest.raises(ValueError, match="Snapshot shots") as caught:
        energy_program.restore(executor, malformed, {"theta": 0.0})
    assert isinstance(caught.value.__cause__, (KeyError, TypeError, ValueError))
    client.jobs.get.assert_not_called()


@pytest.mark.parametrize("native", [False, True], ids=["circuit-job", "hugr-job"])
def test_typed_aggregation_matches_across_job_families(native):
    """Merge equal structures while preserving scalar types and empty shapes."""
    first = {"bits": [False, True], "empty": np.empty((0, 2), dtype=np.int32)}
    values = [
        first,
        {"empty": np.empty((0, 2), dtype=np.int32), "bits": [False, True]},
        True,
        1,
        (1,),
        [1],
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 2), dtype=np.int64),
    ]
    if native:
        job = _HugrSampleJob(CompletedExecutionHandle(values), shots=len(values))
    else:
        job = SampleJob(
            {"0": len(values)},
            lambda _: [(value, 1) for value in values],
            shots=len(values),
        )
    result = job.result()
    assert result.shots == len(values)
    assert [count for _, count in result.results] == [2, 1, 1, 1, 1, 1, 1]
    assert result.results[0][0] is first
    for (value, _), original in zip(result.results, [first, *values[2:]], strict=True):
        assert value is original
