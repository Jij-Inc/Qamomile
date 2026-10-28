"""Restore public jobs after local SDK results cross the native-value boundary."""

from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.circuit.transpiler import JobSnapshot  # noqa: E402
from qamomile.qiskit import QiskitExecutor, QiskitTranspiler  # noqa: E402


@qmc.qkernel
def _sample_pair(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare a rotated bit and an independent zero bit.

    Args:
        theta (qmc.Float): Rotation angle in radians.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Rotated bit followed by the zero bit.
    """
    first = qmc.qubit("first")
    second = qmc.qubit("second")
    first = qmc.ry(first, theta)
    return qmc.measure(first), qmc.measure(second)


@qmc.qkernel
def _local_expectation(theta: qmc.Float, observable: qmc.Observable) -> qmc.Float:
    """Estimate an observable on one rotated qubit.

    Args:
        theta (qmc.Float): Rotation angle in radians.
        observable (qmc.Observable): One-qubit observable.

    Returns:
        qmc.Float: Scalar expectation value.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.ry(qubit, theta)
    return qmc.expval(qubit, observable)


@pytest.fixture(params=["basic", "aer", "aer_from_backend"])
def local_executor(request: pytest.FixtureRequest) -> QiskitExecutor:
    """Create real local backends whose SDK sampling results contain Counts.

    Args:
        request (pytest.FixtureRequest): Selected local backend variant.

    Returns:
        QiskitExecutor: Backend adapter with deterministic simulator settings.
    """
    from qiskit.providers.basic_provider import BasicSimulator

    if request.param == "basic":
        backend: Any = BasicSimulator(seed_simulator=19)
    else:
        aer = pytest.importorskip("qiskit_aer")
        if request.param == "aer":
            backend = aer.AerSimulator(seed_simulator=19)
        else:
            from qiskit.providers.fake_provider import GenericBackendV2

            target = GenericBackendV2(num_qubits=2, noise_info=False, seed=23)
            with pytest.warns(UserWarning, match="has no QubitProperties"):
                backend = aer.AerSimulator.from_backend(target, seed_simulator=19)
    return QiskitExecutor(backend)


@pytest.mark.parametrize("theta,expected_bit", [(0.0, 0), (math.pi, 1)])
def test_local_sample_snapshot_restores_native_counts_without_resubmission(
    local_executor: QiskitExecutor,
    monkeypatch: pytest.MonkeyPatch,
    theta: float,
    expected_bit: int,
) -> None:
    """Ry(0)|0> and Ry(pi)|0> give deterministic typed results on real simulators."""
    program = QiskitTranspiler().transpile(_sample_pair, parameters=["theta"])
    bindings = {"theta": theta}
    original = program.sample(local_executor, shots=17, bindings=bindings)
    snapshot = original.snapshot()
    assert snapshot.executions == ()
    assert snapshot.execution is not None
    assert type(snapshot.execution.value) is dict
    assert all(type(key) is str for key in snapshot.execution.value)
    assert all(type(count) is int for count in snapshot.execution.value.values())
    payload = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))

    submit = Mock(side_effect=AssertionError("Restoration must not submit"))
    lookup = Mock(side_effect=AssertionError("Local results need no provider lookup"))
    monkeypatch.setattr(local_executor, "submit_sample", submit)
    monkeypatch.setattr(local_executor, "restore", lookup)
    restored = program.restore(local_executor, JobSnapshot.from_dict(payload), bindings)

    assert type(restored) is type(original)
    assert restored.result() == original.result()
    assert restored.result().results == [((expected_bit, 0), 17)]
    assert restored.result().shots == 17
    values, count = restored.result().results[0]
    assert type(values) is tuple
    assert all(type(value) is int for value in values)
    assert type(count) is int
    submit.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "observable,expected", [(qm_o.Z(0), -1.0), (qm_o.Hamiltonian.identity(0.25), 0.25)]
)
def test_local_estimator_snapshots_keep_native_float_values(
    observable: qm_o.Hamiltonian,
    expected: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ry(pi)|0> has Z expectation -1; a constant identity retains its coefficient."""
    program = QiskitTranspiler().transpile(
        _local_expectation,
        bindings={"observable": observable},
        parameters=["theta"],
    )
    executor = QiskitExecutor()
    bindings = {"theta": math.pi}
    original = program.run(executor, bindings=bindings)
    snapshot = original.snapshot()
    assert snapshot.execution is not None
    assert type(snapshot.execution.children[0].value) is float
    submit = Mock(side_effect=AssertionError("Restoration must not estimate"))
    lookup = Mock(side_effect=AssertionError("Local results need no provider lookup"))
    monkeypatch.setattr(executor, "submit_estimates", submit)
    monkeypatch.setattr(executor, "restore", lookup)

    restored = program.restore(
        executor,
        JobSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict()))),
        bindings,
    )

    assert type(restored) is type(original)
    assert type(restored.result()) is float
    np.testing.assert_allclose(
        [original.result(), restored.result()],
        [expected, expected],
        atol=1e-12,
        rtol=0.0,
    )
    submit.assert_not_called()
    lookup.assert_not_called()
