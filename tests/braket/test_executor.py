"""Tests for Braket parameter binding, sampling, and estimation."""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.braket
pytest.importorskip("braket.circuits")

from braket.circuits import Circuit, FreeParameter  # noqa: E402

import qamomile.observable as qm_o  # noqa: E402
from qamomile.braket import BraketExecutor  # noqa: E402
from qamomile.circuit.transpiler.executable import (  # noqa: E402
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.execution_request import (  # noqa: E402
    Exact,
    ShotBased,
)


def test_execute_normalizes_braket_qubit_order() -> None:
    """Sampling returns the highest physical qubit on the left."""
    circuit = Circuit().x(0).i(1).i(2)

    counts = BraketExecutor().execute(circuit, shots=32)

    assert counts == {"001": 32}


def test_zero_qubit_execute_avoids_device_task() -> None:
    """A zero-qubit sample returns immediately without creating a device."""
    assert BraketExecutor().execute(Circuit(), shots=17) == {"": 17}


@pytest.mark.parametrize("angle", [0.0, math.pi / 3, math.pi, 2 * math.pi])
def test_bind_parameters_supports_boundary_angles(angle: float) -> None:
    """Named Braket free parameters bind at common angle boundaries."""
    theta = FreeParameter("theta")
    circuit = Circuit().rx(0, theta)
    metadata = ParameterMetadata(
        parameters=[ParameterInfo("theta", "theta", None, theta)]
    )

    bound = BraketExecutor().bind_parameters(circuit, {"theta": angle}, metadata)

    assert not bound.parameters


def test_exact_expectation_includes_noncommuting_terms_and_constant() -> None:
    """Exact estimation sums non-commuting Pauli terms and a constant."""
    circuit = Circuit().h(0).i(1)
    hamiltonian = 0.75 * qm_o.X(0) + 0.5 * qm_o.Z(1)
    hamiltonian.constant = -0.25

    result = BraketExecutor().estimate(circuit, hamiltonian)

    assert np.isclose(result, 1.0, rtol=0.0, atol=1e-10)


def test_sampled_expectation_runs_each_term() -> None:
    """Shot-based estimation evaluates each Pauli term independently."""
    circuit = Circuit().h(0).i(1)
    hamiltonian = qm_o.X(0) + qm_o.Z(1)

    result = BraketExecutor(estimation_shots=1000).estimate(circuit, hamiltonian)

    assert np.isclose(result, 2.0, rtol=0.0, atol=0.15)


def test_negative_estimation_shots_are_rejected() -> None:
    """Executor construction rejects negative expectation shot counts."""
    with pytest.raises(ValueError, match="non-negative"):
        BraketExecutor(estimation_shots=-1)


def test_fractional_estimation_shots_are_rejected() -> None:
    """Executor construction rejects fractional expectation shot counts."""
    with pytest.raises(ValueError, match="integer"):
        BraketExecutor(estimation_shots=1.5)


def test_local_capabilities_report_native_execution_features() -> None:
    """Local Braket reports async submission, batching, and exact estimation."""
    capabilities = BraketExecutor().capabilities

    assert capabilities.supports_async_sampling
    assert capabilities.supports_async_estimation
    assert capabilities.supports_estimation
    assert capabilities.supports_native_batch
    assert capabilities.supports_native_parameter_inputs
    assert not capabilities.supports_restoration
    assert capabilities.estimation_accuracy == frozenset({Exact, ShotBased})
