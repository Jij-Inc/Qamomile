"""Tests for QBraidExecutor constructor, execute, and shared wait helper."""

from unittest.mock import MagicMock, patch

import pytest
from qiskit import QuantumCircuit

from qamomile.qbraid.executor import QBraidExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_device(counts: dict[str, int] | None = None):
    """Create a mock qBraid QuantumDevice that returns given counts."""
    if counts is None:
        counts = {"00": 50, "11": 50}

    device = MagicMock()
    job = MagicMock()
    result = MagicMock()
    result.data.get_counts.return_value = counts

    device.run.return_value = job
    job.wait_for_final_state.return_value = None
    job.result.return_value = result

    return device, job, result


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_device_only(self):
        device, _, _ = _mock_device()
        executor = QBraidExecutor(device=device)
        assert executor.device is device

    def test_device_with_device_id_raises(self):
        device, _, _ = _mock_device()
        with pytest.raises(ValueError, match="must not be specified"):
            QBraidExecutor(device=device, device_id="some_id")

    def test_device_with_provider_raises(self):
        device, _, _ = _mock_device()
        with pytest.raises(ValueError, match="must not be specified"):
            QBraidExecutor(device=device, provider=MagicMock())

    def test_device_with_api_key_raises(self):
        device, _, _ = _mock_device()
        with pytest.raises(ValueError, match="must not be specified"):
            QBraidExecutor(device=device, api_key="key")

    def test_no_device_no_device_id_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            QBraidExecutor()

    def test_provider_only_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            QBraidExecutor(provider=MagicMock())

    def test_api_key_only_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            QBraidExecutor(api_key="key")

    def test_device_id_with_provider(self):
        provider = MagicMock()
        mock_device, _, _ = _mock_device()
        provider.get_device.return_value = mock_device

        executor = QBraidExecutor(device_id="dev123", provider=provider)
        provider.get_device.assert_called_once_with("dev123")
        assert executor.device is mock_device

    @patch("qamomile.qbraid.executor.QBraidExecutor._resolve_device")
    def test_device_id_only(self, mock_resolve):
        mock_dev, _, _ = _mock_device()
        mock_resolve.return_value = mock_dev

        executor = QBraidExecutor(device_id="dev123")
        mock_resolve.assert_called_once_with("dev123", None, None)
        assert executor.device is mock_dev

    @patch("qamomile.qbraid.executor.QBraidExecutor._resolve_device")
    def test_device_id_with_api_key(self, mock_resolve):
        mock_dev, _, _ = _mock_device()
        mock_resolve.return_value = mock_dev

        QBraidExecutor(device_id="dev123", api_key="mykey")
        mock_resolve.assert_called_once_with("dev123", None, "mykey")

    def test_default_params(self):
        device, _, _ = _mock_device()
        executor = QBraidExecutor(device=device)
        assert executor.expval_shots == 4096
        assert executor.timeout is None
        assert executor.poll_interval == 5
        assert executor.run_kwargs == {}

    def test_custom_params(self):
        device, _, _ = _mock_device()
        executor = QBraidExecutor(
            device=device,
            expval_shots=2048,
            timeout=120,
            poll_interval=10,
            run_kwargs={"name": "test"},
        )
        assert executor.expval_shots == 2048
        assert executor.timeout == 120
        assert executor.poll_interval == 10
        assert executor.run_kwargs == {"name": "test"}

    def test_run_kwargs_shots_rejected(self):
        """run_kwargs must not contain 'shots' — it collides with execute()."""
        device, _, _ = _mock_device()
        with pytest.raises(ValueError, match="reserved key"):
            QBraidExecutor(device=device, run_kwargs={"shots": 100})

    def test_run_kwargs_shots_with_other_keys_rejected(self):
        """run_kwargs with 'shots' among other keys is still rejected."""
        device, _, _ = _mock_device()
        with pytest.raises(ValueError, match="shots"):
            QBraidExecutor(device=device, run_kwargs={"name": "job", "shots": 50})


# ---------------------------------------------------------------------------
# Shared wait helper
# ---------------------------------------------------------------------------


class TestSubmitAndWait:
    def test_call_order(self):
        """wait_for_final_state must be called before result()."""
        device, job, result = _mock_device({"0": 100})
        executor = QBraidExecutor(device=device, timeout=60, poll_interval=3)

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        counts = executor._submit_and_wait(qc, shots=100)

        device.run.assert_called_once()
        job.wait_for_final_state.assert_called_once_with(timeout=60, poll_interval=3)
        job.result.assert_called_once()
        assert counts == {"0": 100}

    def test_poll_interval_forwarded(self):
        """poll_interval must reach wait_for_final_state."""
        device, job, _ = _mock_device()
        executor = QBraidExecutor(device=device, poll_interval=7)

        qc = QuantumCircuit(1)
        qc.measure_all()
        executor._submit_and_wait(qc, shots=10)

        _, kwargs = job.wait_for_final_state.call_args
        assert kwargs["poll_interval"] == 7

    def test_run_kwargs_forwarded(self):
        """run_kwargs must be forwarded to device.run."""
        device, _, _ = _mock_device()
        executor = QBraidExecutor(
            device=device, run_kwargs={"name": "my_job", "tags": ["test"]}
        )

        qc = QuantumCircuit(1)
        qc.measure_all()
        executor._submit_and_wait(qc, shots=50)

        _, kwargs = device.run.call_args
        assert kwargs["name"] == "my_job"
        assert kwargs["tags"] == ["test"]
        assert kwargs["shots"] == 50

    def test_post_init_shots_mutation_rejected(self):
        """Mutating run_kwargs after construction to add 'shots' is caught."""
        device, _, _ = _mock_device()
        executor = QBraidExecutor(device=device)
        executor.run_kwargs["shots"] = 7

        qc = QuantumCircuit(1)
        qc.measure_all()
        with pytest.raises(ValueError, match="reserved key"):
            executor._submit_and_wait(qc, shots=5)


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestExecute:
    def test_adds_measurements_when_no_clbits(self):
        """execute() should add measure_all when circuit has no clbits."""
        device, _, _ = _mock_device({"00": 50, "11": 50})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert qc.num_clbits == 0

        counts = executor.execute(qc, shots=100)

        # Verify that the submitted circuit has measurements
        submitted_circuit = device.run.call_args[0][0]
        assert submitted_circuit.num_clbits > 0
        assert counts == {"00": 50, "11": 50}

    def test_preserves_existing_measurements(self):
        """execute() should not add measurements if circuit already has them."""
        device, _, _ = _mock_device({"0": 100})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        counts = executor.execute(qc, shots=100)

        submitted_circuit = device.run.call_args[0][0]
        assert submitted_circuit.num_clbits == 1
        assert counts == {"0": 100}

    def test_does_not_modify_original_circuit(self):
        """execute() must not mutate the input circuit."""
        device, _, _ = _mock_device()
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(2)
        qc.h(0)
        original_clbits = qc.num_clbits

        executor.execute(qc, shots=100)
        assert qc.num_clbits == original_clbits


# ---------------------------------------------------------------------------
# bind_parameters()
# ---------------------------------------------------------------------------


class TestBindParameters:
    def test_bind_parameters(self):
        from qamomile.circuit.transpiler.parameter_binding import (
            ParameterInfo,
            ParameterMetadata,
        )
        from qiskit.circuit import Parameter

        device, _, _ = _mock_device()
        executor = QBraidExecutor(device=device)

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        param_info = ParameterInfo(
            name="theta",
            array_name="theta",
            index=None,
            backend_param=theta,
        )
        metadata = ParameterMetadata(parameters=[param_info])

        bound = executor.bind_parameters(qc, {"theta": 1.5}, metadata)
        assert len(bound.parameters) == 0  # All parameters bound
