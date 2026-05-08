"""Tests for QBraidExecutor constructor, execute, and shared wait helper."""

from unittest.mock import MagicMock, patch

import pytest
from qiskit import QuantumCircuit

from qamomile.circuit.transpiler.errors import ExecutionError
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

        executor = QBraidExecutor(device_id="dev123", api_key="mykey")
        mock_resolve.assert_called_once_with("dev123", None, "mykey")
        assert executor.device is mock_dev

    def test_default_params(self):
        device, _, _ = _mock_device()
        executor = QBraidExecutor(device=device)
        assert executor.device is device
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
        assert executor.device is device
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
            QBraidExecutor(device=device, run_kwargs={"name": "test", "shots": 50})


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
# Counts normalization
# ---------------------------------------------------------------------------


class TestCountsNormalization:
    def test_execute_preserves_qiskit_big_endian_clbit_order(self):
        """execute() keeps canonical Qiskit big-endian classical-bit order."""
        device, _, _ = _mock_device({"001": 5})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(3, 3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        counts = executor.execute(qc, shots=5)

        # QBraidExecutor normalizes width/spacing only. It does not reverse the
        # bitstring, so "001" still means c2=0, c1=0, c0=1.
        assert counts == {"001": 5}

    def test_short_key_padded_2bit(self):
        """Under-width key '1' is zero-padded to '01' for 2-clbit circuit."""
        device, _, _ = _mock_device({"1": 5})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        counts = executor.execute(qc, shots=5)
        assert counts == {"01": 5}

    def test_short_key_padded_3bit(self):
        """Under-width key '10' is zero-padded to '010' for 3-clbit circuit."""
        device, _, _ = _mock_device({"10": 5})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(3, 3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        counts = executor.execute(qc, shots=5)
        assert counts == {"010": 5}

    def test_space_separated_key_flattened(self):
        """Space-separated key '01 0' is flattened to '010'."""
        device, _, _ = _mock_device({"01 0": 5})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(3, 3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        counts = executor.execute(qc, shots=5)
        assert counts == {"010": 5}

    def test_already_padded_key_unchanged(self):
        """Full-width key '010' passes through unchanged."""
        device, _, _ = _mock_device({"010": 3, "111": 7})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(3, 3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        counts = executor.execute(qc, shots=10)
        assert counts == {"010": 3, "111": 7}

    def test_non_binary_key_raises(self):
        """Non-binary key after space removal raises ExecutionError."""
        device, _, _ = _mock_device({"0x1": 5})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        with pytest.raises(ExecutionError, match="non-binary"):
            executor.execute(qc, shots=5)

    def test_width_1_key_unchanged(self):
        """Width-1 keys '0' and '1' pass through unchanged."""
        device, _, _ = _mock_device({"0": 60, "1": 40})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        counts = executor.execute(qc, shots=100)
        assert counts == {"0": 60, "1": 40}

    def test_short_key_via_measure_all(self):
        """Short key is padded when circuit gets measure_all via execute()."""
        device, _, _ = _mock_device({"1": 5})
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(2)
        counts = executor.execute(qc, shots=5)
        assert counts == {"01": 5}

    def test_integration_sample_with_short_key(self):
        """QiskitTranspiler + QBraidExecutor: short key decodes without None."""
        import qamomile.circuit as qm
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def three_bit_kernel() -> tuple[qm.Bit, qm.Bit, qm.Bit]:
            qs = qm.qubit_array(3, "qs")
            return qm.measure(qs[0]), qm.measure(qs[1]), qm.measure(qs[2])

        exe = QiskitTranspiler().transpile(three_bit_kernel)

        device, _, _ = _mock_device({"10": 1})
        executor = QBraidExecutor(device=device)

        sample_results = exe.sample(executor, shots=1).result().results
        assert len(sample_results) == 1
        values, count = sample_results[0]
        assert None not in values, f"None found in decoded values: {values}"
        assert count == 1

    def test_integration_run_with_short_key(self):
        """QiskitTranspiler + QBraidExecutor: short key decodes without None."""
        import qamomile.circuit as qm
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def three_bit_kernel() -> tuple[qm.Bit, qm.Bit, qm.Bit]:
            qs = qm.qubit_array(3, "qs")
            return qm.measure(qs[0]), qm.measure(qs[1]), qm.measure(qs[2])

        exe = QiskitTranspiler().transpile(three_bit_kernel)

        device, _, _ = _mock_device({"10": 1})
        executor = QBraidExecutor(device=device)

        run_result = exe.run(executor).result()
        assert isinstance(run_result, tuple)
        assert None not in run_result, f"None found in run result: {run_result}"


# ---------------------------------------------------------------------------
# Seeded random width-normalization regression
# ---------------------------------------------------------------------------


class TestRandomShortKeyPadding:
    """Seeded random tests verifying width normalization across widths."""

    @pytest.mark.parametrize("seed,width", [(0, 2), (7, 3), (42, 5)])
    def test_random_short_keys_padded_to_target_width(self, seed, width):
        import random

        rng = random.Random(seed)
        max_outcomes = min(8, 2**width)
        num_outcomes = rng.randint(2, max_outcomes)
        outcomes = rng.sample(range(2**width), num_outcomes)

        for outcome in outcomes:
            canonical = format(outcome, f"0{width}b")
            # Simulate qBraid stripping leading zeros
            short = canonical.lstrip("0") or "0"

            device, _, _ = _mock_device({short: 1})
            executor = QBraidExecutor(device=device)

            qc = QuantumCircuit(width, width)
            for i in range(width):
                qc.measure(i, i)
            counts = executor.execute(qc, shots=1)

            for key in counts:
                assert len(key) == width, (
                    f"seed={seed}, width={width}, outcome={outcome}: "
                    f"expected key length {width}, got {len(key)} for key {key!r}"
                )
            assert counts == {canonical: 1}, (
                f"seed={seed}, width={width}, outcome={outcome}: "
                f"expected {{{canonical!r}: 1}}, got {counts}"
            )

    @pytest.mark.parametrize("seed,width", [(0, 2), (7, 3), (42, 5)])
    def test_already_padded_keys_unchanged(self, seed, width):
        import random

        rng = random.Random(seed)
        max_outcomes = min(8, 2**width)
        num_outcomes = rng.randint(2, max_outcomes)
        outcomes = rng.sample(range(2**width), num_outcomes)

        for outcome in outcomes:
            canonical = format(outcome, f"0{width}b")

            device, _, _ = _mock_device({canonical: 1})
            executor = QBraidExecutor(device=device)

            qc = QuantumCircuit(width, width)
            for i in range(width):
                qc.measure(i, i)
            counts = executor.execute(qc, shots=1)

            assert counts == {canonical: 1}, (
                f"seed={seed}, width={width}: "
                f"padded key {canonical!r} should pass through unchanged"
            )


# ---------------------------------------------------------------------------
# bind_parameters()
# ---------------------------------------------------------------------------


class TestBindParameters:
    def test_bind_parameters(self):
        from qiskit.circuit import Parameter

        from qamomile.circuit.transpiler.parameter_binding import (
            ParameterInfo,
            ParameterMetadata,
        )

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
