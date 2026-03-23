"""Tests for QBraidExecutor.estimate() counts-based expectation value."""

import math
from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.observable import Hamiltonian, Pauli, PauliOperator, X, Y, Z
from qamomile.qbraid.executor import QBraidExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_device_multi(counts_sequence: list[dict[str, int]]):
    """Create a mock device that returns different counts on successive calls.

    Each call to device.run() returns a job whose result has the next counts
    from counts_sequence.
    """
    device = MagicMock()
    jobs = []
    for counts in counts_sequence:
        job = MagicMock()
        result = MagicMock()
        result.data.get_counts.return_value = counts
        job.wait_for_final_state.return_value = None
        job.result.return_value = result
        jobs.append(job)

    device.run.side_effect = jobs
    return device


# ---------------------------------------------------------------------------
# Pre-existing clbit rejection
# ---------------------------------------------------------------------------


class TestExistingClbitRejection:
    def test_rejects_circuit_with_clbits(self):
        device = MagicMock()
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(2, 1)
        qc.h(0)
        qc.measure(0, 0)

        hamiltonian = Z(0)

        with pytest.raises(ExecutionError, match="existing classical bits"):
            executor.estimate(qc, hamiltonian)

    def test_rejects_circuit_with_empty_creg(self):
        device = MagicMock()
        executor = QBraidExecutor(device=device)

        qc = QuantumCircuit(1, 1)  # has 1 clbit even without explicit measure

        hamiltonian = Z(0)

        with pytest.raises(ExecutionError, match="num_clbits > 0"):
            executor.estimate(qc, hamiltonian)


# ---------------------------------------------------------------------------
# Single Pauli terms
# ---------------------------------------------------------------------------


class TestSinglePauliEstimate:
    def test_z_expectation_all_zero(self):
        """<0|Z|0> = +1."""
        # All measurements give "0" -> parity 0 for qubit 0 -> eigenvalue +1
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)  # |0> state

        result = executor.estimate(qc, Z(0))
        assert math.isclose(result, 1.0, abs_tol=1e-10)

    def test_z_expectation_all_one(self):
        """<1|Z|1> = -1."""
        device = _mock_device_multi([{"1": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)
        qc.x(0)  # |1> state

        result = executor.estimate(qc, Z(0))
        assert math.isclose(result, -1.0, abs_tol=1e-10)

    def test_x_expectation_plus_state(self):
        """<+|X|+> = +1. H|0> = |+>, X basis measurement = all 0."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)
        qc.h(0)  # |+> state

        result = executor.estimate(qc, X(0))
        assert math.isclose(result, 1.0, abs_tol=1e-10)

    def test_y_expectation(self):
        """Test Y measurement with known state."""
        # S|+> = (|0> + i|1>)/sqrt(2), <Y> = +1
        # After Sdg + H rotation: all "0"
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)  # S|+> state

        result = executor.estimate(qc, Y(0))
        assert math.isclose(result, 1.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Multi-qubit terms
# ---------------------------------------------------------------------------


class TestMultiQubitEstimate:
    def test_zz_bell_state(self):
        """<Bell|ZZ|Bell> = +1 for |Phi+> = (|00> + |11>)/sqrt(2)."""
        # ZZ eigenvalues: |00> -> +1, |11> -> +1 (even parity on both)
        device = _mock_device_multi([{"00": 50, "11": 50}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        hamiltonian = Z(0) * Z(1)
        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, 1.0, abs_tol=1e-10)

    def test_z0_minus_z1_asymmetric(self):
        """H = Z(0) - Z(1) on |10⟩ (qubit 0 = |1>, qubit 1 = |0>).

        <10|Z(0)|10> = -1,  <10|Z(1)|10> = +1
        Expected: -1 - 1 = -2

        Big-endian "01" means qubit1=0, qubit0=1 → this is |10⟩.
        If endianness is wrong, result would be +2 instead of -2.
        """
        # Z0 and Z1 share the same basis (both Z), one circuit submitted
        device = _mock_device_multi([{"01": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)
        qc.x(0)  # qubit 0 → |1⟩

        hamiltonian = Z(0) - Z(1)
        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, -2.0, abs_tol=1e-10)

    def test_multi_term_hamiltonian(self):
        """Test H = 0.5*Z0 + 0.3*Z1 on |00>."""
        # <00|Z0|00> = 1, <00|Z1|00> = 1
        # Each term has a different basis key, so two circuits are submitted.
        device = _mock_device_multi([{"00": 100}, {"00": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)

        hamiltonian = 0.5 * Z(0) + 0.3 * Z(1)
        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, 0.8, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Constant term
# ---------------------------------------------------------------------------


class TestConstantTerm:
    def test_constant_only(self):
        """Hamiltonian with only a constant term."""
        device = MagicMock()
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)

        hamiltonian = Hamiltonian()
        hamiltonian.constant = 3.14

        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, 3.14, abs_tol=1e-10)
        # No circuits should have been submitted
        device.run.assert_not_called()

    def test_constant_plus_pauli(self):
        """H = 2.0 + Z0 on |0>."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)

        hamiltonian = Z(0) + 2.0
        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, 3.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Remapped qubits
# ---------------------------------------------------------------------------


class TestRemappedQubits:
    def test_remapped_z(self):
        """Z on remapped qubit index."""
        # H = Z(2) but circuit only has 3 qubits (index 0,1,2)
        # All measurement "000" -> qubit 2 is 0 -> parity 0 -> +1
        device = _mock_device_multi([{"000": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(3)

        hamiltonian = Z(2)
        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, 1.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Out-of-range Hamiltonian qubit index
# ---------------------------------------------------------------------------


class TestOutOfRangeHamiltonian:
    def test_single_qubit_out_of_range(self):
        """Z(1) on 1-qubit circuit must raise ExecutionError."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)
        with pytest.raises(ExecutionError, match="qubit index 1"):
            executor.estimate(qc, Z(1))

    def test_multi_qubit_term_out_of_range(self):
        """Z(0)*Z(2) on 2-qubit circuit must raise ExecutionError."""
        device = _mock_device_multi([{"00": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)
        hamiltonian = Z(0) * Z(2)
        with pytest.raises(ExecutionError, match="qubit index 2"):
            executor.estimate(qc, hamiltonian)

    def test_in_range_still_works(self):
        """Z(0) on 1-qubit circuit should work normally."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)
        result = executor.estimate(qc, Z(0))
        assert math.isclose(result, 1.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Imaginary part handling
# ---------------------------------------------------------------------------


class TestImaginaryPart:
    def test_negligible_imaginary_accepted(self):
        """Result with negligible imaginary part should be accepted."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)

        # Real coefficient -> real result
        hamiltonian = Z(0)
        result = executor.estimate(qc, hamiltonian)
        assert isinstance(result, float)

    def test_non_negligible_imaginary_rejected(self):
        """Result with large imaginary part should raise ExecutionError."""
        # All measurements "0" -> Z expectation = +1, coefficient 1j -> result = 1j
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)

        # Create Hamiltonian with purely imaginary coefficient
        hamiltonian = Hamiltonian()
        hamiltonian.add_term((PauliOperator(Pauli.Z, 0),), 1j)

        with pytest.raises(ExecutionError, match="imaginary part"):
            executor.estimate(qc, hamiltonian)


# ---------------------------------------------------------------------------
# Shared wait helper reuse
# ---------------------------------------------------------------------------


class TestWaitHelperReuse:
    def test_estimate_uses_submit_and_wait(self):
        """estimate() must use _submit_and_wait, not a separate wait path."""
        # Build job mock directly for inspection
        device = MagicMock()
        job = MagicMock()
        result = MagicMock()
        result.data.get_counts.return_value = {"0": 100}
        job.wait_for_final_state.return_value = None
        job.result.return_value = result
        device.run.return_value = job

        executor = QBraidExecutor(device=device, expval_shots=100, poll_interval=3)

        qc = QuantumCircuit(1)
        executor.estimate(qc, Z(0))

        job.wait_for_final_state.assert_called_once_with(timeout=None, poll_interval=3)
