"""Tests for QBraidExecutor.estimate() counts-based expectation value."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, random_pauli_list

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.observable import Hamiltonian, Pauli, PauliOperator, X, Y, Z
from qamomile.qbraid.executor import QBraidExecutor
from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

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
        assert device.run.call_count == 1

    def test_estimate_uses_qiskit_big_endian_qubit_positions(self):
        """estimate() reads count keys with rightmost bit = qubit 0."""
        device = _mock_device_multi([{"001": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(3)
        qc.x(0)  # Prepare |001> in Qiskit big-endian count notation.

        hamiltonian = Z(0) + 2.0 * Z(1) + 4.0 * Z(2)
        result = executor.estimate(qc, hamiltonian)

        # Big-endian "001" means q2=0, q1=0, q0=1.
        # Therefore Z0=-1, Z1=+1, Z2=+1, giving -1 + 2 + 4 = 5.
        assert math.isclose(result, 5.0, abs_tol=1e-10)

    def test_multi_term_hamiltonian(self):
        """Test H = 0.5*Z0 + 0.3*Z1 on |00>."""
        # <00|Z0|00> = 1, <00|Z1|00> = 1
        # Both terms are measurable from the same all-Z basis circuit.
        device = _mock_device_multi([{"00": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)

        hamiltonian = 0.5 * Z(0) + 0.3 * Z(1)
        result = executor.estimate(qc, hamiltonian)
        assert math.isclose(result, 0.8, abs_tol=1e-10)
        assert device.run.call_count == 1


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


# ---------------------------------------------------------------------------
# Parameter binding
# ---------------------------------------------------------------------------


class TestParameterBinding:
    def test_params_are_bound_before_submission(self):
        """Direct estimate() with params must bind before device.run()."""
        from qiskit.circuit import Parameter

        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        executor.estimate(qc, Z(0), params=[0.5])

        # The circuit submitted to device.run() must have no unbound params.
        submitted = device.run.call_args.args[0]
        assert len(submitted.parameters) == 0

    def test_unbound_params_without_values_rejected(self):
        """Parametric circuit + params=None must raise before submission."""
        from qiskit.circuit import Parameter

        device = MagicMock()
        executor = QBraidExecutor(device=device, expval_shots=100)

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        with pytest.raises(ExecutionError, match="unbound parameter"):
            executor.estimate(qc, Z(0))

        device.run.assert_not_called()

    def test_length_mismatch_raises(self):
        """Wrong number of params must raise ValueError (from Qiskit)."""
        from qiskit.circuit import Parameter

        device = MagicMock()
        executor = QBraidExecutor(device=device, expval_shots=100)

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        with pytest.raises(ValueError):
            executor.estimate(qc, Z(0), params=[0.1, 0.2])

    def test_pre_bound_circuit_no_params_works(self):
        """Already-bound circuit with params=None should work normally."""
        from qiskit.circuit import Parameter

        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc = qc.assign_parameters([0.5])

        result = executor.estimate(qc, Z(0))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Fake statevector device for oracle comparison
# ---------------------------------------------------------------------------


class _FakeResultData:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return self._counts


class _FakeResult:
    def __init__(self, counts):
        self.data = _FakeResultData(counts)


class _FakeJob:
    def __init__(self, counts):
        self._counts = counts

    def wait_for_final_state(self, timeout=None, poll_interval=None):
        return None

    def result(self):
        return _FakeResult(self._counts)


class _FakeStatevectorDevice:
    """Fake device that returns deterministic probability-proportional counts.

    Instead of random sampling, each outcome gets ``round(prob * shots)``
    counts. This eliminates shot noise entirely; the only error is from
    integer rounding (at most 0.5 per outcome).

    Compatible with QBraidExecutor's job contract
    (wait_for_final_state / result().data.get_counts()).
    """

    def run(self, circuit, shots, **kwargs):
        qc = circuit.remove_final_measurements(inplace=False)
        probs = Statevector.from_instruction(qc).probabilities_dict()
        counts: dict[str, int] = {}
        for outcome, prob in probs.items():
            c = round(prob * shots)
            if c > 0:
                counts[outcome] = c
        return _FakeJob(counts)


# ---------------------------------------------------------------------------
# Layer 1: Deterministic arithmetic matrix regression
# ---------------------------------------------------------------------------


class TestArithmeticMatrix:
    """Supported Hamiltonian arithmetic combinations with hand-crafted counts."""

    def test_scalar_multiplication_sign_and_scale(self):
        """``-0.5 * X(0)`` on |+> state: X expectation is +1, scaled to -0.5."""
        # After X-rotation (H gate), |+> -> |0>; all counts are "0".
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)
        qc.h(0)

        result = executor.estimate(qc, -0.5 * X(0))
        assert result == pytest.approx(-0.5, abs=1e-9)

    def test_addition_mixed_basis(self):
        """``0.25 * X(0) + 0.75 * Z(1)`` on |+0>: single group (X on q0, Z on q1)."""
        # X(0) and Z(1) act on different qubits so they merge into one
        # basis group {0: X, 1: Z}. After H rotation on q0, |+0> -> |00>.
        device = _mock_device_multi([{"00": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)
        qc.h(0)

        result = executor.estimate(qc, 0.25 * X(0) + 0.75 * Z(1))
        assert result == pytest.approx(0.25 + 0.75, abs=1e-9)

    def test_hamiltonian_mul_disjoint(self):
        """``X(0) * Z(1)`` on |+0>: disjoint product, expectation +1."""
        # XZ shares one basis group; after X-rotation on q0:
        # |+0> -> measure q0 in X basis (H), q1 in Z basis
        # q0: |+> -> H -> |0>, q1: |0>; bitstring "00" has even parity -> +1
        device = _mock_device_multi([{"00": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)
        qc.h(0)

        result = executor.estimate(qc, X(0) * Z(1))
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_hamiltonian_mul_identity_collapse(self):
        """``X(0) * X(0)`` collapses to identity with coefficient 1."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)

        result = executor.estimate(qc, X(0) * X(0))
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_constant_cross_term(self):
        """``(Z(0) + 1.5) * (Z(1) - 0.5)`` on |00>.

        Expands to: Z0*Z1 - 0.5*Z0 + 1.5*Z1 - 0.75.
        On |00>: Z0=+1, Z1=+1, Z0Z1=+1 -> 1 - 0.5 + 1.5 - 0.75 = 1.25.
        """
        # Z0Z1 and Z0 and Z1 all share Z basis -> single group
        device = _mock_device_multi([{"00": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)

        H = (Z(0) + 1.5) * (Z(1) - 0.5)
        result = executor.estimate(qc, H)
        assert result == pytest.approx(1.25, abs=1e-9)

    def test_unary_negation_multi_term(self):
        """``-(Z(0) + 0.25 * X(1))`` on |0+>.

        Z(0) on |0> = +1, X(1) on |+> = +1.
        Negated: -(1 + 0.25) = -1.25.
        """
        # Two basis groups: Z for qubit 0, X for qubit 1
        device = _mock_device_multi(
            [
                {"00": 100},  # Z(0): qubit 0 in |0> -> +1
                {"00": 50, "01": 50},  # X(1): qubit 1 in |+>, after H -> +1
            ]
        )
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(2)
        qc.h(1)

        H = -(Z(0) + 0.25 * X(1))
        result = executor.estimate(qc, H)
        assert result == pytest.approx(-1.25, abs=1e-9)

    def test_same_qubit_mixed_pauli_product_rejected(self):
        """``X(0) * Y(0)`` produces anti-Hermitian 1j*Z(0); estimate() rejects."""
        device = _mock_device_multi([{"0": 100}])
        executor = QBraidExecutor(device=device, expval_shots=100)

        qc = QuantumCircuit(1)

        with pytest.raises(ExecutionError, match="imaginary"):
            executor.estimate(qc, X(0) * Y(0))


# ---------------------------------------------------------------------------
# Layer 2: Seeded randomized oracle comparison
# ---------------------------------------------------------------------------


def _from_pauli_label(label: str, coeff: float) -> Hamiltonian:
    """Convert a Qiskit Pauli label (big-endian) to a qamomile Hamiltonian.

    ``phase=False`` in ``random_pauli_list`` is required so that only
    real-coefficient Hermitian operators are generated.
    """
    n = len(label)
    h = Hamiltonian(num_qubits=n)
    ops: list[PauliOperator] = []
    for q, ch in enumerate(reversed(label)):
        if ch == "I":
            continue
        ops.append(PauliOperator(getattr(Pauli, ch), q))
    if ops:
        h.add_term(tuple(ops), coeff)
    else:
        h.constant = coeff
    return h


def _bit_for_qubit(bitstring: str, qubit: int, *, reverse_endian: bool = False) -> int:
    """Return the measured bit for ``qubit`` from a big-endian bitstring."""
    if reverse_endian:
        return int(bitstring[qubit])
    return int(bitstring[len(bitstring) - 1 - qubit])


def _prepare_product_eigenstate(
    num_qubits: int, bitstring: str, basis_assignment: dict[int, Pauli]
) -> QuantumCircuit:
    """Prepare a product state with prescribed X/Y/Z eigenvalues per qubit."""
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        # bit=1 means we want the -1 eigenstate for the chosen single-qubit Pauli.
        bit = _bit_for_qubit(bitstring, qubit)
        if bit:
            qc.x(qubit)

        pauli = basis_assignment[qubit]
        # Start from Z eigenstates |0> / |1> and rotate them into X/Y eigenstates
        # when needed.
        if pauli == Pauli.X:
            qc.h(qubit)
        elif pauli == Pauli.Y:
            qc.h(qubit)
            qc.s(qubit)

    return qc


def _random_endian_sensitive_terms(
    seed: int, num_qubits: int, *, basis_assignment: dict[int, Pauli]
) -> tuple[str, Hamiltonian]:
    """Generate a seeded Hamiltonian whose value changes under reversed endian."""

    def pauli_label_eigenvalue(
        label: str, bitstring: str, *, reverse_endian: bool = False
    ) -> float:
        eigenvalue = 1.0
        for qubit, ch in enumerate(reversed(label)):
            if ch != "I" and _bit_for_qubit(
                bitstring, qubit, reverse_endian=reverse_endian
            ):
                eigenvalue *= -1.0
        return eigenvalue

    rng = np.random.default_rng(seed)
    labels: list[str] = []
    for support_mask in range(1, 2**num_qubits):
        chars = ["I"] * num_qubits
        for qubit in range(num_qubits):
            if support_mask & (1 << qubit):
                # Qiskit labels are big-endian: qubit 0 is the rightmost char.
                chars[num_qubits - 1 - qubit] = basis_assignment[qubit].name
        labels.append("".join(chars))

    for _ in range(128):
        bitstring = "".join(
            str(int(bit)) for bit in rng.integers(0, 2, size=num_qubits)
        )
        # Palindromic bitstrings are weak regression cases because reversing the
        # bit order leaves them unchanged.
        if bitstring == bitstring[::-1]:
            continue

        # Re-randomize only the coefficients; the label set itself is the full
        # non-identity basis for this per-qubit Pauli assignment.
        coeffs = rng.uniform(-1.0, 1.0, size=len(labels))
        coeffs = np.where(
            np.abs(coeffs) < 0.2,
            np.where(coeffs < 0.0, -0.2, 0.2),
            coeffs,
        )
        correct = sum(
            float(coeff) * pauli_label_eigenvalue(label, bitstring)
            for label, coeff in zip(labels, coeffs)
        )
        wrong = sum(
            float(coeff) * pauli_label_eigenvalue(label, bitstring, reverse_endian=True)
            for label, coeff in zip(labels, coeffs)
        )

        # Keep only cases that would clearly fail if QBraidExecutor read the
        # measured bitstring in the wrong direction.
        if abs(correct - wrong) <= 0.5:
            continue

        hamiltonian = Hamiltonian(num_qubits=num_qubits)
        for label, coeff in zip(labels, coeffs):
            hamiltonian = hamiltonian + _from_pauli_label(label, float(coeff))
        return bitstring, hamiltonian

    raise AssertionError(
        f"Failed to generate an endian-sensitive random case for seed={seed}, "
        f"num_qubits={num_qubits}."
    )


def _deterministic_rounding_error_bound(
    circuit: QuantumCircuit, hamiltonian: Hamiltonian, shots: int
) -> float:
    """Upper-bound fake-device rounding error for ``QBraidExecutor.estimate()``.

    For a measurement-basis group with ``M`` non-zero outcomes,
    ``_FakeStatevectorDevice`` rounds each ideal count independently:

        c_x = round(shots * p_x) = shots * p_x + e_x,   |e_x| <= 0.5

    Let ``delta = sum_x e_x`` be the total-shot drift and ``Delta`` the
    parity-sum drift for one Pauli term. Then ``|delta| <= M / 2`` and
    ``|Delta| <= M / 2``. Since ``QBraidExecutor.estimate()`` normalizes by
    ``sum(counts.values()) = shots + delta``, one term's expectation-value
    error is bounded by:

        |mu_hat - mu| <= M / (shots - M / 2)

    The Hamiltonian error is the sum of those per-term bounds weighted by
    the absolute values of the coefficients in each basis group.

    Rule of thumb for the test's current ``shots=2_000_000``:

    - ``M = 2`` non-zero outcomes gives about ``1.0e-6`` per unit coefficient.
    - ``M = 4`` non-zero outcomes gives about ``2.0e-6`` per unit coefficient.
    - ``M = 8`` non-zero outcomes gives about ``4.0e-6`` per unit coefficient.

    So, for example:

    - one 2-qubit term with full support (``M = 4``) and ``|c| = 1`` gives a
      bound of about ``2.0e-6``;
    - one 3-qubit term with full support (``M = 8``) and ``|c| = 1`` gives a
      bound of about ``4.0e-6``;
    - three such 3-qubit unit-weight terms would give a worst-case bound of
      about ``1.2e-5``.
    """
    basis_groups: list[
        tuple[dict[int, Pauli], list[tuple[tuple[PauliOperator, ...], complex]]]
    ] = []

    for operators, coeff in hamiltonian:
        basis_assignment = {
            op.index: op.pauli for op in operators if op.pauli != Pauli.I
        }

        for group_basis, group_terms in basis_groups:
            if all(
                group_basis.get(qubit_idx, pauli_type) == pauli_type
                for qubit_idx, pauli_type in basis_assignment.items()
            ):
                group_basis.update(basis_assignment)
                group_terms.append((operators, coeff))
                break
        else:
            basis_groups.append((dict(basis_assignment), [(operators, coeff)]))

    total_bound = 0.0
    for basis_assignment, terms in basis_groups:
        rotated = circuit.copy()
        for qubit_idx, pauli_type in sorted(basis_assignment.items()):
            if pauli_type == Pauli.X:
                rotated.h(qubit_idx)
            elif pauli_type == Pauli.Y:
                rotated.sdg(qubit_idx)
                rotated.h(qubit_idx)

        support_size = len(Statevector.from_instruction(rotated).probabilities_dict())
        per_term_bound = support_size / (shots - support_size / 2)
        total_bound += per_term_bound * sum(abs(complex(coeff)) for _, coeff in terms)

    return total_bound


class TestRandomOracleComparison:
    """Seeded random circuit + random Hermitian Hamiltonian vs statevector oracle."""

    @pytest.mark.parametrize("seed", [offset + 901 for offset in range(30)])
    @pytest.mark.parametrize("num_qubits", [2, 3])
    def test_random_circuit_random_hamiltonian_matches_statevector(
        self, seed, num_qubits
    ):
        depth = 3
        num_terms = 3
        shots = 2_000_000

        # Random state-preparation circuit under test.
        qc = random_circuit(
            num_qubits, depth=depth, max_operands=2, measure=False, seed=seed
        )

        # Random Hermitian Hamiltonian built from Qiskit's big-endian Pauli labels.
        plist = random_pauli_list(num_qubits, size=num_terms, seed=seed, phase=False)
        coeffs = np.random.default_rng(seed).uniform(-1.0, 1.0, size=num_terms)

        H = Hamiltonian(num_qubits=num_qubits)
        for p, c in zip(plist, coeffs):
            H = H + _from_pauli_label(p.to_label(), float(c))

        # Oracle value from Qiskit's exact statevector expectation.
        exact = float(
            Statevector.from_instruction(qc)
            .expectation_value(hamiltonian_to_sparse_pauli_op(H))
            .real
        )

        # QBraidExecutor still reconstructs the expectation from counts, but this
        # fake device removes sampling noise so only deterministic rounding remains.
        device = _FakeStatevectorDevice()
        approx = QBraidExecutor(device=device, expval_shots=shots).estimate(qc, H)

        # Tolerance justification:
        # _FakeStatevectorDevice rounds each outcome count independently, so
        # there is no shot noise, only deterministic integer-rounding error.
        # Because QBraidExecutor normalizes by sum(counts.values()), the error
        # bound must account for both parity-sum rounding and total-shot drift.
        # `_deterministic_rounding_error_bound()` computes a rigorous upper
        # bound for this exact circuit/Hamiltonian pair.
        tolerance = _deterministic_rounding_error_bound(qc, H, shots) + 1e-12
        assert abs(exact - approx) < tolerance, (
            f"seed={seed} num_qubits={num_qubits} depth={depth} shots={shots}: "
            f"exact={exact}, approx={approx}, diff={abs(exact - approx)}"
        )


class TestRandomEndianRegression:
    """Seeded random tests that would fail under reversed bit ordering."""

    @pytest.mark.parametrize("seed", [offset + 901 for offset in range(30)])
    @pytest.mark.parametrize("num_qubits", [2, 3, 5])
    def test_random_raw_counts_big_endian_for_z_terms(self, seed, num_qubits):
        # Use only Z terms so the expected value is determined directly by the
        # reported bitstring, making endian mistakes especially easy to expose.
        basis_assignment = {qubit: Pauli.Z for qubit in range(num_qubits)}
        bitstring, hamiltonian = _random_endian_sensitive_terms(
            seed, num_qubits, basis_assignment=basis_assignment
        )

        # Prepare the computational basis state corresponding to that big-endian
        # bitstring.
        qc = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            if _bit_for_qubit(bitstring, qubit):
                qc.x(qubit)

        # Qiskit statevector acts as the exact oracle for the same Hamiltonian.
        exact = float(
            Statevector.from_instruction(qc)
            .expectation_value(hamiltonian_to_sparse_pauli_op(hamiltonian))
            .real
        )

        # Feed the exact big-endian bitstring back as counts. If QBraidExecutor
        # reads bit positions in the wrong order, this test should fail.
        device = _mock_device_multi([{bitstring: 100}])
        approx = QBraidExecutor(device=device, expval_shots=100).estimate(
            qc, hamiltonian
        )

        assert approx == pytest.approx(exact, abs=1e-10)
        assert device.run.call_count == 1

    @pytest.mark.parametrize("seed", [offset + 901 for offset in range(30)])
    @pytest.mark.parametrize("num_qubits", [2, 3, 5])
    def test_random_product_eigenstates_mixed_bases_are_endian_sensitive(
        self, seed, num_qubits
    ):
        rng = np.random.default_rng(seed)
        paulis = (Pauli.X, Pauli.Y, Pauli.Z)
        # Pick one measurement basis per qubit, then generate a random Hamiltonian
        # whose terms all respect that local basis assignment.
        basis_assignment = {
            qubit: paulis[int(rng.integers(0, len(paulis)))]
            for qubit in range(num_qubits)
        }
        bitstring, hamiltonian = _random_endian_sensitive_terms(
            seed + 10_000, num_qubits, basis_assignment=basis_assignment
        )

        # Prepare a product eigenstate for that mixed X/Y/Z basis pattern.
        qc = _prepare_product_eigenstate(num_qubits, bitstring, basis_assignment)
        exact = float(
            Statevector.from_instruction(qc)
            .expectation_value(hamiltonian_to_sparse_pauli_op(hamiltonian))
            .real
        )

        # This variant checks the full basis-rotation path used by estimate(),
        # not just the direct raw-count interpretation.
        device = _FakeStatevectorDevice()
        approx = QBraidExecutor(device=device, expval_shots=100).estimate(
            qc, hamiltonian
        )

        assert approx == pytest.approx(exact, abs=1e-10)
