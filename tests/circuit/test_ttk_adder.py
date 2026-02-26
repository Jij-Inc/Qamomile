"""Tests for TTK inplace adder (arXiv:0910.2530v1, Section 2.2)."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.operation.composite_gate import CompositeGateType
from qamomile.circuit.stdlib.ttk_adder import TTKInplaceAdder

# ---------------------------------------------------------------------------
# Unit tests for class attributes and resources
# ---------------------------------------------------------------------------


class TestTTKInplaceAdderAttributes:
    def test_custom_name(self) -> None:
        adder = TTKInplaceAdder(3)
        assert adder.custom_name == "ttk_adder"

    def test_gate_type(self) -> None:
        adder = TTKInplaceAdder(3)
        assert adder.gate_type == CompositeGateType.CUSTOM

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_num_target_qubits(self, n: int) -> None:
        adder = TTKInplaceAdder(n)
        assert adder.num_target_qubits == 2 * n + 1

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            TTKInplaceAdder(0)
        with pytest.raises(ValueError, match="n must be >= 1"):
            TTKInplaceAdder(-1)


class TestTTKInplaceAdderResources:
    def test_resources_n1(self) -> None:
        r = TTKInplaceAdder(1)._resources()
        m = r.custom_metadata
        assert m["num_cnot_gates"] == 1
        assert m["num_toffoli_gates"] == 1
        assert m["total_gates"] == 2

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_resources(self, n: int) -> None:
        r = TTKInplaceAdder(n)._resources()
        m = r.custom_metadata
        assert m["num_cnot_gates"] == 5 * n - 5
        assert m["num_toffoli_gates"] == 2 * n - 1
        assert m["total_gates"] == 7 * n - 6
        assert m["depth"] == 5 * n - 3


# ---------------------------------------------------------------------------
# Statevector correctness tests via Qamomile pipeline + Qiskit simulation
# ---------------------------------------------------------------------------


def _make_adder_kernel(n: int):
    """Create an adder qkernel for n-bit operands using qubit_array + ttk_adder."""

    @qm.qkernel
    def _kernel() -> qm.Bit:
        a = qm.qubit_array(n, "a")
        b = qm.qubit_array(n, "b")
        z = qm.qubit("z")
        b, a, z = qm.ttk_adder(b, a, z)
        return qm.measure(z)

    return _kernel


def _build_adder_circuit(n: int, a_val: int, b_val: int):
    """Transpile adder kernel via Qamomile and prepend X gates for inputs.

    Qubit layout (allocation order): a[0..n-1], b[0..n-1], z — 2n+1 qubits.
    """
    from qiskit import QuantumCircuit

    from qamomile.qiskit import QiskitTranspiler

    kernel = _make_adder_kernel(n)
    transpiler = QiskitTranspiler()
    adder_qc = transpiler.to_circuit(kernel)

    # Remove final measurements so we can do statevector simulation
    adder_qc.remove_final_measurements()

    # Prepend X gates to encode a_val and b_val
    prep = QuantumCircuit(adder_qc.num_qubits)
    for i in range(n):
        if (a_val >> i) & 1:
            prep.x(i)  # a register: qubits 0..n-1
        if (b_val >> i) & 1:
            prep.x(n + i)  # b register: qubits n..2n-1

    return prep.compose(adder_qc)


def _simulate_statevector(qc):
    """Simulate circuit and return statevector as numpy array."""
    from qiskit_aer import AerSimulator

    qc.save_statevector()
    simulator = AerSimulator(method="statevector")
    result = simulator.run(qc).result()
    return np.array(result.get_statevector())


def _expected_basis_index(n: int, a_val: int, b_val: int) -> int:
    """Compute expected basis state index after TTK adder.

    Layout: (a_0..a_{n-1}, b_0..b_{n-1}, z), Qiskit little-endian.
    After adder: a restored, b = sum low bits, z = carry.
    """
    total = a_val + b_val
    sum_low = total % (1 << n)
    carry = total >> n
    return a_val | (sum_low << n) | (carry << (2 * n))


@pytest.mark.parametrize(
    "a_val, b_val",
    [(a, b) for a in range(2) for b in range(2)],
)
def test_ttk_adder_correctness_1bit(a_val: int, b_val: int) -> None:
    """Exhaustive test for n=1 (4 cases)."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    n = 1
    qc = _build_adder_circuit(n, a_val, b_val)
    sv = _simulate_statevector(qc)

    expected_idx = _expected_basis_index(n, a_val, b_val)
    num_qubits = qc.num_qubits
    expected_sv = np.zeros(2**num_qubits, dtype=complex)
    expected_sv[expected_idx] = 1.0

    assert np.allclose(np.abs(sv), np.abs(expected_sv), atol=1e-8), (
        f"n={n}, a={a_val}, b={b_val}: "
        f"expected index {expected_idx}, got nonzero at {np.nonzero(np.abs(sv) > 0.5)}"
    )


@pytest.mark.parametrize(
    "a_val, b_val",
    [(a, b) for a in range(4) for b in range(4)],
)
def test_ttk_adder_correctness_2bit(a_val: int, b_val: int) -> None:
    """Exhaustive test for n=2 (16 cases)."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    n = 2
    qc = _build_adder_circuit(n, a_val, b_val)
    sv = _simulate_statevector(qc)

    expected_idx = _expected_basis_index(n, a_val, b_val)
    num_qubits = qc.num_qubits
    expected_sv = np.zeros(2**num_qubits, dtype=complex)
    expected_sv[expected_idx] = 1.0

    assert np.allclose(np.abs(sv), np.abs(expected_sv), atol=1e-8), (
        f"n={n}, a={a_val}, b={b_val}: "
        f"expected index {expected_idx}, got nonzero at {np.nonzero(np.abs(sv) > 0.5)}"
    )


@pytest.mark.parametrize(
    "a_val, b_val",
    [(a, b) for a in range(8) for b in range(8)],
)
def test_ttk_adder_correctness_3bit(a_val: int, b_val: int) -> None:
    """Exhaustive test for n=3 (64 cases)."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    n = 3
    qc = _build_adder_circuit(n, a_val, b_val)
    sv = _simulate_statevector(qc)

    expected_idx = _expected_basis_index(n, a_val, b_val)
    num_qubits = qc.num_qubits
    expected_sv = np.zeros(2**num_qubits, dtype=complex)
    expected_sv[expected_idx] = 1.0

    assert np.allclose(np.abs(sv), np.abs(expected_sv), atol=1e-8), (
        f"n={n}, a={a_val}, b={b_val}: "
        f"expected index {expected_idx}, got nonzero at {np.nonzero(np.abs(sv) > 0.5)}"
    )


# ---------------------------------------------------------------------------
# Random cross-validation: Qamomile pipeline vs Qiskit reference (n=5, 8)
# ---------------------------------------------------------------------------


def _build_reference_circuit(n: int, a_val: int, b_val: int):
    """Build TTK adder directly in Qiskit as independent reference.

    Implements the 6-step algorithm from arXiv:0910.2530v1, Section 2.2.
    Qubit layout: (a_0..a_{n-1}, b_0..b_{n-1}, z) — 2n+1 qubits.
    """
    from qiskit import QuantumCircuit

    num_qubits = 2 * n + 1
    qc = QuantumCircuit(num_qubits)

    def A(i: int) -> int:
        return i

    def B(i: int) -> int:
        return n + i

    Z = 2 * n

    # Encode inputs
    for i in range(n):
        if (a_val >> i) & 1:
            qc.x(A(i))
        if (b_val >> i) & 1:
            qc.x(B(i))

    # TTK adder 6-step algorithm
    if n == 1:
        qc.ccx(B(0), A(0), Z)
        qc.cx(A(0), B(0))
    else:
        # Step 1
        for i in range(1, n):
            qc.cx(A(i), B(i))
        # Step 2
        for i in range(n - 1, 0, -1):
            tgt = Z if i == n - 1 else A(i + 1)
            qc.cx(A(i), tgt)
        # Step 3
        for i in range(n):
            tgt = Z if i == n - 1 else A(i + 1)
            qc.ccx(B(i), A(i), tgt)
        # Step 4
        for i in range(n - 1, 0, -1):
            qc.cx(A(i), B(i))
            qc.ccx(B(i - 1), A(i - 1), A(i))
        # Step 5
        for i in range(1, n - 1):
            qc.cx(A(i), A(i + 1))
        # Step 6
        for i in range(n):
            qc.cx(A(i), B(i))

    return qc


@pytest.mark.parametrize("n", [5, 8])
def test_ttk_adder_random_vs_qiskit(n: int) -> None:
    """Random cross-validation: Qamomile pipeline vs Qiskit reference."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    rng = np.random.default_rng(42)
    num_samples = 10

    for _ in range(num_samples):
        a_val = int(rng.integers(0, 2**n))
        b_val = int(rng.integers(0, 2**n))

        qamomile_qc = _build_adder_circuit(n, a_val, b_val)
        reference_qc = _build_reference_circuit(n, a_val, b_val)

        qamomile_sv = _simulate_statevector(qamomile_qc)
        reference_sv = _simulate_statevector(reference_qc)

        assert np.allclose(np.abs(qamomile_sv), np.abs(reference_sv), atol=1e-8), (
            f"n={n}, a={a_val}, b={b_val}: "
            f"Qamomile and Qiskit reference statevectors differ"
        )
