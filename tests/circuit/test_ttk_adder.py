"""Tests for TTK inplace adder (arXiv:0910.2530v1, Section 2.2)."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.operation.composite_gate import CompositeGateType
from qamomile.circuit.stdlib.ttk_adder import TTKInplaceAdder
from tests.circuit.conftest import run_statevector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_resources(n: int) -> dict[str, int]:
    """Expected TTK adder resource counts for n-bit operands."""
    if n == 1:
        return {
            "num_cnot_gates": 1,
            "num_toffoli_gates": 1,
            "total_gates": 2,
            "depth": 2,
        }
    return {
        "num_cnot_gates": 5 * n - 5,
        "num_toffoli_gates": 2 * n - 1,
        "total_gates": 7 * n - 6,
        "depth": 5 * n - 3,
    }


def _make_adder_kernel(n: int):
    """Create an adder qkernel for n-bit operands."""

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

    # Prepend X gates to encode a_val and b_val
    prep = QuantumCircuit(adder_qc.num_qubits)
    for i in range(n):
        if (a_val >> i) & 1:
            prep.x(i)  # a register: qubits 0..n-1
        if (b_val >> i) & 1:
            prep.x(n + i)  # b register: qubits n..2n-1

    return prep.compose(adder_qc)


def _expected_basis_index(n: int, a_val: int, b_val: int) -> int:
    """Compute expected basis state index after TTK adder.

    Layout: (a_0..a_{n-1}, b_0..b_{n-1}, z), Qiskit little-endian.
    After adder: a restored, b = sum low bits, z = carry.
    """
    total = a_val + b_val
    sum_low = total % (1 << n)
    carry = total >> n
    return a_val | (sum_low << n) | (carry << (2 * n))


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


# ---------------------------------------------------------------------------
# Unit tests: attributes, resources, validation
# ---------------------------------------------------------------------------


class TestTTKInplaceAdder:
    """Unit tests for TTKInplaceAdder construction, attributes, and resources."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_class_attributes(self, n: int) -> None:
        adder = TTKInplaceAdder(n)
        assert adder.custom_name == "ttk_adder"
        assert adder.gate_type == CompositeGateType.CUSTOM
        assert adder.num_target_qubits == 2 * n + 1

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            TTKInplaceAdder(0)
        with pytest.raises(ValueError, match="n must be >= 1"):
            TTKInplaceAdder(-1)

    def test_mismatched_registers(self) -> None:
        """Mismatched register sizes raise ValueError."""
        from qamomile.circuit.stdlib.ttk_adder import ttk_adder

        @qm.qkernel
        def circuit() -> qm.Bit:
            a = qm.qubit_array(2, "a")
            b = qm.qubit_array(3, "b")
            z = qm.qubit("z")
            b, a, z = ttk_adder(b, a, z)
            return qm.measure(z)

        with pytest.raises(ValueError, match="same size"):
            circuit.build()

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_resources(self, n: int) -> None:
        r = TTKInplaceAdder(n)._resources()
        expected = _expected_resources(n)
        for key, value in expected.items():
            assert r.custom_metadata[key] == value


# ---------------------------------------------------------------------------
# Statevector correctness (exhaustive, small n)
# ---------------------------------------------------------------------------


class TestTTKAdderCorrectness:
    """Exhaustive statevector verification for small n."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_exhaustive_addition(self, n: int) -> None:
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        for a_val in range(2**n):
            for b_val in range(2**n):
                qc = _build_adder_circuit(n, a_val, b_val)
                sv = run_statevector(qc)

                expected_idx = _expected_basis_index(n, a_val, b_val)
                num_qubits = qc.num_qubits
                expected_sv = np.zeros(2**num_qubits, dtype=complex)
                expected_sv[expected_idx] = 1.0

                assert np.allclose(np.abs(sv), np.abs(expected_sv), atol=1e-8), (
                    f"n={n}, a={a_val}, b={b_val}: "
                    f"expected index {expected_idx}, "
                    f"got nonzero at {np.nonzero(np.abs(sv) > 0.5)}"
                )


# ---------------------------------------------------------------------------
# Cross-validation: Qamomile pipeline vs Qiskit reference (large n)
# ---------------------------------------------------------------------------


class TestTTKAdderCrossValidation:
    """Random cross-validation against independent Qiskit reference."""

    @pytest.mark.parametrize("n", [5, 8])
    def test_random_vs_qiskit_reference(self, n: int) -> None:
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        rng = np.random.default_rng(42)
        num_samples = 10

        for _ in range(num_samples):
            a_val = int(rng.integers(0, 2**n))
            b_val = int(rng.integers(0, 2**n))

            qamomile_qc = _build_adder_circuit(n, a_val, b_val)
            reference_qc = _build_reference_circuit(n, a_val, b_val)

            qamomile_sv = run_statevector(qamomile_qc)
            reference_sv = run_statevector(reference_qc)

            assert np.allclose(np.abs(qamomile_sv), np.abs(reference_sv), atol=1e-8), (
                f"n={n}, a={a_val}, b={b_val}: "
                f"Qamomile and Qiskit reference statevectors differ"
            )
