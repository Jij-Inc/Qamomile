"""Tests for ``computational_basis_state``.

The kernel prepares ``|bits>`` from ``|0>^n`` via exact conditional X powers.
Sampling must yield exactly the prepared bit string on every shot, and
the expectation value of any diagonal Pauli-Z Hamiltonian must match the
analytical eigenvalue at ``|bits>``.  We pin both paths down on Qiskit,
QURI Parts, and CUDA-Q so that any backend-specific Rx parameter or
measurement-readout bug surfaces here.
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib.state_preparation import computational_basis_state
from qamomile.circuit.stdlib.state_preparation.computational_basis_state import (
    _apply_exact_basis_bit,
)


@qmc.qkernel
def _cb_sample_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Top-level wrapper: prepare |bits> from |0>^n and measure all qubits."""
    q = qmc.qubit_array(n, name="q")
    q = computational_basis_state(q, bits)
    return qmc.measure(q)


@qmc.qkernel
def _cb_expval_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Top-level wrapper: prepare |bits> and return <bits|hamiltonian|bits>."""
    q = qmc.qubit_array(n, name="q")
    q = computational_basis_state(q, bits)
    return qmc.expval(q, hamiltonian)


def _quri_parts_transpiler():
    """Return a ``QuriPartsTranspiler`` or skip the test if unavailable."""
    pytest.importorskip("quri_parts.qulacs")
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


def _cudaq_transpiler():
    """Return a ``CudaqTranspiler`` or skip the test if unavailable."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


def _qiskit_transpiler():
    """Return a ``QiskitTranspiler`` or skip the test if unavailable."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


def _random_bits(rng: np.random.Generator, n: int) -> list[int]:
    """Draw a random {0, 1}^n bit vector as a Python list of ints."""
    return [int(b) for b in rng.integers(0, 2, size=n)]


def _z_hamiltonian(coeffs: list[float]) -> qm_o.Hamiltonian:
    """Build sum_i coeffs[i] * Z(i) as a Hamiltonian."""
    H = qm_o.Hamiltonian()
    for i, c in enumerate(coeffs):
        H += float(c) * qm_o.Z(i)
    return H


def _exact_z_expval(coeffs: list[float], bits: list[int]) -> float:
    """Analytic ``<bits | sum_i c_i Z_i | bits> = sum_i c_i (1 - 2 b_i)``."""
    return float(sum(c * (1.0 - 2.0 * b) for c, b in zip(coeffs, bits)))


def _sample_compile_time(transpiler, n: int, bits: list[int], shots: int):
    """Transpile with ``bits`` bound and return the sample result list."""
    exe = transpiler.transpile(
        _cb_sample_kernel,
        bindings={"n": n, "bits": bits},
    )
    job = exe.sample(transpiler.executor(), bindings={}, shots=shots)
    return job.result().results


def _sample_runtime_bits(transpiler, n: int, bits: list[int], shots: int):
    """Transpile with ``bits`` left as a runtime parameter, bind at sample.

    Only ``n`` is bound at compile time; the kernel's ``for i in qmc.range(n)``
    body fixes the gate count without needing a placeholder for ``bits``.
    """
    exe = transpiler.transpile(
        _cb_sample_kernel,
        bindings={"n": n},
        parameters=["bits"],
    )
    job = exe.sample(transpiler.executor(), bindings={"bits": bits}, shots=shots)
    return job.result().results


def _expval_compile_time(
    transpiler, n: int, bits: list[int], hamiltonian: qm_o.Hamiltonian
) -> float:
    """Transpile with ``bits`` bound and return the expval."""
    exe = transpiler.transpile(
        _cb_expval_kernel,
        bindings={"n": n, "bits": bits, "hamiltonian": hamiltonian},
    )
    return exe.run(transpiler.executor(), bindings={}).result()


def _assert_deterministic_sample(results, expected_bits: list[int], shots: int):
    """All shots must return the prepared bit pattern."""
    expected = tuple(expected_bits)
    total = 0
    for sampled, count in results:
        sampled_tuple = tuple(int(b) for b in sampled)
        assert sampled_tuple == expected, (
            f"sampled {sampled_tuple} != expected {expected} (count={count})"
        )
        total += count
    assert total == shots


# Test grids: register sizes and RNG seeds.  Multiple sizes guard against
# off-by-one bugs in the for-loop bound; multiple seeds randomize the bit
# pattern and hamiltonian coefficients without flake.
_SIZES = [1, 2, 3, 5]
_SEEDS = [0, 1, 2, 42]
_BOUNDARY_SIZES = [1, 3, 4]


@qmc.qkernel
def _controlled_basis_bit_phase_probe(bit: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Expose any conditional-X global phase through control interference."""
    qubits = qmc.qubit_array(2, "qubits")
    control = qubits[0]
    target = qubits[1]
    control = qmc.h(control)
    target = qmc.h(target)
    control, target = qmc.control(_apply_exact_basis_bit)(control, target, bit)
    control = qmc.h(control)
    qubits[0] = control
    qubits[1] = target
    return qmc.measure(qubits)


def test_controlled_basis_bit_has_no_spurious_relative_phase() -> None:
    """Controlled bit=1 acts as controlled-X, not controlled-RX(pi)."""
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    result = (
        transpiler.transpile(
            _controlled_basis_bit_phase_probe,
            bindings={"bit": 1},
        )
        .sample(transpiler.executor(), shots=128)
        .result()
    )

    assert all(sample[0] == 0 for sample, _ in result.results)


class TestComputationalBasisStateSample:
    """Deterministic sampling: |bits> sampled with probability 1 on every backend."""

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_qiskit(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        shots = 64
        results = _sample_compile_time(_qiskit_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_quri_parts(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        shots = 64
        results = _sample_compile_time(_quri_parts_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)

    @pytest.mark.cudaq
    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_cudaq(self, n: int, seed: int) -> None:
        """CUDA-Q leg; ``-m cudaq`` sessions only (see tests/_cudaq_isolation.py)."""
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        shots = 64
        results = _sample_compile_time(_cudaq_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)


class TestComputationalBasisStateRuntimeBits:
    """``bits`` as a runtime parameter: bind at ``sample()`` time, not transpile.

    Mirrors the QeMCMC tutorial use case where the proposal circuit takes
    the current MCMC state as a runtime input.  Validates that the Rx-based
    state preparation transpiles without an unresolved-shape error and that
    runtime binding still produces the deterministic outcome.
    """

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_qiskit(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        shots = 64
        results = _sample_runtime_bits(_qiskit_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_quri_parts(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        shots = 64
        results = _sample_runtime_bits(_quri_parts_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)

    @pytest.mark.cudaq
    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_cudaq(self, n: int, seed: int) -> None:
        """CUDA-Q leg; ``-m cudaq`` sessions only (see tests/_cudaq_isolation.py)."""
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        shots = 64
        results = _sample_runtime_bits(_cudaq_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)


class TestComputationalBasisStateExpval:
    """Expectation values: ``<bits | sum c_i Z_i | bits> = sum c_i (1 - 2 b_i)``."""

    _TOLERANCE = 1e-8

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_qiskit(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        coeffs = rng.uniform(-1.0, 1.0, size=n).tolist()
        H = _z_hamiltonian(coeffs)
        expected = _exact_z_expval(coeffs, bits)

        observed = _expval_compile_time(_qiskit_transpiler(), n, bits, H)
        assert abs(observed - expected) < self._TOLERANCE, (
            f"qiskit expval={observed}, expected={expected}"
        )

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_quri_parts(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        coeffs = rng.uniform(-1.0, 1.0, size=n).tolist()
        H = _z_hamiltonian(coeffs)
        expected = _exact_z_expval(coeffs, bits)

        observed = _expval_compile_time(_quri_parts_transpiler(), n, bits, H)
        assert abs(observed - expected) < self._TOLERANCE, (
            f"quri_parts expval={observed}, expected={expected}"
        )

    @pytest.mark.cudaq
    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_cudaq(self, n: int, seed: int) -> None:
        """CUDA-Q leg; ``-m cudaq`` sessions only (see tests/_cudaq_isolation.py)."""
        rng = np.random.default_rng(seed)
        bits = _random_bits(rng, n)
        coeffs = rng.uniform(-1.0, 1.0, size=n).tolist()
        H = _z_hamiltonian(coeffs)
        expected = _exact_z_expval(coeffs, bits)

        observed = _expval_compile_time(_cudaq_transpiler(), n, bits, H)
        assert abs(observed - expected) < self._TOLERANCE, (
            f"cudaq expval={observed}, expected={expected}"
        )


class TestComputationalBasisStateBoundary:
    """All-zero and all-one bit patterns: identity preparation and full flip."""

    @pytest.mark.parametrize("n", _BOUNDARY_SIZES)
    @pytest.mark.parametrize("pattern", ["all_zero", "all_one"])
    def test_qiskit(self, n: int, pattern: str) -> None:
        bits = [0] * n if pattern == "all_zero" else [1] * n
        shots = 32
        results = _sample_compile_time(_qiskit_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)

    @pytest.mark.parametrize("n", _BOUNDARY_SIZES)
    @pytest.mark.parametrize("pattern", ["all_zero", "all_one"])
    def test_quri_parts(self, n: int, pattern: str) -> None:
        bits = [0] * n if pattern == "all_zero" else [1] * n
        shots = 32
        results = _sample_compile_time(_quri_parts_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)

    @pytest.mark.cudaq
    @pytest.mark.parametrize("n", _BOUNDARY_SIZES)
    @pytest.mark.parametrize("pattern", ["all_zero", "all_one"])
    def test_cudaq(self, n: int, pattern: str) -> None:
        """CUDA-Q leg; ``-m cudaq`` sessions only (see tests/_cudaq_isolation.py)."""
        bits = [0] * n if pattern == "all_zero" else [1] * n
        shots = 32
        results = _sample_compile_time(_cudaq_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)
