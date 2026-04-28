"""Tests for basic rotation and entanglement layers."""

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import (
    computational_basis_state,
    cx_entangling_layer,
    cz_entangling_layer,
    rx_layer,
    ry_layer,
    rz_layer,
)
from qamomile.circuit.estimator import count_gates

# ---------------------------------------------------------------------------
# Symbolic gate count tests (count_gates on IR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_rx_layer_gate_count(num_qubits):
    """Test that rx_layer produces n RX gates."""
    counts = count_gates(rx_layer.block)
    q_dim0 = sp.Symbol("q_dim0", integer=True, positive=True)
    assert counts.single_qubit.subs(q_dim0, num_qubits) == num_qubits
    assert counts.two_qubit.subs(q_dim0, num_qubits) == 0


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_ry_layer_gate_count(num_qubits):
    """Test that ry_layer produces n RY gates."""
    counts = count_gates(ry_layer.block)
    q_dim0 = sp.Symbol("q_dim0", integer=True, positive=True)
    assert counts.single_qubit.subs(q_dim0, num_qubits) == num_qubits
    assert counts.two_qubit.subs(q_dim0, num_qubits) == 0


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_rz_layer_gate_count(num_qubits):
    """Test that rz_layer produces n RZ gates."""
    counts = count_gates(rz_layer.block)
    q_dim0 = sp.Symbol("q_dim0", integer=True, positive=True)
    assert counts.single_qubit.subs(q_dim0, num_qubits) == num_qubits
    assert counts.two_qubit.subs(q_dim0, num_qubits) == 0


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_cz_entangling_layer_gate_count(num_qubits):
    """Test that cz_entangling_layer produces n-1 CZ gates."""
    counts = count_gates(cz_entangling_layer.block)
    q_dim0 = sp.Symbol("q_dim0", integer=True, positive=True)
    assert counts.single_qubit.subs(q_dim0, num_qubits) == 0
    assert counts.two_qubit.subs(q_dim0, num_qubits) == num_qubits - 1


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_cx_entangling_layer_gate_count(num_qubits):
    """Test that cx_entangling_layer produces n-1 CX gates."""
    counts = count_gates(cx_entangling_layer.block)
    q_dim0 = sp.Symbol("q_dim0", integer=True, positive=True)
    assert counts.single_qubit.subs(q_dim0, num_qubits) == 0
    assert counts.two_qubit.subs(q_dim0, num_qubits) == num_qubits - 1


# ---------------------------------------------------------------------------
# Transpiler-based tests (concrete bindings with QiskitTranspiler)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_qubits,offset", [(3, 0), (4, 2), (2, 5)])
def test_rx_layer_transpiled(num_qubits, offset):
    """Test rx_layer produces correct RX gates with concrete bindings."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(
        n: qmc.UInt,
        thetas: qmc.Vector[qmc.Float],
        off: qmc.UInt,
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = rx_layer(q, thetas, off)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    n_params = offset + num_qubits
    thetas_val = [0.1 * i for i in range(n_params)]
    executor = transpiler.transpile(
        circuit,
        bindings={"n": num_qubits, "thetas": thetas_val, "off": offset},
    )

    qc = executor.compiled_quantum[0].circuit
    rx_gates = [inst for inst in qc.data if inst.operation.name == "rx"]
    assert len(rx_gates) == num_qubits, (
        f"Expected {num_qubits} RX gates, got {len(rx_gates)}"
    )

    # Verify parameter values match thetas[offset], thetas[offset+1], ...
    expected_params = [thetas_val[offset + i] for i in range(num_qubits)]
    for i, inst in enumerate(rx_gates):
        actual = float(inst.operation.params[0])
        assert abs(actual - expected_params[i]) < 1e-10, (
            f"RX gate {i}: expected angle {expected_params[i]}, got {actual}"
        )


@pytest.mark.parametrize("num_qubits,offset", [(3, 0), (4, 2), (2, 5)])
def test_ry_layer_transpiled(num_qubits, offset):
    """Test ry_layer produces correct RY gates with concrete bindings."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(
        n: qmc.UInt,
        thetas: qmc.Vector[qmc.Float],
        off: qmc.UInt,
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = ry_layer(q, thetas, off)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    n_params = offset + num_qubits
    thetas_val = [0.1 * i for i in range(n_params)]
    executor = transpiler.transpile(
        circuit,
        bindings={"n": num_qubits, "thetas": thetas_val, "off": offset},
    )

    qc = executor.compiled_quantum[0].circuit
    ry_gates = [inst for inst in qc.data if inst.operation.name == "ry"]
    assert len(ry_gates) == num_qubits, (
        f"Expected {num_qubits} RY gates, got {len(ry_gates)}"
    )

    # Verify parameter values match thetas[offset], thetas[offset+1], ...
    expected_params = [thetas_val[offset + i] for i in range(num_qubits)]
    for i, inst in enumerate(ry_gates):
        actual = float(inst.operation.params[0])
        assert abs(actual - expected_params[i]) < 1e-10, (
            f"RY gate {i}: expected angle {expected_params[i]}, got {actual}"
        )


@pytest.mark.parametrize("num_qubits,offset", [(3, 0), (4, 2), (2, 5)])
def test_rz_layer_transpiled(num_qubits, offset):
    """Test rz_layer produces correct RZ gates with concrete bindings."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(
        n: qmc.UInt,
        thetas: qmc.Vector[qmc.Float],
        off: qmc.UInt,
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = rz_layer(q, thetas, off)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    n_params = offset + num_qubits
    thetas_val = [0.1 * i for i in range(n_params)]
    executor = transpiler.transpile(
        circuit,
        bindings={"n": num_qubits, "thetas": thetas_val, "off": offset},
    )

    qc = executor.compiled_quantum[0].circuit
    rz_gates = [inst for inst in qc.data if inst.operation.name == "rz"]
    assert len(rz_gates) == num_qubits, (
        f"Expected {num_qubits} RZ gates, got {len(rz_gates)}"
    )

    # Verify parameter values match thetas[offset], thetas[offset+1], ...
    expected_params = [thetas_val[offset + i] for i in range(num_qubits)]
    for i, inst in enumerate(rz_gates):
        actual = float(inst.operation.params[0])
        assert abs(actual - expected_params[i]) < 1e-10, (
            f"RZ gate {i}: expected angle {expected_params[i]}, got {actual}"
        )


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_cz_entangling_layer_transpiled(num_qubits):
    """Test cz_entangling_layer produces correct CZ gates with concrete bindings."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(
        n: qmc.UInt,
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = cz_entangling_layer(q)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executor = transpiler.transpile(
        circuit,
        bindings={"n": num_qubits},
    )

    qc = executor.compiled_quantum[0].circuit
    cz_count = sum(1 for inst in qc.data if inst.operation.name == "cz")
    assert cz_count == num_qubits - 1, (
        f"Expected {num_qubits - 1} CZ gates, got {cz_count}"
    )


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_cx_entangling_layer_transpiled(num_qubits):
    """Test cx_entangling_layer produces correct CX gates with concrete bindings."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = cx_entangling_layer(q)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executor = transpiler.transpile(circuit, bindings={"n": num_qubits})
    qc = executor.compiled_quantum[0].circuit
    cx_count = sum(1 for inst in qc.data if inst.operation.name == "cx")
    assert cx_count == num_qubits - 1


# ---------------------------------------------------------------------------
# computational_basis_state — cross-backend sampling + expval
#
# The kernel prepares |bits> from |0>^n via Rx(pi * bits[i]) on each qubit.
# Sampling must yield exactly the prepared bit string on every shot, and
# expval of any diagonal Pauli-Z Hamiltonian must match the analytical
# eigenvalue at |bits>.  We pin both paths down on Qiskit, QURI Parts, and
# CUDA-Q so that any backend-specific Rx parameter or measurement-readout
# bug surfaces here.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _cb_sample_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Top-level wrapper: prepare |bits> from |0>^n and measure all qubits."""
    q = qmc.qubit_array(n, name="q")
    q = computational_basis_state(n, q, bits)
    return qmc.measure(q)


@qmc.qkernel
def _cb_expval_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Top-level wrapper: prepare |bits> and return <bits|hamiltonian|bits>."""
    q = qmc.qubit_array(n, name="q")
    q = computational_basis_state(n, q, bits)
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
    """Transpile with ``bits`` left as a runtime parameter, bind at sample."""
    exe = transpiler.transpile(
        _cb_sample_kernel,
        bindings={"n": n, "bits": [0] * n},  # shape hint
        parameters=["bits"],
    )
    job = exe.sample(
        transpiler.executor(), bindings={"bits": bits}, shots=shots
    )
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

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_cudaq(self, n: int, seed: int) -> None:
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

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_cudaq(self, n: int, seed: int) -> None:
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

    @pytest.mark.parametrize("n", _SIZES)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_cudaq(self, n: int, seed: int) -> None:
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

    @pytest.mark.parametrize("n", _BOUNDARY_SIZES)
    @pytest.mark.parametrize("pattern", ["all_zero", "all_one"])
    def test_cudaq(self, n: int, pattern: str) -> None:
        bits = [0] * n if pattern == "all_zero" else [1] * n
        shots = 32
        results = _sample_compile_time(_cudaq_transpiler(), n, bits, shots)
        _assert_deterministic_sample(results, bits, shots)
