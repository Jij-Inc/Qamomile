"""Rich CUDA-Q frontend-to-backend test suite.

Tests the full pipeline: @qkernel definition -> CudaqTranspiler -> execution.
Covers every frontend gate, gate combinations, control flow (loops), stdlib,
measurement, parametric circuits, expval, and edge cases.

CUDA-Q 0.14.x does not support measurement-dependent conditional branching
(``c_if`` was removed). Tests that rely on Qiskit-specific circuit
introspection are replaced with statevector or sampling verification.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from tests.transpiler.gate_test_specs import (
    GATE_SPECS,
    all_zeros_state,
    computational_basis_state,
    compute_expected_statevector,
    identity,
    statevectors_equal,
    tensor_product,
)

# ---------------------------------------------------------------------------
# Skip entire module if cudaq is not installed
# ---------------------------------------------------------------------------
cudaq = pytest.importorskip("cudaq")

from qamomile.cudaq import CudaqTranspiler  # noqa: E402
from qamomile.cudaq.emitter import CudaqCircuit  # noqa: E402
from qamomile.circuit.transpiler.errors import EmitError  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_statevector(circuit: CudaqCircuit) -> np.ndarray:
    """Run a CudaqCircuit and return the statevector.

    Args:
        circuit: CudaqCircuit wrapper with kernel and qubits.

    Returns:
        Numpy array of complex amplitudes.
    """
    import cudaq

    state = cudaq.get_state(circuit.kernel)
    return np.array(state)


def _transpile_and_get_circuit(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
) -> tuple[Any, CudaqCircuit]:
    """Transpile a qkernel and return the executable and CUDA-Q circuit.

    Args:
        kernel: A ``@qmc.qkernel`` decorated function.
        bindings: Optional parameter bindings dict.
        parameters: Optional list of parameter names to leave symbolic.

    Returns:
        Tuple of (ExecutableProgram, CudaqCircuit).
    """
    transpiler = CudaqTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    circuit = exe.compiled_quantum[0].circuit
    return exe, circuit


# ============================================================================
# 1. Individual Gate Tests – statevector verification
# ============================================================================


class TestSingleQubitGatesFrontend:
    """Test each single-qubit gate through the full frontend pipeline."""

    # -- Hadamard --

    def test_h_statevector(self):
        """H|0> produces equal superposition (|0>+|1>)/sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_qubits == 1
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    # -- Pauli-X --

    def test_x_statevector(self):
        """X|0> = |1>."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_qubits == 1
        sv = _run_statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(sv, expected)

    # -- RX --

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rx_statevector(self, angle):
        """RX(theta) statevector matches analytical RX matrix."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        assert qc.num_qubits == 1
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- RY --

    @pytest.mark.parametrize(
        "angle",
        [np.pi / 2]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_ry_statevector(self, angle):
        """RY(theta) statevector matches analytical RY matrix."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- RZ --

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rz_statevector(self, angle):
        """RZ(theta) statevector matches analytical RZ matrix."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RZ"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- P (Phase) --

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_p_statevector(self, angle):
        """P(theta) statevector matches analytical P matrix."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["P"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- Pauli-Y --

    def test_y_statevector(self):
        """Y|0> = i|1>."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.y(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["Y"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    # -- Z, S, Sdg, T, Tdg --

    def test_z_statevector(self):
        """Z|0> = |0> (no visible change on |0>)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["Z"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_s_statevector(self):
        """S gate on |+> applies phase pi/2."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        h_matrix = GATE_SPECS["H"].matrix_fn()
        s_matrix = GATE_SPECS["S"].matrix_fn()
        expected = compute_expected_statevector(
            compute_expected_statevector(all_zeros_state(1), h_matrix), s_matrix
        )
        assert statevectors_equal(sv, expected)

    def test_sdg_statevector(self):
        """Sdg gate (S-dagger) on |+> applies phase -pi/2."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        h_matrix = GATE_SPECS["H"].matrix_fn()
        sdg_matrix = GATE_SPECS["SDG"].matrix_fn()
        expected = compute_expected_statevector(
            compute_expected_statevector(all_zeros_state(1), h_matrix), sdg_matrix
        )
        assert statevectors_equal(sv, expected)

    def test_t_statevector(self):
        """T gate on |+> applies phase pi/4."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        h_matrix = GATE_SPECS["H"].matrix_fn()
        t_matrix = GATE_SPECS["T"].matrix_fn()
        expected = compute_expected_statevector(
            compute_expected_statevector(all_zeros_state(1), h_matrix), t_matrix
        )
        assert statevectors_equal(sv, expected)

    def test_tdg_statevector(self):
        """Tdg gate (T-dagger) on |+> applies phase -pi/4."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        h_matrix = GATE_SPECS["H"].matrix_fn()
        tdg_matrix = GATE_SPECS["TDG"].matrix_fn()
        expected = compute_expected_statevector(
            compute_expected_statevector(all_zeros_state(1), h_matrix), tdg_matrix
        )
        assert statevectors_equal(sv, expected)


# ============================================================================
# 2. Two-Qubit Gate Tests
# ============================================================================


class TestTwoQubitGatesFrontend:
    """Test two-qubit gates through the frontend pipeline."""

    @pytest.mark.parametrize("initial_state_index", [0, 1, 2, 3])
    def test_cx_statevector(self, initial_state_index):
        """CX on all 4 basis states."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            if initial_state_index & 1:
                q[0] = qmc.x(q[0])
            if initial_state_index & 2:
                q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_qubits == 2
        sv = _run_statevector(qc)
        initial = computational_basis_state(2, initial_state_index)
        expected = compute_expected_statevector(initial, GATE_SPECS["CX"].matrix_fn())
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("initial_state_index", [0, 1, 2, 3])
    def test_cz_statevector(self, initial_state_index):
        """CZ on all 4 basis states."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            if initial_state_index & 1:
                q[0] = qmc.x(q[0])
            if initial_state_index & 2:
                q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        initial = computational_basis_state(2, initial_state_index)
        expected = compute_expected_statevector(initial, GATE_SPECS["CZ"].matrix_fn())
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_cp_statevector(self, angle):
        """CP(theta) on |11> state."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        initial = computational_basis_state(2, 3)
        expected = compute_expected_statevector(
            initial, GATE_SPECS["CP"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rzz_statevector(self, angle):
        """RZZ(theta) on |11> state."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        initial = computational_basis_state(2, 3)
        expected = compute_expected_statevector(
            initial, GATE_SPECS["RZZ"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("initial_state_index", [0, 1, 2, 3])
    def test_swap_statevector(self, initial_state_index):
        """SWAP on all 4 basis states."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            if initial_state_index & 1:
                q[0] = qmc.x(q[0])
            if initial_state_index & 2:
                q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        initial = computational_basis_state(2, initial_state_index)
        expected = compute_expected_statevector(initial, GATE_SPECS["SWAP"].matrix_fn())
        assert statevectors_equal(sv, expected)


# ============================================================================
# 3. Three-Qubit Gate Tests
# ============================================================================


class TestThreeQubitGatesFrontend:
    """Test three-qubit gates through the frontend pipeline."""

    @pytest.mark.parametrize("basis_idx", list(range(8)))
    def test_ccx_statevector(self, basis_idx):
        """Toffoli on all 8 basis states (uses cx([controls], target))."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            if basis_idx & 1:
                q[0] = qmc.x(q[0])
            if basis_idx & 2:
                q[1] = qmc.x(q[1])
            if basis_idx & 4:
                q[2] = qmc.x(q[2])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_qubits == 3
        sv = _run_statevector(qc)
        initial = computational_basis_state(3, basis_idx)
        expected = compute_expected_statevector(
            initial, GATE_SPECS["TOFFOLI"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)


# ============================================================================
# 4. Parametric Gate Tests
# ============================================================================


class TestParametricGates:
    """Test that gates with unbound parameters produce parameterized circuits.

    Each test verifies:
    1. Circuit topology (num_qubits)
    2. Parameter metadata (has_parameters, count, name)
    3. Functional correctness: binding a concrete value produces the correct
       statevector, proving the parameter is wired through the gate.
    """

    def test_rx_parametric(self):
        """RX with unbound theta: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 1
        assert exe.has_parameters
        assert len(exe.parameter_names) == 1
        assert "theta" in exe.parameter_names
        # Bind and verify statevector
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    def test_ry_parametric(self):
        """RY with unbound theta: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 1
        assert exe.has_parameters
        assert len(exe.parameter_names) == 1
        assert "theta" in exe.parameter_names
        # Bind and verify
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi / 2}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(np.pi / 2)
        )
        assert statevectors_equal(sv, expected)

    def test_rz_parametric(self):
        """RZ with unbound theta: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 1
        assert exe.has_parameters
        assert len(exe.parameter_names) == 1
        assert "theta" in exe.parameter_names
        # Bind and verify
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RZ"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    def test_p_parametric(self):
        """P with unbound theta: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 1
        assert exe.has_parameters
        assert len(exe.parameter_names) == 1
        assert "theta" in exe.parameter_names
        # Bind and verify
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["P"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    def test_cp_parametric(self):
        """CP with unbound theta: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 2
        assert exe.has_parameters
        assert len(exe.parameter_names) == 1
        assert "theta" in exe.parameter_names
        # Bind and verify on |11> state
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        initial = computational_basis_state(2, 3)
        expected = compute_expected_statevector(
            initial, GATE_SPECS["CP"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    def test_rzz_parametric(self):
        """RZZ with unbound theta: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 2
        assert exe.has_parameters
        assert len(exe.parameter_names) == 1
        assert "theta" in exe.parameter_names
        # Bind and verify on |11> state
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        initial = computational_basis_state(2, 3)
        expected = compute_expected_statevector(
            initial, GATE_SPECS["RZZ"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    def test_vector_parametric(self):
        """Vector[Float] parameter: metadata + bind-and-verify statevector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["thetas"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 3
        assert exe.has_parameters
        assert len(exe.parameter_names) == 3
        # Bind and verify: RY(pi/2) on each qubit
        # Vector parameters are expanded to individual indexed names
        # (e.g., "thetas[0]", "thetas[1]", "thetas[2]") during transpilation.
        executor = transpiler.executor()
        angles = [np.pi / 2, np.pi / 2, np.pi / 2]
        bindings_expanded = {f"thetas[{i}]": v for i, v in enumerate(angles)}
        bound = executor.bind_parameters(
            qc, bindings_expanded, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        # Each qubit: RY(pi/2)|0> = (|0>+|1>)/sqrt(2)
        # All angles are the same, so kron order does not matter.
        single = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(np.pi / 2)
        )
        expected = tensor_product(tensor_product(single, single), single)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 5. Gate Combination Tests
# ============================================================================


class TestGateCombinations:
    """Test multi-gate circuits through the frontend pipeline."""

    def test_bell_state(self):
        """H + CX creates Bell state (|00> + |11>) / sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_ghz_state_3q(self):
        """H + CX chain creates 3-qubit GHZ state."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_rotation_sequence(self):
        """Sequence of RX, RY, RZ rotations."""

        @qmc.qkernel
        def circuit(alpha: qmc.Float, beta: qmc.Float, gamma: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, alpha)
            q = qmc.ry(q, beta)
            q = qmc.rz(q, gamma)
            return qmc.measure(q)

        alpha, beta, gamma = 0.3, 0.7, 1.2
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"alpha": alpha, "beta": beta, "gamma": gamma}
        )
        sv = _run_statevector(qc)
        state = all_zeros_state(1)
        state = compute_expected_statevector(state, GATE_SPECS["RX"].matrix_fn(alpha))
        state = compute_expected_statevector(state, GATE_SPECS["RY"].matrix_fn(beta))
        expected = compute_expected_statevector(
            state, GATE_SPECS["RZ"].matrix_fn(gamma)
        )
        assert statevectors_equal(sv, expected)

    def test_x_then_h(self):
        """X then H: X|0> = |1>, H|1> = (|0> - |1>)/sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_entangle_then_rotate(self):
        """Bell state then local rotation on qubit 1."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1] = qmc.ry(q[1], theta)
            return qmc.measure(q)

        theta = np.pi / 4
        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(qc)
        # Manual: Bell state then RY(pi/4) on qubit 1
        # In little-endian convention (q0=LSB), kron(Gate_q1, I_q0) applies
        # the gate to qubit 1 (bit 1, MSB position in the kron product).
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        ry_mat = GATE_SPECS["RY"].matrix_fn(theta)
        combined = tensor_product(ry_mat, identity(2))
        expected = combined @ bell
        assert statevectors_equal(sv, expected)

    def test_multiple_cx_sequence(self):
        """Multiple CX gates in sequence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # CX twice cancels: X|0> on q0 = |10>
        expected = computational_basis_state(2, 1)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 6. Control Flow – Range Loops
# ============================================================================


class TestControlFlowRange:
    """Test range-based for loops (unrolled for CUDA-Q)."""

    def test_h_on_all_qubits(self):
        """Apply H to all n qubits using qmc.range(n)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3})
        sv = _run_statevector(qc)
        # H on all 3 qubits: equal superposition
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        assert statevectors_equal(sv, expected)

    def test_range_with_parametric_angles(self):
        """Range loop with vector parameter angles."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.measure(q)

        angles = [0.1, 0.5, 1.0]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "thetas": angles})
        sv = _run_statevector(qc)
        # Compute expected: tensor product of individual RY states
        # In little-endian convention (q0=LSB), each new qubit state goes
        # to higher bits: kron(new_qubit_state, accumulated_lower_qubits).
        states = []
        for angle in angles:
            states.append(
                compute_expected_statevector(
                    all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(angle)
                )
            )
        expected = states[0]
        for s in states[1:]:
            expected = np.kron(s, expected)
        assert statevectors_equal(sv, expected)

    def test_cx_chain_via_range(self):
        """CX chain using qmc.range(n-1)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 4})
        sv = _run_statevector(qc)
        # 4-qubit GHZ state
        expected = np.zeros(16, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[15] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 7. Control Flow – Items Loops
# ============================================================================


class TestControlFlowItems:
    """Test items-based for loops (Ising RZZ pattern)."""

    def test_ising_rzz_layer(self):
        """Ising cost layer: RZZ(gamma * J) for each edge."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            J: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(J):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        J = {(0, 1): 1.0, (1, 2): -0.5}
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "J": J, "gamma": 0.3}
        )
        assert qc.num_qubits == 3
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-6)
        # Compute expected: H^3 |000>, then RZZ(0.3*1.0) on (0,1), RZZ(0.3*-0.5) on (1,2)
        h_all = np.ones(8, dtype=complex) / np.sqrt(8)
        # RZZ(theta) = diag(e^{-i*theta/2}, e^{i*theta/2}, e^{i*theta/2}, e^{-i*theta/2})
        # Applied on qubits (i,j) in a 3-qubit system
        rzz_01 = np.eye(8, dtype=complex)
        theta1 = 0.3 * 1.0
        for k in range(8):
            b0, b1 = (k >> 0) & 1, (k >> 1) & 1
            parity = b0 ^ b1
            rzz_01[k, k] = (
                np.exp(-1j * theta1 / 2) if parity == 0 else np.exp(1j * theta1 / 2)
            )
        rzz_12 = np.eye(8, dtype=complex)
        theta2 = 0.3 * (-0.5)
        for k in range(8):
            b1, b2 = (k >> 1) & 1, (k >> 2) & 1
            parity = b1 ^ b2
            rzz_12[k, k] = (
                np.exp(-1j * theta2 / 2) if parity == 0 else np.exp(1j * theta2 / 2)
            )
        expected = rzz_12 @ rzz_01 @ h_all
        assert statevectors_equal(sv, expected)

    def test_empty_items(self):
        """Empty dict produces no gates from items loop."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            J: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(J):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "J": {}, "gamma": 0.3}
        )
        sv = _run_statevector(qc)
        # Just H on each qubit
        expected = np.ones(4, dtype=complex) / 2.0
        assert statevectors_equal(sv, expected)


# ============================================================================
# 8. Measurement & Execution
# ============================================================================


class TestMeasurement:
    """Test measurement operations and sampling."""

    def test_single_qubit_measure(self):
        """Single qubit measurement returns consistent results."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        # X|0> always measures |1>
        total = sum(count for _, count in result.results)
        assert total == 100
        for value, _ in result.results:
            assert value == 1

    def test_vector_measure(self):
        """Vector qubit measurement returns tuple with correct values."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        total = sum(count for _, count in result.results)
        assert total == 100
        for value, count in result.results:
            assert isinstance(value, tuple)
            assert len(value) == 3
            # X on q[0] and q[2]: expect (1, 0, 1)
            assert value == (1, 0, 1)

    def test_bell_state_sampling(self):
        """Bell state should only produce (0,0) and (1,1)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=1000)
        result = job.result()
        outcomes = {val for val, _ in result.results}
        assert outcomes.issubset({(0, 0), (1, 1)})


class TestExecution:
    """Test circuit execution patterns."""

    def test_sample_with_shots(self):
        """Sample with correct total shot count."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        total_counts = sum(count for _, count in result.results)
        assert total_counts == 500

    def test_parametric_sample(self):
        """Parametric circuit execution with varying parameter values."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        executor = transpiler.executor()

        # theta=0: should measure |0>
        job = exe.sample(executor, bindings={"theta": 0.0}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 0

        # theta=pi: should measure |1>
        job = exe.sample(executor, bindings={"theta": np.pi}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 1


# ============================================================================
# 9. Expectation Value Tests
# ============================================================================


class TestExpvalCudaqPipeline:
    """Test qmc.expval through full CUDA-Q transpile pipeline."""

    def test_expval_transpiles_with_compiled_expval(self):
        """Transpilation with Observable produces compiled_expval segment."""

        @qmc.qkernel
        def vqe(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        H = qm_o.Z(0) * qm_o.Z(1)
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(vqe, bindings={"H": H, "n": 2})
        assert len(exe.compiled_expval) == 1
        assert isinstance(exe.compiled_expval[0].hamiltonian, qm_o.Hamiltonian)

    def test_expval_z_on_zero_state(self):
        """<0|Z|0> = +1.0."""

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            return qmc.expval((q,), H)

        H_label = qm_o.Hamiltonian(num_qubits=1)
        H_label += qm_o.Z(0)
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.1)

    def test_expval_z_on_one_state(self):
        """<1|Z|1> = -1.0."""

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.expval((q,), H)

        H_label = qm_o.Hamiltonian(num_qubits=1)
        H_label += qm_o.Z(0)
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, -1.0, atol=0.1)

    def test_expval_x_on_plus_state(self):
        """<+|X|+> = +1.0."""

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.expval((q,), H)

        H_label = qm_o.Hamiltonian(num_qubits=1)
        H_label += qm_o.X(0)
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.1)

    def test_expval_zz_bell_state(self):
        """<Phi+|ZZ|Phi+> = +1.0."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        H_label = qm_o.Z(0) * qm_o.Z(1)
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label, "n": 2})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.1)

    def test_expval_complex_hamiltonian(self):
        """Multi-term Hamiltonian on Bell state: ZZ + 0.5*X0 + 0.5*X1."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        H_label = qm_o.Z(0) * qm_o.Z(1) + 0.5 * qm_o.X(0) + 0.5 * qm_o.X(1)
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label, "n": 2})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.15)

    def test_expval_missing_observable_raises(self):
        """Transpilation without Observable binding raises RuntimeError."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            return qmc.expval(q, H)

        transpiler = CudaqTranspiler()
        with pytest.raises(RuntimeError, match="Observable.*not found in bindings"):
            transpiler.transpile(circuit, bindings={"n": 2})


# ============================================================================
# 10. Sub-Kernel Inlining
# ============================================================================


class TestSubKernelInlining:
    """Test sub-kernel calls (function composition)."""

    def test_single_sub_kernel(self):
        """Sub-kernel call is inlined correctly."""

        @qmc.qkernel
        def sub(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def main() -> qmc.Bit:
            q = qmc.qubit("q")
            q = sub(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main)
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_chained_sub_kernels(self):
        """Chained sub-kernel calls produce correct statevector."""

        @qmc.qkernel
        def apply_h(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def apply_x(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        @qmc.qkernel
        def main() -> qmc.Bit:
            q = qmc.qubit("q")
            q = apply_x(q)
            q = apply_h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main)
        sv = _run_statevector(qc)
        # X|0> = |1>, H|1> = (|0> - |1>)/sqrt(2)
        expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 11. Stdlib QFT Tests
# ============================================================================


class TestStdlibQFT:
    """Test QFT/IQFT composite gates."""

    def test_qft_on_zero_state(self):
        """QFT|000> = uniform superposition (equal amplitudes)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3})
        assert qc.num_qubits == 3
        sv = _run_statevector(qc)
        # QFT|000> = uniform superposition with equal magnitudes
        expected_magnitude = 1.0 / np.sqrt(8)
        for amp in sv:
            assert np.isclose(abs(amp), expected_magnitude, atol=1e-6)

    def test_qft_then_iqft_identity(self):
        """QFT followed by IQFT is identity on |0...0>."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = qmc.qft(q)
            q = qmc.iqft(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3})
        sv = _run_statevector(qc)
        expected = all_zeros_state(3)
        assert statevectors_equal(sv, expected)

    def test_qft_then_iqft_on_state(self):
        """QFT followed by IQFT preserves |101>."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            q = qmc.qft(q)
            q = qmc.iqft(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3})
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 5)  # |101>
        assert statevectors_equal(sv, expected)

    def test_qft_single_qubit(self):
        """QFT on 1 qubit reduces to H gate: QFT|0> = |+>."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 1})
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 12. Bell States
# ============================================================================


class TestAllFourBellStates:
    """Test all four Bell states."""

    def test_phi_plus(self):
        """Phi+ = (|00> + |11>) / sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_phi_minus(self):
        """Phi- = (|00> - |11>) / sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_psi_plus(self):
        """Psi+ = (|01> + |10>) / sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[1] = qmc.x(q[1])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_psi_minus(self):
        """Psi- = (|01> - |10>) / sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1] = qmc.x(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([0, -1, 1, 0], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 13. GHZ State (Parametrised)
# ============================================================================


class TestGHZStateParametrised:
    """Test GHZ state construction with loop."""

    def test_ghz_4q(self):
        """4-qubit GHZ state: (|0000> + |1111>) / sqrt(2)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 4})
        sv = _run_statevector(qc)
        expected = np.zeros(16, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[15] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_ghz_5q(self):
        """5-qubit GHZ state: (|00000> + |11111>) / sqrt(2)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 5})
        sv = _run_statevector(qc)
        expected = np.zeros(32, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[31] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 14. Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test boundary and special cases."""

    def test_measure_only_circuit(self):
        """Circuit with no gates, just measurement."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 0

    def test_multi_qubit_no_entanglement(self):
        """Independent qubits: X on q0 and q2, measure all."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 5)  # |101>
        assert statevectors_equal(sv, expected)

    def test_zero_angle_rotation(self):
        """RX(0) is identity."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": 0.0})
        sv = _run_statevector(qc)
        expected = all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_transpile_empty_barrier(self):
        """Circuit with barrier transpiles correctly."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # H*H = I
        expected = all_zeros_state(1)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 15. Error Cases
# ============================================================================
# Note: c_if transpile-ok, if-with-else EmitError, and while-loop EmitError
# tests live in test_cudaq.py::TestCudaqControlFlowErrors to avoid duplication.


class TestParametricErrors:
    """Test parametric circuit error conditions."""

    def test_missing_parameter_binding_raises(self) -> None:
        """Missing parameter binding in bind_parameters raises ValueError."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        executor = transpiler.executor()
        with pytest.raises(ValueError, match="Missing binding"):
            executor.bind_parameters(qc, {}, exe.compiled_quantum[0].parameter_metadata)


# ============================================================================
# 16. Qubit Array Patterns
# ============================================================================


class TestQubitArrayPatterns:
    """Test general qubit array manipulation patterns."""

    def test_cx_propagation_array(self):
        """CX propagation: |10000> -> |11111>."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.x(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 5})
        sv = _run_statevector(qc)
        expected = computational_basis_state(5, 31)  # |11111>
        assert statevectors_equal(sv, expected)

    def test_independent_rotations(self):
        """Independent RY rotations on each qubit."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, angles: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], angles[i])
            return qmc.measure(q)

        angles = [np.pi / 6, np.pi / 3, np.pi / 2]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "angles": angles})
        sv = _run_statevector(qc)
        # Verify norm
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-6)
        # Verify independent: tensor product of individual states
        # In little-endian convention (q0=LSB), each new qubit state goes
        # to higher bits: kron(new_qubit_state, accumulated_lower_qubits).
        states = []
        for angle in angles:
            states.append(
                compute_expected_statevector(
                    all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(angle)
                )
            )
        expected = states[0]
        for s in states[1:]:
            expected = np.kron(s, expected)
        assert statevectors_equal(sv, expected)

    def test_entangling_layer(self):
        """CZ entangling layer on neighboring pairs."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 4})
        assert qc.num_qubits == 4
        sv = _run_statevector(qc)
        # H^4|0000> then CZ(0,1), CZ(1,2), CZ(2,3)
        state = np.ones(16, dtype=complex) / 4.0
        for k in range(16):
            if (k >> 0) & 1 and (k >> 1) & 1:  # CZ(0,1)
                state[k] *= -1
        for k in range(16):
            if (k >> 1) & 1 and (k >> 2) & 1:  # CZ(1,2)
                state[k] *= -1
        for k in range(16):
            if (k >> 2) & 1 and (k >> 3) & 1:  # CZ(2,3)
                state[k] *= -1
        assert statevectors_equal(sv, state)

    def test_swap_network(self):
        """Swap network reverses qubit order."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0] = qmc.x(q[0])  # |1000>
            # Reverse: swap (0,3) then (1,2)
            q[0], q[3] = qmc.swap(q[0], q[3])
            q[1], q[2] = qmc.swap(q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = computational_basis_state(4, 8)  # |0001> reversed = |1000> = idx 8
        assert statevectors_equal(sv, expected)


# ============================================================================
# 17. Phase Kickback Pattern
# ============================================================================


class TestPhaseKickbackPattern:
    """Test phase kickback phenomenon."""

    def test_phase_kickback_p_gate(self):
        """P(pi) eigenstate |1> kicks back phase pi to control.

        Circuit: H(q0) X(q1) CP(pi, q0, q1) H(q0)
        With theta=pi, control picks up phase pi, H converts to |1>.
        Result: |11> = computational_basis_state(2, 3).
        """

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])  # Control in superposition
            q[1] = qmc.x(q[1])  # Target in eigenstate |1>
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            q[0] = qmc.h(q[0])  # Interfere to detect phase
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})
        assert qc.num_qubits == 2
        sv = _run_statevector(qc)
        # CP(pi)|11> adds phase e^{i*pi}=-1 to |11> component.
        # H converts: (|0> - |1>)/sqrt(2) -> |1>
        # So result is |11>
        expected = computational_basis_state(2, 3)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 18. Algorithm Patterns – QAOA-like
# ============================================================================


class TestQAOAPattern:
    """Test QAOA-style alternating entangling + mixer pattern."""

    def test_single_qaoa_layer(self):
        """Single QAOA layer: H + CZ entangling + RY mixer.

        Verifies QAOA produces same state as manual matrix computation.
        """

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, beta: qmc.Float, gamma: qmc.Float
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            # Initial superposition
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            # Cost layer (CZ entangling)
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            # Mixer layer (RY rotations)
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], beta)
            return qmc.measure(q)

        beta, n = 0.5, 3
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n, "beta": beta, "gamma": 0.3}
        )
        assert qc.num_qubits == 3
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-6)
        # Compute expected: H^3|000>, CZ(0,1), CZ(1,2), RY(beta)^3
        state = np.ones(8, dtype=complex) / np.sqrt(8)
        # CZ(i,j): flip sign when both qubits are |1>
        for k in range(8):
            if (k >> 0) & 1 and (k >> 1) & 1:  # CZ(0,1)
                state[k] *= -1
        for k in range(8):
            if (k >> 1) & 1 and (k >> 2) & 1:  # CZ(1,2)
                state[k] *= -1
        # RY(beta) on each qubit
        ry = GATE_SPECS["RY"].matrix_fn(beta)
        ry3 = np.kron(np.kron(ry, ry), ry)
        expected = ry3 @ state
        assert statevectors_equal(sv, expected)

    def test_qaoa_parametric(self):
        """QAOA with parametric beta, gamma: metadata + bind-and-verify."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            beta: qmc.Float,
            gamma: qmc.Float,
            J: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            # Cost
            for (i, j), Jij in qmc.items(J):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            # Mixer
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={"n": 3, "J": {(0, 1): 1.0, (1, 2): -0.5}},
            parameters=["beta", "gamma"],
        )
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 3
        assert exe.has_parameters
        param_names = exe.parameter_names
        assert len(param_names) == 2
        assert any("beta" in p for p in param_names)
        assert any("gamma" in p for p in param_names)
        # Bind concrete values and verify statevector is valid
        executor = transpiler.executor()
        bound = executor.bind_parameters(
            qc, {"beta": 0.5, "gamma": 0.3}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel, bound.param_values))
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-6)
        # Verify it's not all zeros (actually computed something)
        assert not np.allclose(sv, all_zeros_state(3), atol=1e-6)


# ============================================================================
# Module-level qkernel definitions for c_if tests
# ============================================================================


@qmc.qkernel
def _c_if_basic() -> qmc.Bit:
    """X(q0) → measure → c_if(bit, X(q1)) → measure(q1)."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q0 = qmc.x(q0)
    b = qmc.measure(q0)
    if b:
        q1 = qmc.x(q1)
    return qmc.measure(q1)


@qmc.qkernel
def _c_if_false_branch() -> qmc.Bit:
    """No X on q0 → measure(q0)=0 → c_if should NOT fire."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    b = qmc.measure(q0)
    if b:
        q1 = qmc.x(q1)
    return qmc.measure(q1)


@qmc.qkernel
def _c_if_with_rotation(theta: qmc.Float) -> qmc.Bit:
    """c_if with parametric RX gate in true branch."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q0 = qmc.x(q0)
    b = qmc.measure(q0)
    if b:
        q1 = qmc.rx(q1, theta)
    return qmc.measure(q1)


@qmc.qkernel
def _c_if_with_else() -> qmc.Bit:
    """c_if with else branch — should raise EmitError."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q0 = qmc.h(q0)
    b = qmc.measure(q0)
    if b:
        q1 = qmc.x(q1)
    else:
        q1 = qmc.h(q1)
    return qmc.measure(q1)


# ============================================================================
# c_if Control Flow Tests
# ============================================================================


class TestCIfControlFlow:
    """Test that measurement-dependent conditional branching is rejected under CUDA-Q 0.14.x."""

    def test_c_if_basic_raises_emit_error(self) -> None:
        """measurement-dependent c_if (if-then, no else) raises EmitError under 0.14.x."""
        with pytest.raises(EmitError, match="measurement-dependent"):
            _transpile_and_get_circuit(_c_if_basic)

    def test_c_if_false_branch_raises_emit_error(self) -> None:
        """measurement-dependent c_if (false-branch path) raises EmitError under 0.14.x."""
        with pytest.raises(EmitError, match="measurement-dependent"):
            _transpile_and_get_circuit(_c_if_false_branch)

    def test_c_if_with_rotation_raises_emit_error(self) -> None:
        """measurement-dependent c_if with parametric body raises EmitError under 0.14.x."""
        with pytest.raises(EmitError, match="measurement-dependent"):
            _transpile_and_get_circuit(_c_if_with_rotation, bindings={"theta": np.pi})

    def test_c_if_with_else_raises_emit_error(self) -> None:
        """Measurement-dependent if-else must raise EmitError on CUDA-Q 0.14.x."""
        with pytest.raises(EmitError, match="measurement-dependent"):
            _transpile_and_get_circuit(_c_if_with_else)

    def test_measurement_without_conditional_transpiles_ok(self) -> None:
        """Mid-circuit measurement without conditional branching must not be rejected."""

        @qmc.qkernel
        def measure_only() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.h(q0)
            return qmc.measure(q0)

        _, qc = _transpile_and_get_circuit(measure_only)
        assert qc.num_qubits == 1

    def test_while_loop_still_raises_emit_error(self) -> None:
        """While loops remain unsupported on CUDA-Q."""

        @qmc.qkernel
        def circuit_with_while() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        with pytest.raises(EmitError, match="while loop control flow"):
            _transpile_and_get_circuit(circuit_with_while)


# ============================================================================
# Sampling Ordering Regression Tests
# ============================================================================


class TestSamplingOrdering:
    """Verify that CUDA-Q executor returns canonical big-endian bitstrings.

    CUDA-Q raw ``__global__`` uses allocation order (first declared qubit =
    leftmost), while the executor contract requires big-endian (highest
    qubit index = leftmost).  These tests ensure the normalization is correct.
    """

    def test_non_symmetric_full_register_sample(self):
        """X on q[0] only — sample must decode logical q0=1."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == (1, 0, 0), f"Expected (1, 0, 0), got {value}"

    def test_reverse_order_selective_measurement_sample(self):
        """Measure q[2] then q[0] with X on q[0] — sample must return (0, 1)."""

        @qmc.qkernel
        def measure_reverse_order() -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.x(qs[0])
            return qmc.measure(qs[2]), qmc.measure(qs[0])

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(measure_reverse_order)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == (0, 1), f"Expected (0, 1), got {value}"

    def test_reverse_order_selective_measurement_run(self):
        """run() with reverse-order selective measurement must return (0, 1)."""

        @qmc.qkernel
        def measure_reverse_order() -> tuple[qmc.Bit, qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.x(qs[0])
            return qmc.measure(qs[2]), qmc.measure(qs[0])

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(measure_reverse_order)
        executor = transpiler.executor()

        result = exe.run(executor).result()
        assert result == (0, 1), f"Expected (0, 1), got {result}"


# ============================================================================
# Bound constant if-condition tests (Issue: bound_constant_if_misclassified)
# ============================================================================


class TestBoundConstantIfConditionCudaq:
    """Test that bound parameters in if-conditions are correctly resolved
    as compile-time constants on CUDA-Q (no measurement-dependent error)."""

    def test_bound_flag_true_no_error(self):
        """bindings={"flag": 1} should not raise measurement-dependent error."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1})
        assert qc.num_qubits == 1

    def test_bound_flag_true_statevector(self):
        """bindings={"flag": 1} if-X should produce |1> statevector."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1})
        sv = _run_statevector(qc)
        expected = computational_basis_state(1, 1)  # |1>
        assert statevectors_equal(sv, expected)

    def test_bound_flag_false_statevector(self):
        """bindings={"flag": 0} if-X should produce |0> statevector."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 0})
        sv = _run_statevector(qc)
        expected = computational_basis_state(1, 0)  # |0>
        assert statevectors_equal(sv, expected)

    def test_runtime_measurement_if_still_rejected(self):
        """Runtime measurement-dependent if still raises EmitError."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        with pytest.raises(EmitError, match="measurement-dependent"):
            _transpile_and_get_circuit(circuit)


# ============================================================================
# Controlled helper kernel tests (Issue: cudaq_controlled_helper_gate_conversion)
# ============================================================================


class TestControlledHelperCudaq:
    """Test qmc.controlled() helper kernels on CUDA-Q."""

    def test_controlled_x_double_control_statevector(self):
        """qmc.controlled(x_gate, num_controls=2) should act as Toffoli."""

        @qmc.qkernel
        def x_gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        ccx = qmc.controlled(x_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |110>: controls ON, target OFF
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |110> → Toffoli → |111>
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected)

    def test_controlled_x_double_control_off(self):
        """CCX with one control off should not flip target."""

        @qmc.qkernel
        def x_gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        ccx = qmc.controlled(x_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |100>: only first control ON
            q[0] = qmc.x(q[0])
            q[0], q[1], q[2] = ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |100> → Toffoli → |100> (no flip)
        expected = computational_basis_state(3, 0b001)
        assert statevectors_equal(sv, expected)

    def test_controlled_swap_statevector(self):
        """Controlled-SWAP (Fredkin) should swap targets when control is ON."""

        @qmc.qkernel
        def swap_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.swap(q0, q1)

        cswap = qmc.controlled(swap_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |011> (q0=1, q1=1, q2=0 in big-endian)
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = cswap(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |011> with ctrl=q0=1 → swap q1,q2 → |101>
        expected = computational_basis_state(3, 0b101)
        assert statevectors_equal(sv, expected)

    def test_controlled_swap_control_off(self):
        """Controlled-SWAP with control OFF should not swap."""

        @qmc.qkernel
        def swap_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.swap(q0, q1)

        cswap = qmc.controlled(swap_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |010> (q0=0, q1=1, q2=0)
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = cswap(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |010> with ctrl=q0=0 → no swap → |010>
        expected = computational_basis_state(3, 0b010)
        assert statevectors_equal(sv, expected)


# ============================================================================
# Compile-time constant if with array quantum phi output
# ============================================================================


class TestCompileTimeIfArrayQuantumPhi:
    """Compile-time constant if with array quantum phi must not raise EmitError."""

    def test_dead_branch_different_array(self):
        """Dead branch rebinds qubit array to a different array."""
        flag = True

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            if flag:
                q[0] = qmc.x(q[0])
            else:
                alt = qmc.qubit_array(2, "alt")
                alt[1] = qmc.x(alt[1])
                q = alt
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc is not None
