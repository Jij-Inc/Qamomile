"""Rich CUDA-Q frontend-to-backend test suite.

Tests the full pipeline: @qkernel definition -> CudaqTranspiler -> execution.
Covers every frontend gate, gate combinations, control flow (loops, if, while),
stdlib, measurement, parametric circuits, expval, and edge cases.

Runtime measurement-dependent control flow (``if bit:``, ``if/else``,
``while bit:``) is supported via ``@cudaq.kernel`` source code generation
and ``cudaq.run()``.  Static circuits use the builder API and
``cudaq.sample()``.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.cudaq

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import (
    cz_entangling_layer,
    rx_layer,
    ry_layer,
    rz_layer,
    superposition_vector,
)
from qamomile.circuit.algorithm.qaoa import (
    ising_cost,
    qaoa_layers,
    qaoa_state,
    x_mixer,
)
from qamomile.circuit.transpiler.errors import DependencyError, EmitError
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

from qamomile.cudaq.emitter import CudaqKernelArtifact, ExecutionMode  # noqa: E402
from tests.transpiler.backends._cudaq_source_assertions import (  # noqa: E402
    ValidatingCudaqTranspiler as CudaqTranspiler,
    assert_inspect_source_matches_artifact,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_statevector(circuit: CudaqKernelArtifact) -> np.ndarray:
    """Run a CudaqKernelArtifact and return the statevector.

    Args:
        circuit: CudaqKernelArtifact with kernel_func.

    Returns:
        Numpy array of complex amplitudes.
    """
    import cudaq

    state = cudaq.get_state(circuit.kernel_func)
    return np.array(state)


def _transpile_and_get_circuit(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
    *,
    smoke_test: bool = False,
    smoke_bindings: dict[str, Any] | None = None,
) -> tuple[Any, CudaqKernelArtifact]:
    """Transpile a qkernel and return the executable and CUDA-Q circuit.

    Args:
        kernel: A ``@qmc.qkernel`` decorated function.
        bindings: Optional parameter bindings dict.
        parameters: Optional list of parameter names to leave symbolic.

    Returns:
        Tuple of (ExecutableProgram, CudaqKernelArtifact).
    """
    transpiler = CudaqTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    circuit = exe.compiled_quantum[0].circuit
    assert circuit.source
    assert_inspect_source_matches_artifact(circuit)
    if smoke_test:
        executor = transpiler.executor()
        bound_bindings = smoke_bindings or {}
        if exe.compiled_expval:
            exe.run(executor, bindings=bound_bindings).result()
        else:
            exe.sample(executor, bindings=bound_bindings, shots=1).result()
    return exe, circuit


def _assert_source_contains(circuit: CudaqKernelArtifact, *fragments: str) -> None:
    """Assert that source contains the given fragments in order."""
    cursor = 0
    for fragment in fragments:
        idx = circuit.source.find(fragment, cursor)
        assert idx >= 0, f"Expected source fragment not found: {fragment}\n{circuit.source}"
        cursor = idx + len(fragment)


def _assert_compiled_source(exe: Any, *fragments: str) -> CudaqKernelArtifact:
    """Assert source invariants for the first compiled CUDA-Q artifact."""
    circuit = exe.compiled_quantum[0].circuit
    assert circuit.source
    assert_inspect_source_matches_artifact(circuit)
    if fragments:
        _assert_source_contains(circuit, *fragments)
    return circuit


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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
# 15b. Dead-branch parameter elimination regression
# ============================================================================


class TestDeadBranchParameterElimination:
    """Regression tests: parameters in compile-time dead branches produce
    parameterless CUDA-Q kernels.

    When ``parameters=["theta"]`` is specified but ``theta`` only appears
    inside a branch that is eliminated at compile time, the generated kernel
    must NOT require a ``thetas`` argument.
    """

    def test_dead_branch_closure_constant(self):
        """Closure constant False eliminates branch → parameterless kernel."""
        use_rotation = False

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            if use_rotation:
                q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit

        # No surviving parameters
        assert not exe.has_parameters
        assert exe.parameter_names == []

        # Kernel signature must be parameterless
        assert "thetas" not in qc.source

        # Execution must succeed without binding
        result = cudaq.sample(qc.kernel_func, shots_count=10)
        assert len(result) > 0

    def test_dead_branch_binding_eliminates_param(self):
        """Binding flag=0 eliminates branch → parameterless kernel."""

        @qmc.qkernel
        def circuit(flag: qmc.Float, theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"flag": 0}, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit

        # No surviving parameters
        assert not exe.has_parameters
        assert exe.parameter_names == []

        # Kernel signature must be parameterless
        assert "thetas" not in qc.source

        # Execution must succeed without binding
        result = cudaq.sample(qc.kernel_func, shots_count=10)
        assert len(result) > 0


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
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
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
    """c_if with else branch."""
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
    """Test runtime measurement-dependent control flow on CUDA-Q via cudaq.run()."""

    def test_c_if_basic_transpiles(self) -> None:
        """Runtime if-then transpiles to RUNNABLE mode."""
        exe, circuit = _transpile_and_get_circuit(_c_if_basic)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE
        assert circuit.kernel_func is not None

    def test_c_if_false_branch_transpiles(self) -> None:
        """Runtime if-then (false path) transpiles to RUNNABLE mode."""
        exe, circuit = _transpile_and_get_circuit(_c_if_false_branch)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE

    def test_c_if_with_rotation_transpiles(self) -> None:
        """Runtime if-then with parametric body transpiles successfully."""
        exe, circuit = _transpile_and_get_circuit(
            _c_if_with_rotation, bindings={"theta": np.pi}
        )
        assert circuit.execution_mode == ExecutionMode.RUNNABLE

    def test_c_if_with_else_transpiles(self) -> None:
        """Runtime if-else transpiles to RUNNABLE mode."""
        exe, circuit = _transpile_and_get_circuit(_c_if_with_else)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE

    def test_measurement_without_conditional_transpiles_ok(self) -> None:
        """Mid-circuit measurement without conditional stays on STATIC mode."""

        @qmc.qkernel
        def measure_only() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.h(q0)
            return qmc.measure(q0)

        _, qc = _transpile_and_get_circuit(measure_only)
        assert qc.execution_mode == ExecutionMode.STATIC
        assert qc.num_qubits == 1

    def test_while_loop_transpiles(self) -> None:
        """While loops transpile to RUNNABLE mode."""

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

        exe, circuit = _transpile_and_get_circuit(circuit_with_while)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE

    def test_c_if_basic_sample(self) -> None:
        """X(q0) → measure → if bit: X(q1). q0=1 always, so q1=1 always."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_basic)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 1, f"Expected bit=1, got {value}"

    def test_c_if_basic_run(self) -> None:
        """run() returns single shot: bit should be 1."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_basic)
        executor = transpiler.executor()
        result = exe.run(executor).result()
        assert result == 1, f"Expected 1, got {result}"

    def test_c_if_false_branch_sample(self) -> None:
        """No X on q0 → q0=0 → if-branch not taken → q1=0 always."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_false_branch)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 0, f"Expected bit=0, got {value}"

    def test_c_if_with_rotation_sample(self) -> None:
        """Bound theta=pi: RX(pi) ≈ X, so q1=1 always when if-branch taken."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_with_rotation, bindings={"theta": np.pi})
        executor = transpiler.executor()
        job = exe.sample(executor, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 1, f"Expected bit=1, got {value}"

    def test_c_if_with_else_sample(self) -> None:
        """H(q0) → measure → if: X(q1) else: H(q1). Both branches should occur."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_with_else)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=200)
        result = job.result()
        values = {v for v, _ in result.results}
        # With H on q0, both branches execute; both 0 and 1 should appear in results
        assert 0 in values and 1 in values, f"Expected both 0 and 1, got {values}"

    def test_while_loop_repeat_until_zero_sample(self) -> None:
        """Repeat-until-zero: H → measure → while bit → H → measure. Always returns 0."""

        @qmc.qkernel
        def repeat_until_zero() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(repeat_until_zero)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=200)
        result = job.result()
        for value, _ in result.results:
            assert value == 0, f"repeat_until_zero must always return 0, got {value}"

    def test_nested_if_inside_while_sample(self) -> None:
        """while loop with if-else inside: deterministic exit via if-branch measurement."""

        @qmc.qkernel
        def nested_if_while() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2")
                    bit = qmc.measure(q2)
                else:
                    q3 = qmc.qubit("q3")
                    bit = qmc.measure(q3)
            return bit

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(nested_if_while)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 0, f"Expected 0 (loop exit), got {value}"

    def test_c_if_with_rotation_late_binding_sample(self) -> None:
        """Parametric if-only with late binding: parameters=['theta'] + run-time bindings."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_with_rotation, parameters=["theta"])
        executor = transpiler.executor()
        job = exe.sample(executor, shots=100, bindings={"theta": np.pi})
        result = job.result()
        for value, _ in result.results:
            assert value == 1, f"Expected bit=1 with RX(pi), got {value}"

    def test_c_if_with_rotation_late_binding_run(self) -> None:
        """Parametric if-only with late binding via run()."""
        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(_c_if_with_rotation, parameters=["theta"])
        executor = transpiler.executor()
        result = exe.run(executor, bindings={"theta": np.pi}).result()
        assert result == 1, f"Expected 1, got {result}"


# ============================================================================
# Logical Clbit Return Contract Tests
# ============================================================================


class TestLogicalClbitReturnContract:
    """Verify RUNNABLE kernels return logical clbit values, not mz(q)."""

    def test_runnable_source_returns_clbit_aggregate(self) -> None:
        """RUNNABLE source must contain 'return [__b' and not 'return mz(q)'."""
        _, circuit = _transpile_and_get_circuit(_c_if_basic)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE
        assert "return mz(q)" not in circuit.source
        assert "return [__b" in circuit.source

    def test_runnable_source_initializes_clbits(self) -> None:
        """RUNNABLE source must initialize __b{i} = False for all clbits."""
        _, circuit = _transpile_and_get_circuit(_c_if_basic)
        for i in range(circuit.num_clbits):
            assert f"__b{i} = False" in circuit.source

    def test_runnable_measurement_qubit_map_empty(self) -> None:
        """RUNNABLE segments must not populate measurement_qubit_map."""
        exe, circuit = _transpile_and_get_circuit(_c_if_basic)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE
        meas_map = exe.compiled_quantum[0].measurement_qubit_map
        assert meas_map == {}

    def test_if_else_different_qubit_sources(self) -> None:
        """Same logical bit from different qubits in if/else branches."""

        @qmc.qkernel
        def branch_sensitive() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            sel = qmc.measure(q0)
            q1 = qmc.x(q1)
            if sel:
                bit = qmc.measure(q1)
            else:
                bit = qmc.measure(q2)
            return bit

        _, circuit = _transpile_and_get_circuit(branch_sensitive)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE
        assert "return mz(q)" not in circuit.source
        assert "return [__b" in circuit.source
        # measurement_qubit_map must be empty
        exe, _ = _transpile_and_get_circuit(branch_sensitive)
        assert exe.compiled_quantum[0].measurement_qubit_map == {}

    def test_while_loop_clbit_return(self) -> None:
        """While loop kernel uses logical clbit return."""

        @qmc.qkernel
        def repeat_until_zero() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q2 = qmc.qubit("q2")
                q2 = qmc.h(q2)
                bit = qmc.measure(q2)
            return bit

        _, circuit = _transpile_and_get_circuit(repeat_until_zero)
        assert circuit.execution_mode == ExecutionMode.RUNNABLE
        assert "return mz(q)" not in circuit.source
        assert "return [__b" in circuit.source


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

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1}, smoke_test=True)
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

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 0}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = computational_basis_state(1, 0)  # |0>
        assert statevectors_equal(sv, expected)

    def test_runtime_measurement_if_uses_runnable_mode(self):
        """Runtime measurement-dependent if produces RUNNABLE-mode artifact."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        exe, qc = _transpile_and_get_circuit(circuit)
        assert qc.execution_mode == ExecutionMode.RUNNABLE

    def test_bound_constant_if_uses_static_mode(self):
        """Compile-time if still produces STATIC-mode artifact."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1})
        assert qc.execution_mode == ExecutionMode.STATIC


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


# ============================================================================
# CUDA-Q helper kernel semantics contract
# (Issue: cudaq_backend_semantics_contract_issue)
# ============================================================================


class TestCudaqHelperKernelSemanticsContract:
    """Test CUDA-Q helper kernel emit with correct operand-to-target mapping.

    Verifies that:
    - Single-control helper touching second target acts on the correct qubit.
    - Multi-control helper touching second target acts on the correct qubit.
    - Helper body with unsupported ops raises EmitError (not silent skip).
    - Existing CCX/CSWAP happy paths remain functional.
    """

    def test_single_control_second_target(self):
        """Single-control helper that flips the second target only."""

        @qmc.qkernel
        def flip_second(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            b = qmc.x(b)
            return a, b

        cx2 = qmc.controlled(flip_second)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Control ON -> should flip q[2] (second target), not q[1]
            q[0] = qmc.x(q[0])
            q[0], q[1], q[2] = cx2(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |001> with ctrl=q0=1 -> flip second target (q2) -> |101>
        expected = computational_basis_state(3, 0b101)
        assert statevectors_equal(sv, expected), (
            f"Single-control second-target helper: expected |101>, got statevector {sv}"
        )

    def test_multi_control_second_target(self):
        """Multi-control helper that flips the second target only."""

        @qmc.qkernel
        def flip_second(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            b = qmc.x(b)
            return a, b

        cc = qmc.controlled(flip_second, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            # Both controls ON -> should flip q[3] (second target)
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2], q[3] = cc(q[0], q[1], q[2], q[3])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |0011> with both controls=1 -> flip q[3] -> |1011>
        expected = computational_basis_state(4, 0b1011)
        assert statevectors_equal(sv, expected), (
            f"Multi-control second-target helper: expected |1011>, got statevector {sv}"
        )

    def test_helper_body_with_loop_multi_control_raises_emit_error(self):
        """Multi-control helper body with ForOperation should raise EmitError."""

        @qmc.qkernel
        def loop_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            for i in qmc.range(1):
                q0 = qmc.x(q0)
                q1 = qmc.x(q1)
            return q0, q1

        cc = qmc.controlled(loop_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2], q[3] = cc(q[0], q[1], q[2], q[3])
            return qmc.measure(q)

        with pytest.raises(EmitError, match="Unsupported operation"):
            _transpile_and_get_circuit(circuit)

    def test_existing_ccx_happy_path_regression(self):
        """Existing CCX (Toffoli) happy path must not regress."""

        @qmc.qkernel
        def x_gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        ccx = qmc.controlled(x_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected)

    def test_existing_cswap_happy_path_regression(self):
        """Existing CSWAP (Fredkin) happy path must not regress."""

        @qmc.qkernel
        def swap_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.swap(q0, q1)

        cswap = qmc.controlled(swap_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = cswap(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 0b101)
        assert statevectors_equal(sv, expected)

    def test_single_control_helper_with_loop_succeeds(self):
        """Single-control helper with compile-time ForOperation should succeed.

        Regression: the CUDA-Q override previously rejected ForOperation
        in all helper bodies, but single-control cases should delegate to
        the shared fallback which supports compile-time loop unrolling.
        """

        @qmc.qkernel
        def loop_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            for i in qmc.range(1):
                q0 = qmc.x(q0)
                q1 = qmc.x(q1)
            return q0, q1

        c1 = qmc.controlled(loop_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1], q[2] = c1(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |001> ctrl=q0=1 -> loop body flips q1 and q2 -> |111>
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected), (
            f"Single-control loop helper: expected |111>, got statevector {sv}"
        )

    def test_single_control_helper_with_loop_index_spec(self):
        """Single-control loop helper via index-spec route should succeed.

        Covers the _emit_controlled_u_with_index_spec entry path to ensure
        the same routing fix applies through both controlled-U entry paths.
        """

        @qmc.qkernel
        def loop_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            for i in qmc.range(1):
                q0 = qmc.x(q0)
                q1 = qmc.x(q1)
            return q0, q1

        c1 = qmc.controlled(loop_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            # Use controlled_indices to specify q[0] as the control
            q = c1(q, controlled_indices=[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # ctrl=q0=1 -> loop body flips q1 and q2 -> |111>
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected), (
            f"Single-control loop helper (index-spec): expected |111>, "
            f"got statevector {sv}"
        )

    def test_controlled_cnot_single_control(self):
        """Positional single-control helper with inner CX should act as Toffoli."""

        @qmc.qkernel
        def cnot_gate(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.cx(a, b)

        c1 = qmc.controlled(cnot_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |011>: outer control ON, inner control ON, target OFF
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = c1(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |011> -> Toffoli (outer ctrl=q0, inner ctrl=q1, target=q2) -> |111>
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected), (
            f"Controlled CNOT (single, positional): expected |111>, got {sv}"
        )

    def test_controlled_cnot_double_control(self):
        """Positional multi-control helper with inner CX should act as C^3X."""

        @qmc.qkernel
        def cnot_gate(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.cx(a, b)

        cccx = qmc.controlled(cnot_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            # Prepare |0111>: all 3 controls ON, target OFF
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[2] = qmc.x(q[2])
            q[0], q[1], q[2], q[3] = cccx(q[0], q[1], q[2], q[3])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |0111> -> C^3X -> |1111>
        expected = computational_basis_state(4, 0b1111)
        assert statevectors_equal(sv, expected), (
            f"Controlled CNOT (double, positional): expected |1111>, got {sv}"
        )

    def test_controlled_cnot_single_control_index_spec(self):
        """Index-spec single-control helper with inner CX should act as Toffoli."""

        @qmc.qkernel
        def cnot_gate(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.cx(a, b)

        c1 = qmc.controlled(cnot_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q = c1(q, controlled_indices=[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # ctrl=q0, inner ctrl=q1, target=q2 -> Toffoli -> |111>
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected), (
            f"Controlled CNOT (single, index-spec): expected |111>, got {sv}"
        )

    def test_controlled_cnot_double_control_index_spec(self):
        """Index-spec multi-control helper with inner CX should act as C^3X."""

        @qmc.qkernel
        def cnot_gate(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.cx(a, b)

        cccx = qmc.controlled(cnot_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[2] = qmc.x(q[2])
            q = cccx(q, controlled_indices=[0, 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # C^3X with all controls ON -> |1111>
        expected = computational_basis_state(4, 0b1111)
        assert statevectors_equal(sv, expected), (
            f"Controlled CNOT (double, index-spec): expected |1111>, got {sv}"
        )


# ============================================================================
# High-Level Frontend Coverage
# ============================================================================


class TestParameterizedCircuits:
    """Test high-level parametric circuit preservation and binding."""

    def test_parametric_rx(self):
        """parameters=['theta'] preserves symbolic metadata."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = _assert_compiled_source(
            exe,
            "def _qamomile_kernel(thetas: list[float]):",
            "q = cudaq.qvector(1)",
            "rx(thetas[0], q[0])",
        )
        result = exe.sample(
            transpiler.executor(), bindings={"theta": np.pi}, shots=32
        ).result()
        assert exe.has_parameters
        assert qc.num_qubits == 1
        assert exe.parameter_names == ["theta"]
        assert {value for value, _ in result.results} == {1}

    def test_parametric_bind_statevector(self):
        """Binding a preserved parameter produces the expected statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = _assert_compiled_source(
            exe,
            "def _qamomile_kernel(thetas: list[float]):",
            "q = cudaq.qvector(1)",
            "rx(thetas[0], q[0])",
        )
        executor = transpiler.executor()
        sample = exe.sample(executor, bindings={"theta": np.pi}, shots=16).result()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(np.pi)
        )
        assert {value for value, _ in sample.results} == {1}
        assert statevectors_equal(sv, expected)

    def test_mixed_bound_and_parametric(self):
        """Structural bindings and symbolic angles can coexist."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["theta"])
        qc = _assert_compiled_source(
            exe,
            "def _qamomile_kernel(thetas: list[float]):",
            "q = cudaq.qvector(3)",
            "rx(thetas[0], q[0])",
        )
        executor = transpiler.executor()
        sample = exe.sample(executor, bindings={"theta": np.pi}, shots=16).result()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
        expected = computational_basis_state(3, 0b111)
        assert exe.has_parameters
        assert qc.num_qubits == 3
        assert statevectors_equal(sv, expected)
        assert {value for value, _ in sample.results} == {(1, 1, 1)}

    def test_rebind_same_compiled_circuit(self):
        """A single compiled parametric circuit can be rebound safely."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = _assert_compiled_source(
            exe,
            "def _qamomile_kernel(thetas: list[float]):",
            "q = cudaq.qvector(1)",
            "ry(thetas[0], q[0])",
        )
        executor = transpiler.executor()

        zero_result = exe.sample(executor, bindings={"theta": 0.0}, shots=16).result()
        pi_result = exe.sample(executor, bindings={"theta": np.pi}, shots=16).result()
        assert {value for value, _ in zero_result.results} == {0}
        assert {value for value, _ in pi_result.results} == {1}


class TestControlFlowNested:
    """Test nested compile-time loop patterns on CUDA-Q."""

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_nested_range_loops_transpile(self, n):
        """Two-level range loops over all CZ pairs transpile cleanly."""

        @qmc.qkernel
        def circuit(num_qubits: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(num_qubits, "q")
            for i in qmc.range(num_qubits):
                q[i] = qmc.h(q[i])
            for i in qmc.range(num_qubits - 1):
                for j in qmc.range(i + 1, num_qubits):
                    q[i], q[j] = qmc.cz(q[i], q[j])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"num_qubits": n}, smoke_test=True
        )
        assert qc.execution_mode == ExecutionMode.STATIC
        assert qc.num_qubits == n
        _assert_source_contains(qc, "def _qamomile_kernel():", f"q = cudaq.qvector({n})")

    def test_cx_ladder_via_range(self):
        """A loop-based CX ladder propagates |1> to every qubit."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.x(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 4}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = computational_basis_state(4, 0b1111)
        assert statevectors_equal(sv, expected)

    def test_nested_range_sampling_ghz(self):
        """Nested control flow can still drive a standard GHZ pattern."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 4})
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(4)",
            "x.ctrl(q[0], q[1])",
        )
        result = exe.sample(transpiler.executor(), shots=256).result()
        outcomes = {value for value, _ in result.results}
        assert outcomes == {(0, 0, 0, 0), (1, 1, 1, 1)}


class TestControlFlowQAOAPattern:
    """QAOA-like range/items combinations should compile on CUDA-Q."""

    def test_qaoa_single_layer_statevector(self):
        """H init + RZZ cost + RX mixer matches the analytical state."""

        @qmc.qkernel
        def qaoa_layer(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.measure(q)

        gamma, beta = 0.5, 0.3
        ising = {(0, 1): 1.0}
        _, qc = _transpile_and_get_circuit(
            qaoa_layer,
            bindings={"n": 2, "ising": ising, "gamma": gamma, "beta": beta},
            smoke_test=True,
        )
        sv = _run_statevector(qc)
        h = GATE_SPECS["H"].matrix_fn()
        rzz = GATE_SPECS["RZZ"].matrix_fn(gamma)
        rx = GATE_SPECS["RX"].matrix_fn(beta)
        expected = (
            tensor_product(rx, rx) @ rzz @ tensor_product(h, h) @ all_zeros_state(2)
        )
        assert statevectors_equal(sv, expected)

    def test_alternating_entangling_step2_transpiles(self):
        """Even/odd CZ sweeps with step=2 stay in STATIC mode."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(0, n - 1, 2):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            for i in qmc.range(1, n - 1, 2):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 6}, smoke_test=True)
        assert qc.execution_mode == ExecutionMode.STATIC
        assert qc.num_qubits == 6
        _assert_source_contains(qc, "def _qamomile_kernel():", "z.ctrl(q[0], q[1])")


class TestStdlibQPE:
    """Test Quantum Phase Estimation through the CUDA-Q pipeline."""

    def test_qpe_transpiles(self):
        """QPE emits a 3-bit phase register plus one target."""

        @qmc.qkernel
        def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.p(q, theta)

        @qmc.qkernel
        def qpe_3bit(phase: qmc.Float) -> qmc.Float:
            phase_register = qmc.qubit_array(3, name="phase_reg")
            target = qmc.qubit(name="target")
            target = qmc.x(target)
            phase_q = qmc.qpe(target, phase_register, phase_gate, theta=phase)
            return qmc.measure(phase_q)

        _, qc = _transpile_and_get_circuit(
            qpe_3bit, bindings={"phase": np.pi / 2}, smoke_test=True
        )
        assert qc.execution_mode == ExecutionMode.STATIC
        assert qc.num_qubits == 4
        _assert_source_contains(qc, "def _qamomile_kernel():", "q = cudaq.qvector(4)")

    def test_qpe_execution(self):
        """QPE returns the exact 0.25 phase estimate for pi/2."""

        @qmc.qkernel
        def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.p(q, theta)

        @qmc.qkernel
        def qpe_3bit(phase: qmc.Float) -> qmc.Float:
            phase_register = qmc.qubit_array(3, name="phase_reg")
            target = qmc.qubit(name="target")
            target = qmc.x(target)
            phase_q = qmc.qpe(target, phase_register, phase_gate, theta=phase)
            return qmc.measure(phase_q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(qpe_3bit, bindings={"phase": np.pi / 2})
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(4)",
            "x(q[3])",
        )
        result = exe.sample(transpiler.executor(), shots=256).result()
        assert len(result.results) == 1
        value, count = result.results[0]
        assert np.isclose(value, 0.25, atol=1e-9)
        assert count == 256


class TestAlgorithmBasicLayers:
    """Test qamomile.circuit.algorithm.basic helpers on CUDA-Q."""

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_ry_layer_statevector(self, n_qubits):
        """ry_layer matches the tensor product of per-qubit RY gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.3 * (i + 1) for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}, smoke_test=True
        )
        sv = _run_statevector(qc)
        matrices = [GATE_SPECS["RY"].matrix_fn(theta) for theta in thetas]
        expected = matrices[-1]
        for matrix in reversed(matrices[:-1]):
            expected = tensor_product(expected, matrix)
        expected = expected @ all_zeros_state(n_qubits)
        assert statevectors_equal(sv, expected)

    def test_rx_layer_statevector(self):
        """rx_layer on 3 qubits matches the analytical statevector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = rx_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.5, 1.0, 1.5]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "thetas": thetas}, smoke_test=True
        )
        sv = _run_statevector(qc)
        expected = tensor_product(
            GATE_SPECS["RX"].matrix_fn(thetas[2]),
            tensor_product(
                GATE_SPECS["RX"].matrix_fn(thetas[1]),
                GATE_SPECS["RX"].matrix_fn(thetas[0]),
            ),
        ) @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)

    def test_rz_layer_statevector(self):
        """rz_layer preserves the correct per-wire phase evolution."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = rz_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.2, 0.4, 0.6]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "thetas": thetas}, smoke_test=True
        )
        sv = _run_statevector(qc)
        expected = tensor_product(
            GATE_SPECS["RZ"].matrix_fn(thetas[2]),
            tensor_product(
                GATE_SPECS["RZ"].matrix_fn(thetas[1]),
                GATE_SPECS["RZ"].matrix_fn(thetas[0]),
            ),
        ) @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)

    def test_cz_entangling_layer_on_plus_state(self):
        """CZ entangling layer on |+>^3 matches the manual construction."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3}, smoke_test=True)
        sv = _run_statevector(qc)
        h = GATE_SPECS["H"].matrix_fn()
        cz = GATE_SPECS["CZ"].matrix_fn()
        state = tensor_product(h, tensor_product(h, h)) @ all_zeros_state(3)
        state = tensor_product(identity(2), cz) @ state
        state = tensor_product(cz, identity(2)) @ state
        assert statevectors_equal(sv, state)

    def test_ry_plus_cz_statevector(self):
        """A variational ry_layer + cz_entangling_layer block is emitted faithfully."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        thetas = [0.5, 1.0, 1.5]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "thetas": thetas}, smoke_test=True
        )
        sv = _run_statevector(qc)
        ry0, ry1, ry2 = [GATE_SPECS["RY"].matrix_fn(theta) for theta in thetas]
        cz = GATE_SPECS["CZ"].matrix_fn()
        state = tensor_product(ry2, tensor_product(ry1, ry0)) @ all_zeros_state(3)
        state = tensor_product(identity(2), cz) @ state
        state = tensor_product(cz, identity(2)) @ state
        assert statevectors_equal(sv, state)


class TestAlgorithmQAOAModules:
    """Test qamomile.circuit.algorithm.qaoa helpers on CUDA-Q."""

    @staticmethod
    def _simple_ising():
        return {(0, 1): 1.0}, {0: 0.5, 1: -0.5}

    def test_superposition_vector(self):
        """superposition_vector produces a uniform |+...+> state."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(qmc.uint(3))
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        sv = _run_statevector(qc)
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        assert statevectors_equal(sv, expected)

    def test_ising_cost(self):
        """ising_cost emits a valid unitary cost layer."""
        quad, linear = self._simple_ising()

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = ising_cost(quad_, linear_, q, gamma)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={"quad_": quad, "linear_": linear, "gamma": 0.5},
            smoke_test=True,
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_x_mixer_circuit(self):
        """x_mixer applies the expected mixer evolution."""

        @qmc.qkernel
        def circuit(beta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = x_mixer(q, beta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"beta": np.pi / 4}, smoke_test=True
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_qaoa_layers_p1(self):
        """Single-layer qaoa_layers emits a normalized state."""
        quad, linear = self._simple_ising()

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = qaoa_layers(qmc.uint(1), quad_, linear_, q, gammas, betas)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": np.array([0.3]),
                "betas": np.array([0.7]),
            },
            smoke_test=True,
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_qaoa_state_end_to_end(self):
        """Full qaoa_state builds a normalized state."""
        quad, linear = self._simple_ising()

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(qmc.uint(1), quad_, linear_, qmc.uint(2), gammas, betas)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": np.array([0.5]),
                "betas": np.array([0.8]),
            },
            smoke_test=True,
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_qaoa_sampling_consistency(self):
        """QAOA state can be sampled through the CUDA-Q executor."""
        quad, linear = self._simple_ising()

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(qmc.uint(1), quad_, linear_, qmc.uint(2), gammas, betas)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": np.array([0.5]),
                "betas": np.array([0.8]),
            },
        )
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(2)",
            "h(q[0])",
        )
        result = exe.sample(transpiler.executor(), shots=200).result()
        assert sum(count for _, count in result.results) == 200


class TestCompileTimeIfPhiPropagation:
    """Compile-time if lowering should preserve selected classical values."""

    def test_direct_classical_if_after_qinit_flag_true(self):
        """flag=True selects the true branch angle."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(1.0)
        )
        assert statevectors_equal(sv, expected)

    def test_direct_classical_if_after_qinit_flag_false(self):
        """flag=False selects the false branch angle."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 0}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(2.0)
        )
        assert statevectors_equal(sv, expected)

    def test_comparison_derived_classical_if_after_qinit(self):
        """Comparison-derived compile-time conditions lower correctly."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag > 0:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(1.0)
        )
        assert statevectors_equal(sv, expected)

    def test_comparison_derived_classical_if_flag_zero(self):
        """Comparison-derived false branches also lower correctly."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag > 0:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 0}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(2.0)
        )
        assert statevectors_equal(sv, expected)

    def test_symbolic_parameter_alias_before_qinit(self):
        """A symbolic angle routed through a compile-time if is preserved."""
        flag = True

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            angle = qmc.float_(0.5)
            if flag:
                angle = theta
            else:
                angle = qmc.float_(0.5)
            q = qmc.qubit("q")
            q = qmc.rx(q, angle)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = _assert_compiled_source(
            exe,
            "def _qamomile_kernel(thetas: list[float]):",
            "q = cudaq.qvector(1)",
            "rx(thetas[0], q[0])",
        )
        executor = transpiler.executor()
        sample = exe.sample(executor, bindings={"theta": np.pi}, shots=16).result()
        bound = executor.bind_parameters(
            qc, {"theta": np.pi}, exe.compiled_quantum[0].parameter_metadata
        )
        sv = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(np.pi)
        )
        assert {value for value, _ in sample.results} == {1}
        assert statevectors_equal(sv, expected)

    def test_bit_vector_phi_merge_flag_true(self):
        """Branch-local Vector[Bit] values merge correctly for the true branch."""
        flag = True

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            alt = qmc.qubit_array(2, "alt")
            q[0] = qmc.x(q[0])
            alt[1] = qmc.x(alt[1])
            if flag:
                bits = qmc.measure(q)
            else:
                bits = qmc.measure(alt)
            return bits

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(4)",
            "x(q[0])",
            "x(q[3])",
        )
        result = exe.sample(transpiler.executor(), shots=32).result()
        assert {value for value, _ in result.results} == {(1, 0)}

    def test_bit_vector_phi_merge_flag_false(self):
        """Branch-local Vector[Bit] values merge correctly for the false branch."""
        flag = False

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            alt = qmc.qubit_array(2, "alt")
            q[0] = qmc.x(q[0])
            alt[1] = qmc.x(alt[1])
            if flag:
                bits = qmc.measure(q)
            else:
                bits = qmc.measure(alt)
            return bits

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(4)",
            "x(q[0])",
            "x(q[3])",
        )
        result = exe.sample(transpiler.executor(), shots=32).result()
        assert {value for value, _ in result.results} == {(0, 1)}


class TestDirectCastMeasure:
    """Direct Vector[Qubit] -> QFixed -> measure should preserve carriers."""

    def test_direct_cast_measure_resolves_carriers(self):
        """A direct cast to QFixed measures both qubit carriers."""

        @qmc.qkernel
        def circuit() -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            qf = qmc.cast(q, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        segment = exe.compiled_quantum[0]
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(2)",
        )
        result = exe.sample(transpiler.executor(), shots=32).result()
        assert len(segment.clbit_map) == 2
        assert len(segment.measurement_qubit_map) == 2
        assert {value for value, _ in result.results} == {0.0}

    def test_cast_after_gate_measure(self):
        """A gate before the cast does not break QFixed measurement emission."""

        @qmc.qkernel
        def circuit() -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            qf = qmc.cast(q, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit)
        segment = exe.compiled_quantum[0]
        _assert_compiled_source(
            exe,
            "def _qamomile_kernel():",
            "q = cudaq.qvector(2)",
            "h(q[0])",
        )
        result = exe.sample(transpiler.executor(), shots=64).result()
        assert len(segment.clbit_map) == 2
        assert len(segment.measurement_qubit_map) == 2
        assert len(result.results) == 2
        assert sum(count for _, count in result.results) == 64


# ============================================================================
# Additional High-Level Patterns
# ============================================================================


class TestCustomCompositeGate:
    """Custom CompositeGate decomposition should work on CUDA-Q."""

    def test_custom_composite_transpiles(self):
        """A BellPair composite gate can be inlined into CUDA-Q source."""
        from qamomile.circuit.frontend.composite_gate import CompositeGate

        class BellPair(CompositeGate):
            custom_name = "bell_pair"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple) -> tuple:
                q0, q1 = qubits
                q0 = qmc.h(q0)
                q0, q1 = qmc.cx(q0, q1)
                return q0, q1

        bell = BellPair()

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = bell(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        assert qc.execution_mode == ExecutionMode.STATIC
        assert qc.num_qubits == 2
        _assert_source_contains(qc, "def _qamomile_kernel():", "h(q[0])", "x.ctrl(q[0], q[1])")

    def test_composite_statevector(self):
        """A BellPair composite gate prepares |Phi+>."""
        from qamomile.circuit.frontend.composite_gate import CompositeGate

        class BellPair(CompositeGate):
            custom_name = "bell_pair"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple) -> tuple:
                q0, q1 = qubits
                q0 = qmc.h(q0)
                q0, q1 = qmc.cx(q0, q1)
                return q0, q1

        bell = BellPair()

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = bell(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        sv = _run_statevector(qc)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2)
        expected[3] = 1.0 / np.sqrt(2)
        assert statevectors_equal(sv, expected)


class TestDeepNestedQKernelComposition:
    """Three-level qkernel nesting should inline correctly on CUDA-Q."""

    def test_three_level_nesting_statevector(self):
        """outer -> middle -> inner applies RX then H."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        @qmc.qkernel
        def middle(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = inner(q, theta)
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def outer(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = middle(q, theta)
            return qmc.measure(q)

        theta = np.pi / 3
        _, qc = _transpile_and_get_circuit(
            outer, bindings={"theta": theta}, smoke_test=True
        )
        sv = _run_statevector(qc)
        expected = (
            GATE_SPECS["H"].matrix_fn()
            @ GATE_SPECS["RX"].matrix_fn(theta)
            @ all_zeros_state(1)
        )
        assert statevectors_equal(sv, expected)

    def test_deep_nesting_two_params(self):
        """Nested kernels can thread multiple floating parameters."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, alpha)
            return q

        @qmc.qkernel
        def middle(q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float) -> qmc.Qubit:
            q = inner(q, alpha)
            q = qmc.ry(q, beta)
            return q

        @qmc.qkernel
        def outer(alpha: qmc.Float, beta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = middle(q, alpha, beta)
            return qmc.measure(q)

        alpha, beta = np.pi / 6, np.pi / 3
        _, qc = _transpile_and_get_circuit(
            outer, bindings={"alpha": alpha, "beta": beta}, smoke_test=True
        )
        sv = _run_statevector(qc)
        expected = (
            GATE_SPECS["RY"].matrix_fn(beta)
            @ GATE_SPECS["RX"].matrix_fn(alpha)
            @ all_zeros_state(1)
        )
        assert statevectors_equal(sv, expected)

    def test_sub_kernel_per_element_statevector(self):
        """A sub-kernel called per element in a loop yields |++>."""

        @qmc.qkernel
        def apply_h(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def main(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = apply_h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main, bindings={"n": 2}, smoke_test=True)
        sv = _run_statevector(qc)
        expected = np.array([1, 1, 1, 1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)


class TestControlledSubRoutines:
    """controlled() should support high-level sub-routines on CUDA-Q."""

    def test_controlled_ry_control_on(self):
        """Controlled-RY with ctrl=|1> matches the CRY unitary."""

        @qmc.qkernel
        def ry_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.ry(q, theta)
            return q

        controlled_ry = qmc.controlled(ry_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = controlled_ry(q[0], q[1], theta=theta)
            return qmc.measure(q)

        theta = np.pi / 2
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": theta}, smoke_test=True
        )
        sv = _run_statevector(qc)
        state = tensor_product(
            identity(2), GATE_SPECS["X"].matrix_fn()
        ) @ all_zeros_state(2)
        expected = GATE_SPECS["CRY"].matrix_fn(theta) @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_ry_control_off(self):
        """Controlled-RY with ctrl=|0> leaves the target untouched."""

        @qmc.qkernel
        def ry_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.ry(q, theta)
            return q

        controlled_ry = qmc.controlled(ry_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = controlled_ry(q[0], q[1], theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": np.pi / 2}, smoke_test=True
        )
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, all_zeros_state(2))

    def test_controlled_rz_statevector(self):
        """Controlled-RZ with ctrl=|1> matches the CRZ unitary."""

        @qmc.qkernel
        def rz_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rz(q, theta)
            return q

        controlled_rz = qmc.controlled(rz_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = controlled_rz(q[0], q[1], theta=theta)
            return qmc.measure(q)

        theta = np.pi / 3
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": theta}, smoke_test=True
        )
        sv = _run_statevector(qc)
        state = tensor_product(
            identity(2), GATE_SPECS["X"].matrix_fn()
        ) @ all_zeros_state(2)
        expected = GATE_SPECS["CRZ"].matrix_fn(theta) @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_multi_gate_kernel(self):
        """A controlled multi-gate helper kernel transpiles."""

        @qmc.qkernel
        def hx_gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        controlled_hx = qmc.controlled(hx_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = controlled_hx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        assert qc.num_qubits == 2
        assert qc.execution_mode == ExecutionMode.STATIC
        _assert_source_contains(qc, "def _qamomile_kernel():", "q = cudaq.qvector(2)")

    def test_controlled_power_2(self):
        """A powered controlled phase helper transpiles."""

        @qmc.qkernel
        def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.p(q, theta)
            return q

        cp_pow = qmc.controlled(p_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = cp_pow(q[0], q[1], power=2, theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": np.pi / 4}, smoke_test=True
        )
        assert qc.num_qubits == 2
        _assert_source_contains(qc, "def _qamomile_kernel():", "q = cudaq.qvector(2)")

    def test_controlled_power_4(self):
        """A higher powered controlled phase helper also transpiles."""

        @qmc.qkernel
        def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.p(q, theta)
            return q

        cp_pow = qmc.controlled(p_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = cp_pow(q[0], q[1], power=4, theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": np.pi / 4}, smoke_test=True
        )
        assert qc.num_qubits == 2
        _assert_source_contains(qc, "def _qamomile_kernel():", "q = cudaq.qvector(2)")


class TestEntanglementAndParityPatterns:
    """Portable entanglement/parity patterns should run on CUDA-Q."""

    def test_parity_check_even(self):
        """Even data parity leaves the ancilla at |0>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[1] = qmc.x(q[1])
            q[2] = qmc.x(q[2])
            q[1], q[0] = qmc.cx(q[1], q[0])
            q[2], q[0] = qmc.cx(q[2], q[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 0b110)
        assert statevectors_equal(sv, expected)

    def test_parity_check_odd(self):
        """Odd data parity flips the ancilla."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[1], q[0] = qmc.cx(q[1], q[0])
            q[2], q[0] = qmc.cx(q[2], q[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 0b101)
        assert statevectors_equal(sv, expected)

    def test_entanglement_witness_via_cz(self):
        """CZ on |++> creates the expected entangled state."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        sv = _run_statevector(qc)
        expected = np.array([1, 1, 1, -1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)

    def test_repetition_code_encode_0(self):
        """Encoding |0> with a repetition code yields |000>."""

        @qmc.qkernel
        def encode() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(encode, smoke_test=True)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, all_zeros_state(3))

    def test_repetition_code_encode_1(self):
        """Encoding |1> with a repetition code yields |111>."""

        @qmc.qkernel
        def encode() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(encode, smoke_test=True)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 0b111)
        assert statevectors_equal(sv, expected)

    def test_swap_test_same_states(self):
        """Swap test on identical inputs returns ancilla |0>."""

        @qmc.qkernel
        def swap_kernel(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.swap(q0, q1)

        cswap = qmc.controlled(swap_kernel)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1], q[2] = cswap(q[0], q[1], q[2])
            q[0] = qmc.h(q[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, all_zeros_state(3))


class TestWhileIfSharedLocalPhi:
    """While-if shared locals should preserve dead/live distinction."""

    def test_while_loop_with_if_else_same_name_dead_local_transpile(self):
        """A dead shared local inside while-if should transpile."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2_t")
                    q2 = qmc.h(q2)
                    bit = qmc.measure(q2)
                else:
                    q2 = qmc.qubit("q2_f")
                    bit = qmc.measure(q2)
            return bit

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        assert qc.num_clbits >= 1
        _assert_source_contains(qc, "while __b0[0]:")

    def test_while_loop_with_if_else_same_name_dead_local_reassigned_transpile(self):
        """A dead shared local that is overwritten later should transpile."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2_t")
                    q2 = qmc.h(q2)
                else:
                    q2 = qmc.qubit("q2_f")
                    q2 = qmc.x(q2)
                q2 = qmc.qubit("q2_new")
                bit = qmc.measure(q2)
            return bit

        _, qc = _transpile_and_get_circuit(circuit, smoke_test=True)
        assert qc.num_clbits >= 1
        _assert_source_contains(qc, "while __b0[0]:")

    def test_while_loop_with_if_else_same_name_live_local_dependency_error(self):
        """A live merged local with incompatible resources should still fail."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2_t")
                    q2 = qmc.h(q2)
                else:
                    q2 = qmc.qubit("q2_f")
                    q2 = qmc.x(q2)
                bit = qmc.measure(q2)
            return bit

        with pytest.raises((DependencyError, EmitError)):
            _transpile_and_get_circuit(circuit)


class TestDeutschJozsaAlgorithm:
    """Deutsch-Jozsa algorithm patterns should execute on CUDA-Q."""

    @staticmethod
    def _prob_input_all_zero(sv: np.ndarray, n_input: int) -> float:
        ancilla_stride = 2**n_input
        return abs(sv[0]) ** 2 + abs(sv[ancilla_stride]) ** 2

    def test_constant_zero_oracle(self):
        """A constant-zero oracle returns the all-zero input signature."""

        @qmc.qkernel
        def dj_constant_zero() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_constant_zero, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(qc, "def _qamomile_kernel():", "q = cudaq.qvector(3)")
        assert np.isclose(self._prob_input_all_zero(sv, 2), 1.0, atol=1e-10)

    def test_constant_1_all_zeros(self):
        """A constant-one oracle still returns the all-zero signature."""

        @qmc.qkernel
        def dj_constant_one() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[2] = qmc.x(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_constant_one, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(qc, "def _qamomile_kernel():", "x(q[2])")
        assert np.isclose(self._prob_input_all_zero(sv, 2), 1.0, atol=1e-10)

    def test_balanced_first_bit_non_zero(self):
        """A balanced oracle should suppress the all-zero input outcome."""

        @qmc.qkernel
        def dj_balanced() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_balanced, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(qc, "def _qamomile_kernel():", "x.ctrl(q[0], q[2])")
        assert self._prob_input_all_zero(sv, 2) < 0.01

    def test_constant_0_oracle_transpiles(self):
        """A sub-kernel oracle version of DJ also transpiles."""

        @qmc.qkernel
        def oracle_c0(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            return inputs, ancilla

        @qmc.qkernel
        def dj_c0(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_c0(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        _, qc = _transpile_and_get_circuit(dj_c0, bindings={"n": 3}, smoke_test=True)
        assert qc.num_qubits == 4
        _assert_source_contains(qc, "def _qamomile_kernel():", "q = cudaq.qvector(4)")

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_balanced_xor_transpiles(self, n):
        """A balanced XOR-style oracle scales to several input sizes."""

        @qmc.qkernel
        def dj_bxor(n_val: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n_val, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            for i in qmc.range(n_val):
                inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        _, qc = _transpile_and_get_circuit(
            dj_bxor, bindings={"n_val": n}, smoke_test=True
        )
        assert qc.num_qubits == n + 1
        _assert_source_contains(qc, "def _qamomile_kernel():", f"q = cudaq.qvector({n + 1})")


class TestBernsteinVaziraniAlgorithm:
    """Bernstein-Vazirani oracle patterns should reveal the secret string."""

    @staticmethod
    def _check_bv_result(
        sv: np.ndarray, n_input: int, expected_secret_idx: int
    ) -> None:
        ancilla_bit = 2**n_input
        probability = (
            abs(sv[expected_secret_idx]) ** 2
            + abs(sv[expected_secret_idx + ancilla_bit]) ** 2
        )
        assert np.isclose(probability, 1.0, atol=1e-6)

    def test_bv_secret_10(self):
        """Secret 10 is recovered from the input register."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(qc, "def _qamomile_kernel():", "x.ctrl(q[0], q[2])")
        self._check_bv_result(sv, n_input=2, expected_secret_idx=0b01)

    def test_bv_secret_01(self):
        """Secret 01 is recovered from the input register."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(qc, "def _qamomile_kernel():", "x.ctrl(q[1], q[2])")
        self._check_bv_result(sv, n_input=2, expected_secret_idx=0b10)

    def test_bv_secret_11(self):
        """Secret 11 is recovered from the input register."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(
            qc,
            "def _qamomile_kernel():",
            "x.ctrl(q[0], q[2])",
            "x.ctrl(q[1], q[2])",
        )
        self._check_bv_result(sv, n_input=2, expected_secret_idx=0b11)

    def test_bv_secret_101(self):
        """A 3-bit secret is recovered exactly."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[3] = qmc.x(q[3])
            q[3] = qmc.h(q[3])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            q[0], q[3] = qmc.cx(q[0], q[3])
            q[2], q[3] = qmc.cx(q[2], q[3])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv, smoke_test=True)
        sv = _run_statevector(qc)
        _assert_source_contains(
            qc,
            "def _qamomile_kernel():",
            "x.ctrl(q[0], q[3])",
            "x.ctrl(q[2], q[3])",
        )
        self._check_bv_result(sv, n_input=3, expected_secret_idx=0b101)


# ============================================================================
# Unresolved non-measurement if-condition rejection
# ============================================================================


class TestUnresolvedNonMeasurementIfRejection:
    """Non-measurement unresolved if-conditions must raise EmitError.

    When a kernel argument is used as an if-condition and left in
    ``parameters`` (not bound), the transpiler must raise an explicit
    error instead of silently dropping the branch.
    """

    def test_unresolved_flag_parameter_raises(self):
        """``if flag:`` with parameters=["flag", "theta"] must raise EmitError."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt, theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.rx(q, theta)
            return qmc.measure(q)

        with pytest.raises(EmitError, match="measurement results"):
            _transpile_and_get_circuit(circuit, parameters=["flag", "theta"])

    def test_unresolved_flag_only_raises(self):
        """``if flag:`` with parameters=["flag"] must raise EmitError."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        with pytest.raises(EmitError, match="measurement results"):
            _transpile_and_get_circuit(circuit, parameters=["flag"])

    def test_bound_flag_still_works(self):
        """``if flag:`` with bindings={"flag": 1} must still work."""

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


class TestUnresolvedStructuralSize:
    """Unresolved structural UInt must raise, not produce zero-size artifact."""

    def test_qubit_array_with_unresolved_size_raises(self):
        """qubit_array(n) with parameters=["n"] must raise EmitError."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        with pytest.raises((EmitError, ValueError)):
            transpiler.transpile(circuit, parameters=["n"])
