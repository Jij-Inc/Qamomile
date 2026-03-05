"""Regression tests for constant folding pass.

Tests verify that BinOp expressions are correctly folded and propagated
to GateOperation.theta during the constant folding pass.
"""

from typing import Any, TYPE_CHECKING

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.qaoa import x_mixier_circuit
from qamomile.circuit.algorithm.basic import ry_layer, cz_entangling_layer
from qamomile.qiskit.transpiler import QiskitTranspiler

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

from tests.transpiler.gate_test_specs import (
    GATE_SPECS,
    all_zeros_state,
    statevectors_equal,
    tensor_product,
)


# ---------------------------------------------------------------------------
# Module-level @qkernel definitions (required for inspect.getsource)
# ---------------------------------------------------------------------------


@qmc.qkernel
def x_mixer_test_circuit(
    n: qmc.UInt, beta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Wrapper that applies x_mixier_circuit and measures."""
    q = qmc.qubit_array(n, "q")
    q = x_mixier_circuit(q, beta)
    return qmc.measure(q)


# --- Float BinOp kernels: one per arithmetic operation ---


@qmc.qkernel
def binop_add_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX(theta + 1.0) on a single qubit."""
    q = qmc.qubit_array(1, "q")
    angle = theta + 1.0
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


@qmc.qkernel
def binop_sub_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX(theta - 0.5) on a single qubit."""
    q = qmc.qubit_array(1, "q")
    angle = theta - 0.5
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


@qmc.qkernel
def binop_mul_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX(theta * 3.0) on a single qubit."""
    q = qmc.qubit_array(1, "q")
    angle = theta * 3.0
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


@qmc.qkernel
def binop_div_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX(theta / 2.0) on a single qubit."""
    q = qmc.qubit_array(1, "q")
    angle = theta / 2.0
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


# --- Compound assignment kernels (+=, -=, *=) ---


@qmc.qkernel
def binop_iadd_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX with += compound assignment: angle = theta; angle += 1.0."""
    q = qmc.qubit_array(1, "q")
    angle = theta
    angle += 1.0
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


@qmc.qkernel
def binop_isub_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX with -= compound assignment: angle = theta; angle -= 0.5."""
    q = qmc.qubit_array(1, "q")
    angle = theta
    angle -= 0.5
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


@qmc.qkernel
def binop_imul_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX with *= compound assignment: angle = theta; angle *= 3.0."""
    q = qmc.qubit_array(1, "q")
    angle = theta
    angle *= 3.0
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


# --- Chained BinOp kernel (multiple operations) ---


@qmc.qkernel
def binop_chained_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """RX((theta * 2.0) + 0.5) on a single qubit — chained MUL then ADD."""
    q = qmc.qubit_array(1, "q")
    angle = theta * 2.0 + 0.5
    q[0] = qmc.rx(q[0], angle=angle)
    return qmc.measure(q)


# --- UInt BinOp kernels (floordiv, pow) — affect loop bounds ---


@qmc.qkernel
def binop_floordiv_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n // 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n // 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_pow_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n ** 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n ** 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def variational_classifier(
    n: qmc.UInt,
    params: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    """Two-layer variational classifier with RY + CZ entangling."""
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(2):
        q = ry_layer(q, params, layer * n)
        q = cz_entangling_layer(q)
    return qmc.expval(q, H)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _transpile_and_get_circuit(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
) -> tuple[Any, "QuantumCircuit"]:
    """Transpile a kernel and return ``(executable, circuit)``.

    Args:
        kernel: A ``@qmc.qkernel``-decorated function.
        bindings: Concrete values for kernel parameters.
        parameters: Names of parameters to keep symbolic.

    Returns:
        Tuple of (executable, Qiskit QuantumCircuit).
    """
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    qc = exe.compiled_quantum[0].circuit
    return exe, qc


def _run_statevector(qc: "QuantumCircuit") -> np.ndarray:
    """Run a Qiskit circuit and return the statevector.

    Removes measurement gates before simulation so the statevector
    is not collapsed.

    Args:
        qc: Qiskit QuantumCircuit (may contain measurements).

    Returns:
        Complex statevector as a numpy array.
    """
    from qiskit import QuantumCircuit as QC
    from qiskit_aer import AerSimulator

    # Build a new circuit without final measurements
    qc_no_meas = qc.remove_final_measurements(inplace=False)

    sim = AerSimulator(method="statevector")
    qc_no_meas.save_statevector()
    result = sim.run(qc_no_meas).result()
    return np.array(result.get_statevector())


def _gate_names(qc: "QuantumCircuit") -> list[str]:
    """Get list of gate names from a Qiskit circuit.

    Args:
        qc: Qiskit QuantumCircuit.

    Returns:
        List of gate operation names.
    """
    return [inst.operation.name for inst in qc.data]


# ---------------------------------------------------------------------------
# Tests — BinOp constant folding into GateOperation.theta
# ---------------------------------------------------------------------------

# Fixed seed for reproducible random beta values
_RNG = np.random.default_rng(42)
_RANDOM_BETAS = _RNG.uniform(0, 2 * np.pi, size=3).tolist()


class TestBinOpConstantFolding:
    """Tests for BinOp constant folding in GateOperation.theta."""

    @pytest.mark.parametrize(
        "n, beta",
        [
            (1, 0.0),                # edge: identity rotation
            (2, 0.4),                # original regression case
            (1, np.pi),              # edge: full π rotation
            (3, np.pi / 4),          # non-trivial angle, larger array
            (2, _RANDOM_BETAS[0]),   # random
            (1, _RANDOM_BETAS[1]),   # random, single qubit
            (3, _RANDOM_BETAS[2]),   # random, 3 qubits
        ],
        ids=[
            "n=1,beta=0",
            "n=2,beta=0.4",
            "n=1,beta=pi",
            "n=3,beta=pi/4",
            "n=2,random_beta_0",
            "n=1,random_beta_1",
            "n=3,random_beta_2",
        ],
    )
    def test_x_mixer_circuit_statevector(self, n: int, beta: float) -> None:
        """Statevector of x_mixier_circuit matches ⊗ RX(2β) |0…0⟩."""
        _, qc = _transpile_and_get_circuit(
            x_mixer_test_circuit, bindings={"n": n, "beta": beta}
        )
        sv = _run_statevector(qc)

        RX = GATE_SPECS["RX"].matrix_fn(2.0 * beta)
        expected = tensor_product(*([RX] * n)) @ all_zeros_state(n)
        assert statevectors_equal(sv, expected)


# ---------------------------------------------------------------------------
# Tests — Variational classifier with loop-dependent BinOp
# ---------------------------------------------------------------------------


class TestVariationalClassifier:
    """Tests for variational classifier patterns with BinOp in loop."""

    @pytest.mark.parametrize(
        "n, num_layers, expected_ry, expected_cz",
        [
            (3, 2, 6, 4),   # original regression case: 3 qubits, 2 layers
            (2, 2, 4, 2),   # 2 qubits, 2 layers
        ],
        ids=["3q_2layer", "2q_2layer"],
    )
    def test_variational_classifier_gate_counts(
        self,
        n: int,
        num_layers: int,
        expected_ry: int,
        expected_cz: int,
    ) -> None:
        """Gate counts match n*layers RY and (n-1)*layers CZ."""
        H_label = qm_o.Hamiltonian(num_qubits=n)
        H_label += qm_o.Z(0)
        total_params = num_layers * n
        params = [0.1 * i for i in range(total_params)]

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            variational_classifier,
            bindings={"n": n, "params": params, "H": H_label},
        )

        qc = exe.compiled_quantum[0].circuit
        ry_count = sum(1 for name in _gate_names(qc) if name == "ry")
        cz_count = sum(1 for name in _gate_names(qc) if name == "cz")

        assert ry_count == expected_ry
        assert cz_count == expected_cz

        # Verify RY angles are non-zero (BinOp folding propagated correctly)
        ry_angles = [
            float(inst.operation.params[0])
            for inst in qc.data
            if inst.operation.name == "ry"
        ]
        for i, angle in enumerate(ry_angles):
            assert np.isclose(angle, params[i]), (
                f"RY[{i}] angle {angle} != expected {params[i]}"
            )


# ---------------------------------------------------------------------------
# Tests — All BinOp kinds: angle propagation + statevector verification
# ---------------------------------------------------------------------------


def _extract_rx_angles(qc: "QuantumCircuit") -> list[float]:
    """Extract all RX gate angles from a Qiskit circuit."""
    return [
        float(inst.operation.params[0])
        for inst in qc.data
        if inst.operation.name == "rx"
    ]


class TestBinOpAllOperations:
    """Tests for all Float BinOp kinds: verify folded angles and statevectors.

    Each test transpiles a kernel that applies a single BinOp to a Float
    parameter, then checks:
      1. The resulting RX gate angle matches the expected value.
      2. The statevector matches RX(expected)|0⟩.
    """

    @pytest.mark.parametrize(
        "kernel, bindings, expected_angle",
        [
            # Basic arithmetic operations
            (binop_add_circuit, {"theta": 0.5}, 0.5 + 1.0),
            (binop_sub_circuit, {"theta": 1.5}, 1.5 - 0.5),
            (binop_mul_circuit, {"theta": 0.3}, 0.3 * 3.0),
            (binop_div_circuit, {"theta": np.pi}, np.pi / 2.0),
            # Compound assignments (same BinOp, different syntax)
            (binop_iadd_circuit, {"theta": 0.5}, 0.5 + 1.0),
            (binop_isub_circuit, {"theta": 1.5}, 1.5 - 0.5),
            (binop_imul_circuit, {"theta": 0.3}, 0.3 * 3.0),
            # Chained operations
            (binop_chained_circuit, {"theta": 0.7}, 0.7 * 2.0 + 0.5),
        ],
        ids=[
            "add(theta+1.0)",
            "sub(theta-0.5)",
            "mul(theta*3.0)",
            "div(theta/2.0)",
            "iadd(theta+=1.0)",
            "isub(theta-=0.5)",
            "imul(theta*=3.0)",
            "chained(theta*2.0+0.5)",
        ],
    )
    def test_binop_angle_propagation(
        self,
        kernel: Any,
        bindings: dict[str, Any],
        expected_angle: float,
    ) -> None:
        """Verify the folded BinOp angle in the RX gate matches expected."""
        _, qc = _transpile_and_get_circuit(kernel, bindings=bindings)
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == 1, f"Expected 1 RX gate, got {len(rx_angles)}"
        assert np.isclose(rx_angles[0], expected_angle), (
            f"RX angle {rx_angles[0]} != expected {expected_angle}"
        )

    @pytest.mark.parametrize(
        "kernel, bindings, expected_angle",
        [
            (binop_add_circuit, {"theta": 0.5}, 0.5 + 1.0),
            (binop_sub_circuit, {"theta": 1.5}, 1.5 - 0.5),
            (binop_mul_circuit, {"theta": 0.3}, 0.3 * 3.0),
            (binop_div_circuit, {"theta": np.pi}, np.pi / 2.0),
            (binop_chained_circuit, {"theta": 0.7}, 0.7 * 2.0 + 0.5),
        ],
        ids=["add", "sub", "mul", "div", "chained"],
    )
    def test_binop_statevector(
        self,
        kernel: Any,
        bindings: dict[str, Any],
        expected_angle: float,
    ) -> None:
        """Statevector matches RX(expected_angle)|0⟩."""
        _, qc = _transpile_and_get_circuit(kernel, bindings=bindings)
        sv = _run_statevector(qc)

        RX = GATE_SPECS["RX"].matrix_fn(expected_angle)
        expected = RX @ all_zeros_state(1)
        assert statevectors_equal(sv, expected)


# ---------------------------------------------------------------------------
# Tests — UInt BinOp (floordiv, pow) folding into loop bounds
# ---------------------------------------------------------------------------


class TestUIntBinOpFolding:
    """Tests for UInt BinOp kinds (``//``, ``**``) that affect loop bounds.

    These operations produce UInt results used as ``qmc.range`` arguments.
    The constant folding pass must resolve them so the loop can be unrolled.
    """

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (4, 0.5, 2),   # 4 // 2 = 2
            (6, 0.3, 3),   # 6 // 2 = 3
            (2, 1.0, 1),   # 2 // 2 = 1
        ],
        ids=["4//2=2", "6//2=3", "2//2=1"],
    )
    def test_floordiv_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n // 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_floordiv_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n//2={n // 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (2, 0.3, 4),   # 2 ** 2 = 4
            (1, 0.5, 1),   # 1 ** 2 = 1
        ],
        ids=["2**2=4", "1**2=1"],
    )
    def test_pow_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n ** 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_pow_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n**2={n ** 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------


class TestConstantFoldEdgeCases:
    """Edge-case and negative tests for constant folding."""

    def test_x_mixer_zero_qubits_produces_empty_circuit(self) -> None:
        """Zero qubits produces a circuit with no RX gates."""
        _, qc = _transpile_and_get_circuit(
            x_mixer_test_circuit, bindings={"n": 0, "beta": 0.5}
        )
        rx_count = sum(1 for name in _gate_names(qc) if name == "rx")
        assert rx_count == 0
