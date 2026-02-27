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

    # Build a new circuit without measurements
    qc_no_meas = QC(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name != "measure":
            qc_no_meas.append(inst)

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
