"""Tests for basic rotation and entanglement layers."""

import pytest
import sympy as sp
import qamomile.circuit as qmc
from qamomile.circuit.estimator import count_gates
from qamomile.circuit.algorithm.basic import (
    rx_layer,
    ry_layer,
    rz_layer,
    cz_entangling_layer,
)


# ---------------------------------------------------------------------------
# Symbolic gate count tests (count_gates on IR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_rx_layer_gate_count(num_qubits):
    """Test that rx_layer produces n RX gates."""
    counts = count_gates(rx_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    assert counts.single_qubit.subs(q_dim0, num_qubits) == num_qubits
    assert counts.two_qubit.subs(q_dim0, num_qubits) == 0


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_ry_layer_gate_count(num_qubits):
    """Test that ry_layer produces n RY gates."""
    counts = count_gates(ry_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    assert counts.single_qubit.subs(q_dim0, num_qubits) == num_qubits
    assert counts.two_qubit.subs(q_dim0, num_qubits) == 0


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_rz_layer_gate_count(num_qubits):
    """Test that rz_layer produces n RZ gates."""
    counts = count_gates(rz_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    assert counts.single_qubit.subs(q_dim0, num_qubits) == num_qubits
    assert counts.two_qubit.subs(q_dim0, num_qubits) == 0


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_cz_entangling_layer_gate_count(num_qubits):
    """Test that cz_entangling_layer produces n-1 CZ gates."""
    counts = count_gates(cz_entangling_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
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
    rx_count = sum(1 for inst in qc.data if inst.operation.name == "rx")
    assert rx_count == num_qubits, f"Expected {num_qubits} RX gates, got {rx_count}"


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
    ry_count = sum(1 for inst in qc.data if inst.operation.name == "ry")
    assert ry_count == num_qubits, f"Expected {num_qubits} RY gates, got {ry_count}"


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
    rz_count = sum(1 for inst in qc.data if inst.operation.name == "rz")
    assert rz_count == num_qubits, f"Expected {num_qubits} RZ gates, got {rz_count}"


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
