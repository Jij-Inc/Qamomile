"""Tests for basic rotation and entanglement layers."""

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import (
    cx_entangling_layer,
    cz_entangling_layer,
    rx_layer,
    ry_layer,
    rz_layer,
)
from qamomile.circuit.estimator import count_gates
from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.graph.graph import Graph
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation

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


def test_cx_entangling_layer():
    """Test cx_entangling_layer produces correct CX gates with concrete bindings."""

    @qmc.qkernel
    def circuit(
        n: qmc.UInt,
    ) -> qmc.Vector[qmc.Qubit]:
        q = qmc.qubit_array(n, "q")
        q = cx_entangling_layer(q)
        return q

    circuit_ir = circuit.build()

    # If the IR is a graph, we can count gates directly without transpilation
    assert isinstance(circuit_ir, Graph)
    operations = circuit_ir.operations
    assert len(operations) == 2
    assert isinstance(operations[0], QInitOperation)
    assert isinstance(operations[1], CallBlockOperation)
    block = operations[1].operands[0]
    assert isinstance(block, BlockValue)
    assert block.name == "cx_entangling_layer"
    assert block.label_args == ["q"]
    block_operations = block.operations
    assert len(block_operations) == 3
    assert isinstance(block_operations[0], BinOp)
    assert isinstance(block_operations[1], ForOperation)
    assert isinstance(block_operations[2], ReturnOperation)
    for_operations = block_operations[1].operations
    assert len(for_operations) == 3
    assert isinstance(for_operations[0], BinOp)  # i+1 in RHS
    assert isinstance(for_operations[1], GateOperation)
    assert isinstance(for_operations[2], BinOp)  # i+1 in LHS
    gate_op = for_operations[1]
    assert gate_op.gate_type == GateOperationType.CX
    # The following two assertions are for now loose.
    assert len(block.input_values) == 1
    assert len(block.return_values) == 1
