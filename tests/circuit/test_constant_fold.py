"""Tests for constant folding of BinOp results used as gate theta."""

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


def test_constant_fold_binop_theta():
    """BinOp result used as GateOperation.theta should be correctly folded."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Bit:
        doubled = theta * 2
        q = qmc.qubit("q")
        q = qmc.rz(q, doubled)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(circuit, bindings={"theta": 0.5})
    rz_params = [
        instr.operation.params[0]
        for instr in executable.quantum_circuit.data
        if instr.operation.name == "rz"
    ]
    assert len(rz_params) == 1
    assert abs(float(rz_params[0]) - 1.0) < 1e-10


def test_constant_fold_chained_binop_theta():
    """Chained BinOp results should be correctly folded through to theta."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Bit:
        a = theta * 2  # 0.5 * 2 = 1.0
        b = a + 0.5  # 1.0 + 0.5 = 1.5
        q = qmc.qubit("q")
        q = qmc.rx(q, b)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(circuit, bindings={"theta": 0.5})
    rx_params = [
        instr.operation.params[0]
        for instr in executable.quantum_circuit.data
        if instr.operation.name == "rx"
    ]
    assert len(rx_params) == 1
    assert abs(float(rx_params[0]) - 1.5) < 1e-10
