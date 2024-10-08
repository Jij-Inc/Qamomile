# File: tests/circuit/test_circuit.py

import pytest
from qamomile.core.circuit import (
    QuantumCircuit,
    Parameter,
    SingleQubitGateType,
    ParametricSingleQubitGateType,
    TwoQubitGateType,
    ThreeQubitGateType,
    MeasurementGate,
    Operator
)


def test_quantum_circuit_initialization():
    qc = QuantumCircuit(2, 2, name="test_circuit")
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    assert qc.name == "test_circuit"


def test_single_qubit_gates():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    qc.y(0)
    qc.z(0)
    qc.s(0)
    qc.t(0)
    assert len(qc.gates) == 6
    assert all(gate.gate in SingleQubitGateType for gate in qc.gates)


def test_parametric_single_qubit_gates():
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.rx(theta, 0)
    qc.ry(theta, 0)
    qc.rz(theta, 0)
    assert len(qc.gates) == 3
    assert all(gate.gate in ParametricSingleQubitGateType for gate in qc.gates)
    assert all(gate.parameter == theta for gate in qc.gates)


def test_two_qubit_gates():
    qc = QuantumCircuit(2)
    qc.cnot(0, 1)
    qc.cz(0, 1)
    assert len(qc.gates) == 2
    assert all(gate.gate in TwoQubitGateType for gate in qc.gates)


def test_three_qubit_gate():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    assert len(qc.gates) == 1
    assert qc.gates[0].gate == ThreeQubitGateType.CCX


def test_measurement():
    qc = QuantumCircuit(2, 2)
    qc.measure(0, 0)
    qc.measure(1, 1)
    assert len(qc.gates) == 2
    assert all(isinstance(gate, MeasurementGate) for gate in qc.gates)


def test_measure_all():
    qc = QuantumCircuit(3)
    qc.measure_all()
    assert len(qc.gates) == 3
    assert all(isinstance(gate, MeasurementGate) for gate in qc.gates)


def test_circuit_append():
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cnot(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.append(qc1)
    assert len(qc2.gates) == 1
    assert isinstance(qc2.gates[0], Operator)


def test_get_parameters():
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    params = qc.get_parameters()
    assert len(params) == 2
    assert theta in params and phi in params


def test_invalid_qubit_index():
    qc = QuantumCircuit(1)
    with pytest.raises(ValueError):
        qc.x(1)


def test_invalid_measurement():
    qc = QuantumCircuit(1, 0)
    with pytest.raises(ValueError):
        qc.measure(0, 0)
