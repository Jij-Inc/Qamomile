# File: tests/circuit/test_circuit.py

import pytest
import random
import math
import qamomile.core.operator as qm_o
from qamomile.core.circuit import (
    SingleQubitGateType,
    SingleQubitGate,
    QuantumCircuit,
    Parameter,
    Value,
    ParametricSingleQubitGateType,
    TwoQubitGateType,
    ThreeQubitGateType,
    MeasurementGate,
    ParametricExpGate,
    Operator,
)


# >>> SingleQubitGate >>>
def test_single_qubit_gate_type():
    """Verify if all single qubit gates are present in the SingleQubitGateType enum.

    Check if
    1. SingleQubitGateType.H exists,
    2. SingleQubitGateType.X exists,
    3. SingleQubitGateType.Y exists,
    4. SingleQubitGateType.Z exists,
    5. SingleQubitGateType.S exists,
    6. SingleQubitGateType.T exists,
    7. The length of SingleQubitGateType is 6.
    """
    # 1. SingleQubitGateType.H exists,
    assert SingleQubitGateType.H
    # 2. SingleQubitGateType.X exists,
    assert SingleQubitGateType.X
    # 3. SingleQubitGateType.Y exists,
    assert SingleQubitGateType.Y
    # 4. SingleQubitGateType.Z exists,
    assert SingleQubitGateType.Z
    # 5. SingleQubitGateType.S exists,
    assert SingleQubitGateType.S
    # 6. SingleQubitGateType.T exists,
    assert SingleQubitGateType.T
    # 7. The length of SingleQubitGateType is 6.
    assert len(SingleQubitGateType) == 6


@pytest.mark.parametrize(
    "gate",
    [single_qubit_gate_type for single_qubit_gate_type in SingleQubitGateType],
)
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_single_qubit_gate(gate, qubit):
    """Create an instance of SingleQubitGate.

    Check if
    1. its gate is the same as the given gate,
    2. its qubit is the same as the given qubit.
    """
    single_qubit_gate = SingleQubitGate(gate, qubit)

    # 1. its gate is the same as the given gate,
    assert single_qubit_gate.gate == gate
    # 2. its qubit is the same as the given qubit.
    assert single_qubit_gate.qubit == qubit


# <<< SingleQubitGate <<<


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
    qc2 = QuantumCircuit(1)
    random_float = random.uniform(0, 4 * math.pi)
    qc2.rx(random_float, 0)
    qc2.ry(random_float, 0)
    qc2.rz(random_float, 0)

    assert len(qc.gates) == 3
    assert all(gate.gate in ParametricSingleQubitGateType for gate in qc.gates)
    assert all(gate.parameter == theta for gate in qc.gates)
    assert all(gate.parameter.value == random_float for gate in qc2.gates)


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


def test_exp_evolution():
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.exp_evolution(theta, hamiltonian)
    assert len(qc.gates) == 1
    assert isinstance(qc.gates[0], ParametricExpGate)
    assert qc.gates[0].parameter == theta
    assert len(qc.gates[0].indices) == 2
    assert qc.gates[0].hamiltonian == hamiltonian

    hamiltonian2 = qm_o.Hamiltonian()
    hamiltonian2 += qm_o.X(0) * qm_o.Y(1) + qm_o.Z(0) * qm_o.X(1)
    qc2 = QuantumCircuit(2)
    qc2.exp_evolution(theta, hamiltonian)
    assert len(qc2.gates) == 1
    assert isinstance(qc2.gates[0], ParametricExpGate)
    assert qc2.gates[0].parameter == theta
    assert len(qc2.gates[0].indices) == 2
    assert qc2.gates[0].hamiltonian == hamiltonian


def test_invalid_exp_evolution():
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    with pytest.raises(ValueError):
        qc.exp_evolution(theta, hamiltonian)


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
