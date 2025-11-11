# File: tests/circuit/test_circuit.py
import math
import random

import numpy as np
import pytest

import qamomile.core.operator as qm_o
from qamomile.core.circuit import (
    QuantumCircuit,
    Parameter,
    SingleQubitGateType,
    ParametricSingleQubitGateType,
    TwoQubitGateType,
    ThreeQubitGateType,
    ParametricTwoQubitGateType,
    MeasurementGate,
    ParametricExpGate,
    Operator,
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


@pytest.mark.parametrize("seed", [901 + i for i in range(50)])
def test_phase_gadget_as_rz(seed):
    """Add phase-gadget to one qubit, which is RZ, randomly.

    Check if
    - the number of gates is 1,
    - the gate is RZ,
    - the qubit is correct,
    - the angle is correct.
    """
    # Fix the seed for reproducibility.
    np.random.seed(seed)

    # Choose the number of qubits randomly between 1 and 100.
    num_qubits = np.random.randint(1, 101)
    qc = QuantumCircuit(num_qubits)
    # Randomem angle between 0 and 4pi
    angle = random.uniform(0, 4 * math.pi)
    # Add phase gadget, which is RZ, for random single qubit.
    qubit = random.randint(0, num_qubits - 1)
    qc.phase_gadget(angle, [qubit])

    # - the number of gates is 1,
    assert len(qc.gates) == 1
    # - the gate is RZ,
    assert qc.gates[0].gate == ParametricSingleQubitGateType.RZ
    # - the qubit is correct,
    assert qc.gates[0].qubit == qubit
    # - the angle is correct.
    assert qc.gates[0].parameter.value == angle


@pytest.mark.parametrize("seed", [901 + i for i in range(50)])
def test_phase_gadget_as_rzz(seed):
    """Add phase-gadget to two-qubit, which is RZZ, randomly.

    Check if
    - the number of gates is 1,
    - the gate is RZZ,
    - the control qubit is correct,
    - the target qubit is correct,
    - the angle is correct.
    """
    # Fix the seed for reproducibility.
    np.random.seed(seed)

    # Choose the number of qubits randomly between 2 and 100.
    num_qubits = np.random.randint(2, 101)
    qc = QuantumCircuit(num_qubits)
    # Randomem angle between 0 and 4pi
    angle = random.uniform(0, 4 * math.pi)
    # Add phase gadget, which is RZZ, for random two qubits.
    indices = np.random.choice(num_qubits, size=2, replace=False)
    qc.phase_gadget(angle, list(indices))

    # - the number of gates is 1,
    assert len(qc.gates) == 1
    # - the gate is RZZ,
    assert qc.gates[0].gate == ParametricTwoQubitGateType.RZZ
    # - the control qubit is correct,
    assert qc.gates[0].control == indices[0]
    # - the target qubit is correct,
    assert qc.gates[0].target == indices[1]
    # - the angle is correct.
    assert qc.gates[0].parameter.value == angle


@pytest.mark.parametrize("seed", [901 + i for i in range(50)])
def test_phase_gadget(seed):
    """Add phase-gagdet to more than two qubits randomly.

    Check if
    - the number of gates is correct,
    - the gates are CNOT...CNOT, RZ, CNOT...CNOT.
    """
    # Fix the seed for reproducibility.
    np.random.seed(seed)

    # Choose the number of qubits randomly between 3 and 100.
    num_qubits = np.random.randint(3, 101)
    qc = QuantumCircuit(num_qubits)
    # Randomem angle between 0 and 4pi
    angle = random.uniform(0, 4 * math.pi)
    # Add phase gadget, which is RZ...Z, for random qubits qubits.
    num_applied_qubits = np.random.randint(3, qc.num_qubits + 1)
    indices = np.random.choice(num_qubits, size=num_applied_qubits, replace=False)
    qc.phase_gadget(angle, list(indices))

    num_chain_cnots = num_applied_qubits - 1
    num_rz = 1
    num_reverse_chain_cnots = num_applied_qubits - 1
    num_gates = num_chain_cnots + num_rz + num_reverse_chain_cnots
    # - the number of gates is correct,
    assert len(qc.gates) == num_gates
    # - the gates are CNOT...CNOT, RZ, CNOT...CNOT.
    for gate in qc.gates[:num_chain_cnots]:
        assert gate.gate == TwoQubitGateType.CNOT
    assert qc.gates[num_chain_cnots].gate == ParametricSingleQubitGateType.RZ
    assert qc.gates[num_chain_cnots].parameter.value == angle
    for gate in qc.gates[num_chain_cnots + num_rz :]:
        assert gate.gate == TwoQubitGateType.CNOT


def test_phase_gadget_to_no_qubits():
    """Add phase-gagdet to zero qubit.

    Check if
    - ValueError is raised.
    """
    qc = QuantumCircuit(3)
    angle = random.uniform(0, 4 * math.pi)
    with pytest.raises(ValueError):
        qc.phase_gadget(angle, [])


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
