# File: tests/circuit/test_circuit.py

import pytest
import random
import math
import qamomile.core.operator as qm_o
from qamomile.core.circuit import (
    SingleQubitGateType,
    SingleQubitGate,
    ParametricSingleQubitGateType,
    ParametricSingleQubitGate,
    TwoQubitGateType,
    TwoQubitGate,
    ParametricTwoQubitGateType,
    ParametricTwoQubitGate,
    ThreeQubitGateType,
    ThreeQubitGate,
    MeasurementGate,
    ParametricExpGate,
    QuantumCircuit,
    Parameter,
    Value,
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
@pytest.mark.parametrize("qubit", [0, 1])
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


# >>> ParametricSingleQubitGate >>>
def test_parametric_single_qubit_gate_type():
    """Verify if all parametric single qubit gates are present in the ParametricSingleQubitGateType enum.

    Check if
    1. ParametricSingleQubitGateType.RX exists,
    2. ParametricSingleQubitGateType.RY exists,
    3. ParametricSingleQubitGateType.RZ exists,
    4. The length of ParametricSingleQubitGateType is 3.
    """
    # 1. ParametricSingleQubitGateType.RX exists,
    assert ParametricSingleQubitGateType.RX
    # 2. ParametricSingleQubitGateType.RY exists,
    assert ParametricSingleQubitGateType.RY
    # 3. ParametricSingleQubitGateType.RZ exists,
    assert ParametricSingleQubitGateType.RZ
    # 4. The length of ParametricSingleQubitGateType is 3.
    assert len(ParametricSingleQubitGateType) == 3


@pytest.mark.parametrize(
    "gate",
    [
        parametric_single_qubit_gate_type
        for parametric_single_qubit_gate_type in ParametricSingleQubitGateType
    ],
)
@pytest.mark.parametrize("qubit", [0, 1])
@pytest.mark.parametrize("parameter", [Parameter("theta"), Value(1.5)])
def test_parametric_single_qubit_gate(gate, qubit, parameter):
    """Create an instance of ParametricSingleQubitGate.

    Check if
    1. its gate is the same as the given gate,
    2. its qubit is the same as the given qubit,
    3. its parameter is the same as the given parameter.
    """
    parametric_single_qubit_gate = ParametricSingleQubitGate(gate, qubit, parameter)

    # 1. its gate is the same as the given gate,
    assert parametric_single_qubit_gate.gate == gate
    # 2. its qubit is the same as the given qubit.
    assert parametric_single_qubit_gate.qubit == qubit
    # 3. its parameter is the same as the given parameter.
    assert parametric_single_qubit_gate.parameter == parameter


# <<< ParametricSingleQubitGate <<<


# >>> TwoQubitGate >>>
def test_two_qubit_gate_type():
    """Verify if all two qubit gates are present in the TwoQubitGateType enum.

    Check if
    1. TwoQubitGateType.CNOT exists,
    2. TwoQubitGateType.CZ exists,
    3. The length of TwoQubitGateType is 2.
    """
    # 1. TwoQubitGateType.CNOT exists,
    assert TwoQubitGateType.CNOT
    # 2. TwoQubitGateType.CZ exists,
    assert TwoQubitGateType.CZ
    # 3. The length of TwoQubitGateType is 2.
    assert len(TwoQubitGateType) == 2


@pytest.mark.parametrize(
    "gate",
    [two_qubit_gate_type for two_qubit_gate_type in TwoQubitGateType],
)
@pytest.mark.parametrize("control", [0, 1])
@pytest.mark.parametrize("target", [2, 3])
def test_two_qubit_gate(gate, control, target):
    """Create an instance of TwoQubitGate.

    Check if
    1. its gate is the same as the given gate,
    2. its control is the same as the given control,
    3. its target is the same as the given target.
    """
    two_qubit_gate = TwoQubitGate(gate, control, target)

    # 1. its gate is the same as the given gate,
    assert two_qubit_gate.gate == gate
    # 2. its control is the same as the given control.
    assert two_qubit_gate.control == control
    # 3. its target is the same as the given target.
    assert two_qubit_gate.target == target


# <<< TwoQubitGate <<<


# >>> ParametricTwoQubitGate >>>
def test_parametric_two_qubit_gate_type():
    """Verify if all parametric two qubit gates are present in the ParametricTwoQubitGateType enum.

    Check if
    1. ParametricTwoQubitGateType.CRX exists,
    2. ParametricTwoQubitGateType.CRY exists,
    3. ParametricTwoQubitGateType.CRZ exists,
    4. ParametricTwoQubitGateType.RXX exists,
    5. ParametricTwoQubitGateType.RYY exists,
    6. ParametricTwoQubitGateType.RZZ exists,
    7. The length of ParametricTwoQubitGateType is 6.
    """
    # 1. ParametricTwoQubitGateType.CRX exists,
    assert ParametricTwoQubitGateType.CRX
    # 2. ParametricTwoQubitGateType.CRY exists,
    assert ParametricTwoQubitGateType.CRY
    # 3. ParametricTwoQubitGateType.CRZ exists,
    assert ParametricTwoQubitGateType.CRZ
    # 4. ParametricTwoQubitGateType.RXX exists,
    assert ParametricTwoQubitGateType.RXX
    # 5. ParametricTwoQubitGateType.RYY exists,
    assert ParametricTwoQubitGateType.RYY
    # 6. ParametricTwoQubitGateType.RZZ exists,
    assert ParametricTwoQubitGateType.RZZ
    # 7. The length of ParametricTwoQubitGateType is 6.
    assert len(ParametricTwoQubitGateType) == 6


@pytest.mark.parametrize(
    "gate",
    [
        parametric_two_qubit_gate_type
        for parametric_two_qubit_gate_type in ParametricTwoQubitGateType
    ],
)
@pytest.mark.parametrize("control", [0, 1])
@pytest.mark.parametrize("target", [2, 3])
@pytest.mark.parametrize("parameter", [Parameter("theta"), Value(1.5)])
def test_parametric_two_qubit_gate(gate, control, target, parameter):
    """Create an instance of ParametricTwoQubitGate.

    Check if
    1. its gate is the same as the given gate,
    2. its control is the same as the given control,
    3. its target is the same as the given target,
    4. its parameter is the same as the given parameter.
    """
    parametric_two_qubit_gate = ParametricTwoQubitGate(gate, control, target, parameter)

    # 1. its gate is the same as the given gate,
    assert parametric_two_qubit_gate.gate == gate
    # 2. its control is the same as the given control.
    assert parametric_two_qubit_gate.control == control
    # 3. its target is the same as the given target.
    assert parametric_two_qubit_gate.target == target
    # 4. its parameter is the same as the given parameter.
    assert parametric_two_qubit_gate.parameter == parameter


# <<< ParametricTwoQubitGate <<<


# >>> ThreeQubitGate >>>
def test_three_qubit_gate_type():
    """Verify if all three qubit gates are present in the ThreeQubitGateType enum.

    Check if
    1. ThreeQubitGateType.CCX exists,
    2. The length of ThreeQubitGateType is 1.
    """
    # 1. ThreeQubitGateType.CCX exists,
    assert ThreeQubitGateType.CCX
    # 2. The length of ThreeQubitGateType is 1.
    assert len(ThreeQubitGateType) == 1


@pytest.mark.parametrize(
    "gate",
    [three_qubit_gate_type for three_qubit_gate_type in ThreeQubitGateType],
)
@pytest.mark.parametrize("control1", [0, 1])
@pytest.mark.parametrize("control2", [2, 3])
@pytest.mark.parametrize("target", [4, 5])
def test_three_qubit_gate(gate, control1, control2, target):
    """Create an instance of ThreeQubitGate.

    Check if
    1. its gate is the same as the given gate,
    2. its control1 is the same as the given control1,
    3. its control2 is the same as the given control2,
    4. its target is the same as the given target.
    """
    three_qubit_gate = ThreeQubitGate(gate, control1, control2, target)

    # 1. its gate is the same as the given gate,
    assert three_qubit_gate.gate == gate
    # 2. its control1 is the same as the given control1.
    assert three_qubit_gate.control1 == control1
    # 3. its control2 is the same as the given control2.
    assert three_qubit_gate.control2 == control2
    # 4. its target is the same as the given target.
    assert three_qubit_gate.target == target


# <<< ThreeQubitGate <<<


# >>> ParametricExpGate >>>
@pytest.mark.parametrize("parameter", [Parameter("theta"), Value(1.5)])
def test_parametric_exp_gate(parameter):
    """Create an instance of ParametricExpGate.

    Check if
    1. its hamiltonian is the same as the given hamiltonian,
    2. its parameter is the same as the given parameter,
    3. its indices is the same as the given indices.
    """
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    indices = [0, 1]
    parametric_exp_gate = ParametricExpGate(hamiltonian, parameter, indices)

    # 1. its hamiltonian is the same as the given hamiltonian,
    assert parametric_exp_gate.hamiltonian == hamiltonian
    # 2. its parameter is the same as the given parameter.
    assert parametric_exp_gate.parameter == parameter
    # 3. its indices is the same as the given indices.
    assert parametric_exp_gate.indices == indices


# <<< ParametricExpGate <<<


# >>> MeasurementGate >>>
@pytest.mark.parametrize("qubit", [0, 1])
@pytest.mark.parametrize("cbit", [2, 3])
def test_measurement_gate(qubit, cbit):
    """Create an instance of MeasurementGate.

    Check if
    1. its qubit is the same as the given qubit,
    2. its cbit is the same as the given cbit.
    """
    measurement_gate = MeasurementGate(qubit, cbit)

    # 1. its qubit is the same as the given qubit,
    assert measurement_gate.qubit == qubit
    # 2. its cbit is the same as the given cbit.
    assert measurement_gate.cbit == cbit


# <<< MeasurementGate <<<


# >>> QuantumCircuit >>>
@pytest.mark.parametrize("num_qubits", [0, 1, 2])
def test_quantum_circuit_creation_default(num_qubits):
    """Create a QuantumCircuit with a given number of qubits.

    Check if
    1. the number of qubits is as expected,
    2. the number of classical bits is zero according to the default constructor,
    3. the name of the circuit is None according to the default constructor,
    4. the gates list is empty,
    5. the _qubits_label is a list whose length is equal to the number of qubits,
    6. the lement of the _qubits_label is "q{0}", "q{1}", etc,
    7. the qubits_label returns _qubits_label.
    """
    qc = QuantumCircuit(num_qubits)
    # 1. the number of qubits is as expected,
    assert qc.num_qubits == num_qubits
    # 2. the number of classical bits is zero according to the default constructor,
    assert qc.num_clbits == 0
    # 3. the name of the circuit is None according to the default constructor,
    assert qc.name is None
    # 4. the gates list is empty,
    assert len(qc.gates) == 0
    # 5. the _qubits_label is a list whose length is equal to the number of qubits,
    assert len(qc.qubits_label) == num_qubits
    # 6. the lement of the _qubits_label is "q{0}", "q{1}", etc,
    for i in range(num_qubits):
        assert qc.qubits_label[i] == f"q_{{{i}}}"
    # 7. the qubits_label returns _qubits_label.
    assert qc.qubits_label == qc._qubits_label


@pytest.mark.parametrize("num_qubits", [0, 1, 2])
@pytest.mark.parametrize("num_cbits", [0, 1])
@pytest.mark.parametrize("name", ["test", "Q!U!A!N!T!U!M!C!I!R!C!U!I!T!", None])
def test_quantum_circuit_creation(num_qubits, num_cbits, name):
    """Create a QuantumCircuit with a given number of qubits.

    Check if
    1. the number of qubits is as expected,
    2. the number of classical bits is as expected,
    3. the name is as expected,
    4. the gates list is empty,
    5. the _qubits_label is a list whose length is equal to the number of qubits,
    6. the lement of the _qubits_label is "q{0}", "q{1}", etc,
    7. the qubits_label returns _qubits_label.
    """
    qc = QuantumCircuit(num_qubits=num_qubits, num_clbits=num_cbits, name=name)

    # 1. the number of qubits is as expected,
    assert qc.num_qubits == num_qubits
    # 2. the number of classical bits is as expected,
    assert qc.num_clbits == num_cbits
    # 3. the name is as expected,
    assert qc.name == name
    # 4. the gates list is empty,
    assert len(qc.gates) == 0
    # 5. the _qubits_label is a list whose length is equal to the number of qubits,
    assert len(qc.qubits_label) == num_qubits
    # 6. the lement of the _qubits_label is "q{0}", "q{1}", etc.
    for i in range(num_qubits):
        assert qc.qubits_label[i] == f"q_{{{i}}}"
    # 7. the qubits_label returns _qubits_label.
    assert qc.qubits_label == qc._qubits_label


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
