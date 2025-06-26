# File: tests/circuit/test_circuit.py
import itertools

import numpy as np
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
    ParameterExpression,
    Parameter,
    Value,
    Operator,
)
from tests.mock import UnsupportedGate


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


@pytest.mark.parametrize(
    "new_qubits_label",
    [
        {0: "Q0", 1: "Q1"},  # Same
        {0: "Q0", 1: "Q1", 2: "Q2"},  # Long
        {0: "Q0"},  # Short
    ],
)
def test_update_qubits_label(new_qubits_label):
    """Update the _qubits_label of a QuantumCircuit.

    Check if
    1. the _qubits_label is updated correctly.
    """
    qc = QuantumCircuit(2)
    qc.update_qubits_label(new_qubits_label)

    for index, qubit_label in enumerate(qc.qubits_label):
        if new_qubits_label.get(index, None) is not None:
            # 1. the _qubits_label is updated correctly.
            assert qubit_label == new_qubits_label[index]


def test_single_qubit_gates():
    """Add all single qubit gates to a QuantumCircuit.

    Check if
    1. the number of gates is 6,
    2. the first gate is a Hadamard gate,
    3. the second gate is an X gate,
    4. the third gate is a Y gate,
    5. the fourth gate is a Z gate,
    6. the fifth gate is an S gate,
    7. the sixth gate is a T gate,
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    qc.y(0)
    qc.z(0)
    qc.s(0)
    qc.t(0)
    # 1. the number of gates is 6,
    assert len(qc.gates) == 6
    # 2. the first gate is a Hadamard gate,
    assert qc.gates[0].gate == SingleQubitGateType.H
    # 3. the second gate is an X gate,
    assert qc.gates[1].gate == SingleQubitGateType.X
    # 4. the third gate is a Y gate,
    assert qc.gates[2].gate == SingleQubitGateType.Y
    # 5. the fourth gate is a Z gate,
    assert qc.gates[3].gate == SingleQubitGateType.Z
    # 6. the fifth gate is an S gate,
    assert qc.gates[4].gate == SingleQubitGateType.S
    # 7. the sixth gate is a T gate,
    assert qc.gates[5].gate == SingleQubitGateType.T


@pytest.mark.parametrize(
    "parameter", [int(1), float(1.0), Parameter("theta"), Value(1.5)]
)
def test_parametric_single_qubit_gates(parameter):
    """Add all parametric single qubit gates to a QuantumCircuit.

    Check if
    1. the number of gates is 3,
    2. the first gate is an RX gate with the given parameter,
    3. the second gate is an RY gate with the given parameter,
    4. the third gate is an RZ gate with the given parameter,
    5. the parameter of each gate is ParameterExpression,
    6. the parameter of each gate is the same as the given value if it is number.
    """
    qc = QuantumCircuit(1)
    qc.rx(parameter, 0)
    qc.ry(parameter, 0)
    qc.rz(parameter, 0)

    # 1. the number of gates is 3,
    assert len(qc.gates) == 3
    # 2. the first gate is an RX gate with the given parameter,
    assert qc.gates[0].gate == ParametricSingleQubitGateType.RX
    # 3. the second gate is an RY gate with the given parameter,
    assert qc.gates[1].gate == ParametricSingleQubitGateType.RY
    # 4. the third gate is an RZ gate with the given parameter,
    assert qc.gates[2].gate == ParametricSingleQubitGateType.RZ
    for gate in qc.gates:
        # 5. the parameter of each gate is ParameterExpression,
        assert isinstance(gate.parameter, ParameterExpression)

        # 6. the parameter of each gate is the same as the given value if the gate has Value.
        if isinstance(gate.parameter, Value):
            if isinstance(parameter, Value):
                assert gate.parameter.value == parameter.value
            else:
                assert gate.parameter.value == parameter


@pytest.mark.parametrize("control, target", [(0, 1), (1, 0), (0, 2), (2, 0)])
def test_two_qubit_gates(control, target):
    """Add all two qubit gates to a QuantumCircuit.

    Check if
    1. the number of gates is 3,
    2. the first gate is a CNOT gate,
    3. the second gat is a CNOT gate,
    4. the third gate is a CZ gate,
    5. the control and target qubits of each gate are as expected.
    """
    qc = QuantumCircuit(3)
    qc.cx(control, target)
    qc.cnot(control, target)
    qc.cz(control, target)

    # 1. the number of gates is 3,
    assert len(qc.gates) == 3
    # 2. the first gate is a CNOT gate,
    assert qc.gates[0].gate == TwoQubitGateType.CNOT
    # 3. the second gat is a CNOT gate,
    assert qc.gates[1].gate == TwoQubitGateType.CNOT
    # 4. the third gate is a CZ gate,
    assert qc.gates[2].gate == TwoQubitGateType.CZ
    # 5. the control and target qubits of each gate are as expected.
    for gate in qc.gates:
        assert gate.control == control
        assert gate.target == target


@pytest.mark.parametrize(
    "parameter", [int(1), float(1.0), Parameter("theta"), Value(1.5)]
)
@pytest.mark.parametrize("control, target", [(0, 1), (1, 2)])
def test_parametric_two_qubit_gates(parameter, control, target):
    """Add all parametric two qubit gates to a QuantumCircuit.

    Check if
    1. the number of gates is 6,
    2. the first gate is a CRX gate with the given parameter,
    3. the second gate is a CRY gate with the given parameter,
    4. the third gate is a CRZ gate with the given parameter,
    5. the fourth gate is an RXX gate with the given parameter,
    6. the fifth gate is an RYY gate with the given parameter,
    7. the sixth gate is an RZZ gate with the given parameter,
    8. the control and target qubits of each gate are as expected,
    9. the parameter of each gate is ParameterExpression,
    10. the parameter of each gate is the same as the given value if it is number.
    """
    qc = QuantumCircuit(3)
    qc.crx(parameter, control, target)
    qc.cry(parameter, control, target)
    qc.crz(parameter, control, target)
    qc.rxx(parameter, control, target)
    qc.ryy(parameter, control, target)
    qc.rzz(parameter, control, target)

    # 1. the number of gates is 6,
    assert len(qc.gates) == 6
    # 2. the first gate is a CRX gate with the given parameter,
    assert qc.gates[0].gate == ParametricTwoQubitGateType.CRX
    # 3. the second gate is a CRY gate with the given parameter,
    assert qc.gates[1].gate == ParametricTwoQubitGateType.CRY
    # 4. the third gate is a CRZ gate with the given parameter,
    assert qc.gates[2].gate == ParametricTwoQubitGateType.CRZ
    # 5. the fourth gate is an RXX gate with the given parameter,
    assert qc.gates[3].gate == ParametricTwoQubitGateType.RXX
    # 6. the fifth gate is an RYY gate with the given parameter,
    assert qc.gates[4].gate == ParametricTwoQubitGateType.RYY
    # 7. the sixth gate is an RZZ gate with the given parameter,
    assert qc.gates[5].gate == ParametricTwoQubitGateType.RZZ
    # 8. the control and target qubits of each gate are as expected,
    for gate in qc.gates:
        assert gate.control == control
        assert gate.target == target

        # 9. the parameter of each gate is ParameterExpression,
        assert isinstance(gate.parameter, ParameterExpression)

        # 10. the parameter of each gate is the same as the given value if it is number.
        if isinstance(gate.parameter, Value):
            print(parameter)
            print(type(parameter))
            if isinstance(parameter, Value):
                assert gate.parameter.value == parameter.value
            else:
                assert gate.parameter.value == parameter


@pytest.mark.parametrize(
    "control1, control2, target", [(0, 1, 2), (1, 2, 0), (0, 2, 4)]
)
def test_three_qubit_gate(control1, control2, target):
    """Add a three qubit gate to a QuantumCircuit.

    Check if
    1. the number of gates is 1,
    2. the gate is a CCX gate,
    3. the control1, control2 and target qubits are as expected.
    """
    qc = QuantumCircuit(5)
    qc.ccx(control1, control2, target)
    # 1. the number of gates is 1,
    assert len(qc.gates) == 1
    # 2. the gate is a CCX gate,
    assert qc.gates[0].gate == ThreeQubitGateType.CCX
    # 3. the control1, control2 and target qubits are as expected.
    for (
        gate
    ) in (
        qc.gates
    ):  # We don't need this for-loop as of now, when ksk-jij implemented this, but for the future, it might be useful.
        assert gate.control1 == control1
        assert gate.control2 == control2
        assert gate.target == target


def test_exp_evolution_manually():
    """Add a manually constructed parametric exp evolution gate to a QuantumCircuit.

    Check if
    1. the number of gates is 1,
    2. the gate is a ParametricExpGate,
    3. the parameter is as expected,
    4. the length of indices are as expected,
    5. the hamiltonian is as expected.
    """
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.exp_evolution(theta, hamiltonian)
    # 1. the number of gates is 1,
    assert len(qc.gates) == 1
    # 2. the gate is a ParametricExpGate,
    assert isinstance(qc.gates[0], ParametricExpGate)
    assert qc.gates[0].parameter == theta
    # 4. the length of indices are as expected,
    assert len(qc.gates[0].indices) == 2
    # 5. the hamiltonian is as expected.
    assert qc.gates[0].hamiltonian == hamiltonian

    hamiltonian2 = qm_o.Hamiltonian()
    hamiltonian2 += qm_o.X(0) * qm_o.Y(1) + qm_o.Z(0) * qm_o.X(1)
    qc2 = QuantumCircuit(2)
    qc2.exp_evolution(theta, hamiltonian)
    # 1. the number of gates is 1,
    assert len(qc2.gates) == 1
    # 2. the gate is a ParametricExpGate,
    assert isinstance(qc2.gates[0], ParametricExpGate)
    # 3. the parameter is as expected,
    assert qc2.gates[0].parameter == theta
    # 4. the length of indices are as expected,
    assert len(qc2.gates[0].indices) == 2
    # 5. the hamiltonian is as expected.
    assert qc2.gates[0].hamiltonian == hamiltonian


def test_invalid_exp_evolution():
    """Add an invalid exp evolution gate to a QuantumCircuit.

    Check if
    1. a ValueError is raised.
    """
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    # 1. a ValueError is raised.
    with pytest.raises(ValueError):
        qc.exp_evolution(theta, hamiltonian)


@pytest.mark.parametrize(
    "num_qubits, num_cbits, target_qubits, target_cbits",
    [
        (1, 1, [0], [0]),
        (1, 2, [0], [1]),
        (2, 1, [1], [0]),
        (2, 2, [0], [0]),
        (2, 2, [1], [1]),
        (2, 2, [0, 1], [0, 1]),
        (2, 2, [0, 1], [1, 0]),
    ],
)
def test_measurement(num_qubits, num_cbits, target_qubits, target_cbits):
    """Add measurements to a QuantumCircuit.

    Check if
    1. the number of gates is equal to the number of target qubits,
    2. the number of gates is equal to the number of target classical bits,
    3. all gates are MeasurementGate instances,
    4. each gate's qubit matches the corresponding target qubit,
    5. each gate's cbit matches the corresponding target cbit.
    """
    # Create a QuantumCircuit and add measurements.
    qc = QuantumCircuit(num_qubits, num_cbits)
    for target_qubit, target_cbit in zip(target_qubits, target_cbits):
        qc.measure(target_qubit, target_cbit)

    # 1. the number of gates is equal to the number of target qubits,
    assert len(qc.gates) == len(target_qubits)
    # 2. the number of gates is equal to the number of target classical bits,
    assert len(qc.gates) == len(target_cbits)
    # 3. all gates are MeasurementGate instances,
    assert all(isinstance(gate, MeasurementGate) for gate in qc.gates)
    for index, gate in enumerate(qc.gates):
        # 4. each gate's qubit matches the corresponding target qubit,
        assert gate.qubit == target_qubits[index]
        # 5. each gate's cbit matches the corresponding target cbit.
        assert gate.cbit == target_cbits[index]


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("num_cbits", [1, 2])
def test_measurement_invalid(num_qubits, num_cbits):
    """Add measurements to a QuantumCircuit with invalid indices.

    Check if
    1. a ValueError is raised when trying to measure a qubit that does not exist,
    2. a ValueError is raised when trying to measure a classical bit that does not exist.
    """
    qc = QuantumCircuit(num_qubits, num_cbits)

    # 1. a ValueError is raised when trying to measure a qubit that does not exist,
    with pytest.raises(ValueError):
        qc.measure(num_qubits, 0)
    # 2. a ValueError is raised when trying to measure a classical bit that does not exist.
    with pytest.raises(ValueError):
        qc.measure(0, num_cbits)


@pytest.mark.parametrize("num_qubits", [1, 2])
def test_measure_all_with_less_cbits(num_qubits):
    """Call measure_all on a QuantumCircuit without classical bits.

    Check if
    1. there is num_cbits classical bits in the circuit,
    2. the number of gates is equal to the number of qubits,
    3. the number of classical bits is equal to the number of qubits after measure_all is called,
    4. all gates are MeasurementGate instances,
    5. each gate's qubit matches its index,
    6. each gate's cbit matches its index.
    """
    for num_cbits in range(num_qubits):
        qc = QuantumCircuit(num_qubits, num_cbits)

        # 1. there is num_cbits classical bits in the circuit,
        assert qc.num_clbits == num_cbits

        qc.measure_all()

        # 2. the number of gates is equal to the number of qubits,
        assert len(qc.gates) == num_qubits
        # 3. the number of classical bits is equal to the number of qubits after measure_all is called,
        assert qc.num_clbits == num_qubits
        # 4. all gates are MeasurementGate instances,
        assert all(isinstance(gate, MeasurementGate) for gate in qc.gates)
        for index, gate in enumerate(qc.gates):
            # 5. each gate's qubit matches its index,
            assert gate.qubit == index
            # 6. each gate's cbit matches its index.
            assert gate.cbit == index


@pytest.mark.parametrize("num_qubits", [1, 2])
def test_measure_all_with_more_or_eqaul_cbits(num_qubits):
    """Call measure_all on a QuantumCircuit with more or equal classical bits than qubits.

    Check if
    1. there is num_cbits classical bits in the circuit,
    2. the number of gates is equal to the number of qubits,
    3. all gates are MeasurementGate instances,
    4. each gate's qubit matches its index,
    5. each gate's cbit matches its index.
    """
    for num_cbits in range(num_qubits, num_qubits + 2):
        qc = QuantumCircuit(num_qubits, num_cbits)

        # 1. there is num_cbits classical bits in the circuit,
        assert qc.num_clbits == num_cbits

        qc.measure_all()

        # 2. the number of gates is equal to the number of qubits,
        assert len(qc.gates) == num_qubits
        # 3. all gates are MeasurementGate instances,
        assert all(isinstance(gate, MeasurementGate) for gate in qc.gates)
        for index, gate in enumerate(qc.gates):
            # 4. each gate's qubit matches its index,
            assert gate.qubit == index
            # 5. each gate's cbit matches its index.
            assert gate.cbit == index


def test_add_gate_to_invalid_qubit():
    """Add a gate to a QuantumCircuit with an invalid qubit index.

    Check if
    1. a ValueError is raised.
    """
    num_qubits = 1
    invalid_qubit = num_qubits + 1
    num_cbits = 1
    invalid_cbit = num_cbits + 1
    qc = QuantumCircuit(num_qubits, num_cbits)
    # SingleQubitGate
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.x(invalid_qubit)
    # ParametricSingleQubitGate
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.rx(Parameter("theta"), invalid_qubit)
    # TwoQubitGate
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.cx(0, invalid_qubit)
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.cx(invalid_qubit, 0)
    # ParametricTwoQubitGate
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.crx(Parameter("theta"), 0, invalid_qubit)
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.crx(Parameter("theta"), invalid_qubit, 0)
    # ThreeQubitGate
    with pytest.raises(ValueError):
        qc.ccx(0, invalid_qubit, invalid_qubit + 1)
    with pytest.raises(ValueError):
        qc.ccx(invalid_qubit + 1, 0, invalid_qubit)
    with pytest.raises(ValueError):
        qc.ccx(invalid_qubit, invalid_qubit + 1, 0)
    # ParametricExpGate
    hamiltonian = qm_o.Hamiltonian(num_qubits=invalid_qubit)
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.exp_evolution(Parameter("theta"), hamiltonian)
    # Operator
    # TODO: Add tests according to how we fix issue #187.
    qc_operator = QuantumCircuit(invalid_qubit)
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.add_gate(Operator(qc_operator))
    # MeasurementGate
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.measure(invalid_qubit, 0)
    with pytest.raises(ValueError):
        # 1. a ValueError is raised.
        qc.measure(0, invalid_cbit)


def test_repr():
    """Call __repr__ on a QuantumCircuit.

    Check if
    1. the representation is f"[{self.gates}]".
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(0)
    qc.y(0)
    qc.z(0)
    qc.s(0)
    qc.t(0)
    qc.rx(1.0, 0)
    qc.ry(1.0, 0)
    qc.rz(1.0, 0)
    qc.cx(0, 1)
    qc.cnot(0, 1)
    qc.cz(0, 1)
    qc.crx(1.0, 0, 1)
    qc.cry(1.0, 0, 1)
    qc.crz(1.0, 0, 1)
    qc.rxx(1.0, 0, 1)
    qc.ryy(1.0, 0, 1)
    qc.rzz(1.0, 0, 1)
    qc.ccx(0, 1, 2)
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Y(1) * qm_o.Z(2)
    qc.exp_evolution(1.0, hamiltonian)
    qc.measure_all()

    expected_repr = f"{qc.gates}"

    assert repr(qc) == expected_repr


def test_circuit_append_manually():
    """Run append on a QuantumCircuit with a manually created Operator.

    Check if
    1. the number of gates is 1,
    2. the gate is an Operator,
    3. the Operator contains the original circuit.
    """
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cnot(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.append(qc1)
    # 1. the number of gates is 1,
    assert len(qc2.gates) == 1
    # 2. the gate is an Operator,
    assert isinstance(qc2.gates[0], Operator)
    # 3. the Operator contains the original circuit.
    assert qc2.gates[0].circuit == qc1


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


# <<< QuantumCircuit <<<


# >>> Operator >>>
def test_operator_creation_default_manually():
    """Create an Operator with default parameters.

    Check if
    1. the circuit is the same as the given circuit,
    2. the label is None,
    """
    qc = QuantumCircuit(2)
    operator = Operator(qc)
    # 1. the circuit is the same as the given circuit,
    assert operator.circuit == qc
    # 2. the name is None,
    assert operator.label is None

    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure_all()
    operator = Operator(qc)
    # 1. the circuit is the same as the given circuit,
    assert operator.circuit == qc
    # 2. the name is None,
    assert operator.label is None


@pytest.mark.parametrize("label", ["test", "Q!U!A!N!T!U!M!O!P!E!R!A!T!O!R!", None])
def test_operator_creation_with_label(label):
    """Create an Operator with default parameters.

    Check if
    1. the circuit is the same as the given circuit,
    2. the label is the given label,
    """
    qc = QuantumCircuit(2)
    operator = Operator(circuit=qc, label=label)
    # 1. the circuit is the same as the given circuit,
    assert operator.circuit == qc
    # 2. the label is the given label,
    assert operator.label == label

    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure_all()
    operator = Operator(circuit=qc, label=label)
    # 1. the circuit is the same as the given circuit,
    assert operator.circuit == qc
    # 2. the label is the given label,
    assert operator.label == label


def test_operated_qubits():
    """Run Operator.operated_qubits on a QuantumCircuit with various gates.

    Check if
    1. the operated qubits are as expected.
    """
    np.random.seed(901)  # For reproducibility

    num_qubits = 10

    num_attempts = 50
    for _ in range(num_attempts):
        qubits = np.random.randint(0, num_qubits, size=27)

        qc = QuantumCircuit(num_qubits)
        qc.h(qubits[0])
        qc.x(qubits[1])
        qc.y(qubits[2])
        qc.z(qubits[3])
        qc.s(qubits[4])
        qc.t(qubits[5])
        qc.cx(qubits[6], qubits[7])
        qc.cz(qubits[8], qubits[9])
        qc.crx(Parameter("theta"), qubits[10], qubits[11])
        qc.cry(Parameter("phi"), qubits[12], qubits[13])
        qc.crz(Parameter("gamma"), qubits[14], qubits[15])
        qc.rxx(Parameter("delta"), qubits[16], qubits[17])
        qc.ryy(Parameter("epsilon"), qubits[18], qubits[19])
        qc.rzz(Parameter("zeta"), qubits[20], qubits[21])
        qc.ccx(qubits[22], qubits[23], qubits[24])
        qc.exp_evolution(Parameter("omega"), qm_o.X(qubits[25]))

        qc_operator = QuantumCircuit(num_qubits)
        qc_operator.h(qubits[26])
        qc.append(qc_operator)

        operated_qubit = set(Operator(qc).operated_qubits())
        expected_qubits = set(qubits)
        # 1. the operated qubits are as expected.
        assert operated_qubit == expected_qubits


def test_operated_qubits_with_unsupported_gate():
    """Run Operator.operated_qubits on a QuantumCircuit with an unsupported gate.

    Check if
    1. a ValueError is raised.
    """
    qc = QuantumCircuit(1)
    qc.append(UnsupportedGate())

    with pytest.raises(ValueError):
        Operator(qc).operated_qubits()


# <<< Operator <<<
