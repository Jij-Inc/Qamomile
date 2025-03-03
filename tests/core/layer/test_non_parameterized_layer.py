import pytest

import qamomile.core.circuit as qm_c
from qamomile.core.layer.non_parameterized_layer import (EntanglementLayer,
                                                         SuperpositionLayer)


def test_entanglement_layer_linear():
    num_qubits = 4
    layer = EntanglementLayer(num_qubits, "linear")
    circuit = layer.get_circuit()
    
    assert circuit.name == "EntanglementLayer"
    assert circuit.num_qubits == num_qubits
    print(circuit.gates)
    for i in range(num_qubits - 1):
        assert circuit.gates[i] == qm_c.TwoQubitGate(qm_c.TwoQubitGateType.CNOT, i, i + 1)

def test_entanglement_layer_full():
    num_qubits = 3
    layer = EntanglementLayer(num_qubits, "full")
    circuit = layer.get_circuit()
    
    assert circuit.name == "EntanglementLayer"
    assert circuit.num_qubits == num_qubits * (num_qubits - 1) // 2
    index = 0
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            assert circuit.gates[index] == qm_c.TwoQubitGate(qm_c.TwoQubitGateType.CNOT, i, j)
            index += 1

def test_entanglement_layer_circular():
    num_qubits = 4
    layer = EntanglementLayer(num_qubits, "circular")
    circuit = layer.get_circuit()
    
    assert circuit.name == "EntanglementLayer"
    assert circuit.num_qubits == num_qubits
    for i in range(num_qubits):
        assert circuit.gates[i] == qm_c.TwoQubitGate(qm_c.TwoQubitGateType.CNOT, i, (i + 1) % num_qubits)

def test_entanglement_layer_reverse_linear():
    num_qubits = 4
    layer = EntanglementLayer(num_qubits, "reverse_linear")
    circuit = layer.get_circuit()
    
    assert circuit.name == "EntanglementLayer"
    assert circuit.num_qubits == num_qubits 
    for i in range(num_qubits - 1, 0, -1):
        assert circuit.gates[num_qubits - 1 - i] == qm_c.TwoQubitGate(qm_c.TwoQubitGateType.CNOT, i, i - 1)

def test_entanglement_layer_invalid_type():
    with pytest.raises(ValueError):
        EntanglementLayer(4, "invalid_type")

def test_superposition_layer():
    num_qubits = 4
    layer = SuperpositionLayer(num_qubits)
    circuit = layer.get_circuit()
    
    assert circuit.name == "SuperpositionLayer"
    assert circuit.num_qubits == num_qubits
    for i in range(num_qubits):
        assert circuit.gates[i] == qm_c.SingleQubitGate(qm_c.SingleQubitGateType.H, i)