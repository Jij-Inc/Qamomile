import pytest

import qamomile.core.circuit as qm_c
from qamomile.core.ansatz.predefined.efficient_su2 import EfficientSU2Ansatz


def test_build_default_parameters():
    reps = 1
    num_qubits = 3
    ansatz = EfficientSU2Ansatz(num_qubits=num_qubits, reps=reps)
    circuit = ansatz.build()
    assert isinstance(circuit, qm_c.QuantumCircuit)
    assert circuit.num_qubits == num_qubits
    assert circuit.name == "EfficientSU2"
    assert len(circuit.get_parameters()) == 6 * reps + 6
    assert len(circuit.gates) == 5 # 4 Rotation gates + 1 entanglement gate
    assert len(circuit.gates[0].circuit.gates) == num_qubits
    assert len(circuit.gates[1].circuit.gates) == num_qubits
    assert len(circuit.gates[2].circuit.gates) == num_qubits - 1
    assert len(circuit.gates[3].circuit.gates) == num_qubits
    assert len(circuit.gates[4].circuit.gates) == num_qubits

def test_build_custom_rotation_types():
    reps = 1
    num_qubits = 3
    ansatz = EfficientSU2Ansatz(num_qubits= num_qubits, rotation_types=["rx", "ry"], reps=reps)
    circuit = ansatz.build()
    assert isinstance(circuit, qm_c.QuantumCircuit)
    assert circuit.num_qubits == 3
    assert circuit.name == "EfficientSU2"
    assert len(circuit.get_parameters()) == 6 * reps + 6
    assert len(circuit.gates) == 5
    assert len(circuit.gates[0].circuit.gates) == num_qubits
    assert len(circuit.gates[1].circuit.gates) == num_qubits
    assert len(circuit.gates[2].circuit.gates) == num_qubits - 1
    assert len(circuit.gates[3].circuit.gates) == num_qubits
    assert len(circuit.gates[4].circuit.gates) == num_qubits

def test_build_custom_entanglement():
    reps = 1
    num_qubits = 3
    ansatz = EfficientSU2Ansatz(num_qubits=num_qubits, entanglement="full", reps=reps)
    circuit = ansatz.build()
    assert isinstance(circuit, qm_c.QuantumCircuit)
    assert circuit.num_qubits == 3
    assert circuit.name == "EfficientSU2"
    assert len(circuit.get_parameters()) == 6 * reps + 6
    assert len(circuit.gates) == 5
    assert len(circuit.gates[0].circuit.gates) == num_qubits
    assert len(circuit.gates[1].circuit.gates) == num_qubits
    assert len(circuit.gates[2].circuit.gates) == num_qubits * (num_qubits - 1) // 2
    assert len(circuit.gates[3].circuit.gates) == num_qubits
    assert len(circuit.gates[4].circuit.gates) == num_qubits


def test_build_skip_final_rotation_layer():
    reps = 1
    num_qubits = 3
    ansatz = EfficientSU2Ansatz(num_qubits= num_qubits, skip_final_rotation_layer=True, reps=reps)
    circuit = ansatz.build()
    assert isinstance(circuit, qm_c.QuantumCircuit)
    assert circuit.num_qubits == num_qubits
    assert circuit.name == "EfficientSU2"
    assert len(circuit.get_parameters()) == 6 * reps
    assert len(circuit.gates) == (2 + 1) * reps
    assert len(circuit.gates[0].circuit.gates) == num_qubits
    assert len(circuit.gates[1].circuit.gates) == num_qubits
    assert len(circuit.gates[2].circuit.gates) == num_qubits - 1


def test_build_multiple_reps():
    reps = 2
    num_qubits = 3
    ansatz = EfficientSU2Ansatz(num_qubits=num_qubits, reps=reps)
    circuit = ansatz.build()
    assert isinstance(circuit, qm_c.QuantumCircuit)
    assert circuit.num_qubits == num_qubits
    assert circuit.name == "EfficientSU2"
    assert len(circuit.get_parameters()) == 6 * reps + 6
    assert len(circuit.gates) == (2 + 1) * reps + 2
    


def test_build_invalid_rotation_type():
    with pytest.raises(ValueError):
        EfficientSU2Ansatz(num_qubits=3, rotation_types=["invalid"]).build()