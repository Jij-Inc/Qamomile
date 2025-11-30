import pytest

from qamomile.core.ansatz.custom_ansatz import CustomAnsatz
from qamomile.core.layer.layer import Layer
from qamomile.core.layer.non_parameterized_layer import EntanglementLayer
from qamomile.core.layer.parameterized_layer import (CostLayer, MixerLayer,
                                                     ParameterizedLayer,
                                                     RotationLayer)


class DummyLayer(Layer):
    def get_circuit(self):
        pass

def test_custom_ansatz_init():
    num_qubits = 4
    layers = [DummyLayer(), DummyLayer()]
    reps = 2
    use_common_parameter_context = False

    ansatz = CustomAnsatz(num_qubits, layers, reps, use_common_parameter_context)

    assert ansatz.num_qubits == num_qubits
    assert ansatz.layers == layers
    assert ansatz.reps == reps
    assert ansatz.use_common_parameter_context == use_common_parameter_context

def test_custom_ansatz_init_default_reps():
    num_qubits = 4
    layers = [DummyLayer(), DummyLayer()]

    ansatz = CustomAnsatz(num_qubits, layers)

    assert ansatz.num_qubits == num_qubits
    assert ansatz.layers == layers
    assert ansatz.reps == 1
    assert ansatz.use_common_parameter_context == True

def test_custom_ansatz_efficient_su2():
    num_qubits = 4
    reps = 2
    layers = [
        RotationLayer(num_qubits, rotation_type="ry"),
        RotationLayer(num_qubits, rotation_type="rz"),
        EntanglementLayer(num_qubits,entangle_type="linear"),
    ]
    custom_ansatz = CustomAnsatz(num_qubits, layers, reps= reps)
    handmade_su2 = custom_ansatz.get_circuit()
    assert handmade_su2.num_qubits == num_qubits
    assert handmade_su2.name == "CustomAnsatz"
    assert len(handmade_su2.gates) == len(layers) * reps
    assert len(handmade_su2.get_parameters()) == num_qubits * 2 * reps
    assert len(handmade_su2.gates[0].circuit.gates) == num_qubits
    assert len(handmade_su2.gates[1].circuit.gates) == num_qubits
    assert len(handmade_su2.gates[2].circuit.gates) == num_qubits - 1
    assert len(handmade_su2.gates[3].circuit.gates) == num_qubits 
    assert len(handmade_su2.gates[4].circuit.gates) == num_qubits
    assert len(handmade_su2.gates[5].circuit.gates) == num_qubits - 1

def test_custom_ansatz_commonparameters():
    num_qubits = 4
    reps = 2
    layers = [
        RotationLayer(num_qubits, rotation_type="ry", symbol="theta"),
        RotationLayer(num_qubits, rotation_type="rz", symbol="phi"),
        EntanglementLayer(num_qubits,entangle_type="linear"),
    ]
    custom_ansatz = CustomAnsatz(num_qubits, layers, reps= reps, use_common_parameter_context=False)
    handmade_su2 = custom_ansatz.get_circuit()
    assert handmade_su2.num_qubits == num_qubits
    assert handmade_su2.name == "CustomAnsatz"
    assert len(handmade_su2.gates) == len(layers) * reps
    assert len(handmade_su2.get_parameters()) == 2 * num_qubits
