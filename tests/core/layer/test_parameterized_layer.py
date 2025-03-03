import pytest

import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.circuit import (Parameter, ParameterExpression,
                                   QuantumCircuit, Value)
from qamomile.core.layer.parameter_context import ParameterContext
from qamomile.core.layer.parameterized_layer import (CostLayer, MixerLayer,
                                                     ParameterizedLayer,
                                                     RotationLayer)
from qamomile.core.operator import Hamiltonian


def test_rotation_layer_initialization():
    num_qubits = 3
    rotation_type = "rx"
    layer = RotationLayer(num_qubits=num_qubits, rotation_type=rotation_type)

    assert layer.num_qubits == num_qubits
    assert layer.rotation_type == rotation_type
    assert len(layer.params) == num_qubits
    assert all(isinstance(param, Parameter) for param in layer.params)

def test_rotation_layer_get_circuit():
    num_qubits = 3
    rotation_type = "ry"
    layer = RotationLayer(num_qubits=num_qubits, rotation_type=rotation_type)
    circuit = layer.get_circuit()

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.name == "RotationLayer: ry"
    for i in range(num_qubits):
        assert circuit.gates[i] == qm_c.ParametricSingleQubitGate(qm_c.ParametricSingleQubitGateType.RY, i, layer.params[i])

def test_cost_layer_initialization():
    hamiltonian = qm_o.Z(0) * qm_o.Y(1)
    layer = CostLayer(hamiltonian=hamiltonian)

    assert layer.num_qubits == hamiltonian.num_qubits
    assert layer.hamiltonian == hamiltonian
    assert len(layer.params) == 1
    assert isinstance(layer.params[0], Parameter)

def test_cost_layer_get_circuit():
    hamiltonian = qm_o.Z(0) * qm_o.Y(1)
    layer = CostLayer(hamiltonian=hamiltonian)
    circuit = layer.get_circuit()

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.name == "CostMixerLayer"
    assert circuit.gates[0] == qm_c.ParametricExpGate(hamiltonian, layer.params[0], [0,1])

def test_mixer_layer_initialization():
    num_qubits = 4
    mixer_type = "z"
    layer = MixerLayer(num_qubits=num_qubits, mixer_type=mixer_type)

    assert layer.num_qubits == num_qubits
    assert layer.mixer_type == mixer_type
    assert len(layer.params) == 1
    assert isinstance(layer.params[0], Parameter)

def test_mixer_layer_get_circuit():
    num_qubits = 4
    mixer_type = "x"
    layer = MixerLayer(num_qubits=num_qubits, mixer_type=mixer_type)
    circuit = layer.get_circuit()

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.name == "MixerLayer"
    for i in range(num_qubits):
        assert circuit.gates[i] == qm_c.ParametricSingleQubitGate(qm_c.ParametricSingleQubitGateType.RX, i, layer.params[0])

def test_parameterized_layer_with_custom_params():
    params = [Value(0.5), Value(1.5)]
    num_params = len(params)
    layer = RotationLayer(num_qubits=num_params, rotation_type="rz", params=params)

    assert len(layer.params) == num_params
    assert all(isinstance(param, Value) for param in layer.params)
    for i in range(num_params):
        assert layer.params[i] == params[i]

    circuit = layer.get_circuit()
    for i in range(num_params):
        assert circuit.gates[i] == qm_c.ParametricSingleQubitGate(qm_c.ParametricSingleQubitGateType.RZ, i, layer.params[i])
    


def test_parameterized_layer_with_invalid_params():
    num_params = 2
    params = [Value(0.5)]

    with pytest.raises(ValueError):
        RotationLayer(num_qubits=num_params, rotation_type="rz", params=params)

def test_parameterized_layer_with_parameter_context():
    num_params = 3
    context = ParameterContext()
    layer = RotationLayer(num_qubits=num_params, rotation_type="rx", parameter_context=context, symbol="alpha")
    print(layer.params)
    assert len(layer.params) == num_params
    assert all(isinstance(param, Parameter) for param in layer.params)
    assert layer.params[0].name == "alpha_{0}"
    assert layer.params[1].name == "alpha_{1}"
    assert layer.params[2].name == "alpha_{2}"

def test_set_parameter_context():
    num_params = 3
    context = ParameterContext()
    layer = RotationLayer(num_qubits=num_params, rotation_type="rx", symbol="alpha")
    layer.set_parameter_context(context, regenerate=True)

    assert len(layer.params) == num_params
    assert all(isinstance(param, Parameter) for param in layer.params)
    assert layer.params[0].name == "alpha_{0}"
    assert layer.params[1].name == "alpha_{1}"
    assert layer.params[2].name == "alpha_{2}"