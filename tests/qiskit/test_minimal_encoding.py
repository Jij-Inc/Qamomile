import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import qiskit as qk
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms.eigensolvers import NumPyEigensolver
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

import qamomile.qiskit.minimal_encoding as minimal_encoding


def generate_circuit(
    num_register_bits: int, reps: int
) -> qk.circuit.quantumcircuit.QuantumCircuit:
    """
    Function to generate qunatum circuit (variational ansatz) for minimal encoding.

    Parameters
    ----------
    num_register_bits : int
        number of register qubits
    reps : int
        number of layer, for this specific circuit one layer consists of C-NOT and Ry rotation gate

    Returns
    -------
    circuit : qiskit.circuit.quantumcircuit.QuantumCircuit
        Parameterised quantum circuit
    """
    # define number of qubits
    num_qubits = num_register_bits + 1
    qreg_q = QuantumRegister(num_register_bits, "q")
    areg_a = AncillaRegister(1, "a")
    circuit = QuantumCircuit(areg_a, qreg_q)

    # initialize a parameters
    parameters = ParameterVector("θ", num_qubits * reps)
    # create a dictionary of parameters with random values and return it
    initial_params = {
        parameter: np.random.random() for parameter in parameters
    }

    # add H gate for each qubit
    circuit.h(areg_a[0])
    for i in range(0, num_register_bits):
        circuit.h(qreg_q[i])
    circuit.barrier()

    # add layers which consist of CNOT and Ry gate
    for j in range(0, reps):
        # CNOT
        # circuit.cx(qreg_q[0],areg_a[0])
        circuit.cx(areg_a[0], qreg_q[0])
        for i in range(num_register_bits):
            if i != 0:
                # circuit.cx(qreg_q[i],qreg_q[i-1])
                circuit.cx(qreg_q[i - 1], qreg_q[i])

        # Ry
        for i in range(num_qubits):
            if i == 0:
                circuit.ry(parameters[num_qubits * j + i], areg_a[i])
            else:
                circuit.ry(parameters[num_qubits * j + i], qreg_q[i - 1])
        circuit.barrier()
    return circuit, initial_params


def test_minimal_encoding_onehot():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    minimal_encoding_builder = minimal_encoding.transpile_to_minimal_encoding(
        compiled_instance
    )
    assert isinstance(
        minimal_encoding_builder,
        minimal_encoding.to_minimal_encoding.MinimalEncodingBuilder,
    )

    minimal_encoded_instance = minimal_encoding_builder.get_encoded_instance(
        multipliers={"onehot": 1.0}
    )
    assert isinstance(
        minimal_encoded_instance,
        minimal_encoding.to_minimal_encoding.MinimalEncodedInstance,
    )


def test_minimal_encoding_onehot_correctness():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 2})
    minimal_encoding_builder = minimal_encoding.transpile_to_minimal_encoding(
        compiled_instance, normalize=False
    )

    minimal_encoded_instance = minimal_encoding_builder.get_encoded_instance(
        multipliers={"onehot": 1.0}
    )
    ansatz, theta = generate_circuit(
        minimal_encoded_instance.num_register_bits, reps=2
    )

    optimized_func = minimal_encoded_instance.get_minimized_function(ansatz)

    result = minimize(
        optimized_func,
        list(theta.values()),
        method="COBYLA",
    )
    sample_set = minimal_encoded_instance.get_optimized_state(ansatz, result.x)

    assert len(sample_set.feasible().record.solution["x"]) == 1
