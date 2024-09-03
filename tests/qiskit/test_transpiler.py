# File: tests/qiskit/test_transpiler.py

import pytest
import numpy as np
import jijmodeling as jm
import jijmodeling_transpiler.core as jmt
import networkx as nx
import qiskit
import qiskit.quantum_info as qk_ope
import qiskit.primitives as qk_pr
from qamomile.core.circuit import QuantumCircuit as QamomileCircuit
from qamomile.core.circuit import Parameter
from qamomile.core.operator import Hamiltonian, PauliOperator, Pauli
import qamomile.core.bitssample as qm_bs

import qamomile.core as qm
from qamomile.qiskit.transpiler import QiskitTranspiler
from qamomile.qiskit.exceptions import QamomileQiskitTranspileError

def graph_coloring_problem() -> jm.Problem:
    # define variables
    V = jm.Placeholder("V")
    E = jm.Placeholder("E", ndim=2)
    N = jm.Placeholder("N")
    x = jm.BinaryVar("x", shape=(V, N))
    n = jm.Element("i", belong_to=(0, N))
    v = jm.Element("v", belong_to=(0, V))
    e = jm.Element("e", belong_to=E)
    # set problem
    problem = jm.Problem("Graph Coloring")
    # set one-hot constraint that each vertex has only one color

    #problem += jm.Constraint("one-color", x[v, :].sum() == 1, forall=v)
    problem += jm.Constraint("one-color", jm.sum(n, x[v, n]) == 1, forall=v)
    # set objective function: minimize edges whose vertices connected by edges are the same color
    problem += jm.sum([n, e], x[e[0], n] * x[e[1], n])
    return problem

def graph_coloring_instance():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3)])
    E = [list(edge) for edge in G.edges]
    num_color = 3
    num_nodes = G.number_of_nodes()
    instance_data = {"V": num_nodes, "N": num_color, "E": E}
    return instance_data

def create_graph_coloring_operator_ansatz_initial_state(
    compiled_instance: jmt.CompiledInstance, num_nodes: int, num_color: int
):
    n = num_color * num_nodes
    qc = QamomileCircuit(n)
    var_map = compiled_instance.var_map.var_map["x"]
    for i in range(num_nodes):
        qc.x(var_map[(i, 0)])  # set all nodes to color 0
    return qc

@pytest.fixture
def transpiler():
    return QiskitTranspiler()


def test_transpile_simple_circuit(transpiler):
    qc = QamomileCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert qiskit_circuit.num_qubits == 2
    assert len(qiskit_circuit.data) == 2
    assert qiskit_circuit.data[0].operation.name == "h"
    assert qiskit_circuit.data[1].operation.name == "cx"


def test_transpile_parametric_circuit(transpiler):
    qc = QamomileCircuit(1)
    theta = Parameter("theta")
    qc.rx(theta, 0)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert len(qiskit_circuit.parameters) == 1
    assert qiskit_circuit.parameters[0].name == "theta"


def test_transpile_complex_circuit(transpiler):
    qc = QamomileCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure(0, 0)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert qiskit_circuit.num_qubits == 3
    assert qiskit_circuit.num_clbits == 3
    assert len(qiskit_circuit.data) == 4


def test_transpile_unsupported_gate(transpiler):
    class UnsupportedGate:
        pass

    qc = QamomileCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(QamomileQiskitTranspileError):
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler):
    # Simulate Qiskit BitArray result
    class MockBitArray:
        def __init__(self, counts, num_bits):
            self.counts = counts
            self.num_bits = num_bits

        def get_int_counts(self):
            return self.counts

    mock_result = MockBitArray({0: 500, 3: 500}, 2)

    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000


def test_transpile_hamiltonian(transpiler):
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.Z, 1),), 2.0)

    qiskit_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(qiskit_hamiltonian, qk_ope.SparsePauliOp)
    assert len(qiskit_hamiltonian) == 2
    assert np.allclose(qiskit_hamiltonian.coeffs, [1.0, 2.0])

    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0), ), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.X, 0), ), 1.0)

    qiskit_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(qiskit_hamiltonian, qk_ope.SparsePauliOp)
    assert len(qiskit_hamiltonian) == 1
    assert np.allclose(qiskit_hamiltonian.coeffs, [2.0])
    assert np.all(qiskit_hamiltonian.paulis == ['X'])


def test_coloring_sample_decode():
    problem = graph_coloring_problem()
    instance_data = graph_coloring_instance()
    compiled_instance = jmt.compile_model(problem, instance_data)
    initial_circuit = create_graph_coloring_operator_ansatz_initial_state(compiled_instance, instance_data['V'], instance_data['N'])
    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})
    
    qk_transpiler = QiskitTranspiler()
    sampler = qk_pr.StatevectorSampler()
    qk_circ = qk_transpiler.transpile_circuit(initial_circuit)
    qk_circ.measure_all()
    job = sampler.run([qk_circ])
    
    job_result = job.result()
    
    sampleset = qaoa_converter.decode(qk_transpiler, job_result[0].data['meas'])
    assert sampleset[0].var_values["x"].values == {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1}
    assert sampleset[0].num_occurrences == 1024

    
