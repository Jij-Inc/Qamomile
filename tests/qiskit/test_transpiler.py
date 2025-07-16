# File: tests/qiskit/test_transpiler.py

import pytest
import numpy as np
import jijmodeling as jm
import ommx.v1
import networkx as nx
import qiskit
import qiskit.quantum_info as qk_ope
import qiskit.primitives as qk_pr

from qamomile.core.circuit import QuantumCircuit as QamomileCircuit
from qamomile.core.circuit import Parameter
from qamomile.core.operator import Hamiltonian, PauliOperator, Pauli, X, Y, Z
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
    compiled_instance: ommx.v1.Instance,
    num_nodes: int,
    num_color: int,
    apply_vars: list[tuple[int, int]],
):
    n = num_color * num_nodes
    qc = QamomileCircuit(n)
    var_map = {tuple(dc.subscripts): dc.id for dc in compiled_instance.decision_variables if dc.name == "x"}
    for pos in apply_vars:
        qc.x(var_map[pos])  # set all nodes to color 0
    return qc


def tsp_problem() -> jm.Problem:
    N = jm.Placeholder("N")
    D = jm.Placeholder("d", ndim=2)
    start = jm.Placeholder("start", latex="N-1")
    x = jm.BinaryVar("x", shape=(N - 1, N - 1))
    t = jm.Element("t", belong_to=N - 2)
    j = jm.Element("j", belong_to=N - 1)
    u = jm.Element("u", belong_to=(0, N - 1))
    v = jm.Element("v", belong_to=(0, N - 1))

    problem = jm.Problem("TSP")
    problem += jm.sum(u, D[start][u] * (x[0][u] + x[N - 2][u])) + jm.sum(
        t, jm.sum(u, jm.sum(v, D[u][v] * x[t][u] * x[t + 1][v]))
    )

    return problem


def tsp_instance():
    N = 5
    np.random.seed(3)

    x_pos = np.random.rand(N)
    y_pos = np.random.rand(N)

    d = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            d[i][j] = np.sqrt((x_pos[i] - x_pos[j]) ** 2 + (y_pos[i] - y_pos[j]) ** 2)

    instance_data = {"N": N, "d": d, "start": N - 1}
    return instance_data


def create_tsp_initial_state(
    compiled_instance: ommx.v1.Instance, num_nodes: int = 4
):
    n = num_nodes * num_nodes
    qc = qm.circuit.QuantumCircuit(n)
    var_map = {
        tuple(dc.subscripts): dc.id for dc in compiled_instance.decision_variables}

    for i in range(num_nodes):
        qc.x(var_map[(i, i)])

    return qc

@pytest.fixture
def transpiler():
    return QiskitTranspiler()


def test_transpile_simple_circuit(transpiler: QiskitTranspiler):
    qc = QamomileCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert qiskit_circuit.num_qubits == 2
    assert len(qiskit_circuit.data) == 2
    assert qiskit_circuit.data[0].operation.name == "h"
    assert qiskit_circuit.data[1].operation.name == "cx"


def test_transpile_parametric_circuit(transpiler: QiskitTranspiler):
    qc = QamomileCircuit(1)
    theta = Parameter("theta")
    qc.rx(theta, 0)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert len(qiskit_circuit.parameters) == 1
    assert qiskit_circuit.parameters[0].name == "theta"


def test_transpile_complex_circuit(transpiler: QiskitTranspiler):
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


def test_transpile_unsupported_gate(transpiler: QiskitTranspiler):
    class UnsupportedGate:
        pass

    qc = QamomileCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(QamomileQiskitTranspileError):
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler: QiskitTranspiler):
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


def test_transpile_hamiltonian(transpiler: QiskitTranspiler):
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
    compiled_instance = jm.Interpreter(instance_data).eval_problem(problem)
    apply_vars = [(0, 0), (1, 0), (2, 0), (3, 0)]
    initial_circuit = create_graph_coloring_operator_ansatz_initial_state(compiled_instance, instance_data['V'], instance_data['N'], apply_vars)
    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})
    
    qk_transpiler = QiskitTranspiler()
    sampler = qk_pr.StatevectorSampler()
    qk_circ = qk_transpiler.transpile_circuit(initial_circuit)
    qk_circ.measure_all()
    job = sampler.run([qk_circ])
    
    job_result = job.result()
    bit_array = job_result[0].data.meas
    sampleset = qaoa_converter.decode(qk_transpiler, bit_array)
    nonzero_results = {k: v for k, v in sampleset.extract_decision_variables("x", 0).items() if v != 0}
    assert nonzero_results == {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1}
    assert len(sampleset.sample_ids) == 1024

def test_parametric_exp_gate(transpiler: QiskitTranspiler):
        hamiltonian = Hamiltonian()
        hamiltonian += X(0) * Z(1)
        qc = QamomileCircuit(2)
        theta = Parameter("theta")
        qc.exp_evolution(theta,hamiltonian)
        qk_circ = transpiler.transpile_circuit(qc)
        
        assert isinstance(qk_circ, qiskit.QuantumCircuit)
        assert len(qk_circ.data) == 1
        assert qk_circ.data[0].operation.name == 'PauliEvolution'
        assert qk_circ.data[0].operation.num_qubits == 2
        assert qk_circ.data[0].qubits[0]._index == 0
        assert qk_circ.data[0].qubits[1]._index == 1
        assert len(qk_circ.data[0].params) == 1

        hamiltonian2 = Hamiltonian()
        hamiltonian2 += X(0) * Y(1) + Z(0) * X(1)
        qc2 = QamomileCircuit(2)
        qc2.exp_evolution(theta,hamiltonian2)
        qk_circ2 = transpiler.transpile_circuit(qc2)

        assert isinstance(qk_circ2, qiskit.QuantumCircuit)
        assert len(qk_circ2.data) == 1
        assert qk_circ2.data[0].operation.name == 'PauliEvolution'
        assert qk_circ2.data[0].operation.num_qubits == 2
        assert qk_circ2.data[0].qubits[0]._index == 0
        assert qk_circ2.data[0].qubits[1]._index == 1
        assert len(qk_circ2.data[0].params) == 1
    
def test_tsp_decode():
    problem = tsp_problem()
    instance_data = tsp_instance()
    compiled_instance = jm.Interpreter(instance_data).eval_problem(problem)

    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})
    initial_circuit = create_tsp_initial_state(compiled_instance)

    qk_transpiler = QiskitTranspiler()
    sampler = qk_pr.StatevectorSampler()
    qk_circ = qk_transpiler.transpile_circuit(initial_circuit)
    qk_circ.measure_all()
    job = sampler.run([qk_circ])
    
    job_result = job.result()
    bit_array = job_result[0].data.meas
    sampleset = qaoa_converter.decode(qk_transpiler, bit_array)

    results = sampleset.extract_decision_variables("x", sample_id=0)
    nonzero_results = {k: v for k, v in results.items() if v != 0}
    assert nonzero_results == {
        (0, 0): 1,
        (1, 1): 1,
        (2, 2): 1,
        (3, 3): 1,
    }
    assert len(sampleset.sample_ids) == 1024

def test_transpile_hamiltonian_with_complex_coeffs(transpiler: QiskitTranspiler):
    """Test transpiling Hamiltonian with complex coefficients."""
    hamiltonian = Hamiltonian()
    
    # Add terms with complex coefficients
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0 + 2.0j)
    hamiltonian.add_term((PauliOperator(Pauli.Y, 1),), -0.5 + 1.5j)
    hamiltonian.add_term((PauliOperator(Pauli.Z, 0), PauliOperator(Pauli.X, 1)), 2.0 - 1.0j)
    
    # Add a constant term (also complex)
    hamiltonian.constant = 0.5 + 0.5j
    
    qiskit_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)
    
    # Verify the result
    assert isinstance(qiskit_hamiltonian, qk_ope.SparsePauliOp)
    assert len(qiskit_hamiltonian) == 4  # 3 Pauli terms + 1 constant term
    
    # Check that coefficients are complex
    assert qiskit_hamiltonian.coeffs.dtype == np.complex128
    
    # Verify specific coefficients
    expected_coeffs = [1.0 + 2.0j, -0.5 + 1.5j, 2.0 - 1.0j, 0.5 + 0.5j]
    
    assert np.allclose(qiskit_hamiltonian.coeffs, expected_coeffs)

def test_transpile_hamiltonian_mixed_real_complex(transpiler: QiskitTranspiler):
    """Test transpiling Hamiltonian with mixed real and complex coefficients."""
    hamiltonian = Hamiltonian()
    
    # Mix real and complex coefficients
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)  # Real
    hamiltonian.add_term((PauliOperator(Pauli.Y, 1),), 1.0j)  # Pure imaginary
    hamiltonian.add_term((PauliOperator(Pauli.Z, 2),), 2.5 + 1.5j)  # Complex
    
    qiskit_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)
    
    # Should automatically promote to complex dtype
    assert qiskit_hamiltonian.coeffs.dtype == np.complex128
    assert len(qiskit_hamiltonian) == 3
    
    # Check values
    expected_coeffs = [1.0 + 0.0j, 0.0 + 1.0j, 2.5 + 1.5j]
    
    assert np.allclose(qiskit_hamiltonian.coeffs, expected_coeffs) 