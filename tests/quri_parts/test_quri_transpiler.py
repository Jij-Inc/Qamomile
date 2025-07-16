import pytest
import numpy as np
import networkx as nx
import collections
import jijmodeling as jm
import ommx.v1
import quri_parts.circuit as qp_c
import quri_parts.core.operator as qp_o
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
import qamomile.core.bitssample as qm_bs
import qamomile.core as qm
from qamomile.core.converters.qaoa import QAOAConverter
from qamomile.quri_parts.transpiler import QuriPartsTranspiler


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

    # problem += jm.Constraint("one-color", x[v, :].sum() == 1, forall=v)
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
    qc = qm_c.QuantumCircuit(n)
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
    var_map = {tuple(dc.subscripts): dc.id for dc in compiled_instance.decision_variables if dc.name == "x"}

    for i in range(num_nodes):
        qc.x(var_map[(i, i)])

    return qc


@pytest.fixture
def transpiler():
    return QuriPartsTranspiler()


def test_transpile_simple_circuit(transpiler):
    qc = qm_c.QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert quri_circuit.qubit_count == 2
    assert len(quri_circuit.gates) == 2
    assert isinstance(quri_circuit.gates[0], qp_c.gate.QuantumGate)
    assert isinstance(quri_circuit.gates[1], qp_c.gate.QuantumGate)


def test_transpile_parametric_circuit(transpiler):
    qc = qm_c.QuantumCircuit(1)
    theta = qm_c.Parameter("theta")
    qc.rx(theta, 0)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert quri_circuit.qubit_count == 1
    assert len(quri_circuit.gates) == 1
    assert isinstance(quri_circuit.gates[0], qp_c.gate.ParametricQuantumGate)

def test_rotation_two_pauli_circuit(transpiler):
    qc = qm_c.QuantumCircuit(2)
    theta = qm_c.Parameter("theta")
    qc.rxx(theta, 0, 1)
    quri_circuit = transpiler.transpile_circuit(qc)

    for gate in quri_circuit.gates:
        assert isinstance(gate, qp_c.gate.ParametricQuantumGate)
        assert gate.pauli_ids == (1,1)

    qc = qm_c.QuantumCircuit(2)
    theta = qm_c.Parameter("theta")
    qc.ryy(theta, 0, 1)
    quri_circuit = transpiler.transpile_circuit(qc)

    for gate in quri_circuit.gates:
        assert isinstance(gate, qp_c.gate.ParametricQuantumGate)
        assert gate.pauli_ids == (2,2)

    qc = qm_c.QuantumCircuit(2)
    theta = qm_c.Parameter("theta")
    qc.rzz(theta, 0, 1)
    quri_circuit = transpiler.transpile_circuit(qc)

    for gate in quri_circuit.gates:
        assert isinstance(gate, qp_c.gate.ParametricQuantumGate)
        assert gate.pauli_ids == (3,3)

def test_transpile_complex_circuit(transpiler):
    qc = qm_c.QuantumCircuit(3, 3)
    qc.h(0)
    qc.cnot(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure(0, 0)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert quri_circuit.qubit_count == 3
    assert quri_circuit.cbit_count == 3
    assert len(quri_circuit.gates) == 3  # QURI-Parts does not support measurement gate


def test_transpile_unsupported_gate(transpiler):
    class UnsupportedGate(qm_c.Gate):
        pass

    qc = qm_c.QuantumCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(Exception):  # Replace with specific exception if known
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler):
    mock_result = (collections.Counter({0: 500, 3: 500}), 2)

    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000


def test_transpile_hamiltonian(transpiler):
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 1),), 2.0)

    quri_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(quri_hamiltonian, qp_o.Operator)
    assert len(quri_hamiltonian) == 2
    assert np.isclose(
        quri_hamiltonian[qp_o.pauli_label([(0, qp_o.SinglePauli.X)])], 1.0
    )
    assert np.isclose(
        quri_hamiltonian[qp_o.pauli_label([(1, qp_o.SinglePauli.Z)])], 2.0
    )


def test_parametric_two_qubit_gate(transpiler):
    qc = qm_c.QuantumCircuit(2)
    theta = qm_c.Parameter("theta")
    qc.gates.append(
        qm_c.ParametricTwoQubitGate(qm_c.ParametricTwoQubitGateType.RXX, 0, 1, theta)
    )
    qc.ry(theta, 0)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert len(quri_circuit.gates) == 2
    assert isinstance(quri_circuit.gates[0], qp_c.ParametricQuantumGate)
    assert quri_circuit.gates[0].target_indices == (0, 1)
    assert quri_circuit.gates[0].pauli_ids == (1, 1)  # XX
    assert quri_circuit.parameter_count == 1

def test_parametric_exp_gate(transpiler):
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    qc = qm_c.QuantumCircuit(2)
    theta = qm_c.Parameter("theta")
    qc.exp_evolution(theta,hamiltonian)
    quri_circuit = transpiler.transpile_circuit(qc)
    
    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert len(quri_circuit.gates) == 1
    assert isinstance(quri_circuit.gates[0], qp_c.ParametricQuantumGate)
    assert quri_circuit.gates[0].target_indices == (0, 1)
    assert quri_circuit.gates[0].pauli_ids == (1, 3)  #XZ
    assert quri_circuit.parameter_count == 1
    
    hamiltonian2 = qm_o.Hamiltonian()
    hamiltonian2 += qm_o.X(0) * qm_o.Y(1) + qm_o.Z(0) * qm_o.X(1)
    qc2 = qm_c.QuantumCircuit(2)
    qc2.exp_evolution(theta,hamiltonian2)
    quri_circuit2 = transpiler.transpile_circuit(qc2)

    assert isinstance(quri_circuit2, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert len(quri_circuit2.gates) == 2
    assert isinstance(quri_circuit2.gates[0], qp_c.ParametricQuantumGate)
    assert isinstance(quri_circuit2.gates[1], qp_c.ParametricQuantumGate)
    assert quri_circuit2.gates[0].target_indices == (0, 1)
    assert quri_circuit2.gates[1].target_indices == (1, 0)
    assert quri_circuit2.gates[0].pauli_ids == (1, 2) #XY
    assert quri_circuit2.gates[1].pauli_ids == (1, 3) #ZX
    assert quri_circuit2.parameter_count == 1


def test_qaoa_circuit():
    import jijmodeling as jm

    x = jm.BinaryVar("x", shape=(3,))
    problem = jm.Problem("qubo")
    problem += -x[0] * x[1] + x[1] * x[2] + x[2] * x[0]

    compiled_instance = jm.Interpreter({}).eval_problem(problem)
    qaoa_converter = QAOAConverter(compiled_instance)

    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(2)

    qp_circuit = QuriPartsTranspiler().transpile_circuit(qaoa_circuit)

    assert isinstance(qp_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert qp_circuit.qubit_count == 3
    assert qp_circuit.parameter_count == 4


def test_coloring_sample_decode():
    problem = graph_coloring_problem()
    instance_data = graph_coloring_instance()
    compiled_instance = jm.Interpreter(instance_data).eval_problem(problem)
    apply_vars = [(0, 0), (1, 0), (2, 0), (3, 0)]
    initial_circuit = create_graph_coloring_operator_ansatz_initial_state(
        compiled_instance, instance_data["V"], instance_data["N"], apply_vars
    )
    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})

    qp_transpiler = QuriPartsTranspiler()
    sampler = create_qulacs_vector_sampler()
    qp_circ = qp_transpiler.transpile_circuit(initial_circuit)
    qp_result = sampler(qp_circ, 10)

    sampleset = qaoa_converter.decode(
        qp_transpiler, (qp_result, initial_circuit.num_qubits)
    )
    nonzero_results = {k: v for k, v in sampleset.extract_decision_variables("x", 0).items() if v != 0}
    assert nonzero_results == {
        (0, 0): 1,
        (1, 0): 1,
        (2, 0): 1,
        (3, 0): 1,
    }
    assert len(sampleset.sample_ids) == 10

    apply_vars = [(0, 0), (1, 1), (2, 2)]
    initial_circuit = create_graph_coloring_operator_ansatz_initial_state(
        compiled_instance, instance_data["V"], instance_data["N"], apply_vars
    )
    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})

    qp_transpiler = QuriPartsTranspiler()
    sampler = create_qulacs_vector_sampler()
    qp_circ = qp_transpiler.transpile_circuit(initial_circuit)
    qp_result = sampler(qp_circ, 10)

    sampleset = qaoa_converter.decode(
        qp_transpiler, (qp_result, initial_circuit.num_qubits)
    )
    
    nonzero_results = {k: v for k, v in sampleset.extract_decision_variables("x", 0).items() if v != 0}
    assert nonzero_results == {(0, 0): 1, (1, 1): 1, (2, 2): 1}
    assert len(sampleset.sample_ids) == 10

    apply_vars = [(0, 0), (0, 1), (0, 2), (2, 2)]
    initial_circuit = create_graph_coloring_operator_ansatz_initial_state(
        compiled_instance, instance_data["V"], instance_data["N"], apply_vars
    )
    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})

    qp_transpiler = QuriPartsTranspiler()
    sampler = create_qulacs_vector_sampler()
    qp_circ = qp_transpiler.transpile_circuit(initial_circuit)
    qp_result = sampler(qp_circ, 10)

    sampleset = qaoa_converter.decode(
        qp_transpiler, (qp_result, initial_circuit.num_qubits)
    )

    nonzero_results = {k: v for k, v in sampleset.extract_decision_variables("x", 0).items() if v != 0}
    assert nonzero_results == {(0, 0): 1, (0, 1): 1, (0, 2): 1, (2, 2): 1}
    assert len(sampleset.sample_ids) == 10


def test_tsp_decode():
    problem = tsp_problem()
    instance_data = tsp_instance()
    compiled_instance = jm.Interpreter(instance_data).eval_problem(problem)

    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})
    initial_circuit = create_tsp_initial_state(compiled_instance)

    qp_transpiler = QuriPartsTranspiler()
    sampler = create_qulacs_vector_sampler()
    qp_circ = qp_transpiler.transpile_circuit(initial_circuit)
    qp_result = sampler(qp_circ, 10)

    sampleset = qaoa_converter.decode(
        qp_transpiler, (qp_result, initial_circuit.num_qubits)
    )

    nonzero_results = {k: v for k, v in sampleset.extract_decision_variables("x", 0).items() if v != 0}
    assert nonzero_results == {
        (0, 0): 1,
        (1, 1): 1,
        (2, 2): 1,
        (3, 3): 1,
    }
    assert len(sampleset.sample_ids) == 10


def test_run_small_circuit():
    circuit = qm_c.QuantumCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure_all()

    transpiler = QuriPartsTranspiler()
    quri_c = transpiler.transpile_circuit(circuit)
    from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
    sampler = create_qulacs_vector_sampler()
    result = sampler(quri_c, 10)
