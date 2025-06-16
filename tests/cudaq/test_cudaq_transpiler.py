# File: tests/qiskit/test_transpiler.py
import cudaq
import jijmodeling as jm
import jijmodeling_transpiler.core as jmt
import networkx as nx
import numpy as np
import pytest

import qamomile.core as qm
from qamomile.core.circuit import QuantumCircuit as QamomileCircuit
from qamomile.core.circuit import Parameter
from qamomile.core.operator import Hamiltonian, PauliOperator, Pauli, X, Y, Z
import qamomile.core.bitssample as qm_bs
from qamomile.cudaq.transpiler import CudaqTranspiler
from qamomile.cudaq.exceptions import QamomileCudaqTranspileError
from tests.utils import *


# >>> Data Preparation >>>
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
    compiled_instance: jmt.CompiledInstance,
    num_nodes: int,
    num_color: int,
    apply_vars: tuple[int, int],
):
    n = num_color * num_nodes
    qc = QamomileCircuit(n)
    var_map = compiled_instance.var_map.var_map["x"]
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
    compiled_instance: jmt.CompiledInstance, num_nodes: int = 4
):
    n = num_nodes * num_nodes
    qc = qm.circuit.QuantumCircuit(n)
    var_map = compiled_instance.var_map.var_map["x"]

    for i in range(num_nodes):
        qc.x(var_map[(i, i)])

    return qc


# <<< Data Preparation <<<


@pytest.fixture
def transpiler():
    return CudaqTranspiler()


def test_transpile_simple_circuit(transpiler: CudaqTranspiler):
    # Create a simple circuit and the expected statevector.
    num_qubits = 2
    qc = QamomileCircuit(num_qubits)
    state_0 = np.kron(KET_0, KET_0)  # |00>
    qc.x(0)
    x_applied_state = np.kron(X_MATRIX, I_MATRIX) @ state_0  # |10>
    qc.cx(0, 1)
    expected_statevector = CX_MATRIX @ x_applied_state  # |11>

    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)
    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Get the statevector of the kernel.
    cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, []))
    # Check if the cudaq statevector matches the expected statevector.
    assert np.allclose(cudaq_statevector, expected_statevector)


def test_transpile_parametric_circuit(transpiler: CudaqTranspiler):
    # Create a simple parametric circuit and the expected statevector.
    num_qubits = 1
    qc = QamomileCircuit(num_qubits)
    state_0 = KET_0  # |0>
    theta = Parameter("theta")
    qc.rx(theta, 0)
    expected_statevector = (
        lambda theta: RX_MATRIX(theta) @ state_0
    )  # |0> rotated by theta
    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)

    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Check if the kernel has only one parameter.
    with pytest.raises(RuntimeError):
        # Zero parameter should raise an error.
        cudaq.sample(cudaq_kernel, [])
    # One parameter should not raise an error.
    cudaq.sample(cudaq_kernel, [0])
    with pytest.raises(RuntimeError):
        # Two parameters should raise an error.
        cudaq.sample(cudaq_kernel, [0, 1])

    # Check if the statevector matches the expected statevector for several thetas.
    np.random.seed(901)
    num_trials = 100
    for _ in range(num_trials):
        theta_value = np.random.uniform(0, 2 * np.pi)
        cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, [theta_value]))
        assert np.allclose(cudaq_statevector, expected_statevector(theta))


def test_transpile_complex_circuit(transpiler: CudaqTranspiler):
    # Create a more complex circuit with multiple gates.
    num_qubits = 3
    num_cbits = 3
    num_measured_cbits = 0
    qc = QamomileCircuit(num_qubits, num_cbits)
    state_0 = take_tensor_product(KET_0, KET_0, KET_0)  # |000>
    qc.h(0)
    state_1 = take_tensor_product(H_MATRIX, I_MATRIX, I_MATRIX) @ state_0
    qc.cx(0, 1)
    state_2 = take_tensor_product(CX_MATRIX, I_MATRIX) @ state_1
    qc.ccx(0, 1, 2)
    expected_statevector = CCX_MATRIX @ state_2
    qc.measure(0, 0)
    num_measured_cbits += 1

    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)

    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Check if the kernel has the expected number of measured classical bits.
    sample = cudaq.sample(cudaq_kernel, [])
    for key, _ in sample.items():
        break
    assert len(key) == num_measured_cbits

    # Check if the statevector matches the expected statevector.
    cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, []))
    assert np.allclose(cudaq_statevector, expected_statevector)


def test_transpile_unsupported_gate(transpiler: CudaqTranspiler):
    class UnsupportedGate:
        pass

    qc = QamomileCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(QamomileCudaqTranspileError):
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler: CudaqTranspiler):
    # Simulate Qiskit BitArray result
    class MockBitArray:
        def __init__(self, counts):
            self.counts = counts

        def get_int_counts(self):
            return self.counts

    mock_result = MockBitArray({0: 500, 3: 500})

    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000


def test_transpile_hamiltonian(transpiler: CudaqTranspiler):
    # Create a Hamiltonian in Qamomile format.
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.Z, 1),), 2.0)

    # Transpile the Hamiltonian to a CUDA-Q Hamiltonian.
    cudaq_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Check if the transpiled Hamiltonian is a CUDA-Q SpinOperator.
    assert isinstance(cudaq_hamiltonian, cudaq.SpinOperator)

    # Get the number of terms and coefficients in the CUDA-Q Hamiltonian.
    num_terms = 0
    cudaq_coeffs = []
    for term in cudaq_hamiltonian:
        coeff = term.coefficient.evaluate()
        if coeff != 0:
            num_terms += 1
            cudaq_coeffs.append(coeff)
    # Check the number of terms and coefficients.
    assert num_terms == 2
    # Check the coefficients of the terms.
    assert np.allclose(cudaq_coeffs, [1.0, 2.0])

    # Create another Hamiltonian with duplicate terms.
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)

    # Transpile the Hamiltonian to a CUDA-Q Hamiltonian.
    cudaq_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Check if the transpiled Hamiltonian is a CUDA-Q SpinOperator.
    assert isinstance(cudaq_hamiltonian, cudaq.SpinOperator)

    # Get the number of terms and coefficients in the CUDA-Q Hamiltonian.
    num_terms = 0
    cudaq_coeffs = []
    for term in cudaq_hamiltonian:
        coeff = term.coefficient.evaluate()
        if coeff != 0:
            num_terms += 1
            cudaq_coeffs.append(coeff)
    # Check the number of terms and coefficients.
    assert num_terms == 1
    # Check the coefficients of the terms.
    assert np.allclose(cudaq_coeffs, [2.0])


def test_coloring_sample_decode(transpiler: CudaqTranspiler):
    # Create a graph coloring problem and instance data.
    problem = graph_coloring_problem()
    instance_data = graph_coloring_instance()
    compiled_instance = jmt.compile_model(problem, instance_data)
    apply_vars = [(0, 0), (1, 0), (2, 0), (3, 0)]
    initial_circuit = create_graph_coloring_operator_ansatz_initial_state(
        compiled_instance, instance_data["V"], instance_data["N"], apply_vars
    )
    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})

    cudaq_kernel = transpiler.transpile_circuit(initial_circuit)
    result = cudaq.sample(cudaq_kernel, [])

    sampleset = qaoa_converter.decode(transpiler, result)
    assert sampleset[0].var_values["x"].values == {
        (0, 0): 1,
        (1, 0): 1,
        (2, 0): 1,
        (3, 0): 1,
    }
    assert sampleset[0].num_occurrences == 1024


def test_parametric_exp_gate(transpiler: CudaqTranspiler):
    hamiltonian = Hamiltonian()
    hamiltonian += X(0) * Z(1)
    qc = QamomileCircuit(2)
    theta = Parameter("theta")
    qc.exp_evolution(theta, hamiltonian)

    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)
    # Check if the kernel has the expected number of qubits.
    qir_str = cudaq.translate(cudaq_kernel, [1], format="qir")
    assert count_qir_parameters(qir_str) == 1  # One parameter for theta
    # Check if the kernel has only one operation, which is the Pauli evolution.
    operations = count_qir_operations(qir_str)
    assert len(operations) == 1
    assert operations["__quantum__qis__exp_pauli"] == 1
    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == 2

    hamiltonian2 = Hamiltonian()
    hamiltonian2 += X(0) * Y(1) + Z(0) * X(1)
    qc2 = QamomileCircuit(2)
    qc2.exp_evolution(theta, hamiltonian2)
    cudaq_kernel2 = transpiler.transpile_circuit(qc2)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel2, cudaq.Kernel)
    # Check if the kernel has the expected number of qubits.
    qir_str2 = cudaq.translate(cudaq_kernel2, [1], format="qir")
    assert count_qir_parameters(qir_str2) == 1  # One parameter for theta
    # Check if the kernel has only one operation, which is the Pauli evolution.
    operations2 = count_qir_operations(qir_str2)
    assert len(operations2) == 1
    assert operations2["__quantum__qis__exp_pauli"] == 1
    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits2 = cudaq.get_state(cudaq_kernel2, []).num_qubits()
    assert cudaq_num_qubits2 == 2


def test_tsp_decode(transpiler: CudaqTranspiler):
    problem = tsp_problem()
    instance_data = tsp_instance()
    compiled_instance = jmt.compile_model(problem, instance_data)

    qaoa_converter = qm.qaoa.QAOAConverter(compiled_instance)
    qaoa_converter.ising_encode(multipliers={"one-color": 1})
    initial_circuit = create_tsp_initial_state(compiled_instance)

    cudaq_kernel = transpiler.transpile_circuit(initial_circuit)
    sample = cudaq.sample(cudaq_kernel, [])

    sampleset = qaoa_converter.decode(transpiler, sample)

    assert sampleset[0].var_values["x"].values == {
        (0, 0): 1,
        (1, 1): 1,
        (2, 2): 1,
        (3, 3): 1,
    }
    assert sampleset[0].num_occurrences == 1024
