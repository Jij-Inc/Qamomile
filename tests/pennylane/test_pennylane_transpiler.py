import pytest
import pennylane as qml
import numpy as np
import ommx.v1
import qamomile
import qamomile.core.bitssample as qm_bs
from qamomile.pennylane.transpiler import PennylaneTranspiler
from qamomile.core.operator import Hamiltonian, Pauli, X, Y, Z
from qamomile.core.circuit import (
    QuantumCircuit,
    SingleQubitGate,
    TwoQubitGate,
    ParametricSingleQubitGate,
    ParametricTwoQubitGate,
    SingleQubitGateType,
    TwoQubitGateType,
    ParametricSingleQubitGateType,
    ParametricTwoQubitGateType,
    Parameter
)

import jijmodeling as jm
import networkx as nx

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
    apply_vars: tuple[int, int],
):
    n = num_color * num_nodes
    qc = QamomileCircuit(n)
    var_map = {tuple(dc.subscripts): dc.id for dc in compiled_instance.decision_variables if dc.name == "x"}
    for pos in apply_vars:
        qc.x(var_map[pos])  # set all nodes to color 0
    return qc

@pytest.fixture
def transpiler():
    """Fixture to initialize the PennylaneTranspiler."""
    return PennylaneTranspiler()

def test_transpile_empty_hamiltonian(transpiler):
    """Test transpiling an empty Hamiltonian."""
    hamiltonian = Hamiltonian()
    pennylane_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)
    assert isinstance(pennylane_hamiltonian, qml.Hamiltonian)
    assert len(pennylane_hamiltonian.operands) == 0
    assert len(pennylane_hamiltonian.coeffs) == 0

def test_transpile_hamiltonian(transpiler):
    """Test the transpilation of Qamomile Hamiltonian to Pennylane Hamiltonian."""
    # Define a Qamomile Hamiltonian
    hamiltonian = qamomile.core.operator.Hamiltonian()
    hamiltonian += qamomile.core.operator.X(0) * qamomile.core.operator.Z(1)

    # Transpile the Hamiltonian
    pennylane_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Assert the result is a Pennylane Hamiltonian
    assert isinstance(pennylane_hamiltonian, qml.Hamiltonian)

    # Validate number of qubits and terms
    assert len(pennylane_hamiltonian.operands) == 1 # Only one term
    assert np.all((pennylane_hamiltonian.coeffs , [1.0, ])) # Default coefficient is 1.0

    # Validate term content
    term_ops = pennylane_hamiltonian.terms()[1]
    # assert isinstance(term_ops, qml.operation)
    assert len(term_ops[0]) == 2  # Two operators in the term
    assert isinstance(term_ops[0][0], qml.PauliX)  # X on qubit 0
    assert isinstance(term_ops[0][1], qml.PauliZ)  # Z on qubit 1

def test_transpile_unsupported_pauli(transpiler):
    """Test transpiling a Hamiltonian with an unsupported Pauli operator."""
    class FakePauli:
        pauli = "W" 

        def __init__(self, index):
            self.index = index

    hamiltonian = Hamiltonian()
    term = (FakePauli(0), ) 
    hamiltonian.terms[term] = 1.0

    with pytest.raises(NotImplementedError, match="Unsupported Pauli operator"):
        transpiler.transpile_hamiltonian(hamiltonian)

def test_transpile_hamiltonian_with_multiple_terms(transpiler):
    """Test transpiling a Hamiltonian with multiple supported terms."""
    hamiltonian = Hamiltonian()
    hamiltonian += X(0)*Z(1)
    hamiltonian += Y(0)*Y(1)*Z(2)*X(3)*X(4)

    pennylane_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)
    assert isinstance(pennylane_hamiltonian, qml.Hamiltonian)
    assert len(pennylane_hamiltonian.operands) == 2
    assert np.allclose(pennylane_hamiltonian.coeffs, [1.0, 1.0])

    ops_first_term = pennylane_hamiltonian.terms()[1][0]
    # X(0)*Z(1)*I(2)*I(3)*I(4)
    assert len(ops_first_term) == 5
    assert isinstance(ops_first_term[0], qml.PauliX)
    assert ops_first_term[0].wires.tolist() == [0]
    assert isinstance(ops_first_term[1], qml.PauliZ)
    assert ops_first_term[1].wires.tolist() == [1]
    assert isinstance(ops_first_term[2], qml.I)
    assert ops_first_term[2].wires.tolist() == [2]

    ops_second_term = pennylane_hamiltonian.terms()[1][1]
    # Y(0)*Y(1)*Z(2)*X(3)*X(4)
    print(ops_second_term)
    assert len(ops_second_term) == 5
    assert all(isinstance(op, (qml.PauliY, qml.PauliZ, qml.PauliX)) for op in ops_second_term)
    wire_sequence = [op.wires.tolist()[0] for op in ops_second_term]
    assert wire_sequence == [0, 1, 2, 3, 4]

def test_transpile_circuit_basic(transpiler):
    """Test transpiling a simple Qamomile circuit to a Pennylane callable."""
    circuit = QuantumCircuit(2)
    circuit.gates.append(SingleQubitGate(SingleQubitGateType.X, 0))
    circuit.gates.append(TwoQubitGate(TwoQubitGateType.CNOT, 0, 1))

    fn = transpiler.transpile_circuit(circuit)
    assert callable(fn)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def test_qnode():
        fn()
        return qml.state()

    state = test_qnode()
    # initial state |00>, apply X(0) become |10>, then CNOT(control=0, target=1) -> |11>
    expected_state = np.zeros(4)
    expected_state[3] = 1.0  # |11>
    assert np.allclose(state, expected_state)

def test_transpile_circuit_with_parameters(transpiler):
    """Test transpiling a circuit with parameters."""

    circuit = QuantumCircuit(2)
    theta = Parameter("theta")

    circuit.rx(2*theta, 0)
    circuit.crx(theta, 0 , 1)

    fn = transpiler.transpile_circuit(circuit)
    assert callable(fn)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def test_qnode(theta):
        fn(theta=theta)
        return qml.expval(qml.PauliZ(1))

    # when theta=0，RX(2*0)=I, CRX(0)=CNOT，|00> still |00>, Z measurement on qubit 1:
    # expval = |0>'s prob - |1>'s prob = 1
    assert np.allclose(test_qnode(0.0), 1.0)

    # when theta=pi，RX(pi)=X gate on qubit 0，|00> -> |10>，then CRX(pi/2)
    # CRX(pi/2) -> |1+> , Z measurement on qubit 0:
    # expval=|0>'s prob - |1>'s prob = 0
    assert np.allclose(test_qnode(np.pi/2), 0)


def test_extract_angle_param_not_found(transpiler):
    """Test extracting an angle when the parameter is not found in params."""
    phi = Parameter("phi")
    gate = ParametricSingleQubitGate(ParametricSingleQubitGateType.RX, 0, phi)

    with pytest.raises(ValueError, match="Parameter 'phi' not found"):
        transpiler._extract_angle(gate, params={})


def test_apply_single_qubit_gate_unsupported(transpiler):
    """Test applying an unsupported single qubit gate."""
    class FakeGate:
        gate = "INVALID"
        qubit = 0

    with pytest.raises(NotImplementedError, match="Unsupported single-qubit gate"):
        transpiler._apply_single_qubit_gate(FakeGate())


def test_apply_two_qubit_gate_unsupported(transpiler):
    """Test applying an unsupported two qubit gate."""
    class FakeTwoQubitGate:
        gate = "INVALID"
        control = 0
        target = 1

    with pytest.raises(NotImplementedError, match="Unsupported two-qubit gate"):
        transpiler._apply_two_qubit_gate(FakeTwoQubitGate())


def test_apply_parametric_single_qubit_gate_unsupported(transpiler):
    """Test applying an unsupported parametric single qubit gate."""
    class MockParamGate:
        gate = "INVALID"
        qubit = 0
        parameter = Parameter("theta")

    with pytest.raises(NotImplementedError, match="Unsupported parametric single-qubit gate"):
        transpiler._apply_parametric_single_qubit_gate(MockParamGate(), params={"theta": np.pi})


def test_apply_parametric_two_qubit_gate_unsupported(transpiler):
    """Test applying an unsupported parametric two qubit gate."""
    class MockParamGate:
        gate = "INVALID"
        control = 0
        target = 1
        parameter = Parameter("theta")

    with pytest.raises(NotImplementedError, match="Unsupported parametric two-qubit gate"):
        transpiler._apply_parametric_two_qubit_gate(MockParamGate(), params={"theta": np.pi})

def test_convert_result(transpiler):

    dict_result = {'00': 500, '11': 500}

    sampleset = transpiler.convert_result(dict_result)

    assert isinstance(sampleset, qm_bs.BitsSampleSet)
    assert len(sampleset.bitarrays) == 2
    assert sampleset.total_samples() == 1000