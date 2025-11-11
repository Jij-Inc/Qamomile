# File: tests/core/qaoa/test_qaoa.py

import pytest
import numpy as np
import jijmodeling as jm
import ommx.v1
from qamomile.core.converters.qaoa import QAOAConverter
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
import qamomile.core.bitssample as qm_bs

from tests.utils import Utils


@pytest.fixture
def simple_qubo_problem():
    Q = jm.Placeholder("Q", ndim=2)
    n = Q.len_at(0, latex="n")
    x = jm.BinaryVar("x", shape=(n,))
    problem = jm.Problem("qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])
    instance_data = {"Q": [[0.1, 0.2], [0.2, 0.3]]}
    instance = jm.Interpreter(instance_data).eval_problem(problem)
    return instance


@pytest.fixture
def qaoa_converter(simple_qubo_problem):
    return QAOAConverter(simple_qubo_problem)


def test_get_cost_ansatz(qaoa_converter: QAOAConverter):

    beta = qm_c.Parameter("beta")
    cost_circuit = qaoa_converter.get_cost_ansatz(beta)

    assert isinstance(cost_circuit, qm_c.QuantumCircuit)
    assert cost_circuit.num_qubits == 2
    assert len(cost_circuit.gates) > 0
    assert any(
        isinstance(gate, qm_c.ParametricSingleQubitGate) for gate in cost_circuit.gates
    )


def test_get_ansatz_circuit(qaoa_converter):
    p = 2
    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(p)

    assert isinstance(qaoa_circuit, qm_c.QuantumCircuit)
    assert qaoa_circuit.num_qubits == 2
    assert len(qaoa_circuit.gates) > 0
    assert len(qaoa_circuit.get_parameters()) == 4  # 2 * p parameters


def test_get_cost_hamiltonian(qaoa_converter):
    hamiltonian = qaoa_converter.get_cost_hamiltonian()

    assert isinstance(hamiltonian, qm_o.Hamiltonian)
    assert len(hamiltonian.terms) > 0
    assert all(isinstance(term[0], qm_o.PauliOperator) for term in hamiltonian.terms)


def test_decode_bits_to_sampleset(qaoa_converter):
    # Create a mock BitsSampleSet
    bits_samples = [
        qm_bs.BitsSample(num_occurrences=3, bits=[0, 0]),
        qm_bs.BitsSample(num_occurrences=1, bits=[1, 1]),
    ]
    bits_sample_set = qm_bs.BitsSampleSet(bits_samples)

    sampleset = qaoa_converter.decode_bits_to_sampleset(bits_sample_set)

    assert isinstance(sampleset, ommx.v1.SampleSet)
    assert len(sampleset.sample_ids) == 4


def test_qaoa_converter_with_larger_problem():
    # Create a larger QUBO problem
    Q = jm.Placeholder("Q", ndim=2)
    n = Q.len_at(0, latex="n")
    x = jm.BinaryVar("x", shape=(n,))
    problem = jm.Problem("large_qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])
    instance_data = {"Q": np.random.rand(10, 10)}
    compiled_instance = jm.Interpreter(instance_data).eval_problem(problem)

    qaoa_converter = QAOAConverter(compiled_instance)

    # Test circuit generation
    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(p=3)
    assert qaoa_circuit.num_qubits == 10
    assert len(qaoa_circuit.get_parameters()) == 6  # 2 * p parameters

    # Test Hamiltonian generation
    hamiltonian = qaoa_converter.get_cost_hamiltonian()
    assert len(hamiltonian.terms) > 0


def test_multipliers():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    y = jm.BinaryVar("y")
    problem = jm.Problem("sample")
    i = jm.Element("i", (0, n))
    problem += jm.Constraint("const1", x[i] + y == 0, forall=i)
    intepreter = jm.Interpreter({"n": 3})
    instance: ommx.v1.Instance = intepreter.eval_problem(problem)

    multipliers = {"const1": 1.5}
    detail_parameters = {"const1": {(0,): 2.0}}

    converter = QAOAConverter(instance)
    qubo, constant = converter.instance_to_qubo(multipliers, detail_parameters)
    # 1.5*2*(x_0 + y)^2 + 1.5*(x_1 + y)^2 + 1.5*(x_2 + y)^2
    # = 6*(x_0*y) + 3*x_0 + ... + 3*(x_2*y) + 1.5*x_2 ... + 3*x_2*y + 3*y^2
    dv_list = instance.decision_variables
    dv_objects = {}
    for dv in dv_list:
        if dv.name not in dv_objects:
            dv_objects[dv.name] = {}
        dv_objects[dv.name][tuple(dv.subscripts)] = dv.id
    x0 = dv_objects["x"][(0,)]
    y = dv_objects["y"][()]
    assert qubo[x0, y] == 6.0
    assert qubo[x0, x0] == 3.0
    x1 = dv_objects["x"][(1,)]
    assert qubo[x1, y] == 3.0
    _ = converter.ising_encode()
    print(converter.int2varlabel)


def test_ommx_support_error_case():
    """This test raised the follwing error with the version whose commit ID 625fd2476dca09c6f4c710ffa7b6ac13b2cd9bf4.

    ---------------------------------------------------------------------------
    KeyError                                  Traceback (most recent call last)
    Cell In[6], line 5
        2 from qamomile.qiskit import QiskitTranspiler
        4 # Build a QAOA ansatz (p = 1)
    ----> 5 circuit_ir = QAOAConverter(compiled_instance).get_qaoa_ansatz(p=1)
        6 # Output platform-specific circuits
        7 qc_qiskit = QiskitTranspiler().transpile_circuit(circuit_ir)

    File ~/dev/git/Qamomile/qamomile/core/converters/qaoa.py:109, in QAOAConverter.get_qaoa_ansatz(self, p, initial_hadamard)
        96 def get_qaoa_ansatz(
        97     self, p: int, initial_hadamard: bool = True
        98 ) -> qm_c.QuantumCircuit:
        99      '''
        100     Generate the complete QAOA ansatz circuit.
        101
    (...)    107         qm_c.QuantumCircuit: The complete QAOA ansatz circuit.
        108     '''
    --> 109     ising = self.get_ising()
        110     num_qubits = ising.num_bits
        111     qaoa_circuit = qm_c.QuantumCircuit(num_qubits, 0, name="QAOA")

    File ~/dev/git/Qamomile/qamomile/core/converters/converter.py:192, in QuantumConverter.get_ising(self)
        184 '''
        185 Get the Ising model representation of the problem.
        186
    (...)    189
        190 '''
        191 if self._ising is None:
    --> 192     self._ising = self.ising_encode()
        193 return self._ising

    File ~/dev/git/Qamomile/qamomile/core/converters/converter.py:234, in QuantumConverter.ising_encode(self, multipliers, detail_parameters)
        232 deci_vars = {dv.id: dv for dv in self.original_instance.decision_variables}
        233 for ising_index, qubo_index in ising.index_map.items():
    --> 234     deci_var = deci_vars[qubo_index]
        235     # TODO: If use log encoding to represent an integer,
        236     #       var_name is ommx.log_encode and subscripts represents [original variable index, encoded binary index].
        237     #       Need to be fixed.
        238     var_name = deci_var.name

    KeyError: 7
    """
    # Define Knapsack problem
    v = jm.Placeholder("v", ndim=1)
    N = v.len_at(0, latex="N")
    w = jm.Placeholder("w", ndim=1)
    W = jm.Placeholder("W")
    x = jm.BinaryVar("x", shape=(N,))
    i = jm.Element("i", belong_to=(0, N))
    problem = jm.Problem("Knapsack", sense=jm.ProblemSense.MAXIMIZE)
    problem += jm.sum(i, v[i] * x[i])
    problem += jm.Constraint("weight", jm.sum(i, w[i] * x[i]) <= W)

    # Prepare instance data
    instance_data = {"v": [12, 7, 19, 5, 11, 3], "w": [4, 2, 7, 3, 6, 1], "W": 15}

    # create the problem model
    compiled_instance = jm.Interpreter(instance_data).eval_problem(problem)

    from qamomile.core.converters.qaoa import QAOAConverter

    # Build a QAOA ansatz (p = 1)
    QAOAConverter(compiled_instance).get_qaoa_ansatz(p=1)


@pytest.mark.parametrize(
    "instance_data",
    [
        {"N": 3, "a": [-1.0, 1.0, -1.0]},
        {"N": 4, "a": [0.5, -0.5, 0.5, -0.5]},
    ],
)
def test_n_body_problem(instance_data):
    """Run get_qaoa_ansatz and get_cost_hamiltonian for N-body problem with different instance data.

    Check if
    - no errors are raised.
    """
    # Get the N-body problem.
    n_body_problem = Utils.get_n_body_problem()
    # Get the ommx instance.
    interpreter = jm.Interpreter(instance_data)
    instance = interpreter.eval_problem(n_body_problem)
    # Create QAOA converter.
    qaoa_converter = QAOAConverter(instance)
    # Get ansatz and cost hamiltonian.
    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(p=1)
    qaoa_cost = qaoa_converter.get_cost_hamiltonian()


def test_decode_error():
    """This test raised the follwing error with the version whose commit ID 01402581faf790965440f9cedc87f1c6f63606b3.

    ---------------------------------------------------------------------------
    KeyError                                  Traceback (most recent call last)
    Cell In[19], line 8
        6 print(job_result.data["meas"].get_counts())
        7 print(qaoa_converter._ising)
    ----> 8 sampleset = qaoa_converter.decode(qk_transpiler, job_result.data["meas"])
        9 energies = []
        10 frequencies = []

    File ~/dev/git/Qamomile/qamomile/core/converters/converter.py:283, in QuantumConverter.decode(self, transpiler, result)
        268 '''
        269 Decode quantum computation results into a SampleSet.
        270
    (...)    280     ommx.v1.SampleSet: The decoded results as a SampleSet.
        281 '''
        282 bitssampleset = transpiler.convert_result(result)
    --> 283 return self.decode_bits_to_sampleset(bitssampleset)

    File ~/dev/git/Qamomile/qamomile/core/converters/converter.py:308, in QuantumConverter.decode_bits_to_sampleset(self, bitssampleset)
        306 sample = {}
        307 for i, bit in enumerate(bitssample.bits):
    --> 308     index = ising.ising2qubo_index(i)
        309     sample[index] = bit
        310 state = ommx.v1.State(entries=sample)

    File ~/dev/git/Qamomile/qamomile/core/ising_qubo.py:45, in IsingModel.ising2qubo_index(self, index)
        44 def ising2qubo_index(self, index: int) -> int:
    ---> 45     return self.index_map[index]

    KeyError: 0
    """
    # Define a problem having decision variables not starting from index 0.
    N = jm.Placeholder("N")
    x = jm.BinaryVar("x", shape=(N,))
    problem = jm.Problem("N-body problem")
    problem += x[1] * x[2]
    # Compile the problem.
    instance_data = {"N": 3}
    interpreter = jm.Interpreter(instance_data)
    instance = interpreter.eval_problem(problem)
    instance.objective.degree()
    # Get the circuit.
    qaoa_converter = QAOAConverter(instance)
    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(p=1)

    from qamomile.qiskit import QiskitTranspiler
    import qiskit.primitives as qk_pr

    # Transpile the circuit.
    qk_transpiler = QiskitTranspiler()
    qk_circuit = qk_transpiler.transpile_circuit(qaoa_circuit)

    # Run it on a simulator with random parameters.
    sampler = qk_pr.StatevectorSampler()
    qk_circuit.measure_all()
    parameters = {
        param: np.random.rand() * 2 * np.pi for param in qk_circuit.parameters
    }
    job = sampler.run([(qk_circuit, parameters)], shots=10000)
    job_result = job.result()[0]
    qaoa_converter.decode(qk_transpiler, job_result.data["meas"])
