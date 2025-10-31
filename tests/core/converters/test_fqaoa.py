# File: tests/core/qaoa/test_qaoa.py

import pytest
import numpy as np
import jijmodeling as jm
import ommx.v1
import qiskit.primitives as qk_pr

from qamomile.qiskit import QiskitTranspiler
from qamomile.core.converters.fqaoa import FQAOAConverter
from qamomile.core.ising_qubo import IsingModel
import qamomile.core.circuit as qm_c
from qamomile.core.circuit.circuit import ParametricTwoQubitGate
import qamomile.core.operator as qm_o
import qamomile.core.bitssample as qm_bs

from tests.utils import Utils


@pytest.fixture
def simple_problem():
    J = jm.Placeholder("J", ndim=2)
    n = J.len_at(0, latex="n")
    D = jm.Placeholder("D")
    x = jm.BinaryVar("x", shape=(n, D))

    problem = jm.Problem("qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    d, d_dash = jm.Element("d", D), jm.Element("d'", D)
    problem += jm.sum([i, j], J[i, j] * jm.sum([d, d_dash], x[i, d] * x[j, d_dash]))
    problem += jm.Constraint("constraint", jm.sum([i, d], x[i, d]) == 4)

    instance_data = {
        "J": [
            [0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "D": 2,
    }
    instance = jm.Interpreter(instance_data).eval_problem(problem)

    return instance


def test_initializaiton(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)

    assert fqaoa_converter.num_integers == 4
    assert fqaoa_converter.num_bits == 2
    assert fqaoa_converter.num_fermions == 4
    assert isinstance(fqaoa_converter.var_map, dict)
    assert isinstance(fqaoa_converter.ising, IsingModel)
    assert fqaoa_converter.num_qubits == 8

    print(fqaoa_converter.int2varlabel)


def test_fqaoa_instance_to_qubo(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    num_constraints = len(simple_problem.constraints)

    qubo, constant = fqaoa_converter.fqaoa_instance_to_qubo()

    assert len(simple_problem.constraints) == num_constraints


def test_fqaoa_ising_encode(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    ising = fqaoa_converter.fqaoa_ising_encode()

    assert fqaoa_converter.int2varlabel == {
        0: "x_{0,0}",
        1: "x_{1,0}",
        2: "x_{2,0}",
        3: "x_{3,0}",
        4: "x_{0,1}",
        5: "x_{1,1}",
        6: "x_{2,1}",
        7: "x_{3,1}",
    }


def test_cyclic_mapping(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)

    assert fqaoa_converter.var_map == {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (3, 0): 3,
        (0, 1): 4,
        (1, 1): 5,
        (2, 1): 6,
        (3, 1): 7,
    }


def test_get_init_state(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    init_circuit = fqaoa_converter.get_init_state()

    assert isinstance(init_circuit, qm_c.QuantumCircuit)
    assert init_circuit.num_qubits == 8
    assert len(init_circuit.gates) > 0

    qk_transpiler = QiskitTranspiler()
    qk_test_circuit = qk_transpiler.transpile_circuit(init_circuit)

    sampler = qk_pr.StatevectorSampler()
    qk_test_circuit.measure_all()
    job = sampler.run([qk_test_circuit], shots=1000)
    job_result = job.result()[0]

    test_sampleset = fqaoa_converter.decode(qk_transpiler, job_result.data["meas"])
    df_test_sampleset = test_sampleset.summary

    assert (df_test_sampleset["feasible"] == True).sum() == 1000


def test_get_mixer_ansatz(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    mixer_circuit = fqaoa_converter.get_mixer_ansatz(beta=2.0)

    assert isinstance(mixer_circuit, qm_c.QuantumCircuit)
    assert mixer_circuit.num_qubits == 8
    assert len(mixer_circuit.gates) > 0

    test_circuit = fqaoa_converter.get_init_state()
    test_circuit.append(mixer_circuit)
    qk_transpiler = QiskitTranspiler()
    qk_test_circuit = qk_transpiler.transpile_circuit(test_circuit)

    sampler = qk_pr.StatevectorSampler()
    qk_test_circuit.measure_all()
    job = sampler.run([qk_test_circuit], shots=1000)
    job_result = job.result()[0]

    test_sampleset = fqaoa_converter.decode(qk_transpiler, job_result.data["meas"])
    df_test_sampleset = test_sampleset.summary

    assert (df_test_sampleset["feasible"] == True).sum() == 1000


def test_get_cost_ansatz(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    cost_circuit = fqaoa_converter.get_cost_ansatz(gamma=2.0)

    # circuit
    assert isinstance(cost_circuit, qm_c.QuantumCircuit)
    assert cost_circuit.num_qubits == 8
    assert len(cost_circuit.gates) > 0

    # gate
    for gate in cost_circuit.gates:
        if isinstance(gate, ParametricTwoQubitGate):
            interaction = (gate.control, gate.target)
            assert interaction in fqaoa_converter.ising.quad

    test_circuit = fqaoa_converter.get_init_state()
    test_circuit.append(cost_circuit)
    qk_transpiler = QiskitTranspiler()
    qk_test_circuit = qk_transpiler.transpile_circuit(test_circuit)

    sampler = qk_pr.StatevectorSampler()
    qk_test_circuit.measure_all()
    job = sampler.run([qk_test_circuit], shots=1000)
    job_result = job.result()[0]

    test_sampleset = fqaoa_converter.decode(qk_transpiler, job_result.data["meas"])
    df_test_sampleset = test_sampleset.summary

    assert (df_test_sampleset["feasible"] == True).sum() == 1000


@pytest.mark.parametrize(
    "instance_data",
    [
        {"N": 3, "a": [-1.0, 1.0, -1.0]},
        {"N": 4, "a": [0.5, -0.5, 0.5, -0.5]},
    ],
)
def test_n_body_problem_with_constraints(instance_data):
    """Create FQAOAConverter with a HUBO problem.

    Check if
    - ValueError is raised.
    """
    # Get the N-body problem.
    n_body_problem = Utils.get_n_body_problem()
    # Get the ommx instance.
    interpreter = jm.Interpreter(instance_data)
    instance = interpreter.eval_problem(n_body_problem)
    # - ValueError is raised.
    with pytest.raises(ValueError):
        FQAOAConverter(instance, num_fermions=0)
