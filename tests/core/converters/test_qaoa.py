# File: tests/core/qaoa/test_qaoa.py

import pytest
import numpy as np
import jijmodeling as jm
import jijmodeling_transpiler.core as jmt
from qamomile.core.converters.qaoa import QAOAConverter
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
import qamomile.core.bitssample as qm_bs


@pytest.fixture
def simple_qubo_problem():
    Q = jm.Placeholder("Q", ndim=2)
    n = Q.len_at(0, latex="n")
    x = jm.BinaryVar("x", shape=(n,))
    problem = jm.Problem("qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])
    instance_data = {"Q": [[0.1, 0.2], [0.2, 0.3]]}
    compiled_instance = jmt.compile_model(problem, instance_data)
    return compiled_instance


@pytest.fixture
def qaoa_converter(simple_qubo_problem):
    return QAOAConverter(simple_qubo_problem)


def test_get_cost_ansatz(qaoa_converter):
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

    assert isinstance(sampleset, jm.experimental.SampleSet)
    assert len(sampleset) == 2
    assert [sample.num_occurrences for sample in sampleset] == [3, 1]


def test_qaoa_converter_with_larger_problem():
    # Create a larger QUBO problem
    Q = jm.Placeholder("Q", ndim=2)
    n = Q.len_at(0, latex="n")
    x = jm.BinaryVar("x", shape=(n,))
    problem = jm.Problem("large_qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])
    instance_data = {"Q": np.random.rand(10, 10)}
    compiled_instance = jmt.compile_model(problem, instance_data)

    qaoa_converter = QAOAConverter(compiled_instance)

    # Test circuit generation
    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(p=3)
    assert qaoa_circuit.num_qubits == 10
    assert len(qaoa_circuit.get_parameters()) == 6  # 2 * p parameters

    # Test Hamiltonian generation
    hamiltonian = qaoa_converter.get_cost_hamiltonian()
    assert len(hamiltonian.terms) > 0
