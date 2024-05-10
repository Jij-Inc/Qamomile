import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import qiskit as qk

from qamomile.quri_parts import (
    transpile_to_qrac21_hamiltonian,
    transpile_to_qrac31_hamiltonian,
    transpile_to_qrac32_hamiltonian,
)


def qubo_problem():
    Q = jm.Placeholder("Q", ndim=2)
    x = jm.BinaryVar("x",shape=(Q.shape[0],))
    i = jm.Element("i", belong_to=(0, Q.shape[0]))
    j = jm.Element("j", belong_to=(0, Q.shape[0]))
    problem = jm.Problem("QUBO")
    problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])
    return problem

def test_transpile_to_qrac31_hamiltonian():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qrac_builder = transpile_to_qrac31_hamiltonian(compiled_instance)

    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian(
        multipliers={"onehot": 1.0}
    )


def test_transpile_to_qrac21_hamiltonian():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qrac_builder = transpile_to_qrac21_hamiltonian(compiled_instance)

    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian(
        multipliers={"onehot": 1.0}
    )


def test_transpile_to_qrac32_hamiltonian():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 6})
    qrac_builder = transpile_to_qrac32_hamiltonian(compiled_instance)

    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian(
        multipliers={"onehot": 1.0}
    )

def test_check_linear_term_from_qubo():
    qubo = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    instance_data = {"Q": qubo}
    compiled_model = jmt.core.compile_model(qubo_problem(), instance_data,{})

    qrac_builder = transpile_to_qrac31_hamiltonian(compiled_model)
    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian()

    qrac_builder = transpile_to_qrac21_hamiltonian(compiled_model)
    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian()

    qrac_builder = transpile_to_qrac32_hamiltonian(compiled_model)
    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian()