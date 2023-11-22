import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import qiskit as qk

from jijmodeling_transpiler_quantum.quri_parts import (
    transpile_to_qrac21_hamiltonian,
    transpile_to_qrac31_hamiltonian,
    transpile_to_qrac32_hamiltonian,
)


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
