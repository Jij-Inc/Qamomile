import qiskit as qk
import qiskit.quantum_info as qk_ope
import numpy as np
import jijmodeling as jm
import jijmodeling_transpiler as jmt

from jijmodeling_transpiler_quantum.qiskit import (
    transpile_to_qrac_space_efficient_hamiltonian,
    transpile_to_qrac31_hamiltonian,
    transpile_to_qrac21_hamiltonian,
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

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qrac_builder = transpile_to_qrac32_hamiltonian(compiled_instance)

    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian(
        multipliers={"onehot": 1.0}
    )


def test_transpile_to_qrac_space_efficient_hamiltonian():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qrac_builder = transpile_to_qrac_space_efficient_hamiltonian(
        compiled_instance
    )

    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian(
        multipliers={"onehot": 1.0}
    )
    assert encoding.color_group == {}
