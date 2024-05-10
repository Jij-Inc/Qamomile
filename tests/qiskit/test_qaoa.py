import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import qiskit as qk
from qiskit.algorithms.eigensolvers import NumPyEigensolver

import qamomile.qiskit as jmt_qk


def test_qaoa_onehot():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qaoa_builder = jmt_qk.transpile_to_qaoa_ansatz(compiled_instance)

    qaoa_ansatz, cost_func, constant = qaoa_builder.get_qaoa_ansatz(p=1)


def test_qaoa_H_eigenvalue():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 2})

    qaoa_builder = jmt_qk.transpile_to_qaoa_ansatz(
        compiled_instance, relax_method=jmt.core.pubo.SquaredPenalty
    )

    # qubo:  (x0 + x1 - 1)^2 = - x0 - x1 + 2x0x1 + 1
    # ising: = - 1/2(1-z0) - 1/2(1-z1) + 1/2(1-z0)(1-z1) + 1
    #        = 1/2*z0*z1 - 0.5 + 1
    hamiltonian, constant = qaoa_builder.get_hamiltonian()
    eigen_solver = NumPyEigensolver()
    result = eigen_solver.compute_eigenvalues(hamiltonian)
    ising_optimal = (np.array(result.eigenvalues) + constant)[0].real

    num_shots = 1
    counts = result.eigenstates[0].sample_counts(shots=num_shots)
    sampleset = qaoa_builder.decode_from_counts(counts)

    assert len(sampleset.feasible().record.solution["x"]) == 1
    assert ising_optimal == 0.0
    assert sampleset.record.num_occurrences == [num_shots]

    num_shots = 10
    counts = result.eigenstates[0].sample_counts(shots=num_shots)
    sampleset = qaoa_builder.decode_from_counts(counts)

    assert len(sampleset.feasible().record.solution["x"]) == 1
    assert ising_optimal == 0.0
    assert sampleset.record.num_occurrences == [num_shots]
