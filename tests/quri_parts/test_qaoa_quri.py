import numpy as np
import jijmodeling as jm
import jijmodeling_transpiler as jmt
import jijmodeling_transpiler_quantum.quri_parts as jmt_qp
from quri_parts.core.operator import pauli_label, PAULI_IDENTITY, Operator


def test_qaoa_onehot():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)
    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qaoa_builder = jmt_qp.transpile_to_qaoa_ansatz(compiled_instance)
    qaoa_ansatz, cost_func, constant = qaoa_builder.get_qaoa_ansatz(p=1)

    # qubo:  (x0 + x1 + x2 - 1)^2 = - x0 - x1 - x2 + 2x0x1 + 2x1x2 + 2x0x2+ 1
    # ising: = - 1/2(1-z0) - 1/2(1-z1) - 1/2(1-z3) + 1/2(1-z0)(1-z1) + 1/2(1-z1)(1-z2) + 1/2(1-z0)(1-z2)+ 1
    #        = 1/2*z0*z1 - 0.5 + 1

    ANS_op = Operator(
        {
            pauli_label("Z0"): -1,
            pauli_label("Z1"): -1,
            pauli_label("Z2"): -1,
            pauli_label("Z0 Z1"): 0.5,
            pauli_label("Z0 Z2"): 0.5,
            pauli_label("Z1 Z2"): 0.5,
            PAULI_IDENTITY: 1.5,
        }
    )

    assert qaoa_ansatz.qubit_count == 3
    assert qaoa_ansatz.parameter_count == 2
    assert cost_func == ANS_op
    assert constant == 2.5


# def test_qaoa_H_eigenvalue():
#     n = jm.Placeholder("n")
#     x = jm.BinaryVar("x", shape=(n,))
#     i = jm.Element("i", belong_to=n)
#     problem = jm.Problem("sample")
#     problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)

#     compiled_instance = jmt.core.compile_model(problem, {"n": 2})

#     qaoa_builder = jmt_qp.transpile_to_qaoa_ansatz(
#         compiled_instance, relax_method=jmt.core.pubo.SquaredPenalty
#     )

#     # qubo:  (x0 + x1 - 1)^2 = - x0 - x1 + 2x0x1 + 1
#     # ising: = - 1/2(1-z0) - 1/2(1-z1) + 1/2(1-z0)(1-z1) + 1
#     #        = 1/2*z0*z1 - 0.5 + 1

#     hamiltonian, constant = qaoa_builder.get_hamiltonian()
#     print(hamiltonian)
#     eigen_solver = NumPyEigensolver()
#     result = eigen_solver.compute_eigenvalues(hamiltonian)
#     ising_optimal = (np.array(result.eigenvalues) + constant)[0].real

#     counts = result.eigenstates[0].sample_counts(shots=1)
#     sampleset = qaoa_builder.decode_from_counts(counts)
#     print()

#     assert len(sampleset.feasible().record.solution["x"]) == 1
#     assert ising_optimal == 0.0
