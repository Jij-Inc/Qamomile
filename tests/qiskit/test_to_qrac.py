import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import qiskit as qk
import qiskit.quantum_info as qk_ope
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.primitives import Estimator, Sampler
from scipy.optimize import minimize

import qamomile.qiskit as jtq
from qamomile.qiskit import (
    transpile_to_qrac21_hamiltonian,
    transpile_to_qrac31_hamiltonian,
    transpile_to_qrac32_hamiltonian,
    transpile_to_qrac_space_efficient_hamiltonian,
)

def qubo_problem():
    Q = jm.Placeholder("Q", ndim=2)
    x = jm.BinaryVar("x",shape=(Q.shape[0],))
    i = jm.Element("i", belong_to=(0, Q.shape[0]))
    j = jm.Element("j", belong_to=(0, Q.shape[0]))
    problem = jm.Problem("QUBO")
    problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])
    return problem

def calculate_eigenvalue_three_one_pauli(
    hamiltonian: qk_ope.SparsePauliOp,
    encoding_cache: jtq.qrao.to_qrac.QRACEncodingCache,
    num_qubits: int,
):
    variable_ops = create_rounding_operator(encoding_cache, num_qubits)

    eigen_solver = NumPyMinimumEigensolver()
    result = eigen_solver.compute_minimum_eigenvalue(
        operator=hamiltonian, aux_operators=variable_ops
    )

    return pauli_rounding(result.aux_operators_evaluated)


def create_rounding_operator(
    encoding_cache: jtq.qrao.to_qrac.QRACEncodingCache, num_qubits: int
) -> list[qk_ope.SparsePauliOp]:
    """creating rounding operator

        Pauli Rounding is a method for estimating the state by calculating the expectation value of Pauli operators for the encoded state.

    Args:
        encoding_cache (jtq.qrao.QracEncodingCache): encoding_cache
        num_qubits (int): number of qubits

    Returns:
        list[qk_info.SparsePauliOp]: List of the rounding operator
    """
    encoded_op = encoding_cache.encoding
    sorted_op = sorted(encoded_op.items())

    variable_ops = [
        jtq.qrao.qrao31.create_pauli_term([pauli_kind], [color], num_qubits)
        for idx, (color, pauli_kind) in sorted_op
    ]
    return variable_ops


def pauli_rounding(aux_op_evaluated: list[tuple[complex, dict]]):
    """do pauli rounding

    Return the rounded value of the Pauli operator expectation value.

    Args:
        aux_op_evaluated (list[tuple[complex, dict]]): List of the expectation value of the Pauli operator

    """
    pauli_base_state = [val.real for val, _ in aux_op_evaluated]

    rng = np.random.default_rng()

    def sign(val) -> int:
        return 0 if (val > 0) else 1

    rounded_vars = np.array(
        [
            sign(e) if not np.isclose(0, e) else rng.choice([0, 1])
            for e in pauli_base_state
        ]
    )
    return rounded_vars


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


def test_transpile_to_qrac31_decode():
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
    pauli_vars = calculate_eigenvalue_three_one_pauli(
        qrac_hamiltonian, encoding, 3
    )

    sampleset = qrac_builder.decode_from_binary_values([pauli_vars])

    assert len(sampleset.feasible().record.solution["x"]) == 1


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

    qrac_builder = transpile_to_qrac_space_efficient_hamiltonian(compiled_model)
    qrac_hamiltonian, offset, encoding = qrac_builder.get_hamiltonian()