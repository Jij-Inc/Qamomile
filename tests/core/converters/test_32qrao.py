import numpy as np
import jijmodeling as jm
import pytest

from qamomile.core.ising_qubo import IsingModel
from qamomile.core.converters.qrao.qrao32 import (
    create_x_prime,
    create_y_prime,
    create_z_prime,
    qrac32_encode_ising,
    create_prime_operator,
    QRAC32Converter,
)
import qamomile.core.operator as qm_o
from qamomile.core.converters.qrao.graph_coloring import (
    greedy_graph_coloring,
    check_linear_term,
)

from tests.utils import Utils


def test_create_x_prime():
    X_prime = create_x_prime(0)
    coeff = 1.0 / np.sqrt(6)
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
    expected = qm_o.Hamiltonian()
    expected.add_term((X0, X1), 1 / 2 * coeff)
    expected.add_term((X0, Z1), 1 / 2 * coeff)
    expected.add_term((Z0,), 1 * coeff)
    assert X_prime == expected


def test_create_y_prime():
    Y_prime = create_y_prime(0)
    coeff = 1.0 / np.sqrt(6)
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    expected = qm_o.Hamiltonian()
    expected.add_term((X1,), 1 / 2 * coeff)
    expected.add_term((Z1,), 1 * coeff)
    expected.add_term((Y0, Y1), 1 / 2 * coeff)
    assert Y_prime == expected


def test_create_z_prime():
    Z_prime = create_z_prime(0)
    coeff = 1.0 / np.sqrt(6)
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
    expected = qm_o.Hamiltonian()
    expected.add_term((Z0, Z1), 1 * coeff)
    expected.add_term((X0,), -1 / 2 * coeff)
    expected.add_term((Z0, X1), -1 / 2 * coeff)
    assert Z_prime == expected


def test_qrac32_encode_ising():
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    X2 = qm_o.PauliOperator(qm_o.Pauli.X, 2)
    Y2 = qm_o.PauliOperator(qm_o.Pauli.Y, 2)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
    Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)
    Z3 = qm_o.PauliOperator(qm_o.Pauli.Z, 3)

    ising = IsingModel({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0}, 6.0)

    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, encoding = qrac32_encode_ising(ising, color_group)
    num_terms = len(ising.linear.keys()) + len(ising.quad.keys())

    expected_hamiltonian = qm_o.Hamiltonian()
    expected_hamiltonian.constant = 6.0
    Z0_prime = create_prime_operator(Z0)
    Z1_prime = create_prime_operator(Z1)
    Z2_prime = create_prime_operator(Z2)
    X1_prime = create_prime_operator(X1)

    expected_hamiltonian += 6 * 2.0 * Z0_prime * Z1_prime
    expected_hamiltonian += 6 * 1.0 * Z0_prime * X1_prime
    expected_hamiltonian += np.sqrt(6) * 5.0 * X1_prime
    expected_hamiltonian += np.sqrt(6) * 2.0 * Z2_prime
    # expected_hamiltonian = {
    #     (Z0, Z1): max_color_group_size * 2.0,
    #     (Z0, X1): max_color_group_size * 1.0,
    #     (X1,): np.sqrt(max_color_group_size) * 5.0,
    #     (Z2,): np.sqrt(max_color_group_size) * 2.0,
    # }

    assert len(encoding) == ising.num_bits
    assert qrac_hamiltonian == expected_hamiltonian


def test_QRAC32Converter():
    problem = jm.Problem("sample")
    x = jm.BinaryVar("x", shape=(3,))
    problem += x[1] * x[2] + x[0]
    instance = jm.Interpreter({}).eval_problem(problem)

    converter = QRAC32Converter(instance)

    # Test get_cost_hamiltonian method
    cost_hamiltonian = converter.get_cost_hamiltonian()

    pauli_list = converter.get_encoded_pauli_list()
    assert len(pauli_list) == 3


@pytest.mark.parametrize(
    "instance_data",
    [
        {"N": 3, "a": [-1.0, 1.0, -1.0]},
        {"N": 4, "a": [0.5, -0.5, 0.5, -0.5]},
    ],
)
def test_n_body_problem(instance_data):
    """Create 31QRACConverter with a HUBO problem.

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
        QRAC32Converter(instance)
