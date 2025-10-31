import numpy as np
import jijmodeling as jm
import pytest

from qamomile.core.ising_qubo import IsingModel
from qamomile.core.converters.qrao.graph_coloring import (
    greedy_graph_coloring,
    check_linear_term,
)
from qamomile.core.converters.qrao.qrao31 import qrac31_encode_ising, QRAC31Converter
from qamomile.core.converters.qrao.qrao21 import qrac21_encode_ising, QRAC21Converter
from qamomile.core.converters.qrao.qrao_space_efficient import (
    numbering_space_efficient_encode,
    qrac_space_efficient_encode_ising,
    QRACSpaceEfficientConverter,
)
import qamomile.core.operator as qm_o

from tests.utils import Utils


def test_check_linear_term_qrao31():
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

    qrac_hamiltonian, encoding = qrac31_encode_ising(ising, color_group)
    num_terms = len(ising.linear.keys()) + len(ising.quad.keys())

    expected_hamiltonian = {
        (Z0, Z1): max_color_group_size * 2.0,
        (Z0, X1): max_color_group_size * 1.0,
        (X1,): np.sqrt(max_color_group_size) * 5.0,
        (Z2,): np.sqrt(max_color_group_size) * 2.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits
    assert len(encoding) == ising.num_bits
    assert qrac_hamiltonian.terms == expected_hamiltonian

    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
        6.0,
    )
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, encoding = qrac31_encode_ising(ising, color_group)

    num_terms = len(ising.linear.keys()) + len(ising.quad.keys())

    expected_hamiltonian = {
        (Z0, Z1): max_color_group_size * 2.0,
        (Z0, X1): max_color_group_size * 1.0,
        (X1,): np.sqrt(max_color_group_size) * 5.0,
        (Z2,): np.sqrt(max_color_group_size) * 2.0,
        (X2,): np.sqrt(max_color_group_size) * 1.0,
        (Y2,): np.sqrt(max_color_group_size) * 1.0,
        (Z3,): np.sqrt(max_color_group_size) * 1.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits
    assert len(encoding) == ising.num_bits
    assert qrac_hamiltonian.terms == expected_hamiltonian


def test_QRAC31Converter():
    problem = jm.Problem("sample")
    x = jm.BinaryVar("x", shape=(3,))
    problem += x[1]
    problem += jm.Constraint("const", x[0] + x[2] == 1)
    compiled_instance = jm.Interpreter({}).eval_problem(problem)

    converter = QRAC31Converter(compiled_instance)

    # Test get_cost_hamiltonian method
    cost_hamiltonian = converter.get_cost_hamiltonian()

    pauli_list = converter.get_encoded_pauli_list()
    assert len(pauli_list) == 3


def test_check_linear_term_qrao21():
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    X2 = qm_o.PauliOperator(qm_o.Pauli.X, 2)
    X3 = qm_o.PauliOperator(qm_o.Pauli.X, 3)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
    Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)
    Z3 = qm_o.PauliOperator(qm_o.Pauli.Z, 3)

    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
        6.0,
    )
    max_color_group_size = 2
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, encoding = qrac21_encode_ising(ising, color_group)

    num_terms = len(ising.linear.keys()) + len(ising.quad.keys())

    expected_hamiltonian = {
        (Z0, Z1): max_color_group_size * 2.0,
        (Z0, X1): max_color_group_size * 1.0,
        (X1,): np.sqrt(max_color_group_size) * 5.0,
        (Z2,): np.sqrt(max_color_group_size) * 2.0,
        (X2,): np.sqrt(max_color_group_size) * 1.0,
        (Z3,): np.sqrt(max_color_group_size) * 1.0,
        (X3,): np.sqrt(max_color_group_size) * 1.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits
    assert len(encoding) == ising.num_bits
    assert qrac_hamiltonian.terms == expected_hamiltonian


def test_check_no_quad_term_quri():
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)

    ising = IsingModel({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, encoding = qrac31_encode_ising(ising, color_group)

    num_terms = len(ising.linear.keys()) + len(ising.quad.keys())

    expected_hamiltonian = {
        (Z0,): np.sqrt(max_color_group_size) * 1.0,
        (X0,): np.sqrt(max_color_group_size) * 1.0,
        (Y0,): np.sqrt(max_color_group_size) * 5.0,
        (Z1,): np.sqrt(max_color_group_size) * 2.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits
    assert len(encoding) == ising.num_bits
    assert qrac_hamiltonian.terms == expected_hamiltonian

    ising = IsingModel({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 2
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, encoding = qrac21_encode_ising(ising, color_group)

    num_terms = len(ising.linear.keys()) + len(ising.quad.keys())

    expected_hamiltonian = {
        (Z0,): np.sqrt(max_color_group_size) * 1.0,
        (X0,): np.sqrt(max_color_group_size) * 1.0,
        (Z1,): np.sqrt(max_color_group_size) * 5.0,
        (X1,): np.sqrt(max_color_group_size) * 2.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits
    assert len(encoding) == ising.num_bits
    assert qrac_hamiltonian.terms == expected_hamiltonian


def test_QRAC21Converter():
    problem = jm.Problem("sample")
    x = jm.BinaryVar("x", shape=(3,))
    problem += x[1]
    problem += jm.Constraint("const", x[0] + x[2] == 1)
    compiled_instance = jm.Interpreter({}).eval_problem(problem)

    converter = QRAC21Converter(compiled_instance)

    # Test get_cost_hamiltonian method
    cost_hamiltonian = converter.get_cost_hamiltonian()

    pauli_list = converter.get_encoded_pauli_list()
    assert len(pauli_list) == 3


def test_numbering_space_efficient_encode():
    ising = IsingModel({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0}, 6.0)
    encoding = numbering_space_efficient_encode(ising)
    expected_encoding = {
        0: qm_o.PauliOperator(qm_o.Pauli.X, 0),
        1: qm_o.PauliOperator(qm_o.Pauli.Y, 0),
        2: qm_o.PauliOperator(qm_o.Pauli.X, 1),
        3: qm_o.PauliOperator(qm_o.Pauli.Y, 1),
    }
    assert encoding == expected_encoding


def test_qrac_space_efficient_encode_ising():
    ising = IsingModel({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0}, 6.0)
    expected_hamiltonian = qm_o.Hamiltonian()
    expected_hamiltonian.constant = 6.0

    expected_hamiltonian.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 1),), np.sqrt(3) * 5.0
    )
    expected_hamiltonian.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1),), np.sqrt(3) * 2.0
    )
    expected_hamiltonian.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.X, 1)),
        3 * 1.0,
    )
    expected_hamiltonian.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, 0),), np.sqrt(3) * 2.0
    )

    expected_encoding = {
        0: qm_o.PauliOperator(qm_o.Pauli.X, 0),
        1: qm_o.PauliOperator(qm_o.Pauli.Y, 0),
        2: qm_o.PauliOperator(qm_o.Pauli.X, 1),
        3: qm_o.PauliOperator(qm_o.Pauli.Y, 1),
    }

    hamiltonian, encoding = qrac_space_efficient_encode_ising(ising)

    assert hamiltonian == expected_hamiltonian
    assert encoding == expected_encoding


def test_QRACSpaceEfficientConverter():
    problem = jm.Problem("sample")
    x = jm.BinaryVar("x", shape=(3,))
    problem += x[0] + x[1]
    problem += jm.Constraint("const", x[0] + x[1] + x[2] == 1)
    compiled_instance = jm.Interpreter({}).eval_problem(problem)

    # Create an instance of QRACSpaceEfficientConverter
    converter = QRACSpaceEfficientConverter(compiled_instance)

    # Test get_cost_hamiltonian method
    cost_hamiltonian = converter.get_cost_hamiltonian()
    assert converter.num_qubits == 2
    pauli_list = converter.get_encoded_pauli_list()
    assert len(pauli_list) == 3
    assert np.all(pauli_list == [qm_o.X(0), qm_o.Y(0), qm_o.X(1)])


@pytest.mark.parametrize(
    "instance_data",
    [
        {"N": 3, "a": [-1.0, 1.0, -1.0]},
        {"N": 4, "a": [0.5, -0.5, 0.5, -0.5]},
    ],
)
@pytest.mark.parametrize(
    "converter_class", [QRAC21Converter, QRAC31Converter, QRACSpaceEfficientConverter]
)
def test_n_body_problem(converter_class, instance_data):
    """Create converter_class with a HUBO problem.

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
        converter_class(instance)
