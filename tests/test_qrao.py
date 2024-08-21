import numpy as np
from qamomile.core.ising_qubo import IsingModel, qubo_to_ising
from qamomile.core.converters.qrao.graph_coloring import (
    greedy_graph_coloring,
    check_linear_term,
)
from qamomile.core.converters.qrao.qrao31 import qrac31_encode_ising
from qamomile.core.converters.qrao.qrao21 import qrac21_encode_ising
import qamomile.core.operator as qm_o


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
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
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
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
    assert qrac_hamiltonian.terms == expected_hamiltonian


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
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
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
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
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
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
    assert qrac_hamiltonian.terms == expected_hamiltonian
