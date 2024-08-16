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
        (qm_o.X(0), qm_o.X(1)): max_color_group_size * 2.0,
        (qm_o.X(0), qm_o.Y(1)): max_color_group_size * 1.0,
        (qm_o.Y(1),): np.sqrt(max_color_group_size) * 5.0,
        (qm_o.X(2),): np.sqrt(max_color_group_size) * 2.0,
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
        (qm_o.X(0), qm_o.X(1)): max_color_group_size * 2.0,
        (qm_o.X(0), qm_o.Y(1)): max_color_group_size * 1.0,
        (qm_o.Y(1),): np.sqrt(max_color_group_size) * 5.0,
        (qm_o.X(2),): np.sqrt(max_color_group_size) * 2.0,
        (qm_o.Y(2),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.Z(2),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.X(3),): np.sqrt(max_color_group_size) * 1.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
    assert qrac_hamiltonian.terms == expected_hamiltonian


def test_check_linear_term_qrao21():
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
        (qm_o.X(0), qm_o.X(1)): max_color_group_size * 2.0,
        (qm_o.X(0), qm_o.Y(1)): max_color_group_size * 1.0,
        (qm_o.Y(1),): np.sqrt(max_color_group_size) * 5.0,
        (qm_o.X(2),): np.sqrt(max_color_group_size) * 2.0,
        (qm_o.Y(2),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.X(3),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.Y(3),): np.sqrt(max_color_group_size) * 1.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
    assert qrac_hamiltonian.terms == expected_hamiltonian


def test_check_no_quad_term_quri():
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
        (qm_o.X(0),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.Y(0),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.Z(0),): np.sqrt(max_color_group_size) * 5.0,
        (qm_o.X(1),): np.sqrt(max_color_group_size) * 2.0,
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
        (qm_o.X(0),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.Y(0),): np.sqrt(max_color_group_size) * 1.0,
        (qm_o.X(1),): np.sqrt(max_color_group_size) * 5.0,
        (qm_o.Y(1),): np.sqrt(max_color_group_size) * 2.0,
    }
    assert len(qrac_hamiltonian.terms) == num_terms
    assert qrac_hamiltonian.num_qubits < ising.num_bits()
    assert len(encoding) == ising.num_bits()
    assert qrac_hamiltonian.terms == expected_hamiltonian
