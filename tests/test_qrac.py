import jijmodeling as jm
import jijmodeling_transpiler.core as jtc
import jijmodeling_transpiler_quantum.qiskit as jtqk
import jijmodeling_transpiler_quantum.quri_parts as jtqp
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel, qubo_to_ising
from jijmodeling_transpiler_quantum.core.qrac import (
    check_linear_term,
    greedy_graph_coloring,
)


def test_check_linear_term_qiskit():
    ising = IsingModel({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, offset, encoding = jtqk.qrao.qrac31_encode_ising(
        ising, color_group
    )

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

    qrac_hamiltonian, offset, encoding = jtqk.qrao.qrac31_encode_ising(
        ising, color_group
    )

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

    qrac_hamiltonian, offset, encoding = jtqk.qrao.qrac21_encode_ising(
        ising, color_group
    )


def test_check_linear_term_quri():
    ising = IsingModel({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, offset, encoding = jtqp.qrao.qrac31_encode_ising(
        ising, color_group
    )

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

    qrac_hamiltonian, offset, encoding = jtqp.qrao.qrac31_encode_ising(
        ising, color_group
    )

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

    qrac_hamiltonian, offset, encoding = jtqp.qrao.qrac21_encode_ising(
        ising, color_group
    )


def test_check_no_quad_term_quri():
    ising = IsingModel({}, {0:1.0,1:1.0,2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, offset, encoding = jtqp.qrao.qrac31_encode_ising(
        ising, color_group
    )

    ising = IsingModel({}, {2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, offset, encoding = jtqp.qrao.qrac31_encode_ising(
        ising, color_group
    )

    ising = IsingModel(
        {},
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

    qrac_hamiltonian, offset, encoding = jtqp.qrao.qrac21_encode_ising(
        ising, color_group
    )
    