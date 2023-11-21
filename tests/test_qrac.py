import jijmodeling_transpiler_quantum.qiskit as jtq
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel
from jijmodeling_transpiler_quantum.core.qrac import (
    check_linear_term,
    greedy_graph_coloring,
)
from jijmodeling_transpiler_quantum.qiskit.qrao import (
    qrac21_encode_ising,
    qrac31_encode_ising,
)


def test_check_linear_term():
    ising = IsingModel({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0}, 6.0)
    max_color_group_size = 3
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=max_color_group_size
    )
    color_group = check_linear_term(
        color_group, ising.linear.keys(), max_color_group_size
    )

    qrac_hamiltonian, offset, encoding = qrac31_encode_ising(
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

    qrac_hamiltonian, offset, encoding = qrac31_encode_ising(
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

    qrac_hamiltonian, offset, encoding = qrac21_encode_ising(
        ising, color_group
    )
