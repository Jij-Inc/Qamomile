from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel
from jijmodeling_transpiler_quantum.core.qrac import greedy_graph_coloring
from jijmodeling_transpiler_quantum.qiskit.qrao import qrac31_encode_ising


# TODO:test name
def test_greedy():
    ising = IsingModel({(0, 1): 2.0}, {2: 5.0}, 6.0)
    _, color_group = greedy_graph_coloring(
        ising.quad.keys(), max_color_group_size=3
    )
    qrac_hamiltonian, offset, encoding = qrac31_encode_ising(
        ising, color_group
    )
