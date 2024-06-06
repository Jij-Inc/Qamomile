from qamomile.core import Hamiltonian, pauli_x, pauli_y, pauli_z
import jijmodeling as jm


def test_Hamiltonian():
    X = pauli_x(10)
    Y = pauli_y(10)
    Z = pauli_z(10)
    i = jm.Element("i", belong_to=(0, 10))
    N = jm.Placeholder("N")
    expr = jm.sum(
        [i], X[i] * X[(i + 1) % N] + Y[i] * Y[(i + 1) % N] + Z[i] * Z[(i + 1) % N]
    )
    op = Hamiltonian(expr, "Heisenberg_model")

    assert op.name == "Heisenberg_model"
    assert jm.is_same(op.hamiltonian, expr)

    X = pauli_x(10)
    Y = pauli_y(10)
    Z = pauli_z(10)
    i = jm.Element("i", belong_to=(0, 10))
    N = jm.Placeholder("N")
    expr = jm.sum(
        [i], X[i] * X[(i + 1) % N] + Y[i] * Y[(i + 1) % N] + Z[i] * Z[(i + 1) % N]
    )
    op = Hamiltonian(expr)

    assert op.name == ""
    assert jm.is_same(op.hamiltonian, expr)
