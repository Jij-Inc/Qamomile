import qamomile.core.modeler as qm_m
import qamomile.core.operator as qm_o
import jijmodeling as jm
import networkx as nx

def test_HamiltonianExpr():
    N = jm.Placeholder("N")
    Z = qm_m.PauliExpr.z(shape=(N,))
    i = jm.Element("i", belong_to=(0, N))
    expr = jm.sum(i, Z[i])

    hamiltonian_expr = qm_m.HamiltonianExpr(expr, name="test")
    assert str(hamiltonian_expr.hamiltonian) == str(expr)
    assert hamiltonian_expr.name == "test"
    assert hamiltonian_expr._repr_latex_() == expr._repr_latex_()

def test_HamiltonianBuilder_simple():
    X = qm_m.PauliExpr.x(10)
    i = jm.Element("i", belong_to=(0, 10))
    expr = jm.sum(i, X[i])

    hamiltonian_expr = qm_m.HamiltonianExpr(expr, name="test")
    builder = qm_m.HamiltonianBuilder(hamiltonian_expr, {})
    op = builder.build()

    expected_hamiltonian = qm_o.Hamiltonian()
    for i in range(10):
        expected_hamiltonian += qm_o.X(i)
    
    assert op == expected_hamiltonian

    N = jm.Placeholder("N")
    Z = qm_m.PauliExpr.z(shape=(N,))
    i = jm.Element("i", belong_to=(0, N))
    expr = jm.sum(i, Z[i])

    hamiltonian_expr = qm_m.HamiltonianExpr(expr, name="test")
    instance_data = {"N": 3}
    builder = qm_m.HamiltonianBuilder(hamiltonian_expr, instance_data)
    op = builder.build()

    expected_hamiltonian = qm_o.Hamiltonian()
    for i in range(3):
        expected_hamiltonian += qm_o.Z(i)
    
    assert op == expected_hamiltonian

def test_HamiltonianBuilder_graph():
    E = jm.Placeholder("E", ndim=2)
    e = jm.Element("e", belong_to=E)
    Z = qm_m.PauliExpr.z(shape=3)
    expr = jm.sum(e, Z[e[0]] * Z[e[1]])
    
    hamiltonian_expr = qm_m.HamiltonianExpr(expr, name="test")
    
    E = [[0, 1], [1, 2], [2, 0]]
    
    builder = qm_m.HamiltonianBuilder(hamiltonian_expr, {"E": E})
    op = builder.build()

    expected_hamiltonian = qm_o.Hamiltonian()
    for i,j in E:
        expected_hamiltonian += qm_o.Z(i) * qm_o.Z(j)

    assert op == expected_hamiltonian

    