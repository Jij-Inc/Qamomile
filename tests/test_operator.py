import qamomile.core.operator as qm_o

def test_pauli_operator_creation():
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    assert X0.pauli == qm_o.Pauli.X
    assert X0.index == 0

    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    assert Y1.pauli == qm_o.Pauli.Y
    assert Y1.index == 1

    Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)
    assert Z2.pauli == qm_o.Pauli.Z
    assert Z2.index == 2

# def test_multiplication_by_scalar():
#     X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
#     h = qm_o.Hamiltonian()
#     h.add_term((X0,), 2.0)
#     op = X0 * 2.0
#     assert h == op

#     Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
#     h = qm_o.Hamiltonian()
#     h.add_term((Y1,), 3.0)
#     op = 3.0 * Y1
#     assert h == op

# def test_muliiplication_by_operator():
#     X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
#     Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
#     h = qm_o.Hamiltonian()
#     h.add_term((X0, Y1), 1.0)
#     op = X0 * Y1
#     assert h == op

#     X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
#     Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
#     op = X0 * Y0
#     h = qm_o.Hamiltonian()
#     h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0), ), 1.0j)
#     assert h == op

#     op = Y0 * X0
#     h = qm_o.Hamiltonian()
#     h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0), ), -1.0j)
#     assert h == op

def test_pauli_multiplication():
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0), ), 1.0j)
    assert h == qm_o.pauli_multiplication(X0, Y0)

    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0), ), -1.0j)
    assert h == qm_o.pauli_multiplication(X0, Z0)

    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0), ), 1.0j)
    assert h == qm_o.pauli_multiplication(Y0, Z0)

    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    h = qm_o.Hamiltonian()
    assert h == qm_o.pauli_multiplication(X0, X0)

    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    h = qm_o.Hamiltonian()
    assert h == qm_o.pauli_multiplication(Y0, Y0)

    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    h = qm_o.Hamiltonian()
    assert h == qm_o.pauli_multiplication(Z0, Z0)

def test_Hamiltonian_add():
    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 1.0)
    h = h1 + h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 1.0)
    assert h == expected_h

def test_Hamiltonian_add_scalar():
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h = h + 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.constant = 2.0
    assert h == expected_h

    h = 2.0 + h
    expected_h.constant += 2.0
    assert h == expected_h

# def test_Hamiltonian_multiplication():
#     h = qm_o.Hamiltonian()
#     h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
#     h = h * 2.0
#     expected_h = qm_o.Hamiltonian()
#     expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 2.0)
#     assert h == expected_h