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


def test_add_term():
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    Y2 = qm_o.PauliOperator(qm_o.Pauli.Y, 2)

    h = qm_o.Hamiltonian()
    h.add_term((X0,), 1.0)
    assert h.terms == {(X0,): 1.0}

    h.add_term((Y1, Y2), 3.0)
    assert h.terms == {(X0,): 1.0, (Y1, Y2): 3.0}

    h.add_term((X0,), 1.0)
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}

    h.add_term((X0, X0), -1.0)
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}
    assert h.constant == -1.0

    h.add_term((X0, Y0), -4.0)
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0, (Z0,): -4.0j}
    assert h.constant == -1.0
    

def test_pauli_hamiltonian_creation():
    x0 = qm_o.X(0)
    _x0 = qm_o.Hamiltonian()
    _x0.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    assert x0 == _x0

    y1 = qm_o.Y(1)
    _y1 = qm_o.Hamiltonian()
    _y1.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 1.0)
    assert y1 == _y1

    z2 = qm_o.Z(2)
    _z2 = qm_o.Hamiltonian()
    _z2.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 2),), 1.0)
    assert z2 == _z2


def test_multiply_pauli_same_qubit():
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    I0 = qm_o.PauliOperator(qm_o.Pauli.I, 0)

    ops, phase = qm_o.multiply_pauli_same_qubit(X0, Y0)
    assert ops == Z0
    assert phase == 1.0j

    ops, phase = qm_o.multiply_pauli_same_qubit(X0, Z0)
    assert ops == Y0
    assert phase == -1.0j

    ops, phase = qm_o.multiply_pauli_same_qubit(Y0, Z0)
    assert ops == X0
    assert phase == 1.0j

    ops, phase = qm_o.multiply_pauli_same_qubit(X0, X0)
    assert ops == I0
    assert phase == 1.0

    ops, phase = qm_o.multiply_pauli_same_qubit(Y0, Y0)
    assert ops == I0
    assert phase == 1.0

    ops, phase = qm_o.multiply_pauli_same_qubit(Z0, I0)
    assert ops == Z0
    assert phase == 1.0

    ops, phase = qm_o.multiply_pauli_same_qubit(I0, Z0)
    assert ops == Z0
    assert phase == 1.0


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

def test_Hamiltonian_sub():
    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 1.0)
    h = h1 - h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    assert h == expected_h

    h = h1 - 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.constant = -2.0
    assert h == expected_h

    h = 2.0 - h1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), -1.0)
    expected_h.constant = 2.0
    assert h == expected_h

def test_Hamiltonian_scalar_multiplication():
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h = h * 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 2.0)
    assert h == expected_h

    h = 2.0 * h
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 4.0)
    assert h == expected_h


def test_simplify_pauliop_terms():
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    X1 = qm_o.PauliOperator(qm_o.Pauli.X, 1)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    Z1 = qm_o.PauliOperator(qm_o.Pauli.Z, 1)
    paulis, phase = qm_o.simplify_pauliop_terms((X0, Y1))
    assert set(paulis) == set((X0, Y1))
    assert phase == 1.0

    paulis, phase = qm_o.simplify_pauliop_terms((X1, Y0) + (X0,))
    assert set(paulis) == set((X1, Z0))
    assert phase == -1.0j

    paulis, coeffs = qm_o.simplify_pauliop_terms((Z0,) + (X0,))
    assert set(paulis) == set((Y0,))
    assert coeffs == 1.0j

    paulis, coeffs = qm_o.simplify_pauliop_terms((X0, X0))

    assert set(paulis) == set(())
    assert coeffs == 1.0

    paulis, coeffs = qm_o.simplify_pauliop_terms((Y0, Y0, X0))
    assert set(paulis) == set((X0,))
    assert coeffs == 1.0

    paulis, coeffs = qm_o.simplify_pauliop_terms((Y0, Y0, X0, X1, Z1))
    assert set(paulis) == set((X0, Y1))
    assert coeffs == -1.0j

    paulis, coeffs = qm_o.simplify_pauliop_terms(())
    assert set(paulis) == set(())
    assert coeffs == 1.0


def test_Hamiltonian_multiplication():
    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1), qm_o.PauliOperator(qm_o.Pauli.Y, 2)), 3.0
    )
    h = h1 * h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.X, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
            qm_o.PauliOperator(qm_o.Pauli.Y, 2),
        ),
        3.0,
    )
    assert h == expected_h

    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 2.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1), qm_o.PauliOperator(qm_o.Pauli.Y, 2)), 3.0
    )
    h = h1 * h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.X, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
            qm_o.PauliOperator(qm_o.Pauli.Y, 2),
        ),
        3.0,
    )
    expected_h.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
            qm_o.PauliOperator(qm_o.Pauli.Y, 2),
        ),
        6.0,
    )
    assert h == expected_h

    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 2.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 3.0)
    h2.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 2),), 4.0)
    h = h1 * h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 1)), 3.0
    )
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 2)), 4.0
    )
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 1)), 6.0
    )
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 2)), 8.0
    )
    assert h == expected_h

    x0 = qm_o.X(0)
    y0 = qm_o.Y(0)
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0j)
    op = x0 * y0
    assert h == op

    x1 = qm_o.X(1)
    y1 = qm_o.Y(1)

    h1 = x0 + y1
    h2 = x1 + y0
    h = h1 * h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.X, 1)), 1.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0j)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 1),), -1.0j)
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 1)), 1.0
    )
    assert h == expected_h

    h1 = 2.0 * x0 + y1
    h2 = x0 + y0
    h = h1 * h2

    expected_h = qm_o.Hamiltonian()
    expected_h.constant += 2.0
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 1)), 1.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 2.0j)
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 1)), 1.0
    )
    assert h == expected_h

    h1 = 2.0 * x0 + 1.0
    h = y1 * h1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1), qm_o.PauliOperator(qm_o.Pauli.X, 0)), 2.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 1.0)
    assert h == expected_h

    h = h1 * y1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1), qm_o.PauliOperator(qm_o.Pauli.X, 0)), 2.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 1.0)
    assert h == expected_h

    h = h1 * h1
    expected_h = qm_o.Hamiltonian()
    expected_h.constant += 5.0
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 4.0)

    assert h == expected_h

    h2 = 4 * y1 + 3
    h = h1 * h2
    expected_h = qm_o.Hamiltonian()
    expected_h.constant += 3.0
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Y, 1)), 8.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 6.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 4.0)
    assert h == expected_h


def test_Hamiltonian_neg():
    x0 = qm_o.X(0)
    y1 = qm_o.Y(1)

    h = -x0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), -1.0)
    assert h == expected_h

    h1 = -(2.0 * x0 + y1)
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), -2.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), -1.0)
    assert h1 == expected_h

def test_num_qubits():
    h = qm_o.Hamiltonian(num_qubits=3)
    h += 1.0
    assert h.num_qubits == 3
    h *= qm_o.X(0)
    assert h.num_qubits == 3
    h *= qm_o.X(1)
    assert h.num_qubits == 3
    h *= qm_o.X(3)
    assert h.num_qubits == 4

    h = qm_o.Hamiltonian(num_qubits=3)
    h += qm_o.X(0) 
    assert h.num_qubits == 3
    h += qm_o.X(3) 
    assert h.num_qubits == 4
    
def test_coeff_complex():
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    h *= (1 + 1j * qm_o.Y(0))

    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    assert h == expected_h

    