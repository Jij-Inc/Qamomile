import qamomile.core.operator as qm_o


# >>> Pauli >>>
def test_pauli():
    """Test Pauli class.

    Check if
    1. The Pauli class has the correct attributes for I, X, Y, Z,
    2. The Pauli class has only four attributes.
    """
    # 1. The Pauli class has the correct attributes for I, X, Y, Z,
    #    It is alright to just access the attributes.
    #    If error arises, then the attribute is not defined, which is a failure of the test.
    qm_o.Pauli.I
    qm_o.Pauli.X
    qm_o.Pauli.Y
    qm_o.Pauli.Z
    # 2. The Pauli class has only four attributes.
    assert len(qm_o.Pauli) == 4


# <<< Pauli <<<


# >>> PauliOperator >>>
def test_pauli_operator_creation():
    """Create PauliOperators.

    Check if
    1. The pauli attribute is correctly set.
    2. The index is set correctly.
    """
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    # 1. The pauli attribute is correctly set.
    assert X0.pauli == qm_o.Pauli.X
    # 2. The index is set correctly.
    assert X0.index == 0

    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    # 1. The pauli attribute is correctly set.
    assert Y1.pauli == qm_o.Pauli.Y
    # 2. The index is set correctly.
    assert Y1.index == 1

    Z2 = qm_o.PauliOperator(qm_o.Pauli.Z, 2)
    # 1. The pauli attribute is correctly set.
    assert Z2.pauli == qm_o.Pauli.Z
    # 2. The index is set correctly.
    assert Z2.index == 2


def test_add_term():
    """Test Hamiltonian.add_term for various PauliOperator combinations.

    Check if
    1. Terms are added and accumulated correctly,
    2. Adding the same term accumulates the coefficient,
    3. Adding a term with two identical PauliOperators on the same qubit updates the constant,
    4. Adding a term with X and Y on the same qubit produces a Z term with correct phase.
    """
    # 1. Terms are added and accumulated correctly
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
    # 2. Adding the same term accumulates the coefficient
    h.add_term((X0,), 1.0)
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}
    # 3. Adding a term with two identical PauliOperators on the same qubit updates the constant
    h.add_term((X0, X0), -1.0)
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}
    assert h.constant == -1.0
    # 4. Adding a term with X and Y on the same qubit produces a Z term with correct phase
    h.add_term((X0, Y0), -4.0)
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0, (Z0,): -4.0j}
    assert h.constant == -1.0


# >>> multiply_pauli_same_qubit >>>
def test_multiply_pauli_same_qubit():
    """Test multiply_pauli_same_qubit for all Pauli combinations on the same qubit.

    Check if
    1. The correct PauliOperator and phase are returned for each combination.
    """
    # 1. The correct PauliOperator and phase are returned for each combination
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


# <<< multiply_pauli_same_qubit <<<

# <<< PauliOperator <<<


# >>> Hamiltonian >>>
def test_pauli_hamiltonian_creation():
    """Test creation of Hamiltonians using X, Y, Z helpers and manual construction.

    Check if
    1. The helper functions X, Y, Z create correct Hamiltonians,
    2. Manually constructed Hamiltonians are equal to those from helpers.
    """
    # 1. The helper functions X, Y, Z create correct Hamiltonians
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


def test_Hamiltonian_add():
    """Test Hamiltonian addition with other Hamiltonians.

    Check if
    1. Adding two Hamiltonians accumulates all terms correctly.
    """
    # 1. Adding two Hamiltonians accumulates all terms correctly
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
    """Test Hamiltonian addition with scalars.

    Check if
    1. Adding a scalar updates the constant term,
    2. Adding a scalar from the left also updates the constant term.
    """
    # 1. Adding a scalar updates the constant term
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h = h + 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.constant = 2.0
    assert h == expected_h
    # 2. Adding a scalar from the left also updates the constant term
    h = 2.0 + h
    expected_h.constant += 2.0
    assert h == expected_h


def test_Hamiltonian_sub():
    """Test Hamiltonian subtraction with other Hamiltonians and scalars.

    Check if
    1. Subtracting two Hamiltonians accumulates all terms correctly,
    2. Subtracting a scalar updates the constant term,
    3. Subtracting a Hamiltonian from a scalar negates the terms and adds the scalar to the constant.
    """
    # 1. Subtracting two Hamiltonians accumulates all terms correctly
    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 1.0)
    h = h1 - h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    assert h == expected_h
    # 2. Subtracting a scalar updates the constant term
    h = h1 - 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.constant = -2.0
    assert h == expected_h
    # 3. Subtracting a Hamiltonian from a scalar negates the terms and adds the scalar to the constant
    h = 2.0 - h1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), -1.0)
    expected_h.constant = 2.0
    assert h == expected_h


def test_Hamiltonian_scalar_multiplication():
    """Test Hamiltonian scalar multiplication (left and right).

    Check if
    1. Multiplying a Hamiltonian by a scalar updates all coefficients,
    2. Multiplying by a scalar from the left also updates all coefficients.
    """
    # 1. Multiplying a Hamiltonian by a scalar updates all coefficients
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h = h * 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 2.0)
    assert h == expected_h
    # 2. Multiplying by a scalar from the left also updates all coefficients
    h = 2.0 * h
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 4.0)
    assert h == expected_h


def test_Hamiltonian_multiplication():
    """Test Hamiltonian multiplication with other Hamiltonians and scalars.

    Check if
    1. Multiplying two Hamiltonians produces the correct terms and coefficients,
    2. Multiplying by a scalar updates all coefficients,
    3. Multiplying by a Hamiltonian with multiple terms accumulates all products.
    """
    # 1. Multiplying two Hamiltonians produces the correct terms and coefficients
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
    # 2. Multiplying by a Hamiltonian with multiple terms accumulates all products
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
    # 3. Multiplying by a Hamiltonian with multiple terms accumulates all products
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
    """Test negation of Hamiltonian.

    Check if
    1. Negating a Hamiltonian negates all coefficients and the constant.
    """
    # 1. Negating a Hamiltonian negates all coefficients and the constant
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
    """Test num_qubits property of Hamiltonian.

    Check if
    1. num_qubits is updated correctly when adding terms or multiplying by operators on new qubits.
    """
    # 1. num_qubits is updated correctly when adding terms or multiplying by operators on new qubits
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
    """Test Hamiltonian with complex coefficients.

    Check if
    1. Complex coefficients are handled correctly in multiplication and addition.
    """
    # 1. Complex coefficients are handled correctly in multiplication and addition
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    h *= 1 + 1j * qm_o.Y(0)
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    assert h == expected_h


# <<< Hamiltonian <<<


# >>> simplify_pauliop_terms >>>
def test_simplify_pauliop_terms():
    """Test simplify_pauliop_terms for various PauliOperator tuples.

    Check if
    1. The correct simplified PauliOperator tuple and phase are returned for each input.
    """
    # 1. The correct simplified PauliOperator tuple and phase are returned for each input
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


# <<< simplify_pauliop_terms <<<
