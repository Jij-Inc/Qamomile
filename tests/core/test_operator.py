import pytest

import qamomile.core.operator as qm_o
from tests.utils import Utils


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
@pytest.mark.parametrize(
    "pauli",
    [(qm_o.Pauli.X), (qm_o.Pauli.Y), (qm_o.Pauli.Z), (qm_o.Pauli.I)],
)
@pytest.mark.parametrize(
    "index",
    [0, 1, 2],
)
def test_pauli_operator_creation(pauli, index):
    """Create PauliOperators.

    Check if
    1. The pauli attribute is correctly set.
    2. The index is set correctly.
    3. The return value of repr function of PauliOperator is correct.
    4. The return value of str function of PauliOperator is correct.
    5. The return value of hash function of PauliOperator is a tuple of its pauli and its index in this order.
    """
    pauli_operator = qm_o.PauliOperator(pauli, index)
    # 1. The pauli attribute is correctly set.
    assert pauli_operator.pauli == pauli
    # 2. The index is set correctly.
    assert pauli_operator.index == index
    # 3. The return value of repr function of PauliOperator is correct.
    assert repr(pauli_operator) == f"{pauli.name}{index}"
    # 4. The return value of str function of PauliOperator is correct.
    assert str(pauli_operator) == f"{pauli.name}{index}"
    # 5. The return value of hash function of PauliOperator is a tuple of its pauli and its index in this order.
    assert hash(pauli_operator) == hash((pauli, index))


# <<< PauliOperator <<<


# >>> multiply_pauli_same_qubit >>>
@pytest.mark.parametrize(
    "pauli1",
    [qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z, qm_o.Pauli.I],
)
@pytest.mark.parametrize(
    "pauli2",
    [qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z, qm_o.Pauli.I],
)
def test_multiply_pauli_same_qubit_on_same_qubit(pauli1, pauli2):
    """Test multiply_pauli_same_qubit for all Pauli combinations on the same qubit.

    Check if
    1. The returned PauliOperator's index is the same as the input PauliOperators' index,
    2. The returned PauliOperator's pauli is correct according to the multiplication rules,
    3. The returned phase is correct according to the multiplication rules.
    """
    # Create PauliOperators for the same qubit.
    index = 0
    pauli_op1 = qm_o.PauliOperator(pauli1, index)
    pauli_op2 = qm_o.PauliOperator(pauli2, index)
    # Run multiply_pauli_same_qubit for those PauliOperators.
    ops, phase = qm_o.multiply_pauli_same_qubit(pauli_op1, pauli_op2)

    # 1. The returned PauliOperator's index is the same as the input PauliOperators' index,
    assert ops.index == index

    # Get the expected Pauli.
    expected_pauli, expected_phase = Utils.PAULI_PRODUCT_TABLE[(pauli1, pauli2)]

    # 2. The returned PauliOperator's pauli is correct according to the multiplication rules,
    assert ops.pauli == expected_pauli
    # 3. The returned phase is correct according to the multiplication rules.
    assert phase == expected_phase


@pytest.mark.parametrize(
    "pauli1",
    [qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z, qm_o.Pauli.I],
)
@pytest.mark.parametrize(
    "pauli2",
    [qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z, qm_o.Pauli.I],
)
def test_multiply_pauli_same_qubit_on_different_qubits(pauli1, pauli2):
    """Test multiply_pauli_same_qubit for PauliOperators on different qubits.

    Check if
    1. ValueError arises.
    """
    max_index = 5
    for first_index in range(1, max_index + 1):
        for second_index in range(first_index + 1, max_index + 1):

            pauli_op1 = qm_o.PauliOperator(pauli1, first_index)
            pauli_op2 = qm_o.PauliOperator(pauli2, second_index)
            # 1. ValueError arises.
            with pytest.raises(ValueError):
                qm_o.multiply_pauli_same_qubit(pauli_op1, pauli_op2)
            # 1. ValueError arises.
            with pytest.raises(ValueError):
                qm_o.multiply_pauli_same_qubit(pauli_op2, pauli_op1)


# <<< multiply_pauli_same_qubit <<<


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
    h *= 1 + 1j * qm_o.Y(0)

    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    assert h == expected_h
