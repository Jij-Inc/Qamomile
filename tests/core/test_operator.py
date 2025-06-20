import itertools

import numpy as np
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
    "pauli, index",
    [
        # Standard cases with different Pauli and indices
        (qm_o.Pauli.X, 3),
        (qm_o.Pauli.Y, 2),
        (qm_o.Pauli.Z, 1),
        (qm_o.Pauli.I, 0),
    ],
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

    # Get the expected Pauli.
    expected_pauli, expected_phase = Utils.PAULI_PRODUCT_TABLE[(pauli1, pauli2)]

    # Run multiply_pauli_same_qubit for those PauliOperators.
    ops, phase = qm_o.multiply_pauli_same_qubit(pauli_op1, pauli_op2)

    # 1. The returned PauliOperator's index is the same as the input PauliOperators' index,
    assert ops.index == index

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
    max_index = 10
    for first_index in range(1, max_index + 1):
        for second_index in range(first_index + 1, max_index + 1):

            # 1. ValueError arises.
            pauli_op1 = qm_o.PauliOperator(pauli1, first_index)
            pauli_op2 = qm_o.PauliOperator(pauli2, second_index)
            with pytest.raises(ValueError):
                qm_o.multiply_pauli_same_qubit(pauli_op1, pauli_op2)
            pauli_op1 = qm_o.PauliOperator(pauli1, second_index)
            pauli_op2 = qm_o.PauliOperator(pauli2, first_index)
            with pytest.raises(ValueError):
                qm_o.multiply_pauli_same_qubit(pauli_op1, pauli_op2)


# <<< multiply_pauli_same_qubit <<<


# >>> Hamiltonian >>>
@pytest.mark.parametrize(
    "num_qubits",
    [0, 1, 2, 3],
)
def test_hamiltonian_creation_with_num_qubits(num_qubits):
    """Create Hamiltonian with specifying num_qubits.

    Check if
    1. Its num_qubits is set correctly,
    2. Its terms is an empty dictionary,
    3. Its constasnt is set to zero.
    """
    h = qm_o.Hamiltonian(num_qubits=num_qubits)
    # 1. Its num_qubits is set correctly,
    assert h.num_qubits == num_qubits
    # 2. Its terms is an empty dictionary,
    assert h.terms == {}
    # 3. Its constasnt is set to zero.
    assert h.constant == 0.0


def test_hamiltonian_creation_without_num_qubits():
    """Create Hamiltonian without specifying num_qubits.

    Check if
    1. Its num_qubits is zero,
    2. Its terms is an empty dictionary,
    3. Its constasnt is set to zero.
    """
    h = qm_o.Hamiltonian()
    # 1. Its num_qubits is set correctly,
    assert h.num_qubits == 0
    # 2. Its terms is an empty dictionary,
    assert h.terms == {}
    # 3. Its constasnt is set to zero.
    assert h.constant == 0.0


def test_add_term():
    """Test Hamiltonian.add_term for various PauliOperator combinations.

    Check if
    1. The terms are just qm_o.Pauli.X at index 0 with coefficient 1.0,
    2. The constnat is 0.0,
    3. qm_o.Pauli.Y are added at index 1 and 2 with coefficient 3.0,
    4. The constnat is 0.0,
    5. qm_o.Pauli.X is added at index 0 with coefficient 1.0, which is combined with the previous term and becomes 2.0,
    6. The constnat is 0.0,
    7. Two qm_o.Pauli.X are added at index 0 with coefficient -1.0, which converts to an identity operator and the terms are not changed,
    8. The constnat is -1.0,
    9. qm_o.Pauli.X and qm_o.Pauli.Y are added at index 0 with coefficient -4.0, which converts to a Z operator and is added to the terms,
    10. The constnat is -1.0.
    """
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    Y2 = qm_o.PauliOperator(qm_o.Pauli.Y, 2)
    h = qm_o.Hamiltonian()

    h.add_term((X0,), 1.0)
    # 1. The terms are just qm_o.Pauli.X at index 0 with coefficient 1.0,
    assert h.terms == {(X0,): 1.0}
    # 2. The constnat is 0.0,
    assert h.constant == 0.0

    h.add_term((Y1, Y2), 3.0)
    # 3. qm_o.Pauli.Y are added at index 1 and 2 with coefficient 3.0,
    assert h.terms == {(X0,): 1.0, (Y1, Y2): 3.0}
    # 4. The constnat is 0.0,
    assert h.constant == 0.0

    h.add_term((X0,), 1.0)
    # 5. qm_o.Pauli.X is added at index 0 with coefficient 1.0, which is combined with the previous term and becomes 2.0,
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}
    # 6. The constnat is 0.0,
    assert h.constant == 0.0

    h.add_term((X0, X0), -1.0)
    # 7. Two qm_o.Pauli.X are added at index 0 with coefficient -1.0, which converts to an identity operator and the terms are not changed,
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}
    # 8. The constnat is -1.0,
    assert h.constant == -1.0

    h.add_term((X0, Y0), -4.0)
    # 9. qm_o.Pauli.X and qm_o.Pauli.Y are added at index 0 with coefficient -4.0, which converts to a Z operator and is added to the terms,
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0, (Z0,): -4.0j}
    # 10. The constnat is -1.0.
    assert h.constant == -1.0


# @pytest.mark.parametrize("num_qubits", [0, 1, 2, 3])
# def test_num_qubits_with_num_qubits(num_qubits):
#     """Call num_qubits property of Hamiltonian with num_qubits.

#     Check if
#     1. num_qubits is the same as the given num_qubits when creating Hamiltonian with num_qubits without adding terms,
#     2. num_qubits is the same as the given num_qubits when creating Hamiltonian with num_qubits with adding terms,
#     """
#     hamiltonian = qm_o.Hamiltonian(num_qubits=num_qubits)

#     # 1. num_qubits is the same as the given num_qubits when creating Hamiltonian with num_qubits without adding terms,
#     assert hamiltonian.num_qubits == num_qubits

#     max_iterations = 10
#     index = 0
#     for _ in range(max_iterations):
#         hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), index)
#         index += 1
#         hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), index)
#         index += 1
#         hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), index)
#         index += 1
#         # 2. num_qubits is the same as the given num_qubits when creating Hamiltonian with num_qubits with adding terms.
#         assert hamiltonian.num_qubits == num_qubits


# def test_num_qubits_without_num_qubits():
#     """Call num_qubits property of Hamiltonian without num_qubits.

#     Check if
#     2. num_qubits is zero when creating Hamiltonian without num_qubits and without adding terms,
#     3. num_qubits is 1 + the maximum index of PauliOperators in the terms when creating Hamiltonian without num_qubits and adding terms.
#     """


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
def test_simplify_pauliop_terms_on_same_qubit():
    """Run simplify_pauliop_terms for various Paulis on the same qubit.

    Check if
    1. The simplified PauliOperator tuple is correct,
    2. The simplified phase is correct.
    """
    index = 0
    max_num_paulis = 4
    for length in range(2, max_num_paulis + 1):
        # Get all combinations of Pauli products of the given length.
        pauli_products = list(itertools.product(list(qm_o.Pauli), repeat=length))

        # Iterate over each combination of pauli_products.
        for pauli_product in pauli_products:
            # Run simplify_pauliop_terms on the current combination of Pauli products.
            pauli_ops = [qm_o.PauliOperator(pauli, index) for pauli in pauli_product]
            simplified_pauli_ops, simplified_phase = qm_o.simplify_pauliop_terms(
                pauli_ops
            )

            # Compute the expected PauliOperator and phase.
            expected_pauli = pauli_product[0]
            phases = []
            for pauli in pauli_product[1:]:
                expected_pauli, phase = Utils.PAULI_PRODUCT_TABLE[
                    (expected_pauli, pauli)
                ]
                phases.append(phase)

            if expected_pauli == qm_o.Pauli.I:
                expected_pauli_op = ()
            else:
                expected_pauli_op = (qm_o.PauliOperator(expected_pauli, index),)
            expected_phase = np.prod(phases)

            # 1. The simplified PauliOperator tuple is correct,
            assert simplified_pauli_ops == expected_pauli_op
            # 2. The simplified phase is correct.
            assert simplified_phase == expected_phase


@pytest.mark.parametrize(
    "pauli_ops, expected_pauli_ops_set, expected_phase",
    [
        # Simple case: One Pauli operator on each different qubit
        (
            (
                qm_o.PauliOperator(qm_o.Pauli.X, 0),
                qm_o.PauliOperator(qm_o.Pauli.Y, 1),
            ),
            set(
                (
                    qm_o.PauliOperator(qm_o.Pauli.X, 0),
                    qm_o.PauliOperator(qm_o.Pauli.Y, 1),
                )
            ),
            1.0,
        ),
        # Complex case: Multiple Pauli operators on different qubits
        (
            (
                qm_o.PauliOperator(qm_o.Pauli.X, 1),
                qm_o.PauliOperator(qm_o.Pauli.Y, 0),
                qm_o.PauliOperator(qm_o.Pauli.X, 0),
            ),
            set(
                (
                    qm_o.PauliOperator(qm_o.Pauli.X, 1),
                    qm_o.PauliOperator(qm_o.Pauli.Z, 0),
                )
            ),
            -1.0j,
        ),
        # Complex case: Multiple Pauli operators on different qubits with more than two terms
        (
            (
                qm_o.PauliOperator(qm_o.Pauli.Y, 0),
                qm_o.PauliOperator(qm_o.Pauli.Y, 0),
                qm_o.PauliOperator(qm_o.Pauli.X, 0),
                qm_o.PauliOperator(qm_o.Pauli.X, 1),
                qm_o.PauliOperator(qm_o.Pauli.Z, 1),
            ),
            set(
                (
                    qm_o.PauliOperator(qm_o.Pauli.X, 0),
                    qm_o.PauliOperator(qm_o.Pauli.Y, 1),
                )
            ),
            -1.0j,
        ),
        ((), set(()), 1),
    ],
)
def test_simplify_pauliop_terms_on_different_qubits(
    pauli_ops, expected_pauli_ops_set, expected_phase
):
    """Run simplify_pauliop_terms for various PauliOperators on different qubits.

    Check if
    1. The simplified PauliOperator tuple is correct,
    2. The simplified phase is correct.
    """
    simplified_pauli_ops, simplified_phase = qm_o.simplify_pauliop_terms(pauli_ops)

    # 1. The simplified PauliOperator tuple is correct,
    assert set(simplified_pauli_ops) == expected_pauli_ops_set
    # 2. The simplified phase is correct.
    assert simplified_phase == expected_phase


# <<< simplify_pauliop_terms <<<
