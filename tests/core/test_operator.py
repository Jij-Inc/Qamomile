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
@pytest.mark.parametrize(
    "num_qubits",
    [0, 1, 2],
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


@pytest.mark.parametrize("num_qubits", [0, 1, 2, 3])
def test_num_qubits_with_num_qubits(num_qubits):
    """Call num_qubits property of Hamiltonian with num_qubits.

    Check if
    1. num_qubits is the same as the given num_qubits without adding terms,
    2-1. num_qubits is the same as the given num_qubits if the largest specified qubit index + 1 is smaller than the given num_qubits.
    2-2. num_qubits is the largest specified qubit index + 1 if the largest specified qubit index + 1 is greater than or equal to the given num_qubits.
    """
    hamiltonian = qm_o.Hamiltonian(num_qubits=num_qubits)

    # 1. num_qubits is the same as the given num_qubits without adding terms,
    assert hamiltonian.num_qubits == num_qubits

    max_iterations = 10
    index = 0
    for _ in range(max_iterations):
        hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.X, index),), 1.0)

        # 2-1. num_qubits is the same as the given num_qubits if the largest specified qubit index + 1 is smaller than the given num_qubits.
        if index + 1 < num_qubits:
            assert hamiltonian.num_qubits == num_qubits
        # 2-2. num_qubits is the largest specified qubit index + 1 if the largest specified qubit index + 1 is greater than or equal to the given num_qubits.
        else:
            assert hamiltonian.num_qubits == index + 1

        index += 1


def test_num_qubits_without_num_qubits():
    """Call num_qubits property of Hamiltonian without num_qubits.

    Check if
    1. num_qubits is zero without adding terms,
    2. num_qubits is the largest specified qubit index + 1.
    """
    hamiltonian = qm_o.Hamiltonian()

    # 1. num_qubits is zero without adding terms,
    assert hamiltonian.num_qubits == 0

    max_iterations = 10
    index = 0
    for _ in range(max_iterations):
        hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.X, index),), 1.0)

        # 2. num_qubits is the largest specified qubit index + 1.
        assert hamiltonian.num_qubits == index + 1

        index += 1


def test_num_qubits_manually():
    """Call num_qubits property of manually evolved Hamiltonian.

    Check if
    1. The number of qubits is correct after evolving manually.
    """
    h = qm_o.Hamiltonian(num_qubits=3)
    h += 1.0
    # 1. The number of qubits is correct after evolving manually.
    assert h.num_qubits == 3
    h *= qm_o.X(0)
    # 1. The number of qubits is correct after evolving manually.
    assert h.num_qubits == 3
    h *= qm_o.X(1)
    # 1. The number of qubits is correct after evolving manually.
    assert h.num_qubits == 3
    h *= qm_o.X(3)
    # 1. The number of qubits is correct after evolving manually.
    assert h.num_qubits == 4

    h = qm_o.Hamiltonian(num_qubits=3)
    h += qm_o.X(0)
    # 1. The number of qubits is correct after evolving manually.
    assert h.num_qubits == 3
    h += qm_o.X(3)
    # 1. The number of qubits is correct after evolving manually.
    assert h.num_qubits == 4


def test_add_term_manually():
    """Add manually decided terms to a Hamiltonian.

    Check if
    1. The terms are added correctly,
    2. The constant is set correctly.
    """
    X0 = qm_o.PauliOperator(qm_o.Pauli.X, 0)
    Y0 = qm_o.PauliOperator(qm_o.Pauli.Y, 0)
    Z0 = qm_o.PauliOperator(qm_o.Pauli.Z, 0)
    Y1 = qm_o.PauliOperator(qm_o.Pauli.Y, 1)
    Y2 = qm_o.PauliOperator(qm_o.Pauli.Y, 2)

    h = qm_o.Hamiltonian()
    h.add_term((X0,), 1.0)
    # 1. The terms are added correctly,
    assert h.terms == {(X0,): 1.0}

    h.add_term((Y1, Y2), 3.0)
    # 1. The terms are added correctly,
    assert h.terms == {(X0,): 1.0, (Y1, Y2): 3.0}

    h.add_term((X0,), 1.0)
    # 1. The terms are added correctly,
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}

    h.add_term((X0, X0), -1.0)
    # 1. The terms are added correctly,
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0}
    # 2. The constant is set correctly.
    assert h.constant == -1.0

    h.add_term((X0, Y0), -4.0)
    # 1. The terms are added correctly,
    assert h.terms == {(X0,): 2.0, (Y1, Y2): 3.0, (Z0,): -4.0j}
    # 2. The constant is set correctly.
    assert h.constant == -1.0


@pytest.mark.parametrize(
    "pauli_combinations",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize("positive_coefficient", [1, 2, 1.1])
def test_to_latex(pauli_combinations, positive_coefficient):
    """ """
    index = 0
    for pauli_combination in pauli_combinations:
        h = qm_o.Hamiltonian()

        expected_strs = []
        for pauli in pauli_combination:
            # Add terms to the Hamiltonian.
            h.add_term((qm_o.PauliOperator(pauli, index),), positive_coefficient)

            # Create the expected string representation.
            if pauli != qm_o.Pauli.I:
                pauli_str = Utils.get_pauli_string(pauli)
                _expected_str = f"{pauli_str}_" + "{" + f"{index}" + "}"
                if positive_coefficient != 1:
                    _expected_str = f"{positive_coefficient:.1f}" + _expected_str
                expected_strs.append(_expected_str)
        expected_str = "+".join(expected_strs)

        assert h.to_latex() == expected_str


@pytest.mark.parametrize(
    "pauli_combinations",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize("negative_coefficient", [-1, -2, -1.1])
def test_to_latex(pauli_combinations, negative_coefficient):
    """ """
    index = 0
    for pauli_combination in pauli_combinations:
        h = qm_o.Hamiltonian()

        expected_strs = []
        for pauli in pauli_combination:
            # Add terms to the Hamiltonian.
            h.add_term((qm_o.PauliOperator(pauli, index),), negative_coefficient)

            # Create the expected string representation.
            if pauli != qm_o.Pauli.I:
                pauli_str = Utils.get_pauli_string(pauli)
                _expected_str = f"{pauli_str}_" + "{" + f"{index}" + "}"
                if negative_coefficient != -1:
                    _expected_str = f"{negative_coefficient:.1f}" + _expected_str
                else:
                    _expected_str = "-" + _expected_str
                expected_strs.append(_expected_str)
        expected_str = "".join(expected_strs)

        assert h.to_latex() == expected_str


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_add_wrt_same_qubit(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian addition with other Hamiltonians whose all the terms are with respect to the same qubits.

    Check if
    1. Adding two Hamiltonians accumulates all terms correctly.
    """
    index = 0
    # Iterate over pauli_combinations1.
    for pauli_combination1 in pauli_combinations1:
        # Create the first Hamiltonian.
        h1 = qm_o.Hamiltonian()
        for pauli1 in pauli_combination1:
            h1.add_term((qm_o.PauliOperator(pauli1, index),), 1.0)

        # Iterate over pauli_combinations2.
        for pauli_combination2 in pauli_combinations2:
            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to to the same qubit as the first Hamiltonian: index.
                h2.add_term((qm_o.PauliOperator(pauli2, index),), 1.0)

            # Add the two Hamiltonians.
            h = h1 + h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            pauli_combination = list(pauli_combination1) + list(pauli_combination2)
            for pauli in pauli_combination:
                expected_h.add_term((qm_o.PauliOperator(pauli, index),), 1.0)

            # 1. Adding two Hamiltonians accumulates all terms correctly.
            assert h == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_add_wrt_different_qubits(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian addition with other Hamiltonians whose all the terms are with respect to different qubits.

    Check if
    1. Adding two Hamiltonians accumulates all terms correctly.
    """
    index1 = 0
    index2 = 1
    # Iterate over palui_combinations1.
    for pauli_combination1 in pauli_combinations1:
        # Create the first Hamiltonian.
        h1 = qm_o.Hamiltonian()
        for pauli1 in pauli_combination1:
            h1.add_term((qm_o.PauliOperator(pauli1, index1),), 1.0)

        # Iterate over palui_combinations2.
        for pauli_combination2 in pauli_combinations2:
            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to a different qubit than the first Hamiltonian: index2.
                h2.add_term((qm_o.PauliOperator(pauli2, index2),), 1.0)

            # Add the two Hamiltonians.
            h = h1 + h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            for pauli in pauli_combination1:
                expected_h.add_term((qm_o.PauliOperator(pauli, index1),), 1.0)
            for pauli in pauli_combination2:
                expected_h.add_term((qm_o.PauliOperator(pauli, index2),), 1.0)

            # 1. Adding two Hamiltonians accumulates all terms correctly.
            assert h == expected_h


@pytest.mark.parametrize(
    "constant",
    [
        int(1),
        float(1.0),
        float(1.1),
        complex(1.0, 1.0),
        complex(1.0, 0.0),
        complex(0.0, 1.0),
    ],
)
def test_Hamiltonian_add_wrt_valid_constants(constant):
    """Add Hamiltonian with valid constants.

    Check if
    1. Adding a constant to one whose constant is zero updates the constant term,
    2. Adding a constant to one whose constant is not zero updates the constant term.
    """
    # Add constant to Hamiltonian with constant zero.
    h = qm_o.Hamiltonian()
    h = h + constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = constant
    # 1. Adding a constant to one whose constant is zero updates the constant term,
    assert h == expected_h

    # Add constant to Hamiltonian with constant not zero.
    h = qm_o.Hamiltonian()
    initial_constant = 1.0
    h.constant = initial_constant
    h = h + constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = initial_constant + constant
    # 2. Adding a constant to one whose constant is not zero updates the constant term.
    assert h == expected_h


@pytest.mark.parametrize("invalid_constant", [str(1), list([1, 2, 3]), dict({1: 2})])
def test_Hamiltonian_add_wrt_invalid_constants(invalid_constant):
    """Add Hamiltonian with invalid constants.

    Check if
    1. ValueError arises.
    """
    h = qm_o.Hamiltonian()
    # 1. ValueError arises.
    with pytest.raises(ValueError):
        h + invalid_constant


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_radd_wrt_same_qubit(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian right addition with other Hamiltonians whose all the terms are with respect to the same qubits.

    Check if
    1. Adding two Hamiltonians accumulates all terms correctly.
    """
    index = 0
    # Iterate over pauli_combinations1:
    for pauli_combination1 in pauli_combinations1:

        # Iterate over pauli_combinations2:
        for pauli_combination2 in pauli_combinations2:
            # Create the first Hamiltonian.
            #    Note: We now test the right addition operator, so we need to create the first Hamiltonian every time.
            h1 = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                h1.add_term((qm_o.PauliOperator(pauli1, index),), 1.0)

            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to the same qubit as the first Hamiltonian: index.
                h2.add_term((qm_o.PauliOperator(pauli2, index),), 1.0)

            # Add the two Hamiltonians.
            h1 += h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            all_perms = list(pauli_combination1) + list(pauli_combination2)
            for pauli in all_perms:
                expected_h.add_term((qm_o.PauliOperator(pauli, index),), 1.0)

            # 1. Adding two Hamiltonians accumulates all terms correctly.
            assert h1 == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_radd_wrt_different_qubits(
    pauli_combinations1, pauli_combinations2
):
    """Test Hamiltonian right addition with other Hamiltonians whose all the terms are with respect to different qubits.

    Check if
    1. Adding two Hamiltonians accumulates all terms correctly.
    """
    index1 = 0
    index2 = 1
    # Iterate over pauli_combinations1:
    for pauli_combination1 in pauli_combinations1:

        # Iterate over pauli_combinations2:
        for pauli_combination2 in pauli_combinations2:
            # Create the first Hamiltonian.
            #    Note: We now test the right addition operator, so we need to create the first Hamiltonian every time.
            h1 = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                h1.add_term((qm_o.PauliOperator(pauli1, index1),), 1.0)

            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to a different qubit than the first Hamiltonian: index2.
                h2.add_term((qm_o.PauliOperator(pauli2, index2),), 1.0)

            # Add the two Hamiltonians.
            h1 += h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            for pauli in pauli_combination1:
                expected_h.add_term((qm_o.PauliOperator(pauli, index1),), 1.0)
            for pauli in pauli_combination2:
                expected_h.add_term((qm_o.PauliOperator(pauli, index2),), 1.0)

            # 1. Adding two Hamiltonians accumulates all terms correctly.
            assert h1 == expected_h


@pytest.mark.parametrize(
    "constant", [int(1), float(1.0), float(1.1), complex(1.0, 1.0), complex(1.0, 0.0)]
)
def test_Hamiltonian_radd_wrt_valid_constants(constant):
    """Right add Hamiltonian with valid constants.

    Check if
    1. Adding a constant to one whose constant is zero updates the constant term,
    2. Adding a constant to one whose constant is not zero updates the constant term.
    """
    # Add constant to Hamiltonian with constant zero.
    h = qm_o.Hamiltonian()
    h += constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = constant
    # 1. Adding a constant to one whose constant is zero updates the constant term,
    assert h == expected_h

    # Add constant to Hamiltonian with constant not zero.
    h = qm_o.Hamiltonian()
    initial_constant = 1.0
    h.constant = initial_constant
    h += constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = initial_constant + constant
    # 2. Adding a constant to one whose constant is not zero updates the constant term.
    assert h == expected_h


@pytest.mark.parametrize("invalid_constant", [str(1), list([1, 2, 3]), dict({1: 2})])
def test_Hamiltonian_radd_wrt_invalid_constants(invalid_constant):
    """Right add Hamiltonian with invalid constants.

    Check if
    1. ValueError arises.
    """
    h = qm_o.Hamiltonian()
    # 1. Adding a string raises TypeError,
    with pytest.raises(ValueError):
        h += invalid_constant


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_sub_wrt_same_qubit(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian subtraction with other Hamiltonians whose all the terms are with respect to the same qubits.

    Check if
    1. Subtracting two Hamiltonians accumulates all terms correctly.
    """
    index = 0
    # Iterate over pauli_combinations1:
    for pauli_combination1 in pauli_combinations1:
        # Crate the first Hamiltonian.
        h1 = qm_o.Hamiltonian()
        for pauli1 in pauli_combination1:
            h1.add_term((qm_o.PauliOperator(pauli1, index),), 1.0)

        # Iterate over pauli_combination2.
        for pauli_combination2 in pauli_combinations1:
            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to the same qubit as the first Hamiltonian: index.
                h2.add_term((qm_o.PauliOperator(pauli2, index),), 1.0)

            # Sub the two Hamiltonians.
            h = h1 - h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            for pauli in pauli_combination1:
                expected_h.add_term((qm_o.PauliOperator(pauli, index),), 1.0)
            for pauli in pauli_combination2:
                expected_h.add_term((qm_o.PauliOperator(pauli, index),), -1.0)

            # 1. Subtracting two Hamiltonians accumulates all terms correctly.
            assert h == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_sub_wrt_different_qubits(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian subtraction with other Hamiltonians whose all the terms are with respect to different qubits.

    Check if
    1. Subtracting two Hamiltonians accumulates all terms correctly.
    """
    index1 = 0
    index2 = 1
    # Iterate over pauli_combinations1:
    for pauli_combination1 in pauli_combinations1:
        # Create the first Hamiltonian.
        h1 = qm_o.Hamiltonian()
        for pauli1 in pauli_combination1:
            h1.add_term((qm_o.PauliOperator(pauli1, index1),), 1.0)

        # Iterate over pauli_combinations2.
        for pauli_combination2 in pauli_combinations2:
            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to a different qubit than the first Hamiltonian: index2.
                h2.add_term((qm_o.PauliOperator(pauli2, index2),), 1.0)

            # Sub the two Hamiltonians.
            h = h1 - h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            for pauli in pauli_combination1:
                expected_h.add_term((qm_o.PauliOperator(pauli, index1),), 1.0)
            for pauli in pauli_combination2:
                expected_h.add_term((qm_o.PauliOperator(pauli, index2),), -1.0)

            # 1. Subtracting two Hamiltonians accumulates all terms correctly.
            assert h == expected_h


@pytest.mark.parametrize(
    "constant", [int(1), float(1.0), float(1.1), complex(1.0, 1.0), complex(1.0, 0.0)]
)
def test_Hamiltonian_sub_wrt_constants(constant):
    """Sub Hamiltonian with constants.

    Check if
    1. Subtracting a constant to one whose constant is zero updates the constant term,
    2. Subtracting a constant to one whose constant is not zero updates the constant term.
    """
    # Sub constant to Hamiltonian with constant zero.
    h = qm_o.Hamiltonian()
    h = h - constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = -constant
    # 1. Subtracting a constant to one whose constant is zero updates the constant term,
    assert h == expected_h

    # Sub constant to Hamiltonian with constant not zero.
    h = qm_o.Hamiltonian()
    initial_constant = 1.0
    h.constant = initial_constant
    h = h - constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = initial_constant - constant
    # 2. Subtracting a constant to one whose constant is not zero updates the constant term.
    assert h == expected_h


def test_Hamiltonian_sub_manulally():
    """Substract manually decided terms from a Hamiltonian.

    Check if
    1. The terms are subtracted correctly,
    """
    h1 = qm_o.Hamiltonian()
    h1.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h2 = qm_o.Hamiltonian()
    h2.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 1.0)
    h = h1 - h2
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    # 1. The terms are subtracted correctly,
    assert h == expected_h

    h = h1 - 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    expected_h.constant = -2.0
    # 1. The terms are subtracted correctly,
    assert h == expected_h

    h = 2.0 - h1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), -1.0)
    expected_h.constant = 2.0
    # 1. The terms are subtracted correctly,
    assert h == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_rsub_wrt_same_qubit(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian right subtraction with other Hamiltonians whose all the terms are with respect to the same qubits.

    Check if
    1. Subtracting two Hamiltonians accumulates all terms correctly.
    """
    index = 0
    # Iterate over pauli_combinations1:
    for pauli_combination1 in pauli_combinations1:

        # Iterate over pauli_combinations2:
        for pauli_combination2 in pauli_combinations2:
            # Create the first Hamiltonian.
            #    Note: We now test the right subtraction operator, so we need to create the first Hamiltonian every time.
            h1 = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                h1.add_term((qm_o.PauliOperator(pauli1, index),), 1.0)

            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to the same qubit as the first Hamiltonian: index.
                h2.add_term((qm_o.PauliOperator(pauli2, index),), 1.0)

            # Sub the two Hamiltonians.
            h1 -= h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            for pauli in pauli_combination1:
                expected_h.add_term((qm_o.PauliOperator(pauli, index),), 1.0)
            for pauli in pauli_combination2:
                expected_h.add_term((qm_o.PauliOperator(pauli, index),), -1.0)

            # 1. Subtracting two Hamiltonians accumulates all terms correctly.
            assert h1 == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_rsub_wrt_different_qubits(
    pauli_combinations1, pauli_combinations2
):
    """Test Hamiltonian right subtraction with other Hamiltonians whose all the terms are with respect to different qubits.

    Check if
    1. Subtracting two Hamiltonians accumulates all terms correctly.
    """
    index1 = 0
    index2 = 1
    # Iterate over pauli_combinations1:
    for pauli_combination1 in pauli_combinations1:

        # Iterate over pauli_combinations2:
        for pauli_combination2 in pauli_combinations2:
            # Create the first Hamiltonian.
            #    Note: We now test the right subtraction operator, so we need to create the first Hamiltonian every time.
            h1 = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                h1.add_term((qm_o.PauliOperator(pauli1, index1),), 1.0)

            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to a different qubit than the first Hamiltonian: index2.
                h2.add_term((qm_o.PauliOperator(pauli2, index2),), 1.0)

            # Sub the two Hamiltonians.
            h1 -= h2

            # Calculate the expected Hamiltonian.
            expected_h = qm_o.Hamiltonian()
            for pauli in pauli_combination1:
                expected_h.add_term((qm_o.PauliOperator(pauli, index1),), 1.0)
            for pauli in pauli_combination2:
                expected_h.add_term((qm_o.PauliOperator(pauli, index2),), -1.0)

            # 1. Subtracting two Hamiltonians accumulates all terms correctly.
            assert h1 == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_mul_wrt_same_qubit(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian multiplication with other Hamiltonians whose all the terms are with respect to the same qubits.

    Check if
    1. Multiplicating two Hamiltonians accumulates all terms correctly.
    """
    index = 0
    # Iterate over pauli_combinations1.
    for pauli_combination1 in pauli_combinations1:
        # Create the first Hamiltonian.
        h1 = qm_o.Hamiltonian()
        for pauli1 in pauli_combination1:
            h1.add_term((qm_o.PauliOperator(pauli1, index),), 1.0)

        # Iterate over pauli_combinations2.
        for pauli_combination2 in pauli_combinations2:
            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to to the same qubit as the first Hamiltonian: index.
                h2.add_term((qm_o.PauliOperator(pauli2, index),), 1.0)

            # Add the two Hamiltonians.
            h = h1 * h2

            # Calculate the expected Hamiltonian.
            #    Note: If h1 = X + Y + Z + I and h2 = X + Y + Z + I, then
            #          h = (X + Y + Z + I) * (X + Y + Z + I)
            #          = XX + XY + XZ + XI + YX + YY + YZ + YI + ZX + ZY + ZZ + ZI + IX + IY + IZ + II.
            #          Thus, the following loop accumulates all terms correctly.
            expected_h = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                for pauli2 in pauli_combination2:
                    expected_h.add_term(
                        (
                            qm_o.PauliOperator(pauli1, index),
                            qm_o.PauliOperator(pauli2, index),
                        ),
                        1.0,
                    )

            # 1. Multiplicating two Hamiltonians accumulates all terms correctly.
            assert h == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_mul_wrt_different_qubits(pauli_combinations1, pauli_combinations2):
    """Test Hamiltonian multiplication with other Hamiltonians whose all the terms are with respect to different qubits.

    Check if
    1. multiplicating two Hamiltonians accumulates all terms correctly.
    """
    index1 = 0
    index2 = 1
    # Iterate over palui_combinations1.
    for pauli_combination1 in pauli_combinations1:
        # Create the first Hamiltonian.
        h1 = qm_o.Hamiltonian()
        for pauli1 in pauli_combination1:
            h1.add_term((qm_o.PauliOperator(pauli1, index1),), 1.0)

        # Iterate over palui_combinations2.
        for pauli_combination2 in pauli_combinations2:
            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to a different qubit than the first Hamiltonian: index2.
                h2.add_term((qm_o.PauliOperator(pauli2, index2),), 1.0)

            # Mul the two Hamiltonians.
            h = h1 * h2

            # Calculate the expected Hamiltonian.
            #    Note: If h1 = X1 + Y1 + Z1 + I1 and h2 = X2 + Y2 + Z2 + I2, then
            #          h = (X1 + Y1 + Z1 + I1) * (X2 + Y2 + Z2 + I2)
            #          = X1X2 + X1Y2 + X1Z2 + X1I2 + Y1X2 + Y1Y2 + Y1Z2 + Y1I2 + Z1X2 + Z1Y2 + Z1Z2 + Z1I2 + I1X2 + I1Y2 + I1Z2 + I1I2.
            #          Thus, the following loop accumulates all terms correctly.
            expected_h = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                for pauli2 in pauli_combination2:
                    expected_h.add_term(
                        (
                            qm_o.PauliOperator(pauli1, index1),
                            qm_o.PauliOperator(pauli2, index2),
                        ),
                        1.0,
                    )

            # 1. Multiplicating two Hamiltonians accumulates all terms correctly.
            assert h == expected_h


@pytest.mark.parametrize(
    "constant",
    [
        int(2),
        float(2.0),
        float(1.1),
        complex(2.0, 2.0),
        complex(2.0, 0.0),
        complex(0.0, 2.0),
    ],
)
def test_Hamiltonian_mul_wrt_valid_constants(constant):
    """Mul Hamiltonian with valid constants.

    Check if
    1. Multiplicating a constant to one whose constant is zero updates the constant term,
    2. Multiplicating a constant to one whose constant is not zero updates the constant term.
    """
    # Mul constant to Hamiltonian with constant zero.
    h = qm_o.Hamiltonian()
    h = h * constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = expected_h.constant * constant
    # 1. Multiplicating a constant to one whose constant is zero updates the constant term,
    assert h == expected_h

    # Mul constant to Hamiltonian with constant not zero.
    h = qm_o.Hamiltonian()
    initial_constant = 1.0
    h.constant = initial_constant
    h = h * constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = initial_constant * constant
    # 2. Multiplicating a constant to one whose constant is not zero updates the constant term.
    assert h == expected_h


@pytest.mark.parametrize("invalid_constant", [str(1), list([1, 2, 3]), dict({1: 2})])
def test_Hamiltonian_mul_wrt_invalid_constants(invalid_constant):
    """Mul Hamiltonian with invalid constants.

    Check if
    1. ValueError arises.
    """
    h = qm_o.Hamiltonian()
    # 1. ValueError arises.
    with pytest.raises(ValueError):
        h * invalid_constant


def test_Hamiltonian_mul_scalar_manually():
    """Mul Hamiltonian with manually decided scalars.

    Check if
    1. The terms are multiplied correctly,
    """
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    h = h * 2.0
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 2.0)
    # 1. The terms are multiplied correctly,
    assert h == expected_h

    h = 2.0 * h
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 4.0)
    # 1. The terms are multiplied correctly,
    assert h == expected_h


def test_Hamiltonian_mul_manually():
    """Multiply Hamiltonian with manually decided terms.

    Check if
    1. The terms are multiplied correctly.
    """
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
    # 1. The terms are multiplied correctly.
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
    # 1. The terms are multiplied correctly.
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
    # 1. The terms are multiplied correctly.
    assert h == expected_h

    x0 = qm_o.X(0)
    y0 = qm_o.Y(0)
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0j)
    op = x0 * y0
    # 1. The terms are multiplied correctly.
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
    # 1. The terms are multiplied correctly.
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
    # 1. The terms are multiplied correctly.
    assert h == expected_h

    h1 = 2.0 * x0 + 1.0
    h = y1 * h1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1), qm_o.PauliOperator(qm_o.Pauli.X, 0)), 2.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 1.0)
    # 1. The terms are multiplied correctly.
    assert h == expected_h

    h = h1 * y1
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1), qm_o.PauliOperator(qm_o.Pauli.X, 0)), 2.0
    )
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 1.0)
    # 1. The terms are multiplied correctly.
    assert h == expected_h

    h = h1 * h1
    expected_h = qm_o.Hamiltonian()
    expected_h.constant += 5.0
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 4.0)

    # 1. The terms are multiplied correctly.
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
    # 1. The terms are multiplied correctly.
    assert h == expected_h


@pytest.mark.parametrize(
    "pauli_combinations1",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
@pytest.mark.parametrize(
    "pauli_combinations2",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_rmul_wrt_different_qubits(
    pauli_combinations1, pauli_combinations2
):
    """Test Hamiltonian right multiplication with other Hamiltonians whose all the terms are with respect to different qubits.

    Check if
    1. multiplicating two Hamiltonians accumulates all terms correctly.
    """
    index1 = 0
    index2 = 1
    # Iterate over palui_combinations1.
    for pauli_combination1 in pauli_combinations1:

        # Iterate over palui_combinations2.
        for pauli_combination2 in pauli_combinations2:
            # Create the first Hamiltonian.
            #    Note: We now test the right multiplication operator, so we need to create the first Hamiltonian every time.
            h1 = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                h1.add_term((qm_o.PauliOperator(pauli1, index1),), 1.0)

            # Create the second Hamiltonian.
            h2 = qm_o.Hamiltonian()
            for pauli2 in pauli_combination2:
                # Add terms to a different qubit than the first Hamiltonian: index2.
                h2.add_term((qm_o.PauliOperator(pauli2, index2),), 1.0)

            # Mul the two Hamiltonians.
            h1 *= h2

            # Calculate the expected Hamiltonian.
            #    Note: If h1 = X1 + Y1 + Z1 + I1 and h2 = X2 + Y2 + Z2 + I2, then
            #          h = (X1 + Y1 + Z1 + I1) * (X2 + Y2 + Z2 + I2)
            #          = X1X2 + X1Y2 + X1Z2 + X1I2 + Y1X2 + Y1Y2 + Y1Z2 + Y1I2 + Z1X2 + Z1Y2 + Z1Z2 + Z1I2 + I1X2 + I1Y2 + I1Z2 + I1I2.
            #          Thus, the following loop accumulates all terms correctly.
            expected_h = qm_o.Hamiltonian()
            for pauli1 in pauli_combination1:
                for pauli2 in pauli_combination2:
                    expected_h.add_term(
                        (
                            qm_o.PauliOperator(pauli1, index1),
                            qm_o.PauliOperator(pauli2, index2),
                        ),
                        1.0,
                    )

            # 1. Multiplicating two Hamiltonians accumulates all terms correctly.
            assert h1 == expected_h


@pytest.mark.parametrize(
    "constant",
    [
        int(2),
        float(2.0),
        float(1.1),
        complex(2.0, 2.0),
        complex(2.0, 0.0),
        complex(0.0, 2.0),
    ],
)
def test_Hamiltonian_rmul_wrt_valid_constants(constant):
    """Right mul Hamiltonian with valid constants.

    Check if
    1. Multiplicating a constant to one whose constant is zero updates the constant term,
    2. Multiplicating a constant to one whose constant is not zero updates the constant term.
    """
    # Mul constant to Hamiltonian with constant zero.
    h = qm_o.Hamiltonian()
    h *= constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = expected_h.constant * constant
    # 1. Multiplicating a constant to one whose constant is zero updates the constant term,
    assert h == expected_h

    # Mul constant to Hamiltonian with constant not zero.
    h = qm_o.Hamiltonian()
    initial_constant = 1.0
    h.constant = initial_constant
    h *= constant
    # Create the expected Hamiltonian with the constant.
    expected_h = qm_o.Hamiltonian()
    expected_h.constant = initial_constant * constant
    # 2. Multiplicating a constant to one whose constant is not zero updates the constant term.
    assert h == expected_h


@pytest.mark.parametrize(
    "pauli_combinations",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_Hamiltonian_neg(pauli_combinations):
    """Test Hamiltonian negation with other Hamiltonians whose all the terms are with respect to the same qubits.

    Check if
    1. negating Hamiltonian shows all terms correctly.
    """
    index = 0
    # Iterate over pauli_combinations1.
    for pauli_combination in pauli_combinations:
        # Create the first Hamiltonian.
        h = qm_o.Hamiltonian()
        for pauli in pauli_combination:
            h.add_term((qm_o.PauliOperator(pauli, index),), 1.0)

        # Negate the Hamiltonian.
        h = -h

        # Calculate the expected Hamiltonian.
        expected_h = qm_o.Hamiltonian()
        for pauli in pauli_combination:
            expected_h.add_term(
                (qm_o.PauliOperator(pauli, index),),
                -1.0,
            )

        # 1. negating Hamiltonian shows all terms correctly.
        assert h == expected_h


def test_Hamiltonian_neg_manually():
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
    # 1. Negating a Hamiltonian negates all coefficients and the constant.
    assert h == expected_h
    h1 = -(2.0 * x0 + y1)
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), -2.0)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), -1.0)
    # 1. Negating a Hamiltonian negates all coefficients and the constant.
    assert h1 == expected_h


def test_coeff_complex_manually():
    """Test Hamiltonian with complex coefficients.

    Check if
    1. Complex coefficients are handled correctly in multiplication and addition.
    """
    h = qm_o.Hamiltonian()
    h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    h *= 1 + 1j * qm_o.Y(0)
    expected_h = qm_o.Hamiltonian()
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.I, 0),), 1.0j)
    expected_h.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), -1.0)
    # 1. Complex coefficients are handled correctly in multiplication and addition
    assert h == expected_h


# <<< Hamiltonian <<<


# >>> X, Y, Z >>>
@pytest.mark.parametrize("index", range(3))
def test_X(index):
    """Call X function.

    Check if
    1. the returned value is Hamltonian,
    2. the returned value's terms is {(X as PauliOperator, index): 1.0},
    3. the returned value's constant is 0.0,
    4. the returned value's num_qubits is index + 1.
    """
    h = qm_o.X(index)
    # 1. the returned value is Hamltonian,
    assert isinstance(h, qm_o.Hamiltonian)
    # 2. the returned value's terms is {(X as PauliOperator, index): 1.0},
    assert h.terms == {(qm_o.PauliOperator(qm_o.Pauli.X, index),): 1.0}
    # 3. the returned value's constant is 0.0,
    assert h.constant == 0.0
    # 4. the returned value's num_qubits is index + 1.
    assert h.num_qubits == index + 1


@pytest.mark.parametrize("index", range(3))
def test_Y(index):
    """Call Y function.

    Check if
    1. the returned value is Hamltonian,
    2. the returned value's terms is {(Y as PauliOperator, index): 1.0},
    3. the returned value's constant is 0.0,
    4. the returned value's num_qubits is index + 1.
    """
    h = qm_o.Y(index)
    # 1. the returned value is Hamltonian,
    assert isinstance(h, qm_o.Hamiltonian)
    # 2. the returned value's terms is {(X as PauliOperator, index): 1.0},
    assert h.terms == {(qm_o.PauliOperator(qm_o.Pauli.Y, index),): 1.0}
    # 3. the returned value's constant is 0.0,
    assert h.constant == 0.0
    # 4. the returned value's num_qubits is index + 1.
    assert h.num_qubits == index + 1


@pytest.mark.parametrize("index", range(3))
def test_Z(index):
    """Call Z function.

    Check if
    1. the returned value is Hamltonian,
    2. the returned value's terms is {(Z as PauliOperator, index): 1.0},
    3. the returned value's constant is 0.0,
    4. the returned value's num_qubits is index + 1.
    """
    h = qm_o.Z(index)
    # 1. the returned value is Hamltonian,
    assert isinstance(h, qm_o.Hamiltonian)
    # 2. the returned value's terms is {(X as PauliOperator, index): 1.0},
    assert h.terms == {(qm_o.PauliOperator(qm_o.Pauli.Z, index),): 1.0}
    # 3. the returned value's constant is 0.0,
    assert h.constant == 0.0
    # 4. the returned value's num_qubits is index + 1.
    assert h.num_qubits == index + 1


# <<< X, Y, Z <<<


# >>> simplify_pauliop_terms >>>
@pytest.mark.parametrize(
    "pauli_combinations",
    [
        list(itertools.permutations(qm_o.Pauli, 1)),
        list(itertools.permutations(qm_o.Pauli, 2)),
        list(itertools.permutations(qm_o.Pauli, 3)),
        list(itertools.permutations(qm_o.Pauli, 4)),
    ],
)
def test_simplify_pauliop_terms_on_same_qubit(pauli_combinations):
    """Run simplify_pauliop_terms for various Paulis on the same qubit.

    Check if
    1. The simplified PauliOperator tuple is correct,
    2. The simplified phase is correct.
    """
    index = 0
    # Iterate over pauli_combinations.
    for pauli_combination in pauli_combinations:
        # Run simplify_pauliop_terms on the current combination of Pauli products.
        pauli_ops = [qm_o.PauliOperator(pauli, index) for pauli in pauli_combination]
        simplified_pauli_ops, simplified_phase = qm_o.simplify_pauliop_terms(pauli_ops)

        # Compute the expected PauliOperator and phase.
        expected_pauli = pauli_combination[0]
        phases = []
        for pauli in pauli_combination[1:]:
            expected_pauli, phase = Utils.PAULI_PRODUCT_TABLE[(expected_pauli, pauli)]
            phases.append(phase)
        if expected_pauli == qm_o.Pauli.I:
            # If it is the identity, we expect an empty tuple.
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
def test_simplify_pauliop_terms_manually(
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
