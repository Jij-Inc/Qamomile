from qamomile.core.compile import PauliType, PauliOperator

def test_pauli_type():
    assert PauliType.X.value == "X"
    assert PauliType.Y.value == "Y"
    assert PauliType.Z.value == "Z"


def test_pauli_type_hash():
    assert hash(PauliType.X) != hash(PauliType.Y)
    assert hash(PauliType.X) != hash(PauliType.Z)
    assert hash(PauliType.Y) != hash(PauliType.Z)

def test_pauli_oprator():
    pauli_operator = PauliOperator(PauliType.X, 0)
    assert pauli_operator.pauli_type == PauliType.X
    assert pauli_operator.qubit_index == 0

    pauli_operator = PauliOperator(PauliType.Y, 1)
    assert pauli_operator.pauli_type == PauliType.Y
    assert pauli_operator.qubit_index == 1

    pauli_operator = PauliOperator(PauliType.Z, 2)
    assert pauli_operator.pauli_type == PauliType.Z
    assert pauli_operator.qubit_index == 2

def test_pauli_operator_hash():
    assert hash(PauliOperator(PauliType.X, 0)) != hash(PauliOperator(PauliType.Y, 0))
    assert hash(PauliOperator(PauliType.X, 0)) != hash(PauliOperator(PauliType.Z, 0))
    assert hash(PauliOperator(PauliType.Y, 0)) != hash(PauliOperator(PauliType.Z, 0))
    assert hash(PauliOperator(PauliType.X, 0)) != hash(PauliOperator(PauliType.X, 1))
    assert hash(PauliOperator(PauliType.Y, 0)) != hash(PauliOperator(PauliType.Y, 1))
    assert hash(PauliOperator(PauliType.Z, 0)) != hash(PauliOperator(PauliType.Z, 1))

    assert PauliOperator(PauliType.X, 0) != PauliOperator(PauliType.Y, 0)
    assert PauliOperator(PauliType.X, 0) != PauliOperator(PauliType.Z, 0)
    assert PauliOperator(PauliType.Y, 0) != PauliOperator(PauliType.Z, 0)
    assert PauliOperator(PauliType.X, 0) != PauliOperator(PauliType.X, 1)
    assert PauliOperator(PauliType.Y, 0) != PauliOperator(PauliType.Y, 1)
    assert PauliOperator(PauliType.Z, 0) != PauliOperator(PauliType.Z, 1)

    assert PauliOperator(PauliType.X, 0) == PauliOperator(PauliType.X, 0)
    assert PauliOperator(PauliType.Y, 0) == PauliOperator(PauliType.Y, 0)
    assert PauliOperator(PauliType.Z, 0) == PauliOperator(PauliType.Z, 0)