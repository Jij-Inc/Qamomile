from qamomile.core import pauli_x, pauli_y, pauli_z
import jijmodeling as jm

def test_pauli_x():
    op = pauli_x(10)
    assert op.description == "PauliX"
    assert op.shape[0].value == jm.NumberLit(10).value

    op = pauli_x((10,5))
    assert op.description == "PauliX"
    assert op.shape[0].value == jm.NumberLit(10).value
    assert op.shape[1].value == jm.NumberLit(5).value

    op = pauli_x((10,))
    assert op.description == "PauliX"
    assert op.shape[0].value == jm.NumberLit(10).value

def test_pauli_y():
    op = pauli_y(10)
    assert op.description == "PauliY"
    assert op.shape[0].value == jm.NumberLit(10).value

    op = pauli_y((10,5))
    assert op.description == "PauliY"
    assert op.shape[0].value == jm.NumberLit(10).value
    assert op.shape[1].value == jm.NumberLit(5).value

    op = pauli_y((10,))
    assert op.description == "PauliY"
    assert op.shape[0].value == jm.NumberLit(10).value


def test_pauli_z():
    op = pauli_z(10)
    assert op.description == "PauliZ"
    assert op.shape[0].value == jm.NumberLit(10).value

    op = pauli_z((10,5))
    assert op.description == "PauliZ"
    assert op.shape[0].value == jm.NumberLit(10).value
    assert op.shape[1].value == jm.NumberLit(5).value

    op = pauli_z((10,))
    assert op.description == "PauliZ"
    assert op.shape[0].value == jm.NumberLit(10).value