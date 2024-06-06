from qamomile.core.compile import compile_ising_model
from qamomile.core.compile import PauliOperator, PauliType, SubstitutedQuantumExpression
from qamomile.core import IsingModel

def test_compile_ising_model():
    ising = IsingModel({(0, 1): 2.0}, {0: 4.0, 1: 5.0}, 6.0)
    compiled = compile_ising_model(ising)
    assert compiled.constant == 6.0
    print(compiled.coeff)
    assert compiled.coeff == {
        (PauliOperator(PauliType.Z, 0), PauliOperator(PauliType.Z, 1)): 2.0,
        (PauliOperator(PauliType.Z, 0),): 4.0,
        (PauliOperator(PauliType.Z, 1),): 5.0,
    }

    ising = IsingModel({(0, 1): 2.0}, {}, 0.0)
    compiled = compile_ising_model(ising)
    assert compiled.constant == 0.0
    print(compiled.coeff)
    assert compiled.coeff == {
        (PauliOperator(PauliType.Z, 0), PauliOperator(PauliType.Z, 1)): 2.0,
    }

    ising = IsingModel({}, {0: 4.0, 1: 5.0}, 0.0)
    compiled = compile_ising_model(ising)
    assert compiled.constant == 0.0
    print(compiled.coeff)
    assert compiled.coeff == {
        (PauliOperator(PauliType.Z, 0),): 4.0,
        (PauliOperator(PauliType.Z, 1),): 5.0,
    }