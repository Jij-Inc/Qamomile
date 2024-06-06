from qamomile.core.compile.substitued_quantum_expr import SubstitutedQuantumExpression
from qamomile.core.compile import PauliOperator, PauliType

def test_SubstitutedQuantumExpression():
    quantum_expression = SubstitutedQuantumExpression(coeff={}, constant=0.0, order=0)
    assert quantum_expression.coeff == {}
    assert quantum_expression.constant == 0.0
    assert quantum_expression.order == 0

    quantum_expression = SubstitutedQuantumExpression(coeff={PauliOperator(PauliType.X, 0): 1.0}, constant=1.0, order=1)
    assert quantum_expression.coeff[PauliOperator(PauliType.X, 0)] == 1.0
    assert quantum_expression.constant == 1.0
    assert quantum_expression.order == 1

    quantum_expression = SubstitutedQuantumExpression(coeff={(PauliOperator(PauliType.X, 0),PauliOperator(PauliType.X, 1)): 1.0}, constant=-1.0, order=2)
    assert quantum_expression.coeff[(PauliOperator(PauliType.X, 0),PauliOperator(PauliType.X, 1))] == 1.0
    assert quantum_expression.constant == -1.0
    assert quantum_expression.order == 2
