from qamomile.core.compile import CompiledHamiltonian, PauliOperator, PauliType, SubstitutedQuantumExpression
from qamomile.core import Hamiltonian, pauli_x
import jijmodeling_transpiler.core as jtc
import jijmodeling as jm

def test_CompiledHamiltonian():
    n = 2
    X = pauli_x(n)
    N = jm.Placeholder("N")
    i = jm.Element("i", belong_to=(0,N))
    expr = X[i]
    compiled_hamiltonian = CompiledHamiltonian(substituted_hamiltonian=SubstitutedQuantumExpression(coeff={PauliOperator(PauliType.X, 0): 1.0}, constant=1.0, order=1), hamiltonian=Hamiltonian(expr), data=jtc.InstanceData({'N':n},{},[],{}),var_map = jtc.VariableMap({}, var_num=0, integer_bound={}),  qubit_index_map={})
    assert compiled_hamiltonian.substituted_hamiltonian.coeff[PauliOperator(PauliType.X, 0)] == 1.0
    assert compiled_hamiltonian.substituted_hamiltonian.constant == 1.0
    assert compiled_hamiltonian.substituted_hamiltonian.order == 1
    assert jm.is_same(compiled_hamiltonian.hamiltonian.hamiltonian, expr)
    assert compiled_hamiltonian.data.tensor_data == {'N':n}
    assert compiled_hamiltonian.var_map == jtc.VariableMap({}, var_num=0, integer_bound={})
    assert compiled_hamiltonian.qubit_index_map == {}



