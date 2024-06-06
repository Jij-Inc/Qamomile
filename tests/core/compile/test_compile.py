import qamomile.core as qamo
from qamomile.core.compile.compile import (
    make_reverse_var_map,
    make_substituted_hamiltonian,
    PauliOperator,
    PauliType,
    make_compiled_hamiltonian,
    compile_hamiltonian,
    qubit_index_mapper,
)
import jijmodeling_transpiler.core as jtc
import jijmodeling as jm


def heisenberg_model(n=3):
    X = qamo.pauli_x(n)
    Y = qamo.pauli_y(n)
    Z = qamo.pauli_z(n)
    N = jm.Placeholder("N")
    i = jm.Element("i", belong_to=(0, N))
    expr = jm.sum(
        [i], X[i] * X[(i + 1) % N] + Y[i] * Y[(i + 1) % N] + Z[i] * Z[(i + 1) % N]
    )
    op = qamo.Hamiltonian(expr, "Heisenberg_model")
    instance_data = {"N": n}
    return op, instance_data


def simple_hamiltonian(n=2):
    X = qamo.pauli_x(n)

    N = jm.Placeholder("N")
    i = jm.Element("i", belong_to=(0, N))
    expr = jm.sum([i], X[i] * X[i])
    op = qamo.Hamiltonian(expr, "simple_model")
    instance_data = {"N": n}
    return op, instance_data


def test_make_substituted_hamiltonian():
    op, instance = heisenberg_model()
    subsituted_expr, var_map, instance_data = make_substituted_hamiltonian(op, instance)
    assert isinstance(var_map, jtc.VariableMap)

    assert var_map.var_map == {
        "_PauliX": {(0,): 0, (1,): 1, (2,): 6},
        "_PauliY": {(0,): 2, (1,): 3, (2,): 7},
        "_PauliZ": {(0,): 4, (1,): 5, (2,): 8},
    }
    assert isinstance(subsituted_expr, jtc.substitute.SubstitutedExpression)

    assert subsituted_expr.coeff == {
        (0, 1): 1,
        (2, 3): 1,
        (4, 5): 1,
        (1, 6): 1,
        (3, 7): 1,
        (5, 8): 1,
        (0, 6): 1,
        (2, 7): 1,
        (4, 8): 1,
    }

    assert isinstance(instance_data, jtc.InstanceData)
    assert instance_data.tensor_data == {"N": 3}


def test_qubit_index_mapper():
    var_map = jtc.VariableMap(
        {
            "_PauliX": {(0,): 0, (1,): 1, (2,): 6},
            "_PauliY": {(0,): 2, (1,): 3, (2,): 7},
            "_PauliZ": {(0,): 4, (1,): 5, (2,): 8},
        },
        var_num=0,
        integer_bound={},
    )
    qubit_index_map = qubit_index_mapper(var_map)
    assert qubit_index_map == {(0,): 0, (1,): 1, (2,): 2}

    var_map = jtc.VariableMap(
        {
            "_PauliX": {(0, 1): 0, (1, 1): 1, (2, 1): 6},
            "_PauliY": {(0, 1): 2, (1, 1): 3, (2, 1): 7},
            "_PauliZ": {(0, 1): 4, (1, 1): 5, (2, 1): 8},
        },
        var_num=0,
        integer_bound={},
    )
    qubit_index_map = qubit_index_mapper(var_map)
    assert qubit_index_map == {(0, 1): 0, (1, 1): 1, (2, 1): 2}

    var_map = jtc.VariableMap(
        {
            "_PauliX": {(0, 0): 0, (1, 0): 1, (2, 0): 6},
            "_PauliY": {(0, 1): 2, (1, 1): 3, (2, 1): 7},
            "_PauliZ": {(0, 2): 4, (1, 2): 5, (2, 2): 8},
        },
        var_num=0,
        integer_bound={},
    )
    qubit_index_map = qubit_index_mapper(var_map)
    assert qubit_index_map == {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (0, 1): 3,
        (1, 1): 4,
        (2, 1): 5,
        (0, 2): 6,
        (1, 2): 7,
        (2, 2): 8,
    }


def test_make_reverse_var_map():
    op, instance = heisenberg_model()
    _, var_map, _ = make_substituted_hamiltonian(op, instance)
    reverse_var_map, _ = make_reverse_var_map(var_map)
    assert reverse_var_map == {
        0: PauliOperator(PauliType.X, 0),
        1: PauliOperator(PauliType.X, 1),
        2: PauliOperator(PauliType.Y, 0),
        3: PauliOperator(PauliType.Y, 1),
        4: PauliOperator(PauliType.Z, 0),
        5: PauliOperator(PauliType.Z, 1),
        6: PauliOperator(PauliType.X, 2),
        7: PauliOperator(PauliType.Y, 2),
        8: PauliOperator(PauliType.Z, 2),
    }

    var_map = jtc.VariableMap(
        {
            "_PauliX": {(0, 0): 0, (1, 0): 1, (2, 0): 6},
            "_PauliY": {(0, 1): 2, (1, 1): 3, (2, 1): 7},
            "_PauliZ": {(0, 2): 4, (1, 2): 5, (2, 2): 8},
        },
        var_num=0,
        integer_bound={},
    )
    reverse_var_map, _ = make_reverse_var_map(var_map)
    
    assert reverse_var_map == {
        0: PauliOperator(PauliType.X, 0),
        1: PauliOperator(PauliType.X, 1),
        6: PauliOperator(PauliType.X, 2),
        2: PauliOperator(PauliType.Y, 3),
        3: PauliOperator(PauliType.Y, 4),
        7: PauliOperator(PauliType.Y, 5),
        4: PauliOperator(PauliType.Z, 6),
        5: PauliOperator(PauliType.Z, 7),
        8: PauliOperator(PauliType.Z, 8),
    }


def test_make_compiled_hamiltonian():
    op, instance = heisenberg_model()
    subsituted_expr, var_map, instance_data = make_substituted_hamiltonian(op, instance)
    reverse_var_map, _ = make_reverse_var_map(var_map)
    subsituted_hamiltonian = make_compiled_hamiltonian(subsituted_expr, reverse_var_map)

    assert isinstance(subsituted_hamiltonian, qamo.compile.SubstitutedQuantumExpression)

    assert subsituted_hamiltonian.coeff == {
        (PauliOperator(PauliType.X, 0), PauliOperator(PauliType.X, 1)): 1,
        (PauliOperator(PauliType.Y, 0), PauliOperator(PauliType.Y, 1)): 1,
        (PauliOperator(PauliType.Z, 0), PauliOperator(PauliType.Z, 1)): 1,
        (PauliOperator(PauliType.X, 1), PauliOperator(PauliType.X, 2)): 1,
        (PauliOperator(PauliType.Y, 1), PauliOperator(PauliType.Y, 2)): 1,
        (PauliOperator(PauliType.Z, 1), PauliOperator(PauliType.Z, 2)): 1,
        (PauliOperator(PauliType.X, 0), PauliOperator(PauliType.X, 2)): 1,
        (PauliOperator(PauliType.Y, 0), PauliOperator(PauliType.Y, 2)): 1,
        (PauliOperator(PauliType.Z, 0), PauliOperator(PauliType.Z, 2)): 1,
    }
    assert subsituted_hamiltonian.constant == 0
    assert subsituted_hamiltonian.order == 2


def test_simple_model():
    op, instance = simple_hamiltonian()
    subsituted_expr, var_map, instance_data = make_substituted_hamiltonian(op, instance)
    reverse_var_map,_ = make_reverse_var_map(var_map)
    subsituted_hamiltonian = make_compiled_hamiltonian(subsituted_expr, reverse_var_map)

    assert isinstance(subsituted_hamiltonian, qamo.compile.SubstitutedQuantumExpression)
    print(subsituted_hamiltonian.coeff)

    # X_0 * X_0 + X_1 * X_1 cannot be simplified
    assert subsituted_hamiltonian.coeff == {
        (PauliOperator(PauliType.X, 0), PauliOperator(PauliType.X, 0)): 1,
        (PauliOperator(PauliType.X, 1), PauliOperator(PauliType.X, 1)): 1,
    }
    assert subsituted_hamiltonian.constant == 0
    assert subsituted_hamiltonian.order == 2

    assert isinstance(instance_data, jtc.InstanceData)
    assert instance_data.tensor_data == {"N": 2}

# def test_compile_hamiltonian():
#     op, instance = heisenberg_model()
#     subsituted_expr, var_map, instance_data = make_substituted_hamiltonian(op, instance)
#     reverse_var_map = make_reverse_var_map(var_map)
#     subsituted_hamiltonian = make_compiled_hamiltonian(subsituted_expr, reverse_var_map)
#     compiled_hamiltonian = compile_hamiltonian(op, instance)

#     assert isinstance(compiled_hamiltonian, qamo.compile.CompiledHamiltonian)
#     assert compiled_hamiltonian.substituted_hamiltonian == subsituted_hamiltonian
#     assert compiled_hamiltonian.hamiltonian == op
#     assert compiled_hamiltonian.data == instance_data
#     assert compiled_hamiltonian.var_map == var_map