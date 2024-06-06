from __future__ import annotations
import jijmodeling_transpiler.core as jtc
from qamomile.core.compile import (
    CompiledHamiltonian,
    PauliOperator,
    PauliType,
    SubstitutedQuantumExpression,
)
from qamomile.core import Hamiltonian


def make_substituted_hamiltonian(
    hamiltonian: Hamiltonian, instance: dict
) -> tuple[jtc.substitute.SubstitutedExpression, jtc.VariableMap, jtc.InstanceData]:
    el_map = {}
    subs, el_map = jtc.substitute.convert_to_substitutable(
        hamiltonian.hamiltonian, el_map
    )
    var_map = jtc.VariableMap({}, var_num=0, integer_bound={})

    ph_vars = subs.extract_type(jtc.substitute.substitutable_expr.Placeholder)
    ph_names = [p.name for p in ph_vars]
    instance_data = jtc.InstanceData(instance, {}, ph_names, indices={})

    subsituted_expr = subs.substitute(var_map, instance_data)

    return subsituted_expr, var_map, instance_data


def pauli_str_to_pauli_type(pauli_str: str) -> PauliType:
    if pauli_str == "_PauliX":
        return PauliType.X
    elif pauli_str == "_PauliY":
        return PauliType.Y
    elif pauli_str == "_PauliZ":
        return PauliType.Z
    else:
        raise ValueError("Invalid Pauli string")


def qubit_index_mapper(var_map: jtc.VariableMap) -> dict[tuple[int, ...], int]:
    qubit_index_map = {}
    qubit_index = 0
    for pauli_op, indices_map in var_map.var_map.items():
        for indices, _ in indices_map.items():
            if indices not in qubit_index_map.keys():
                qubit_index_map[indices] = qubit_index
                qubit_index += 1

    return qubit_index_map


def make_reverse_var_map(var_map: jtc.VariableMap) -> dict[int, PauliOperator]:
    reverse_var_map = {}
    qubit_index_map = qubit_index_mapper(var_map)
    for pauli_op, indices_map in var_map.var_map.items():
        pauli_type = pauli_str_to_pauli_type(pauli_op)
        for indices, var in indices_map.items():
            qubit_index = qubit_index_map[indices]
            reverse_var_map[var] = PauliOperator(pauli_type, qubit_index)
    return reverse_var_map, qubit_index_map


def make_compiled_hamiltonian(
    subsituted_expr: jtc.substitute.SubstitutedExpression,
    reverse_var_map: dict[int, PauliOperator],
):
    substituted_paulis = {}
    for indices, coeff in subsituted_expr.coeff.items():
        pauli_ops = []
        for i in indices:
            pauli_ops.append(reverse_var_map[i])

        substituted_paulis[tuple(pauli_ops)] = coeff
    return SubstitutedQuantumExpression(
        coeff=substituted_paulis,
        constant=subsituted_expr.constant,
        order=subsituted_expr.order,
    )


def compile_hamiltonian(
    hamiltonian: Hamiltonian, instance: dict
) -> CompiledHamiltonian:
    subsituted_expr, var_map, instance_data = make_substituted_hamiltonian(
        hamiltonian, instance
    )
    reverse_var_map, qubit_index_map = make_reverse_var_map(var_map)
    substituted_hamiltonian = make_compiled_hamiltonian(
        subsituted_expr, reverse_var_map
    )
    return CompiledHamiltonian(
        substituted_hamiltonian,
        hamiltonian,
        instance_data,
        var_map,
        var_map.deci_var_shape,
    )
