from __future__ import annotations
import enum
import numpy as np
import qiskit.quantum_info as qk_ope
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel


class Pauli(enum.Enum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


def color_group_to_qrac_encode(
    color_group: dict[int, list[int]]
) -> dict[int, tuple[int, Pauli]]:
    """qrac encode

    Args:
        color_group (dict[int, list[int]]): key is color (qubit's index). value is list of bit's index.

    Returns:
        dict[int, tuple[int, Pauli]]: key is bit's index. value is tuple of qubit's index and Pauli operator kind.

    Examples:
        >>> color_group = {0: [0, 1, 2], 1: [3, 4], 2: [6,]}
        >>> color_group_for_qrac_encode(color_group)
        {0: (0, <Pauli.Z: 3>), 1: (0, <Pauli.X: 1>), 2: (0, <Pauli.Y: 2>), 3: (1, <Pauli.Z: 3>), 4: (1, <Pauli.X: 1>), 6: (2, <Pauli.Z: 3>)}

    """
    qrac31 = {}
    pauli_ope = [Pauli.Z, Pauli.X, Pauli.Y]
    for color, group in color_group.items():
        for ope_idx, bit_index in enumerate(group):
            qrac31[bit_index] = (color, pauli_ope[ope_idx])
    return qrac31


def create_pauli_term(operators: list[Pauli], indices: list[int], n_qubit: int):
    z_p = np.zeros(n_qubit, dtype=bool)
    x_p = np.zeros(n_qubit, dtype=bool)
    for ope, idx in zip(operators, indices):
        if ope == Pauli.X:
            x_p[idx] = True
        elif ope == Pauli.Y:
            x_p[idx] = True
            z_p[idx] = True
        elif ope == Pauli.Z:
            z_p[idx] = True
    return qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)))


def qrac31_encode_ising(
    ising: IsingModel, color_group: dict[int, list[int]]
) -> tuple[qk_ope.SparsePauliOp, float, dict[int, tuple[int, Pauli]]]:
    encoded_ope = color_group_to_qrac_encode(color_group)

    pauli_terms: list[qk_ope.SparsePauliOp] = []

    offset = ising.constant
    n_qubit = len(color_group)

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if coeff == 0.0:
            continue

        color, pauli_kind = encoded_ope[idx]
        pauli_operator = create_pauli_term([pauli_kind], [color], n_qubit)

        pauli_terms.append(qk_ope.SparsePauliOp(pauli_operator, np.sqrt(3) * coeff))

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        if i == j:
            offset += coeff
            continue

        color_i, pauli_kind_i = encoded_ope[i]

        color_j, pauli_kind_j = encoded_ope[j]

        pauli_ope = create_pauli_term(
            [pauli_kind_i, pauli_kind_j], [color_i, color_j], n_qubit
        )

        pauli_terms.append(qk_ope.SparsePauliOp(pauli_ope, 3 * coeff))

    if pauli_terms:
        # Remove paulis whose coefficients are zeros.

        qubit_op = sum(pauli_terms).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        n_qubit = max(1, n_qubit)
        qubit_op = qk_ope.SparsePauliOp("I" * n_qubit, 0)

    return qubit_op, offset, encoded_ope
