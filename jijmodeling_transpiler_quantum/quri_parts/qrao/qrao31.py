from __future__ import annotations

import enum

import numpy as np
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label

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


def create_pauli_term(
    operators: list[Pauli], indices: list[int], n_qubit: int
) -> str:
    """Create a Pauli term string given a list of operators and indices.

    Args:
        operators (list[Pauli]): A list of Pauli operators.
        indices (list[int]): A list of indices corresponding to each operator.
        n_qubit (int): The total number of qubits.

    Returns:
        str: The created Pauli term string.
    """
    pauli_str = ""
    for ope, idx in zip(operators, indices):
        if ope == Pauli.X:
            pauli_str += f"X{idx} "
        elif ope == Pauli.Y:
            pauli_str += f"Y{idx} "
        elif ope == Pauli.Z:
            pauli_str += f"Z{idx} "
    return pauli_str.rstrip()


def qrac31_encode_ising(
    ising: IsingModel, color_group: dict[int, list[int]]
) -> tuple[Operator, float, dict[int, tuple[int, Pauli]]]:
    """Encode an Ising model and a color group into QRAC31.

    Args:
        ising (IsingModel): The Ising model to be encoded.
        color_group (dict[int, list[int]]): The color group mapping for encoding.

    Returns:
        tuple[Operator, float, dict[int, tuple[int, Pauli]]]: The encoded quantum operator,
        the offset of the Ising model, and the encoded operation as a dictionary.
    """
    encoded_ope = color_group_to_qrac_encode(color_group)

    pauli_terms: list[Operator] = []

    offset = ising.constant
    n_qubit = len(color_group)

    for idx, coeff in ising.linear.items():
        if coeff == 0.0:
            continue

        color, pauli_kind = encoded_ope[idx]
        pauli_operator = create_pauli_term([pauli_kind], [color], n_qubit)

        pauli_terms.append(
            Operator({pauli_label(pauli_operator): np.sqrt(3) * coeff})
        )

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
        pauli_terms.append(Operator({pauli_label(pauli_ope): 3 * coeff}))

    if pauli_terms:
        qubit_op = Operator()
        for term in pauli_terms:
            qubit_op += term
    else:
        n_qubit = max(1, n_qubit)
        qubit_op = Operator({PAULI_IDENTITY * n_qubit: 0})

    return qubit_op, offset, encoded_ope
