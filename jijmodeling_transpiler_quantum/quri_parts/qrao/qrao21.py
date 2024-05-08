from __future__ import annotations

import numpy as np
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label

from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel

from .qrao31 import Pauli, color_group_to_qrac_encode, create_pauli_term


def qrac21_encode_ising(
    ising: IsingModel, color_group: dict[int, list[int]]
) -> tuple[Operator, float, dict[int, tuple[int, Pauli]]]:
    """Encode an Ising model and a color group into QRAC21.

    This function encodes an Ising model into the operators, and returns as the QURI Parts Operator,
    the offset(constant) of the Ising model, and the encoded operation as a dictionary.

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
        pauli_str = create_pauli_term([pauli_kind], [color], n_qubit)
        pauli_terms.append(
            Operator({pauli_label(pauli_str): np.sqrt(2) * coeff})
        )

    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        if i == j:
            offset += coeff
            continue

        color_i, pauli_kind_i = encoded_ope[i]

        color_j, pauli_kind_j = encoded_ope[j]

        pauli_str = create_pauli_term(
            [pauli_kind_i, pauli_kind_j], [color_i, color_j], n_qubit
        )

        pauli_terms.append(Operator({pauli_label(pauli_str): 2 * coeff}))

    if pauli_terms:
        qubit_op = Operator()
        for term in pauli_terms:
            qubit_op += term
    else:
        n_qubit = max(1, n_qubit)
        qubit_op = Operator({PAULI_IDENTITY * n_qubit: 0})

    return qubit_op, offset, encoded_ope
