from __future__ import annotations
import numpy as np
import qiskit.quantum_info as qk_ope
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel
from .qrao31 import Pauli, color_group_to_qrac_encode, create_pauli_term


def qrac21_encode_ising(
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

        pauli_terms.append(qk_ope.SparsePauliOp(pauli_operator, np.sqrt(2) * coeff))

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

        pauli_terms.append(qk_ope.SparsePauliOp(pauli_ope, 2 * coeff))

    if pauli_terms:
        # Remove paulis whose coefficients are zeros.

        qubit_op = sum(pauli_terms).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        n_qubit = max(1, n_qubit)
        qubit_op = qk_ope.SparsePauliOp("I" * n_qubit, 0)

    return qubit_op, offset, encoded_ope
