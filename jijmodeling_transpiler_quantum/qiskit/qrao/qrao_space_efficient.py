from __future__ import annotations
import numpy as np
import qiskit.quantum_info as qk_ope
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel
from .qrao31 import Pauli, create_pauli_term


def numbering_space_efficient_encode(
    ising: IsingModel,
) -> dict[int, tuple[int, Pauli]]:
    max_quad_index = max(max(t) for t in ising.quad.keys())
    max_linear_index = max(ising.linear.keys())
    num_vars = max(max_quad_index, max_linear_index) + 1

    encode = {}
    pauli_ope = [Pauli.X, Pauli.Y]
    for i in range(num_vars):
        qubit_index = i // 2
        color = i % 2
        encode[i] = (qubit_index, pauli_ope[color])
    return encode


def qrac_space_efficient_encode_ising(
    ising: IsingModel,
) -> tuple[qk_ope.SparsePauliOp, float, dict[int, tuple[int, Pauli]]]:
    encoded_ope = numbering_space_efficient_encode(ising)

    pauli_terms: list[qk_ope.SparsePauliOp] = []

    n_qubit = max(t[0] for t in encoded_ope.values()) + 1

    offset = ising.constant

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

        if color_i == color_j:
            pauli_ope = create_pauli_term([Pauli.Z], [color_i], n_qubit)
            pauli_terms.append(qk_ope.SparsePauliOp(pauli_ope, np.sqrt(3) * coeff))
        else:
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
