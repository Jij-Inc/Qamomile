from __future__ import annotations
import enum
import numpy as np
import qiskit.quantum_info as qk_ope
from jijtranspiler_qiskit.ising_qubo.ising_qubo import qubo_to_ising


class Pauli(enum.Enum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


def color_group_for_qrac_encode(color_group: dict[int, list[int]]):
    qrac31 = {}
    pauli_ope = [Pauli.Z, Pauli.X, Pauli.Y]
    for color, group in color_group.items():
        for ope_idx, bit_index in enumerate(group):
            qrac31[bit_index] = (color, pauli_ope[ope_idx])
    return qrac31


def qrac31_encode(
    qubo: dict[tuple[int, int], float],
    color_group: dict[int, list[int]]
) -> tuple[qk_ope.SparsePauliOp, float]:
    ising = qubo_to_ising(qubo)

    encoded_ope = color_group_for_qrac_encode(color_group)

    pauli_terms: list[qk_ope.SparsePauliOp] = []

    offset = ising.constant
    n_qubit = len(color_group)
    zero = np.zeros(n_qubit, dtype=bool)

    operator_parity = {Pauli.X: (False, True), Pauli.Y: (True, False), Pauli.Z: (True, False)}

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coef in ising.linear.items():
        if coef == 0.0:
            continue
 
        weight = coef
        color, idx_pauli = encoded_ope[idx]
        z_parity, x_parity = operator_parity[idx_pauli]
        if z_parity:
            z_p = zero.copy()
            z_p[color] = True
        else:
            z_p = zero
        if x_parity:
            x_p = zero.copy()
            x_p[color] = True
        else:
            x_p = zero

        pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)), -weight))
        offset += weight

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        weight = coeff

        if i == j:
            offset += weight
            continue

        color_i, pauli_i_index = encoded_ope[i]
        zi_p, xi_p = operator_parity[pauli_i_index]

        color_j, pauli_j_index = encoded_ope[j]
        zj_p, xj_p = operator_parity[pauli_j_index]

        if zi_p or zj_p:
            z_p = zero.copy()
            z_p[color_i] = zi_p
            z_p[color_j] = zj_p
        else:
            z_p = zero

        if xi_p or xj_p:
            x_p = zero.copy()
            x_p[color_i] = xi_p
            x_p[color_j] = xj_p
        else:
            x_p = zero

        pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)), weight))

    if pauli_terms:
        # Remove paulis whose coefficients are zeros.
        qubit_op = sum(pauli_terms).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        n_qubit = max(1, n_qubit)
        qubit_op = qk_ope.SparsePauliOp("I" * n_qubit, 0)

    return qubit_op, offset
