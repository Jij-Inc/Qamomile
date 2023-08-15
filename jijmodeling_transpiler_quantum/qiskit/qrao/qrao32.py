from __future__ import annotations
import numpy as np
import qiskit.quantum_info as qk_ope
from jijmodeling_transpiler_quantum.core.ising_qubo import IsingModel
from .qrao31 import Pauli, color_group_to_qrac_encode


def create_pauli_x_prime_term(
    xps: list[np.ndarray], zps: list[np.ndarray], coeffs: list[float], idx: int
):
    xps[0][idx] = True
    xps[0][idx + 1] = True
    coeffs[0] *= 1 / 2

    xps[1][idx] = True
    zps[1][idx + 1] = True
    coeffs[1] *= 1 / 2

    zps[2][idx] = True


def create_pauli_y_prime_term(
    xps: list[np.ndarray], zps: list[np.ndarray], coeffs: list[float], idx: int
):
    xps[0][idx + 1] = True
    coeffs[0] *= 1 / 2

    zps[1][idx + 1] = True

    xps[2][idx] = True
    zps[2][idx] = True
    xps[2][idx + 1] = True
    zps[2][idx + 1] = True
    coeffs[2] *= 1 / 2


def create_pauli_z_prime_term(
    xps: list[np.ndarray], zps: list[np.ndarray], coeffs: list[float], idx: int
):
    zps[0][idx] = True
    zps[0][idx + 1] = True

    xps[1][idx] = True
    coeffs[1] *= -1 / 2

    zps[2][idx] = True
    xps[2][idx + 1] = True
    coeffs[2] *= -1 / 2


def create_pauli_prime_terms(operator: Pauli, index: int, n_qubit: int):
    coeffs = [1.0, 1.0, 1.0]
    zps = [
        np.zeros(n_qubit, dtype=bool),
        np.zeros(n_qubit, dtype=bool),
        np.zeros(n_qubit, dtype=bool),
    ]
    xps = [
        np.zeros(n_qubit, dtype=bool),
        np.zeros(n_qubit, dtype=bool),
        np.zeros(n_qubit, dtype=bool),
    ]
    if operator == Pauli.X:
        create_pauli_x_prime_term(xps, zps, coeffs, 2 * index)
    elif operator == Pauli.Y:
        create_pauli_y_prime_term(xps, zps, coeffs, 2 * index)
    elif operator == Pauli.Z:
        create_pauli_z_prime_term(xps, zps, coeffs, 2 * index)

    return xps, zps, coeffs


def create_pauli_linear_term(operator: Pauli, index: int, n_qubit: int):
    xps, zps, coeffs = create_pauli_prime_terms(operator, index, n_qubit)

    _pauli_terms: list[qk_ope.SparsePauliOp] = []
    for z_p, x_p, coeff in zip(zps, xps, coeffs):
        _pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)), coeff))

    return _pauli_terms


def create_pauli_quad_term(operators: list[Pauli], indices: list[int], n_qubit: int):
    xps_i, zps_i, coeffs_i = create_pauli_prime_terms(operators[0], indices[0], n_qubit)
    xps_j, zps_j, coeffs_j = create_pauli_prime_terms(operators[1], indices[1], n_qubit)

    _pauli_terms: list[qk_ope.SparsePauliOp] = []
    for x_p_i, z_p_i, coeff_i in zip(xps_i, zps_i, coeffs_i):
        for x_p_j, z_p_j, coeff_j in zip(xps_j, zps_j, coeffs_j):
            _pauli_terms.append(
                qk_ope.SparsePauliOp(
                    qk_ope.Pauli((z_p_i | z_p_j, x_p_i | x_p_j)),
                    coeff_i * coeff_j,
                )
            )

    return _pauli_terms


def qrac32_encode_ising(
    ising: IsingModel, color_group: dict[int, list[int]]
) -> tuple[qk_ope.SparsePauliOp, float, dict[int, tuple[int, Pauli]]]:
    encoded_ope = color_group_to_qrac_encode(color_group)

    pauli_terms: list[qk_ope.SparsePauliOp] = []

    offset = ising.constant
    n_qubit = 2 * len(color_group)

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if coeff == 0.0:
            continue

        color, pauli_kind = encoded_ope[idx]
        pauli_operator_terms = create_pauli_linear_term(pauli_kind, color, n_qubit)
        pauli_terms.extend(pauli_operator_terms)

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        if i == j:
            offset += coeff
            continue

        color_i, pauli_kind_i = encoded_ope[i]

        color_j, pauli_kind_j = encoded_ope[j]

        pauli_operator_terms = create_pauli_quad_term(
            [pauli_kind_i, pauli_kind_j], [color_i, color_j], n_qubit
        )

        pauli_terms.extend(pauli_operator_terms)

    if pauli_terms:
        # Remove paulis whose coefficients are zeros.

        qubit_op = sum(pauli_terms).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        n_qubit = max(1, n_qubit)
        qubit_op = qk_ope.SparsePauliOp("I" * n_qubit, 0)

    return qubit_op, offset, encoded_ope
