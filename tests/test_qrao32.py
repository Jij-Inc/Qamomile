import qiskit as qk
import qiskit.quantum_info as qk_ope
import numpy as np
from jijtranspiler_qiskit.qrao.qrao32 import (
    create_pauli_x_prime_term,
    create_pauli_y_prime_term,
    create_pauli_z_prime_term,
)


def test_create_pauli_x_prime_term():
    # The numbering of qubit is different between Qiskit and the paper
    # The operator X' is defined as follows (in qiskit notation):
    # X' = 1/2XX + 1/2ZX + IZ
    xps = [
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
    ]
    zps = [
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
    ]
    coeffs = [1.0, 1.0, 1.0]
    idx = 0

    create_pauli_x_prime_term(xps, zps, coeffs, idx)

    _pauli_terms: list[qk_ope.SparsePauliOp] = []
    for z_p, x_p, coeff in zip(zps, xps, coeffs):
        _pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)), coeff))

    answer = [
        qk_ope.SparsePauliOp(qk_ope.Pauli("XX"), 1 / 2),
        qk_ope.SparsePauliOp(qk_ope.Pauli("ZX"), 1 / 2),
        qk_ope.SparsePauliOp(qk_ope.Pauli("IZ"), 1.0),
    ]

    assert _pauli_terms == answer


def test_create_pauli_y_prime_term():
    # The numbering of qubit is different between Qiskit and the paper
    # The operator Y' is defined as follows (in qiskit notation):
    # Y' = 1/2XI + ZI + 1/2YY
    xps = [
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
    ]
    zps = [
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
    ]
    coeffs = [1.0, 1.0, 1.0]
    idx = 0

    create_pauli_y_prime_term(xps, zps, coeffs, idx)

    _pauli_terms: list[qk_ope.SparsePauliOp] = []
    for z_p, x_p, coeff in zip(zps, xps, coeffs):
        _pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)), coeff))

    answer = [
        qk_ope.SparsePauliOp(qk_ope.Pauli("XI"), 1 / 2),
        qk_ope.SparsePauliOp(qk_ope.Pauli("ZI"), 1.0),
        qk_ope.SparsePauliOp(qk_ope.Pauli("YY"), 1 / 2),
    ]

    assert _pauli_terms == answer


def test_create_pauli_z_prime_term():
    # The numbering of qubit is different between Qiskit and the paper
    # The operator Z' is defined as follows (in qiskit notation):
    # Z' = ZZ - 1/2IX - 1/2XZ
    xps = [
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
    ]
    zps = [
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
    ]
    coeffs = [1.0, 1.0, 1.0]
    idx = 0

    create_pauli_z_prime_term(xps, zps, coeffs, idx)

    _pauli_terms: list[qk_ope.SparsePauliOp] = []
    for z_p, x_p, coeff in zip(zps, xps, coeffs):
        _pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, x_p)), coeff))

    answer = [
        qk_ope.SparsePauliOp(qk_ope.Pauli("ZZ"), 1.0),
        qk_ope.SparsePauliOp(qk_ope.Pauli("IX"), -1 / 2),
        qk_ope.SparsePauliOp(qk_ope.Pauli("XZ"), -1 / 2),
    ]

    assert _pauli_terms == answer
