from __future__ import annotations
import numpy as np
import qiskit.quantum_info as qk_ope

from jijmodeling_transpiler_quantum.core import qubo_to_ising


def to_ising_operator_from_qubo(
    qubo: dict[tuple[int, int], float], n_qubit: int
) -> tuple[qk_ope.SparsePauliOp, float]:
    """Returns a quantum circuit that represents the QUBO."""
    ising = qubo_to_ising(qubo)
    pauli_terms: list[qk_ope.SparsePauliOp] = []

    offset = ising.constant
    zero = np.zeros(n_qubit, dtype=bool)

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if coeff == 0.0:
            continue

        z_p = zero.copy()
        z_p[idx] = True

        pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, zero)), coeff))

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        if i == j:
            offset += coeff
        else:
            z_p = zero.copy()
            z_p[i] = True
            z_p[j] = True
            pauli_terms.append(qk_ope.SparsePauliOp(qk_ope.Pauli((z_p, zero)), coeff))

    if pauli_terms:
        # Remove paulis whose coefficients are zeros.
        qubit_op = sum(pauli_terms).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        n_qubit = max(1, n_qubit)
        qubit_op = qk_ope.SparsePauliOp("I" * n_qubit, 0)

    return qubit_op, offset
