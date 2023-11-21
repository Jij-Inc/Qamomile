from __future__ import annotations

import numpy as np
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label

from jijmodeling_transpiler_quantum.core import qubo_to_ising


def to_ising_operator_from_qubo(
    qubo: dict[tuple[int, int], float], n_qubit: int
) -> tuple[Operator, float]:
    """Convert a given QUBO to an Ising operator.

    This function converts a quadratic unconstrained binary optimization (QUBO) problem
    into an Ising operator and returns the operator and the constant offset of the
    Ising model.

    Args:
        qubo (dict[tuple[int, int], float]): The QUBO to be converted. It is a dictionary
        where the keys are tuples representing the variables in the quadratic term
        and the values are the coefficients of these terms.
        n_qubit (int): The total number of qubits.

    Returns:
        tuple[Operator, float]: The Ising operator and the constant offset.
    """
    ising = qubo_to_ising(qubo)
    offset = ising.constant
    quri_operator = Operator()

    # convert linear parts of the objective function into Operator.
    for idx, coeff in ising.linear.items():
        if coeff != 0.0:
            quri_operator += Operator({pauli_label(f"Z{idx}"): coeff})

    # convert quadratic parts of the objective function into Operator.
    for (i, j), coeff in ising.quad.items():
        if coeff != 0.0:
            quri_operator += Operator({pauli_label(f"Z{i} Z{j}"): coeff})
    quri_operator.constant = offset
    # Add the constant part to the operator.
    # op += Operator({pauli_label(''): ising.offset})

    return quri_operator, offset
