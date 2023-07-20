from __future__ import annotations
import numpy as np
import qiskit.quantum_info as qk_ope

from jijmodeling_transpiler_quantum.core import qubo_to_ising

from quri_parts.core.operator import pauli_label
from quri_parts.core.operator import PAULI_IDENTITY
from quri_parts.core.operator import Operator


def to_ising_operator_from_qubo_quri(
    qubo: dict[tuple[int, int], float], n_qubit: int
) -> tuple[Operator, float]:
    """Returns a quantum circuit that represents the QUBO."""
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
