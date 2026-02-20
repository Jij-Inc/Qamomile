"""QURI Parts observable support.

This module provides conversion from qamomile.observable.Hamiltonian
to QURI Parts Operator for use with QURI Parts estimator primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import qamomile.observable as qm_o

if TYPE_CHECKING:
    from quri_parts.core.operator import Operator


def hamiltonian_to_quri_operator(hamiltonian: qm_o.Hamiltonian) -> "Operator":
    """Convert qamomile.observable.Hamiltonian to QURI Parts Operator.

    Args:
        hamiltonian: The qamomile.observable.Hamiltonian to convert

    Returns:
        QURI Parts Operator representation

    Example:
        ```python
        import qamomile.observable as qm_o
        from qamomile.quri_parts.observable import hamiltonian_to_quri_operator

        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))
        operator = hamiltonian_to_quri_operator(H)
        ```
    """
    from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY

    pauli_id_map = {
        qm_o.Pauli.X: 1,
        qm_o.Pauli.Y: 2,
        qm_o.Pauli.Z: 3,
    }

    terms: dict = {}

    for operators, coeff in hamiltonian.terms.items():
        pauli_indices = []
        for op in operators:
            if op.pauli == qm_o.Pauli.I:
                continue
            pauli_indices.append((op.index, pauli_id_map[op.pauli]))

        if pauli_indices:
            label = pauli_label(pauli_indices)
            terms[label] = terms.get(label, 0) + coeff
        else:
            terms[PAULI_IDENTITY] = terms.get(PAULI_IDENTITY, 0) + coeff

    if abs(hamiltonian.constant) > 1e-15:
        terms[PAULI_IDENTITY] = terms.get(PAULI_IDENTITY, 0) + hamiltonian.constant

    return Operator(terms)


# Convenience alias
to_quri_operator = hamiltonian_to_quri_operator
