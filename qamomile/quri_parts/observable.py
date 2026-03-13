"""QURI Parts observable support.

This module provides conversion from qamomile.observable.Hamiltonian
to QURI Parts Operator for use with QURI Parts estimator primitives.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

# qamomile.observable is a core module, always available (not optional).
import qamomile.observable as qm_o

if TYPE_CHECKING:
    from quri_parts.core.operator import Operator

# Threshold for treating a coefficient as zero.
# Matches the tolerance used in qamomile.qiskit.observable for consistency.
_ZERO_THRESHOLD = 1e-15


def hamiltonian_to_quri_operator(hamiltonian: qm_o.Hamiltonian) -> "Operator":
    """Convert qamomile.observable.Hamiltonian to QURI Parts Operator.

    Args:
        hamiltonian (qm_o.Hamiltonian): The qamomile Hamiltonian to convert.

    Returns:
        Operator: QURI Parts Operator representation.

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

    terms: dict[Any, complex] = {}

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

    if abs(hamiltonian.constant) > _ZERO_THRESHOLD:
        terms[PAULI_IDENTITY] = terms.get(PAULI_IDENTITY, 0) + hamiltonian.constant

    return Operator(terms)


def to_quri_operator(hamiltonian: qm_o.Hamiltonian) -> "Operator":
    """Convert Hamiltonian to QURI Parts Operator.

    Convenience alias for :func:`hamiltonian_to_quri_operator`.

    Args:
        hamiltonian (qm_o.Hamiltonian): The qamomile Hamiltonian to convert.

    Returns:
        Operator: QURI Parts Operator representation.
    """
    return hamiltonian_to_quri_operator(hamiltonian)
