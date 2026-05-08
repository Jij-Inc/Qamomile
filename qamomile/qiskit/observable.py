"""Qiskit observable support.

This module provides conversion from qamomile.observable.Hamiltonian
to Qiskit SparsePauliOp for use with Qiskit Estimator primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import qamomile.observable as qm_o

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp


def hamiltonian_to_sparse_pauli_op(hamiltonian: qm_o.Hamiltonian) -> "SparsePauliOp":
    """Convert qamomile.observable.Hamiltonian to Qiskit SparsePauliOp.

    Args:
        hamiltonian: The qamomile.observable.Hamiltonian to convert

    Returns:
        Qiskit SparsePauliOp representation

    Example:
        ```python
        import qamomile.observable as qm_o
        from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

        # Build Hamiltonian
        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))

        # Convert to Qiskit
        sparse_pauli_op = hamiltonian_to_sparse_pauli_op(H)
        ```
    """
    from qiskit.quantum_info import SparsePauliOp

    n_qubits = hamiltonian.num_qubits
    if n_qubits == 0:
        # Handle empty Hamiltonian - just the constant term
        return SparsePauliOp.from_list([("I", hamiltonian.constant)])

    pauli_list = []

    # Add constant term if non-zero
    if abs(hamiltonian.constant) > 1e-15:
        pauli_list.append(("I" * n_qubits, hamiltonian.constant))

    # Pauli character mapping
    pauli_char_map = {
        qm_o.Pauli.I: "I",
        qm_o.Pauli.X: "X",
        qm_o.Pauli.Y: "Y",
        qm_o.Pauli.Z: "Z",
    }

    # Convert each term
    for operators, coeff in hamiltonian.terms.items():
        # Build Pauli string
        paulis = ["I"] * n_qubits
        for op in operators:
            paulis[op.index] = pauli_char_map[op.pauli]

        # Qiskit uses little-endian: rightmost character is qubit 0
        # So we need to reverse the order
        pauli_str = "".join(reversed(paulis))
        pauli_list.append((pauli_str, coeff))  # type: ignore[arg-type]

    if not pauli_list:
        # Empty Hamiltonian (zero operator)
        pauli_list = [("I" * max(1, n_qubits), 0.0)]

    return SparsePauliOp.from_list(pauli_list)
