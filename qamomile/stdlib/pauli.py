# ruff: noqa: F821 -- ``width`` is closed into a concrete dependent shape.

"""Exact block encodings for backend-neutral Pauli Hamiltonians."""

from __future__ import annotations

import qamomile as qm
from qamomile.observable import Hamiltonian, Pauli, PauliFactor, PauliWord

from ._structure import compose_unary, identity
from .block_encoding import BlockEncoding
from .lcu import lcu_block_encoding


def pauli_block_encoding(operator: Hamiltonian) -> BlockEncoding:
    """Construct an exact LCU block encoding of a Pauli Hamiltonian.

    The observable algebra already canonicalizes identity and Pauli-word
    coefficients. This adapter turns each word into an ordinary unary
    ``QKernel`` and delegates all PREPARE/SELECT behavior to
    :func:`lcu_block_encoding`. It introduces no producer descriptor, QCore
    operation, resource rule, or backend lowering path.

    Args:
        operator: Canonical backend-neutral Pauli sum over at least one qubit.

    Returns:
        Construction-independent exact block-encoding contract.

    Raises:
        TypeError: If ``operator`` is not a :class:`Hamiltonian`.
        ValueError: If the operator has no system qubit or a coefficient is
            nonfinite.
    """

    if not isinstance(operator, Hamiltonian):
        raise TypeError("operator must be a Hamiltonian")
    if operator.num_qubits <= 0:
        raise ValueError("pauli block encoding requires at least one system qubit")
    terms: list[tuple[complex, qm.QKernel]] = []
    if operator.constant:
        terms.append((operator.constant, identity))
    terms.extend(
        (coefficient, _pauli_word(word, operator.num_qubits))
        for word, coefficient in operator.terms
    )
    return lcu_block_encoding(terms, system_width=operator.num_qubits)


def _pauli_word(word: PauliWord, width: int) -> qm.QKernel:
    """Close one canonical Pauli word into an ordinary unary function."""

    steps = tuple(_pauli_factor(factor, width) for factor in word.factors)
    return compose_unary(width=width, steps=steps)


def _pauli_factor(factor: PauliFactor, width: int) -> qm.QKernel:
    """Construct one fixed-target Pauli gate function."""

    target = factor.index
    if factor.pauli is Pauli.X:

        @qm.qkernel
        def apply_x(
            state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Apply Pauli X to the configured component.

            Args:
                state (qm.Q[qm.Bits[width]]): Owned state to transform.

            Returns:
                qm.Q[qm.Bits[width]]: State after the Pauli X operation.
            """

            return qm.x(state, target=target)

        return apply_x
    if factor.pauli is Pauli.Y:

        @qm.qkernel
        def apply_y(
            state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Apply Pauli Y to the configured component.

            Args:
                state (qm.Q[qm.Bits[width]]): Owned state to transform.

            Returns:
                qm.Q[qm.Bits[width]]: State after the Pauli Y operation.
            """

            return qm.y(state, target=target)

        return apply_y

    @qm.qkernel
    def apply_z(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply Pauli Z to the configured component.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned state to transform.

        Returns:
            qm.Q[qm.Bits[width]]: State after the Pauli Z operation.
        """

        return qm.z(state, target=target)

    return apply_z


__all__ = ["pauli_block_encoding"]
