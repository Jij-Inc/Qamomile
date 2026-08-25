r"""Compose ordered products of exact block encodings.

This module is the multiplicative counterpart of :mod:`.lcu`: where
:func:`~qamomile.circuit.lcu_block_encoding` encodes a weighted *sum* of child
operators, :func:`product_block_encoding` encodes their ordered *product*.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel

from .lcu import LCUBlockEncoding

__all__ = [
    "product_block_encoding",
]


def product_block_encoding(
    encodings: Sequence[LCUBlockEncoding],
) -> LCUBlockEncoding:
    r"""Compose an ordered product of exact block encodings.

    Given children ``U_j`` satisfying
    :math:`V_{0,j}^\dagger U_j V_{0,j} = A_j / \alpha_j` on a common system
    register, this factory encodes the ordered product

    .. math::

        A = A_{M-1} \cdots A_0,
        \qquad
        \alpha = \prod_j \alpha_j,

    by applying the children in sequence on a shared system register, each on
    its own disjoint slice of one flat signal register. Because child ``j``
    acts as identity on every other child's ancillas, projecting the whole
    signal register onto zero picks out exactly
    :math:`\prod_j A_j / \alpha_j` with no cross terms. This holds for any
    ordered product; the children need not commute.

    The returned descriptor may itself be used as a child of this factory or
    of :func:`~qamomile.circuit.lcu_block_encoding`.

    Args:
        encodings (Sequence[LCUBlockEncoding]): Nonempty ordered children,
            applied left to right, so ``encodings[0]`` acts first and is the
            rightmost matrix factor. Every child must report the same
            ``num_system_qubits``.

    Returns:
        LCUBlockEncoding: Composable descriptor whose ``num_signal_qubits`` is
            the sum of the children's and whose ``normalization`` is their
            product.

    Raises:
        TypeError: If ``encodings`` is not an ordered sequence or contains a
            value that is not an :class:`LCUBlockEncoding`.
        ValueError: If ``encodings`` is empty, child system widths differ, or
            the composed normalization overflows or is non-finite.

    Example:
        >>> import qamomile.circuit as qmc
        >>> first = qmc.ising_z_block_encoding({(0,): 1.0}, 1)
        >>> second = qmc.identity_block_encoding(1)
        >>> qmc.product_block_encoding([first, second]).num_signal_qubits
        2
    """
    if not isinstance(encodings, Sequence):
        raise TypeError("encodings must be an ordered sequence.")
    children = tuple(encodings)
    if not children:
        raise ValueError("encodings must be nonempty; the system width is unknowable.")
    # A str is a Sequence, so its characters land here and are rejected as
    # non-descriptors; no separate str/bytes guard is needed.
    for child in children:
        if not isinstance(child, LCUBlockEncoding):
            raise TypeError("each entry of encodings must be an LCUBlockEncoding.")

    system_width = children[0].num_system_qubits
    if any(child.num_system_qubits != system_width for child in children):
        raise ValueError("every child block encoding must use one system width.")

    normalization = 1.0
    for child in children:
        normalization *= child.normalization
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("composed normalization must be finite and positive.")

    widths = tuple(child.num_signal_qubits for child in children)
    offsets = tuple(sum(widths[:index]) for index in range(len(widths)))
    signal_width = sum(widths)

    def emit_chain(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Emit every child on its own signal slice, in order.

        Args:
            signal (Vector[Qubit]): Flat concatenated child signal register.
            system (Vector[Qubit]): Shared ordered system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and system
                registers.
        """
        # NOTE: kept out of the qkernel body on purpose. The qkernel AST
        # transformer rejects direct iteration over a sequence, and this loop
        # cannot become qmc.range(...) anyway: every iteration calls a
        # different child kernel, so the unrolling has to happen on the host.
        # The transformer does not descend into called helpers, so the loop
        # runs at trace time with the real handles.
        for child, offset, width in zip(children, offsets, widths):
            stop = offset + width
            signal[offset:stop], system = child.unitary(signal[offset:stop], system)
        return signal, system

    @qkernel
    def unitary(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the ordered product of the child block encodings.

        Args:
            signal (Vector[Qubit]): Flat concatenated child signal register.
            system (Vector[Qubit]): Shared ordered system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and system
                registers.
        """
        return emit_chain(signal, system)

    return LCUBlockEncoding(
        unitary=unitary,
        normalization=normalization,
        num_signal_qubits=signal_width,
        num_system_qubits=system_width,
    )
