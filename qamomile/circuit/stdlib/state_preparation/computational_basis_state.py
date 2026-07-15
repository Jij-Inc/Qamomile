from __future__ import annotations

import math
from typing import cast

import qamomile.circuit as qmc


@qmc.qkernel
def computational_basis_state(
    q: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare the computational basis state labeled by ``bits``.

    Applies ``Rx(pi * bits[i])`` to ``q[i]``: identity when ``bits[i] == 0``
    and ``X`` (up to a ``-i`` global phase) when ``bits[i] == 1``.

    Assumes ``q`` starts in :math:`\\lvert 0 \\rangle^{\\otimes n}` and
    ``q.shape[0] == bits.shape[0]``.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register, expected to start in |0>^n.
        bits (qmc.Vector[qmc.UInt]): Classical bit register specifying the target state.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register prepared in the |bits> state
        (up to a global phase).
    """
    for i in qmc.range(q.shape[0]):
        angle = cast(qmc.Float, math.pi * bits[i])
        q[i] = qmc.rx(q[i], angle)
    return q
