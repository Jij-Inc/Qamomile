"""Modular increment and decrement operators for basis-state registers."""

from __future__ import annotations

import qamomile.circuit as qmc


@qmc.qkernel
def modular_increment(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular increment ``|j> -> |j + 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to increment in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    for k in qmc.range(1, n):
        target_index = n - k
        mcx = qmc.control(qmc.x, num_controls=target_index)
        controls = q[0:target_index]
        target = q[target_index]
        controls, target = mcx(controls, target)
        q[0:target_index] = controls
        q[target_index] = target
    q[0] = qmc.x(q[0])
    return q


@qmc.qkernel
def modular_decrement(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular decrement ``|j> -> |j - 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to decrement in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    q[0] = qmc.x(q[0])
    for target_index in qmc.range(1, n):
        mcx = qmc.control(qmc.x, num_controls=target_index)
        controls = q[0:target_index]
        target = q[target_index]
        controls, target = mcx(controls, target)
        q[0:target_index] = controls
        q[target_index] = target
    return q


__all__ = [
    "modular_decrement",
    "modular_increment",
]
