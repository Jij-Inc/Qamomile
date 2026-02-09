"""Basic building blocks for variational quantum circuits.

This module provides fundamental rotation layers and entanglement layers
that can be composed to build variational ansatze.
"""

from __future__ import annotations

import qamomile.circuit as qmc


@qmc.qkernel
def rx_layer(
    q: qmc.Vector[qmc.Qubit],
    thetas: qmc.Vector[qmc.Float],
    offset: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply RX rotation to each qubit.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas (consumes q.shape[0] parameters)

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], thetas[offset + i])
    return q


@qmc.qkernel
def ry_layer(
    q: qmc.Vector[qmc.Qubit],
    thetas: qmc.Vector[qmc.Float],
    offset: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply RY rotation to each qubit.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas (consumes q.shape[0] parameters)

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], thetas[offset + i])
    return q


@qmc.qkernel
def rz_layer(
    q: qmc.Vector[qmc.Qubit],
    thetas: qmc.Vector[qmc.Float],
    offset: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply RZ rotation to each qubit.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas (consumes q.shape[0] parameters)

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rz(q[i], thetas[offset + i])
    return q


@qmc.qkernel
def cz_entangling_layer(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply CZ entangling layer with linear connectivity.

    Applies CZ gates between consecutive qubits: (0,1), (1,2), ..., (n-2,n-1).

    Args:
        q: Qubit vector

    Returns:
        Qubit vector after entanglement
    """
    n = q.shape[0]
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
    return q
