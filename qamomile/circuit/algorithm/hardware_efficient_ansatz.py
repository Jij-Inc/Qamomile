"""Hardware-efficient ansatz for variational quantum algorithms.

This module provides hardware-efficient ansatz circuits that can be used
for VQE, QRAO, and other variational algorithms.
"""

from __future__ import annotations

import qamomile.circuit as qmc


@qmc.qkernel
def ry_rz_layer(
    q: qmc.Vector[qmc.Qubit],
    thetas: qmc.Vector[qmc.Float],
    offset: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply RY-RZ rotation layer to all qubits.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], thetas[offset + 2 * i])
        q[i] = qmc.rz(q[i], thetas[offset + 2 * i + 1])
    return q


@qmc.qkernel
def cz_entangling_layer(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply CZ entangling layer with linear connectivity.

    Args:
        q: Qubit vector

    Returns:
        Qubit vector after entanglement
    """
    n = q.shape[0]
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
    return q


@qmc.qkernel
def hardware_efficient_ansatz(
    q: qmc.Vector[qmc.Qubit],
    depth: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Apply hardware-efficient ansatz with RY-RZ rotations and CZ entanglement.

    Args:
        q: Input qubit vector (should be in |+>^n state for good performance)
        depth: Number of variational layers
        thetas: Variational parameters (2 * n * depth total)

    Returns:
        Qubit vector after applying the ansatz
    """
    n = q.shape[0]
    for layer in qmc.range(depth):
        q = ry_rz_layer(q, thetas, 2 * n * layer)
        q = cz_entangling_layer(q)
    return q


def num_parameters(n: int, depth: int) -> int:
    """Calculate total number of parameters for hardware-efficient ansatz.

    Args:
        n: Number of qubits
        depth: Number of variational layers

    Returns:
        Total number of parameters (2 * n * depth)
    """
    return 2 * n * depth
