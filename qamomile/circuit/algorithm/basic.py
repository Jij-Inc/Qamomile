"""Basic building blocks for variational quantum circuits.

This module provides fundamental rotation layers and entanglement layers
that can be composed to build variational ansatze.
"""

from __future__ import annotations

import qamomile.circuit as qmc


@qmc.qkernel
def superposition_vector(
    n: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Create a uniform superposition state by applying Hadamard to all qubits.

    Args:
        n (qmc.UInt): Number of qubits.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register in the |+>^n state.
    """
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


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


@qmc.qkernel
def cx_entangling_layer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply CX entangling layer with linear connectivity.

    Applies CX gates between consecutive qubits: (0,1), (1,2), ..., (n-2,n-1).

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit vector

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit vector after entanglement
    """
    n = q.shape[0]
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return q


@qmc.qkernel
def phase_gadget(
    q: qmc.Vector[qmc.Qubit],
    indices: qmc.Vector[qmc.UInt],
    angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply exp(-i * angle/2 * Z_{i0} Z_{i1} ... Z_{ik-1}).

    Decomposes a k-body Z-rotation into CX + RZ primitives.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        indices (qmc.Vector[qmc.UInt]): Qubit indices for the interaction term.
            Must be non-empty.
        angle (qmc.Float): Rotation angle in radians.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    # Preconditions: indices must be non-empty.
    k = indices.shape[0]
    last = k - 1
    # CX forward ladder
    for step in qmc.range(last):
        next_step = step + 1
        left = indices[step]
        right = indices[next_step]
        q[left], q[right] = qmc.cx(q[left], q[right])

    # Apply RZ on the last qubit
    q[indices[last]] = qmc.rz(q[indices[last]], angle=angle)

    # CX reverse ladder (using positive-step range for qkernel compatibility)
    for step in qmc.range(last):
        rev = k - step - 2
        rev_next = rev + 1
        left = indices[rev]
        right = indices[rev_next]
        q[left], q[right] = qmc.cx(q[left], q[right])
    return q
