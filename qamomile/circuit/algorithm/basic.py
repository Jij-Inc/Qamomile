"""Basic building blocks for variational quantum circuits.

This module provides fundamental rotation layers and entanglement layers
that can be composed to build variational ansatze.
"""

from __future__ import annotations

from qamomile.circuit.frontend.handle import Float, Qubit, UInt, Vector
from qamomile.circuit.frontend.operation.control_flow import range
from qamomile.circuit.frontend.operation.qubit_gates import cz, rx, ry, rz
from qamomile.circuit.frontend.qkernel import qkernel


@qkernel
def rx_layer(
    q: Vector[Qubit],
    thetas: Vector[Float],
    offset: UInt,
) -> Vector[Qubit]:
    """Apply RX rotation to each qubit.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas (consumes q.shape[0] parameters)

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in range(n):
        q[i] = rx(q[i], thetas[offset + i])
    return q


@qkernel
def ry_layer(
    q: Vector[Qubit],
    thetas: Vector[Float],
    offset: UInt,
) -> Vector[Qubit]:
    """Apply RY rotation to each qubit.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas (consumes q.shape[0] parameters)

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in range(n):
        q[i] = ry(q[i], thetas[offset + i])
    return q


@qkernel
def rz_layer(
    q: Vector[Qubit],
    thetas: Vector[Float],
    offset: UInt,
) -> Vector[Qubit]:
    """Apply RZ rotation to each qubit.

    Args:
        q: Qubit vector
        thetas: Parameter vector
        offset: Starting index in thetas (consumes q.shape[0] parameters)

    Returns:
        Qubit vector after rotations
    """
    n = q.shape[0]
    for i in range(n):
        q[i] = rz(q[i], thetas[offset + i])
    return q


@qkernel
def cz_entangling_layer(
    q: Vector[Qubit],
) -> Vector[Qubit]:
    """Apply CZ entangling layer with linear connectivity.

    Applies CZ gates between consecutive qubits: (0,1), (1,2), ..., (n-2,n-1).

    Args:
        q: Qubit vector

    Returns:
        Qubit vector after entanglement
    """
    n = q.shape[0]
    for i in range(n - 1):
        q[i], q[i + 1] = cz(q[i], q[i + 1])
    return q
