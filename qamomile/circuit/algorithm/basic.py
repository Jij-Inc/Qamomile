"""Basic building blocks for variational quantum circuits.

This module provides fundamental rotation layers and entanglement layers
that can be composed to build variational ansatze.
"""

from __future__ import annotations

import math

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


@qmc.qkernel
def computational_basis_state(
    n: qmc.UInt,
    q: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare the computational basis state labeled by ``bits``.

    Applies ``Rx(pi * bits[i])`` to ``q[i]``: identity when ``bits[i] == 0``
    and ``X`` (up to a ``-i`` global phase) when ``bits[i] == 1``. Equivalent
    to a conditional ``X`` for any measurement-based use case, but uses a
    parameterized rotation so the circuit can be transpiled even when
    ``bits`` is supplied as a runtime parameter (the runtime ``if`` form
    cannot be emitted because backends require ``if`` conditions to be
    compile-time constants or measurement results).

    The explicit length ``n`` keeps the circuit's qubit count fixed at
    compile time independently of ``bits``. Bind ``n`` at transpile so the
    for-loop bound is concrete even when ``bits`` is left as a runtime
    parameter; otherwise the transpile would fail with an unresolved-shape
    error on ``bits.shape[0]``.

    Assumes ``q`` starts in :math:`\\lvert 0 \\rangle^{\\otimes n}` and
    ``q.shape[0] == bits.shape[0] == n``.

    Args:
        n (qmc.UInt): Length of the registers. Bind at transpile time.
        q (qmc.Vector[qmc.Qubit]): Qubit register, expected to start in |0>^n.
        bits (qmc.Vector[qmc.UInt]): Classical bit register specifying the target state.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register prepared in the |bits> state
        (up to a global phase).
    """
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], math.pi * bits[i])
    return q
