"""FQAOA (Fermionic QAOA) circuit building blocks.

This module provides the quantum circuit components for the Fermionic Quantum
Approximate Optimization Algorithm (FQAOA), including Givens rotations for
initial state preparation, hopping gates for the fermionic mixer, and cost
layer construction.

All functions are decorated with ``@qm_c.qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.
"""

import numpy as np

import qamomile.circuit as qmc


@qmc.qkernel
def _ry_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    return qmc.ry(q, theta)


_controlled_ry = qmc.controlled(_ry_gate)


@qmc.qkernel
def initial_occupations(
    q: qmc.Vector[qmc.Qubit],
    num_fermions: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply X gates to the first ``num_fermions`` qubits."""
    for i in qmc.range(num_fermions):
        q[i] = qmc.x(q[i])
    return q


@qmc.qkernel
def givens_rotation(
    q: qmc.Vector[qmc.Qubit],
    i: qmc.UInt,
    j: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a single Givens rotation between qubits *i* and *j*."""
    q[j], q[i] = qmc.cx(q[j], q[i])
    q[i], q[j] = _controlled_ry(q[i], q[j], theta=-2.0 * theta)
    q[j], q[i] = qmc.cx(q[j], q[i])
    return q


@qmc.qkernel
def givens_rotations(
    q: qmc.Vector[qmc.Qubit],
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Apply a sequence of Givens rotations.

    Args:
        q: Qubit register.
        givens_ij: Matrix of shape ``(N, 2)`` where each row ``[i, j]``
            contains the qubit indices for one Givens rotation.
        givens_theta: Vector of length ``N`` with the rotation angles.
    """
    n_rotations = givens_ij.shape[0]
    for k in qmc.range(n_rotations):
        q = givens_rotation(q, givens_ij[k, 0], givens_ij[k, 1], givens_theta[k])
    return q


@qmc.qkernel
def hopping_gate(
    q: qmc.Vector[qmc.Qubit],
    i: qmc.UInt,
    j: qmc.UInt,
    beta: qmc.Float,
    hopping: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the fermionic hopping gate between qubits *i* and *j*."""
    q[i] = qmc.rx(q[i], -0.5 * np.pi)
    q[j] = qmc.rx(q[j], 0.5 * np.pi)
    q[i], q[j] = qmc.cx(q[i], q[j])
    q[i] = qmc.rx(q[i], -1.0 * beta * hopping)
    q[j] = qmc.rz(q[j], beta * hopping)
    q[i], q[j] = qmc.cx(q[i], q[j])
    q[i] = qmc.rx(q[i], 0.5 * np.pi)
    q[j] = qmc.rx(q[j], -0.5 * np.pi)
    return q


@qmc.qkernel
def mixer_layer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
    hopping: qmc.Float,
    num_qubits: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the fermionic mixer layer (even-odd-boundary hopping)."""
    last_qubit = num_qubits - 1
    for i in qmc.range(0, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    for i in qmc.range(1, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    q = hopping_gate(q, qmc.uint(0), last_qubit, beta, hopping)
    return q


@qmc.qkernel
def cost_layer(
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the cost (phase separation) layer."""
    for i, hi in qmc.items(linear):
        q[i] = qmc.rz(q[i], 2 * hi * gamma)

    for (i, j), Jij in qmc.items(quad):
        q[i], q[j] = qmc.rzz(q[i], q[j], 2 * Jij * gamma)

    return q


@qmc.qkernel
def fqaoa_layers(
    q: qmc.Vector[qmc.Qubit],
    betas: qmc.Vector[qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    p: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    hopping: qmc.Float,
    num_qubits: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply *p* layers of cost + mixer."""
    for layer in qmc.range(p):
        q = cost_layer(q, gammas[layer], linear, quad)
        q = mixer_layer(q, betas[layer], hopping, num_qubits)
    return q


@qmc.qkernel
def fqaoa_state(
    p: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    num_qubits: qmc.UInt,
    num_fermions: qmc.UInt,
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate complete FQAOA state.

    Args:
        p: Number of FQAOA layers.
        linear: Linear coefficients of Ising model.
        quad: Quadratic coefficients of Ising model.
        num_qubits: Number of qubits.
        num_fermions: Number of fermions for initial state.
        givens_ij: Matrix of shape ``(N, 2)`` with qubit index pairs for
            Givens rotations.
        givens_theta: Vector of length ``N`` with Givens rotation angles.
        hopping: Hopping integral for the mixer.
        gammas: Vector of gamma parameters.
        betas: Vector of beta parameters.

    Returns:
        FQAOA state vector.
    """
    q = qmc.qubit_array(num_qubits, name="q")
    q = initial_occupations(q, num_fermions)
    q = givens_rotations(q, givens_ij, givens_theta)
    q = fqaoa_layers(q, betas, gammas, p, linear, quad, hopping, num_qubits)
    return q
