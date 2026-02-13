"""FQAOA (Fermionic QAOA) circuit building blocks.

This module provides the quantum circuit components for the Fermionic Quantum
Approximate Optimization Algorithm (FQAOA), including Givens rotations for
initial state preparation, hopping gates for the fermionic mixer, and cost
layer construction.

All functions are decorated with ``@qm_c.qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.
"""

import numpy as np

import qamomile.circuit as qm_c


@qm_c.qkernel
def _ry_gate(q: qm_c.Qubit, theta: qm_c.Float) -> qm_c.Qubit:
    return qm_c.ry(q, theta)


_controlled_ry = qm_c.controlled(_ry_gate)


@qm_c.qkernel
def initial_occupations(
    q: qm_c.Vector[qm_c.Qubit],
    num_fermions: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply X gates to the first ``num_fermions`` qubits."""
    for i in qm_c.range(num_fermions):
        q[i] = qm_c.x(q[i])
    return q


@qm_c.qkernel
def givens_rotation(
    q: qm_c.Vector[qm_c.Qubit],
    i: qm_c.UInt,
    j: qm_c.UInt,
    theta: qm_c.Float,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply a single Givens rotation between qubits *i* and *j*."""
    q[j], q[i] = qm_c.cx(q[j], q[i])
    q[i], q[j] = _controlled_ry(q[i], q[j], theta=-2.0 * theta)
    q[j], q[i] = qm_c.cx(q[j], q[i])
    return q


@qm_c.qkernel
def givens_rotations(
    q: qm_c.Vector[qm_c.Qubit],
    givens_ij: qm_c.Matrix[qm_c.UInt],
    givens_theta: qm_c.Vector[qm_c.Float],
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply a sequence of Givens rotations.

    Args:
        q: Qubit register.
        givens_ij: Matrix of shape ``(N, 2)`` where each row ``[i, j]``
            contains the qubit indices for one Givens rotation.
        givens_theta: Vector of length ``N`` with the rotation angles.
    """
    n_rotations = givens_ij.shape[0]
    for k in qm_c.range(n_rotations):
        q = givens_rotation(q, givens_ij[k, 0], givens_ij[k, 1], givens_theta[k])
    return q


@qm_c.qkernel
def hopping_gate(
    q: qm_c.Vector[qm_c.Qubit],
    i: qm_c.UInt,
    j: qm_c.UInt,
    beta: qm_c.Float,
    hopping: qm_c.Float,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply the fermionic hopping gate between qubits *i* and *j*."""
    q[i] = qm_c.rx(q[i], -0.5 * np.pi)
    q[j] = qm_c.rx(q[j], 0.5 * np.pi)
    q[i], q[j] = qm_c.cx(q[i], q[j])
    q[i] = qm_c.rx(q[i], -1.0 * beta * hopping)
    q[j] = qm_c.rz(q[j], beta * hopping)
    q[i], q[j] = qm_c.cx(q[i], q[j])
    q[i] = qm_c.rx(q[i], 0.5 * np.pi)
    q[j] = qm_c.rx(q[j], -0.5 * np.pi)
    return q


@qm_c.qkernel
def mixer_layer(
    q: qm_c.Vector[qm_c.Qubit],
    beta: qm_c.Float,
    hopping: qm_c.Float,
    num_qubits: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply the fermionic mixer layer (even-odd-boundary hopping)."""
    last_qubit = num_qubits - 1
    for i in qm_c.range(0, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    for i in qm_c.range(1, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    q = hopping_gate(q, qm_c.uint(0), last_qubit, beta, hopping)
    return q


@qm_c.qkernel
def cost_layer(
    q: qm_c.Vector[qm_c.Qubit],
    gamma: qm_c.Float,
    linear: qm_c.Dict[qm_c.UInt, qm_c.Float],
    quad: qm_c.Dict[qm_c.Tuple[qm_c.UInt, qm_c.UInt], qm_c.Float],
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply the cost (phase separation) layer."""
    for i, hi in qm_c.items(linear):
        q[i] = qm_c.rz(q[i], 2 * hi * gamma)

    for (i, j), Jij in qm_c.items(quad):
        q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gamma)

    return q


@qm_c.qkernel
def fqaoa_layers(
    q: qm_c.Vector[qm_c.Qubit],
    betas: qm_c.Vector[qm_c.Float],
    gammas: qm_c.Vector[qm_c.Float],
    p: qm_c.UInt,
    linear: qm_c.Dict[qm_c.UInt, qm_c.Float],
    quad: qm_c.Dict[qm_c.Tuple[qm_c.UInt, qm_c.UInt], qm_c.Float],
    hopping: qm_c.Float,
    num_qubits: qm_c.UInt,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply *p* layers of cost + mixer."""
    for layer in qm_c.range(p):
        q = cost_layer(q, gammas[layer], linear, quad)
        q = mixer_layer(q, betas[layer], hopping, num_qubits)
    return q


@qm_c.qkernel
def fqaoa_state(
    p: qm_c.UInt,
    linear: qm_c.Dict[qm_c.UInt, qm_c.Float],
    quad: qm_c.Dict[qm_c.Tuple[qm_c.UInt, qm_c.UInt], qm_c.Float],
    num_qubits: qm_c.UInt,
    num_fermions: qm_c.UInt,
    givens_ij: qm_c.Matrix[qm_c.UInt],
    givens_theta: qm_c.Vector[qm_c.Float],
    hopping: qm_c.Float,
    gammas: qm_c.Vector[qm_c.Float],
    betas: qm_c.Vector[qm_c.Float],
) -> qm_c.Vector[qm_c.Qubit]:
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
    q = qm_c.qubit_array(num_qubits, name="q")
    q = initial_occupations(q, num_fermions)
    q = givens_rotations(q, givens_ij, givens_theta)
    q = fqaoa_layers(q, betas, gammas, p, linear, quad, hopping, num_qubits)
    return q
