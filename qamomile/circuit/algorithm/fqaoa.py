"""FQAOA (Fermionic QAOA) circuit building blocks.

This module provides the quantum circuit components for the Fermionic Quantum
Approximate Optimization Algorithm (FQAOA), including Givens rotations for
initial state preparation, hopping gates for the fermionic mixer, and cost
layer construction.
"""

import numpy as np

import qamomile.circuit as qm_c


@qm_c.qkernel
def _ry_gate(q: qm_c.Qubit, theta: qm_c.Float) -> qm_c.Qubit:
    return qm_c.ry(q, theta)


_controlled_ry = qm_c.controlled(_ry_gate)


def initial_occupations(
    q: qm_c.Vector[qm_c.Qubit],
    num_fermions: int,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply X gates to the first ``num_fermions`` qubits."""
    for i in range(num_fermions):
        q[i] = qm_c.x(q[i])
    return q


def givens_rotation(
    q: qm_c.Vector[qm_c.Qubit],
    i: int,
    j: int,
    theta: float,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply a single Givens rotation between qubits *i* and *j*."""
    q[j], q[i] = qm_c.cx(q[j], q[i])
    q[i], q[j] = _controlled_ry(q[i], q[j], theta=-2.0 * theta)
    q[j], q[i] = qm_c.cx(q[j], q[i])
    return q


def givens_rotations(
    q: qm_c.Vector[qm_c.Qubit],
    rotations: list[tuple[tuple[int, int], float]] | list[list],
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply a sequence of Givens rotations."""
    for (i, j), theta in rotations:
        q = givens_rotation(q, i, j, theta)
    return q


def hopping_gate(
    q: qm_c.Vector[qm_c.Qubit],
    i: int,
    j: int,
    beta: qm_c.Float,
    hopping: float,
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


def mixer_layer(
    q: qm_c.Vector[qm_c.Qubit],
    beta: qm_c.Float,
    hopping: float,
    num_qubits: int,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply the fermionic mixer layer (even-odd-boundary hopping)."""
    for i in range(0, num_qubits - 1, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    for i in range(1, num_qubits - 1, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    q = hopping_gate(q, 0, num_qubits - 1, beta, hopping)
    return q


def cost_layer(
    q: qm_c.Vector[qm_c.Qubit],
    gamma: qm_c.Float,
    linear: dict[int, float],
    quad: dict[tuple[int, int], float],
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply the cost (phase separation) layer.

    For the transpilable (Dict-typed) version, use
    :func:`qamomile.circuit.algorithm.qaoa.ising_cost_circuit`.
    """
    for i, hi in linear.items():
        q[i] = qm_c.rz(q[i], 2 * hi * gamma)

    for (i, j), Jij in quad.items():
        q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gamma)

    return q


def fqaoa_layers(
    q: qm_c.Vector[qm_c.Qubit],
    betas: qm_c.Vector[qm_c.Float],
    gammas: qm_c.Vector[qm_c.Float],
    p: int,
    linear: dict[int, float],
    quad: dict[tuple[int, int], float],
    hopping: float,
    num_qubits: int,
) -> qm_c.Vector[qm_c.Qubit]:
    """Apply *p* layers of cost + mixer."""
    for layer in range(p):
        q = cost_layer(q, gammas[layer], linear, quad)
        q = mixer_layer(q, betas[layer], hopping, num_qubits)
    return q
