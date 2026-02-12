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


def _apply_initial_occupations(
    q: qm_c.Vector[qm_c.Qubit],
    num_fermions: int,
) -> qm_c.Vector[qm_c.Qubit]:
    for i in range(num_fermions):
        q[i] = qm_c.x(q[i])
    return q


def _apply_givens_rotation(
    q: qm_c.Vector[qm_c.Qubit],
    i: int,
    j: int,
    theta: float,
) -> qm_c.Vector[qm_c.Qubit]:
    q[j], q[i] = qm_c.cx(q[j], q[i])
    q[i], q[j] = _controlled_ry(q[i], q[j], theta=-2.0 * theta)
    q[j], q[i] = qm_c.cx(q[j], q[i])
    return q


def _apply_givens_rotations(
    q: qm_c.Vector[qm_c.Qubit],
    givens_rotations: list[tuple[tuple[int, int], float]] | list[list],
) -> qm_c.Vector[qm_c.Qubit]:
    for (i, j), theta in givens_rotations:
        q = _apply_givens_rotation(q, i, j, theta)
    return q


def _apply_hopping_gate(
    q: qm_c.Vector[qm_c.Qubit],
    i: int,
    j: int,
    beta: qm_c.Float,
    hopping: float,
) -> qm_c.Vector[qm_c.Qubit]:
    q[i] = qm_c.rx(q[i], -0.5 * np.pi)
    q[j] = qm_c.rx(q[j], 0.5 * np.pi)
    q[i], q[j] = qm_c.cx(q[i], q[j])
    q[i] = qm_c.rx(q[i], -1.0 * beta * hopping)
    q[j] = qm_c.rz(q[j], beta * hopping)
    q[i], q[j] = qm_c.cx(q[i], q[j])
    q[i] = qm_c.rx(q[i], 0.5 * np.pi)
    q[j] = qm_c.rx(q[j], -0.5 * np.pi)
    return q


def _apply_mixer_layer(
    q: qm_c.Vector[qm_c.Qubit],
    beta: qm_c.Float,
    hopping: float,
    num_qubits: int,
) -> qm_c.Vector[qm_c.Qubit]:
    for i in range(0, num_qubits - 1, 2):
        q = _apply_hopping_gate(q, i, i + 1, beta, hopping)
    for i in range(1, num_qubits - 1, 2):
        q = _apply_hopping_gate(q, i, i + 1, beta, hopping)
    q = _apply_hopping_gate(q, 0, num_qubits - 1, beta, hopping)
    return q


def _apply_cost_layer(
    q: qm_c.Vector[qm_c.Qubit],
    gamma: qm_c.Float,
    linear: dict[int, float],
    quad: dict[tuple[int, int], float],
) -> qm_c.Vector[qm_c.Qubit]:
    for i, hi in linear.items():
        q[i] = qm_c.rz(q[i], 2 * hi * gamma)

    for (i, j), Jij in quad.items():
        q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gamma)

    return q


def _apply_fqaoa_layers(
    q: qm_c.Vector[qm_c.Qubit],
    betas: qm_c.Vector[qm_c.Float],
    gammas: qm_c.Vector[qm_c.Float],
    p: int,
    linear: dict[int, float],
    quad: dict[tuple[int, int], float],
    hopping: float,
    num_qubits: int,
) -> qm_c.Vector[qm_c.Qubit]:
    for layer in range(p):
        q = _apply_cost_layer(q, gammas[layer], linear, quad)
        q = _apply_mixer_layer(q, betas[layer], hopping, num_qubits)
    return q
