"""FQAOA (Fermionic QAOA) circuit building blocks.

This module provides the quantum circuit components for the Fermionic Quantum
Approximate Optimization Algorithm (FQAOA), including Givens rotations for
initial state preparation, hopping gates for the fermionic mixer, and cost
layer construction.

All functions are decorated with ``@qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.
"""

import numpy as np

from qamomile.circuit.frontend.constructors import qubit_array, uint
from qamomile.circuit.frontend.handle import (
    Dict,
    Float,
    Matrix,
    Qubit,
    Tuple,
    UInt,
    Vector,
)
from qamomile.circuit.frontend.operation.control_flow import items, range
from qamomile.circuit.frontend.operation.controlled import controlled
from qamomile.circuit.frontend.operation.qubit_gates import cx, rx, ry, rz, rzz, x
from qamomile.circuit.frontend.qkernel import qkernel


@qkernel
def _ry_gate(q: Qubit, theta: Float) -> Qubit:
    return ry(q, theta)


_controlled_ry = controlled(_ry_gate)


@qkernel
def initial_occupations(
    q: Vector[Qubit],
    num_fermions: UInt,
) -> Vector[Qubit]:
    """Apply X gates to the first ``num_fermions`` qubits."""
    for i in range(num_fermions):
        q[i] = x(q[i])
    return q


@qkernel
def givens_rotation(
    q: Vector[Qubit],
    i: UInt,
    j: UInt,
    theta: Float,
) -> Vector[Qubit]:
    """Apply a single Givens rotation between qubits *i* and *j*."""
    q[j], q[i] = cx(q[j], q[i])
    q[i], q[j] = _controlled_ry(q[i], q[j], theta=-2.0 * theta)
    q[j], q[i] = cx(q[j], q[i])
    return q


@qkernel
def givens_rotations(
    q: Vector[Qubit],
    givens_ij: Matrix[UInt],
    givens_theta: Vector[Float],
) -> Vector[Qubit]:
    """Apply a sequence of Givens rotations.

    Args:
        q: Qubit register.
        givens_ij: Matrix of shape ``(N, 2)`` where each row ``[i, j]``
            contains the qubit indices for one Givens rotation.
        givens_theta: Vector of length ``N`` with the rotation angles.
    """
    n_rotations = givens_ij.shape[0]
    for k in range(n_rotations):
        q = givens_rotation(q, givens_ij[k, 0], givens_ij[k, 1], givens_theta[k])
    return q


@qkernel
def hopping_gate(
    q: Vector[Qubit],
    i: UInt,
    j: UInt,
    beta: Float,
    hopping: Float,
) -> Vector[Qubit]:
    """Apply the fermionic hopping gate between qubits *i* and *j*."""
    q[i] = rx(q[i], -0.5 * np.pi)
    q[j] = rx(q[j], 0.5 * np.pi)
    q[i], q[j] = cx(q[i], q[j])
    q[i] = rx(q[i], -1.0 * beta * hopping)
    q[j] = rz(q[j], beta * hopping)
    q[i], q[j] = cx(q[i], q[j])
    q[i] = rx(q[i], 0.5 * np.pi)
    q[j] = rx(q[j], -0.5 * np.pi)
    return q


@qkernel
def mixer_layer(
    q: Vector[Qubit],
    beta: Float,
    hopping: Float,
    num_qubits: UInt,
) -> Vector[Qubit]:
    """Apply the fermionic mixer layer (even-odd-boundary hopping)."""
    last_qubit = num_qubits - 1
    for i in range(0, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    for i in range(1, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    q = hopping_gate(q, uint(0), last_qubit, beta, hopping)
    return q


@qkernel
def cost_layer(
    q: Vector[Qubit],
    gamma: Float,
    linear: Dict[UInt, Float],
    quad: Dict[Tuple[UInt, UInt], Float],
) -> Vector[Qubit]:
    """Apply the cost (phase separation) layer."""
    for i, hi in items(linear):
        q[i] = rz(q[i], 2 * hi * gamma)

    for (i, j), Jij in items(quad):
        q[i], q[j] = rzz(q[i], q[j], 2 * Jij * gamma)

    return q


@qkernel
def fqaoa_layers(
    q: Vector[Qubit],
    betas: Vector[Float],
    gammas: Vector[Float],
    p: UInt,
    linear: Dict[UInt, Float],
    quad: Dict[Tuple[UInt, UInt], Float],
    hopping: Float,
    num_qubits: UInt,
) -> Vector[Qubit]:
    """Apply *p* layers of cost + mixer."""
    for layer in range(p):
        q = cost_layer(q, gammas[layer], linear, quad)
        q = mixer_layer(q, betas[layer], hopping, num_qubits)
    return q


@qkernel
def fqaoa_state(
    p: UInt,
    linear: Dict[UInt, Float],
    quad: Dict[Tuple[UInt, UInt], Float],
    num_qubits: UInt,
    num_fermions: UInt,
    givens_ij: Matrix[UInt],
    givens_theta: Vector[Float],
    hopping: Float,
    gammas: Vector[Float],
    betas: Vector[Float],
) -> Vector[Qubit]:
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
    q = qubit_array(num_qubits, name="q")
    q = initial_occupations(q, num_fermions)
    q = givens_rotations(q, givens_ij, givens_theta)
    q = fqaoa_layers(q, betas, gammas, p, linear, quad, hopping, num_qubits)
    return q
