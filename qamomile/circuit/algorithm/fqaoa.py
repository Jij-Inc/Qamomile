"""FQAOA (Fermionic QAOA) circuit building blocks.

This module provides the quantum circuit components for the Fermionic Quantum
Approximate Optimization Algorithm (FQAOA), including Givens rotations for
initial state preparation, hopping gates for the fermionic mixer, and cost
layer construction.

All functions are decorated with ``@qm_c.qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.
"""

from __future__ import annotations

import numpy as np

import qamomile.circuit as qmc

from . import basic as _basic

# ------------------------------------------------------------------
# Initial state preparation
# ------------------------------------------------------------------


@qmc.qkernel
def initial_occupations(
    q: qmc.Vector[qmc.Qubit],
    num_fermions: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply X gates to the first ``num_fermions`` qubits.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        num_fermions (qmc.UInt): Number of fermions to occupy.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register with the first
            ``num_fermions`` qubits flipped to |1>.
    """
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
    """Apply a single Givens rotation between qubits *i* and *j*.

    The controlled-RY factory is constructed inside the function rather
    than at module level so importing ``fqaoa`` does not trigger
    eager wrapper synthesis (compile/exec + tracing) for every install.
    The synthesized wrapper is cached per-callable inside
    ``qmc.controlled``, so the only real cost happens on the first
    Givens rotation; subsequent calls hit the cache.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        i (qmc.UInt): Index of the first qubit.
        j (qmc.UInt): Index of the second qubit.
        theta (qmc.Float): Givens rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    controlled_ry = qmc.controlled(qmc.ry)
    q[j], q[i] = qmc.cx(q[j], q[i])
    q[i], q[j] = controlled_ry(q[i], q[j], angle=-2.0 * theta)
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
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        givens_ij (qmc.Matrix[qmc.UInt]): Matrix of shape ``(N, 2)`` where
            each row ``[i, j]`` contains the qubit indices for one Givens
            rotation.
        givens_theta (qmc.Vector[qmc.Float]): Vector of length ``N`` with
            the rotation angles.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n_rotations = givens_ij.shape[0]
    for k in qmc.range(n_rotations):
        q = givens_rotation(q, givens_ij[k, 0], givens_ij[k, 1], givens_theta[k])
    return q


# ------------------------------------------------------------------
# Mixer
# ------------------------------------------------------------------


@qmc.qkernel
def hopping_gate(
    q: qmc.Vector[qmc.Qubit],
    i: qmc.UInt,
    j: qmc.UInt,
    beta: qmc.Float,
    hopping: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the fermionic hopping gate between qubits *i* and *j*.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        i (qmc.UInt): Index of the first qubit.
        j (qmc.UInt): Index of the second qubit.
        beta (qmc.Float): Variational parameter for the mixer layer.
        hopping (qmc.Float): Hopping integral strength.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
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
) -> qmc.Vector[qmc.Qubit]:
    """Apply the fermionic mixer layer (even-odd-boundary hopping).

    Applies hopping gates in even-pair, odd-pair, and boundary order.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        beta (qmc.Float): Variational parameter for the mixer layer.
        hopping (qmc.Float): Hopping integral strength.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    last_qubit = q.shape[0] - 1
    for i in qmc.range(0, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    for i in qmc.range(1, last_qubit, 2):
        q = hopping_gate(q, i, i + 1, beta, hopping)
    q = hopping_gate(q, qmc.uint(0), last_qubit, beta, hopping)
    return q


# ------------------------------------------------------------------
# Quadratic cost + layers + state
# ------------------------------------------------------------------


@qmc.qkernel
def cost_layer(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the cost (phase separation) layer.

    Applies RZZ gates for quadratic terms and RZ gates for linear terms,
    each scaled by the variational parameter gamma.

    Args:
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients J_{ij} of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients h_i of the Ising model.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gamma (qmc.Float): Variational parameter for the cost layer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    for (i, j), Jij in qmc.items(quad):
        q[i], q[j] = qmc.rzz(q[i], q[j], 2 * Jij * gamma)

    for i, hi in qmc.items(linear):
        q[i] = qmc.rz(q[i], 2 * hi * gamma)

    return q


@qmc.qkernel
def fqaoa_layers(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply *p* layers of cost + mixer.

    Each layer applies the Ising cost circuit followed by the fermionic
    mixer circuit.

    Args:
        p (qmc.UInt): Number of FQAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        hopping (qmc.Float): Hopping integral for the mixer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = cost_layer(quad, linear, q, gammas[layer])
        q = mixer_layer(q, betas[layer], hopping)
    return q


@qmc.qkernel
def fqaoa_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    num_fermions: qmc.UInt,
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate complete FQAOA state.

    Args:
        p (qmc.UInt): Number of FQAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of Ising model.
        n (qmc.UInt): Number of qubits.
        num_fermions (qmc.UInt): Number of fermions for initial state.
        givens_ij (qmc.Matrix[qmc.UInt]): Matrix of shape ``(N, 2)`` with
            qubit index pairs for Givens rotations.
        givens_theta (qmc.Vector[qmc.Float]): Vector of length ``N`` with
            Givens rotation angles.
        hopping (qmc.Float): Hopping integral for the mixer.
        gammas (qmc.Vector[qmc.Float]): Vector of gamma parameters.
        betas (qmc.Vector[qmc.Float]): Vector of beta parameters.

    Returns:
        qmc.Vector[qmc.Qubit]: FQAOA state vector.
    """
    q = qmc.qubit_array(n, name="q")
    q = initial_occupations(q, num_fermions)
    q = givens_rotations(q, givens_ij, givens_theta)
    q = fqaoa_layers(p, quad, linear, q, gammas, betas, hopping)
    return q


# ------------------------------------------------------------------
# HUBO cost + layers + state
# ------------------------------------------------------------------


@qmc.qkernel
def hubo_cost_layer(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the cost (phase separation) layer including higher-order terms.

    Applies the standard quadratic cost layer, then decomposes each
    higher-order term into phase gadgets.

    Args:
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients J_{ij} of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients h_i of the Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gamma (qmc.Float): Variational parameter for the cost layer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    q = cost_layer(quad, linear, q, gamma)
    for indices, coeff in qmc.items(higher):
        q = _basic.phase_gadget(q, indices, coeff * gamma)
    return q


@qmc.qkernel
def hubo_fqaoa_layers(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply *p* layers of HUBO FQAOA circuit (cost + mixer).

    Each layer applies the HUBO cost circuit (quadratic + higher-order terms)
    followed by the fermionic mixer circuit.

    Args:
        p (qmc.UInt): Number of FQAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        hopping (qmc.Float): Hopping integral for the mixer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = hubo_cost_layer(quad, linear, higher, q, gammas[layer])
        q = mixer_layer(q, betas[layer], hopping)
    return q


@qmc.qkernel
def hubo_fqaoa_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    num_fermions: qmc.UInt,
    givens_ij: qmc.Matrix[qmc.UInt],
    givens_theta: qmc.Vector[qmc.Float],
    hopping: qmc.Float,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate complete HUBO FQAOA state.

    Creates the fermionic initial state via Givens rotations and applies
    *p* layers of the HUBO FQAOA circuit.

    Args:
        p (qmc.UInt): Number of FQAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        n (qmc.UInt): Number of qubits.
        num_fermions (qmc.UInt): Number of fermions for initial state.
        givens_ij (qmc.Matrix[qmc.UInt]): Matrix of shape ``(N, 2)`` with
            qubit index pairs for Givens rotations.
        givens_theta (qmc.Vector[qmc.Float]): Vector of length ``N`` with
            Givens rotation angles.
        hopping (qmc.Float): Hopping integral for the mixer.
        gammas (qmc.Vector[qmc.Float]): Vector of gamma parameters.
        betas (qmc.Vector[qmc.Float]): Vector of beta parameters.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO FQAOA state vector.
    """
    q = qmc.qubit_array(n, name="q")
    q = initial_occupations(q, num_fermions)
    q = givens_rotations(q, givens_ij, givens_theta)
    q = hubo_fqaoa_layers(p, quad, linear, higher, q, gammas, betas, hopping)
    return q
