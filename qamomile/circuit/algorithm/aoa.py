"""AOA (Alternating Operator Ansatz) circuit building blocks.

This module provides the quantum circuit components for the Alternating Operator Ansatz (AOA), 
including Givens circuit for efficient initial Dicke state preparation and xy mixer.

All functions are decorated with ``@qm_c.qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.
"""

import numpy as np

import qamomile.circuit as qmc

from . import basic as _basic
from .qaoa import hubo_ising_cost, ising_cost

@qmc.qkernel
def xy_pair_rotation(
    q: qmc.Vector[qmc.Qubit],
    i: qmc.UInt,
    j: qmc.UInt,
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """
    Apply exp(-i beta (X_i X_j + Y_i Y_j)) to qubits i and j.
    The decomposition into CNOT,RX and RZ gates is exact.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        i (qmc.UInt): Index of the first qubit in the pair.
        j (qmc.UInt): Index of the second qubit in the pair.
        beta (qmc.Float): Rotation angle for the XY interaction.
    
    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after applying the XY rotation.
    """
    q[i] = qmc.rx(q[i], -0.5 * np.pi)
    q[j] = qmc.rx(q[j], 0.5 * np.pi)
    q[i], q[j] = qmc.cx(q[i], q[j])
    q[i] = qmc.rx(q[i], beta)
    q[j] = qmc.rz(q[j], (-1.0) * beta)
    q[i], q[j] = qmc.cx(q[i], q[j])
    q[i] = qmc.rx(q[i], 0.5 * np.pi)
    q[j] = qmc.rx(q[j], -0.5 * np.pi)
    return q

@qmc.qkernel
def xy_mixer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
    pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """
    Apply a parity XY mixer on each one-hot color register.

    The Python side precomputes the qubit pairs for the parity schedule and
    passes them as ``pair_indices``.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        beta (qmc.Float): Rotation angle for the XY interaction.
        pair_indices (qmc.Matrix[qmc.UInt]): Matrix of shape (num_pairs, 2) containing the qubit index pairs for the XY mixer.
    
    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after applying the XY mixer.
    """
    num_pairs = pair_indices.shape[0]

    for pair_idx in qmc.range(num_pairs):
        q = xy_pair_rotation(
            q,
            pair_indices[pair_idx, 0],
            pair_indices[pair_idx, 1],
            beta,
        )

    return q

@qmc.qkernel
def aoa_layers(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Apply p layers of the AOA circuit (cost + XY mixer).

    Each layer applies the Ising cost circuit followed by the XY mixer.

    Args:
        p (qmc.UInt): Number of AOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        pair_indices (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = ising_cost(quad, linear, q, gammas[layer])
        q = xy_mixer(q, betas[layer], pair_indices)
    return q

@qmc.qkernel
def aoa_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Generate AOA State for Ising model.

    Args:
        p (qmc.UInt): Number of AOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        pair_indices (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: AOA state vector.
    """
    q = _basic.superposition_vector(n)
    q = aoa_layers(p, quad, linear, q, gammas, betas, pair_indices)
    return q


@qmc.qkernel
def hubo_aoa_layers(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Apply p layers of the HUBO AOA circuit (cost + XY mixer).

    Uses phase-gadget decomposition for higher-order terms in the cost layer,
    then applies the XY mixer.

    Args:
        p (qmc.UInt): Number of AOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        pair_indices (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = hubo_ising_cost(quad, linear, higher, q, gammas[layer])
        q = xy_mixer(q, betas[layer], pair_indices)
    return q


@qmc.qkernel
def hubo_aoa_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Generate HUBO AOA state starting from the uniform superposition.

    Args:
        p (qmc.UInt): Number of AOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        pair_indices (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO AOA state vector.
    """
    q = _basic.superposition_vector(n)
    q = hubo_aoa_layers(p, quad, linear, higher, q, gammas, betas, pair_indices)
    return q