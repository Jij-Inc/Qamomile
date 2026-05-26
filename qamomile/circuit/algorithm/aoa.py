"""AOA (Alternating Operator Ansatz) circuit building blocks.

This module provides the quantum circuit components for the Alternating Operator Ansatz (AOA),
including Dicke state preparation (using Bartschi-Eidenbenz SCS construction) and the XY mixer.

All functions are decorated with ``@qmc.qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.
"""

import numpy as np

import qamomile.circuit as qmc

from . import basic as _basic
from .qaoa import hubo_ising_cost, ising_cost
from .state_preparation.dicke import prepare_dicke


@qmc.qkernel
def xy_pair_rotation(
    q: qmc.Vector[qmc.Qubit],
    i: qmc.UInt,
    j: qmc.UInt,
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply exp(-i beta/2 (X_i X_j + Y_i Y_j)) to qubits i and j.

    The decomposition into CNOT, RX, and RZ gates is exact.

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
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Apply XY pair rotations along the supplied mixer schedule.

    Each row of ``pair_indices_mixer`` selects a qubit pair on which
    :func:`xy_pair_rotation` is applied with the same ``beta``.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        beta (qmc.Float): Rotation angle for the XY interaction.
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Matrix of shape (num_pairs, 2) containing the qubit index pairs for the XY mixer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after applying the XY mixer.
    """
    num_pairs = pair_indices_mixer.shape[0]

    for pair_idx in qmc.range(num_pairs):
        q = xy_pair_rotation(
            q,
            pair_indices_mixer[pair_idx, 0],
            pair_indices_mixer[pair_idx, 1],
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
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
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
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = ising_cost(quad, linear, q, gammas[layer])
        q = xy_mixer(q, betas[layer], pair_indices_mixer)
    return q


@qmc.qkernel
def aoa_state_superposition(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
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
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: AOA state vector.
    """
    q = _basic.superposition_vector(n)
    q = aoa_layers(p, quad, linear, q, gammas, betas, pair_indices_mixer)
    return q


@qmc.qkernel
def basis_state_preparation(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a computational basis state from a list of ``|1>`` indices.

    Args:
        n (qmc.UInt): Number of qubits.
        initial_ones (qmc.Vector[qmc.UInt]): Indices initialized to ``|1>``.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register in the requested basis state.
    """
    q = qmc.qubit_array(n, name="q")
    for idx in qmc.range(initial_ones.shape[0]):
        qubit_index = initial_ones[idx]
        q[qubit_index] = qmc.x(q[qubit_index])
    return q


@qmc.qkernel
def aoa_state_basis_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
    initial_ones: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Generate AOA state starting from a computational basis state.

    Args:
        p (qmc.UInt): Number of AOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer.
        initial_ones (qmc.Vector[qmc.UInt]): Indices initialized to ``|1>``.

    Returns:
        qmc.Vector[qmc.Qubit]: AOA state vector.
    """
    q = basis_state_preparation(n, initial_ones)
    q = aoa_layers(p, quad, linear, q, gammas, betas, pair_indices_mixer)
    return q


@qmc.qkernel
def aoa_state_dicke(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
    initial_ones: qmc.Vector[qmc.UInt],
    pairs_dicke: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    triplets_dicke: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate AOA State, whose initial state is a Dicke state, for Ising model.

    Args:
        p (qmc.UInt): Number of AOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits that are initially in the ``|1>`` state for Dicke state preparation.
        pairs_dicke (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Precomputed
            mapping from ``(t, c)`` qubit pairs to rotation angles for the 2-qubit SCS blocks.
        triplets_dicke (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Precomputed mapping
            from ``(t, c1, c2)`` qubit index triples to rotation angles for the 3-qubit
            SCS blocks. Keys are length-3 index vectors.

    Returns:
        qmc.Vector[qmc.Qubit]: AOA state vector.
    """
    q = prepare_dicke(
        n,
        initial_ones,
        pairs_dicke,
        triplets_dicke,
    )
    q = aoa_layers(p, quad, linear, q, gammas, betas, pair_indices_mixer)
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
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
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
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = hubo_ising_cost(quad, linear, higher, q, gammas[layer])
        q = xy_mixer(q, betas[layer], pair_indices_mixer)
    return q


@qmc.qkernel
def hubo_aoa_state_superposition(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
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
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO AOA state vector.
    """
    q = _basic.superposition_vector(n)
    q = hubo_aoa_layers(p, quad, linear, higher, q, gammas, betas, pair_indices_mixer)
    return q


@qmc.qkernel
def hubo_aoa_state_basis_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
    initial_ones: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Generate HUBO AOA state starting from a computational basis state.

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
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer.
        initial_ones (qmc.Vector[qmc.UInt]): Indices initialized to ``|1>``.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO AOA state vector.
    """
    q = basis_state_preparation(n, initial_ones)
    q = hubo_aoa_layers(p, quad, linear, higher, q, gammas, betas, pair_indices_mixer)
    return q


@qmc.qkernel
def hubo_aoa_state_dicke(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
    initial_ones: qmc.Vector[qmc.UInt],
    pairs_dicke: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    triplets_dicke: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate HUBO AOA state, whose initial state is a Dicke state.

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
        pair_indices_mixer (qmc.Matrix[qmc.UInt]): Qubit pairs for the XY mixer parity schedule.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits that are initially in the ``|1>`` state for Dicke state preparation.
        pairs_dicke (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Precomputed
            mapping from ``(t, c)`` qubit pairs to rotation angles for the 2-qubit SCS blocks.
        triplets_dicke (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Precomputed mapping
            from ``(t, c1, c2)`` qubit index triples to rotation angles for the 3-qubit
            SCS blocks. Keys are length-3 index vectors.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO AOA state vector.
    """
    q = prepare_dicke(
        n,
        initial_ones,
        pairs_dicke,
        triplets_dicke,
    )
    q = hubo_aoa_layers(p, quad, linear, higher, q, gammas, betas, pair_indices_mixer)
    return q
