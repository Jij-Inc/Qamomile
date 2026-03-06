from __future__ import annotations

import qamomile.circuit as qmc

from . import basic as _basic


@qmc.qkernel
def ising_cost(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the Ising cost layer for quadratic interactions.

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
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * gamma)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * gamma)
    return q


@qmc.qkernel
def x_mixer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the X-mixer layer.

    Applies RX(2*beta) to every qubit in the register.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        beta (qmc.Float): Variational parameter for the mixer layer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return q


@qmc.qkernel
def qaoa_layers(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Apply p layers of the QAOA circuit (cost + mixer).

    Each layer applies the Ising cost circuit followed by the X-mixer circuit.

    Args:
        p (qmc.UInt): Number of QAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p):
        q = ising_cost(quad, linear, q, gammas[layer])
        q = x_mixer(q, betas[layer])
    return q


@qmc.qkernel
def qaoa_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate QAOA State for Ising model.

    Args:
        p (qmc.UInt): Number of QAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.

    Returns:
        qmc.Vector[qmc.Qubit]: QAOA state vector.
    """
    q = _basic.superposition_vector(n)
    q = qaoa_layers(p, quad, linear, q, gammas, betas)
    return q


@qmc.qkernel
def hubo_ising_cost(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the full cost layer including higher-order terms.

    Applies the standard quadratic Ising cost circuit, then decomposes each
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
    q = ising_cost(quad, linear, q, gamma)
    for indices, coeff in qmc.items(higher):
        q = _basic.phase_gadget(q, indices, coeff * gamma)
    return q


@qmc.qkernel
def hubo_qaoa_layers(
    p_val: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Apply p layers of the HUBO QAOA circuit (cost + mixer).

    Each layer applies the HUBO cost circuit (quadratic + higher-order terms)
    followed by the X-mixer circuit.

    Args:
        p_val (qmc.UInt): Number of QAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p_val):
        q = hubo_ising_cost(quad, linear, higher, q, gammas[layer])
        q = x_mixer(q, betas[layer])
    return q


@qmc.qkernel
def hubo_qaoa_state(
    p_val: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Generate HUBO QAOA state.

    Creates a uniform superposition and applies p layers of the HUBO QAOA
    circuit.

    Args:
        p_val (qmc.UInt): Number of QAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        higher (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Higher-order coefficients keyed by index vectors.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO QAOA state vector.
    """
    q = _basic.superposition_vector(n)
    q = hubo_qaoa_layers(p_val, quad, linear, higher, q, gammas, betas)
    return q
