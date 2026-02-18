from __future__ import annotations

import qamomile.circuit as qmc


def apply_phase_gadget(
    q: qmc.Vector[qmc.Qubit],
    indices: list[int],
    angle: qmc.Float | float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply exp(-i * angle/2 * Z_{i0} Z_{i1} ... Z_{ik-1}).

    Decomposes a k-body Z-rotation into CX + RZ primitives.
    This is a plain Python function (not a @qkernel) intended to be
    called during qkernel tracing.

    Args:
        q: Qubit register.
        indices: Concrete qubit indices for the interaction term.
        angle: Rotation angle in radians.

    Returns:
        Updated qubit register.
    """
    k = len(indices)
    if k == 0:
        return q
    if k == 1:
        q[indices[0]] = qmc.rz(q[indices[0]], angle=angle)
        return q
    if k == 2:
        q[indices[0]], q[indices[1]] = qmc.rzz(
            q[indices[0]], q[indices[1]], angle=angle
        )
        return q
    # k >= 3: CX forward ladder
    for step in range(k - 1):
        q[indices[step]], q[indices[step + 1]] = qmc.cx(
            q[indices[step]], q[indices[step + 1]]
        )
    # RZ on last qubit
    q[indices[-1]] = qmc.rz(q[indices[-1]], angle=angle)
    # CX reverse ladder
    for step in range(k - 2, -1, -1):
        q[indices[step]], q[indices[step + 1]] = qmc.cx(
            q[indices[step]], q[indices[step + 1]]
        )
    return q


@qmc.qkernel
def ising_cost_circuit(
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
def x_mixier_circuit(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the X-mixer layer.

    Applies RX(2*beta) to every qubit in the register.

    Note:
        The function name contains a typo (``mixier`` instead of ``mixer``)
        preserved for backward compatibility.

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
def qaoa_circuit(
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
        q = ising_cost_circuit(quad, linear, q, gammas[layer])
        q = x_mixier_circuit(q, betas[layer])
    return q


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


def hubo_cost_circuit(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
    higher_terms: list[tuple[tuple[int, ...], float]],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the full cost layer including higher-order terms.

    Applies the standard quadratic Ising cost circuit, then decomposes each
    higher-order term into phase gadgets. This is a plain Python function
    (not a @qkernel) that iterates over concrete precomputed data during
    qkernel tracing.

    Args:
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients J_{ij} of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients h_i of the Ising model.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gamma (qmc.Float): Variational parameter for the cost layer.
        higher_terms (list[tuple[tuple[int, ...], float]]):
            Sorted list of ``(index_tuple, coefficient)`` pairs for
            higher-order interactions.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    q = ising_cost_circuit(quad, linear, q, gamma)
    for indices, coeff in higher_terms:
        q = apply_phase_gadget(q, list(indices), coeff * gamma)
    return q


def hubo_qaoa_circuit(
    p_val: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    higher_terms: list[tuple[tuple[int, ...], float]],
) -> qmc.Vector[qmc.Qubit]:
    """Apply p layers of the HUBO QAOA circuit (cost + mixer).

    Each layer applies the HUBO cost circuit (quadratic + higher-order terms)
    followed by the X-mixer circuit. This is a plain Python function (not a
    @qkernel) because it wraps ``hubo_cost_circuit`` which iterates over
    concrete precomputed data.

    Args:
        p_val (qmc.UInt): Number of QAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        higher_terms (list[tuple[tuple[int, ...], float]]):
            Sorted list of ``(index_tuple, coefficient)`` pairs.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after all layers.
    """
    for layer in qmc.range(p_val):
        q = hubo_cost_circuit(quad, linear, q, gammas[layer], higher_terms)
        q = x_mixier_circuit(q, betas[layer])
    return q


def hubo_qaoa_state(
    p_val: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    higher_terms: list[tuple[tuple[int, ...], float]],
) -> qmc.Vector[qmc.Qubit]:
    """Generate HUBO QAOA state.

    Creates a uniform superposition and applies p layers of the HUBO QAOA
    circuit. This is a plain Python function (not a @qkernel) because it
    wraps ``hubo_qaoa_circuit``.

    Args:
        p_val (qmc.UInt): Number of QAOA layers.
        quad (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]):
            Quadratic coefficients of the Ising model.
        linear (qmc.Dict[qmc.UInt, qmc.Float]):
            Linear coefficients of the Ising model.
        n (qmc.UInt): Number of qubits.
        gammas (qmc.Vector[qmc.Float]): Cost-layer parameters, one per layer.
        betas (qmc.Vector[qmc.Float]): Mixer-layer parameters, one per layer.
        higher_terms (list[tuple[tuple[int, ...], float]]):
            Sorted list of ``(index_tuple, coefficient)`` pairs.

    Returns:
        qmc.Vector[qmc.Qubit]: HUBO QAOA state vector.
    """
    q = superposition_vector(n)
    q = hubo_qaoa_circuit(p_val, quad, linear, q, gammas, betas, higher_terms)
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
        p: Number of QAOA layers.
        quad: Quadratic coefficients of Ising model.
        linear: Linear coefficients of Ising model.
        n: Number of qubits.
        gammas: Vector of gamma parameters.
        betas: Vector of beta parameters.

    Returns:
        QAOA state vector.
    """
    q = superposition_vector(n)
    q = qaoa_circuit(p, quad, linear, q, gammas, betas)
    return q
