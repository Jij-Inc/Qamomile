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
    for layer in qmc.range(p):
        q = ising_cost_circuit(quad, linear, q, gammas[layer])
        q = x_mixier_circuit(q, betas[layer])
    return q


@qmc.qkernel
def superposition_vector(
    n: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
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

