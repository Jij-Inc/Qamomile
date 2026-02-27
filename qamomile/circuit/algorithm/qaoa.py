from __future__ import annotations

from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Dict, Float, Qubit, Tuple, UInt, Vector
from qamomile.circuit.frontend.operation.control_flow import range
from qamomile.circuit.frontend.operation.qubit_gates import h, rx, rz, rzz
from qamomile.circuit.frontend.qkernel import qkernel


@qkernel
def ising_cost_circuit(
    quad: Dict[Tuple[UInt, UInt], Float],
    linear: Dict[UInt, Float],
    q: Vector[Qubit],
    gamma: Float,
) -> Vector[Qubit]:
    for (i, j), Jij in quad.items():
        q[i], q[j] = rzz(q[i], q[j], angle=Jij * gamma)
    for i, hi in linear.items():
        q[i] = rz(q[i], angle=hi * gamma)
    return q


@qkernel
def x_mixier_circuit(
    q: Vector[Qubit],
    beta: Float,
) -> Vector[Qubit]:
    n = q.shape[0]
    for i in range(n):
        q[i] = rx(q[i], angle=2.0 * beta)
    return q


@qkernel
def qaoa_circuit(
    p: UInt,
    quad: Dict[Tuple[UInt, UInt], Float],
    linear: Dict[UInt, Float],
    q: Vector[Qubit],
    gammas: Vector[Float],
    betas: Vector[Float],
) -> Vector[Qubit]:
    for layer in range(p):
        q = ising_cost_circuit(quad, linear, q, gammas[layer])
        q = x_mixier_circuit(q, betas[layer])
    return q


@qkernel
def superposition_vector(
    n: UInt
) -> Vector[Qubit]:
    q = qubit_array(n, name="q")
    for i in range(n):
        q[i] = h(q[i])
    return q


@qkernel
def qaoa_state(
    p: UInt,
    quad: Dict[Tuple[UInt, UInt], Float],
    linear: Dict[UInt, Float],
    n: UInt,
    gammas: Vector[Float],
    betas: Vector[Float],
) -> Vector[Qubit]:
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
