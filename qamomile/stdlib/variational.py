# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.

"""Functional variational-algorithm building blocks."""

from __future__ import annotations

import qamomile as qm


@qm.qkernel
def uniform_superposition(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply a Hadamard transform to a fixed register prefix."""

    for index in range(size):
        state = qm.h(state, target=index)
    return state


@qm.qkernel
def rx_layer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
    angles: qm.Param[qm.Vector[qm.Real, size]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply one runtime-parametric X-rotation layer."""

    for index in range(size):
        state = qm.rx(state, angles[index], target=index)
    return state


@qm.qkernel
def ry_layer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
    angles: qm.Param[qm.Vector[qm.Real, size]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply one runtime-parametric Y-rotation layer."""

    for index in range(size):
        state = qm.ry(state, angles[index], target=index)
    return state


@qm.qkernel
def rz_layer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
    angles: qm.Param[qm.Vector[qm.Real, size]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply one runtime-parametric Z-rotation layer."""

    for index in range(size):
        state = qm.rz(state, angles[index], target=index)
    return state


@qm.qkernel
def cx_entangling_layer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply a nearest-neighbor controlled-X chain."""

    for index in range(size - 1):
        state = qm.cx(state, control=index, target=index + 1)
    return state


@qm.qkernel
def cz_entangling_layer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply a nearest-neighbor controlled-Z chain."""

    for index in range(size - 1):
        state = qm.cz(state, control=index, target=index + 1)
    return state


@qm.qkernel
def phase_gadget(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    count: qm.Const[qm.Int],  # type: ignore[valid-type]
    indices: qm.Const[qm.Vector[qm.Int, count]],  # type: ignore[valid-type]
    angle: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply a multi-component Z phase gadget to a nonempty index vector."""

    for offset in range(count - 1):
        state = qm.cx(
            state,
            control=indices[offset],
            target=indices[offset + 1],
        )
    state = qm.rz(state, angle, target=indices[count - 1])
    for offset in range(count - 1):
        reverse = count - offset - 2
        state = qm.cx(
            state,
            control=indices[reverse],
            target=indices[reverse + 1],
        )
    return state


@qm.qkernel
def ising_cost(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    gamma: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply an exact commuting Ising cost Hamiltonian."""

    return qm.pauli_evolve(state, hamiltonian, gamma)


@qm.qkernel
def x_mixer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
    beta: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply the standard transverse-field X mixer."""

    for index in range(size):
        state = qm.rx(state, 2.0 * beta, target=index)
    return state


@qm.qkernel
def qaoa_layers(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply cost and mixer transformations for every QAOA layer."""

    for layer in range(layers):
        state = ising_cost(state, hamiltonian, gammas[layer])
        state = x_mixer(state, width, betas[layer])
    return state


@qm.qkernel
def qaoa_state(
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Prepare a QAOA state from one backend-neutral Ising Hamiltonian."""

    state = uniform_superposition(qm.qubits(width), width)
    return qaoa_layers(state, width, layers, hamiltonian, gammas, betas)


@qm.qkernel
def initial_occupations(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    occupations: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Occupy the leading fermionic modes in computational order."""

    for index in range(occupations):
        state = qm.x(state, target=index)
    return state


@qm.qkernel
def givens_rotation(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    left: qm.Int,  # type: ignore[valid-type]
    right: qm.Int,  # type: ignore[valid-type]
    theta: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply one number-preserving real Givens rotation."""

    state = qm.cx(state, control=right, target=left)
    state = qm.ry(state, -theta, target=right)
    state = qm.cx(state, control=left, target=right)
    state = qm.ry(state, theta, target=right)
    state = qm.cx(state, control=left, target=right)
    state = qm.cx(state, control=right, target=left)
    return state


@qm.qkernel
def givens_rotations(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    givens_count: qm.Const[qm.Int],  # type: ignore[valid-type]
    pairs: qm.Const[  # type: ignore[valid-type]
        qm.Vector[tuple[qm.Int, qm.Int], givens_count]
    ],
    angles: qm.Param[qm.Vector[qm.Real, givens_count]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply an immutable sequence of indexed Givens rotations."""

    for index in range(givens_count):
        pair = pairs[index]
        state = givens_rotation(state, pair[0], pair[1], angles[index])
    return state


@qm.qkernel
def hopping_gate(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    left: qm.Int,  # type: ignore[valid-type]
    right: qm.Int,  # type: ignore[valid-type]
    beta: qm.Real,  # type: ignore[valid-type]
    hopping: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply one fermionic hopping transform between two logical modes."""

    state = qm.rx(state, -1.5707963267948966, target=left)
    state = qm.rx(state, 1.5707963267948966, target=right)
    state = qm.cx(state, control=left, target=right)
    state = qm.rx(state, -beta * hopping, target=left)
    state = qm.rz(state, beta * hopping, target=right)
    state = qm.cx(state, control=left, target=right)
    state = qm.rx(state, 1.5707963267948966, target=left)
    state = qm.rx(state, -1.5707963267948966, target=right)
    return state


@qm.qkernel
def fermionic_mixer(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    beta: qm.Real,  # type: ignore[valid-type]
    hopping: qm.Real,  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply the even-odd-periodic schedule for a multi-mode register."""

    for left in range(0, width - 1, 2):
        state = hopping_gate(state, left, left + 1, beta, hopping)
    for left in range(1, width - 1, 2):
        state = hopping_gate(state, left, left + 1, beta, hopping)
    state = hopping_gate(state, 0, width - 1, beta, hopping)
    return state


@qm.qkernel
def fqaoa_layers(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    hopping: qm.Real,  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Apply fermionic cost and mixer layers."""

    for layer in range(layers):
        state = ising_cost(state, hamiltonian, gammas[layer])
        state = fermionic_mixer(state, width, betas[layer], hopping)
    return state


@qm.qkernel
def fqaoa_state(
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    occupations: qm.Const[qm.Int],  # type: ignore[valid-type]
    givens_count: qm.Const[qm.Int],  # type: ignore[valid-type]
    pairs: qm.Const[  # type: ignore[valid-type]
        qm.Vector[tuple[qm.Int, qm.Int], givens_count]
    ],
    givens_angles: qm.Param[qm.Vector[qm.Real, givens_count]],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    hopping: qm.Param[qm.Real],  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
    """Prepare a complete fermionic-QAOA state."""

    state = initial_occupations(qm.qubits(width), occupations)
    state = givens_rotations(state, givens_count, pairs, givens_angles)
    return fqaoa_layers(
        state,
        width,
        layers,
        hamiltonian,
        hopping,
        gammas,
        betas,
    )


__all__ = [
    "cx_entangling_layer",
    "cz_entangling_layer",
    "fermionic_mixer",
    "fqaoa_layers",
    "fqaoa_state",
    "givens_rotation",
    "givens_rotations",
    "hopping_gate",
    "initial_occupations",
    "ising_cost",
    "phase_gadget",
    "qaoa_layers",
    "qaoa_state",
    "rx_layer",
    "ry_layer",
    "rz_layer",
    "uniform_superposition",
    "x_mixer",
]
