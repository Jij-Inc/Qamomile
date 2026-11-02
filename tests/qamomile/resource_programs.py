# ruff: noqa: F821 -- Qamomile dependent shape binders.
"""Quantum program fixtures for symbolic resource and execution regressions."""

from __future__ import annotations

import qamomile as qm
from qamomile import resources as qr
from qamomile.stdlib import phase_estimation, qft


# %% QFT: a triangular phase network, then inspect its elementary price.
@qm.qkernel
def fourier(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:
    """Apply the textbook QFT to zero and measure the register."""
    return qm.measure(qft(qm.qubits(width), width))


# %% QPE: the oracle has a visible body, not a manually supplied query rule.
@qm.qkernel
def evolution(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Give the eigenstate |1> phase one quarter of a full turn."""
    return qm.p(state, 1.5707963267948966)


@qm.qkernel
def phase_sample(precision: qm.Const[qm.Int]) -> qm.Shot[qm.UFixed[0, precision]]:
    """Estimate the prepared eigenstate's phase, including preparation/readout."""
    target = qm.x(qm.qubits(1))
    phase, target = phase_estimation(evolution, target, precision)
    result = qm.measure(phase)
    _target_result = qm.measure(target)
    return result


# %% Amplitude amplification: expose the cost of both reflections.
@qm.qkernel
def uniform(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    """Prepare the uniform superposition over eight possible answers."""
    return qm.h(state)


@qm.qkernel
def mark_good(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    """Reflect the marked answer |111>, preserving its full phase."""
    state = qm.h(state, target=2)
    state = qm.ccx(state)
    return qm.h(state, target=2)


@qm.qkernel
def search_step(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    """Apply a marked reflection and a uniform-state reflection."""
    state = mark_good(state)
    state = qm.adjoint(uniform, state)
    state = qm.x(state)
    state = mark_good(state)
    state = qm.x(state)
    return uniform(state)


@qm.qkernel
def amplified_search(iterations: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[3]]:
    """Sample a Grover search after a symbolic number of iterations."""
    state = uniform(qm.qubits(3))
    for _ in range(iterations):
        state = search_step(state)
    return qm.measure(state)


# %% Product formula: H = X + Z, exact finite cost at a chosen slice count.
@qm.qkernel
def strang_slice(state: qm.Q[qm.Bits[1]], dt: qm.Real) -> qm.Q[qm.Bits[1]]:
    """Implement exp(-i X dt/2) exp(-i Z dt) exp(-i X dt/2)."""
    state = qm.rx(state, dt)
    state = qm.rz(state, 2.0 * dt)
    return qm.rx(state, dt)


@qm.qkernel
def product_formula(
    steps: qm.Const[qm.Int], time: qm.Param[qm.Real]
) -> qm.Shot[qm.Bits[1]]:
    """Run positive steps of the second-order product formula."""
    state = qm.qubits(1)
    for _ in range(steps):
        state = strang_slice(state, time / steps)
    return qm.measure(state)


@qm.qkernel
def fused_product_formula(
    steps: qm.Const[qm.Int], time: qm.Param[qm.Real]
) -> qm.Shot[qm.Bits[1]]:
    """Implement the same positive-step product with adjacent X rotations fused."""
    dt = time / steps
    state = qm.rx(qm.qubits(1), dt)
    for _ in range(steps - 1):
        state = qm.rz(state, 2.0 * dt)
        state = qm.rx(state, 2.0 * dt)
    state = qm.rz(state, 2.0 * dt)
    state = qm.rx(state, dt)
    return qm.measure(state)


# %% Canonical amplitude estimation: phase estimation of a Grover iterate.
@qm.qkernel
def prepare_amplitude(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Encode success probability sin(pi/8)**2."""
    return qm.ry(state, 0.7853981633974483)


@qm.qkernel
def amplitude_step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Rotate the success angle by pi/4 using two reflections."""
    state = qm.z(state)
    state = qm.adjoint(prepare_amplitude, state)
    state = qm.z(state)
    return prepare_amplitude(state)


@qm.qkernel
def amplitude_sample(precision: qm.Const[qm.Int]) -> qm.Shot[qm.UFixed[0, precision]]:
    """Return a phase; classical sin(pi*phase)**2 recovers the probability."""
    target = prepare_amplitude(qm.qubits(1))
    phase, target = phase_estimation(amplitude_step, target, precision)
    result = qm.measure(phase)
    _target_result = qm.measure(target)
    return result


def studies() -> dict[str, qr.ResourceStudy]:
    """Analyze each authored family once, before any numeric size sweep."""
    return {
        "QFT": qr.logical(fourier),
        "QPE": qr.logical(phase_sample),
        "amplification": qr.logical(amplified_search),
        "product_formula": qr.logical(product_formula),
        "amplitude_estimation": qr.logical(amplitude_sample),
    }
