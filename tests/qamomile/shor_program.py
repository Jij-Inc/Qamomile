# ruff: noqa: F821 -- ``n`` is a Qamomile dependent shape binder.

"""Resource regression fixture for Shor order finding with 2 modulo 15.

This file deliberately contains only ordinary Qamomile source. Resource
estimation discovers its call graph, loops, gates, and live quantum state from
QCore; there is no Shor-specific estimator or algorithm registry entry.
"""

from __future__ import annotations

import qamomile as qm


@qm.qkernel
def t_dagger(state: qm.Q[qm.Bits[n]], target: qm.Int) -> qm.Q[qm.Bits[n]]:
    # T**7 equals T dagger.
    for _ in range(7):
        state = qm.t(state, target=target)
    return state


@qm.qkernel
def toffoli(
    state: qm.Q[qm.Bits[n]],
    control_a: qm.Int,
    control_b: qm.Int,
    target: qm.Int,
) -> qm.Q[qm.Bits[n]]:
    # Ancilla-free Clifford+T Toffoli decomposition.
    state = qm.h(state, target=target)
    state = qm.cx(state, control=control_b, target=target)
    state = t_dagger(state, target)
    state = qm.cx(state, control=control_a, target=target)
    state = qm.t(state, target=target)
    state = qm.cx(state, control=control_b, target=target)
    state = t_dagger(state, target)
    state = qm.cx(state, control=control_a, target=target)
    state = qm.t(state, target=control_b)
    state = qm.t(state, target=target)
    state = qm.h(state, target=target)
    state = qm.cx(state, control=control_a, target=control_b)
    state = qm.t(state, target=control_a)
    state = t_dagger(state, control_b)
    state = qm.cx(state, control=control_a, target=control_b)
    return state


@qm.qkernel
def controlled_swap(
    state: qm.Q[qm.Bits[n]],
    control: qm.Int,
    left: qm.Int,
    right: qm.Int,
) -> qm.Q[qm.Bits[n]]:
    # Fredkin from one Toffoli and two CNOTs.
    state = qm.cx(state, control=right, target=left)
    state = toffoli(state, control, left, right)
    state = qm.cx(state, control=right, target=left)
    return state


@qm.qkernel
def controlled_rotate_work(
    state: qm.Q[qm.Bits[n]],
    control: qm.Int,
    offset: qm.Int,
    shift: qm.Int,
) -> qm.Q[qm.Bits[n]]:
    # A cyclic rotation realizes multiplication by a power of two modulo 15.
    for _ in range(shift):
        state = controlled_swap(state, control, offset + 2, offset + 3)
        state = controlled_swap(state, control, offset + 1, offset + 2)
        state = controlled_swap(state, control, offset, offset + 1)
    return state


@qm.qkernel
def controlled_phase(
    state: qm.Q[qm.Bits[n]],
    control: qm.Int,
    target: qm.Int,
    angle: qm.Real,
) -> qm.Q[qm.Bits[n]]:
    # Controlled phase through RZ and CX primitives.
    state = qm.rz(state, angle / 2.0, target=target)
    state = qm.cx(state, control=control, target=target)
    state = qm.rz(state, -angle / 2.0, target=target)
    state = qm.cx(state, control=control, target=target)
    state = qm.rz(state, angle / 2.0, target=control)
    return state


@qm.qkernel
def inverse_qft(state: qm.Q[qm.Bits[n]], precision: qm.Int) -> qm.Q[qm.Bits[n]]:
    # Triangular controlled-phase network followed by bit reversal.
    for target in range(precision):
        for control in range(target):
            distance = target - control + 1
            angle = -6.283185307179586 / (2**distance)
            state = controlled_phase(state, control, target, angle)
        state = qm.h(state, target=target)
    for left in range(precision // 2):
        state = qm.swap(state, left=left, right=precision - left - 1)
    return state


@qm.qkernel
def shor_15() -> qm.Bits[12]:
    precision = 8
    state = qm.qubits(12)
    state = qm.x(state, target=precision)
    for control in range(precision):
        state = qm.h(state, target=control)

    # Multiplication by 2 mod 15 is a cyclic rotation of four little-endian
    # bits. Its squared action is a rotation by two; all higher QPE powers are
    # identity because 2 has order four modulo 15.
    state = controlled_rotate_work(state, 0, precision, 1)
    state = controlled_rotate_work(state, 1, precision, 2)
    state = inverse_qft(state, precision)
    return qm.measure(state)
