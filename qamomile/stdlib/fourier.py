# ruff: noqa: F821 -- ``n`` is a Qamomile dependent shape binder.

"""Functional quantum Fourier transforms written in ordinary Qamomile source."""

from __future__ import annotations

import qamomile as qm


@qm.qkernel
def qft(state: qm.Q[qm.Bits[n]], size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:  # type: ignore[valid-type]
    """Apply the exact quantum Fourier transform to a register prefix.

    Args:
        state (qm.Q[qm.Bits[n]]): Owned quantum state whose leading ``size``
            components are transformed.
        size (qm.Const[qm.Int]): Compile-time width of the transformed prefix.

    Returns:
        qm.Q[qm.Bits[n]]: Transformed state in little-endian source order.
    """

    for offset in range(size):
        target = size - 1 - offset
        state = qm.h(state, target=target)
        for delta in range(target):
            control = target - 1 - delta
            angle = 3.141592653589793 / (2 ** (target - control))
            state = qm.cp(state, angle, control=control, target=target)
    for left in range(size // 2):
        state = qm.swap(state, left=left, right=size - left - 1)
    return state


@qm.qkernel
def approximate_qft(
    state: qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
    size: qm.Const[qm.Int],  # type: ignore[valid-type]
    max_distance: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[n]]:  # type: ignore[valid-type]
    """Apply a Fourier transform with small controlled rotations omitted.

    The cutoff changes the operator. Distances at least ``size - 1`` give the
    exact QFT; nonpositive distances retain only Hadamards and reversal swaps.
    Use ``qm.adjoint(approximate_qft, state, size, max_distance)`` for its inverse.

    Args:
        state (qm.Q[qm.Bits[n]]): Owned state whose leading ``size`` components
            are transformed. Components outside that prefix remain unchanged.
        size (qm.Const[qm.Int]): Compile-time width of the transformed prefix.
        max_distance (qm.Const[qm.Int]): Largest retained control-target distance.
            A rotation at distance ``d`` has angle ``pi / 2**d``.

    Returns:
        qm.Q[qm.Bits[n]]: Approximately Fourier-transformed successor state in
            little-endian source order.
    """

    for offset in range(size):
        target = size - 1 - offset
        state = qm.h(state, target=target)
        for delta in range(target):
            control = target - 1 - delta
            if target - control <= max_distance:
                angle = 3.141592653589793 / (2 ** (target - control))
                state = qm.cp(state, angle, control=control, target=target)
    for left in range(size // 2):
        state = qm.swap(state, left=left, right=size - left - 1)
    return state


@qm.qkernel
def inverse_qft(state: qm.Q[qm.Bits[n]], size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:  # type: ignore[valid-type]
    """Apply the compiler-derived exact inverse Fourier transform.

    Args:
        state (qm.Q[qm.Bits[n]]): Owned quantum state whose leading ``size``
            components are transformed.
        size (qm.Const[qm.Int]): Compile-time width of the transformed prefix.

    Returns:
        qm.Q[qm.Bits[n]]: Inverse-transformed state in little-endian source order.
    """

    return qm.adjoint(qft, state, size)
