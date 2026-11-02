# ruff: noqa: F821 -- ``n`` and ``precision`` are dependent shape binders.

"""Quantum phase estimation as one ordinary higher-order qkernel."""

from __future__ import annotations

import qamomile as qm

from .fourier import inverse_qft


@qm.qkernel
def windowed_phase_estimation(
    unitary: qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]],  # type: ignore[valid-type]
    window: qm.Q[qm.Bits[precision]],  # type: ignore[valid-type]
    target: qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
    precision: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> tuple[
    qm.Q[qm.UFixed[0, precision]],  # type: ignore[valid-type]
    qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
]:
    """Estimate an eigenphase from one prepared phase-window state.

    The counting register is exposed as an unsigned fixed-point phase in the
    half-open interval ``[0, 1)``. Its fractional width is ``precision``. The
    standard QFT uses little-endian source order, so the final register reversal
    maps those lanes to the canonical most-significant-first ``UFixed`` digits.
    ``window`` may be a rectangular, Kaiser, sine, or application-specific
    state; the function consumes it exactly once and returns only its numeric
    phase view.

    Args:
        unitary (qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]]): Unary exact unitary
            applied to the target register.
        window (qm.Q[qm.Bits[precision]]): Prepared phase-window register. Its
            computational-basis index supplies the coherent evolution time.
        target (qm.Q[qm.Bits[n]]): Owned eigenstate retained by the result.
        precision (qm.Const[qm.Int]): Positive specialization-time number of
            phase bits.

    Returns:
        tuple[qm.Q[qm.UFixed[0, precision]], qm.Q[qm.Bits[n]]]: The coherent
        numeric phase and the retained target state.

    Raises:
        ValueError: If ``precision`` is not positive during specialization.
    """

    system = (window, target)
    for control_index in range(precision):
        counting, target = system
        system = (counting, target)
        for _ in range(2**control_index):
            counting, target = system
            counting, target = qm.control(
                unitary,
                counting,
                target,
                at=control_index,
            )
            system = (counting, target)

    counting, target = system
    counting = inverse_qft(counting, precision)
    for left in range(precision // 2):
        counting = qm.swap(
            counting,
            left=left,
            right=precision - left - 1,
        )
    return qm.as_fixed(counting, integer_bits=0), target


@qm.qkernel
def phase_estimation(
    unitary: qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]],  # type: ignore[valid-type]
    target: qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
    precision: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> tuple[
    qm.Q[qm.UFixed[0, precision]],  # type: ignore[valid-type]
    qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
]:
    """Estimate an eigenphase while preserving the target owner.

    This is textbook phase estimation with a rectangular window. The counting
    register is prepared in uniform superposition and then passed through
    :func:`windowed_phase_estimation`, which owns the shared phase-kickback,
    inverse-QFT, and result-ordering semantics.

    Args:
        unitary (qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]]): Unary exact unitary
            applied to the target register.
        target (qm.Q[qm.Bits[n]]): Owned eigenstate retained by the result.
        precision (qm.Const[qm.Int]): Positive specialization-time number of
            phase bits.

    Returns:
        tuple[qm.Q[qm.UFixed[0, precision]], qm.Q[qm.Bits[n]]]: The coherent
        numeric phase and the retained target state.

    Raises:
        ValueError: If ``precision`` is not positive during specialization.
    """

    window = qm.qubits(precision)
    for index in range(precision):
        window = qm.h(window, target=index)
    return windowed_phase_estimation(unitary, window, target, precision)


__all__ = ["phase_estimation", "windowed_phase_estimation"]
