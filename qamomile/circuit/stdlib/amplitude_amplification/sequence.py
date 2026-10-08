r"""Emit an amplitude-amplification sequence into an enclosing qkernel.

:func:`amplitude_amplification` applies a state preparation and then one round
per phase pair in a schedule, raising the probability of landing in the good
subspace — the all-zero state of the signal register. That is exactly the
success event of a block encoding's post-selection.

The sequence is unrolled in Python, so this is a plain function rather than a
qkernel: a Python loop inside a qkernel body would be rewritten into traced
control flow.
"""

from __future__ import annotations

from typing import Any

import qamomile.circuit as qmc

from .preparation import _validate_state_preparation
from .reflection import _good_phase_rotation, _zero_phase_rotation
from .schedule import AmplificationSchedule


def _resolve_preparation(preparation: object) -> Any:
    """Return the preparation kernel behind a descriptor or a bare callable.

    Accepts a concrete :class:`~qamomile.circuit.stdlib.AmplificationPreparation`,
    the static-binding proxy for one, or any callable already carrying the
    ``(signal, system)`` ABI such as ``encoding.unitary``. The descriptor is
    detected by attribute rather than ``isinstance`` so that a binding proxy
    resolves the same way a concrete descriptor does.

    Args:
        preparation (object): Descriptor, binding proxy, or bare callable.

    Returns:
        Any: Callable with the block-encoding ABI.

    Raises:
        TypeError: If the resolved callable does not expose the required ABI.
    """
    member = getattr(preparation, "preparation", None)
    return _validate_state_preparation(
        preparation if member is None else member,
    )


def _resolve_schedule(schedule: object) -> tuple[float, ...]:
    """Return the interleaved phases of an amplification schedule.

    Args:
        schedule (object): Candidate schedule descriptor.

    Returns:
        tuple[float, ...]: Interleaved reflection phases.

    Raises:
        TypeError: If ``schedule`` is not an ``AmplificationSchedule``.
    """
    if not isinstance(schedule, AmplificationSchedule):
        raise TypeError(
            "schedule must be an AmplificationSchedule. Build one with "
            "standard_amplification_schedule or "
            "fixed_point_amplification_schedule."
        )
    return schedule.phases


def _emit_amplification_sequence(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    preparation: Any,
    phases: tuple[float, ...],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Emit the preparation followed by one round per phase pair.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register in the all-zero state.
        system (qmc.Vector[qmc.Qubit]): System register in the all-zero state.
        preparation (Any): Validated preparation callable.
        phases (tuple[float, ...]): Interleaved reflection phases.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Registers after
            amplification.
    """
    signal, system = preparation(signal, system)
    if not phases:
        return signal, system

    auxiliary = qmc.qubit("amplitude_amplification_auxiliary")
    inverse_preparation = qmc.inverse(preparation)
    for index in range(len(phases) // 2):
        good_phase = phases[2 * index]
        zero_phase = phases[2 * index + 1]
        signal, auxiliary = _good_phase_rotation(signal, auxiliary, good_phase)
        signal, system = inverse_preparation(signal, system)
        signal, system, auxiliary = _zero_phase_rotation(
            signal,
            system,
            auxiliary,
            zero_phase,
        )
        signal, system = preparation(signal, system)
    return signal, system


def amplitude_amplification(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    preparation: Any,
    schedule: AmplificationSchedule,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    r"""Amplify the all-zero signal subspace of a prepared state.

    Applies ``preparation`` to the all-zero registers and then, for each
    consecutive phase pair ``(good, zero)`` of ``schedule``, one round of

    ``S_good(good)``, ``A^dagger``, ``S_zero(zero)``, ``A``.

    A standard schedule makes every round the Brassard-Hoyer-Mosca-Tapp
    iterate ``A S_0 A^dagger S_good`` — the Grover iterate up to a global sign
    — reaching ``sin^2((2k + 1) theta)``. A fixed-point schedule reaches at
    least ``1 - delta^2`` for every initial success probability above its
    bound, without overshooting. The schedule predicts the outcome
    analytically through
    :meth:`~qamomile.circuit.stdlib.AmplificationSchedule.success_probability`.

    One clean auxiliary qubit is allocated and reused by every reflection, so
    the circuit uses ``num_signal_qubits + num_system_qubits + 1`` qubits. A
    kernel that calls this function therefore cannot itself be passed to
    :func:`~qamomile.circuit.inverse`, which rejects kernels that allocate
    qubits internally.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register in the all-zero state
            on entry. Its all-zero subspace is the good subspace.
        system (qmc.Vector[qmc.Qubit]): System register in the all-zero state
            on entry.
        preparation (Any): An
            :class:`~qamomile.circuit.stdlib.AmplificationPreparation`, a
            static binding of one, or any callable with the block-encoding ABI
            such as ``encoding.unitary``. Must be invertible by
            :func:`~qamomile.circuit.inverse`.
        schedule (AmplificationSchedule): Phase schedule to apply. An empty
            schedule emits the bare preparation.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Signal and system
            registers after amplification.

    Raises:
        TypeError: If ``preparation`` does not expose the required ABI or
            ``schedule`` is not an ``AmplificationSchedule``.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import (
        ...     amplitude_amplification,
        ...     fixed_point_amplification_schedule,
        ... )
        >>> def build(encoding):
        ...     schedule = fixed_point_amplification_schedule(0.05, 0.1)
        ...
        ...     @qmc.qkernel
        ...     def amplified() -> qmc.Vector[qmc.Bit]:
        ...         signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        ...         system = qmc.qubit_array(encoding.num_system_qubits, "system")
        ...         signal, _ = amplitude_amplification(
        ...             signal, system, encoding.unitary, schedule
        ...         )
        ...         return qmc.measure(signal)
        ...
        ...     return amplified
    """
    resolved_preparation = _resolve_preparation(preparation)
    phases = _resolve_schedule(schedule)
    return _emit_amplification_sequence(
        signal,
        system,
        resolved_preparation,
        phases,
    )


__all__ = ["amplitude_amplification"]
