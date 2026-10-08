r"""Generalized reflections used by every amplitude-amplification schedule.

Both composites realize ``S(phase) = I - (1 - exp(i * phase)) * Pi`` for a
projector ``Pi``: :func:`_good_phase_rotation` projects onto the all-zero
signal register (the good subspace), and :func:`_zero_phase_rotation` projects
onto the joint all-zero signal-plus-system state. A hard reflection is
``phase = pi``.

This convention uses a phase gate rather than an ``rz`` rotation, so it leaves
no residual global phase. The emitted sequence is therefore exactly the
intended unitary and stays correct under
:func:`~qamomile.circuit.control`. Both constructions are branch-free and
width-agnostic, so a one-qubit signal register needs no special case, and both
leave the auxiliary qubit in ``|0>`` so one clean auxiliary serves an entire
amplification sequence.
"""

from __future__ import annotations

from typing import Any, cast

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.ir.operation.callable import CallPolicy


@qmc.composite_gate(name="amplitude_amplification_good_phase_rotation")
def _good_phase_rotation(
    signal: qmc.Vector[qmc.Qubit],
    auxiliary: qmc.Qubit,
    phase: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    r"""Apply ``I - (1 - exp(i * phase)) * Pi`` for the all-zero signal subspace.

    The signal register is inverted around two multi-controlled X gates so a
    clean auxiliary carries the "signal is all-zero" flag, takes the phase, and
    is uncomputed. The auxiliary returns to zero and is reusable for every
    phase in one amplification sequence. The construction is branch-free and
    width-agnostic, so it also supports a one-qubit signal register.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Non-empty signal register whose
            all-zero state defines the good subspace.
        auxiliary (qmc.Qubit): Clean reusable auxiliary qubit.
        phase (qmc.Float): Reflection phase in radians. ``pi`` gives the hard
            reflection ``I - 2 * Pi``.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]: Phased signal register and
            restored clean auxiliary.
    """
    signal = qmc.x(signal)
    signal, auxiliary = qmc.mcx(signal, auxiliary)
    auxiliary = qmc.p(auxiliary, phase)
    signal, auxiliary = qmc.mcx(signal, auxiliary)
    signal = qmc.x(signal)
    return signal, auxiliary


configure_composite(
    _good_phase_rotation,
    name="amplitude_amplification_good_phase_rotation",
    namespace="qamomile.stdlib",
    policy=CallPolicy.NATIVE_FIRST,
)


@qmc.composite_gate(name="amplitude_amplification_zero_phase_rotation")
def _zero_phase_rotation(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    auxiliary: qmc.Qubit,
    phase: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    r"""Apply ``I - (1 - exp(i * phase)) * |0><0|`` over both registers jointly.

    Both registers are inverted so the joint all-zero state becomes the
    all-ones state. A clean auxiliary carries the "signal is all-zero" flag and
    a multi-controlled phase gate over the system register applies the phase
    only when the system is also all-zero. The auxiliary is then uncomputed.

    The controls are kept on a single register per controlled operation. A
    multi-slot control spanning both registers would resolve its control count
    symbolically and fail during constant folding.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Non-empty signal register.
        system (qmc.Vector[qmc.Qubit]): Non-empty system register.
        auxiliary (qmc.Qubit): Clean reusable auxiliary qubit.
        phase (qmc.Float): Reflection phase in radians. ``pi`` gives the hard
            reflection ``I - 2 * |0><0|``.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]: Phased
            signal and system registers and the restored clean auxiliary.
    """
    signal = qmc.x(signal)
    system = qmc.x(system)
    signal, auxiliary = qmc.mcx(signal, auxiliary)
    controlled_phase = cast(Any, qmc.control(qmc.p, num_controls=system.shape[0]))
    system, auxiliary = controlled_phase(system, auxiliary, phase)
    signal, auxiliary = qmc.mcx(signal, auxiliary)
    signal = qmc.x(signal)
    system = qmc.x(system)
    return signal, system, auxiliary


configure_composite(
    _zero_phase_rotation,
    name="amplitude_amplification_zero_phase_rotation",
    namespace="qamomile.stdlib",
    policy=CallPolicy.NATIVE_FIRST,
)


__all__: list[str] = []
