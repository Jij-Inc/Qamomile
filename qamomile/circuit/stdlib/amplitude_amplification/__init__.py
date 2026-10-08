r"""General-purpose amplitude amplification for post-selected quantum states.

This subpackage raises the probability of landing in a *good subspace* defined
by "the signal register is all-zero". That is exactly the success event of a
:class:`~qamomile.circuit.stdlib.LCUBlockEncoding`, so the primary use is
boosting block-encoding post-selection, but the routines accept any state
preparation with the block-encoding ABI.

The pieces mirror the block-encoding descriptors: frozen objects built by named
factories, driving one circuit-emitting function.

Schedules describe *how* to amplify.
    :class:`StandardAmplificationSchedule` is the Brassard-Hoyer-Mosca-Tapp
    rotation. Starting from ``A|0>`` with success probability
    ``a = sin^2(theta)``, ``k`` rounds reach ``sin^2((2k + 1) theta)``. It is
    optimal when ``a`` is known and overshoots when it is not, so the round
    count comes from :func:`amplitude_amplification_iteration_count`.

    :class:`FixedPointAmplificationSchedule` is the Yoder-Low-Chuang
    fixed-point schedule. It needs only a *lower bound* ``lambda <= a`` and
    reaches at least ``1 - delta^2`` for every ``a >= lambda`` without
    overshooting, using ``L = O(log(2 / delta) / sqrt(lambda))`` queries. This
    is the useful variant for block encodings, whose exact success amplitude is
    generically unknown.

    Both answer
    :meth:`AmplificationSchedule.success_probability`, so the outcome of a
    sequence is predictable before a single gate is emitted.
    :class:`AmplificationSchedule` itself accepts any hand-designed phase list.

:class:`AmplificationPreparation` describes *what* to amplify: a state
    preparation together with its register widths. It is registered for static
    qkernel binding, so a template can be serialized before any concrete
    preparation exists and bound later.

:func:`amplitude_amplification` emits the sequence.

Reflection convention:
    Rounds are built from generalized reflections
    ``S(phase) = I - (1 - exp(i * phase)) * Pi``, with ``Pi`` projecting onto
    the all-zero signal register or onto the joint all-zero signal-plus-system
    state. A hard reflection is ``phase = pi``. This convention leaves no
    residual global phase, so the emitted sequence is exactly the intended
    unitary and stays correct under :func:`~qamomile.circuit.control`.

Cost:
    One clean auxiliary qubit is allocated per amplification and reused by
    every reflection; it returns to ``|0>`` after each one. The total width is
    ``num_signal_qubits + num_system_qubits + 1``.

Bit-flip oracles:
    A marker that flips an ancilla instead of applying a phase is bridged by
    making that ancilla the signal register: the good subspace is then the
    all-zero flag, which is what these routines reflect about.

Example:
    ```python
    import qamomile.circuit as qmc

    encoding = qmc.pauli_lcu_block_encoding(lcu)
    schedule = qmc.fixed_point_amplification_schedule(0.05, 0.1)


    @qmc.qkernel
    def amplified() -> qmc.Vector[qmc.Bit]:
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, system = qmc.amplitude_amplification(
            signal, system, encoding.unitary, schedule
        )
        return qmc.measure(signal)
    ```
"""

from .preparation import AmplificationPreparation, amplification_preparation
from .schedule import (
    AmplificationSchedule,
    FixedPointAmplificationSchedule,
    StandardAmplificationSchedule,
    amplitude_amplification_iteration_count,
    fixed_point_amplification_schedule,
    standard_amplification_schedule,
)
from .sequence import amplitude_amplification

__all__ = [
    "AmplificationSchedule",
    "StandardAmplificationSchedule",
    "FixedPointAmplificationSchedule",
    "standard_amplification_schedule",
    "fixed_point_amplification_schedule",
    "amplitude_amplification_iteration_count",
    "AmplificationPreparation",
    "amplification_preparation",
    "amplitude_amplification",
]
