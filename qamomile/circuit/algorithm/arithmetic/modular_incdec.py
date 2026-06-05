"""Modular increment and decrement operators for basis-state registers."""

from __future__ import annotations

import qamomile.circuit as qmc


@qmc.qkernel
def modular_increment(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular increment ``|j> -> |j + 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to increment in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    for k in qmc.range(n):
        target_index = n - 1 - k
        if target_index == 0:
            q[target_index] = qmc.x(q[target_index])
        else:
            mcx = qmc.control(qmc.x, num_controls=target_index)
            controls = q[0:target_index]
            target = q[target_index]
            controls, target = mcx(controls, target)
            q[0:target_index] = controls
            q[target_index] = target
    return q


@qmc.qkernel
def modular_decrement(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular decrement ``|j> -> |j - 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to decrement in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    for target_index in qmc.range(n):
        if target_index == 0:
            q[target_index] = qmc.x(q[target_index])
        else:
            mcx = qmc.control(qmc.x, num_controls=target_index)
            controls = q[0:target_index]
            target = q[target_index]
            controls, target = mcx(controls, target)
            q[0:target_index] = controls
            q[target_index] = target
    return q


@qmc.qkernel
def controlled_modular_increment(
    control: qmc.Qubit,
    q: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
    """Apply controlled modular increment to a target register.

    The target vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer. When ``control`` is
    ``|1>``, the target maps as ``|j> -> |j + 1 mod 2^n>``; when
    ``control`` is ``|0>``, the target is unchanged.

    Args:
        control (qmc.Qubit): External control qubit.
        q (qmc.Vector[qmc.Qubit]): Target qubit register.

    Returns:
        tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]: Updated control qubit
        and target register.
    """
    n = q.shape[0]
    for k in qmc.range(n):
        target_index = n - 1 - k
        mcx = qmc.control(qmc.x, num_controls=target_index + 1)
        prefix = q[0:target_index]
        target = q[target_index]
        control, prefix, target = mcx(control, prefix, target)
        q[0:target_index] = prefix
        q[target_index] = target
    return control, q


@qmc.qkernel
def controlled_modular_decrement(
    control: qmc.Qubit,
    q: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
    """Apply controlled modular decrement to a target register.

    The target vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer. When ``control`` is
    ``|1>``, the target maps as ``|j> -> |j - 1 mod 2^n>``; when
    ``control`` is ``|0>``, the target is unchanged.

    Args:
        control (qmc.Qubit): External control qubit.
        q (qmc.Vector[qmc.Qubit]): Target qubit register.

    Returns:
        tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]: Updated control qubit
        and target register.
    """
    n = q.shape[0]
    for target_index in qmc.range(n):
        mcx = qmc.control(qmc.x, num_controls=target_index + 1)
        prefix = q[0:target_index]
        target = q[target_index]
        control, prefix, target = mcx(control, prefix, target)
        q[0:target_index] = prefix
        q[target_index] = target
    return control, q


@qmc.qkernel
def controlled_modular_increment_by_index(
    q: qmc.Vector[qmc.Qubit],
    control_index: qmc.UInt,
    num_system: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply controlled modular increment inside a shared register.

    The layout is ``q[0:num_system]`` for the little-endian system
    register and ``q[control_index]`` for the external control qubit.
    When the control qubit is ``|1>``, the system maps as
    ``|j> -> |j + 1 mod 2^num_system>``; when it is ``|0>``, the
    system is unchanged.

    Args:
        q (qmc.Vector[qmc.Qubit]): Register containing both the system
            qubits and the external control qubit.
        control_index (qmc.UInt): Index of the external control qubit
            in ``q``. It is expected to be outside ``q[0:num_system]``.
        num_system (qmc.UInt): Number of system qubits at the front of
            ``q``.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated shared register.
    """
    for k in qmc.range(num_system):
        target_index = num_system - 1 - k
        mcx = qmc.control(qmc.x, num_controls=target_index + 1)
        control = q[control_index]
        prefix = q[0:target_index]
        target = q[target_index]
        control, prefix, target = mcx(control, prefix, target)
        q[control_index] = control
        q[0:target_index] = prefix
        q[target_index] = target
    return q


@qmc.qkernel
def controlled_modular_decrement_by_index(
    q: qmc.Vector[qmc.Qubit],
    control_index: qmc.UInt,
    num_system: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply controlled modular decrement inside a shared register.

    The layout is ``q[0:num_system]`` for the little-endian system
    register and ``q[control_index]`` for the external control qubit.
    When the control qubit is ``|1>``, the system maps as
    ``|j> -> |j - 1 mod 2^num_system>``; when it is ``|0>``, the
    system is unchanged.

    Args:
        q (qmc.Vector[qmc.Qubit]): Register containing both the system
            qubits and the external control qubit.
        control_index (qmc.UInt): Index of the external control qubit
            in ``q``. It is expected to be outside ``q[0:num_system]``.
        num_system (qmc.UInt): Number of system qubits at the front of
            ``q``.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated shared register.
    """
    for target_index in qmc.range(num_system):
        mcx = qmc.control(qmc.x, num_controls=target_index + 1)
        control = q[control_index]
        prefix = q[0:target_index]
        target = q[target_index]
        control, prefix, target = mcx(control, prefix, target)
        q[control_index] = control
        q[0:target_index] = prefix
        q[target_index] = target
    return q


__all__ = [
    "controlled_modular_decrement",
    "controlled_modular_decrement_by_index",
    "controlled_modular_increment",
    "controlled_modular_increment_by_index",
    "modular_decrement",
    "modular_increment",
]
