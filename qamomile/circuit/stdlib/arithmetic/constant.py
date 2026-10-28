"""Implement Fourier-based constant addition primitives."""

from __future__ import annotations

import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector


@qmc.qkernel
def _extended_qft(
    target: Vector[Qubit],
    overflow: Qubit,
) -> tuple[Vector[Qubit], Qubit]:
    """Apply QFT to ``overflow || target`` without allocating a vector.

    Args:
        target (Vector[Qubit]): Little-endian low bits of the value.
        overflow (Qubit): Most-significant bit of the value.

    Returns:
        tuple[Vector[Qubit], Qubit]: Fourier-transformed register handles.
    """
    size = target.shape[0]
    overflow = qmc.h(overflow)
    for delta in qmc.range(size):
        control_index = size - 1 - delta
        angle = math.pi / (2 ** (size - control_index))
        overflow, target[control_index] = qmc.cp(
            overflow,
            target[control_index],
            angle,
        )
    for offset in qmc.range(size):
        target_index = size - 1 - offset
        target[target_index] = qmc.h(target[target_index])
        for delta in qmc.range(target_index):
            control_index = target_index - 1 - delta
            angle = math.pi / (2 ** (target_index - control_index))
            target[target_index], target[control_index] = qmc.cp(
                target[target_index],
                target[control_index],
                angle,
            )
    target[0], overflow = qmc.swap(target[0], overflow)
    for index in qmc.range(1, (size + 1) // 2):
        mirror = size - index
        target[index], target[mirror] = qmc.swap(target[index], target[mirror])
    return target, overflow


@qmc.qkernel
def _extended_iqft(
    target: Vector[Qubit],
    overflow: Qubit,
) -> tuple[Vector[Qubit], Qubit]:
    """Undo QFT on ``overflow || target`` without allocating a vector.

    Args:
        target (Vector[Qubit]): Fourier-encoded low-bit handles.
        overflow (Qubit): Fourier-encoded most-significant-bit handle.

    Returns:
        tuple[Vector[Qubit], Qubit]: Computational-basis register handles.
    """
    size = target.shape[0]
    target[0], overflow = qmc.swap(target[0], overflow)
    for index in qmc.range(1, (size + 1) // 2):
        mirror = size - index
        target[index], target[mirror] = qmc.swap(target[index], target[mirror])
    for target_index in qmc.range(size):
        for control_index in qmc.range(target_index):
            angle = -math.pi / (2 ** (target_index - control_index))
            target[target_index], target[control_index] = qmc.cp(
                target[target_index],
                target[control_index],
                angle,
            )
        target[target_index] = qmc.h(target[target_index])
    for control_index in qmc.range(size):
        angle = -math.pi / (2 ** (size - control_index))
        overflow, target[control_index] = qmc.cp(
            overflow,
            target[control_index],
            angle,
        )
    overflow = qmc.h(overflow)
    return target, overflow


@qmc.qkernel
def _add_const_body(
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
) -> tuple[Vector[Qubit], Qubit]:
    """Add a classical value to an extended quantum register.

    Args:
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to add.

    Returns:
        tuple[Vector[Qubit], Qubit]: Updated low bits and overflow bit.
    """
    target, overflow = _extended_qft(target, overflow)
    for bit_index in qmc.range(target.shape[0]):
        angle = 2 * math.pi * value / (2 ** (target.shape[0] + 1 - bit_index))
        target[bit_index] = qmc.p(target[bit_index], angle)
    overflow = qmc.p(
        overflow,
        math.pi * value,
    )
    target, overflow = _extended_iqft(target, overflow)
    return target, overflow


@qmc.qkernel
def _subtract_const_body(
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
) -> tuple[Vector[Qubit], Qubit]:
    """Subtract a classical value from an extended quantum register.

    Args:
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to subtract.

    Returns:
        tuple[Vector[Qubit], Qubit]: Updated low bits and overflow bit.
    """
    target, overflow = _extended_qft(target, overflow)
    for bit_index in qmc.range(target.shape[0]):
        angle = -2 * math.pi * value / (2 ** (target.shape[0] + 1 - bit_index))
        target[bit_index] = qmc.p(target[bit_index], angle)
    overflow = qmc.p(
        overflow,
        -math.pi * value,
    )
    target, overflow = _extended_iqft(target, overflow)
    return target, overflow


@qmc.qkernel
def _controlled_add_const_body(
    control: Qubit,
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
) -> tuple[Qubit, Vector[Qubit], Qubit]:
    """Condition an extended-register constant addition on one qubit.

    Args:
        control (Qubit): Quantum control preserved by the operation.
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to add.

    Returns:
        tuple[Qubit, Vector[Qubit], Qubit]: Updated control, low bits, and
            overflow bit.
    """
    target, overflow = _extended_qft(target, overflow)
    for bit_index in qmc.range(target.shape[0]):
        angle = 2 * math.pi * value / (2 ** (target.shape[0] + 1 - bit_index))
        control, target[bit_index] = qmc.cp(
            control,
            target[bit_index],
            angle,
        )
    control, overflow = qmc.cp(
        control,
        overflow,
        math.pi * value,
    )
    target, overflow = _extended_iqft(target, overflow)
    return control, target, overflow


@qmc.qkernel
def _controlled_subtract_const_body(
    control: Qubit,
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
) -> tuple[Qubit, Vector[Qubit], Qubit]:
    """Condition an extended-register constant subtraction on one qubit.

    Args:
        control (Qubit): Quantum control preserved by the operation.
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to subtract.

    Returns:
        tuple[Qubit, Vector[Qubit], Qubit]: Updated control, low bits, and
            overflow bit.
    """
    target, overflow = _extended_qft(target, overflow)
    for bit_index in qmc.range(target.shape[0]):
        angle = -2 * math.pi * value / (2 ** (target.shape[0] + 1 - bit_index))
        control, target[bit_index] = qmc.cp(
            control,
            target[bit_index],
            angle,
        )
    control, overflow = qmc.cp(
        control,
        overflow,
        -math.pi * value,
    )
    target, overflow = _extended_iqft(target, overflow)
    return control, target, overflow


def _apply_const_add(
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
    *,
    control: Qubit | None,
    inverse: bool,
) -> tuple[Qubit | None, Vector[Qubit], Qubit]:
    """Add or subtract a classical value with an optional quantum control.

    Args:
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to add or subtract.
        control (Qubit | None): Optional quantum control. Defaults to None.
        inverse (bool): Whether to subtract instead of add.

    Returns:
        tuple[Qubit | None, Vector[Qubit], Qubit]: Updated control, low bits,
            and overflow bit.
    """
    if control is None:
        operation = _subtract_const_body if inverse else _add_const_body
        target, overflow = operation(target, overflow, value)
        return None, target, overflow
    operation = (
        _controlled_subtract_const_body if inverse else _controlled_add_const_body
    )
    control, target, overflow = operation(control, target, overflow, value)
    return control, target, overflow


@qmc.composite_gate(name="add_const")
def add_const(
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
) -> tuple[Vector[Qubit], Qubit]:
    """Add a classical constant to an extended quantum register.

    Args:
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to add.

    Returns:
        tuple[Vector[Qubit], Qubit]: Updated low bits and overflow bit.
    """
    _, target, overflow = _apply_const_add(
        target,
        overflow,
        value,
        control=None,
        inverse=False,
    )
    return target, overflow


@qmc.composite_gate(name="controlled_add_const")
def controlled_add_const(
    control: Qubit,
    target: Vector[Qubit],
    overflow: Qubit,
    value: UInt,
) -> tuple[Qubit, Vector[Qubit], Qubit]:
    """Condition a classical constant addition on one qubit.

    Args:
        control (Qubit): Quantum control preserved by the operation.
        target (Vector[Qubit]): Little-endian low bits to update.
        overflow (Qubit): Most-significant bit of the extended value.
        value (UInt): Classical non-negative value to add.

    Returns:
        tuple[Qubit, Vector[Qubit], Qubit]: Updated control, low bits, and
            overflow bit.
    """
    control_out, target, overflow = _apply_const_add(
        target,
        overflow,
        value,
        control=control,
        inverse=False,
    )
    assert control_out is not None
    return control_out, target, overflow
