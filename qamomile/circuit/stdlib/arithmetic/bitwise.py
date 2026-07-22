"""Implement bitwise helpers for constant modular multiplication."""

from __future__ import annotations

import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector


@qmc.qkernel
def _xor_bit(target: Qubit, bit: UInt) -> Qubit:
    """Apply an exact X power selected by a classical bit.

    ``RX(pi)`` equals ``-i X`` rather than ``X``.  The compensating global
    phase is irrelevant for a standalone call but becomes observable when
    this helper is controlled, as modular multiplication is in order finding.

    Args:
        target (Qubit): Qubit to update.
        bit (UInt): Zero or one selecting identity or X.

    Returns:
        Qubit: Updated qubit.
    """
    target = qmc.rx(target, math.pi * bit)
    return target


@qmc.qkernel
def _xor_constant(register: Vector[Qubit], value: UInt) -> Vector[Qubit]:
    """XOR a classical integer into a little-endian quantum register.

    Args:
        register (Vector[Qubit]): Register to update.
        value (UInt): Classical integer whose low bits are applied.

    Returns:
        Vector[Qubit]: Updated register.
    """
    for index in qmc.range(register.shape[0]):
        bit = (value // (2**index)) % 2
        exact_x_power = qmc.global_phase(_xor_bit, 0.5 * math.pi * bit)
        register[index] = exact_x_power(register[index], bit)
    return register


@qmc.qkernel
def _phase_flip_if(measurement: qmc.Bit, target: Qubit) -> Qubit:
    """Apply a Z correction selected by a measurement result.

    Args:
        measurement (qmc.Bit): X-basis vent result controlling the correction.
        target (Qubit): Dirty workspace qubit receiving the phase correction.

    Returns:
        Qubit: Corrected target qubit.
    """
    if measurement:
        target = qmc.z(target)
    return target


@qmc.qkernel
def _phase_shift_if(
    measurement: qmc.Bit,
    target: Qubit,
    angle: qmc.Float,
) -> Qubit:
    """Apply a phase shift selected by a prior measurement.

    Keeping the condition measurement-backed lets FTQC backends lower this
    semiclassical inverse-QFT correction to dynamic control flow.

    Args:
        measurement (qmc.Bit): Prior measurement controlling the correction.
        target (Qubit): Qubit receiving the phase shift when the bit is one.
        angle (qmc.Float): Phase angle in radians.

    Returns:
        Qubit: Corrected target qubit.
    """
    if measurement:
        target = qmc.p(target, angle)
    return target


def _apply_offset_x(
    control: Qubit | None,
    target: Qubit,
    bit: UInt,
) -> tuple[Qubit | None, Qubit]:
    """XOR one bit of a classical offset into a qubit.

    When ``control`` is present, the classical bit selects a CNOT from the
    control instead. The explicit phase compensation keeps the selected X
    exact when the operation is nested under coherent controls.

    Args:
        control (Qubit | None): Optional control for the classical offset.
        target (Qubit): Qubit to update.
        bit (UInt): Zero-or-one classical selector.

    Returns:
        tuple[Qubit | None, Qubit]: Updated optional control and target.
    """
    if control is None:
        exact_x = qmc.global_phase(_xor_bit, 0.5 * math.pi * bit)
        return None, exact_x(target, bit)
    selected_x = qmc.control(_xor_bit)
    control, target = selected_x(
        control,
        target,
        bit,
        global_phase=0.5 * math.pi * bit,
    )
    return control, target


def _apply_offset_parity_ccx(
    control: Qubit | None,
    first: Qubit,
    second: Qubit,
    target: Qubit,
    bit: UInt,
) -> tuple[Qubit | None, Qubit, Qubit, Qubit]:
    """Toggle a target from ``first & (second xor offset_bit)``.

    Args:
        control (Qubit | None): Optional control replacing a set offset bit.
        first (Qubit): First Toffoli control.
        second (Qubit): Second control, parity-conjugated by the offset bit.
        target (Qubit): Toffoli target.
        bit (UInt): Classical offset bit.

    Returns:
        tuple[Qubit | None, Qubit, Qubit, Qubit]: Updated handles.
    """
    control, second = _apply_offset_x(control, second, bit)
    first, second, target = qmc.ccx(first, second, target)
    control, second = _apply_offset_x(control, second, bit)
    return control, first, second, target


def _apply_two_offset_parities_ccx(
    control: Qubit | None,
    first: Qubit,
    second: Qubit,
    target: Qubit,
    bit: UInt,
) -> tuple[Qubit | None, Qubit, Qubit, Qubit]:
    """Toggle a target from two controls XORed with one offset bit.

    Args:
        control (Qubit | None): Optional control replacing a set offset bit.
        first (Qubit): First parity-conjugated Toffoli control.
        second (Qubit): Second parity-conjugated Toffoli control.
        target (Qubit): Toffoli target.
        bit (UInt): Classical offset bit.

    Returns:
        tuple[Qubit | None, Qubit, Qubit, Qubit]: Updated handles.
    """
    control, first = _apply_offset_x(control, first, bit)
    control, second = _apply_offset_x(control, second, bit)
    first, second, target = qmc.ccx(first, second, target)
    control, second = _apply_offset_x(control, second, bit)
    control, first = _apply_offset_x(control, first, bit)
    return control, first, second, target


def _apply_offset_and(
    control: Qubit | None,
    source: Qubit,
    target: Qubit,
    bit: UInt,
) -> tuple[Qubit | None, Qubit, Qubit]:
    """XOR ``source & offset_bit`` into a target.

    Args:
        control (Qubit | None): Optional control replacing a set offset bit.
        source (Qubit): Other conjunction input.
        target (Qubit): Conjunction target.
        bit (UInt): Classical offset bit.

    Returns:
        tuple[Qubit | None, Qubit, Qubit]: Updated handles.
    """
    if control is None:
        selected_x = qmc.control(_xor_bit)
        source, target = selected_x(
            source,
            target,
            bit,
            global_phase=0.5 * math.pi * bit,
        )
        return None, source, target

    selected_x = qmc.control(_xor_bit, num_controls=2)
    source, control, target = selected_x(
        source,
        control,
        target,
        bit,
        global_phase=0.5 * math.pi * bit,
    )
    return control, source, target


def _apply_offset_parity_and(
    control: Qubit | None,
    source: Qubit,
    target: Qubit,
    bit: UInt,
) -> tuple[Qubit | None, Qubit, Qubit]:
    """XOR ``(source xor offset_bit) & offset_bit`` into a target.

    Args:
        control (Qubit | None): Optional control replacing a set offset bit.
        source (Qubit): Parity-conjugated conjunction input.
        target (Qubit): Conjunction target.
        bit (UInt): Classical offset bit.

    Returns:
        tuple[Qubit | None, Qubit, Qubit]: Updated handles.
    """
    control, source = _apply_offset_x(control, source, bit)
    control, source, target = _apply_offset_and(control, source, target, bit)
    control, source = _apply_offset_x(control, source, bit)
    return control, source, target
