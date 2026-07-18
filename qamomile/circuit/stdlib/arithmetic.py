"""Provide executable constant modular-arithmetic building blocks."""

from __future__ import annotations

import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.ir.operation.callable import CallPolicy


@qmc.qkernel
def _majority(
    left: Qubit,
    right: Qubit,
    carry: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Propagate one ripple-carry majority step.

    Args:
        left (Qubit): Addend qubit for the current bit.
        right (Qubit): Accumulator qubit for the current bit.
        carry (Qubit): Incoming carry qubit.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Updated left, right, and carry qubits.
    """
    left, right = qmc.cx(left, right)
    left, carry = qmc.cx(left, carry)
    right, carry, left = qmc.ccx(right, carry, left)
    return left, right, carry


@qmc.qkernel
def _unmajority_add(
    left: Qubit,
    right: Qubit,
    carry: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Uncompute one majority step while writing the sum bit.

    Args:
        left (Qubit): Addend qubit for the current bit.
        right (Qubit): Accumulator qubit for the current bit.
        carry (Qubit): Outgoing carry qubit from the majority sweep.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Restored left, summed right, and restored
            carry qubits.
    """
    right, carry, left = qmc.ccx(right, carry, left)
    left, carry = qmc.cx(left, carry)
    carry, right = qmc.cx(carry, right)
    return left, right, carry


@qmc.qkernel
def _inverse_majority(
    left: Qubit,
    right: Qubit,
    carry: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Reverse one ripple-carry majority step.

    Args:
        left (Qubit): Addend qubit after majority propagation.
        right (Qubit): Accumulator qubit after majority propagation.
        carry (Qubit): Propagated carry qubit.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Qubits before the majority step.
    """
    right, carry, left = qmc.ccx(right, carry, left)
    left, carry = qmc.cx(left, carry)
    left, right = qmc.cx(left, right)
    return left, right, carry


@qmc.qkernel
def _inverse_unmajority_add(
    left: Qubit,
    right: Qubit,
    carry: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Reverse one unmajority-and-add step.

    Args:
        left (Qubit): Restored addend qubit.
        right (Qubit): Sum qubit.
        carry (Qubit): Restored carry qubit.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Qubits before the unmajority step.
    """
    carry, right = qmc.cx(carry, right)
    left, carry = qmc.cx(left, carry)
    right, carry, left = qmc.ccx(right, carry, left)
    return left, right, carry


@qmc.composite_gate(name="ripple_carry_add")
def ripple_carry_add(
    left: Vector[Qubit],
    right: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
) -> tuple[Vector[Qubit], Vector[Qubit], Qubit, Qubit]:
    """Add ``left`` into ``right`` with a reversible ripple-carry network.

    The equally sized registers are little-endian. ``carry`` must start in
    ``|0>`` and is restored to ``|0>``. ``overflow`` receives the final carry,
    so the output represents the full ``n + 1`` bit sum without discarding
    quantum information.

    Args:
        left (Vector[Qubit]): Little-endian addend register.
        right (Vector[Qubit]): Little-endian accumulator register.
        carry (Qubit): Clean carry workspace qubit, restored on return.
        overflow (Qubit): Qubit receiving the most significant carry bit.

    Returns:
        tuple[Vector[Qubit], Vector[Qubit], Qubit, Qubit]: Preserved addend,
            summed accumulator, restored carry, and updated overflow qubit.
    """
    size = left.shape[0]
    left[0], right[0], carry = _majority(left[0], right[0], carry)
    for index in qmc.range(1, size):
        left[index], right[index], left[index - 1] = _majority(
            left[index],
            right[index],
            left[index - 1],
        )

    left[size - 1], overflow = qmc.cx(left[size - 1], overflow)

    for index in qmc.range(size - 1, 0, -1):
        left[index], right[index], left[index - 1] = _unmajority_add(
            left[index],
            right[index],
            left[index - 1],
        )
    left[0], right[0], carry = _unmajority_add(left[0], right[0], carry)
    return left, right, carry, overflow


configure_composite(
    ripple_carry_add,
    namespace="qamomile.stdlib",
    policy=CallPolicy.NATIVE_FIRST,
)


@qmc.composite_gate(name="ripple_carry_subtract")
def _ripple_carry_subtract(
    left: Vector[Qubit],
    right: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
) -> tuple[Vector[Qubit], Vector[Qubit], Qubit, Qubit]:
    """Subtract ``left`` from ``right`` by reversing the adder body.

    Args:
        left (Vector[Qubit]): Little-endian subtrahend register.
        right (Vector[Qubit]): Little-endian accumulator register.
        carry (Qubit): Carry workspace restored on return.
        overflow (Qubit): High bit participating in the subtraction.

    Returns:
        tuple[Vector[Qubit], Vector[Qubit], Qubit, Qubit]: Preserved
            subtrahend, difference, carry, and high-bit handles.
    """
    size = left.shape[0]
    left[0], right[0], carry = _inverse_unmajority_add(left[0], right[0], carry)
    for index in qmc.range(1, size):
        left[index], right[index], left[index - 1] = _inverse_unmajority_add(
            left[index],
            right[index],
            left[index - 1],
        )

    left[size - 1], overflow = qmc.cx(left[size - 1], overflow)

    for index in qmc.range(size - 1, 0, -1):
        left[index], right[index], left[index - 1] = _inverse_majority(
            left[index],
            right[index],
            left[index - 1],
        )
    left[0], right[0], carry = _inverse_majority(left[0], right[0], carry)
    return left, right, carry, overflow


def _apply_ripple_carry_add(
    left: Vector[Qubit],
    right: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    *,
    control: Qubit | None,
    inverse: bool,
) -> tuple[Qubit | None, Vector[Qubit], Vector[Qubit], Qubit, Qubit]:
    """Apply a direct or inverse adder with an optional control.

    Args:
        left (Vector[Qubit]): Addend register.
        right (Vector[Qubit]): Accumulator register.
        carry (Qubit): Carry workspace.
        overflow (Qubit): Overflow qubit.
        control (Qubit | None): Optional control qubit.
        inverse (bool): Whether to subtract by applying the inverse adder.

    Returns:
        tuple[Qubit | None, Vector[Qubit], Vector[Qubit], Qubit, Qubit]: Updated
            control, addend, accumulator, carry, and overflow handles.
    """
    operation = _ripple_carry_subtract if inverse else ripple_carry_add
    if control is None:
        left, right, carry, overflow = operation(left, right, carry, overflow)
        return None, left, right, carry, overflow
    controlled = qmc.control(operation)
    control, left, right, carry, overflow = controlled(
        control,
        left,
        right,
        carry,
        overflow,
    )
    return control, left, right, carry, overflow


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


def _modular_add_const_body(
    target: Vector[Qubit],
    overflow: Qubit,
    flag: Qubit,
    addend: UInt,
    modulus: UInt,
    control: Qubit | None,
) -> tuple[Qubit | None, Vector[Qubit], Qubit, Qubit]:
    """Add a classical constant modulo another classical constant.

    Inputs must satisfy ``target < modulus``. The operation restores the
    overflow and flag work qubits and extends to a reversible permutation via
    the same compare-subtract-uncompute pattern as ``modular_add``.

    Args:
        target (Vector[Qubit]): Modular target register.
        overflow (Qubit): Clean high-bit workspace restored on return.
        flag (Qubit): Clean modular-reduction flag restored on return.
        addend (UInt): Classical value to add.
        modulus (UInt): Classical modulus.
        control (Qubit | None): Optional control for the addition.

    Returns:
        tuple[Qubit | None, Vector[Qubit], Qubit, Qubit]: Updated control,
            modular target, overflow, and flag.
    """
    control, target, overflow = _apply_const_add(
        target,
        overflow,
        addend,
        control=control,
        inverse=False,
    )
    _, target, overflow = _apply_const_add(
        target,
        overflow,
        modulus,
        control=None,
        inverse=True,
    )
    overflow, flag = qmc.cx(overflow, flag)
    flag, target, overflow = controlled_add_const(
        flag,
        target,
        overflow,
        modulus,
    )
    control, target, overflow = _apply_const_add(
        target,
        overflow,
        addend,
        control=control,
        inverse=True,
    )
    overflow = qmc.x(overflow)
    overflow, flag = qmc.cx(overflow, flag)
    overflow = qmc.x(overflow)
    control, target, overflow = _apply_const_add(
        target,
        overflow,
        addend,
        control=control,
        inverse=False,
    )
    return control, target, overflow, flag


@qmc.composite_gate(name="modular_add_const")
def modular_add_const(
    target: Vector[Qubit],
    overflow: Qubit,
    flag: Qubit,
    addend: UInt,
    modulus: UInt,
) -> tuple[Vector[Qubit], Qubit, Qubit]:
    """Add a classical constant to a quantum register modulo a constant.

    Args:
        target (Vector[Qubit]): Modular target register.
        overflow (Qubit): Clean high-bit workspace restored on return.
        flag (Qubit): Clean modular-reduction flag restored on return.
        addend (UInt): Classical value to add.
        modulus (UInt): Classical modulus.

    Returns:
        tuple[Vector[Qubit], Qubit, Qubit]: Modular target and restored
            workspace qubits.
    """
    _, target, overflow, flag = _modular_add_const_body(
        target,
        overflow,
        flag,
        addend,
        modulus,
        None,
    )
    return target, overflow, flag


@qmc.composite_gate(name="controlled_modular_add_const")
def controlled_modular_add_const(
    control: Qubit,
    target: Vector[Qubit],
    overflow: Qubit,
    flag: Qubit,
    addend: UInt,
    modulus: UInt,
) -> tuple[Qubit, Vector[Qubit], Qubit, Qubit]:
    """Condition a constant modular addition on one qubit.

    Args:
        control (Qubit): Quantum control preserved by the operation.
        target (Vector[Qubit]): Modular target register.
        overflow (Qubit): Clean high-bit workspace restored on return.
        flag (Qubit): Clean modular-reduction flag restored on return.
        addend (UInt): Classical value to add when enabled.
        modulus (UInt): Classical modulus.

    Returns:
        tuple[Qubit, Vector[Qubit], Qubit, Qubit]: Updated control, modular
            target, and restored workspace qubits.
    """
    control_out, target, overflow, flag = _modular_add_const_body(
        target,
        overflow,
        flag,
        addend,
        modulus,
        control,
    )
    assert control_out is not None
    return control_out, target, overflow, flag


def _modular_add_const_modulus_body(
    control: Qubit,
    addend: Vector[Qubit],
    target: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    flag: Qubit,
    modulus: UInt,
) -> tuple[Qubit, Vector[Qubit], Vector[Qubit], Qubit, Qubit, Qubit]:
    """Add a quantum register modulo a classical constant.

    Args:
        control (Qubit): Control for the addend contribution.
        addend (Vector[Qubit]): Quantum addend preserved on return.
        target (Vector[Qubit]): Modular target register.
        carry (Qubit): Clean ripple-carry workspace restored on return.
        overflow (Qubit): Clean high-bit workspace restored on return.
        flag (Qubit): Clean reduction flag restored on return.
        modulus (UInt): Classical modulus.

    Returns:
        tuple[Qubit, Vector[Qubit], Vector[Qubit], Qubit, Qubit, Qubit]:
            Preserved control and addend, modular target, and workspaces.
    """
    control_out, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=False,
    )
    assert control_out is not None
    control = control_out
    _, target, overflow = _apply_const_add(
        target,
        overflow,
        modulus,
        control=None,
        inverse=True,
    )
    overflow, flag = qmc.cx(overflow, flag)
    flag, target, overflow = controlled_add_const(
        flag,
        target,
        overflow,
        modulus,
    )
    control_out, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=True,
    )
    assert control_out is not None
    control = control_out
    overflow = qmc.x(overflow)
    overflow, flag = qmc.cx(overflow, flag)
    overflow = qmc.x(overflow)
    control_out, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=False,
    )
    assert control_out is not None
    control = control_out
    return control, addend, target, carry, overflow, flag


@qmc.composite_gate(name="controlled_modular_add_const_modulus")
def controlled_modular_add_const_modulus(
    control: Qubit,
    addend: Vector[Qubit],
    target: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    flag: Qubit,
    modulus: UInt,
) -> tuple[Qubit, Vector[Qubit], Vector[Qubit], Qubit, Qubit, Qubit]:
    """Condition a quantum-register addition modulo a classical constant.

    Args:
        control (Qubit): Control for the modular addition.
        addend (Vector[Qubit]): Quantum addend preserved on return.
        target (Vector[Qubit]): Modular target register.
        carry (Qubit): Clean carry workspace restored on return.
        overflow (Qubit): Clean high-bit workspace restored on return.
        flag (Qubit): Clean reduction flag restored on return.
        modulus (UInt): Classical modulus.

    Returns:
        tuple[Qubit, Vector[Qubit], Vector[Qubit], Qubit, Qubit, Qubit]:
            Preserved control and addend, modular target, and workspaces.
    """
    return _modular_add_const_modulus_body(
        control,
        addend,
        target,
        carry,
        overflow,
        flag,
        modulus,
    )


def _modular_add_body(
    addend: Vector[Qubit],
    modulus: Vector[Qubit],
    target: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    flag: Qubit,
    control: Qubit | None,
) -> tuple[
    Qubit | None,
    Vector[Qubit],
    Vector[Qubit],
    Vector[Qubit],
    Qubit,
    Qubit,
    Qubit,
]:
    """Apply modular addition and restore every workspace qubit.

    Args:
        addend (Vector[Qubit]): Register containing the value to add.
        modulus (Vector[Qubit]): Register containing the modulus.
        target (Vector[Qubit]): Register updated modulo ``modulus``.
        carry (Qubit): Clean ripple-carry workspace.
        overflow (Qubit): Clean high bit used for subtraction underflow.
        flag (Qubit): Clean flag used for conditional modulus restoration.
        control (Qubit | None): Optional control for the addend contribution.

    Returns:
        tuple[Qubit | None, Vector[Qubit], Vector[Qubit], Vector[Qubit], Qubit,
        Qubit, Qubit]: Updated control, preserved addend and modulus, modular
            target, and restored workspace qubits.
    """
    control, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=False,
    )
    _, modulus, target, carry, overflow = _apply_ripple_carry_add(
        modulus,
        target,
        carry,
        overflow,
        control=None,
        inverse=True,
    )
    overflow, flag = qmc.cx(overflow, flag)

    flag, modulus, target, carry, overflow = qmc.control(ripple_carry_add)(
        flag,
        modulus,
        target,
        carry,
        overflow,
    )

    control, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=True,
    )
    overflow = qmc.x(overflow)
    overflow, flag = qmc.cx(overflow, flag)
    overflow = qmc.x(overflow)
    control, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=False,
    )
    return control, addend, modulus, target, carry, overflow, flag


@qmc.composite_gate(name="modular_add")
def modular_add(
    addend: Vector[Qubit],
    modulus: Vector[Qubit],
    target: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    flag: Qubit,
) -> tuple[
    Vector[Qubit],
    Vector[Qubit],
    Vector[Qubit],
    Qubit,
    Qubit,
    Qubit,
]:
    """Add one register into another modulo a preserved modulus register.

    All registers are equally sized and little-endian. Inputs must encode
    ``addend < modulus`` and ``target < modulus``. The three workspace qubits
    must start in ``|0>`` and are restored before returning.

    Args:
        addend (Vector[Qubit]): Value to add, preserved on return.
        modulus (Vector[Qubit]): Modulus value, preserved on return.
        target (Vector[Qubit]): Value updated to ``(target + addend) % modulus``.
        carry (Qubit): Clean ripple-carry workspace.
        overflow (Qubit): Clean high bit for underflow detection.
        flag (Qubit): Clean conditional-restoration flag.

    Returns:
        tuple[Vector[Qubit], Vector[Qubit], Vector[Qubit], Qubit, Qubit, Qubit]:
            Preserved addend and modulus, modular sum, and restored workspaces.
    """
    _, addend, modulus, target, carry, overflow, flag = _modular_add_body(
        addend,
        modulus,
        target,
        carry,
        overflow,
        flag,
        None,
    )
    return addend, modulus, target, carry, overflow, flag


@qmc.composite_gate(name="controlled_modular_add")
def controlled_modular_add(
    control: Qubit,
    addend: Vector[Qubit],
    modulus: Vector[Qubit],
    target: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    flag: Qubit,
) -> tuple[
    Qubit,
    Vector[Qubit],
    Vector[Qubit],
    Vector[Qubit],
    Qubit,
    Qubit,
    Qubit,
]:
    """Condition a modular addition on one qubit.

    Args:
        control (Qubit): Control qubit preserved by the operation.
        addend (Vector[Qubit]): Value to add when ``control`` is one.
        modulus (Vector[Qubit]): Modulus value, preserved on return.
        target (Vector[Qubit]): Modular accumulator register.
        carry (Qubit): Clean ripple-carry workspace.
        overflow (Qubit): Clean high bit for underflow detection.
        flag (Qubit): Clean conditional-restoration flag.

    Returns:
        tuple[Qubit, Vector[Qubit], Vector[Qubit], Vector[Qubit], Qubit, Qubit,
        Qubit]: Preserved control and constants, conditionally updated target,
            and restored workspaces.
    """
    control_out, addend, modulus, target, carry, overflow, flag = _modular_add_body(
        addend,
        modulus,
        target,
        carry,
        overflow,
        flag,
        control,
    )
    assert control_out is not None
    return control_out, addend, modulus, target, carry, overflow, flag


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


def _carry_xor_const(
    target: Vector[Qubit],
    dirty: Vector[Qubit],
    value: UInt,
    control: Qubit | None,
) -> tuple[Qubit | None, Vector[Qubit], Vector[Qubit]]:
    """XOR right-shifted addition carries into dirty workspace.

    This is the carry-xor block from the constant-workspace classical-quantum
    adder. ``dirty`` is restored by applying this block twice around the phase
    corrections generated by carry venting.

    Args:
        target (Vector[Qubit]): Low bits of the post-addition target.
        dirty (Vector[Qubit]): Dirty carry workspace to toggle.
        value (UInt): Classical offset used by the addition.
        control (Qubit | None): Optional coherent control for the offset.

    Returns:
        tuple[Qubit | None, Vector[Qubit], Vector[Qubit]]: Updated handles.
    """
    dirty_size = get_size(dirty)
    for index in range(dirty_size - 1, 0, -1):
        bit = (value // (2**index)) % 2
        source = target[index]
        previous = dirty[index - 1]
        current = dirty[index]
        control, previous, source, current = _apply_offset_parity_ccx(
            control,
            previous,
            source,
            current,
            bit,
        )
        target[index] = source
        dirty[index - 1] = previous
        dirty[index] = current

    for index in range(dirty_size):
        bit = (value // (2**index)) % 2
        current = dirty[index]
        control, current = _apply_offset_x(control, current, bit)
        dirty[index] = current

    bit = value % 2
    source = target[0]
    current = dirty[0]
    control, source, current = _apply_offset_parity_and(
        control,
        source,
        current,
        bit,
    )
    target[0] = source
    dirty[0] = current

    for index in range(1, dirty_size):
        bit = (value // (2**index)) % 2
        source = target[index]
        previous = dirty[index - 1]
        current = dirty[index]
        control, source, previous, current = _apply_two_offset_parities_ccx(
            control,
            source,
            previous,
            current,
            bit,
        )
        target[index] = source
        dirty[index - 1] = previous
        dirty[index] = current
    return control, target, dirty


def _dirty_const_add_extended(
    target: Vector[Qubit],
    overflow: Qubit,
    dirty: Vector[Qubit],
    clean_first: Qubit,
    clean_second: Qubit,
    value: UInt,
    control: Qubit | None,
) -> tuple[
    Qubit | None,
    Vector[Qubit],
    Qubit,
    Vector[Qubit],
    Qubit,
    Qubit,
]:
    """Add a classical value with two clean and linear dirty workspace.

    Implements the carry-venting construction of Craig Gidney, "A
    Classical-Quantum Adder with Constant Workspace and Linear Gates"
    (arXiv:2507.23079), specialized to an ``overflow || target`` register.
    X-basis carry measurements are corrected coherently, so the operation is
    valid inside controlled modular arithmetic.

    Args:
        target (Vector[Qubit]): Little-endian low target bits.
        overflow (Qubit): Most-significant target bit.
        dirty (Vector[Qubit]): At least ``len(target) - 1`` dirty qubits,
            restored exactly.
        clean_first (Qubit): First clean streaming-carry qubit.
        clean_second (Qubit): Second clean streaming-carry qubit.
        value (UInt): Classical value to add modulo ``2**(len(target) + 1)``.
        control (Qubit | None): Optional coherent control for the addition.

    Returns:
        tuple[Qubit | None, Vector[Qubit], Qubit, Vector[Qubit], Qubit,
        Qubit]: Updated control and target with all workspace restored.

    Raises:
        ValueError: If the target is too small or dirty workspace is short.
    """
    target_size = get_size(target)
    dirty_size = get_size(dirty)
    if target_size < 2:
        raise ValueError("carry-venting constant addition requires two low bits.")
    if dirty_size < target_size - 1:
        raise ValueError(
            "carry-venting constant addition requires len(target) - 1 dirty qubits."
        )
    dirty_root = dirty
    dirty = dirty_root[: target_size - 1]

    for index in range(target_size):
        bit = (value // (2**index)) % 2
        target_bit = target[index]
        control, target_bit = _apply_offset_x(control, target_bit, bit)
        target[index] = target_bit
    overflow_bit = (value // (2**target_size)) % 2
    control, overflow = _apply_offset_x(control, overflow, overflow_bit)

    vents: list[qmc.Bit] = []
    for index in range(target_size):
        bit = (value // (2**index)) % 2
        source = target[index]
        if index == 0:
            next_carry = clean_second
            control, source, next_carry = _apply_offset_and(
                control,
                source,
                next_carry,
                bit,
            )
            clean_second = next_carry
        else:
            current = clean_second if index % 2 == 1 else clean_first
            next_is_overflow = index == target_size - 1
            next_carry = (
                overflow
                if next_is_overflow
                else (clean_first if index % 2 == 1 else clean_second)
            )
            control, source, current, next_carry = _apply_offset_parity_ccx(
                control,
                source,
                current,
                next_carry,
                bit,
            )
            current, source = qmc.cx(current, source)
            dirty_slot = dirty[index - 1]
            current, dirty_slot = qmc.cx(current, dirty_slot)
            dirty[index - 1] = dirty_slot
            current = qmc.h(current)
            current, vent = qmc.measure_reset(current)
            vents.append(vent)
            if index % 2 == 1:
                clean_second = current
            else:
                clean_first = current
            if next_is_overflow:
                overflow = next_carry
            elif index % 2 == 1:
                clean_first = next_carry
            else:
                clean_second = next_carry
        target[index] = source
        if index < target_size - 1:
            next_carry = clean_first if index % 2 == 1 else clean_second
            control, next_carry = _apply_offset_x(control, next_carry, bit)
            if index % 2 == 1:
                clean_first = next_carry
            else:
                clean_second = next_carry
        else:
            control, overflow = _apply_offset_x(control, overflow, bit)

    target = qmc.x(target)
    overflow = qmc.x(overflow)
    for index, vent in enumerate(vents):
        dirty_bit = dirty[index]
        dirty_bit = _phase_flip_if(vent, dirty_bit)
        dirty[index] = dirty_bit
    control, target, dirty = _carry_xor_const(target, dirty, value, control)
    for index, vent in enumerate(vents):
        dirty_bit = dirty[index]
        dirty_bit = _phase_flip_if(vent, dirty_bit)
        dirty[index] = dirty_bit
    target = qmc.x(target)
    overflow = qmc.x(overflow)
    dirty_root[: target_size - 1] = dirty
    return control, target, overflow, dirty_root, clean_first, clean_second


def modmul_const(
    reg: Vector[Qubit],
    *,
    multiplier: int | UInt,
    modulus: int | UInt,
    window_size: int = 2,
    inverse_multiplier: int | UInt | None = None,
    control: Qubit | None = None,
) -> Vector[Qubit] | tuple[Qubit, Vector[Qubit]]:
    """Apply constant modular multiplication ``|x> -> |a*x mod N>``.

    The standard implementation uses lookup windows and reversible modular
    additions. Resource estimation walks this same executable body; it never
    substitutes an external arithmetic cost formula. Basis states ``x >= N``
    are left unchanged so the operation is a unitary permutation over the full
    register space. The FTQC body contains measurement and reset operations;
    condition it through the ``control`` argument rather than wrapping the
    complete operation with :func:`qamomile.circuit.control`.

    Args:
        reg (Vector[Qubit]): Little-endian register to multiply in place. Its
            width must be known when the qkernel is traced.
        multiplier (int | UInt): Positive multiplier ``a``.
        modulus (int | UInt): Modulus ``N``.
        window_size (int): Lookup address width. Defaults to 2.
        inverse_multiplier (int | UInt | None): Multiplicative inverse of
            ``multiplier`` modulo ``modulus``. Python integer inputs compute it
            automatically. Symbolic inputs must provide it explicitly. Defaults
            to ``None``.
        control (Qubit | None): Optional control qubit. When provided the
            multiplication is applied conditionally (as Shor's order finding
            conditions each modular multiplication on an exponent qubit).
            Defaults to ``None``.

    Returns:
        Vector[Qubit] | tuple[Qubit, Vector[Qubit]]: The register after modular
        multiplication, or ``(control, register)`` when a control qubit is
        supplied.

    Raises:
        ValueError: If concrete constants are invalid or symbolic constants omit
            ``inverse_multiplier``.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import modmul_const
        >>> @qmc.qkernel
        ... def mul(reg: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        ...     return modmul_const(reg, multiplier=2, modulus=15)
    """
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if isinstance(multiplier, int) and isinstance(modulus, int):
        if modulus < 2 or multiplier <= 0 or math.gcd(multiplier, modulus) != 1:
            raise ValueError("multiplier must be positive and coprime to modulus.")
        if inverse_multiplier is None:
            inverse_multiplier = pow(multiplier, -1, modulus)
    elif inverse_multiplier is None:
        raise ValueError("symbolic constants require inverse_multiplier.")
    assert inverse_multiplier is not None

    if isinstance(modulus, int):
        try:
            register_size = get_size(reg)
        except ValueError:
            register_size = None
        if register_size is not None and modulus >= 2**register_size:
            raise ValueError(
                f"modulus={modulus} does not fit in a {register_size}-qubit "
                "register; require modulus < 2**register_size."
            )

    try:
        size = get_size(reg)
    except ValueError as exc:
        raise ValueError(
            "modmul_const requires a concrete register width; specialize the "
            "enclosing kernel before applying FTQC arithmetic."
        ) from exc
    accumulator = qmc.qubit_array(size, name="modmul_accumulator")
    lookup = qmc.qubit_array(size, name="modmul_lookup")
    address = qmc.qubit_array(window_size, name="modmul_address")
    carry = qmc.qubit("modmul_carry")
    vent = qmc.qubit("modmul_vent")
    overflow = qmc.qubit("modmul_overflow")
    flag = qmc.qubit("modmul_flag")
    domain = qmc.qubit("modmul_domain")
    enable = qmc.qubit("modmul_enable")
    internal_control = control if control is not None else qmc.qubit("modmul_control")
    if control is None:
        internal_control = qmc.x(internal_control)
    multiplier_value = (
        qmc.uint(multiplier) if isinstance(multiplier, int) else multiplier
    )
    inverse_multiplier_value = (
        qmc.uint(inverse_multiplier)
        if isinstance(inverse_multiplier, int)
        else inverse_multiplier
    )
    modulus_value = qmc.uint(modulus) if isinstance(modulus, int) else modulus
    internal_control, reg, _, _, _, _, _, _, _, _, _ = _modmul_const_body(
        internal_control,
        reg,
        accumulator,
        lookup,
        address,
        carry,
        vent,
        overflow,
        flag,
        domain,
        enable,
        multiplier_value,
        inverse_multiplier_value,
        modulus_value,
        size,
        window_size,
    )
    if control is None:
        qmc.x(internal_control)
        return reg
    return internal_control, reg


@qmc.qkernel
def lookup_xor(
    address: Vector[Qubit],
    target: Vector[Qubit],
    scale: UInt,
    modulus: UInt,
) -> tuple[Vector[Qubit], Vector[Qubit]]:
    """XOR a modular multiplication lookup into a clean target register.

    The table maps ``j`` to ``(scale * j) % modulus``. Its body is expressed
    entirely in Qamomile operations, so resource estimation counts the actual
    unary-iteration lookup network rather than an opaque table-cost formula.

    Args:
        address (Vector[Qubit]): Little-endian lookup address, preserved.
        target (Vector[Qubit]): Register XORed with the selected table value.
        scale (UInt): Classical scale factor applied to each address.
        modulus (UInt): Classical modulus applied to each table value.

    Returns:
        tuple[Vector[Qubit], Vector[Qubit]]: Preserved address and updated
            lookup target.
    """
    selected_x = qmc.control(_xor_bit, num_controls=address.shape[0])
    for candidate in qmc.range(2 ** address.shape[0]):
        for address_index in qmc.range(address.shape[0]):
            candidate_bit = (candidate // (2**address_index)) % 2
            flip = 1 - candidate_bit
            exact_x = qmc.global_phase(_xor_bit, 0.5 * math.pi * flip)
            address[address_index] = exact_x(address[address_index], flip)
        table_value = (scale * candidate) % modulus
        for target_index in qmc.range(target.shape[0]):
            table_bit = (table_value // (2**target_index)) % 2
            address, target[target_index] = selected_x(
                address,
                target[target_index],
                table_bit,
                global_phase=0.5 * math.pi * table_bit,
            )
        for address_index in qmc.range(address.shape[0]):
            candidate_bit = (candidate // (2**address_index)) % 2
            flip = 1 - candidate_bit
            exact_x = qmc.global_phase(_xor_bit, 0.5 * math.pi * flip)
            address[address_index] = exact_x(address[address_index], flip)
    return address, target


def _controlled_modular_add_const_modulus_dirty(
    control: Qubit,
    addend: Vector[Qubit],
    target: Vector[Qubit],
    dirty: Vector[Qubit],
    carry: Qubit,
    vent: Qubit,
    overflow: Qubit,
    flag: Qubit,
    modulus: UInt,
    register_size: int,
) -> tuple[
    Qubit,
    Vector[Qubit],
    Vector[Qubit],
    Vector[Qubit],
    Qubit,
    Qubit,
    Qubit,
    Qubit,
]:
    """Add a quantum lookup register modulo a classical constant linearly.

    Quantum addition uses the Cuccaro ripple network. Constant subtraction
    and conditional restoration use the carry-venting adder while borrowing
    the multiplication source register as dirty workspace. Consequently every
    modular addition has linear gate cost without another n-qubit register.

    Args:
        control (Qubit): Control for the addend contribution.
        addend (Vector[Qubit]): Quantum lookup addend, preserved.
        target (Vector[Qubit]): Modular accumulator.
        dirty (Vector[Qubit]): Multiplication source borrowed as dirty space.
        carry (Qubit): Clean ripple and venting workspace.
        vent (Qubit): Second clean venting workspace.
        overflow (Qubit): Extended-register high bit.
        flag (Qubit): Clean modular-reduction flag.
        modulus (UInt): Classical modulus.
        register_size (int): Concrete width of the low registers.

    Returns:
        tuple[Qubit, Vector[Qubit], Vector[Qubit], Vector[Qubit], Qubit,
        Qubit, Qubit, Qubit]: Updated accumulator with all controls, addends,
            dirty workspace, and clean ancillas restored.
    """
    control_out, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=False,
    )
    assert control_out is not None
    control = control_out
    negative_modulus = (2 ** (register_size + 1) - modulus) % (2 ** (register_size + 1))
    _, target, overflow, dirty, carry, vent = _dirty_const_add_extended(
        target,
        overflow,
        dirty,
        carry,
        vent,
        negative_modulus,
        None,
    )
    overflow, flag = qmc.cx(overflow, flag)
    flag_out, target, overflow, dirty, carry, vent = _dirty_const_add_extended(
        target,
        overflow,
        dirty,
        carry,
        vent,
        modulus,
        flag,
    )
    assert flag_out is not None
    flag = flag_out
    control_out, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=True,
    )
    assert control_out is not None
    control = control_out
    overflow = qmc.x(overflow)
    overflow, flag = qmc.cx(overflow, flag)
    overflow = qmc.x(overflow)
    control_out, addend, target, carry, overflow = _apply_ripple_carry_add(
        addend,
        target,
        carry,
        overflow,
        control=control,
        inverse=False,
    )
    assert control_out is not None
    return (
        control_out,
        addend,
        target,
        dirty,
        carry,
        vent,
        overflow,
        flag,
    )


def _modmul_const_body(
    control: Qubit,
    register: Vector[Qubit],
    accumulator: Vector[Qubit],
    lookup: Vector[Qubit],
    address_workspace: Vector[Qubit],
    carry: Qubit,
    vent: Qubit,
    overflow: Qubit,
    flag: Qubit,
    domain: Qubit,
    enable: Qubit,
    multiplier: UInt,
    inverse_multiplier: UInt,
    modulus: UInt,
    register_size: int,
    window_size: int,
) -> tuple[
    Qubit,
    Vector[Qubit],
    Vector[Qubit],
    Vector[Qubit],
    Vector[Qubit],
    Qubit,
    Qubit,
    Qubit,
    Qubit,
    Qubit,
    Qubit,
]:
    """Multiply modulo a constant using windowed lookup additions.

    Args:
        control (Qubit): Control for the complete multiplication.
        register (Vector[Qubit]): Value transformed in place.
        accumulator (Vector[Qubit]): Clean multiplication accumulator.
        lookup (Vector[Qubit]): Clean lookup output register.
        address_workspace (Vector[Qubit]): Clean fixed-width lookup address.
        carry (Qubit): Clean ripple-carry workspace.
        vent (Qubit): Second clean carry-venting workspace.
        overflow (Qubit): Clean extended-adder high bit.
        flag (Qubit): Clean modular-reduction flag.
        domain (Qubit): Clean valid-domain flag.
        enable (Qubit): Clean conjunction workspace.
        multiplier (UInt): Classical modular multiplier.
        inverse_multiplier (UInt): Modular inverse of ``multiplier``.
        modulus (UInt): Classical modulus.
        register_size (int): Concrete number of modular-value bits.
        window_size (int): Number of source bits per lookup.

    Returns:
        tuple[Qubit, Vector[Qubit], Vector[Qubit], Vector[Qubit],
        Vector[Qubit], Qubit, Qubit, Qubit, Qubit, Qubit, Qubit]: Multiplied
            register and restored workspaces.
    """
    negative_modulus = (2 ** (register_size + 1) - modulus) % (2 ** (register_size + 1))
    (
        _,
        register,
        overflow,
        accumulator,
        carry,
        vent,
    ) = _dirty_const_add_extended(
        register,
        overflow,
        accumulator,
        carry,
        vent,
        negative_modulus,
        None,
    )
    overflow, domain = qmc.cx(overflow, domain)
    _, register, overflow, accumulator, carry, vent = _dirty_const_add_extended(
        register,
        overflow,
        accumulator,
        carry,
        vent,
        modulus,
        None,
    )

    enable_window = qmc.control(qmc.x, num_controls=2)
    window_count = (register_size + window_size - 1) // window_size
    for window_index in range(window_count):
        offset = window_index * window_size
        stop = min(offset + window_size, register_size)
        width = stop - offset
        for address_index in range(width):
            source = register[offset + address_index]
            source, address_workspace[address_index] = qmc.cx(
                source, address_workspace[address_index]
            )
            register[offset + address_index] = source
        scale = (multiplier * (2**offset)) % modulus
        address_workspace, lookup = lookup_xor(
            address_workspace, lookup, scale, modulus
        )
        control, domain, enable = enable_window(control, domain, enable)
        (
            enable,
            lookup,
            accumulator,
            register,
            carry,
            vent,
            overflow,
            flag,
        ) = _controlled_modular_add_const_modulus_dirty(
            enable,
            lookup,
            accumulator,
            register,
            carry,
            vent,
            overflow,
            flag,
            modulus,
            register_size,
        )
        control, domain, enable = enable_window(control, domain, enable)
        address_workspace, lookup = lookup_xor(
            address_workspace, lookup, scale, modulus
        )
        for address_index in range(width):
            source = register[offset + address_index]
            source, address_workspace[address_index] = qmc.cx(
                source, address_workspace[address_index]
            )
            register[offset + address_index] = source

    controlled_swap = qmc.control(qmc.swap, num_controls=2)
    for index in range(register_size):
        control, domain, register[index], accumulator[index] = controlled_swap(
            control, domain, register[index], accumulator[index]
        )

    for window_index in range(window_count):
        offset = window_index * window_size
        stop = min(offset + window_size, register_size)
        width = stop - offset
        for address_index in range(width):
            source = register[offset + address_index]
            source, address_workspace[address_index] = qmc.cx(
                source, address_workspace[address_index]
            )
            register[offset + address_index] = source
        scale = (modulus - inverse_multiplier * (2**offset)) % modulus
        address_workspace, lookup = lookup_xor(
            address_workspace, lookup, scale, modulus
        )
        control, domain, enable = enable_window(control, domain, enable)
        (
            enable,
            lookup,
            accumulator,
            register,
            carry,
            vent,
            overflow,
            flag,
        ) = _controlled_modular_add_const_modulus_dirty(
            enable,
            lookup,
            accumulator,
            register,
            carry,
            vent,
            overflow,
            flag,
            modulus,
            register_size,
        )
        control, domain, enable = enable_window(control, domain, enable)
        address_workspace, lookup = lookup_xor(
            address_workspace, lookup, scale, modulus
        )
        for address_index in range(width):
            source = register[offset + address_index]
            source, address_workspace[address_index] = qmc.cx(
                source, address_workspace[address_index]
            )
            register[offset + address_index] = source

    _, register, overflow, accumulator, carry, vent = _dirty_const_add_extended(
        register,
        overflow,
        accumulator,
        carry,
        vent,
        negative_modulus,
        None,
    )
    overflow, domain = qmc.cx(overflow, domain)
    _, register, overflow, accumulator, carry, vent = _dirty_const_add_extended(
        register,
        overflow,
        accumulator,
        carry,
        vent,
        modulus,
        None,
    )
    return (
        control,
        register,
        accumulator,
        lookup,
        address_workspace,
        carry,
        vent,
        overflow,
        flag,
        domain,
        enable,
    )


__all__ = [
    "add_const",
    "controlled_add_const",
    "controlled_modular_add",
    "controlled_modular_add_const",
    "controlled_modular_add_const_modulus",
    "lookup_xor",
    "modular_add",
    "modular_add_const",
    "modmul_const",
    "ripple_carry_add",
]
