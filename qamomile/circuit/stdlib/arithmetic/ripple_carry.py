"""Implement reversible ripple-carry addition primitives."""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, Vector
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
