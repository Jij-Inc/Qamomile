"""Implement modular addition primitives."""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector

from .constant import _apply_const_add, controlled_add_const
from .ripple_carry import _apply_ripple_carry_add, ripple_carry_add


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
