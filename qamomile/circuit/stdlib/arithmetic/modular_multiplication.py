"""Implement constant modular multiplication primitives."""

from __future__ import annotations

import math

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector
from qamomile.circuit.frontend.handle.utils import get_size

from .bitwise import _xor_bit
from .carry_venting import _dirty_const_add_extended
from .ripple_carry import _apply_ripple_carry_add


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
