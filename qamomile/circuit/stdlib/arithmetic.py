"""Provide executable constant modular-arithmetic building blocks."""

from __future__ import annotations

import math
from typing import Any

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.operation.callable import CallPolicy

_ModularQKernel = QKernel[..., Vector[Qubit]]


def _is_zero_length_register(q: qmc.Vector[qmc.Qubit]) -> bool:
    """Return whether ``q`` has a concrete zero length.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register handle to inspect.

    Returns:
        bool: ``True`` when the register shape is concretely known to be
            zero; otherwise ``False``.
    """
    if not q.shape:
        return False
    dim = q.shape[0]
    if isinstance(dim, int):
        return dim == 0
    value = getattr(dim, "value", None)
    return bool(
        value is not None and value.is_constant() and int(value.get_const()) == 0
    )


class _CheckedModularQKernel:
    """Reject empty registers before delegating to a modular qkernel.

    Args:
        kernel (_ModularQKernel): Underlying qkernel that implements the
            non-empty modular arithmetic primitive.
        op_name (str): User-facing operation name used in error messages
            and callable metadata.
    """

    def __init__(self, kernel: _ModularQKernel, op_name: str) -> None:
        """Initialize the checked qkernel wrapper.

        Args:
            kernel (_ModularQKernel): Underlying qkernel to delegate to.
            op_name (str): User-facing operation name.
        """
        self._kernel = kernel
        self._op_name = op_name
        self.name = op_name
        if hasattr(kernel, "name"):
            kernel.name = op_name
        self.__name__ = op_name
        self.__doc__ = getattr(kernel, "__doc__", None)

    def __getattr__(self, name: str) -> Any:
        """Forward qkernel attributes to the wrapped kernel.

        Args:
            name (str): Attribute name to retrieve.

        Returns:
            Any: Attribute value from the wrapped qkernel.

        Raises:
            AttributeError: If the wrapped qkernel does not expose the
                requested attribute.
        """
        return getattr(self._kernel, name)

    def __call__(self, q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the wrapped primitive after validating ``q``.

        Args:
            q (qmc.Vector[qmc.Qubit]): Qubit register to update.

        Returns:
            qmc.Vector[qmc.Qubit]: Updated qubit register.

        Raises:
            ValueError: If ``q`` is a concrete zero-length register.
        """
        if _is_zero_length_register(q):
            raise ValueError(
                f"{self._op_name} requires at least one qubit; "
                f"got a zero-length register."
            )
        return self._kernel(q)


@qmc.qkernel
def _modular_increment(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular increment ``|j> -> |j + 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to increment in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    for k in qmc.range(1, n):
        target_index = n - k
        mcx = qmc.control(qmc.x, num_controls=target_index)
        q[0:target_index], q[target_index] = mcx(q[0:target_index], q[target_index])
    q[0] = qmc.x(q[0])
    return q


@qmc.qkernel
def _modular_decrement(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply the modular decrement ``|j> -> |j - 1 mod 2^n>``.

    The qubit vector is interpreted in little-endian order: ``q[0]`` is
    the least-significant bit of the encoded integer.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to decrement in place.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    n = q.shape[0]
    q[0] = qmc.x(q[0])
    for target_index in qmc.range(1, n):
        mcx = qmc.control(qmc.x, num_controls=target_index)
        q[0:target_index], q[target_index] = mcx(q[0:target_index], q[target_index])
    return q


def _apply_fixed_window_periodic_shift(
    system: qmc.Vector[qmc.Qubit],
    start: int,
    width: int,
    shift: int,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a constant modular shift to one fixed window of a flat register.

    Each set bit ``i`` of the shift magnitude is implemented by one
    little-endian increment or decrement ladder on the window starting at bit
    ``i``. Such a ladder changes the complete axis value by ``2**i`` modulo
    ``2**width``. Addressing the parent vector directly avoids constructing a
    nested symbolic slice while a concrete axis view is borrowed by a caller.

    Args:
        system (qmc.Vector[qmc.Qubit]): Flattened parent register to update.
        start (int): First qubit index of the target window.
        width (int): Positive number of qubits in the target window.
        shift (int): Signed modular displacement. Positive values increment
            and negative values decrement. Multiples of ``2**width`` are the
            identity.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated flattened parent register.
    """
    increment = shift > 0
    magnitude = abs(shift) % (1 << width)
    for bit_index in range(width):
        if not magnitude & (1 << bit_index):
            continue

        subwindow_start = start + bit_index
        subwindow_width = width - bit_index
        if not increment:
            system[subwindow_start] = qmc.x(system[subwindow_start])

        target_indices = (
            range(subwindow_width - 1, 0, -1)
            if increment
            else range(1, subwindow_width)
        )
        for target_index in target_indices:
            operands = tuple(
                system[subwindow_start + index] for index in range(target_index + 1)
            )
            results = qmc.control(qmc.x, num_controls=target_index)(*operands)
            for index, result in enumerate(results):
                system[subwindow_start + index] = result

        if increment:
            system[subwindow_start] = qmc.x(system[subwindow_start])
    return system


modular_increment = _CheckedModularQKernel(
    _modular_increment,
    "modular_increment",
)
modular_decrement = _CheckedModularQKernel(
    _modular_decrement,
    "modular_decrement",
)


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


@qmc.composite_gate(name="modmul_const")
def _modmul_const_body(
    control: Qubit,
    register: Vector[Qubit],
    accumulator: Vector[Qubit],
    addend: Vector[Qubit],
    modulus_register: Vector[Qubit],
    carry: Qubit,
    overflow: Qubit,
    flag: Qubit,
    domain: Qubit,
    enable: Qubit,
    multiplier: UInt,
    inverse_multiplier: UInt,
    modulus: UInt,
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
]:
    """Multiply one register modulo a classical constant and clean workspace.

    The operation applies multiplication on basis states below ``modulus`` and
    acts as the identity on states at or above ``modulus``, yielding a unitary
    permutation over the register's complete computational basis.

    Args:
        control (Qubit): Control qubit for the complete multiplication.
        register (Vector[Qubit]): Value transformed in place.
        accumulator (Vector[Qubit]): Clean output/uncomputation register.
        addend (Vector[Qubit]): Clean constant-addend workspace.
        modulus_register (Vector[Qubit]): Clean modulus workspace.
        carry (Qubit): Clean adder carry workspace.
        overflow (Qubit): Clean adder overflow workspace.
        flag (Qubit): Clean modular-reduction flag workspace.
        domain (Qubit): Clean input-domain flag workspace.
        enable (Qubit): Clean conjunction workspace.
        multiplier (UInt): Classical multiplier.
        inverse_multiplier (UInt): Multiplicative inverse modulo ``modulus``.
        modulus (UInt): Classical modulus.

    Returns:
        tuple[Qubit, Vector[Qubit], Vector[Qubit], Vector[Qubit],
        Vector[Qubit], Qubit, Qubit, Qubit, Qubit, Qubit]: Preserved control,
            multiplied register, and restored workspace handles.
    """
    size = register.shape[0]
    modulus_register = _xor_constant(modulus_register, modulus)
    _, modulus_register, register, carry, overflow = _apply_ripple_carry_add(
        modulus_register,
        register,
        carry,
        overflow,
        control=None,
        inverse=True,
    )
    overflow, domain = qmc.cx(overflow, domain)
    _, modulus_register, register, carry, overflow = _apply_ripple_carry_add(
        modulus_register,
        register,
        carry,
        overflow,
        control=None,
        inverse=False,
    )
    enable_term = qmc.control(qmc.x, num_controls=3)
    for source_index in qmc.range(size):
        addend_value = (multiplier * (2**source_index)) % modulus
        addend = _xor_constant(addend, addend_value)
        control, domain, register[source_index], enable = enable_term(
            control,
            domain,
            register[source_index],
            enable,
        )
        (
            enable,
            addend,
            modulus_register,
            accumulator,
            carry,
            overflow,
            flag,
        ) = controlled_modular_add(
            enable,
            addend,
            modulus_register,
            accumulator,
            carry,
            overflow,
            flag,
        )
        control, domain, register[source_index], enable = enable_term(
            control,
            domain,
            register[source_index],
            enable,
        )
        addend = _xor_constant(addend, addend_value)

    controlled_swap = qmc.control(qmc.swap, num_controls=2)
    for index in qmc.range(size):
        control, domain, register[index], accumulator[index] = controlled_swap(
            control,
            domain,
            register[index],
            accumulator[index],
        )

    subtract_modular = qmc.inverse(controlled_modular_add)
    for source_index in qmc.range(size):
        addend_value = (inverse_multiplier * (2**source_index)) % modulus
        addend = _xor_constant(addend, addend_value)
        control, domain, register[source_index], enable = enable_term(
            control,
            domain,
            register[source_index],
            enable,
        )
        (
            enable,
            addend,
            modulus_register,
            accumulator,
            carry,
            overflow,
            flag,
        ) = subtract_modular(
            enable,
            addend,
            modulus_register,
            accumulator,
            carry,
            overflow,
            flag,
        )
        control, domain, register[source_index], enable = enable_term(
            control,
            domain,
            register[source_index],
            enable,
        )
        addend = _xor_constant(addend, addend_value)
    _, modulus_register, register, carry, overflow = _apply_ripple_carry_add(
        modulus_register,
        register,
        carry,
        overflow,
        control=None,
        inverse=True,
    )
    overflow, domain = qmc.cx(overflow, domain)
    _, modulus_register, register, carry, overflow = _apply_ripple_carry_add(
        modulus_register,
        register,
        carry,
        overflow,
        control=None,
        inverse=False,
    )
    modulus_register = _xor_constant(modulus_register, modulus)
    return (
        control,
        register,
        accumulator,
        addend,
        modulus_register,
        carry,
        overflow,
        flag,
        domain,
        enable,
    )


def modmul_const(
    reg: Vector[Qubit],
    *,
    multiplier: int | UInt,
    modulus: int | UInt,
    inverse_multiplier: int | UInt | None = None,
    control: Qubit | None = None,
) -> Vector[Qubit] | tuple[Qubit, Vector[Qubit]]:
    """Apply constant modular multiplication ``|x> -> |a*x mod N>``.

    The implementation uses reversible modular additions and scales
    polynomially with the register width. Resource estimation walks this same
    executable body; it never substitutes an external arithmetic cost formula.
    Basis states ``x >= N`` are left unchanged so the operation is a unitary
    permutation over the full register space.

    Args:
        reg (Vector[Qubit]): Little-endian register to multiply in place. Its
            width must be known when the qkernel is traced.
        multiplier (int | UInt): Positive multiplier ``a``.
        modulus (int | UInt): Modulus ``N``.
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
    if isinstance(multiplier, int) and isinstance(modulus, int):
        if modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {modulus}.")
        if multiplier <= 0 or math.gcd(multiplier, modulus) != 1:
            raise ValueError(
                "multiplier must be a positive integer coprime to modulus; "
                f"got multiplier={multiplier}, modulus={modulus}."
            )
        if inverse_multiplier is None:
            inverse_multiplier = pow(multiplier, -1, modulus)
    elif inverse_multiplier is None:
        raise ValueError(
            "symbolic modular multiplication requires inverse_multiplier so "
            "the reversible workspace can be uncomputed."
        )
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

    size = reg.shape[0]
    accumulator = qmc.qubit_array(size, name="modmul_accumulator")
    addend = qmc.qubit_array(size, name="modmul_addend")
    modulus_register = qmc.qubit_array(size, name="modmul_modulus")
    carry = qmc.qubit("modmul_carry")
    overflow = qmc.qubit("modmul_overflow")
    flag = qmc.qubit("modmul_flag")
    domain = qmc.qubit("modmul_domain")
    enable = qmc.qubit("modmul_enable")
    internal_control = control if control is not None else qmc.qubit("modmul_control")
    if control is None:
        internal_control = qmc.x(internal_control)
    (
        internal_control,
        reg,
        _accumulator,
        _addend,
        _modulus_register,
        _carry,
        _overflow,
        _flag,
        _domain,
        _enable,
    ) = _modmul_const_body(
        internal_control,
        reg,
        accumulator,
        addend,
        modulus_register,
        carry,
        overflow,
        flag,
        domain,
        enable,
        multiplier,
        inverse_multiplier,
        modulus,
    )
    if control is None:
        qmc.x(internal_control)
        return reg
    return internal_control, reg


__all__ = [
    "controlled_modular_add",
    "modular_decrement",
    "modular_increment",
    "modular_add",
    "modmul_const",
    "ripple_carry_add",
]
