"""Functional little-endian modular arithmetic."""

from __future__ import annotations

import qamomile as qm
from qamomile._native import compiler as native

from ._primitives import flip
from ._structure import compose_unary, lift_indices


def add_constant(*, width: int, value: int) -> qm.QKernel:
    """Construct ``|j> -> |j + value mod 2**width>`` compositionally.

    The constant is reduced modulo the register domain.  Each set bit applies
    the existing increment function to one little-endian suffix, so source and
    resource size scale with register width rather than with ``abs(value)``.

    Args:
        width: Positive register width.
        value: Compile-time signed integer constant.

    Returns:
        A source-backed unary transform over ``Q[Bits[width]]``.

    Raises:
        TypeError: If ``width`` or ``value`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    _validate_integer(value, name="value")
    offsets = _native_wrapping_offsets(width, value)
    steps = tuple(_shifted_increment(width=width, offset=offset) for offset in offsets)
    return compose_unary(width=width, steps=steps)


def subtract_constant(*, width: int, value: int) -> qm.QKernel:
    """Construct exact subtraction as the adjoint of constant addition.

    Args:
        width: Positive register width.
        value: Compile-time signed integer constant.

    Returns:
        A compiler-derived unary inverse over ``Q[Bits[width]]``.

    Raises:
        TypeError: If ``width`` or ``value`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    addition = add_constant(width=width, value=value)

    @qm.qkernel
    def subtract(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Subtract the configured constant from an owned register.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned value register.

        Returns:
            qm.Q[qm.Bits[width]]: Register after modular subtraction.
        """

        return qm.adjoint(addition, state)

    return subtract


def add_registers(*, width: int) -> qm.QKernel:
    """Construct an abstract ``|a>|b> -> |a>|b + a mod 2**width>`` transform.

    The input is one ``Q[Bits[2 * width]]`` whose first little-endian half is
    the retained addend and whose second half is the target. Rust retains one
    semantic addition until target emission chooses an exact realization.

    Args:
        width (int): Positive width of each register half.

    Returns:
        qm.QKernel: A source-backed unary transform over both registers.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    total = 2 * width

    @qm.qkernel
    def addition(state: qm.Q[qm.Bits[total]]) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Add the retained low half into the high half.

        Args:
            state (qm.Q[qm.Bits[total]]): Equal little-endian operand registers.

        Returns:
            qm.Q[qm.Bits[total]]: Retained addend and wrapping target sum.
        """

        return qm.add_into(state)

    return addition


def subtract_registers(*, width: int) -> qm.QKernel:
    """Construct register subtraction as the adjoint of register addition.

    Args:
        width: Positive width of each register half.

    Returns:
        The exact ``|a>|b> -> |a>|b - a mod 2**width>`` transform.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    addition = add_registers(width=width)
    total = 2 * width

    @qm.qkernel
    def subtract(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Subtract the retained addend from the target register.

        Args:
            state (qm.Q[qm.Bits[total]]): Addend and target registers.

        Returns:
            qm.Q[qm.Bits[total]]: Retained addend and updated target.
        """

        return qm.adjoint(addition, state)

    return subtract


def compare_unsigned(*, width: int) -> qm.QKernel:
    """Compare two unsigned registers without exposing clean workspace.

    The input contains ``left`` followed by ``right``. The result retains both
    registers and appends one bit equal to ``left < right``. The body delegates
    the exact comparison to the first-class QCore operation.

    Args:
        width (int): Positive width of each operand.

    Returns:
        qm.QKernel: A ``Q[Bits[2 * width]] -> Q[Bits[2 * width + 1]]``
        isometry.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    if width > 4:
        raise ValueError("workspace-free comparison currently supports width <= 4")
    input_width = 2 * width
    result_width = input_width + 1

    @qm.qkernel
    def compare(
        operands: qm.Q[qm.Bits[input_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[result_width]]:  # type: ignore[valid-type]
        """Return retained operands followed by their comparison bit.

        Args:
            operands (qm.Q[qm.Bits[input_width]]): Left and right operands.

        Returns:
            qm.Q[qm.Bits[result_width]]: Operands followed by ``left < right``.
        """

        left, right = qm.split_at(operands, width)
        flag = qm.as_uint(qm.qubits(1))
        flag, left, right = qm.less_than_xor(flag, qm.as_uint(left), qm.as_uint(right))
        return qm.concat(qm.as_bits(left), qm.as_bits(right), qm.as_bits(flag))

    return compare


def multiply_add(*, width: int) -> qm.QKernel:
    """Construct reversible multiplication into a retained accumulator.

    The transform acts on ``Q[Bits[4 * width]]``. The first two
    ``width``-component little-endian registers are the retained factors and
    the final ``2 * width`` components are the product accumulator. It maps
    ``|a>|b>|p>`` to ``|a>|b>|p + a * b mod 2**(2 * width)>``. A zero
    accumulator therefore receives the exact product of the two factors.

    The body delegates the retained multiply-accumulate transition to QCore.

    Args:
        width (int): Positive width of each factor register.

    Returns:
        qm.QKernel: Reversible multiply-accumulate transform over the retained
            factors and double-width accumulator.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    total = 4 * width

    @qm.qkernel
    def multiply(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Update the accumulator while retaining both packed factors.

        Args:
            state (qm.Q[qm.Bits[total]]): Factors followed by accumulator.

        Returns:
            qm.Q[qm.Bits[total]]: Factors and updated accumulator.
        """

        left, tail = qm.split_at(state, width)
        right, product = qm.split_at(tail, width)
        product, left, right = qm.mul_add(
            qm.as_uint(product), qm.as_uint(left), qm.as_uint(right)
        )
        return qm.concat(qm.as_bits(left), qm.as_bits(right), qm.as_bits(product))

    return multiply


def less_than_registers(*, width: int) -> qm.QKernel:
    """Construct an unsigned coherent less-than comparison.

    The transform acts on ``Q[Bits[2 * width + 2]]`` laid out as a left
    little-endian operand, a right little-endian operand, one output flag, and
    one clean work component. It maps the output flag to
    ``flag xor (left < right)`` while preserving both operands and restoring
    the work component to zero.

    Comparison is expressed as controlled subtraction into an extended copy
    of the left operand, copying its temporary sign component to the output,
    and applying the common adjoint. It introduces no comparator opcode,
    backend branch, or resource-analysis rule.

    Args:
        width: Positive width of each unsigned operand.

    Returns:
        A source-backed comparator over two operands, an output flag, and one
        clean work component.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    total = 2 * width + 2
    comparison = compose_unary(
        width=total,
        steps=tuple(
            _comparison_subtract_bit(width=width, index=index) for index in range(width)
        ),
    )
    flag = 2 * width
    work = flag + 1

    @qm.qkernel
    def less_than(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Toggle the output flag when the left operand is smaller.

        Args:
            state (qm.Q[qm.Bits[total]]): Operands, output flag, and clean work
                component.

        Returns:
            qm.Q[qm.Bits[total]]: Preserved operands, comparison output, and
            restored work component.
        """

        state = comparison(state)
        state = qm.cx(state, control=work, target=flag)
        return qm.adjoint(comparison, state)

    return less_than


def gf2_multiply_add(*, width: int, modulus: int) -> qm.QKernel:
    """Construct polynomial multiply-add over a binary quotient ring.

    The transform acts on ``Q[Bits[3 * width]]`` laid out as two retained
    little-endian polynomial-coefficient registers and one accumulator. It
    maps ``|a>|b>|c>`` to ``|a>|b>|c xor (a * b mod modulus)>``. When
    ``modulus`` is irreducible, the quotient ring is the finite field
    :math:`GF(2^{width})`.

    ``modulus`` uses the usual integer bit representation and must be monic of
    degree ``width`` with a nonzero constant coefficient. Host preprocessing
    reduces each product monomial once; the quantum body is only a bounded
    sequence of ordinary Toffoli gates. The transform is its own inverse and
    introduces no field-arithmetic opcode or backend rule.

    Args:
        width: Positive coefficient width of each field element.
        modulus: Degree-``width`` binary modulus polynomial.

    Returns:
        A source-backed multiply-add transform over two factors and an
        accumulator.

    Raises:
        TypeError: If ``width`` or ``modulus`` is not an integer.
        ValueError: If the width or modulus representation is invalid.
    """

    _validate_width(width)
    _validate_integer(modulus, name="modulus")
    if 3 * width > 512:
        raise ValueError("GF(2^m) arithmetic exceeds the 512-qubit budget")
    if modulus >> width != 1 or modulus & 1 == 0:
        raise ValueError(
            "modulus must be monic of degree width with a nonzero constant"
        )
    steps = tuple(
        _gf2_product_coefficient(
            width=width,
            left_index=left_index,
            right_index=right_index,
            output_index=output_index,
        )
        for left_index in range(width)
        for right_index in range(width)
        for output_index in _gf2_monomial_support(
            degree=left_index + right_index,
            width=width,
            modulus=modulus,
        )
    )
    return compose_unary(width=3 * width, steps=steps)


def has_duplicates_sorted(*, length: int, item_width: int) -> qm.QKernel:
    """Construct a clean duplicate test for one sorted register list.

    The transform acts on a flattened little-endian list followed by one
    output flag, ``length - 1`` clean comparison flags, and one clean
    comparison component. It toggles the output flag exactly when the sorted
    list contains a duplicate, preserves every item, and restores all work
    components to zero.

    Each adjacent pair is compared with :func:`less_than_registers`. A single
    multi-controlled flip recognizes a strictly increasing list, and the
    comparisons are then uncomputed. The construction adds no list opcode,
    target rule, or second inverse implementation.

    Args:
        length: Number of items in the sorted list. Must be at least two.
        item_width: Positive bit width of each unsigned item.

    Returns:
        A source-backed duplicate test over the list, output, and clean work
        components.

    Raises:
        TypeError: If either argument is not an integer.
        ValueError: If ``length`` is smaller than two or ``item_width`` is not
            positive.
    """

    _validate_integer(length, name="length")
    _validate_width(item_width)
    if length < 2:
        raise ValueError("length must be at least two")

    data_width = length * item_width
    output = data_width
    comparison_flags = tuple(range(output + 1, output + length))
    comparison_work = output + length
    total = comparison_work + 1
    comparator = less_than_registers(width=item_width)
    comparisons = tuple(
        lift_indices(
            total_width=total,
            indices=(
                *range(index * item_width, (index + 1) * item_width),
                *range((index + 1) * item_width, (index + 2) * item_width),
                comparison_flags[index],
                comparison_work,
            ),
            transform=comparator,
        )
        for index in range(length - 1)
    )
    inverse_comparisons = comparisons[::-1]
    recognize_unique = lift_indices(
        total_width=total,
        indices=(*comparison_flags, output),
        transform=_all_controls_flip(control_width=length - 1),
    )
    toggle_output = lift_indices(
        total_width=total,
        indices=(output,),
        transform=flip,
    )
    return compose_unary(
        width=total,
        steps=(
            *comparisons,
            toggle_output,
            recognize_unique,
            *inverse_comparisons,
        ),
    )


def add_modulo(*, width: int) -> qm.QKernel:
    """Add one retained register modulo another over the complete basis.

    The public state contains only ``addend``, ``modulus``, and ``target``.
    When the modulus is nonzero and both values are below it, the target is
    updated to ``(target + addend) % modulus``. Every other basis state is an
    identity state, making the operation a unique unitary without a domain
    promise or caller-provided workspace.

    Args:
        width (int): Width of each unsigned register, currently at most three.

    Returns:
        qm.QKernel: A total reversible transform over three registers.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is outside the bounded synthesis domain.
    """

    _validate_width(width)
    if width > 3:
        raise ValueError(
            "total register-modular synthesis currently supports width <= 3"
        )
    total = 3 * width

    @qm.qkernel
    def addition(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Add the retained register modulo the retained modulus.

        Args:
            state (qm.Q[qm.Bits[total]]): Addend, modulus, and target.

        Returns:
            qm.Q[qm.Bits[total]]: Retained inputs and updated target.
        """

        addend, tail = qm.split_at(state, width)
        modulus, target = qm.split_at(tail, width)
        target, addend, modulus = qm.mod_add(
            qm.as_uint(target), qm.as_uint(addend), qm.as_uint(modulus)
        )
        return qm.concat(qm.as_bits(addend), qm.as_bits(modulus), qm.as_bits(target))

    return addition


def subtract_modulo(*, width: int) -> qm.QKernel:
    """Subtract one retained register modulo another over the complete basis.

    Args:
        width (int): Width accepted by :func:`add_modulo`.

    Returns:
        qm.QKernel: Exact adjoint of the total modular addition permutation.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is outside the bounded synthesis domain.
    """

    addition = add_modulo(width=width)
    total = 3 * width

    @qm.qkernel
    def subtract(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Apply the exact inverse modular permutation.

        Args:
            state (qm.Q[qm.Bits[total]]): Addend, modulus, and target.

        Returns:
            qm.Q[qm.Bits[total]]: Inverse-updated semantic registers.
        """

        return qm.adjoint(addition, state)

    return subtract


def add_register_modulo(*, width: int) -> qm.QKernel:
    """Construct addition by a retained register modulo a retained register.

    The transform acts on ``Q[Bits[3 * width + 3]]``. The first three
    little-endian registers are the addend, modulus, and target. They are
    followed by a clean source-extension component, a clean target-overflow
    component, and a clean reduction flag. For inputs satisfying
    ``addend < modulus`` and ``target < modulus``, the target becomes
    ``(target + addend) % modulus`` and every other component is restored.

    Fixed tensor permutations lift the existing register adder onto the
    required components. The compare-subtract-restore body therefore adds no
    register-modulus operation, inverse implementation, backend branch, or
    resource-analysis rule.

    Args:
        width: Positive width of each retained value register.

    Returns:
        A source-backed transform over three registers and three clean work
        components.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    extended_width = width + 1
    total = 3 * width + 3
    source_extension = 3 * width
    overflow = source_extension + 1
    flag = overflow + 1
    addend = (*range(width), source_extension)
    modulus = (*range(width, 2 * width), source_extension)
    target = (*range(2 * width, 3 * width), overflow)
    addition = add_registers(width=extended_width)
    subtraction = subtract_registers(width=extended_width)
    add_addend = lift_indices(
        total_width=total,
        indices=(*addend, *target),
        transform=addition,
    )
    subtract_addend = lift_indices(
        total_width=total,
        indices=(*addend, *target),
        transform=subtraction,
    )
    subtract_modulus = lift_indices(
        total_width=total,
        indices=(*modulus, *target),
        transform=subtraction,
    )
    restore_modulus = lift_indices(
        total_width=total,
        indices=(flag, *modulus, *target),
        transform=_controlled_unary(width=2 * extended_width, transform=addition),
    )

    @qm.qkernel
    def modular_add(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Add the retained addend modulo the retained modulus.

        Args:
            state (qm.Q[qm.Bits[total]]): Value registers and clean workspace.

        Returns:
            qm.Q[qm.Bits[total]]: Updated target with workspace restored.
        """

        state = add_addend(state)
        state = subtract_modulus(state)
        state = qm.cx(state, control=overflow, target=flag)
        state = restore_modulus(state)
        state = subtract_addend(state)
        state = qm.x(state, target=overflow)
        state = qm.cx(state, control=overflow, target=flag)
        state = qm.x(state, target=overflow)
        return add_addend(state)

    return modular_add


def subtract_register_modulo(*, width: int) -> qm.QKernel:
    """Construct register-modulus subtraction as the common adjoint.

    Args:
        width: Positive width of each retained value register.

    Returns:
        The exact inverse of :func:`add_register_modulo`.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    addition = add_register_modulo(width=width)
    total = 3 * width + 3

    @qm.qkernel
    def subtract(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Subtract the retained addend modulo the retained modulus.

        Args:
            state (qm.Q[qm.Bits[total]]): Value registers and clean workspace.

        Returns:
            qm.Q[qm.Bits[total]]: Updated target with workspace restored.
        """

        return qm.adjoint(addition, state)

    return subtract


def _add_constant_modulo_with_work(
    *,
    width: int,
    value: int,
    modulus: int,
) -> qm.QKernel:
    """Construct an internal reversible modular-addition work circuit.

    The transform acts on ``Q[Bits[width + 2]]`` laid out as one leading
    reduction flag, ``width`` little-endian value components, and one trailing
    overflow component. The two workspace components must start in ``|0>``
    when the value is in ``[0, modulus)`` and are restored before return.

    The body is the usual reversible compare-subtract-restore construction,
    expressed only through constant addition, coherent control, and ordinary
    calls. The supplied signed value is reduced modulo ``modulus``.

    Args:
        width (int): Positive modular-value width.
        value (int): Compile-time signed constant to add.
        modulus (int): Classical modulus fitting strictly inside the register
            domain.

    Returns:
        qm.QKernel: Source-backed transform over the value and two clean work
            components.

    Raises:
        TypeError: If an argument is not an integer or the native plan is
            unavailable or malformed.
        ValueError: If ``width`` or ``modulus`` is outside its valid domain.
    """

    _validate_width(width)
    _validate_integer(value, name="value")
    _validate_integer(modulus, name="modulus")
    normalized = _native_modular_residue(width, value, modulus)
    total = width + 2
    if normalized == 0:
        return compose_unary(width=total, steps=())

    extended_width = width + 1
    add_value = _lift_suffix(
        total_width=total,
        offset=1,
        transform=add_constant(width=extended_width, value=normalized),
    )
    subtract_value = _lift_suffix(
        total_width=total,
        offset=1,
        transform=subtract_constant(width=extended_width, value=normalized),
    )
    subtract_modulus = _lift_suffix(
        total_width=total,
        offset=1,
        transform=subtract_constant(width=extended_width, value=modulus),
    )
    restore_modulus = _controlled_suffix_addition(
        total_width=total,
        value=modulus,
    )
    overflow = total - 1

    @qm.qkernel
    def modular_add(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Add the configured constant modulo the configured modulus.

        Args:
            state (qm.Q[qm.Bits[total]]): Value register and clean workspace.

        Returns:
            qm.Q[qm.Bits[total]]: Updated value with workspace restored.
        """

        state = add_value(state)
        state = subtract_modulus(state)
        state = qm.cx(state, control=overflow, target=0)
        state = restore_modulus(state)
        state = subtract_value(state)
        state = qm.x(state, target=overflow)
        state = qm.cx(state, control=overflow, target=0)
        state = qm.x(state, target=overflow)
        return add_value(state)

    return modular_add


def _subtract_constant_modulo_with_work(
    *,
    width: int,
    value: int,
    modulus: int,
) -> qm.QKernel:
    """Construct an internal modular-subtraction work circuit.

    Args:
        width (int): Positive modular-value width.
        value (int): Compile-time signed constant to subtract.
        modulus (int): Classical modulus fitting strictly inside the register
            domain.

    Returns:
        qm.QKernel: Exact inverse over the modular value and two work
            components.

    Raises:
        TypeError: If an argument is not an integer or the native plan is
            unavailable or malformed.
        ValueError: If ``width`` or ``modulus`` is outside its valid domain.
    """

    addition = _add_constant_modulo_with_work(
        width=width,
        value=value,
        modulus=modulus,
    )
    total = width + 2

    @qm.qkernel
    def modular_subtract(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Subtract the configured constant modulo the configured modulus.

        Args:
            state (qm.Q[qm.Bits[total]]): Value register and clean workspace.

        Returns:
            qm.Q[qm.Bits[total]]: Updated value with workspace restored.
        """

        return qm.adjoint(addition, state)

    return modular_subtract


def multiply_constant_modulo(
    *,
    width: int,
    value: int,
    modulus: int,
) -> qm.QKernel:
    """Construct in-place multiplication by a unit modulo ``modulus``.

    The transform acts on ``Q[Bits[2 * width + 2]]``. Its first ``width``
    components contain the little-endian modular value. The remaining
    ``width + 2`` components use the clean-workspace layout accepted by
    :func:`_add_constant_modulo_with_work` and must start in zero. The product replaces
    the first register and every workspace component is restored.

    Multiplication is synthesized by accumulating controlled shifted constants
    into the second register, swapping value registers, and uncomputing the
    old value with the classical modular inverse. It introduces no
    multiplication operation or backend/resource special case.

    Args:
        width: Positive modular-value width.
        value: Compile-time multiplier coprime to ``modulus``.
        modulus: Classical modulus fitting strictly inside the register domain.

    Returns:
        A source-backed reversible multiplication transform.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the modulus is invalid or the multiplier is not a unit.
    """

    _validate_width(width)
    _validate_integer(value, name="value")
    _validate_integer(modulus, name="modulus")
    accumulate_values, erase_values = _native_modular_multiplication_plan(
        width,
        value,
        modulus,
    )
    total = 2 * width + 2
    accumulate = tuple(
        _controlled_modular_accumulation(
            width=width,
            control=index,
            value=shifted,
            modulus=modulus,
        )
        for index, shifted in enumerate(accumulate_values)
    )
    erase = tuple(
        _controlled_modular_accumulation(
            width=width,
            control=index,
            value=shifted,
            modulus=modulus,
            subtract=True,
        )
        for index, shifted in enumerate(erase_values)
    )
    return compose_unary(
        width=total,
        steps=(*accumulate, _swap_modular_registers(width=width), *erase),
    )


def divide_constant_modulo(
    *,
    width: int,
    value: int,
    modulus: int,
) -> qm.QKernel:
    """Construct modular division as the common multiplication adjoint.

    Args:
        width: Positive modular-value width.
        value: Compile-time divisor coprime to ``modulus``.
        modulus: Classical modulus fitting strictly inside the register domain.

    Returns:
        The exact inverse of :func:`multiply_constant_modulo`.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the modulus is invalid or the divisor is not a unit.
    """

    multiplication = multiply_constant_modulo(
        width=width,
        value=value,
        modulus=modulus,
    )
    total = 2 * width + 2

    @qm.qkernel
    def divide(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Divide by the configured unit modulo the configured modulus.

        Args:
            state (qm.Q[qm.Bits[total]]): Value register and clean workspace.

        Returns:
            qm.Q[qm.Bits[total]]: Updated value with workspace restored.
        """

        return qm.adjoint(multiplication, state)

    return divide


def multiply_modulo_constant(
    *,
    width: int,
    value: int,
    modulus: int,
) -> qm.QKernel:
    """Multiply by a modular unit without exposing an accumulator or scratch.

    Inputs below ``modulus`` are multiplied modulo it. Inputs at or above the
    modulus are identity states, so the operation is total on the register.

    Args:
        width (int): Positive value-register width.
        value (int): Multiplier coprime to ``modulus``.
        modulus (int): Positive modulus strictly below ``2**width``.

    Returns:
        qm.QKernel: Total in-place modular multiplication permutation.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the modulus is invalid or the multiplier is not a unit.
    """

    _validate_width(width)
    _validate_integer(value, name="value")
    _validate_integer(modulus, name="modulus")
    if width > 8:
        raise ValueError(
            "total modular-constant synthesis currently supports width <= 8"
        )
    _native_modular_multiplication_plan(width, value, modulus)

    @qm.qkernel
    def multiplication(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply the exact constant modular multiplier to one packed word.

        Args:
            state (qm.Q[qm.Bits[width]]): Value register.

        Returns:
            qm.Q[qm.Bits[width]]: Updated register.
        """

        return qm.as_bits(qm.mod_mul(qm.as_uint(state), factor=value, modulus=modulus))

    return multiplication


def divide_modulo_constant(
    *,
    width: int,
    value: int,
    modulus: int,
) -> qm.QKernel:
    """Divide by a modular unit using the total multiplication permutation.

    Args:
        width (int): Positive value-register width.
        value (int): Divisor coprime to ``modulus``.
        modulus (int): Positive modulus strictly below ``2**width``.

    Returns:
        qm.QKernel: Exact adjoint of :func:`multiply_modulo_constant`.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the modulus is invalid or the divisor is not a unit.
    """

    multiplication = multiply_modulo_constant(
        width=width,
        value=value,
        modulus=modulus,
    )

    @qm.qkernel
    def divide(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply the inverse modular multiplication permutation.

        Args:
            state (qm.Q[qm.Bits[width]]): Modular value register.

        Returns:
            qm.Q[qm.Bits[width]]: Inverse-updated value register.
        """

        return qm.adjoint(multiplication, state)

    return divide


def modular_increment(*, width: int) -> qm.QKernel:
    """Construct ``|j> -> |j + 1 mod 2**width>`` as function composition.

    The returned unary transform interprets logical component zero as the
    least-significant bit. Every carry step is an ordinary coherently
    controlled call; this factory introduces no arithmetic opcode or backend
    rule.

    Args:
        width: Positive register width.

    Returns:
        A source-backed unary transform over ``Q[Bits[width]]``.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    _validate_width(width)
    steps = tuple(
        _controlled_flip(width=width, target=target)
        for target in range(width - 1, 0, -1)
    )
    return compose_unary(width=width, steps=(*steps, _lowest_flip(width=width)))


def modular_decrement(*, width: int) -> qm.QKernel:
    """Construct ``|j> -> |j - 1 mod 2**width>`` as the increment adjoint.

    Args:
        width: Positive register width.

    Returns:
        A compiler-derived exact inverse of ``modular_increment``.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive.
    """

    increment = modular_increment(width=width)

    @qm.qkernel
    def decrement(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Decrement an owned register modulo its width.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned value register.

        Returns:
            qm.Q[qm.Bits[width]]: Decremented register.
        """

        return qm.adjoint(increment, state)

    return decrement


def _controlled_flip(*, width: int, target: int) -> qm.QKernel:
    """Construct one fixed-position carry step without exposing qubit atoms."""

    if target == width - 1:

        @qm.qkernel
        def highest(
            state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Apply a carry-controlled flip to the highest component.

            Args:
                state (qm.Q[qm.Bits[width]]): Owned value register.

            Returns:
                qm.Q[qm.Bits[width]]: Register after the carry step.
            """

            controls, target_state = qm.split_at(state, target)
            controls, target_state = qm.control(flip, controls, target_state)
            return qm.concat(controls, target_state)

        return highest

    prefix_width = target + 1

    @qm.qkernel
    def interior(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply a carry-controlled flip to an interior component.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned value register.

        Returns:
            qm.Q[qm.Bits[width]]: Register after the carry step.
        """

        prefix, suffix = qm.split_at(state, prefix_width)
        controls, target_state = qm.split_at(prefix, target)
        controls, target_state = qm.control(flip, controls, target_state)
        prefix = qm.concat(controls, target_state)
        return qm.concat(prefix, suffix)

    return interior


def _lowest_flip(*, width: int) -> qm.QKernel:
    """Lift the primitive one-bit flip to the least-significant component."""

    if width == 1:
        return flip

    @qm.qkernel
    def lowest(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Flip the least-significant component.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned value register.

        Returns:
            qm.Q[qm.Bits[width]]: Register with its lowest component flipped.
        """

        return qm.x(state, target=0)

    return lowest


def _shifted_increment(*, width: int, offset: int) -> qm.QKernel:
    """Lift increment of bits ``offset:`` into the complete register."""

    increment = modular_increment(width=width - offset)
    return _lift_suffix(total_width=width, offset=offset, transform=increment)


def _comparison_subtract_bit(*, width: int, index: int) -> qm.QKernel:
    """Subtract one right-operand bit from the extended left operand."""

    total = 2 * width + 2
    right_index = width + index
    work = total - 1
    subtraction = subtract_constant(width=width + 1, value=1 << index)
    controlled = _controlled_unary(width=width + 1, transform=subtraction)
    return lift_indices(
        total_width=total,
        indices=(right_index, *range(width), work),
        transform=controlled,
    )


def _gf2_product_coefficient(
    *,
    width: int,
    left_index: int,
    right_index: int,
    output_index: int,
) -> qm.QKernel:
    """Toggle one reduced product coefficient under two factor bits."""

    total = 3 * width

    @qm.qkernel
    def toggle(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Accumulate one binary polynomial product coefficient.

        Args:
            state (qm.Q[qm.Bits[total]]): Two factors and the product accumulator.

        Returns:
            qm.Q[qm.Bits[total]]: Retained factors and updated accumulator.
        """

        return qm.ccx(
            state,
            control1=left_index,
            control2=width + right_index,
            target=2 * width + output_index,
        )

    return toggle


def _gf2_monomial_support(
    *,
    degree: int,
    width: int,
    modulus: int,
) -> tuple[int, ...]:
    """Return coefficient indices of one monomial reduced modulo ``modulus``."""

    remainder = 1 << degree
    while remainder.bit_length() - 1 >= width:
        shift = remainder.bit_length() - width - 1
        remainder ^= modulus << shift
    return tuple(index for index in range(width) if remainder & (1 << index))


def _controlled_suffix_addition(
    *,
    total_width: int,
    value: int,
) -> qm.QKernel:
    """Control addition on the suffix by the leading one-bit flag."""

    addition = add_constant(width=total_width - 1, value=value)
    return _controlled_unary(width=total_width - 1, transform=addition)


def _controlled_unary(*, width: int, transform: qm.QKernel) -> qm.QKernel:
    """Control one unary transform with a leading coherent component."""

    total = width + 1

    @qm.qkernel
    def controlled(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Apply the configured transform under a leading control.

        Args:
            state (qm.Q[qm.Bits[total]]): Control and target state.

        Returns:
            qm.Q[qm.Bits[total]]: Retained control and updated target.
        """

        flag, target = qm.split_at(state, 1)
        flag, target = qm.control(transform, flag, target)
        return qm.concat(flag, target)

    return controlled


def _all_controls_flip(*, control_width: int) -> qm.QKernel:
    """Construct a target flip controlled by an all-one register."""

    total = control_width + 1

    @qm.qkernel
    def controlled_flip(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Flip the trailing target when every leading control is one.

        Args:
            state (qm.Q[qm.Bits[total]]): Controls followed by the target.

        Returns:
            qm.Q[qm.Bits[total]]: Retained controls and conditionally flipped
            target.
        """

        controls, target = qm.split_at(state, control_width)
        controls, target = qm.control(flip, controls, target)
        return qm.concat(controls, target)

    return controlled_flip


def _controlled_modular_accumulation(
    *,
    width: int,
    control: int,
    value: int,
    modulus: int,
    subtract: bool = False,
) -> qm.QKernel:
    """Control one modular accumulator update by a multiplicand component."""

    operation = (
        _subtract_constant_modulo_with_work(width=width, value=value, modulus=modulus)
        if subtract
        else _add_constant_modulo_with_work(width=width, value=value, modulus=modulus)
    )
    total = 2 * width + 2

    @qm.qkernel
    def controlled_accumulate(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Apply one controlled modular accumulator update.

        Args:
            state (qm.Q[qm.Bits[total]]): Multiplicand and accumulator state.

        Returns:
            qm.Q[qm.Bits[total]]: Retained multiplicand and updated accumulator.
        """

        multiplicand, accumulator = qm.split_at(state, width)
        multiplicand, accumulator = qm.control(
            operation,
            multiplicand,
            accumulator,
            at=control,
        )
        return qm.concat(multiplicand, accumulator)

    return controlled_accumulate


def _swap_modular_registers(*, width: int) -> qm.QKernel:
    """Swap the value register with the accumulator's value components."""

    total = 2 * width + 2

    @qm.qkernel
    def swap_registers(
        state: qm.Q[qm.Bits[total]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Swap the modular value and accumulator value registers.

        Args:
            state (qm.Q[qm.Bits[total]]): Value and accumulator state.

        Returns:
            qm.Q[qm.Bits[total]]: State with the value registers exchanged.
        """

        for index in range(width):
            state = qm.swap(state, left=index, right=width + 1 + index)
        return state

    return swap_registers


def _lift_suffix(
    *,
    total_width: int,
    offset: int,
    transform: qm.QKernel,
) -> qm.QKernel:
    """Lift one unary transform over a fixed suffix of an owned register."""

    if offset == 0:
        return transform

    @qm.qkernel
    def lifted(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the configured transform to a fixed suffix.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Owned complete register.

        Returns:
            qm.Q[qm.Bits[total_width]]: Register with its suffix transformed.
        """

        prefix, suffix = qm.split_at(state, offset)
        suffix = transform(suffix)
        return qm.concat(prefix, suffix)

    return lifted


def _fixed_x(*, width: int, target: int) -> qm.QKernel:
    """Construct one fixed single-component X transform.

    Args:
        width (int): Register width.
        target (int): Component to flip.

    Returns:
        qm.QKernel: Unary fixed-target X transform.
    """

    @qm.qkernel
    def transform(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Flip the configured component.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned coherent bits.

        Returns:
            qm.Q[qm.Bits[width]]: Updated coherent bits.
        """

        return qm.x(state, target=target)

    return transform


def _validate_width(width: object) -> None:
    """Validate one host-side arithmetic width."""

    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")


def _validate_integer(value: object, *, name: str) -> None:
    """Reject Boolean and noninteger host configuration values."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _native_wrapping_offsets(width: int, value: int) -> tuple[int, ...]:
    """Cross the required native wrapping-addition planning boundary."""

    derive = getattr(native, "wrapping_addition_offsets", None)
    if derive is None:
        raise TypeError("Q7636: native wrapping-addition planning is unavailable")
    projection = derive(str(width), str(value))
    if (
        not isinstance(projection, list)
        or any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in projection
        )
        or projection != sorted(set(projection))
        or any(offset < 0 or offset >= width for offset in projection)
    ):
        raise TypeError("Q7637: malformed native wrapping-addition plan")
    return tuple(projection)


def _native_modular_residue(width: int, value: int, modulus: int) -> int:
    """Cross the required native modular-addition planning boundary."""

    derive = getattr(native, "modular_addition_residue", None)
    if derive is None:
        raise TypeError("Q7636: native modular-addition planning is unavailable")
    residue = _decimal_integer(
        derive(str(width), str(value), str(modulus)),
        error="Q7637: malformed native modular-addition plan",
    )
    if residue < 0 or residue >= modulus:
        raise TypeError("Q7637: malformed native modular-addition plan")
    return residue


def _native_modular_multiplication_plan(
    width: int,
    value: int,
    modulus: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Cross the required native modular-multiplication planning boundary."""

    derive = getattr(native, "plan_modular_multiplication", None)
    if derive is None:
        raise TypeError("Q7636: native modular-multiplication planning is unavailable")
    projection = derive(str(width), str(value), str(modulus))
    try:
        projected_width = projection.width
        projected_modulus = _decimal_integer(projection.modulus, error="")
        multiplier = _decimal_integer(projection.multiplier, error="")
        inverse = _decimal_integer(projection.inverse, error="")
        accumulate = _decimal_integer_rows(projection.accumulate)
        erase = _decimal_integer_rows(projection.erase)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "Q7637: malformed native modular-multiplication plan"
        ) from error
    if (
        isinstance(projected_width, bool)
        or not isinstance(projected_width, int)
        or projected_width != width
        or projected_modulus != modulus
        or not 0 <= multiplier < modulus
        or not 0 <= inverse < modulus
        or multiplier * inverse % modulus != 1
        or len(accumulate) != width
        or len(erase) != width
        or any(not 0 <= residue < modulus for residue in (*accumulate, *erase))
    ):
        raise TypeError("Q7637: malformed native modular-multiplication plan")
    return accumulate, erase


def _decimal_integer(value: object, *, error: str) -> int:
    """Decode one canonical unsigned decimal integer projection."""

    if not isinstance(value, str) or not value.isdecimal():
        raise TypeError(error)
    return int(value)


def _decimal_integer_rows(value: object) -> tuple[int, ...]:
    """Decode one list of canonical unsigned decimal integer projections."""

    if not isinstance(value, list):
        raise TypeError
    return tuple(_decimal_integer(component, error="") for component in value)


__all__ = [
    "add_constant",
    "add_modulo",
    "add_register_modulo",
    "add_registers",
    "compare_unsigned",
    "divide_modulo_constant",
    "divide_constant_modulo",
    "gf2_multiply_add",
    "has_duplicates_sorted",
    "less_than_registers",
    "modular_decrement",
    "modular_increment",
    "multiply_add",
    "multiply_modulo_constant",
    "multiply_constant_modulo",
    "subtract_constant",
    "subtract_modulo",
    "subtract_register_modulo",
    "subtract_registers",
]
