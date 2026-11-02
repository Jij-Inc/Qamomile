# ruff: noqa: F821 -- ``n``, ``p``, ``width``, and ``precision`` are dependent shapes.

"""Executable Shor order finding from ordinary functional composition."""

from __future__ import annotations

import hashlib
import math

import qamomile as qm
from qamomile._native import compiler as native

from ._structure import compose_unary, lift_slice
from .arithmetic import multiply_constant_modulo
from .fourier import inverse_qft, qft
from .phase_estimation import phase_estimation
from .state_preparation import amplitude_encoding

_MAX_REGEV_DIMENSIONS = 128
_MAX_EXACT_GAUSSIAN_WIDTH = 20


def modular_multi_exponentiation(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
) -> qm.QKernel:
    """Construct independent modular powers into one shared product.

    The callable is an isometry from packed coordinate registers to the same
    coordinates followed by a newly allocated modular product initialized to
    one. Its body is the baseline realization: one semantic ``mod_pow`` per
    coordinate. Because the complete computation is body-backed, a
    :class:`qamomile.RealizationOptions` mapping may replace it as one verified
    callable without changing callers.

    Args:
        bases (tuple[int, ...]): Nonempty modular units, one per coordinate.
        modulus (int): Positive classical modulus greater than two.
        coordinate_width (int): Positive width of every exponent coordinate.

    Returns:
        qm.QKernel: Isometry from coordinates to coordinates plus their shared
            modular product.

    Raises:
        TypeError: If an argument has the wrong host type.
        ValueError: If the dimensions or modular units are invalid.
    """

    target_width = _validate_multi_exponentiation_configuration(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
    )
    configuration = repr((bases, modulus, coordinate_width)).encode()
    digest = hashlib.sha256(configuration).hexdigest()[:16]
    function_name = f"modular_multi_exponentiation_{digest}"
    source = _modular_multi_exponentiation_source(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
        target_width=target_width,
        function_name=function_name,
    )
    return qm.QKernel(
        module_name=f"qamomile.stdlib.modular_multi_exponentiation_{digest}",
        function_name=function_name,
        source_text=source,
        display_name="modular_multi_exponentiation.py",
    )


def product_tree_multi_exponentiation(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
) -> qm.QKernel:
    """Construct product-tree multi-exponentiation with repeated squaring.

    For each coordinate bit from most to least significant, the implementation
    forms the selected product of the small bases in a balanced ordinary
    multiplication tree, squares the running modular value, and multiplies by
    that selected product. The final value is copied out before the complete
    computation is uncomputed, so all history and arithmetic workspace is
    clean. Modular products use the common ``mod_mul_add`` semantic operation;
    its schoolbook/ripple realization therefore supplies circuit gates,
    makespan, and workspace without a Regev-specific resource rule.

    Args:
        bases (tuple[int, ...]): Nonempty small modular units. Their complete
            product must be below ``modulus`` so the ordinary product-tree root
            is already a standard residue.
        modulus (int): Positive classical modulus greater than two.
        coordinate_width (int): Positive width of every exponent coordinate.

    Returns:
        qm.QKernel: Isometry with the same ABI and denotation as
            :func:`modular_multi_exponentiation`.

    Raises:
        TypeError: If an argument has the wrong host type.
        ValueError: If the configuration violates a dimension, residue, or
            bounded-source-generation requirement.
    """

    target_width = _validate_multi_exponentiation_configuration(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
    )
    if len(bases) == 1:
        product = _regev_modular_power_step(
            coordinate_total=coordinate_width,
            coordinate_width=coordinate_width,
            target_width=target_width,
            coordinate=0,
            base=bases[0],
            modulus=modulus,
        )
        coordinate_total = coordinate_width
        total_width = coordinate_total + target_width

        @qm.qkernel
        def single_leaf_product_tree(
            coordinates: qm.Q[qm.Bits[coordinate_total]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Use the one-leaf degeneration of the product tree.

            Args:
                coordinates (qm.Q[qm.Bits[coordinate_total]]): Sole exponent
                    coordinate.

            Returns:
                qm.Q[qm.Bits[total_width]]: Coordinate and modular power.
            """

            target = qm.x(qm.qubits(target_width), target=0)
            return product(qm.concat(coordinates, target))

        return single_leaf_product_tree
    padded_count = 1 << (len(bases) - 1).bit_length()
    leaf_width = max(1, max(base.bit_length() for base in bases))
    product_width = padded_count * leaf_width
    if math.prod(bases) >= modulus:
        raise ValueError("the complete selected-base product must be below modulus")
    if product_width > target_width:
        raise ValueError("the product-tree root must fit the modular register width")
    source = _compact_product_tree_multi_exponentiation_source(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
        target_width=target_width,
        padded_count=padded_count,
        leaf_width=leaf_width,
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return qm.QKernel(
        module_name=f"qamomile.stdlib.product_tree_multi_exponentiation_{digest}",
        function_name="product_tree_multi_exponentiation",
        source_text=source,
        display_name="product_tree_multi_exponentiation.py",
    )


def _modular_multi_exponentiation_source(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
    target_width: int,
    function_name: str,
) -> str:
    """Generate the independent-modular-power baseline source.

    Args:
        bases (tuple[int, ...]): Checked modular units.
        modulus (int): Checked classical modulus.
        coordinate_width (int): Width of every packed coordinate.
        target_width (int): Width of the retained modular product.
        function_name (str): Configuration-unique public symbol.

    Returns:
        str: Complete source for the body-backed baseline callable.
    """

    coordinate_total = len(bases) * coordinate_width
    total_width = coordinate_total + target_width
    lines = ["import qamomile as qm", ""]
    for coordinate, base in enumerate(bases):
        start = coordinate * coordinate_width
        suffix_width = coordinate_total - start - coordinate_width
        lines.extend(
            [
                "@qm.qkernel",
                f"def power_{coordinate}(state: qm.Q[qm.Bits[{total_width}]]) -> qm.Q[qm.Bits[{total_width}]]:",
                f"    coordinates, target = qm.split_at(state, {coordinate_total})",
            ]
        )
        if start:
            lines.append(f"    prefix, selected = qm.split_at(coordinates, {start})")
        else:
            lines.append("    selected = coordinates")
        if suffix_width:
            lines.append(
                f"    exponent, suffix = qm.split_at(selected, {coordinate_width})"
            )
        else:
            lines.append("    exponent = selected")
        lines.extend(
            [
                f"    target, exponent = qm.mod_pow(qm.as_uint(target), qm.as_uint(exponent), base={base}, modulus={modulus})",
                "    exponent = qm.as_bits(exponent)",
            ]
        )
        coordinate_parts = [
            *(["prefix"] if start else []),
            "exponent",
            *(["suffix"] if suffix_width else []),
        ]
        lines.extend(
            [
                "    coordinates = qm.concat(" + ", ".join(coordinate_parts) + ")"
                if len(coordinate_parts) > 1
                else "    coordinates = exponent",
                "    return qm.concat(coordinates, qm.as_bits(target))",
                "",
            ]
        )
    lines.extend(
        [
            "@qm.qkernel",
            f"def {function_name}(coordinates: qm.Q[qm.Bits[{coordinate_total}]]) -> qm.Q[qm.Bits[{total_width}]]:",
            f"    target = qm.x(qm.qubits({target_width}), target=0)",
            "    state = qm.concat(coordinates, target)",
        ]
    )
    for coordinate in range(len(bases)):
        lines.append(f"    state = power_{coordinate}(state)")
    lines.extend(["    return state", ""])
    return "\n".join(lines)


def _compact_product_tree_multi_exponentiation_source(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
    target_width: int,
    padded_count: int,
    leaf_width: int,
) -> str:
    """Generate a compact repeated-round product-tree source graph.

    One round callable always consumes the most-significant coordinate bit.
    A cyclic coordinate rotation exposes the next bit, allowing the same
    verified body to be called ``coordinate_width`` times without duplicating
    its operations in QCore.

    Args:
        bases (tuple[int, ...]): Checked small modular units.
        modulus (int): Checked classical modulus.
        coordinate_width (int): Width of every packed coordinate.
        target_width (int): Modular register width.
        padded_count (int): Power-of-two leaf count.
        leaf_width (int): Uniform width of each product-tree leaf.

    Returns:
        str: Complete compact Qamomile source.
    """

    coordinate_total = len(bases) * coordinate_width
    values = [(f"value_{index}", target_width) for index in range(coordinate_width + 1)]
    levels: list[list[tuple[str, int]]] = [
        [(f"leaf_{index}", leaf_width) for index in range(padded_count)]
    ]
    level_width = leaf_width
    level_index = 0
    while len(levels[-1]) > 1:
        previous = levels[-1]
        level_width *= 2
        levels.append(
            [
                (f"node_{level_index}_{index}", level_width)
                for index in range(len(previous) // 2)
            ]
        )
        level_index += 1
    root_name, root_width = levels[-1][0]
    padding_width = target_width - root_width
    scratch: list[tuple[str, int]] = [
        ("square_copy", target_width),
        ("square", target_width),
        *(row for level in levels for row in level),
    ]
    if padding_width:
        scratch.append(("product_padding", padding_width))
    scratch_width = sum(width for _, width in scratch)
    work = [*values, *scratch]
    work_width = sum(width for _, width in work)
    compute_width = coordinate_total + work_width
    round_width = coordinate_total + 2 * target_width + scratch_width
    lines = ["import qamomile as qm", ""]
    lines.extend(
        [
            "@qm.qkernel",
            f"def modular_product(state: qm.Q[qm.Bits[{3 * target_width}]]) -> qm.Q[qm.Bits[{3 * target_width}]]:",
            f"    accumulator, tail = qm.split_at(state, {target_width})",
            f"    left, right = qm.split_at(tail, {target_width})",
            f"    accumulator, left, right = qm.mod_mul_add(qm.as_uint(accumulator), qm.as_uint(left), qm.as_uint(right), modulus={modulus})",
            "    return qm.concat(qm.as_bits(accumulator), qm.as_bits(left), qm.as_bits(right))",
            "",
        ]
    )
    for width in sorted({width for level in levels[:-1] for _, width in level}):
        lines.extend(
            [
                "@qm.qkernel",
                f"def multiply_{width}(state: qm.Q[qm.Bits[{4 * width}]]) -> qm.Q[qm.Bits[{4 * width}]]:",
                f"    product, tail = qm.split_at(state, {2 * width})",
                f"    left, right = qm.split_at(tail, {width})",
                "    product, left, right = qm.mul_add(qm.as_uint(product), qm.as_uint(left), qm.as_uint(right))",
                "    return qm.concat(qm.as_bits(product), qm.as_bits(left), qm.as_bits(right))",
                "",
            ]
        )
    lines.extend(
        [
            "@qm.qkernel",
            f"def copy_word(state: qm.Q[qm.Bits[{2 * target_width}]]) -> qm.Q[qm.Bits[{2 * target_width}]]:",
            f"    for index in range({target_width}):",
            f"        state = qm.cx(state, control=index, target={target_width} + index)",
            "    return state",
            "",
            "@qm.qkernel",
            f"def rotate_coordinates(coordinates: qm.Q[qm.Bits[{coordinate_total}]]) -> qm.Q[qm.Bits[{coordinate_total}]]:",
        ]
    )
    for coordinate in range(len(bases)):
        start = coordinate * coordinate_width
        for bit in reversed(range(1, coordinate_width)):
            lines.append(
                f"    coordinates = qm.swap(coordinates, left={start + bit}, right={start + bit - 1})"
            )
    lines.extend(["    return coordinates", ""])
    lines.extend(
        [
            "@qm.qkernel",
            f"def round_step(state: qm.Q[qm.Bits[{round_width}]]) -> qm.Q[qm.Bits[{round_width}]]:",
            f"    coordinates, remainder = qm.split_at(state, {coordinate_total})",
            f"    current, remainder = qm.split_at(remainder, {target_width})",
            f"    following, remainder = qm.split_at(remainder, {target_width})",
        ]
    )
    for index, (name, width) in enumerate(scratch):
        lines.append(
            f"    {name} = remainder"
            if index + 1 == len(scratch)
            else f"    {name}, remainder = qm.split_at(remainder, {width})"
        )
    lines.extend(
        [
            "    pair = copy_word(qm.concat(current, square_copy))",
            f"    current, square_copy = qm.split_at(pair, {target_width})",
            "    square_state = modular_product(qm.concat(square, current, square_copy))",
            f"    square, square_tail = qm.split_at(square_state, {target_width})",
            f"    current, square_copy = qm.split_at(square_tail, {target_width})",
        ]
    )
    selected_bit = coordinate_width - 1
    for leaf_index, (leaf_name, _) in enumerate(levels[0]):
        lines.append(f"    {leaf_name} = qm.x({leaf_name}, target=0)")
        if leaf_index >= len(bases):
            continue
        base = bases[leaf_index]
        coordinate_bit = leaf_index * coordinate_width + selected_bit
        for leaf_bit in range(leaf_width):
            if ((base >> leaf_bit) & 1) == (leaf_bit == 0):
                continue
            lines.extend(
                [
                    f"    selected = qm.concat(coordinates, {leaf_name})",
                    f"    selected = qm.cx(selected, control={coordinate_bit}, target={coordinate_total + leaf_bit})",
                    f"    coordinates, {leaf_name} = qm.split_at(selected, {coordinate_total})",
                ]
            )
    for tree_level in range(1, len(levels)):
        previous = levels[tree_level - 1]
        factor_width = previous[0][1]
        for node_index, (node_name, _) in enumerate(levels[tree_level]):
            left_name = previous[2 * node_index][0]
            right_name = previous[2 * node_index + 1][0]
            lines.extend(
                [
                    f"    multiplied = multiply_{factor_width}(qm.concat({node_name}, {left_name}, {right_name}))",
                    f"    {node_name}, multiplied_tail = qm.split_at(multiplied, {2 * factor_width})",
                    f"    {left_name}, {right_name} = qm.split_at(multiplied_tail, {factor_width})",
                ]
            )
    factor = (
        root_name if not padding_width else f"qm.concat({root_name}, product_padding)"
    )
    lines.extend(
        [
            f"    product_state = modular_product(qm.concat(following, square, {factor}))",
            f"    following, product_tail = qm.split_at(product_state, {target_width})",
            f"    square, factor = qm.split_at(product_tail, {target_width})",
        ]
    )
    lines.append(
        f"    {root_name} = factor"
        if not padding_width
        else f"    {root_name}, product_padding = qm.split_at(factor, {root_width})"
    )
    for tree_level in reversed(range(1, len(levels))):
        previous = levels[tree_level - 1]
        factor_width = previous[0][1]
        for node_index, (node_name, _) in reversed(list(enumerate(levels[tree_level]))):
            left_name = previous[2 * node_index][0]
            right_name = previous[2 * node_index + 1][0]
            lines.extend(
                [
                    f"    multiplied = qm.adjoint(multiply_{factor_width}, qm.concat({node_name}, {left_name}, {right_name}))",
                    f"    {node_name}, multiplied_tail = qm.split_at(multiplied, {2 * factor_width})",
                    f"    {left_name}, {right_name} = qm.split_at(multiplied_tail, {factor_width})",
                ]
            )
    for leaf_index, (leaf_name, _) in reversed(list(enumerate(levels[0]))):
        if leaf_index < len(bases):
            base = bases[leaf_index]
            coordinate_bit = leaf_index * coordinate_width + selected_bit
            for leaf_bit in reversed(range(leaf_width)):
                if ((base >> leaf_bit) & 1) == (leaf_bit == 0):
                    continue
                lines.extend(
                    [
                        f"    selected = qm.concat(coordinates, {leaf_name})",
                        f"    selected = qm.cx(selected, control={coordinate_bit}, target={coordinate_total + leaf_bit})",
                        f"    coordinates, {leaf_name} = qm.split_at(selected, {coordinate_total})",
                    ]
                )
        lines.append(f"    {leaf_name} = qm.x({leaf_name}, target=0)")
    lines.extend(
        [
            "    square_state = qm.adjoint(modular_product, qm.concat(square, current, square_copy))",
            f"    square, square_tail = qm.split_at(square_state, {target_width})",
            f"    current, square_copy = qm.split_at(square_tail, {target_width})",
            "    pair = copy_word(qm.concat(current, square_copy))",
            f"    current, square_copy = qm.split_at(pair, {target_width})",
            "    return qm.concat(coordinates, current, following, "
            + ", ".join(name for name, _ in scratch)
            + ")",
            "",
            "@qm.qkernel",
            f"def compute(state: qm.Q[qm.Bits[{compute_width}]]) -> qm.Q[qm.Bits[{compute_width}]]:",
            f"    coordinates, remainder = qm.split_at(state, {coordinate_total})",
        ]
    )
    for index, (name, width) in enumerate(work):
        lines.append(
            f"    {name} = remainder"
            if index + 1 == len(work)
            else f"    {name}, remainder = qm.split_at(remainder, {width})"
        )
    lines.append("    value_0 = qm.x(value_0, target=0)")
    scratch_names = ", ".join(name for name, _ in scratch)
    for round_index in range(coordinate_width):
        lines.extend(
            [
                "    round_state = round_step(qm.concat(coordinates, "
                f"value_{round_index}, value_{round_index + 1}, {scratch_names}))",
                f"    coordinates, round_tail = qm.split_at(round_state, {coordinate_total})",
                f"    value_{round_index}, round_tail = qm.split_at(round_tail, {target_width})",
                f"    value_{round_index + 1}, round_tail = qm.split_at(round_tail, {target_width})",
            ]
        )
        for index, (name, width) in enumerate(scratch):
            lines.append(
                f"    {name} = round_tail"
                if index + 1 == len(scratch)
                else f"    {name}, round_tail = qm.split_at(round_tail, {width})"
            )
        lines.append("    coordinates = rotate_coordinates(coordinates)")
    lines.extend(
        [
            "    return qm.concat(coordinates, "
            + ", ".join(name for name, _ in work)
            + ")",
            "",
            "@qm.qkernel",
            f"def product_tree_multi_exponentiation(coordinates: qm.Q[qm.Bits[{coordinate_total}]]) -> qm.Q[qm.Bits[{coordinate_total + target_width}]]:",
            f"    work = qm.qubits({work_width})",
            "    computed = compute(qm.concat(coordinates, work))",
            f"    output = qm.qubits({target_width})",
            "    complete = qm.concat(computed, output)",
            f"    for index in range({target_width}):",
            f"        complete = qm.cx(complete, control={coordinate_total + coordinate_width * target_width} + index, target={compute_width} + index)",
            f"    computed, output = qm.split_at(complete, {compute_width})",
            "    cleaned = qm.adjoint(compute, computed)",
            f"    coordinates, work = qm.split_at(cleaned, {coordinate_total})",
            "    return qm.concat(coordinates, output)",
            "",
        ]
    )
    return "\n".join(lines)


def regev_factoring_state(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
    radius: float,
) -> qm.QKernel:
    """Construct Regev's coherent multidimensional factoring sample.

    Each supplied small base ``b_i`` contributes ``a_i = b_i**2 mod modulus``.
    The returned state prepares the exact finite discrete-Gaussian amplitude
    window on every coordinate, computes ``product(a_i**z_i) mod modulus`` in
    one retained target, and applies an exact QFT to each coordinate.

    This is a direct, executable realization of the quantum sampling stage. It
    deliberately composes one existing ``mod_pow`` operation per coordinate;
    it does not claim Regev's shared small-base product optimization. Resource
    projection therefore exposes the cost of the current realization instead
    of attaching an algorithm-specific asymptotic formula.

    Args:
        bases (tuple[int, ...]): Nonempty small modular units ``b_i``.
        modulus (int): Positive modulus represented by the target register.
        coordinate_width (int): Positive width ``log2(D)`` of every coordinate.
        radius (float): Positive Gaussian radius ``R`` satisfying the paper's
            ``2 * sqrt(d) * R <= D < 4 * sqrt(d) * R`` condition.

    Returns:
        qm.QKernel: Zero-argument coherent function returning all coordinates
        followed by the retained modular product target.

    Raises:
        TypeError: If a configuration value has the wrong host type.
        ValueError: If a dimension, modular unit, Gaussian budget, or the
            ``D``--``R`` relation is invalid.
    """

    squared_bases, target_width, coordinate_count = _validate_regev_configuration(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
        radius=radius,
    )
    coordinate_total = coordinate_count * coordinate_width
    total_width = coordinate_total + target_width
    window = _regev_gaussian_window(width=coordinate_width, radius=radius)
    preparation = compose_unary(
        width=coordinate_total,
        steps=tuple(
            lift_slice(
                total_width=coordinate_total,
                start=index * coordinate_width,
                width=coordinate_width,
                transform=window,
            )
            for index in range(coordinate_count)
        ),
    )
    product = modular_multi_exponentiation(
        bases=squared_bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
    )
    coordinate_qft = _fixed_qft(coordinate_width)
    fourier = compose_unary(
        width=coordinate_total,
        steps=tuple(
            lift_slice(
                total_width=coordinate_total,
                start=index * coordinate_width,
                width=coordinate_width,
                transform=coordinate_qft,
            )
            for index in range(coordinate_count)
        ),
    )

    @qm.qkernel
    def sample_state() -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Prepare one configured multidimensional factoring sample.

        Returns:
            qm.Q[qm.Bits[total_width]]: Fourier coordinates followed by the
            retained modular product target.
        """

        coordinates = preparation(qm.qubits(coordinate_total))
        state = product(coordinates)
        coordinates, target = qm.split_at(state, coordinate_total)
        return qm.concat(fourier(coordinates), target)

    return sample_state


def regev_factoring(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
    radius: float,
) -> qm.QKernel:
    """Construct an executable sampler for Regev's quantum procedure.

    The result contains ``len(bases)`` adjacent ``coordinate_width``-bit
    Fourier samples in source order. Classical lattice reconstruction and the
    final greatest-common-divisor step remain host-side research code.

    Args:
        bases (tuple[int, ...]): Nonempty small modular units ``b_i``.
        modulus (int): Positive modulus represented by the target register.
        coordinate_width (int): Positive width ``log2(D)`` of every coordinate.
        radius (float): Positive discrete-Gaussian radius ``R``.

    Returns:
        qm.QKernel: Zero-argument sampled function returning packed coordinates.

    Raises:
        TypeError: If a configuration value has the wrong host type.
        ValueError: If the Regev configuration is invalid.
    """

    _, _, coordinate_count = _validate_regev_configuration(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
        radius=radius,
    )
    result_width = coordinate_count * coordinate_width
    state = regev_factoring_state(
        bases=bases,
        modulus=modulus,
        coordinate_width=coordinate_width,
        radius=radius,
    )

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bits[result_width]]:  # type: ignore[valid-type]
        """Measure the configured Fourier-coordinate registers.

        Returns:
            qm.Shot[qm.Bits[result_width]]: Packed multidimensional sample.
        """

        coordinates, target = qm.split_at(state(), result_width)
        result = qm.measure(coordinates)
        _ignored = qm.measure(target)
        return result

    return sample


def ekera_hastad_factoring_state(
    *,
    generator: int,
    modulus: int,
    parameter_width: int | None = None,
) -> qm.QKernel:
    """Construct the coherent Ekerå–Håstad short-DLP quantum stage.

    For ``n = modulus.bit_length()`` and ``m = ceil(n / 2) + 1``, the returned
    state contains a ``2m``-bit phase estimate for multiplication by ``g``, an
    ``m``-bit estimate for multiplication by ``y**-1``, and the shared modular
    target, where ``y = g**(modulus + 1) mod modulus``. Classical lattice
    post-processing remains an ordinary host function.

    The construction is two ordinary phase-estimation functions over existing
    modular multiplication. It adds no algorithm operation, estimator rule, or
    backend case.

    Args:
        generator (int): Group element coprime to ``modulus``.
        modulus (int): Candidate balanced semiprime greater than two.
        parameter_width (int | None): Positive short-phase width. Defaults to
            ``ceil(n / 2) + 1``; an explicit smaller value is useful for
            finite-precision experiments.

    Returns:
        qm.QKernel: Zero-argument coherent short-DLP function.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the modular maps are not reversible.
    """

    (
        width,
        selected_width,
        long_precision,
        short_precision,
        target_width,
        inverse_y,
    ) = _validate_short_dlp(
        generator=generator,
        modulus=modulus,
        parameter_width=parameter_width,
    )
    long_unitary = multiply_constant_modulo(
        width=width,
        value=generator,
        modulus=modulus,
    )
    short_unitary = multiply_constant_modulo(
        width=width,
        value=inverse_y,
        modulus=modulus,
    )
    total = long_precision + short_precision + target_width
    target_bits = (*_little_endian(1, width), *((0,) * (width + 2)))

    @qm.qkernel
    def short_dlp_state() -> qm.Q[qm.Bits[total]]:  # type: ignore[valid-type]
        """Prepare the configured short discrete-logarithm state.

        Returns:
            qm.Q[qm.Bits[total]]: Coherent phase and modular target registers.
        """

        target = qm.prepare(target_bits)
        long_count, target = phase_estimation(
            long_unitary,
            target,
            long_precision,
        )
        short_count, target = phase_estimation(
            short_unitary,
            target,
            short_precision,
        )
        return qm.concat(long_count, short_count, target)

    return short_dlp_state


def ekera_hastad_factoring(
    *,
    generator: int,
    modulus: int,
    parameter_width: int | None = None,
) -> qm.QKernel:
    """Construct an executable Ekerå–Håstad short-DLP sampler.

    Args:
        generator (int): Group element coprime to ``modulus``.
        modulus (int): Candidate balanced semiprime greater than two.
        parameter_width (int | None): Positive short-phase width. Defaults to the
            Ekerå–Håstad choice ``ceil(n / 2) + 1``.

    Returns:
        qm.QKernel: Sampled function returning ``2m`` long bits followed by
        ``m`` short bits in Qamomile's source order.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the modular maps are not reversible.
    """

    _, selected_width, _, _, _, _ = _validate_short_dlp(
        generator=generator,
        modulus=modulus,
        parameter_width=parameter_width,
    )
    result_width = 3 * selected_width
    state = ekera_hastad_factoring_state(
        generator=generator,
        modulus=modulus,
        parameter_width=selected_width,
    )

    @qm.qkernel
    def sample_short_dlp() -> qm.Shot[qm.Bits[result_width]]:  # type: ignore[valid-type]
        """Sample the configured short discrete-logarithm phase registers.

        Returns:
            qm.Shot[qm.Bits[result_width]]: Measured long and short phase bits.
        """

        phase, target = qm.split_at(state(), result_width)
        result = qm.measure(phase)
        _ignored = qm.measure(target)
        return result

    return sample_short_dlp


@qm.qkernel
def shor_order_finding_state(
    counting: qm.Q[qm.Bits[p]],  # type: ignore[valid-type]
    target: qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
    base: qm.Const[qm.Int],  # type: ignore[valid-type]
    modulus: qm.Const[qm.Int],  # type: ignore[valid-type]
    precision: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> tuple[
    qm.Q[qm.Bits[p]],  # type: ignore[valid-type]
    qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
]:
    """Apply the symbolic Shor transform to two owned registers.

    Modular exponentiation remains one semantic operation. Resource projection
    can therefore compare exact square-and-multiply and windowed lookup
    implementations without changing this algorithm body.

    Args:
        counting (qm.Q[qm.Bits[p]]): Owned phase-estimation register.
        target (qm.Q[qm.Bits[n]]): Owned modular-value register initialized to
            ``|1>``.
        base (qm.Const[qm.Int]): Specialization-time modular unit whose order is
            sampled.
        modulus (qm.Const[qm.Int]): Specialization-time modulus represented by
            the target width.
        precision (qm.Const[qm.Int]): Positive counting-register width.

    Returns:
        tuple[qm.Q[qm.Bits[p]], qm.Q[qm.Bits[n]]]: Transformed counting and
        modular-value registers.

    Raises:
        ValueError: If specialization produces modular exponentiation that
            is not a total unitary.
    """

    for index in range(precision):
        counting = qm.h(counting, target=index)

    value, exponent = qm.mod_pow(
        qm.as_uint(target),
        qm.as_uint(counting),
        base=base,
        modulus=modulus,
    )
    target = qm.as_bits(value)
    counting = qm.as_bits(exponent)
    counting = inverse_qft(counting, precision)
    for left in range(precision // 2):
        counting = qm.swap(
            counting,
            left=left,
            right=precision - left - 1,
        )
    return counting, target


@qm.qkernel
def shor_order_finding(
    base: qm.Const[qm.Int],  # type: ignore[valid-type]
    modulus: qm.Const[qm.Int],  # type: ignore[valid-type]
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    precision: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[precision]]:  # type: ignore[valid-type]
    """Sample symbolic Shor order finding from one dependent-width body.

    Resource analysis may inspect this unspecialized qkernel directly. Execution
    specializes all four ``Const`` arguments, after which the same verified body
    is compiled for the selected target.

    Args:
        base (qm.Const[qm.Int]): Specialization-time modular unit whose order is
            sampled.
        modulus (qm.Const[qm.Int]): Specialization-time modulus represented by
            ``width`` bits.
        width (qm.Const[qm.Int]): Positive modular-register width.
        precision (qm.Const[qm.Int]): Positive counting-register width.

    Returns:
        qm.Shot[qm.Bits[precision]]: Measured counting-register bits.

    Raises:
        ValueError: If specialization produces invalid dimensions or a modular
            multiplication that is not a total unitary.
    """

    counting = qm.qubits(precision)
    target = qm.x(qm.qubits(width), target=0)
    counting, target = shor_order_finding_state(
        counting,
        target,
        base,
        modulus,
        precision,
    )
    result = qm.measure(counting)
    _ignored = qm.measure(target)
    return result


def _validate_short_dlp(
    *,
    generator: int,
    modulus: int,
    parameter_width: int | None,
) -> tuple[int, int, int, int, int, int]:
    """Validate one short-DLP configuration and return its coordinates.

    Args:
        generator (int): Candidate multiplicative group element.
        modulus (int): Candidate balanced semiprime greater than two.
        parameter_width (int | None): Explicit short phase width, or ``None``
            for the Ekerå–Håstad default.

    Returns:
        tuple[int, int, int, int, int, int]: Value width, selected parameter
        width, long and short precisions, target width, and inverse group value.

    Raises:
        TypeError: If an argument or native plan is malformed.
        ValueError: If the group configuration or width is invalid.
    """

    for name, value in (("generator", generator), ("modulus", modulus)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if parameter_width is not None and (
        isinstance(parameter_width, bool) or not isinstance(parameter_width, int)
    ):
        raise TypeError("parameter_width must be an integer")
    if parameter_width is not None and parameter_width <= 0:
        raise ValueError("parameter_width must be positive")
    derive = getattr(native, "plan_ekera_hastad", None)
    if derive is None:
        raise TypeError("Q7644: native short-DLP planning is unavailable")
    projection = derive(
        str(generator),
        str(modulus),
        None if parameter_width is None else str(parameter_width),
    )
    try:
        width = projection.width
        selected = projection.parameter_width
        long_precision = projection.long_precision
        short_precision = projection.short_precision
        target_width = projection.target_width
        inverse_y = _decimal_integer(projection.inverse_y)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("Q7645: malformed native short-DLP plan") from error
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                width,
                selected,
                long_precision,
                short_precision,
                target_width,
            )
        )
        or long_precision != 2 * selected
        or short_precision != selected
        or target_width != 2 * width + 2
        or not 0 < inverse_y < modulus
        or (parameter_width is not None and selected != parameter_width)
    ):
        raise TypeError("Q7645: malformed native short-DLP plan")
    return (
        width,
        selected,
        long_precision,
        short_precision,
        target_width,
        inverse_y,
    )


def _validate_multi_exponentiation_configuration(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
) -> int:
    """Validate one bounded modular multi-exponentiation configuration.

    Args:
        bases (tuple[int, ...]): Candidate modular units.
        modulus (int): Candidate classical modulus.
        coordinate_width (int): Candidate width of every coordinate.

    Returns:
        int: Width of the modular target register.

    Raises:
        TypeError: If a value has the wrong host type.
        ValueError: If a dimension, modulus, or base is invalid.
    """

    if not isinstance(bases, tuple):
        raise TypeError("bases must be a tuple of integers")
    if not bases:
        raise ValueError("bases must not be empty")
    if len(bases) > _MAX_REGEV_DIMENSIONS:
        raise ValueError(
            f"multi-exponentiation exceeds the {_MAX_REGEV_DIMENSIONS}-coordinate budget"
        )
    if isinstance(modulus, bool) or not isinstance(modulus, int):
        raise TypeError("modulus must be an integer")
    if modulus <= 2 or modulus.bit_length() >= 63:
        raise ValueError(
            "modulus must be greater than two and fit a signed 63-bit word"
        )
    if isinstance(coordinate_width, bool) or not isinstance(coordinate_width, int):
        raise TypeError("coordinate_width must be an integer")
    if coordinate_width <= 0:
        raise ValueError("coordinate_width must be positive")
    if len(bases) * coordinate_width > 4096:
        raise ValueError("packed coordinates exceed the 4096-bit source budget")
    for base in bases:
        if isinstance(base, bool) or not isinstance(base, int):
            raise TypeError("each base must be an integer")
        if not 0 < base < modulus:
            raise ValueError("each base must lie between zero and modulus")
        if math.gcd(base, modulus) != 1:
            raise ValueError("each base must be coprime to modulus")
    return modulus.bit_length()


def _validate_regev_configuration(
    *,
    bases: tuple[int, ...],
    modulus: int,
    coordinate_width: int,
    radius: float,
) -> tuple[tuple[int, ...], int, int]:
    """Validate one bounded Regev sampling configuration.

    Args:
        bases (tuple[int, ...]): Candidate small modular units.
        modulus (int): Candidate modulus.
        coordinate_width (int): Width of every coordinate register.
        radius (float): Discrete-Gaussian radius.

    Returns:
        tuple[tuple[int, ...], int, int]: Squared modular bases, target width,
        and coordinate count.

    Raises:
        TypeError: If a value has the wrong host type.
        ValueError: If a value, budget, or paper parameter relation is invalid.
    """

    if not isinstance(bases, tuple):
        raise TypeError("bases must be a tuple of integers")
    if not bases:
        raise ValueError("bases must not be empty")
    if len(bases) > _MAX_REGEV_DIMENSIONS:
        raise ValueError(
            f"Regev coordinate count exceeds the {_MAX_REGEV_DIMENSIONS}-dimension budget"
        )
    if isinstance(modulus, bool) or not isinstance(modulus, int):
        raise TypeError("modulus must be an integer")
    if modulus <= 2:
        raise ValueError("modulus must be greater than two")
    if isinstance(coordinate_width, bool) or not isinstance(coordinate_width, int):
        raise TypeError("coordinate_width must be an integer")
    if coordinate_width <= 0:
        raise ValueError("coordinate_width must be positive")
    if coordinate_width > _MAX_EXACT_GAUSSIAN_WIDTH:
        raise ValueError(
            "exact Gaussian preparation exceeds the 1048576-entry amplitude budget"
        )
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError("radius must be a real number")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    for base in bases:
        if isinstance(base, bool) or not isinstance(base, int):
            raise TypeError("each base must be an integer")
        if not 1 < base < modulus:
            raise ValueError("each base must lie strictly between one and modulus")
        if math.gcd(base, modulus) != 1:
            raise ValueError("each base must be coprime to modulus")
    dimension = len(bases)
    domain_size = 1 << coordinate_width
    scaled_radius = math.sqrt(dimension) * radius
    if not 2.0 * scaled_radius <= domain_size < 4.0 * scaled_radius:
        raise ValueError(
            "coordinate_width and radius must satisfy "
            "2 * sqrt(len(bases)) * radius <= 2**coordinate_width "
            "< 4 * sqrt(len(bases)) * radius"
        )
    return (
        tuple(pow(base, 2, modulus) for base in bases),
        modulus.bit_length(),
        dimension,
    )


def _regev_gaussian_window(*, width: int, radius: float) -> qm.QKernel:
    """Construct one exact finite discrete-Gaussian amplitude window.

    Args:
        width (int): Positive coordinate width.
        radius (float): Positive finite Gaussian radius.

    Returns:
        qm.QKernel: Unary state-preparation transform for one coordinate.
    """

    domain_size = 1 << width
    midpoint = domain_size // 2
    amplitudes = tuple(
        math.exp(-math.pi * ((value - midpoint) / radius) ** 2)
        for value in range(domain_size)
    )
    return amplitude_encoding(amplitudes)


def _fixed_qft(width: int) -> qm.QKernel:
    """Close the ordinary exact QFT over one fixed-width coordinate.

    Args:
        width (int): Positive coordinate width.

    Returns:
        qm.QKernel: Unary exact Fourier transform.
    """

    @qm.qkernel
    def transform(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply the exact QFT to one complete coordinate.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned coordinate register.

        Returns:
            qm.Q[qm.Bits[width]]: Fourier-transformed coordinate.
        """

        return qft(state, width)

    return transform


def _regev_modular_power_step(
    *,
    coordinate_total: int,
    coordinate_width: int,
    target_width: int,
    coordinate: int,
    base: int,
    modulus: int,
) -> qm.QKernel:
    """Lift one coordinate's modular power onto the shared product target.

    Args:
        coordinate_total (int): Width of all packed coordinate registers.
        coordinate_width (int): Width of the selected coordinate.
        target_width (int): Width of the trailing modular target.
        coordinate (int): Zero-based selected coordinate.
        base (int): Squared modular base for the selected coordinate.
        modulus (int): Positive modulus.

    Returns:
        qm.QKernel: Unary transform over packed coordinates and target.
    """

    prefix_width = coordinate * coordinate_width
    suffix_width = coordinate_total - prefix_width - coordinate_width
    total_width = coordinate_total + target_width

    if prefix_width == 0 and suffix_width == 0:

        @qm.qkernel
        def only_coordinate(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply modular power for the only coordinate.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Coordinate and target.

            Returns:
                qm.Q[qm.Bits[total_width]]: Updated target and retained coordinate.
            """

            exponent, target = qm.split_at(state, coordinate_width)
            target, exponent = qm.mod_pow(
                qm.as_uint(target),
                qm.as_uint(exponent),
                base=base,
                modulus=modulus,
            )
            return qm.concat(qm.as_bits(exponent), qm.as_bits(target))

        return only_coordinate

    if prefix_width == 0:

        @qm.qkernel
        def leading_coordinate(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply modular power for the leading coordinate.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Coordinates and target.

            Returns:
                qm.Q[qm.Bits[total_width]]: Updated target and retained coordinates.
            """

            exponent, tail = qm.split_at(state, coordinate_width)
            suffix, target = qm.split_at(tail, suffix_width)
            target, exponent = qm.mod_pow(
                qm.as_uint(target),
                qm.as_uint(exponent),
                base=base,
                modulus=modulus,
            )
            return qm.concat(qm.as_bits(exponent), suffix, qm.as_bits(target))

        return leading_coordinate

    if suffix_width == 0:

        @qm.qkernel
        def trailing_coordinate(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply modular power for the trailing coordinate.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Coordinates and target.

            Returns:
                qm.Q[qm.Bits[total_width]]: Updated target and retained coordinates.
            """

            prefix, tail = qm.split_at(state, prefix_width)
            exponent, target = qm.split_at(tail, coordinate_width)
            target, exponent = qm.mod_pow(
                qm.as_uint(target),
                qm.as_uint(exponent),
                base=base,
                modulus=modulus,
            )
            return qm.concat(prefix, qm.as_bits(exponent), qm.as_bits(target))

        return trailing_coordinate

    @qm.qkernel
    def interior_coordinate(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply modular power for one interior coordinate.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Coordinates and target.

        Returns:
            qm.Q[qm.Bits[total_width]]: Updated target and retained coordinates.
        """

        prefix, tail = qm.split_at(state, prefix_width)
        exponent, tail = qm.split_at(tail, coordinate_width)
        suffix, target = qm.split_at(tail, suffix_width)
        target, exponent = qm.mod_pow(
            qm.as_uint(target),
            qm.as_uint(exponent),
            base=base,
            modulus=modulus,
        )
        return qm.concat(
            prefix,
            qm.as_bits(exponent),
            suffix,
            qm.as_bits(target),
        )

    return interior_coordinate


def _little_endian(value: int, width: int) -> tuple[int, ...]:
    """Encode one nonnegative host integer into fixed little-endian bits.

    Args:
        value (int): Nonnegative integer to encode.
        width (int): Number of low-order bits to return.

    Returns:
        tuple[int, ...]: Fixed-width little-endian bit tuple.
    """

    return tuple((value >> index) & 1 for index in range(width))


def _decimal_integer(value: object) -> int:
    """Decode one canonical unsigned decimal integer projection.

    Args:
        value (object): Candidate canonical decimal text.

    Returns:
        int: Decoded nonnegative integer.

    Raises:
        TypeError: If ``value`` is not canonical unsigned decimal text.
    """

    if not isinstance(value, str) or not value.isdecimal():
        raise TypeError
    return int(value)


__all__ = [
    "ekera_hastad_factoring",
    "ekera_hastad_factoring_state",
    "regev_factoring",
    "regev_factoring_state",
    "shor_order_finding",
    "shor_order_finding_state",
]
