"""Shared call-site pairing for operation-owned blocks."""

from __future__ import annotations

from collections.abc import Sequence

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.value import ValueBase


def align_formal_operands(
    formals: Sequence[ValueBase],
    quantum_operands: Sequence[ValueBase],
    parameter_operands: Sequence[ValueBase],
) -> list[ValueBase]:
    """Align split call-site operand pools to formal declaration order.

    Operation-owned call sites store quantum operands separately from
    classical/object operands, while a block keeps the Python declaration
    order and may interleave those categories. Reweaving the two pools here
    gives every consumer one canonical formal-to-actual convention.

    Args:
        formals (Sequence[ValueBase]): Formal inputs in declaration order.
        quantum_operands (Sequence[ValueBase]): Quantum actual operands in
            their call-site order.
        parameter_operands (Sequence[ValueBase]): Classical/object actual
            operands in their call-site order.

    Returns:
        list[ValueBase]: Actual operands aligned with ``formals``.
            The list stops at the first category shortfall so a downstream
            positional pairing cannot silently consume an operand of the wrong
            category.
    """
    quantum_iter = iter(quantum_operands)
    parameter_iter = iter(parameter_operands)
    aligned: list[ValueBase] = []
    for formal in formals:
        operands = quantum_iter if formal.type.is_quantum() else parameter_iter
        actual = next(operands, None)
        if actual is None:
            break
        aligned.append(actual)
    return aligned


def pair_block_operands(
    block: Block,
    operands: Sequence[ValueBase],
) -> list[tuple[ValueBase, ValueBase]]:
    """Pair all block inputs with category-grouped call-site operands.

    Args:
        block (Block): Operation-owned block whose inputs are being bound.
        operands (Sequence[ValueBase]): Call-site operands after any controls
            that are external to ``block`` have been removed.

    Returns:
        list[tuple[ValueBase, ValueBase]]: Formal/actual pairs in the block's
        declaration order.
    """
    quantum_operands = [value for value in operands if value.type.is_quantum()]
    parameter_operands = [
        value
        for value in operands
        if value.type.is_classical() or value.type.is_object()
    ]
    aligned = align_formal_operands(
        block.input_values,
        quantum_operands,
        parameter_operands,
    )
    return list(zip(block.input_values, aligned))


def pair_block_parameter_operands(
    block: Block,
    param_operands: Sequence[ValueBase],
) -> list[tuple[ValueBase, ValueBase]]:
    """Pair a block's classical/object inputs with call-site operands.

    Both compile-time lowering and emission bind an operation-owned block's
    non-quantum inputs by their declaration order. Keeping the filtering and
    pairing here ensures those stages cannot adopt different positional
    conventions.

    Args:
        block (Block): Operation-owned block whose formal inputs define the
            declaration order.
        param_operands (Sequence[ValueBase]): Classical or object operands at the
            call site, already ordered according to the operation signature.

    Returns:
        list[tuple[ValueBase, ValueBase]]: ``(formal, actual)`` pairs in
        declaration order. Missing actual operands leave trailing formals
        unpaired, so they can remain symbolic and be provided at emit time.
    """
    param_inputs = [
        value
        for value in block.input_values
        if value.type.is_classical() or value.type.is_object()
    ]
    return list(zip(param_inputs, param_operands))


def block_parameter_binding_keys(parameter: ValueBase) -> tuple[str, ...]:
    """Return the sanctioned emit-binding keys for an inner formal.

    Emit resolution accepts a kernel parameter name before its UUID for public
    API compatibility, so an operation-owned fresh scope must write both keys.
    Keeping the key choice here prevents individual emit paths from reviving a
    parent binding that merely shares the inner formal's display name.

    Args:
        parameter (ValueBase): Classical/object formal input being bound.

    Returns:
        tuple[str, ...]: UUID followed by the formal parameter provenance name
            and nonempty display-name compatibility key, without duplicates.
    """
    keys = [parameter.uuid]
    if parameter.is_parameter():
        parameter_name = parameter.parameter_name()
        if parameter_name and parameter_name not in keys:
            keys.append(parameter_name)
    if parameter.name and parameter.name not in keys:
        keys.append(parameter.name)
    return tuple(keys)
