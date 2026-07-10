"""Shared call-site pairing for operation-owned blocks."""

from __future__ import annotations

from collections.abc import Sequence

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.value import Value


def pair_block_parameter_operands(
    block: Block,
    param_operands: Sequence[Value],
) -> list[tuple[Value, Value]]:
    """Pair a block's classical/object inputs with call-site operands.

    Both compile-time lowering and emission bind an operation-owned block's
    non-quantum inputs by their declaration order. Keeping the filtering and
    pairing here ensures those stages cannot adopt different positional
    conventions.

    Args:
        block (Block): Operation-owned block whose formal inputs define the
            declaration order.
        param_operands (Sequence[Value]): Classical or object operands at the
            call site, already ordered according to the operation signature.

    Returns:
        list[tuple[Value, Value]]: ``(formal, actual)`` pairs in declaration
        order. Missing actual operands leave trailing formals unpaired, so
        they can remain symbolic and be provided at emit time.
    """
    param_inputs = [
        value
        for value in block.input_values
        if value.type.is_classical() or value.type.is_object()
    ]
    return list(zip(param_inputs, param_operands))
