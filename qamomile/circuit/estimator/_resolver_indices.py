"""Index immutable block structure for classical expression resolution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ArrayValue, Value


def _compute_input_shape_dimension_aliases(block: Block) -> dict[str, str]:
    """Compute collision-free public aliases for root array dimensions.

    Args:
        block (Block): Root block whose input dimensions should be named.

    Returns:
        dict[str, str]: Dimension UUID to deterministic, unique input alias.
    """
    occupied = {
        *block.label_args,
        *(slot.name for slot in block.param_slots),
        *block.parameters,
    }
    aliases: dict[str, str] = {}
    for input_value in block.input_values:
        if not isinstance(input_value, ArrayValue):
            continue
        for dimension in input_value.shape:
            alias = dimension.name
            if not alias:
                alias = f"array_dim_{len(aliases)}"
            if alias in occupied:
                base = f"{alias}__shape"
                alias = base
                suffix = 2
                while alias in occupied:
                    alias = f"{base}_{suffix}"
                    suffix += 1
            aliases[dimension.uuid] = alias
            occupied.add(alias)
    return aliases


class _ResolverBlockIndex:
    """Own immutable producer and input-shape indexes for one resolver tree.

    The identity-keyed caches retain each block strongly and recheck object
    identity on every lookup. This prevents a recycled ``id`` from returning
    an index built for a previously collected block.

    """

    __slots__ = (
        "_input_shape_alias_maps",
        "_producer_maps",
    )

    def __init__(self) -> None:
        """Initialize empty block-identity indexes."""
        self._input_shape_alias_maps: dict[
            int,
            tuple[Block, dict[str, str]],
        ] = {}
        self._producer_maps: dict[
            int,
            tuple[Any, dict[str, Operation]],
        ] = {}

    def producer_map(self, block: Any) -> dict[str, Operation]:
        """Return the cached producer index for one block.

        Args:
            block (Any): Block-like object exposing an ``operations`` list.

        Returns:
            dict[str, Operation]: Result UUID to defining operation. All
                operation results are indexed, not only the first result.
        """
        block_id = id(block)
        cached = self._producer_maps.get(block_id)
        if cached is None or cached[0] is not block:
            producers = {
                result.uuid: operation
                for operation in block.operations
                for result in operation.results
            }
            self._producer_maps[block_id] = (block, producers)
            return producers
        return cached[1]

    def input_shape_alias_map(self, block: Block) -> dict[str, str]:
        """Return the cached input-dimension alias index for one block.

        Args:
            block (Block): Block whose immutable interface is indexed.

        Returns:
            dict[str, str]: Input-dimension UUID to collision-free public
                alias.
        """
        block_id = id(block)
        cached = self._input_shape_alias_maps.get(block_id)
        if cached is None or cached[0] is not block:
            aliases = _compute_input_shape_dimension_aliases(block)
            self._input_shape_alias_maps[block_id] = (block, aliases)
            return aliases
        return cached[1]

    def array_producer(
        self,
        array: ArrayValue,
        blocks: Iterable[Any],
    ) -> Operation | None:
        """Return the operation producing one array SSA version.

        Args:
            array (ArrayValue): Array value whose producer is requested.
            blocks (Iterable[Any]): Blocks to search in resolution order.

        Returns:
            Operation | None: Producing operation, or ``None`` when the array
                is an input or initializer.
        """
        for block in blocks:
            if block is None:
                continue
            producer = self.producer_map(block).get(array.uuid)
            if producer is not None:
                return producer
            for nested in walk_operations(block.operations):
                if any(result.uuid == array.uuid for result in nested.results):
                    return nested
        return None

    def input_shape_dimension_alias(
        self,
        value: Value,
        blocks: Iterable[Any],
    ) -> str | None:
        """Return the public alias for one input-array dimension.

        Args:
            value (Value): Unresolved value considered for symbolic fallback.
            blocks (Iterable[Any]): Blocks to search in resolution order.

        Returns:
            str | None: Stable input alias when ``value`` is an input-array
                dimension, otherwise ``None``.
        """
        for block in blocks:
            if not isinstance(block, Block):
                continue
            alias = self.input_shape_alias_map(block).get(value.uuid)
            if alias is not None:
                return alias
        return None
