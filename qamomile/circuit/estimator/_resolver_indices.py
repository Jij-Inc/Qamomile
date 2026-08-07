"""Index immutable block structure for classical expression resolution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import sympy as sp

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


def _resolve_concrete_array_payload(
    array: ArrayValue,
    bindings: Mapping[str, Any],
    *,
    resolve_expression: Callable[[Value], sp.Expr] | None = None,
    specialize: Callable[[sp.Expr], sp.Expr] | None = None,
    source: str,
) -> Any | None:
    """Resolve a rank-one array or view to its concrete Python payload.

    This helper supplements :class:`ValueResolver` with estimator-local
    expression resolution. In particular, slice lengths, starts, and steps may
    be caller expressions that become concrete only after ``inputs`` are
    substituted. Nested callable aliases are followed without guessing when a
    required expression remains symbolic.

    Args:
        array (ArrayValue): Root array or nested slice view to resolve.
        bindings (Mapping[str, Any]): Active UUID and public-name bindings.
        resolve_expression (Callable[[Value], sp.Expr] | None): Resolver for
            scalar IR values in the array's current callable scope. Defaults to
            ``None``.
        specialize (Callable[[sp.Expr], sp.Expr] | None): Concrete-input
            substitution applied after expression resolution. Defaults to
            ``None``.
        source (str): Human-readable operation name used in diagnostics.

    Returns:
        Any | None: Concrete array payload, or ``None`` when the payload or a
        required view expression remains unresolved.

    Raises:
        ValueError: If a concrete view has invalid bounds or addresses an
            element outside its bound root payload.
    """
    active_bindings = dict(bindings)
    value_resolver = ValueResolver()
    visited: set[str] = set()

    def concrete_integer(value: Value) -> int | None:
        """Resolve one scalar view expression to a concrete integer.

        Args:
            value (Value): Scalar shape or slice value to resolve.

        Returns:
            int | None: Concrete integer, or ``None`` while unresolved.
        """
        if resolve_expression is not None:
            expression = resolve_expression(value)
            if specialize is not None:
                expression = specialize(expression)
            if expression.is_number and expression.is_integer is True:
                return int(expression)
        return value_resolver.resolve_int_value(value, active_bindings)

    def resolve(current: ArrayValue) -> Any | None:
        """Resolve one array while guarding recursive aliases.

        Args:
            current (ArrayValue): Current root, view, or alias.

        Returns:
            Any | None: Concrete payload, or ``None`` while unresolved.

        Raises:
            ValueError: If a concrete view is malformed or out of bounds.
        """
        if current.uuid in visited:
            return None
        visited.add(current.uuid)
        try:
            payload = value_resolver.resolve_bound_value(current, active_bindings)
            if payload is not None and not isinstance(payload, ArrayValue):
                return payload
            if isinstance(payload, ArrayValue) and payload is not current:
                return resolve(payload)
            if not current.is_slice():
                return None
            if len(current.shape) != 1:
                return None
            length = concrete_integer(current.shape[0])
            if length is None:
                return None
            if length < 0:
                raise ValueError(f"{source} array view has negative length {length}.")

            start = 0
            step = 1
            root = current
            while root.slice_of is not None:
                if root.slice_start is None or root.slice_step is None:
                    return None
                sub_start = concrete_integer(root.slice_start)
                sub_step = concrete_integer(root.slice_step)
                if sub_start is None or sub_step is None:
                    return None
                if sub_start < 0 or sub_step <= 0:
                    raise ValueError(
                        f"{source} array view resolved to start={sub_start}, "
                        f"step={sub_step}; start must be non-negative and "
                        "step positive."
                    )
                start = sub_start + sub_step * start
                step = sub_step * step
                root = root.slice_of

            root_payload = resolve(root)
            if root_payload is None:
                return None
            try:
                return tuple(
                    root_payload[start + step * index] for index in range(length)
                )
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"{source} array view exceeds its bound root payload."
                ) from error
        finally:
            visited.discard(current.uuid)

    return resolve(array)


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
