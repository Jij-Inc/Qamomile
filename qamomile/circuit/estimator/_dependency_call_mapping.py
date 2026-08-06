"""Map dependency metadata across callable boundaries."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING

import sympy as sp

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
    _resource_max,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._dependency_footprints import _quantum_value_wire_keys
from qamomile.circuit.estimator._dependency_indices import (
    _UNKNOWN_WIRE_INDEX,
    WireKey,
    _WireRangeIndex,
)
from qamomile.circuit.estimator._quantum_values import (
    _array_wire_key_at_index,
    _quantum_allocation_owner,
)


def _map_value_dependency_keys(
    source: Value,
    actual: Value,
    dependency_keys: frozenset[WireKey],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[WireKey]:
    """Map dependency keys owned by one body value onto a caller value.

    Args:
        source (Value): Body input or output whose owner appears in the mask.
        actual (Value): Caller-side operand or result corresponding to source.
        dependency_keys (frozenset[WireKey]): Body-scoped touched-wire mask.
        resolver (ExprResolver): Caller resolver for arrays and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[WireKey]: Caller-scoped dependency keys for this correspondence.
    """
    source_owner = _quantum_allocation_owner(source)
    mapped: set[WireKey] = set()
    for owner, index in dependency_keys:
        if owner != source_owner:
            continue
        if index is None or not isinstance(actual, ArrayValue):
            mapped.update(
                _quantum_value_wire_keys(
                    actual,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
        elif index is _UNKNOWN_WIRE_INDEX:
            mapped.add((_quantum_allocation_owner(actual), _UNKNOWN_WIRE_INDEX))
        elif isinstance(index, _WireRangeIndex):
            mapped_owner, mapped_index = _array_wire_key_at_index(
                actual,
                index.index_at_offset,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            if isinstance(mapped_index, (int, sp.Expr)):
                mapped.add(
                    (
                        mapped_owner,
                        _WireRangeIndex(
                            index_at_offset=_expr(mapped_index),
                            iterations=index.iterations,
                        ),
                    )
                )
            else:
                mapped.add((mapped_owner, mapped_index))
        else:
            mapped.add(
                _array_wire_key_at_index(
                    actual,
                    index,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    return mapped


def _map_body_dependency_keys(
    block: Block,
    body_estimate: ResourceEstimate,
    actual_operands: Sequence[ValueBase],
    caller_results: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> frozenset[WireKey] | None:
    """Translate a body's touched inputs and outputs to caller wire keys.

    Callee-local scratch allocations deliberately disappear at the boundary;
    their depth remains part of the call duration but cannot block unrelated
    caller wires. Returned allocations are mapped through ``output_values`` so
    a caller gate cannot start before the producing body work completes.

    Args:
        block (Block): Evaluated callable implementation.
        body_estimate (ResourceEstimate): Body-scoped estimate carrying its
            touched-wire mask.
        actual_operands (Sequence[ValueBase]): Caller operands aligned by the
            shared block-pairing convention.
        caller_results (Sequence[ValueBase]): Caller results aligned with the
            block outputs.
        resolver (ExprResolver): Caller-side value resolver.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        frozenset[WireKey] | None: Caller-scoped mask, or ``None`` when the
            body did not provide an authoritative mask.
    """
    dependency_keys = body_estimate._dependency_keys
    if dependency_keys is None:
        return None
    mapped: set[WireKey] = set()
    for formal, actual in pair_block_operands(block, actual_operands):
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and formal.type.is_quantum()
            and actual.type.is_quantum()
        ):
            mapped.update(
                _map_value_dependency_keys(
                    formal,
                    actual,
                    dependency_keys,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    if len(block.output_values) == len(caller_results):
        for output, result in zip(
            block.output_values,
            caller_results,
            strict=True,
        ):
            if (
                isinstance(output, Value)
                and isinstance(result, Value)
                and output.type.is_quantum()
                and result.type.is_quantum()
            ):
                mapped.update(
                    _map_value_dependency_keys(
                        output,
                        result,
                        dependency_keys,
                        resolver,
                        scalar_values=scalar_values,
                        used_names=used_names,
                    )
                )
    return frozenset(mapped)


def _map_body_dependency_completion(
    block: Block,
    body_estimate: ResourceEstimate,
    actual_operands: Sequence[ValueBase],
    caller_results: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> dict[WireKey, ResourceExpr] | None:
    """Translate per-wire body completion onto caller-scoped wire keys.

    Args:
        block (Block): Evaluated callable implementation.
        body_estimate (ResourceEstimate): Body-scoped estimate carrying
            per-wire dependency completion depths.
        actual_operands (Sequence[ValueBase]): Caller operands aligned with
            the block inputs.
        caller_results (Sequence[ValueBase]): Caller results aligned with the
            block outputs.
        resolver (ExprResolver): Caller-side value resolver.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        dict[WireKey, ResourceExpr] | None: Caller-scoped completion depths,
        or ``None`` when the body did not provide them.
    """
    completion = body_estimate._dependency_completion
    if completion is None:
        return None
    mapped: dict[WireKey, ResourceExpr] = {}

    def map_value(source: Value, actual: Value) -> None:
        """Map all completion depths owned by one body value.

        Args:
            source (Value): Body-side quantum value.
            actual (Value): Corresponding caller-side quantum value.
        """
        source_owner = _quantum_allocation_owner(source)
        for key, depth in completion.items():
            if key[0] != source_owner:
                continue
            caller_keys = _map_value_dependency_keys(
                source,
                actual,
                frozenset((key,)),
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            for caller_key in caller_keys:
                mapped[caller_key] = _resource_max(
                    mapped.get(caller_key, _ZERO),
                    depth,
                )

    for formal, actual in pair_block_operands(block, actual_operands):
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and formal.type.is_quantum()
            and actual.type.is_quantum()
        ):
            map_value(formal, actual)
    if len(block.output_values) == len(caller_results):
        for output, result in zip(
            block.output_values,
            caller_results,
            strict=True,
        ):
            if (
                isinstance(output, Value)
                and isinstance(result, Value)
                and output.type.is_quantum()
                and result.type.is_quantum()
            ):
                map_value(output, result)
    return mapped
