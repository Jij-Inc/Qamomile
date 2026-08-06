"""Compose private dependency-scheduling metadata for resource estimates."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    _UNKNOWN_WIRE_INDEX,
    WireKey,
    _normalize_wire_index,
    _symbolic_wire_range_index,
    _WireRangeIndex,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import (
    _ConditionIndicator,
    _expr,
    _resource_activity_condition,
    _resource_max,
)
from qamomile.circuit.estimator._scheduling import (
    _estimate_has_nonzero_depth,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate

from qamomile.circuit.estimator._constants import (
    _ZERO,
)


def _normalized_dependency_completion(
    estimate: ResourceEstimate,
) -> dict[WireKey, ResourceExpr] | None:
    """Return an explicit per-wire completion map when one can be recovered.

    Args:
        estimate (ResourceEstimate): Estimate carrying dependency metadata.

    Returns:
        dict[WireKey, ResourceExpr] | None: Caller-visible local completion
        depths, an empty map for a proven empty estimate, or ``None`` for an
        unknown footprint.
    """
    if estimate._dependency_completion is not None:
        return dict(estimate._dependency_completion)
    keys = estimate._dependency_keys
    if keys is None:
        if not _estimate_has_nonzero_depth(estimate):
            return {}
        return None
    return {key: estimate.depth.depth for key in keys}


def _merge_dependency_accesses(
    left: ResourceEstimate,
    right: ResourceEstimate,
    *,
    writes: bool,
) -> frozenset[WireKey] | None:
    """Merge body-level scheduler reads or writes across two estimates.

    Args:
        left (ResourceEstimate): Left composition operand.
        right (ResourceEstimate): Right composition operand.
        writes (bool): Whether to merge write sets instead of read sets.

    Returns:
        frozenset[WireKey] | None: Union of known access sets, or ``None``
            when a nonempty operand has no recoverable footprint.
    """

    def normalized(estimate: ResourceEstimate) -> frozenset[WireKey] | None:
        """Recover one access set from explicit or symmetric metadata.

        Args:
            estimate (ResourceEstimate): Estimate whose access set is needed.

        Returns:
            frozenset[WireKey] | None: Explicit access set, symmetric fallback,
                empty set for zero depth, or ``None`` when unknown.
        """
        accesses = estimate._dependency_writes if writes else estimate._dependency_reads
        if accesses is not None:
            return accesses
        if estimate._dependency_keys is not None:
            return estimate._dependency_keys
        if not _estimate_has_nonzero_depth(estimate):
            return frozenset()
        return None

    left_accesses = normalized(left)
    right_accesses = normalized(right)
    if left_accesses is None or right_accesses is None:
        return None
    return left_accesses | right_accesses


def _project_dependency_metadata_over_symbol(
    estimate: ResourceEstimate,
    symbol: sp.Symbol,
    *,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> ResourceEstimate:
    """Project loop-local scalar wire keys into the enclosing scope.

    A loop induction symbol is local to one iteration. Once the loop is
    summarized, an address such as ``register[i]`` represents a set of
    caller-visible wires rather than one scalar wire. Concrete bounded loops
    enumerate those addresses exactly. A symbolic or large one-dimensional
    loop retains a canonical range descriptor, while a nested range that
    cannot be represented without leaking an outer binder falls back to an
    unknown-scalar marker. Neither representation pretends that the loop
    definitely touched the whole allocation.

    Args:
        estimate (ResourceEstimate): Estimate whose dependency metadata may
            contain the bound symbol.
        symbol (sp.Symbol): Bound symbol leaving scope.
        start (ResourceExpr): First Python-range value.
        step (ResourceExpr): Python-range step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        ResourceEstimate: Estimate with concrete loop addresses enumerated and
            unresolved ranges represented without leaking local symbols.
    """

    start_expr = _expr(start)
    step_expr = _expr(step)
    iterations_expr = _expr(iterations)
    concrete_values: tuple[sp.Integer, ...] | None = None
    if all(
        bound.is_number and _is_concrete_integer(bound)
        for bound in (start_expr, step_expr, iterations_expr)
    ):
        concrete_start = int(start_expr)
        concrete_step = int(step_expr)
        concrete_iterations = int(iterations_expr)
        if (
            concrete_step != 0
            and 0 <= concrete_iterations <= _MAX_EXACT_LOOP_WIRE_EXPANSION
        ):
            concrete_values = tuple(
                sp.Integer(concrete_start + concrete_step * offset)
                for offset in range(concrete_iterations)
            )

    def project(key: WireKey) -> tuple[WireKey, ...]:
        """Project one symbol-dependent scalar address.

        Args:
            key (WireKey): Allocation-owner and scalar-index address.

        Returns:
            tuple[WireKey, ...]: Concrete projected addresses, one unknown
                scalar address, or the unchanged address.
        """
        owner, index = key
        if isinstance(index, _WireRangeIndex):
            range_symbols = (
                index.index_at_offset.free_symbols | index.iterations.free_symbols
            )
            if symbol not in range_symbols:
                return (key,)
            if concrete_values is None:
                return ((owner, _UNKNOWN_WIRE_INDEX),)
            return tuple(
                (
                    owner,
                    index.mapped(
                        lambda expression: expression.subs(
                            symbol,
                            value,
                            simultaneous=True,
                        )
                    ),
                )
                for value in concrete_values
            )
        if not isinstance(index, sp.Expr) or symbol not in index.free_symbols:
            return (key,)
        if concrete_values is None:
            return (
                (
                    owner,
                    _symbolic_wire_range_index(
                        index,
                        symbol,
                        start=start_expr,
                        step=step_expr,
                        iterations=iterations_expr,
                    ),
                ),
            )
        return tuple(
            (
                owner,
                _normalize_wire_index(
                    cast(
                        ResourceExpr,
                        index.subs(symbol, value, simultaneous=True),
                    )
                ),
            )
            for value in concrete_values
        )

    keys = estimate._dependency_keys
    projected_keys = (
        frozenset(projected for key in keys for projected in project(key))
        if keys is not None
        else None
    )
    reads = estimate._dependency_reads
    projected_reads = (
        frozenset(projected for key in reads for projected in project(key))
        if reads is not None
        else None
    )
    writes = estimate._dependency_writes
    projected_writes = (
        frozenset(projected for key in writes for projected in project(key))
        if writes is not None
        else None
    )
    completion = estimate._dependency_completion
    projected_completion: dict[WireKey, ResourceExpr] | None
    if completion is None:
        projected_completion = None
    else:
        projected_completion = {}
        for key, depth in completion.items():
            for projected in project(key):
                projected_completion[projected] = _resource_max(
                    projected_completion.get(projected, _ZERO),
                    depth,
                )
    return dataclasses.replace(
        estimate,
        _dependency_keys=projected_keys,
        _dependency_reads=projected_reads,
        _dependency_writes=projected_writes,
        _dependency_completion=projected_completion,
    )


def _seq_dependency_completion(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> dict[WireKey, ResourceExpr] | None:
    """Compose per-wire completion depths sequentially.

    Args:
        left (ResourceEstimate): First composition operand.
        right (ResourceEstimate): Second composition operand.

    Returns:
        dict[WireKey, ResourceExpr] | None: Sequential completion depths, or
        ``None`` if either nonempty footprint is unknown.
    """
    left_completion = _normalized_dependency_completion(left)
    right_completion = _normalized_dependency_completion(right)
    if left_completion is None or right_completion is None:
        return None
    merged = dict(left_completion)
    for key, completion in right_completion.items():
        previous = merged.get(key, _ZERO)
        active = _ConditionIndicator(_resource_activity_condition(completion))
        merged[key] = cast(
            ResourceExpr,
            previous + active * (left.depth.depth + completion - previous),
        )
    return merged


def _max_dependency_completion(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> dict[WireKey, ResourceExpr] | None:
    """Merge per-wire completion depths by their maximum.

    Args:
        left (ResourceEstimate): First composition operand.
        right (ResourceEstimate): Second composition operand.

    Returns:
        dict[WireKey, ResourceExpr] | None: Per-wire maximums, or ``None`` if
        either nonempty footprint is unknown.
    """
    left_completion = _normalized_dependency_completion(left)
    right_completion = _normalized_dependency_completion(right)
    if left_completion is None or right_completion is None:
        return None
    return {
        key: _resource_max(
            left_completion.get(key, _ZERO),
            right_completion.get(key, _ZERO),
        )
        for key in left_completion.keys() | right_completion.keys()
    }


def _conditional_dependency_completion(
    true_estimate: ResourceEstimate,
    false_estimate: ResourceEstimate,
    condition: sp.Basic,
) -> dict[WireKey, ResourceExpr] | None:
    """Select per-wire completion between two symbolic branches.

    Args:
        true_estimate (ResourceEstimate): Estimate selected when true.
        false_estimate (ResourceEstimate): Estimate selected when false.
        condition (sp.Basic): Branch predicate.

    Returns:
        dict[WireKey, ResourceExpr] | None: Exact branch-selected completion
        depths, or ``None`` when either nonempty branch is unknown.
    """
    true_completion = _normalized_dependency_completion(true_estimate)
    false_completion = _normalized_dependency_completion(false_estimate)
    if true_completion is None or false_completion is None:
        return None
    return {
        key: cast(
            ResourceExpr,
            false_completion.get(key, _ZERO)
            + _ConditionIndicator(condition)
            * (true_completion.get(key, _ZERO) - false_completion.get(key, _ZERO)),
        )
        for key in true_completion.keys() | false_completion.keys()
    }


def _map_dependency_key(
    key: WireKey,
    fn: Any,
) -> WireKey:
    """Rewrite the symbolic index of one private dependency key.

    Args:
        key (WireKey): Allocation-owner and optional scalar-index address.
        fn (Any): Symbolic expression rewrite callable.

    Returns:
        WireKey: Address with its symbolic scalar index rewritten and
            normalized.
    """
    owner, index = key
    if isinstance(index, _WireRangeIndex):
        return owner, index.mapped(fn)
    if not isinstance(index, sp.Expr):
        return key
    return owner, _normalize_wire_index(cast(ResourceExpr, fn(index)))


def _map_dependency_keys(
    keys: frozenset[WireKey] | None,
    fn: Any,
) -> frozenset[WireKey] | None:
    """Rewrite every symbolic private dependency key.

    Args:
        keys (frozenset[WireKey] | None): Dependency addresses, or ``None``
            when the footprint is unavailable.
        fn (Any): Symbolic expression rewrite callable.

    Returns:
        frozenset[WireKey] | None: Rewritten dependency addresses.
    """
    if keys is None:
        return None
    return frozenset(_map_dependency_key(key, fn) for key in keys)


def _map_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    fn: Any,
) -> dict[WireKey, ResourceExpr] | None:
    """Rewrite per-wire addresses and completion depths.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): Completion depths
            to rewrite.
        fn (Any): Symbolic expression rewrite callable.

    Returns:
        dict[WireKey, ResourceExpr] | None: Rewritten nonzero completion
            depths with colliding addresses merged.
    """
    if completion is None:
        return None
    mapped: dict[WireKey, ResourceExpr] = {}
    for key, value in completion.items():
        rewritten = cast(ResourceExpr, fn(value))
        if rewritten != _ZERO:
            mapped_key = _map_dependency_key(key, fn)
            mapped[mapped_key] = _resource_max(
                mapped.get(mapped_key, _ZERO),
                rewritten,
            )
    return mapped


def _complete_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    keys: Iterable[WireKey],
    *,
    fallback_depth: ResourceExpr,
) -> dict[WireKey, ResourceExpr]:
    """Fill missing caller-wire completion with an aggregate depth.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): Precisely mapped
            completion depths when available.
        keys (Iterable[WireKey]): Complete caller-visible dependency keys.
        fallback_depth (ResourceExpr): Aggregate completion used for keys
            whose individual latency is unavailable.

    Returns:
        dict[WireKey, ResourceExpr]: Complete per-wire completion map.
    """
    completed = dict(completion or {})
    for key in keys:
        completed.setdefault(key, fallback_depth)
    return completed
