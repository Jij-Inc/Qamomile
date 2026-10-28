"""Compose private dependency-scheduling metadata for resource estimates."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    _UNKNOWN_WIRE_INDEX,
    WireKey,
    _normalize_wire_index,
    _symbolic_wire_range_index,
    _WireRangeIndex,
)
from qamomile.circuit.estimator._dependency_synchronization import (
    _merge_synchronized_entry_certificates,
    _SynchronizedEntryCertificate,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import (
    _activation_over_range,
    _boolean_condition,
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


def _merge_synchronized_entry_conditions(
    *condition_maps: Mapping[WireKey, sp.Basic],
) -> dict[WireKey, Boolean]:
    """Merge guarded synchronized-entry requirements by physical wire.

    Args:
        *condition_maps (Mapping[WireKey, sp.Basic]): Requirement conditions
            to combine.

    Returns:
        dict[WireKey, Boolean]: Conditions merged with logical OR for each
        physical wire.
    """
    merged: dict[WireKey, Boolean] = {}
    for conditions in condition_maps:
        for key, condition in conditions.items():
            active = _boolean_condition(condition)
            if active is sp.false:
                continue
            merged[key] = _boolean_condition(sp.Or(merged.get(key, sp.false), active))
    return merged


def _synchronized_entry_activity_condition(
    estimate: ResourceEstimate,
) -> Boolean:
    """Return when an estimate requires synchronized input-wire readiness.

    Args:
        estimate (ResourceEstimate): Estimate carrying guarded entry
            requirements.

    Returns:
        Boolean: Logical union of every guarded entry requirement.
    """
    conditions = (
        *estimate._dependency_synchronized_entry_conditions.values(),
        *(
            certificate.active_when
            for certificate in estimate._dependency_synchronized_entry_certificates
            if certificate.coverage
        ),
    )
    return cast(
        Boolean,
        sp.Or(*conditions) if conditions else sp.false,
    )


def _synchronized_entry_condition_symbols(
    estimate: ResourceEstimate,
) -> set[sp.Symbol]:
    """Return symbols used by synchronized-entry activation guards.

    These guards are scheduler-local metadata rather than public resource
    expressions. Loop aggregation still needs to see their bound induction
    symbols so it does not choose a binder-independent repetition shortcut.

    Args:
        estimate (ResourceEstimate): Estimate carrying guarded entry
            requirements.

    Returns:
        set[sp.Symbol]: Free symbols used by the requirement conditions.
    """
    conditions = (
        *estimate._dependency_synchronized_entry_conditions.values(),
        *(
            certificate.active_when
            for certificate in estimate._dependency_synchronized_entry_certificates
            if certificate.coverage
        ),
    )
    return cast(
        set[sp.Symbol],
        {symbol for condition in conditions for symbol in condition.free_symbols},
    )


def _dependency_completion_symbols(
    estimate: ResourceEstimate,
) -> set[sp.Symbol]:
    """Return symbols used only by per-wire completion metadata.

    Completion expressions are scheduler-local rather than public resource
    fields, but a loop binder appearing only here still requires range-aware
    aggregation so that it cannot escape its scope.

    Args:
        estimate (ResourceEstimate): Estimate carrying per-wire completion
            metadata.

    Returns:
        set[sp.Symbol]: Free symbols used by completion expressions.
    """
    completion = estimate._dependency_completion or {}
    return cast(
        set[sp.Symbol],
        {
            symbol
            for depth in completion.values()
            for symbol in sp.sympify(depth).free_symbols
        },
    )


def _dependency_metadata_symbols(
    estimate: ResourceEstimate,
) -> set[sp.Symbol]:
    """Return symbols retained only by dependency-scheduling metadata.

    Args:
        estimate (ResourceEstimate): Estimate carrying dependency keys,
            completion expressions, synchronized-entry conditions, and
            grouped entry certificates.

    Returns:
        set[sp.Symbol]: Free symbols used by private dependency metadata.
    """

    def key_symbols(key: WireKey) -> set[sp.Symbol]:
        """Return symbolic variables retained by one wire address.

        Args:
            key (WireKey): Allocation owner and scalar or range address.

        Returns:
            set[sp.Symbol]: Free symbols used by the wire address.
        """
        index = key[1]
        if isinstance(index, _WireRangeIndex):
            return cast(
                set[sp.Symbol],
                sp.sympify(index.index_at_offset).free_symbols
                | sp.sympify(index.iterations).free_symbols,
            )
        if isinstance(index, sp.Expr):
            return cast(set[sp.Symbol], index.free_symbols)
        return set()

    symbols = _dependency_completion_symbols(estimate)
    symbols.update(_synchronized_entry_condition_symbols(estimate))
    key_groups: tuple[Iterable[WireKey], ...] = (
        estimate._dependency_keys or (),
        estimate._dependency_reads or (),
        estimate._dependency_writes or (),
        (estimate._dependency_completion or {}).keys(),
        estimate._dependency_synchronized_entry_conditions.keys(),
        *(
            certificate.coverage
            for certificate in estimate._dependency_synchronized_entry_certificates
        ),
        *(
            certificate.frontier
            for certificate in estimate._dependency_synchronized_entry_certificates
        ),
    )
    for keys in key_groups:
        for key in keys:
            symbols.update(key_symbols(key))
    return symbols


def _dependency_keys_depend_on_symbol(
    keys: frozenset[WireKey] | None,
    symbol: sp.Symbol,
) -> bool:
    """Return whether a dependency footprint retains one local symbol.

    Args:
        keys (frozenset[WireKey] | None): Evaluated dependency footprint, or
            ``None`` when no precise footprint is available.
        symbol (sp.Symbol): Loop-local symbol to find.

    Returns:
        bool: Whether a scalar address or symbolic range uses ``symbol``.
    """
    if keys is None:
        return False
    for _owner, index in keys:
        if isinstance(index, _WireRangeIndex):
            if symbol in (
                sp.sympify(index.index_at_offset).free_symbols
                | sp.sympify(index.iterations).free_symbols
            ):
                return True
        elif isinstance(index, sp.Expr) and symbol in index.free_symbols:
            return True
    return False


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

    def project_at_value(key: WireKey, value: sp.Integer) -> WireKey:
        """Project one key for one concrete loop value.

        Args:
            key (WireKey): Body-scoped dependency address.
            value (sp.Integer): Concrete induction value.

        Returns:
            WireKey: Address specialized to that one iteration.
        """
        owner, index = key
        if isinstance(index, _WireRangeIndex):
            symbols = index.index_at_offset.free_symbols | index.iterations.free_symbols
            return (
                owner,
                index.mapped(
                    lambda expression: expression.subs(
                        symbol,
                        value,
                        simultaneous=True,
                    )
                )
                if symbol in symbols
                else index,
            )
        if isinstance(index, sp.Expr) and symbol in index.free_symbols:
            return (
                owner,
                _normalize_wire_index(
                    cast(
                        ResourceExpr,
                        index.subs(symbol, value, simultaneous=True),
                    )
                ),
            )
        return key

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
    projected_entry_conditions: dict[WireKey, Boolean] = {}
    for key, condition in estimate._dependency_synchronized_entry_conditions.items():
        if concrete_values is not None:
            for value in concrete_values:
                active = _boolean_condition(
                    condition.subs(symbol, value, simultaneous=True)
                )
                if active is sp.false:
                    continue
                projected = project_at_value(key, value)
                projected_entry_conditions[projected] = _boolean_condition(
                    sp.Or(
                        projected_entry_conditions.get(projected, sp.false),
                        active,
                    )
                )
            continue
        active = _boolean_condition(
            _activation_over_range(
                condition,
                symbol,
                start_expr,
                step_expr,
                iterations_expr,
            )
        )
        if active is sp.false:
            continue
        for projected in project(key):
            projected_entry_conditions[projected] = _boolean_condition(
                sp.Or(
                    projected_entry_conditions.get(projected, sp.false),
                    active,
                )
            )
    projected_certificates: list[_SynchronizedEntryCertificate] = []
    for certificate in estimate._dependency_synchronized_entry_certificates:
        projected_coverage = frozenset(
            projected for key in certificate.coverage for projected in project(key)
        )
        active = _boolean_condition(
            _activation_over_range(
                certificate.active_when,
                symbol,
                start_expr,
                step_expr,
                iterations_expr,
            )
        )
        if not projected_coverage or active is sp.false:
            continue
        projected_certificates.append(
            _SynchronizedEntryCertificate(
                coverage=projected_coverage,
                frontier=frozenset(),
                active_when=active,
            )
        )
    return dataclasses.replace(
        estimate,
        _dependency_keys=projected_keys,
        _dependency_reads=projected_reads,
        _dependency_writes=projected_writes,
        _dependency_completion=projected_completion,
        _dependency_synchronized_entry_conditions=projected_entry_conditions,
        _dependency_synchronized_entry_certificates=(
            _merge_synchronized_entry_certificates(projected_certificates)
        ),
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


def _map_synchronized_entry_conditions(
    conditions: Mapping[WireKey, sp.Basic],
    key_fn: Any,
    guard_fn: Any,
) -> dict[WireKey, Boolean]:
    """Rewrite guarded synchronized-entry addresses and conditions.

    Args:
        conditions (Mapping[WireKey, sp.Basic]): Requirements to rewrite.
        key_fn (Any): Symbolic rewrite callable for wire indices.
        guard_fn (Any): Symbolic rewrite callable for Boolean conditions.

    Returns:
        dict[WireKey, Boolean]: Rewritten nonfalse requirements with colliding
        addresses merged by logical OR.
    """
    mapped: dict[WireKey, Boolean] = {}
    for key, condition in conditions.items():
        active = _boolean_condition(guard_fn(condition))
        if active is sp.false:
            continue
        mapped_key = _map_dependency_key(key, key_fn)
        mapped[mapped_key] = _boolean_condition(
            sp.Or(mapped.get(mapped_key, sp.false), active)
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
