"""Map dependency metadata across callable boundaries."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
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
from qamomile.circuit.estimator._dependency_footprints import (
    _expand_dependency_owner_aliases,
    _quantum_value_wire_keys,
)
from qamomile.circuit.estimator._dependency_indices import (
    _UNKNOWN_WIRE_INDEX,
    WireKey,
    _WireRangeIndex,
)
from qamomile.circuit.estimator._dependency_synchronization import (
    _merge_synchronized_entry_certificates,
    _SynchronizedEntryCertificate,
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


def _map_body_synchronized_entry_conditions(
    block: Block,
    body_estimate: ResourceEstimate,
    actual_operands: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> dict[WireKey, Boolean]:
    """Translate callee entry requirements onto caller input operands.

    Only formal quantum inputs participate. A returned callee-local
    allocation is not an entry dependency and therefore must not be mapped
    through the callable's output/result pairing.

    Args:
        block (Block): Evaluated callable implementation.
        body_estimate (ResourceEstimate): Body estimate carrying guarded entry
            requirements.
        actual_operands (Sequence[ValueBase]): Caller operands aligned with
            the block inputs.
        resolver (ExprResolver): Caller-side value resolver.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        dict[WireKey, Boolean]: Caller-scoped guarded entry requirements.
    """
    conditions = body_estimate._dependency_synchronized_entry_conditions
    if not conditions:
        return {}
    mapped: dict[WireKey, Boolean] = {}
    for formal, actual in pair_block_operands(block, actual_operands):
        if not (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and formal.type.is_quantum()
            and actual.type.is_quantum()
        ):
            continue
        source_owner = _quantum_allocation_owner(formal)
        for key, condition in conditions.items():
            if key[0] != source_owner:
                continue
            caller_keys = _map_value_dependency_keys(
                formal,
                actual,
                frozenset((key,)),
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            for caller_key in caller_keys:
                mapped[caller_key] = _boolean_condition(
                    sp.Or(mapped.get(caller_key, sp.false), condition)
                )
    return mapped


def _map_or_widen_certificate_coverage_key(
    source: Value,
    actual: Value,
    key: WireKey,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
) -> set[WireKey]:
    """Map one formal coverage key without silently losing caller scope.

    Args:
        source (Value): Callee formal quantum value owning ``key``.
        actual (Value): Caller quantum value paired with ``source``.
        key (WireKey): One callee synchronized-entry coverage key.
        resolver (ExprResolver): Caller resolver for arrays and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.
        owner_aliases (Mapping[str, frozenset[str]] | None): Optional caller
            owner aliases to include conservatively. Defaults to ``None``.

    Returns:
        set[WireKey]: Nonempty caller coverage, widened to the actual operand
            footprint or owner when exact key mapping is unavailable.
    """
    mapped = _map_value_dependency_keys(
        source,
        actual,
        frozenset((key,)),
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if not mapped:
        mapped = _quantum_value_wire_keys(
            actual,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
            owner_aliases=owner_aliases,
        )
    if not mapped:
        mapped = {(_quantum_allocation_owner(actual), None)}
    return _expand_dependency_owner_aliases(mapped, owner_aliases)


def _map_exact_certificate_frontier_key(
    source: Value,
    actual: Value,
    key: WireKey,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
) -> WireKey | None:
    """Map one formal frontier key only through an exact scalar image.

    Args:
        source (Value): Callee formal quantum value owning ``key``.
        actual (Value): Caller quantum value paired with ``source``.
        key (WireKey): One callee first-gate frontier key.
        resolver (ExprResolver): Caller resolver for arrays and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.
        owner_aliases (Mapping[str, frozenset[str]] | None): Optional caller
            owner aliases. Any nontrivial expansion rejects the frontier.
            Defaults to ``None``.

    Returns:
        WireKey | None: Unique caller root-scalar image, or ``None`` when the
            mapping widens, aliases, or remains unresolved.
    """
    source_index = key[1]
    if source_index is _UNKNOWN_WIRE_INDEX or isinstance(
        source_index,
        _WireRangeIndex,
    ):
        return None
    if source_index is None and (
        isinstance(source, ArrayValue) or source.parent_array is not None
    ):
        return None
    if actual.is_cast_result() or actual.metadata.array_runtime is not None:
        return None
    mapped = _map_value_dependency_keys(
        source,
        actual,
        frozenset((key,)),
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    expanded = _expand_dependency_owner_aliases(mapped, owner_aliases)
    if len(expanded) != 1:
        return None
    caller_key = next(iter(expanded))
    caller_index = caller_key[1]
    if caller_index is _UNKNOWN_WIRE_INDEX or isinstance(
        caller_index,
        _WireRangeIndex,
    ):
        return None
    if caller_index is None and (
        isinstance(actual, ArrayValue) or actual.parent_array is not None
    ):
        return None
    return caller_key


def _map_body_synchronized_entry_certificates(
    block: Block,
    body_estimate: ResourceEstimate,
    actual_operands: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
) -> tuple[_SynchronizedEntryCertificate, ...]:
    """Translate grouped callee entry certificates onto caller inputs.

    Callee-local and output-only owners have no caller entry state and are
    removed. Coverage may widen when a formal input cannot be mapped exactly,
    but it never disappears. The entire frontier is cleared unless every
    formal frontier key has exactly one root-scalar caller image.

    Args:
        block (Block): Evaluated callable implementation.
        body_estimate (ResourceEstimate): Body estimate carrying grouped entry
            certificates.
        actual_operands (Sequence[ValueBase]): Caller operands aligned with
            the block inputs.
        resolver (ExprResolver): Caller resolver for arrays and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.
        owner_aliases (Mapping[str, frozenset[str]] | None): Optional caller
            owner aliases used to reject ambiguous frontiers and widen
            coverage. Defaults to ``None``.

    Returns:
        tuple[_SynchronizedEntryCertificate, ...]: Caller-scoped grouped
            certificates with fail-closed frontiers.
    """
    certificates = body_estimate._dependency_synchronized_entry_certificates
    if not certificates:
        return ()
    pairs_by_owner: dict[str, list[tuple[Value, Value]]] = {}
    for formal, actual in pair_block_operands(block, actual_operands):
        if not (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and formal.type.is_quantum()
            and actual.type.is_quantum()
        ):
            continue
        pairs_by_owner.setdefault(_quantum_allocation_owner(formal), []).append(
            (formal, actual)
        )

    mapped_certificates: list[_SynchronizedEntryCertificate] = []
    for certificate in certificates:
        mapped_coverage: set[WireKey] = set()
        for key in certificate.coverage:
            for source, actual in pairs_by_owner.get(key[0], ()):
                mapped_coverage.update(
                    _map_or_widen_certificate_coverage_key(
                        source,
                        actual,
                        key,
                        resolver,
                        scalar_values=scalar_values,
                        used_names=used_names,
                        owner_aliases=owner_aliases,
                    )
                )
        if not mapped_coverage:
            continue

        mapped_frontier: set[WireKey] = set()
        frontier_is_exact = True
        for key in certificate.frontier:
            source_pairs = pairs_by_owner.get(key[0], ())
            if not source_pairs:
                continue
            key_images: set[WireKey] = set()
            for source, actual in source_pairs:
                image = _map_exact_certificate_frontier_key(
                    source,
                    actual,
                    key,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                    owner_aliases=owner_aliases,
                )
                if image is None:
                    frontier_is_exact = False
                    break
                key_images.add(image)
            if not frontier_is_exact or len(key_images) != 1:
                frontier_is_exact = False
                break
            mapped_frontier.update(key_images)
        if not frontier_is_exact:
            mapped_frontier.clear()
        mapped_certificates.append(
            _SynchronizedEntryCertificate(
                coverage=frozenset(mapped_coverage),
                frontier=frozenset(mapped_frontier),
                active_when=certificate.active_when,
            )
        )
    return _merge_synchronized_entry_certificates(mapped_certificates)
