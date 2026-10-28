"""Collect classical and quantum dependency footprints."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._dependency_indices import (
    _MAX_CONCRETE_VIEW_WIRE_EXPANSION,
    _UNKNOWN_WIRE_INDEX,
    WireKey,
    _normalize_wire_index,
    _specialize_dependency_expression,
)
from qamomile.circuit.estimator._quantum_values import (
    _array_wire_key_at_index,
    _cast_carrier_wire_keys,
    _quantum_allocation_owner,
    _quantum_element_index_expression,
    _qubit_value_size,
    _runtime_carrier_wire_keys,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import _boolean_condition
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)

_WireFootprint = tuple[frozenset[WireKey], frozenset[WireKey]]


_CLASSICAL_DEPENDENCY_OWNER_PREFIX = "$resource_classical:"


def _classical_dependency_key(uuid: str) -> WireKey:
    """Return the scheduler key for one immutable classical SSA value.

    Args:
        uuid (str): Classical value UUID.

    Returns:
        WireKey: Dependency key used to carry the value's readiness vector.
    """
    return (f"{_CLASSICAL_DEPENDENCY_OWNER_PREFIX}{uuid}", None)


def _is_classical_dependency_key(key: WireKey) -> bool:
    """Return whether a dependency key denotes immutable classical data.

    Args:
        key (WireKey): Scheduler dependency key.

    Returns:
        bool: Whether ``key`` is a classical SSA readiness token.
    """
    return key[0].startswith(_CLASSICAL_DEPENDENCY_OWNER_PREFIX)


def _classical_dependency_footprint(
    source_token_conditions: Mapping[str, Boolean],
    written_source_token_conditions: Mapping[str, Boolean],
) -> tuple[frozenset[WireKey], frozenset[WireKey], Boolean]:
    """Collect classical source-token reads and result writes.

    Classical facts have already resolved structural SSA ancestry into stable
    source tokens. Reads constrain an operation's start, while only newly
    published result tokens receive a completion write.

    Args:
        source_token_conditions (Mapping[str, Boolean]): Classical source
            tokens read by the operation and their activation guards.
        written_source_token_conditions (Mapping[str, Boolean]): Classical
            source tokens first published by the operation's results.

    Returns:
        tuple[frozenset[WireKey], frozenset[WireKey], Boolean]: Read tokens,
            written tokens, and the union of their activation conditions.
    """
    read_uuids = {
        token
        for token, condition in source_token_conditions.items()
        if _boolean_condition(condition) is not sp.false
    }
    written_uuids = {
        token
        for token, condition in written_source_token_conditions.items()
        if _boolean_condition(condition) is not sp.false
    }
    conditions = [
        *(source_token_conditions[token] for token in read_uuids),
        *(written_source_token_conditions[token] for token in written_uuids),
    ]
    active = _boolean_condition(sp.Or(*conditions)) if conditions else sp.false
    return (
        frozenset(_classical_dependency_key(uuid) for uuid in read_uuids),
        frozenset(_classical_dependency_key(uuid) for uuid in written_uuids),
        active,
    )


def _quantum_value_wire_keys(
    value: Value | ArrayValue,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> set[WireKey]:
    """Return dependency keys for one quantum value.

    A scalar array element uses its root allocation and physical scalar index.
    Small concrete views expand to their physical scalar keys. Whole
    registers and larger views use an owner-wide key, while an unresolved
    scalar retains an unknown-scalar marker. Independent scalar qubits use
    their own logical identity with an owner-wide key.

    Args:
        value (Value | ArrayValue): Quantum scalar, array, or array view.
        resolver (ExprResolver): Resolver for element and view indices.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.
        owner_aliases (Mapping[str, frozenset[str]] | None): Optional
            conditional-result owners mapped to every physical owner they may
            select. Defaults to ``None``.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional root
            allocation UUID-to-logical-owner mapping for synthetic carriers.
            Defaults to ``None``.

    Returns:
        set[WireKey]: Root-owner and optional scalar-index keys.
    """
    carrier_keys = _runtime_carrier_wire_keys(
        value,
        allocation_owners_by_uuid or {},
    )
    if carrier_keys is None:
        carrier_keys = _cast_carrier_wire_keys(value)
    if carrier_keys is not None:
        return _expand_dependency_owner_aliases(carrier_keys, owner_aliases)
    owner = _quantum_allocation_owner(value)
    if isinstance(value, ArrayValue):
        if value.slice_of is None:
            return _expand_dependency_owner_aliases(
                {(owner, None)},
                owner_aliases,
            )
        size = _specialize_dependency_expression(
            _qubit_value_size(value, resolver),
            scalar_values,
            used_names,
        )
        if (
            size.is_number
            and _is_concrete_integer(size)
            and 0 <= size <= _MAX_CONCRETE_VIEW_WIRE_EXPANSION
        ):
            keys = {
                _array_wire_key_at_index(
                    value,
                    index,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
                for index in range(int(size))
            }
            return _expand_dependency_owner_aliases(keys, owner_aliases)
        return _expand_dependency_owner_aliases(
            {(owner, None)},
            owner_aliases,
        )
    if value.parent_array is None:
        return _expand_dependency_owner_aliases(
            {(owner, None)},
            owner_aliases,
        )
    index = _quantum_element_index_expression(
        value,
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    return _expand_dependency_owner_aliases(
        {
            (
                owner,
                (None if index is None else _normalize_wire_index(index)),
            )
        },
        owner_aliases,
    )


def _expand_dependency_owner_aliases(
    keys: set[WireKey],
    owner_aliases: Mapping[str, frozenset[str]] | None,
) -> set[WireKey]:
    """Add every transitive physical-owner alias for dependency keys.

    Conditional merge results have their own SSA owner even though a later
    access physically touches one of the branch-source allocations. Retaining
    both the result key and all possible source keys lets call-boundary mapping
    see the result while preventing either possible source from running in the
    same layer.

    Args:
        keys (set[WireKey]): Dependency keys before alias expansion.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases, or ``None`` when no aliases are known.

    Returns:
        set[WireKey]: Original keys plus transitive owner aliases at the same
        scalar index.
    """
    if not owner_aliases:
        return keys
    expanded = set(keys)
    pending = list(keys)
    while pending:
        owner, index = pending.pop()
        for alias in owner_aliases.get(owner, frozenset()):
            key = (alias, index)
            if key in expanded:
                continue
            expanded.add(key)
            pending.append(key)
    return expanded


def _iter_quantum_carrier_values(value: ValueBase) -> Sequence[Value | ArrayValue]:
    """Return scalar or array quantum leaves nested in one IR carrier.

    Args:
        value (ValueBase): Scalar, array, tuple, or dictionary carrier.

    Returns:
        Sequence[Value | ArrayValue]: Quantum leaves in deterministic order.
    """
    if isinstance(value, TupleValue):
        return tuple(
            leaf
            for element in value.elements
            for leaf in _iter_quantum_carrier_values(element)
        )
    if isinstance(value, DictValue):
        return tuple(
            leaf
            for key, entry_value in value.entries
            for element in (key, entry_value)
            for leaf in _iter_quantum_carrier_values(element)
        )
    if isinstance(value, (Value, ArrayValue)) and value.type.is_quantum():
        return (value,)
    return ()


def _wire_keys_for_values(
    values: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[WireKey]:
    """Collect caller dependency keys for quantum values.

    Args:
        values (Sequence[ValueBase]): Values whose physical keys are needed.
        resolver (ExprResolver): Resolver for array elements and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[WireKey]: Union of all quantum value footprints.
    """
    keys: set[WireKey] = set()
    for value in values:
        for quantum_value in _iter_quantum_carrier_values(value):
            keys.update(
                _quantum_value_wire_keys(
                    quantum_value,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    return keys


def _controlled_u_control_wire_keys(
    operation: ControlledUOperation,
    resolved_indices: Sequence[ResourceExpr],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[WireKey]:
    """Return only the active control-pool wires of a controlled call.

    Args:
        operation (ControlledUOperation): Controlled operation whose control
            prefix may be a selected array pool.
        resolved_indices (Sequence[ResourceExpr]): Resolved ``control_indices``
            values, empty when the whole prefix is active.
        resolver (ExprResolver): Caller-side value resolver.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[WireKey]: Selected physical control keys. Symbolic selections keep
            symbolic scalar addresses, and malformed or otherwise unresolved
            selections keep an unknown-scalar marker.
    """
    controls = operation.control_operands
    if not resolved_indices:
        return _wire_keys_for_values(
            controls,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
    if len(controls) != 1 or not isinstance(controls[0], ArrayValue):
        return _wire_keys_for_values(
            controls,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
    pool = controls[0]
    keys: set[WireKey] = set()
    for index in resolved_indices:
        keys.add(
            _array_wire_key_at_index(
                pool,
                index,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
        )
    return keys


def _quantum_wire_keys(
    operation: Operation,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> tuple[set[WireKey], set[WireKey]]:
    """Collect quantum logical wires read and written by one operation.

    Args:
        operation (Operation): Operation whose dependency footprint is needed.
        resolver (ExprResolver): Resolver for array element and view indices.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.
        owner_aliases (Mapping[str, frozenset[str]] | None): Optional
            conditional-result owner aliases. Defaults to ``None``.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional root
            allocation UUID-to-logical-owner mapping for synthetic carriers.
            Defaults to ``None``.

    Returns:
        tuple[set[WireKey], set[WireKey]]: Physical owner/index keys read and
            written. Nested control flow conservatively treats every touched
            wire as both read and written at the enclosing boundary.
    """
    reads: set[WireKey] = set()
    for value in operation.all_input_values():
        for quantum_value in _iter_quantum_carrier_values(value):
            reads.update(
                _quantum_value_wire_keys(
                    quantum_value,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                    owner_aliases=owner_aliases,
                    allocation_owners_by_uuid=allocation_owners_by_uuid,
                )
            )
    writes: set[WireKey] = set()
    for value in operation.results:
        for quantum_value in _iter_quantum_carrier_values(value):
            writes.update(
                _quantum_value_wire_keys(
                    quantum_value,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                    owner_aliases=owner_aliases,
                    allocation_owners_by_uuid=allocation_owners_by_uuid,
                )
            )
    if isinstance(operation, HasNestedOps):
        nested_keys: set[WireKey] = set()
        for body in operation.nested_op_lists():
            for child in body:
                child_reads, child_writes = _quantum_wire_keys(
                    child,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                    owner_aliases=owner_aliases,
                    allocation_owners_by_uuid=allocation_owners_by_uuid,
                )
                nested_keys |= child_reads | child_writes
        reads |= nested_keys
        writes |= nested_keys
    return reads, writes


def _operation_has_unresolved_quantum_index(
    operation: Operation,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> bool:
    """Return whether a quantum element aliases its whole owner conservatively.

    Args:
        operation (Operation): Operation whose quantum values are inspected.
        resolver (ExprResolver): Resolver for element and view expressions.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        bool: Whether any scalar quantum element remains physically unresolved.
    """
    values = [*operation.all_input_values(), *operation.results]
    for value in values:
        if isinstance(value, Value) and value.type.is_quantum():
            runtime = value.metadata.array_runtime
            if runtime is not None and any(
                parent_uuid and parent_index < 0
                for parent_uuid, parent_index in zip(
                    runtime.element_parent_uuids,
                    runtime.element_parent_indices,
                )
            ):
                return True
        if (
            isinstance(value, ArrayValue)
            and value.type.is_quantum()
            and value.slice_of is not None
        ):
            keys = _quantum_value_wire_keys(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            owner = _quantum_allocation_owner(value)
            if (owner, None) in keys or (owner, _UNKNOWN_WIRE_INDEX) in keys:
                return True
            continue
        if (
            not isinstance(value, Value)
            or not value.type.is_quantum()
            or value.parent_array is None
        ):
            continue
        index = _quantum_element_index_expression(
            value,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        if (
            index is not None
            and _normalize_wire_index(index) is not _UNKNOWN_WIRE_INDEX
        ):
            continue
        return True
    return False
