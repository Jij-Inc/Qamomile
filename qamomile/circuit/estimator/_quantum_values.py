"""Resolve quantum IR values to physical owners, indices, and widths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._dependency_indices import (
    WireKey,
    _normalize_wire_index,
    _specialize_dependency_expression,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import _expr
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.types.q_register import (
    QFixedType,
    QUIntType,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    split_indexed_identifier,
)


def _quantum_element_index_expression(
    value: Value,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> ResourceExpr | None:
    """Resolve an array element to its root-array index expression.

    Args:
        value (Value): Quantum scalar that may carry array-element ancestry.
        resolver (ExprResolver): Resolver for symbolic indices and view bounds.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values used only to resolve physical indices. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with dependency input
            names. Defaults to ``None``.

    Returns:
        ResourceExpr | None: Root-array index expression, or ``None`` when the
            value is not a one-dimensional array element.
    """
    if value.parent_array is None or len(value.element_indices) != 1:
        return None
    resolved = _resolve_root_array_index_expression(
        value.parent_array,
        resolver.resolve(value.element_indices[0]),
        resolver,
        scalar_values,
        used_names,
    )
    return None if resolved is None else resolved[1]


def _resolve_root_array_index_expression(
    array: ArrayValue,
    local_index: ResourceExpr,
    resolver: ExprResolver,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[ArrayValue, ResourceExpr] | None:
    """Compose one array index through a validated symbolic view chain.

    This is the resolver-aware counterpart of
    :func:`resolve_root_array_index`. Each concrete affine component is
    checked at the hop where it appears so malformed raw IR cannot compose an
    invalid negative stride into an apparently valid root index.

    Args:
        array (ArrayValue): Array whose local index is being resolved.
        local_index (ResourceExpr): Index relative to ``array``.
        resolver (ExprResolver): Resolver for symbolic view bounds.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        tuple[ArrayValue, ResourceExpr] | None: Root array and composed index,
        or ``None`` when a component is missing, unresolved for a concrete
        address, or violates the nonnegative-start/positive-step contract.
    """

    def valid_concrete_component(
        expression: ResourceExpr,
        *,
        positive: bool,
    ) -> bool:
        """Validate one concrete index, start, or stride expression.

        Args:
            expression (ResourceExpr): Component to inspect.
            positive (bool): Require strict positivity instead of
                nonnegativity.

        Returns:
            bool: ``False`` only for a concrete malformed component.
        """
        if not expression.is_number:
            return True
        if not _is_concrete_integer(expression):
            return False
        return bool(expression > 0) if positive else bool(expression >= 0)

    index = _specialize_dependency_expression(
        local_index,
        scalar_values,
        used_names,
    )
    if not valid_concrete_component(index, positive=False):
        return None
    current = array
    while current.slice_of is not None:
        if current.slice_start is None or current.slice_step is None:
            return None
        start = _specialize_dependency_expression(
            resolver.resolve(current.slice_start),
            scalar_values,
            used_names,
        )
        step = _specialize_dependency_expression(
            resolver.resolve(current.slice_step),
            scalar_values,
            used_names,
        )
        if not valid_concrete_component(
            start,
            positive=False,
        ) or not valid_concrete_component(step, positive=True):
            return None
        index = cast(ResourceExpr, start + step * index)
        current = current.slice_of
    return current, index


def _quantum_element_wire_index(
    value: Value,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> int | None:
    """Resolve an array element to its concrete root-array scalar index.

    Args:
        value (Value): Quantum scalar that may carry array-element ancestry.
        resolver (ExprResolver): Resolver for symbolic indices and view bounds.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values used only to resolve physical indices. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with dependency input
            names. Defaults to ``None``.

    Returns:
        int | None: Nonnegative scalar index in the root array, or ``None``
            when the element or any view mapping remains unresolved.
    """
    index = _quantum_element_index_expression(
        value,
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if (
        index is None
        or not index.is_number
        or not _is_concrete_integer(index)
        or index < 0
    ):
        return None
    return int(index)


def _array_wire_key_at_index(
    array: ArrayValue,
    index: ResourceExpr | int,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> WireKey:
    """Map one concrete array slot through any caller-side view chain.

    Args:
        array (ArrayValue): Actual array or view supplied at a call boundary.
        index (ResourceExpr | int): Concrete or symbolic element index
            relative to ``array``.
        resolver (ExprResolver): Resolver for view starts and strides.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        WireKey: Root-owner scalar key, including a symbolic root index when
            available, or an owner-wide key when the address cannot be
            resolved safely.
    """
    owner = _quantum_allocation_owner(array)
    resolved = _resolve_root_array_index_expression(
        array,
        _expr(index),
        resolver,
        scalar_values,
        used_names,
    )
    if resolved is None:
        return owner, None
    return owner, _normalize_wire_index(resolved[1])


def _quantum_root_array(value: Value) -> ArrayValue | None:
    """Return the root array allocation aliased by a quantum value.

    Args:
        value (Value): Quantum scalar, array, or array view.

    Returns:
        ArrayValue | None: Root array, or ``None`` for an independent scalar.
    """
    if isinstance(value, ArrayValue):
        array = value
    else:
        array = value.parent_array
    if array is None:
        return None
    while array.slice_of is not None:
        array = array.slice_of
    return array


def _runtime_carrier_wire_keys(
    value: Value | ArrayValue,
    allocation_owners_by_uuid: Mapping[str, str],
) -> set[WireKey] | None:
    """Recover physical keys stored by a synthetic quantum array carrier.

    Tuple-form expectation values and similar frontend adapters pack otherwise
    unrelated qubits into an ``ArrayValue``.  The carrier's own logical ID is
    not physical storage; its runtime metadata retains each standalone logical
    ID or root-array UUID/index instead.

    Args:
        value (Value | ArrayValue): Candidate synthetic quantum carrier.
        allocation_owners_by_uuid (Mapping[str, str]): Root allocation UUIDs
            mapped to logical scheduler owners.

    Returns:
        set[WireKey] | None: Physical element keys, or ``None`` when no complete
            carrier metadata is present.
    """
    runtime = value.metadata.array_runtime
    if runtime is None or not runtime.element_uuids or not runtime.element_logical_ids:
        return None
    if len(runtime.element_uuids) != len(runtime.element_logical_ids):
        return None
    parent_addresses = value.get_element_parent_addresses()
    keys: set[WireKey] = set()
    for index, logical_id in enumerate(runtime.element_logical_ids):
        address = parent_addresses[index] if index < len(parent_addresses) else None
        if address is not None:
            parent_uuid, parent_index = address
            owner = allocation_owners_by_uuid.get(parent_uuid)
            if owner is not None:
                keys.add((owner, parent_index))
                continue
        if (
            index < len(runtime.element_parent_uuids)
            and index < len(runtime.element_parent_indices)
            and runtime.element_parent_uuids[index]
            and runtime.element_parent_indices[index] < 0
        ):
            owner = allocation_owners_by_uuid.get(runtime.element_parent_uuids[index])
            if owner is not None:
                keys.add((owner, None))
                continue
        indexed = split_indexed_identifier(logical_id)
        if indexed is not None:
            owner, scalar_index = indexed
            keys.add((owner, int(scalar_index)))
        else:
            keys.add((logical_id, None))
    return keys


def _cast_carrier_wire_keys(value: Value) -> set[WireKey] | None:
    """Return physical dependency keys recorded by quantum cast metadata.

    Concrete vector-to-register casts retain root-space logical identifiers
    such as ``"<root>_3"`` for every carrier. Symbolic casts cannot enumerate
    carriers, so their source logical ID remains an owner-wide conservative
    alias.

    Args:
        value (Value): Candidate cast result.

    Returns:
        set[WireKey] | None: Carrier keys, an owner-wide fallback, or ``None``
            when ``value`` is not a cast result.
    """
    if not value.is_cast_result():
        return None
    logical_ids = value.get_cast_qubit_logical_ids() or ()
    if logical_ids:
        keys: set[WireKey] = set()
        for logical_id in logical_ids:
            indexed = split_indexed_identifier(logical_id)
            if indexed is None:
                keys.add((logical_id, None))
                continue
            owner, index = indexed
            keys.add((owner, int(index)))
        return keys
    source = value.get_cast_source_logical_id()
    return {(source, None)} if source is not None else {(value.logical_id, None)}


def _quantum_allocation_owner(value: Value) -> str:
    """Return the root logical allocation identity for a quantum value.

    Array elements and sliced arrays have their own SSA/logical identities but
    alias storage owned by the root array allocation. Treating those views as
    fresh returned qubits inflates liveness by one per element operation.

    Args:
        value (Value): Quantum scalar, array, or array view.

    Returns:
        str: Root array logical ID for a view/element, otherwise the value's
        own logical ID.
    """
    carrier_keys = _cast_carrier_wire_keys(value)
    if carrier_keys:
        owners = {owner for owner, _index in carrier_keys}
        if len(owners) == 1:
            return next(iter(owners))
    array = _quantum_root_array(value)
    if array is None:
        return value.logical_id
    return array.logical_id


def _qubit_value_size(value: Value, resolver: ExprResolver) -> ResourceExpr:
    """Return the logical width represented by one quantum value.

    Args:
        value (Value): Scalar or array quantum value.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        ResourceExpr: Number of represented qubits.
    """
    if isinstance(value, ArrayValue) and isinstance(value.type, QubitType):
        runtime = value.metadata.array_runtime
        if runtime is not None and runtime.element_uuids and not value.shape:
            return sp.Integer(len(runtime.element_uuids))
        count: ResourceExpr = _ONE
        for dim in value.shape:
            count *= resolver.resolve(dim)
        return count
    if isinstance(value.type, QubitType):
        return _ONE
    if isinstance(value.type, QUIntType):
        return _quantum_register_component_size(value.type.width, resolver)
    if isinstance(value.type, QFixedType):
        return _quantum_register_component_size(
            value.type.integer_bits,
            resolver,
        ) + _quantum_register_component_size(
            value.type.fractional_bits,
            resolver,
        )
    return _ZERO


def _quantum_register_component_size(
    component: int | Value,
    resolver: ExprResolver,
) -> ResourceExpr:
    """Resolve one integer component of a quantum-register width.

    Args:
        component (int | Value): Concrete bit count or symbolic UInt value.
        resolver (ExprResolver): Resolver for symbolic register widths.

    Returns:
        ResourceExpr: Resolved nonnegative width component expression.
    """
    if isinstance(component, Value):
        return resolver.resolve(component)
    return sp.Integer(component)


def _count_qinit(operation: QInitOperation, resolver: ExprResolver) -> ResourceExpr:
    """Count qubits allocated by a qinit operation.

    Args:
        operation (QInitOperation): Qubit initialization operation.
        resolver (ExprResolver): Resolver for symbolic array shapes.

    Returns:
        ResourceExpr: Allocated-qubit count.
    """
    result = operation.results[0]
    if isinstance(result, ArrayValue) and isinstance(result.type, QubitType):
        count: ResourceExpr = _ONE
        for dim in result.shape:
            count *= resolver.resolve(dim)
        return count
    if isinstance(result.type, QubitType):
        return _ONE
    return _ZERO
