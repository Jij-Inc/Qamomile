"""Dependency scheduling and logical-width liveness analysis.

The helpers in this module map IR values to physical wire identities, schedule
resource summaries along those dependencies, and track allocation liveness.
They consume metric records without owning interpretation or decomposition
policy.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from collections import ChainMap
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._metrics import (
    _ONE,
    _ZERO,
    DepthResources,
    EstimateQuality,
    ResourceAssumption,
    ResourceExpr,
    WidthResources,
    _and_conditions,
    _boolean_condition,
    _conditional_depth,
    _ConditionIndicator,
    _expr,
    _is_concrete_integer,
    _is_structurally_nonnegative,
    _maximum_expr_over_range,
    _piecewise,
    _resource_activity_condition,
    _resource_expr,
    _resource_max,
    _resource_max_many,
    _safe_simplify,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
    input_shape_dimension_aliases,
)
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.types.q_register import QFixedType, QUIntType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    split_indexed_identifier,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
    from qamomile.circuit.frontend.qkernel import QKernel


class _WireIndexMarker(enum.Enum):
    """Identify a scalar wire address that could not be resolved."""

    UNKNOWN = enum.auto()


_UNKNOWN_WIRE_INDEX = _WireIndexMarker.UNKNOWN


@dataclasses.dataclass(frozen=True)
class _WireRangeIndex:
    """Describe scalar addresses touched across a symbolic loop range.

    Args:
        index_at_offset (ResourceExpr): Root-array index as a function of the
            canonical zero-based iteration offset.
        iterations (ResourceExpr): Number of offsets represented.
    """

    index_at_offset: ResourceExpr
    iterations: ResourceExpr

    def mapped(self, fn: Any) -> _WireRangeIndex:
        """Rewrite symbols in both range expressions.

        Args:
            fn (Any): Symbolic expression rewrite callable.

        Returns:
            _WireRangeIndex: Rewritten symbolic range descriptor.
        """
        return _WireRangeIndex(
            index_at_offset=cast(ResourceExpr, fn(self.index_at_offset)),
            iterations=cast(ResourceExpr, fn(self.iterations)),
        )


WireIndex = int | ResourceExpr | _WireIndexMarker | _WireRangeIndex | None
WireKey = tuple[str, WireIndex]
_WireFootprint = tuple[frozenset[WireKey], frozenset[WireKey]]
_WIRE_RANGE_OFFSET = sp.Dummy(
    "wire_range_offset",
    integer=True,
    nonnegative=True,
)
_MAX_EXACT_LOOP_WIRE_EXPANSION = 256
_MAX_EXACT_LOOP_DISJOINTNESS_EXPANSION = 4096
_MAX_CONCRETE_VIEW_WIRE_EXPANSION = 4096
_CLASSICAL_DEPENDENCY_OWNER_PREFIX = "$resource_classical:"


class _WireRelation(enum.Enum):
    """Classify whether two physical wire addresses overlap."""

    DEFINITE_OVERLAP = enum.auto()
    DISJOINT = enum.auto()
    POSSIBLE_OVERLAP = enum.auto()


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


def _symbolic_wire_range_index(
    index: ResourceExpr,
    symbol: sp.Symbol,
    *,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> _WireRangeIndex:
    """Canonicalize one loop-dependent scalar address range.

    Args:
        index (ResourceExpr): Root-array index using the local loop symbol.
        symbol (sp.Symbol): Local loop symbol leaving scope.
        start (ResourceExpr): First Python-range value.
        step (ResourceExpr): Python-range step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        _WireRangeIndex: Binder-independent range descriptor.
    """
    index_at_offset = cast(
        ResourceExpr,
        index.subs(
            symbol,
            start + step * _WIRE_RANGE_OFFSET,
            simultaneous=True,
        ),
    )
    return _WireRangeIndex(
        index_at_offset=_safe_simplify(index_at_offset),
        iterations=_safe_simplify(iterations),
    )


@dataclasses.dataclass
class _OwnerWireIndices:
    """Index previously scheduled addresses for one allocation owner.

    Args:
        concrete (set[int]): Concrete scalar indices.
        symbolic (set[ResourceExpr]): Symbolic scalar indices.
        symbolic_offsets (dict[ResourceExpr, dict[ResourceExpr, ResourceExpr]]):
            Symbolic indices grouped by their nonconstant expression and
            additive offset.
        residual_symbolic (set[ResourceExpr]): Symbolic indices without a
            usable additive-offset signature.
        ranges (set[_WireRangeIndex]): Symbolic loop-range footprints.
        whole_owner (bool): Whether the whole allocation was touched.
        unknown (bool): Whether an unresolved scalar footprint was touched.
    """

    concrete: set[int] = dataclasses.field(default_factory=set)
    symbolic: set[ResourceExpr] = dataclasses.field(default_factory=set)
    symbolic_offsets: dict[
        ResourceExpr,
        dict[ResourceExpr, ResourceExpr],
    ] = dataclasses.field(default_factory=dict)
    residual_symbolic: set[ResourceExpr] = dataclasses.field(default_factory=set)
    ranges: set[_WireRangeIndex] = dataclasses.field(default_factory=set)
    whole_owner: bool = False
    unknown: bool = False

    def add(self, index: WireIndex) -> None:
        """Record one owner-local wire address.

        Args:
            index (WireIndex): Concrete, symbolic, owner-wide, or unknown
                address to record.
        """
        if index is None:
            self.whole_owner = True
        elif index is _UNKNOWN_WIRE_INDEX:
            self.unknown = True
        elif isinstance(index, int):
            self.concrete.add(index)
        elif isinstance(index, _WireRangeIndex):
            self.ranges.add(index)
        else:
            self.symbolic.add(index)
            signature = _symbolic_offset_signature(index)
            if signature is None:
                self.residual_symbolic.add(index)
            else:
                base, offset = signature
                self.symbolic_offsets.setdefault(base, {})[offset] = index

    def candidates(self, index: WireIndex) -> tuple[WireIndex, ...]:
        """Return prior addresses that may overlap a queried address.

        Concrete queries use constant-time lookup for other concrete
        addresses and compare only against symbolic addresses. Symbolic,
        owner-wide, and unknown queries must inspect every prior address.

        Args:
            index (WireIndex): Owner-local address being queried.

        Returns:
            tuple[WireIndex, ...]: Prior addresses whose relation may be
                overlap rather than proven disjointness.
        """
        common: list[WireIndex] = []
        if self.whole_owner:
            common.append(None)
        if self.unknown:
            common.append(_UNKNOWN_WIRE_INDEX)
        if isinstance(index, int):
            if index in self.concrete:
                common.append(index)
            common.extend(self.symbolic)
            common.extend(self.ranges)
            return tuple(common)
        common.extend(self.concrete)
        if isinstance(index, sp.Expr):
            signature = _symbolic_offset_signature(index)
            if signature is None:
                common.extend(self.symbolic)
            else:
                base, offset = signature
                same_family = self.symbolic_offsets.get(base)
                if same_family is not None and offset in same_family:
                    common.append(same_family[offset])
                common.extend(self.residual_symbolic)
                for other_base, members in self.symbolic_offsets.items():
                    if other_base != base:
                        common.extend(members.values())
        else:
            common.extend(self.symbolic)
        common.extend(self.ranges)
        return tuple(common)


def _symbolic_offset_signature(
    index: ResourceExpr,
) -> tuple[ResourceExpr, ResourceExpr] | None:
    """Split a symbolic integer index into a base and numeric offset.

    Equal bases with distinct numeric offsets are provably disjoint without
    comparing every pair through SymPy. Indices with different bases remain
    candidates because they may still coincide for some input values.

    Args:
        index (ResourceExpr): Normalized symbolic wire index.

    Returns:
        tuple[ResourceExpr, ResourceExpr] | None: Nonconstant base and numeric
            additive offset, or ``None`` when this bounded proof does not
            apply.
    """
    offset, base = index.as_coeff_Add()
    normalized_base = _safe_simplify(cast(ResourceExpr, base))
    normalized_offset = _safe_simplify(cast(ResourceExpr, offset))
    if (
        normalized_base == _ZERO
        or normalized_offset.is_number is not True
        or normalized_offset.is_finite is not True
    ):
        return None
    return normalized_base, normalized_offset


def _normalize_wire_index(expression: ResourceExpr | int) -> WireIndex:
    """Normalize one root-array index for dependency comparisons.

    Args:
        expression (ResourceExpr | int): Concrete or symbolic root-array
            index.

    Returns:
        WireIndex: A concrete nonnegative integer, a simplified symbolic
            expression, or the unknown marker when the expression is not
            provably a finite integer or a concrete value is negative.
    """
    normalized = _safe_simplify(_expr(expression))
    if not normalized.is_number:
        return (
            normalized
            if normalized.is_integer is True and normalized.is_finite is True
            else _UNKNOWN_WIRE_INDEX
        )
    if not _is_concrete_integer(normalized) or normalized < 0:
        return _UNKNOWN_WIRE_INDEX
    return int(normalized)


@functools.lru_cache(maxsize=4096)
def _wire_index_relation(left: WireIndex, right: WireIndex) -> _WireRelation:
    """Classify overlap between two indices of the same allocation owner.

    ``None`` is the owner-wide address and therefore overlaps every scalar
    address. The unknown marker may overlap any same-owner address. Symbolic
    scalar addresses are compared only with deterministic SymPy facts; an
    unproved relation remains conservative.

    Args:
        left (WireIndex): First owner-local wire index.
        right (WireIndex): Second owner-local wire index.

    Returns:
        _WireRelation: Proven overlap, proven disjointness, or possible
            overlap.
    """
    if left is None or right is None:
        return _WireRelation.DEFINITE_OVERLAP
    if left is _UNKNOWN_WIRE_INDEX or right is _UNKNOWN_WIRE_INDEX:
        return _WireRelation.POSSIBLE_OVERLAP
    if left == right:
        return _WireRelation.DEFINITE_OVERLAP
    if isinstance(left, _WireRangeIndex) or isinstance(right, _WireRangeIndex):
        return _WireRelation.POSSIBLE_OVERLAP
    difference = _safe_simplify(cast(ResourceExpr, _expr(left) - _expr(right)))
    if difference == _ZERO or difference.is_zero is True:
        return _WireRelation.DEFINITE_OVERLAP
    if difference.is_zero is False:
        return _WireRelation.DISJOINT
    return _WireRelation.POSSIBLE_OVERLAP


def _estimate_has_nonzero_depth(estimate: ResourceEstimate) -> bool:
    """Return whether any depth metric is structurally nonzero.

    Args:
        estimate (ResourceEstimate): Estimate whose depth should be inspected.

    Returns:
        bool: Whether at least one depth field is not the exact zero
            expression.
    """
    return any(
        getattr(estimate.depth, field.name) != _ZERO
        for field in dataclasses.fields(DepthResources)
    )


def _estimate_depth_activity_condition(estimate: ResourceEstimate) -> Boolean:
    """Return when at least one declared depth field is active.

    Aggregate opaque costs may provide category depths without also providing
    ``depth``. Such a declaration still participates in dependency barriers
    for every category, including categories where its own duration is zero.

    Args:
        estimate (ResourceEstimate): Estimate whose depth fields define
            operation activity.

    Returns:
        Boolean: Symbolic condition under which any depth field is nonzero.
    """
    active: Boolean = sp.false
    for field in dataclasses.fields(DepthResources):
        field_active = _resource_activity_condition(
            cast(ResourceExpr, getattr(estimate.depth, field.name))
        )
        if field_active is sp.true:
            return sp.true
        if field_active is not sp.false:
            active = cast(Boolean, sp.Or(active, field_active))
    return active


def _scheduled_depth_activity_conditions(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
) -> tuple[Boolean, ...]:
    """Compute depth-activity guards once for a scheduled operation list.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations
            paired with their resource summaries in program order.

    Returns:
        tuple[Boolean, ...]: Activity guards aligned with ``scheduled``.
    """
    return tuple(
        _estimate_depth_activity_condition(estimate)
        for _operation, estimate in scheduled
    )


def _merge_dependency_keys(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> frozenset[WireKey] | None:
    """Merge proven dependency masks without erasing unknown footprints.

    ``None`` means that the enclosing operation must supply a conservative
    footprint, while an empty set means that the estimate has proven no
    caller-visible blocking wires. A structurally zero estimate is therefore
    the neutral element even when its default mask is ``None``.

    Args:
        left (ResourceEstimate): Left composition operand.
        right (ResourceEstimate): Right composition operand.

    Returns:
        frozenset[WireKey] | None: Union of known masks, or ``None`` when a
            nonzero operand still has an unknown footprint.
    """

    def normalized(estimate: ResourceEstimate) -> frozenset[WireKey] | None:
        """Treat a zero-depth unknown mask as the empty dependency set.

        Args:
            estimate (ResourceEstimate): Estimate whose mask is normalized.

        Returns:
            frozenset[WireKey] | None: Explicit dependency mask or ``None``.
        """
        if estimate._dependency_keys is not None:
            return estimate._dependency_keys
        if not _estimate_has_nonzero_depth(estimate):
            return frozenset()
        return None

    left_keys = normalized(left)
    right_keys = normalized(right)
    if left_keys is None or right_keys is None:
        return None
    return frozenset((*left_keys, *right_keys))


class _LocalBlock:
    """Provide a minimal operation container for nested resource scopes.

    Args:
        operations (list[Operation]): Operations visible in the nested scope.
    """

    __slots__ = ("operations",)

    def __init__(self, operations: list[Operation]) -> None:
        """Initialize a local operation container.

        Args:
            operations (list[Operation]): Operations visible in the nested
                scope.
        """
        self.operations = operations


def _specialize_dependency_expression(
    expression: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None,
    used_names: set[str] | None = None,
) -> ResourceExpr:
    """Apply supplied scalar inputs only for physical dependency resolution.

    Resource formulas remain symbolic elsewhere. Dependency scheduling may
    nevertheless use a supplied array/control index to distinguish physical
    wires without expanding a problem-sized loop or circuit.

    Args:
        expression (ResourceExpr): Resolved symbolic index or view expression.
        scalar_values (Mapping[str, sp.Expr] | None): Supplied numeric values
            keyed by qkernel argument name.
        used_names (set[str] | None): Optional set updated with names that
            participated in dependency resolution.

    Returns:
        ResourceExpr: Expression specialized by matching supplied values.
    """
    if not scalar_values or not expression.free_symbols:
        return expression
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for symbol in expression.free_symbols:
        if isinstance(symbol, sp.Dummy):
            continue
        typed_symbol = cast(sp.Symbol, symbol)
        name = _symbol_display_name(typed_symbol)
        if name not in scalar_values:
            continue
        substitutions[typed_symbol] = scalar_values[name]
        if used_names is not None:
            used_names.add(name)
    if not substitutions:
        return expression
    return cast(
        ResourceExpr,
        expression.subs(list(substitutions.items()), simultaneous=True).doit(),
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


def _with_body_boundary_depth_metadata(
    estimate: ResourceEstimate,
    body_estimate: ResourceEstimate,
    *,
    source: str,
    zero_controls: ResourceExpr | int = 0,
    scalar_broadcast: ResourceExpr | int = 1,
) -> ResourceEstimate:
    """Classify conservative completion at a callable boundary.

    Args:
        estimate (ResourceEstimate): Caller-scoped body estimate.
        body_estimate (ResourceEstimate): Original body-scoped estimate.
        source (str): Callable name for the modeling assumption.
        zero_controls (ResourceExpr | int): Open-control X brackets surrounding
            the body. Defaults to zero.
        scalar_broadcast (ResourceExpr | int): Number of actual target
            elements receiving one scalar body. Defaults to one.

    Returns:
        ResourceEstimate: Estimate with conservative boundary metadata.
    """
    keys = body_estimate._dependency_keys
    if keys is None or not _estimate_has_nonzero_depth(body_estimate):
        return estimate
    activity_condition = _estimate_depth_activity_condition(estimate)
    bracket_condition = _and_conditions(
        sp.Gt(_expr(zero_controls), _ZERO),
        activity_condition,
    )
    if keys and bracket_condition is not sp.false:
        assumption = ResourceAssumption(
            "open-control brackets may finish on a different layer than the "
            "controlled body targets",
            source=source,
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=bracket_condition,
        )
    broadcast_condition = _and_conditions(
        sp.Gt(_expr(scalar_broadcast), _ONE),
        activity_condition,
    )
    if broadcast_condition is not sp.false:
        assumption = ResourceAssumption(
            "scalar-to-vector broadcast uses aggregate call latency for each "
            "target element's completion",
            source=source,
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=broadcast_condition,
        )
    return estimate


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


def _operation_depth_is_dependency_schedulable(
    operation: Operation,
) -> bool:
    """Return whether wire dependencies fully describe an operation's depth.

    Every supported aggregate carries its conditional global-ordering need in
    ``ResourceEstimate._global_barrier_condition``. This classifier therefore
    handles only operation kinds whose structure cannot be represented by the
    ordinary quantum footprint at all; it must not inspect nested bodies and
    thereby erase their branch activation conditions.

    Args:
        operation (Operation): Operation to classify.
    Returns:
        bool: Whether dependency scheduling is exact for this operation.
    """
    if isinstance(
        operation,
        (
            GateOperation,
            QInitOperation,
            MeasureOperation,
            MeasureVectorOperation,
            MeasureQFixedOperation,
            ProjectOperation,
            ResetOperation,
            GlobalPhaseOperation,
        ),
    ):
        return True
    if isinstance(operation, InvokeOperation):
        # The selected body or opaque model is scheduled on the invocation's
        # actual operands. ``operation.effects`` is a conservative union over
        # every backend/strategy implementation and must not serialize an
        # unrelated unitary selection.
        return True
    if isinstance(
        operation,
        (ControlledUOperation, InverseBlockOperation, SelectOperation),
    ):
        return True
    if isinstance(operation, (IfOperation, ForOperation, ForItemsOperation)):
        return True
    if isinstance(operation, WhileOperation):
        return True
    if not isinstance(operation, HasNestedOps):
        # Every supported atomic operation either has an explicit quantum
        # footprint or zero duration. Unknown atomic operations are rejected
        # by the interpreter before scheduling reaches this classifier.
        return True
    return False


def _operation_has_uniform_intrinsic_completion(
    operation: Operation,
    estimate: ResourceEstimate,
    dependency_keys: frozenset[WireKey],
    *,
    surrounding_controls: ResourceExpr,
) -> bool:
    """Return whether one atomic operation finishes all operands uniformly.

    Aggregate operations such as Pauli evolution and opaque calls may contain
    different internal exit layers even when they do not expose a dependency
    summary. A lowered multi-layer gate may likewise finish its controls,
    targets, or hidden ancillas on different layers. Only operations whose
    one-step resource semantics proves one uniform layer are accepted here.

    Args:
        operation (Operation): Operation whose intrinsic completion contract
            should be classified.
        estimate (ResourceEstimate): Resource summary produced for the
            operation.
        dependency_keys (frozenset[WireKey]): Caller-visible wires touched by
            the operation.
        surrounding_controls (ResourceExpr): Number of coherent controls
            surrounding the operation.

    Returns:
        bool: Whether all operand completions equal every aggregate depth-field
            peak for this atomic operation.
    """
    if not isinstance(
        operation,
        (
            GateOperation,
            QInitOperation,
            MeasureOperation,
            MeasureVectorOperation,
            MeasureQFixedOperation,
            ProjectOperation,
            ResetOperation,
            GlobalPhaseOperation,
            ExpvalOp,
        ),
    ):
        return False
    if any(
        demand != _ZERO
        for demand in (
            estimate.width.allocated_qubits,
            estimate.width.clean_ancilla_qubits,
            estimate.width.dirty_ancilla_qubits,
        )
    ):
        return False
    if (
        surrounding_controls == _ZERO
        and len(dependency_keys) <= 1
        or surrounding_controls == _ONE
        and not dependency_keys
    ):
        return True
    return _expressions_proven_equal_without_simplify(estimate.depth.depth, _ONE)


def _expressions_proven_equal_without_simplify(
    left: ResourceExpr,
    right: ResourceExpr,
) -> bool:
    """Prove equality without sending the expression to general simplify.

    This helper is used only to prove completion uniformity. Declining an
    expensive proof makes the estimate conservative but cannot undercount its
    gate or depth resources.

    Args:
        left (ResourceExpr): First resource expression.
        right (ResourceExpr): Second resource expression.

    Returns:
        bool: Whether structural comparison or SymPy assumptions prove
            equality.
    """
    if left == right:
        return True
    difference = cast(ResourceExpr, left - right)
    if difference == _ZERO or difference.is_zero is True:
        return True
    return False


def _disjoint_concrete_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    body_dependency_keys: frozenset[WireKey] | None = None,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    loop_symbol: sp.Symbol,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> DepthResources | None:
    """Parallelize concrete loop iterations with disjoint quantum footprints.

    This optimization is deliberately bounded: it proves pairwise-disjoint
    physical owner/index keys only for a small concrete loop. Whole registers,
    unresolved views, shared clean ancillas, large loops, or any overlapping
    element conservatively retain sequential loop depth.

    Args:
        operation (ForOperation): Loop whose independent body was summarized.
        resolver (ExprResolver): Resolver for the enclosing scope.
        body_depth (DepthResources): One symbolic body-iteration depth.
        body_dependency_keys (frozenset[WireKey] | None): Evaluated body
            footprint, including nested operation summaries when available.
            Defaults to ``None``.
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        loop_symbol (sp.Symbol): Internal body loop-variable symbol.
        allocated_qubits (ResourceExpr): Body-local allocation demand reused
            between sequential iterations.
        clean_ancillas (ResourceExpr): Shared fallback ancilla demand.
        dirty_ancillas (ResourceExpr): Shared dirty-ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources | None: Parallel critical-path depth when disjointness
            is proven, otherwise ``None``.
    """
    iterations = _bounded_concrete_loop_values(
        start,
        stop,
        step,
        limit=_MAX_EXACT_LOOP_DISJOINTNESS_EXPANSION,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if iterations is None or any(
        demand != _ZERO for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
    ):
        return None

    seen: dict[str, _OwnerWireIndices] = {}
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    depth_expressions = {
        field: cast(sp.Expr, getattr(body_depth, field)) for field in fields
    }
    varying_depths = {
        field: expression
        for field, expression in depth_expressions.items()
        if loop_symbol in expression.free_symbols
    }
    if len(iterations) > 0:
        for field, expression in depth_expressions.items():
            if field in varying_depths:
                continue
            invariant_depth = _safe_simplify(cast(sp.Expr, expression.doit()))
            peaks[field] = sp.Max(_ZERO, invariant_depth)
    for loop_value in iterations:
        value = sp.Integer(loop_value)
        if operation.loop_var_value is None:
            return None
        child = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context={operation.loop_var_value.uuid: value},
            extra_loop_vars={operation.loop_var: value},
        )
        footprint: set[WireKey] = set()
        if body_dependency_keys is not None:
            for key in body_dependency_keys:
                footprint.update(
                    _specialize_loop_dependency_key(
                        key,
                        loop_symbol,
                        value,
                    )
                )
        else:
            for body_operation in operation.operations:
                reads, writes = _quantum_wire_keys(
                    body_operation,
                    child,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
                footprint |= reads | writes
        if not footprint and any(
            getattr(body_depth, field) != _ZERO for field in fields
        ):
            return None
        if not _record_disjoint_wire_footprint(seen, footprint):
            return None
        for field, expression in varying_depths.items():
            iteration_depth = _safe_simplify(
                cast(sp.Expr, expression.subs(loop_symbol, value).doit())
            )
            peaks[field] = sp.Max(peaks[field], iteration_depth)
    return DepthResources(**peaks)


def _bounded_concrete_loop_values(
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    *,
    limit: int = _MAX_EXACT_LOOP_WIRE_EXPANSION,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> range | None:
    """Return a safely bounded concrete Python range for wire analysis.

    Args:
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        limit (int): Maximum iteration count to enumerate. Defaults to the
            compact dependency-metadata expansion budget.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        range | None: Concrete range within the exact-expansion budget, or
            ``None`` when a bound is unresolved, malformed, or too large.
    """
    bounds = tuple(
        _specialize_dependency_expression(
            bound,
            scalar_values,
            used_names,
        )
        for bound in (start, stop, step)
    )
    if not all(value.is_number and _is_concrete_integer(value) for value in bounds):
        return None
    concrete_start, concrete_stop, concrete_step = (int(value) for value in bounds)
    if concrete_step == 0:
        return None
    iterations = range(concrete_start, concrete_stop, concrete_step)
    if len(iterations[: limit + 1]) > limit:
        return None
    return iterations


def _concrete_loop_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    loop_symbol: sp.Symbol,
    *,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> dict[WireKey, ResourceExpr] | None:
    """Project exact local wire completion through a small concrete loop.

    A disjoint loop runs every iteration from the same logical start layer.
    Its caller-visible completion therefore comes from the corresponding
    iteration-local completion, not from the sequential sum used before the
    disjoint-depth proof.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): One-iteration
            body completion map.
        loop_symbol (sp.Symbol): Internal loop-variable symbol.
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        dict[WireKey, ResourceExpr] | None: Exact projected completion map, or
            ``None`` when exact bounded projection is unavailable.
    """
    if completion is None:
        return None
    iterations = _bounded_concrete_loop_values(
        start,
        stop,
        step,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if iterations is None:
        return None
    projected: dict[WireKey, ResourceExpr] = {}
    for loop_value in iterations:
        value = sp.Integer(loop_value)
        for key, depth in completion.items():
            local_depth = _safe_simplify(
                cast(
                    ResourceExpr,
                    depth.subs(
                        loop_symbol,
                        value,
                        simultaneous=True,
                    ).doit(),
                )
            )
            for specialized_key in _specialize_loop_dependency_key(
                key,
                loop_symbol,
                value,
            ):
                projected[specialized_key] = _resource_max(
                    projected.get(specialized_key, _ZERO),
                    local_depth,
                )
    return projected


def _uniform_parallel_loop_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    *,
    body_depth: ResourceExpr,
    projected_keys: frozenset[WireKey] | None,
    parallel_depth: ResourceExpr,
    loop_symbol: sp.Symbol,
) -> dict[WireKey, ResourceExpr] | None:
    """Recover compact completion for an invariant uniform loop body.

    This path avoids enumerating a large or symbolic disjoint loop. It is
    exact only when every body-visible wire finishes at the body's aggregate
    depth and that depth does not vary with the loop induction value.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): One-iteration
            body completion map.
        body_depth (ResourceExpr): One-iteration aggregate depth.
        projected_keys (frozenset[WireKey] | None): Caller-visible loop
            footprint after range projection.
        parallel_depth (ResourceExpr): Aggregate depth after the disjoint-loop
            proof.
        loop_symbol (sp.Symbol): Internal loop-variable symbol.

    Returns:
        dict[WireKey, ResourceExpr] | None: Compact exact completion map, or
            ``None`` when uniformity cannot be proven.
    """
    if completion is None or projected_keys is None:
        return None
    if loop_symbol in body_depth.free_symbols:
        return None
    if any(
        not _expressions_proven_equal_without_simplify(depth, body_depth)
        for depth in completion.values()
    ):
        return None
    return {key: parallel_depth for key in projected_keys}


def _specialize_loop_dependency_key(
    key: WireKey,
    loop_symbol: sp.Symbol,
    loop_value: sp.Integer,
) -> set[WireKey]:
    """Specialize and, when finite, expand one loop-body dependency key.

    Args:
        key (WireKey): Body-scoped dependency address.
        loop_symbol (sp.Symbol): Loop-local symbol being bound.
        loop_value (sp.Integer): Concrete value for one loop iteration.

    Returns:
        set[WireKey]: Specialized scalar or symbolic-range addresses.
    """
    owner, index = key
    if isinstance(index, sp.Expr):
        return {
            (
                owner,
                _normalize_wire_index(
                    cast(
                        ResourceExpr,
                        index.subs(
                            loop_symbol,
                            loop_value,
                            simultaneous=True,
                        ),
                    )
                ),
            )
        }
    if not isinstance(index, _WireRangeIndex):
        return {key}
    specialized = index.mapped(
        lambda expression: expression.subs(
            loop_symbol,
            loop_value,
            simultaneous=True,
        )
    )
    iterations = specialized.iterations
    if (
        not iterations.is_number
        or not _is_concrete_integer(iterations)
        or not 0 <= iterations <= _MAX_EXACT_LOOP_WIRE_EXPANSION
    ):
        return {(owner, specialized)}
    return {
        (
            owner,
            _normalize_wire_index(
                cast(
                    ResourceExpr,
                    specialized.index_at_offset.subs(
                        _WIRE_RANGE_OFFSET,
                        offset,
                        simultaneous=True,
                    ),
                )
            ),
        )
        for offset in range(int(iterations))
    }


def _record_disjoint_wire_footprint(
    seen: dict[str, _OwnerWireIndices],
    footprint: set[WireKey],
) -> bool:
    """Record one footprint only when it is disjoint from all prior wires.

    An owner-wide ``None`` index aliases every scalar index for the same
    allocation owner. Scalar indices are retained in per-owner sets and
    compared with the same symbolic relation used by the dependency scheduler.

    Args:
        seen (dict[str, _OwnerWireIndices]): Previously recorded addresses
            indexed by allocation owner.
        footprint (set[WireKey]): Candidate physical wire keys.

    Returns:
        bool: Whether the candidate was disjoint and has been recorded.
    """
    for owner, index in footprint:
        owner_indices = seen.get(owner)
        if owner_indices is None:
            continue
        if any(
            _wire_index_relation(index, previous) is not _WireRelation.DISJOINT
            for previous in owner_indices.candidates(index)
        ):
            return False
    for owner, index in footprint:
        seen.setdefault(owner, _OwnerWireIndices()).add(index)
    return True


def _aggregate_completion_overlap_condition(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    wire_footprints: Sequence[_WireFootprint | None],
    *,
    activity_conditions: Sequence[Boolean] | None = None,
) -> Boolean:
    """Return when an aggregate completion can delay a later operation.

    A nonuniform operation is harmless when it is the last user of its wires:
    its aggregate critical-path depth is still exact. The resource estimate
    becomes conservative only when that scalar latency is reused as a
    caller-visible wire completion for a subsequent operation.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations
            paired with their resource summaries in program order.
        wire_footprints (Sequence[_WireFootprint | None]): Read/write
            footprints aligned with ``scheduled``.
        activity_conditions (Sequence[Boolean] | None): Optional precomputed
            depth-activity guards aligned with ``scheduled``. Defaults to
            computing them once for this call.

    Returns:
        Boolean: Guard under which a nonuniform aggregate may over-serialize
            a later wire dependency.

    Raises:
        AssertionError: If the operation, footprint, and activity sequences
            differ in length.
    """
    if activity_conditions is None:
        activity_conditions = _scheduled_depth_activity_conditions(scheduled)
    if not (len(scheduled) == len(wire_footprints) == len(activity_conditions)):
        raise AssertionError(
            "Scheduled operations, wire footprints, and activity conditions "
            "must have equal lengths."
        )
    uncertain_indices: dict[str, _OwnerWireIndices] = {}
    uncertain_activity: dict[WireKey, Boolean] = {}
    overlap_conditions: set[Boolean] = set()
    for (_operation, estimate), footprint, active in zip(
        scheduled,
        wire_footprints,
        activity_conditions,
        strict=True,
    ):
        if not _estimate_has_nonzero_depth(estimate):
            continue
        if footprint is None:
            raise AssertionError(
                "A nonzero-depth scheduled operation requires a wire footprint."
            )
        reads, writes = map(set, footprint)
        for owner, index in reads:
            owner_indices = uncertain_indices.get(owner)
            if owner_indices is None:
                continue
            for candidate in owner_indices.candidates(index):
                if _wire_index_relation(index, candidate) is _WireRelation.DISJOINT:
                    continue
                condition = _and_conditions(
                    uncertain_activity[(owner, candidate)],
                    active,
                )
                if condition is not sp.false:
                    overlap_conditions.add(condition)
        if estimate._dependency_completion_uniform is True:
            continue
        for key in reads | writes:
            owner, index = key
            uncertain_indices.setdefault(owner, _OwnerWireIndices()).add(index)
            previous = uncertain_activity.get(key, sp.false)
            uncertain_activity[key] = cast(Boolean, sp.Or(previous, active))
    return cast(
        Boolean,
        sp.Or(*overlap_conditions) if overlap_conditions else sp.false,
    )


def _loop_body_has_symbolic_quantum_index(
    operation: ForOperation,
    resolver: ExprResolver,
    loop_symbol: sp.Symbol,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> bool:
    """Return whether a loop body addresses an array through its loop value.

    Args:
        operation (ForOperation): Loop whose body values are inspected.
        resolver (ExprResolver): Loop-body resolver.
        loop_symbol (sp.Symbol): Symbol representing the current iteration.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        bool: Whether any quantum element index depends on ``loop_symbol``.
    """
    return any(
        loop_symbol in index.free_symbols
        for body_operation in operation.operations
        for value in (*body_operation.all_input_values(), *body_operation.results)
        if isinstance(value, Value)
        and value.type.is_quantum()
        and value.parent_array is not None
        and (
            index := _quantum_element_index_expression(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
        )
        is not None
    )


def _dependency_keys_depend_on_symbol(
    keys: frozenset[WireKey] | None,
    symbol: sp.Symbol,
) -> bool:
    """Return whether an evaluated footprint still depends on a local symbol.

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
                index.index_at_offset.free_symbols | index.iterations.free_symbols
            ):
                return True
        elif isinstance(index, sp.Expr) and symbol in index.free_symbols:
            return True
    return False


def _symbolic_disjoint_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    loop_symbol: sp.Symbol,
    iterations: ResourceExpr,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> DepthResources | None:
    """Prove parallel loop depth for one injective affine element footprint.

    For each root allocation, every depth-carrying body access must use the
    same affine index ``a * i + b`` with a provably nonzero ``a``. Distinct
    Python-range iterations then touch distinct physical slots, so iteration
    gate layers can run in parallel while additive gate counts still scale by
    the trip count.

    Args:
        operation (ForOperation): Loop whose footprint should be proven.
        resolver (ExprResolver): Resolver scoped to the loop body.
        body_depth (DepthResources): One iteration's dependency depth.
        loop_symbol (sp.Symbol): Symbol representing the loop value.
        iterations (ResourceExpr): Python-range iteration count.
        allocated_qubits (ResourceExpr): Body-local allocation demand reused
            between sequential iterations.
        clean_ancillas (ResourceExpr): Shared decomposition ancilla demand.
        dirty_ancillas (ResourceExpr): Shared dirty-ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources | None: One-iteration depth guarded by a nonempty range,
            or ``None`` when injectivity cannot be proven.
    """
    if any(
        demand != _ZERO for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
    ) or any(
        loop_symbol in cast(ResourceExpr, getattr(body_depth, field.name)).free_symbols
        for field in dataclasses.fields(DepthResources)
    ):
        return None
    ignored_operations = (
        QInitOperation,
        GlobalPhaseOperation,
        StoreArrayElementOperation,
        ReturnQuantumArrayElementOperation,
    )
    indices_by_owner: dict[str, list[ResourceExpr]] = {}
    pending_operations = list(reversed(operation.operations))
    while pending_operations:
        body_operation = pending_operations.pop()
        if isinstance(body_operation, ignored_operations):
            continue
        if isinstance(body_operation, HasNestedOps):
            if not isinstance(body_operation, IfOperation):
                return None
            for region in reversed(body_operation.nested_regions()):
                pending_operations.extend(reversed(region.operations))
            # The branch's own merge values only alias the physical values
            # already inspected in its regions. Including those aggregate
            # ArrayValues would turn an otherwise injective ``register[i]``
            # body into an owner-wide footprint and defeat the proof.
            continue
        quantum_values = [
            value
            for value in (
                *body_operation.all_input_values(),
                *body_operation.results,
            )
            if isinstance(value, Value) and value.type.is_quantum()
        ]
        for value in quantum_values:
            if isinstance(value, ArrayValue) or value.parent_array is None:
                return None
            index = _quantum_element_index_expression(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            if index is None:
                return None
            indices_by_owner.setdefault(
                _quantum_allocation_owner(value),
                [],
            ).append(index)
    if not indices_by_owner and any(
        getattr(body_depth, field.name) != _ZERO
        for field in dataclasses.fields(DepthResources)
    ):
        return None
    for indices in indices_by_owner.values():
        representative = _normalize_wire_index(indices[0])
        if any(
            _wire_index_relation(
                _normalize_wire_index(index),
                representative,
            )
            is not _WireRelation.DEFINITE_OVERLAP
            for index in indices
        ):
            return None
        if not isinstance(representative, sp.Expr):
            return None
        slope = _safe_simplify(cast(ResourceExpr, sp.diff(representative, loop_symbol)))
        if loop_symbol in slope.free_symbols or slope.is_zero is not False:
            return None
    return _conditional_depth(
        body_depth,
        DepthResources.zero(),
        sp.Gt(iterations, _ZERO),
    )


def _conditional_completion(
    active: ResourceExpr,
    inactive: ResourceExpr,
    condition: Boolean,
) -> ResourceExpr:
    """Select a completion depth without expanding a Boolean to ``ITE``.

    Args:
        active (ResourceExpr): Completion when the operation executes.
        inactive (ResourceExpr): Previous completion when it does not.
        condition (Boolean): Symbolic operation-activity predicate.

    Returns:
        ResourceExpr: Binder-safe conditional completion expression.
    """
    return cast(
        ResourceExpr,
        inactive + _ConditionIndicator(condition) * (active - inactive),
    )


def _completion_after_conditional_duration(
    finish: ResourceExpr,
    start: ResourceExpr,
    inactive: ResourceExpr,
    active_when: Boolean,
) -> ResourceExpr:
    """Keep one scheduled completion compact across an inactive duration.

    When ``start`` already equals the inactive completion, ``finish`` differs
    only by the operation duration. That duration is zero whenever its
    activity guard is false, so wrapping the same condition in an additional
    :class:`_ConditionIndicator` is redundant.

    Args:
        finish (ResourceExpr): Completion after adding the operation duration.
        start (ResourceExpr): Dependency start selected for the operation.
        inactive (ResourceExpr): Completion retained when the duration is zero.
        active_when (Boolean): Guard under which the duration may be nonzero.

    Returns:
        ResourceExpr: Exact completion with no redundant activity indicator
            when structural equality proves it unnecessary.
    """
    if active_when is sp.true or _expressions_proven_equal_without_simplify(
        start,
        inactive,
    ):
        return finish
    return _conditional_completion(finish, inactive, active_when)


def _dependency_depth(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    wire_footprints: Sequence[_WireFootprint | None],
    *,
    activity_conditions: Sequence[Boolean] | None = None,
    read_conditions: Sequence[Mapping[WireKey, Boolean]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[DepthResources, dict[WireKey, ResourceExpr], Boolean, bool]:
    """Schedule operation summaries by wire dependencies and hybrid barriers.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations in
            program order paired with their internally computed summaries.
        wire_footprints (Sequence[_WireFootprint | None]): Precomputed read and
            write keys aligned with ``scheduled``. Zero-depth operations use
            ``None``.
        activity_conditions (Sequence[Boolean] | None): Optional precomputed
            depth-activity guards aligned with ``scheduled``. Defaults to
            computing them once for this call.
        read_conditions (Sequence[Mapping[WireKey, Boolean]] | None): Optional
            per-operation guards for individual dependency reads. Missing keys
            are unconditional. Defaults to unconditional reads.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values used only to prove completion uniformity. Defaults to
            ``None``.
        used_names (set[str] | None): Optional set updated with supplied names
            used by the uniformity proof. Defaults to ``None``.

    Returns:
        tuple[DepthResources, dict[WireKey, ResourceExpr], Boolean, bool]:
            Critical-path depths, caller-visible completion time of each
            touched wire for the total-depth field, and the condition under
            which a possible symbolic alias affected scheduling, followed by
            whether every touched wire is proven to complete at every
            aggregate depth-field peak.

    Raises:
        AssertionError: If a footprint is missing, or the operation,
            footprint, and activity sequences are not aligned.
    """
    if activity_conditions is None:
        activity_conditions = _scheduled_depth_activity_conditions(scheduled)
    if read_conditions is None:
        read_conditions = tuple({} for _ in scheduled)
    if not (
        len(scheduled)
        == len(wire_footprints)
        == len(activity_conditions)
        == len(read_conditions)
    ):
        raise AssertionError(
            "Scheduled operations, wire footprints, activity conditions, and "
            "read conditions must have equal lengths."
        )
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    availability: dict[
        str,
        dict[str, dict[WireIndex, ResourceExpr]],
    ] = {field: {} for field in fields}
    indices_by_owner: dict[str, _OwnerWireIndices] = {}
    barrier_availability: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    possible_alias_conditions: set[Boolean] = set()
    completion_is_uniform = True
    for (operation, estimate), footprint, operation_active, operation_reads in zip(
        scheduled,
        wire_footprints,
        activity_conditions,
        read_conditions,
        strict=True,
    ):
        structurally_schedulable = _operation_depth_is_dependency_schedulable(operation)
        barrier_condition = (
            estimate._global_barrier_condition if structurally_schedulable else sp.true
        )
        footprint_reads = set(footprint[0]) if footprint is not None else set()
        footprint_writes = set(footprint[1]) if footprint is not None else set()
        if (
            not _estimate_has_nonzero_depth(estimate)
            and barrier_condition is sp.false
            and not footprint_reads
            and not footprint_writes
        ):
            continue
        if footprint is None and _estimate_has_nonzero_depth(estimate):
            raise AssertionError(
                "A nonzero-depth scheduled operation requires a wire footprint."
            )
        reads = footprint_reads
        writes = footprint_writes
        if estimate.width.clean_ancilla_qubits != _ZERO:
            # Width reports one reusable clean-ancilla pool (a maximum across
            # sequential operations). Treat that pool as a shared dependency
            # wire so depth never assumes two independent decompositions use
            # the same ancillas simultaneously.
            shared_pool = ("$resource_clean_ancilla_pool", None)
            reads.add(shared_pool)
            writes.add(shared_pool)
        if estimate.width.dirty_ancilla_qubits != _ZERO:
            shared_dirty_pool = ("$resource_dirty_ancilla_pool", None)
            reads.add(shared_dirty_pool)
            writes.add(shared_dirty_pool)
        if (
            _estimate_has_nonzero_depth(estimate)
            and estimate.width.allocated_qubits != _ZERO
        ):
            anonymous_workspace = (
                f"$resource_anonymous_workspace:{id(operation)}",
                None,
            )
            reads.add(anonymous_workspace)
            writes.add(anonymous_workspace)
        # Quantum resources are occupied for the whole operation even when an
        # operation consumes them without returning a quantum result (for
        # example measurement).  Classical SSA tokens are immutable and may
        # fan out, so reading one must not delay another independent consumer.
        occupied = {
            key for key in reads | writes if not _is_classical_dependency_key(key)
        }
        occupied.update(key for key in writes if _is_classical_dependency_key(key))
        if (
            estimate._dependency_keys is not None
            and estimate._dependency_completion_uniform is not True
        ):
            completion_is_uniform = False
        scheduling_active = _boolean_condition(
            sp.Or(operation_active, barrier_condition)
        )
        if scheduling_active is sp.false:
            continue
        for field in fields:
            duration = cast(ResourceExpr, getattr(estimate.depth, field))
            owner_depths = availability[field]
            definite_dependencies: list[ResourceExpr] = []
            possible_dependencies: list[ResourceExpr] = []
            for owner, index in reads:
                read_condition = operation_reads.get((owner, index), sp.true)
                if read_condition is sp.false:
                    continue
                owner_indices = indices_by_owner.get(owner)
                if owner_indices is None:
                    continue
                wire_depths = owner_depths.get(owner, {})
                for candidate_index in owner_indices.candidates(index):
                    depth = wire_depths.get(candidate_index)
                    if depth is None:
                        continue
                    if read_condition is not sp.true:
                        depth = _conditional_completion(
                            depth,
                            _ZERO,
                            read_condition,
                        )
                    relation = _wire_index_relation(index, candidate_index)
                    if relation is _WireRelation.DISJOINT:
                        continue
                    if relation is _WireRelation.DEFINITE_OVERLAP:
                        definite_dependencies.append(depth)
                    else:
                        possible_dependencies.append(depth)
            if structurally_schedulable:
                baseline_start = _resource_max_many(
                    [*definite_dependencies, barrier_availability[field]]
                )
                dependency_start = _resource_max_many(
                    [baseline_start, *possible_dependencies]
                )
                barrier_start = _resource_max(
                    peaks[field],
                    barrier_availability[field],
                )
                start = _conditional_completion(
                    barrier_start,
                    dependency_start,
                    barrier_condition,
                )
                for dependency_depth in possible_dependencies:
                    condition = _and_conditions(
                        _and_conditions(
                            operation_active,
                            cast(Boolean, sp.Not(barrier_condition)),
                        ),
                        _and_conditions(
                            _resource_activity_condition(dependency_depth),
                            sp.Gt(
                                dependency_depth,
                                baseline_start,
                            ),
                        ),
                    )
                    if condition is not sp.false:
                        possible_alias_conditions.add(condition)
            else:
                start = _resource_max(
                    peaks[field],
                    barrier_availability[field],
                )
            finish = start + duration
            peaks[field] = _resource_max(peaks[field], finish)
            if barrier_condition is not sp.false:
                previous_barrier = barrier_availability[field]
                barrier_availability[field] = _completion_after_conditional_duration(
                    finish,
                    start,
                    previous_barrier,
                    barrier_condition,
                )
            for owner, index in occupied:
                wire_depth = owner_depths.setdefault(owner, {})
                previous = wire_depth.get(index, _ZERO)
                wire_depth[index] = _completion_after_conditional_duration(
                    finish,
                    start,
                    previous,
                    scheduling_active,
                )
        for owner, index in occupied:
            indices_by_owner.setdefault(owner, _OwnerWireIndices()).add(index)
    completion_keys = {
        (owner, index)
        for owner, owner_depths in availability["depth"].items()
        for index in owner_depths
    }
    if completion_is_uniform:
        for field in fields:
            peak = _specialize_dependency_expression(
                peaks[field],
                scalar_values,
                used_names,
            )
            if any(
                not _expressions_proven_equal_without_simplify(
                    _specialize_dependency_expression(
                        cast(
                            ResourceExpr,
                            availability[field].get(owner, {}).get(index, _ZERO),
                        ),
                        scalar_values,
                        used_names,
                    ),
                    peak,
                )
                for owner, index in completion_keys
            ):
                completion_is_uniform = False
                break
    return (
        DepthResources(**peaks),
        {
            (owner, index): depth
            for owner, owner_depths in availability["depth"].items()
            for index, depth in owner_depths.items()
        },
        cast(
            Boolean,
            sp.Or(*possible_alias_conditions)
            if possible_alias_conditions
            else sp.false,
        ),
        completion_is_uniform,
    )


def _block_input_allocations(
    block: Block,
    resolver: ExprResolver,
) -> dict[str, ResourceExpr]:
    """Return caller-owned quantum widths keyed by logical wire identity.

    Traced qkernel blocks contain declaration-like ``QInitOperation`` nodes
    for their formal quantum inputs. Seeding those logical IDs lets liveness
    distinguish the declarations from true body allocations.

    Args:
        block (Block): Block whose formal quantum inputs should be seeded.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        dict[str, ResourceExpr]: Formal quantum widths by logical ID.
    """
    return {
        value.logical_id: _qubit_value_size(value, resolver)
        for value in block.input_values
        if isinstance(value, Value) and value.type.is_quantum()
    }


def _root_callable_resource_attrs(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
) -> Mapping[str, Any]:
    """Return resource metadata preserved on a root qkernel-like object.

    A plain :class:`Block` intentionally has no callable identity or attrs.
    Serialized qkernels and stdlib descriptors preserve their definition attrs
    on ``_callable_attrs_override``, which lets root estimation enforce the
    same contract as nested invocation operations without importing frontend
    helpers into the estimator core.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Estimator
            input before it is coerced to IR.

    Returns:
        Mapping[str, Any]: Preserved callable attrs, or an empty mapping.
    """
    attrs = getattr(kernel, "_callable_attrs_override", None)
    return attrs if isinstance(attrs, Mapping) else {}


def _root_callable_shape_inputs(
    attrs: Mapping[str, Any],
    block_or_ops: Block | Sequence[Operation],
    explicit_inputs: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, int]:
    """Infer one-dimensional root quantum-port widths from exact metadata.

    A callable resource contract describes total scalar widths. That value
    uniquely determines the shape of a one-dimensional quantum vector, but it
    cannot safely choose dimensions for a higher-rank array. Explicit port or
    dimension inputs always take precedence so a conflicting value reaches
    the normal contract validator and produces a useful error.

    Args:
        attrs (Mapping[str, Any]): Root callable definition attributes.
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.
        explicit_inputs (Mapping[str, Any]): User-supplied specialization
            inputs before build/estimation partitioning.
        source (str): Callable name used in malformed-contract diagnostics.

    Returns:
        dict[str, int]: Inferred one-dimensional quantum-port widths keyed by
            their public argument names.

    Raises:
        ValueError: If present resource metadata is malformed.
    """
    if not isinstance(block_or_ops, Block):
        return {}
    quantum_ports = [
        (name, value)
        for name, value in zip(
            block_or_ops.label_args,
            block_or_ops.input_values,
        )
        if isinstance(value, Value) and value.type.is_quantum()
    ]
    shape_aliases = input_shape_dimension_aliases(block_or_ops)
    inferred: dict[str, int] = {}
    for entry in quantum_operand_widths(attrs, source=source):
        if entry.index >= len(quantum_ports):
            # The constraint builder reports the complete call-shape
            # diagnostic before inference runs.
            continue
        name, value = quantum_ports[entry.index]
        if not isinstance(value, ArrayValue) or len(value.shape) != 1:
            continue
        if value.shape[0].is_constant():
            continue
        dimension_name = shape_aliases.get(value.shape[0].uuid)
        if name in explicit_inputs or (
            dimension_name is not None and dimension_name in explicit_inputs
        ):
            continue
        inferred[name] = entry.width
    return inferred


def _captured_quantum_allocations(
    operations: Sequence[Operation],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Return outer quantum allocations captured by a nested operation list.

    Branch liveness must start with captured wires live so measuring or
    replacing them can release capacity before a branch-local allocation. A
    value is captured when it is read by the nested list but is not produced by
    any operation in that same list. Array elements and slice views carry fresh
    SSA UUIDs, so a value whose root owner comes from a body-local QInit is also
    excluded explicitly.

    Args:
        operations (Sequence[Operation]): Nested operations to inspect.
        resolver (ExprResolver): Resolver for symbolic array dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            enclosing QInit UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Captured root allocation widths by owner.
    """
    produced = {
        result.uuid
        for operation in operations
        for result in operation.results
        if isinstance(result, Value)
    }
    local_allocation_owners_by_uuid = {
        result.uuid: _quantum_allocation_owner(result)
        for operation in operations
        if isinstance(operation, QInitOperation)
        for result in operation.results
        if isinstance(result, Value) and result.type.is_quantum()
    }
    local_allocation_owners = frozenset(local_allocation_owners_by_uuid.values())
    resolved_allocation_owners: Mapping[str, str] = ChainMap(
        local_allocation_owners_by_uuid,
        cast(MutableMapping[str, str], allocation_owners_by_uuid or {}),
    )
    captured: dict[str, ResourceExpr] = {}
    for operation in operations:
        for value in operation.all_input_values():
            if (
                not isinstance(value, Value)
                or not value.type.is_quantum()
                or value.uuid in produced
            ):
                continue
            runtime_sizes = _runtime_carrier_owner_sizes(
                value,
                resolved_allocation_owners,
            )
            if runtime_sizes is not None:
                for owner, size in runtime_sizes.items():
                    if owner in local_allocation_owners:
                        continue
                    captured[owner] = captured.get(owner, _ZERO) + size
                continue
            owner = _quantum_allocation_owner(value)
            if owner in local_allocation_owners:
                continue
            captured[owner] = _quantum_owner_capacity(value, resolver)
    return captured


def _branch_owner_sizes(
    true_sizes: Mapping[str, ResourceExpr],
    false_sizes: Mapping[str, ResourceExpr],
    *,
    condition: sp.Basic,
    runtime_condition: bool,
) -> dict[str, ResourceExpr]:
    """Combine captured owner widths across two conditional branches.

    Args:
        true_sizes (Mapping[str, ResourceExpr]): True-branch captured widths.
        false_sizes (Mapping[str, ResourceExpr]): False-branch captured widths.
        condition (sp.Basic): Compile-time branch predicate.
        runtime_condition (bool): Whether the branch is selected at runtime.

    Returns:
        dict[str, ResourceExpr]: Captured width for each possible owner.
    """
    return {
        owner: (
            sp.Max(true_sizes.get(owner, _ZERO), false_sizes.get(owner, _ZERO))
            if runtime_condition
            else _piecewise(
                true_sizes.get(owner, _ZERO),
                false_sizes.get(owner, _ZERO),
                condition,
            )
        )
        for owner in true_sizes.keys() | false_sizes.keys()
    }


def _with_operation_output_summary(
    estimate: ResourceEstimate,
    operation: ForOperation | WhileOperation | ForItemsOperation,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic,
    body_output_sizes: Mapping[str, ResourceExpr] | None = None,
    additional_consumed_allocations: Mapping[str, ResourceExpr] | None = None,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> ResourceEstimate:
    """Attach authoritative live quantum results to a nested estimate.

    Args:
        estimate (ResourceEstimate): Nested operation estimate.
        operation (ForOperation | WhileOperation | ForItemsOperation):
            Enclosing control-flow operation.
        resolver (ExprResolver): Resolver after publishing carried results.
        active_when (sp.Basic): Condition under which the body executes at
            least once.
        body_output_sizes (Mapping[str, ResourceExpr] | None): Authoritative
            retained live-owner summary across modeled body executions.
            Loop evaluators may provide owner-wise maxima when a final-state
            release cannot be proven. Defaults to ``None``.
        additional_consumed_allocations (Mapping[str, ResourceExpr] | None):
            Captured owners proven consumed by the union of concrete loop
            iterations, in addition to whole-owner body operations. Defaults
            to ``None``.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            enclosing QInit UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        ResourceEstimate: Estimate whose output summary may be empty when every
        captured quantum input was consumed.
    """
    captured = _captured_quantum_allocations(
        operation.operations,
        resolver,
        allocation_owners_by_uuid,
    )
    body_consumed = _definitely_consumed_captured_allocations(
        operation.operations,
        captured,
        resolver,
        allocation_owners_by_uuid,
    )
    body_consumed.update(additional_consumed_allocations or {})
    condition = _boolean_condition(active_when)
    consumed = {
        owner: _piecewise(size, _ZERO, condition)
        for owner, size in body_consumed.items()
    }
    output_sizes = _quantum_result_owner_sizes(
        operation.results,
        resolver,
        allocation_owners_by_uuid,
    )
    residual = sum(
        (
            size
            for owner, size in (body_output_sizes or {}).items()
            if owner not in captured
        ),
        _ZERO,
    )
    guarded_residual = _piecewise(residual, _ZERO, condition)
    if guarded_residual != _ZERO:
        output_sizes[f"{type(operation).__name__}:{id(operation)}/live"] = (
            guarded_residual
        )
    return dataclasses.replace(
        estimate,
        _output_sizes=output_sizes,
        _input_sizes=consumed,
        _has_output_summary=True,
    )


def _definitely_consumed_captured_allocations(
    operations: Sequence[Operation],
    captured: Mapping[str, ResourceExpr],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Find captured owners whose final full-width action consumes them.

    Partial element operations are deliberately ignored: without tracking the
    exact final element set, retaining the owner is safer than releasing live
    siblings. Full-width gates keep an owner live, while a later full-width
    measurement or replacement consumes it.

    Args:
        operations (Sequence[Operation]): Nested body operations in order.
        captured (Mapping[str, ResourceExpr]): Captured owner capacities.
        resolver (ExprResolver): Resolver for symbolic operand widths.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            enclosing QInit UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Owners proven fully consumed by the body.
    """
    consumed = {owner: False for owner in captured}
    for operation in operations:
        if isinstance(operation, HasNestedOps):
            continue
        input_sizes = _quantum_result_owner_sizes(
            [
                value
                for value in operation.all_input_values()
                if isinstance(value, Value) and value.type.is_quantum()
            ],
            resolver,
            allocation_owners_by_uuid,
        )
        result_sizes = _quantum_result_owner_sizes(
            [result for result in operation.results if result.type.is_quantum()],
            resolver,
            allocation_owners_by_uuid,
        )
        for owner, capacity in captured.items():
            touched = input_sizes.get(owner, _ZERO)
            if _safe_simplify(touched - capacity) != _ZERO:
                continue
            returned = result_sizes.get(owner, _ZERO)
            consumed[owner] = _safe_simplify(returned - capacity) != _ZERO
    return {
        owner: captured[owner] for owner, is_consumed in consumed.items() if is_consumed
    }


def _loop_captured_observation_consumption(
    operations: Sequence[Operation],
    captured: Mapping[str, ResourceExpr],
    resolvers: Sequence[ExprResolver],
    *,
    definite_iterations: bool,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> tuple[dict[str, ResourceExpr], bool]:
    """Project destructive loop observations onto captured allocation slots.

    Every resolver represents one possible loop iteration. When the complete
    iteration set is concrete, unconditional observations are unioned by
    physical owner/index and an owner is released only if every scalar slot is
    covered. Observations nested under control flow are never treated as
    definite because they may not execute on every runtime path. For symbolic
    or truncated iteration sets, the helper reports uncertainty instead of
    claiming an exact final liveness state.

    Args:
        operations (Sequence[Operation]): Loop-body operations.
        captured (Mapping[str, ResourceExpr]): Live captured owner capacities.
        resolvers (Sequence[ExprResolver]): Per-iteration resolvers, or one
            symbolic probe resolver when the iteration set is unresolved.
        definite_iterations (bool): Whether ``resolvers`` enumerates every
            executed iteration exactly.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional QInit
            UUID to physical owner mapping. Defaults to ``None``.

    Returns:
        tuple[dict[str, ResourceExpr], bool]: Captured owners proven fully
            consumed, and whether any observed captured owner remains
            conservatively live.
    """
    if not resolvers or not captured:
        return {}, False
    owner_map = allocation_owners_by_uuid or {}
    fully_consumed: set[str] = set()
    covered_indices: dict[str, set[int]] = {}
    uncertain_owners: set[str] = set()
    destructive_types = (
        MeasureOperation,
        MeasureVectorOperation,
        MeasureQFixedOperation,
        ExpvalOp,
    )

    def record_observation(
        operation: Operation,
        resolver: ExprResolver,
        *,
        unconditional: bool,
    ) -> None:
        """Record one destructive operation for one iteration resolver.

        Args:
            operation (Operation): Destructive observation operation.
            resolver (ExprResolver): Resolver for one iteration.
            unconditional (bool): Whether the operation is outside nested
                control flow and therefore executes on every represented path.
        """
        quantum_inputs = [
            value
            for value in operation.all_input_values()
            if isinstance(value, Value) and value.type.is_quantum()
        ]
        for value in quantum_inputs:
            sizes = _quantum_result_owner_sizes([value], resolver, owner_map)
            keys = _quantum_value_wire_keys(
                value,
                resolver,
                allocation_owners_by_uuid=owner_map,
            )
            for owner, size in sizes.items():
                capacity = captured.get(owner)
                if capacity is None:
                    continue
                if not definite_iterations or not unconditional:
                    uncertain_owners.add(owner)
                    continue
                if _safe_simplify(size - capacity) == _ZERO:
                    fully_consumed.add(owner)
                    continue
                concrete_keys = 0
                for key_owner, index in keys:
                    if key_owner != owner:
                        continue
                    if not isinstance(index, (int, sp.Expr)):
                        uncertain_owners.add(owner)
                        continue
                    normalized = _normalize_wire_index(index)
                    if isinstance(normalized, int):
                        covered_indices.setdefault(owner, set()).add(normalized)
                        concrete_keys += 1
                    elif isinstance(normalized, sp.Integer):
                        covered_indices.setdefault(owner, set()).add(int(normalized))
                        concrete_keys += 1
                if (
                    not size.is_number
                    or not _is_concrete_integer(size)
                    or concrete_keys < int(size)
                ):
                    uncertain_owners.add(owner)

    def visit(
        body: Sequence[Operation],
        resolver: ExprResolver,
        *,
        unconditional: bool,
    ) -> None:
        """Visit observations while retaining control-path certainty.

        Args:
            body (Sequence[Operation]): Operations to inspect.
            resolver (ExprResolver): Resolver for one iteration.
            unconditional (bool): Whether every operation in ``body`` is on
                the unconditional loop path.
        """
        for operation in body:
            if isinstance(operation, destructive_types):
                record_observation(
                    operation,
                    resolver,
                    unconditional=unconditional,
                )
            if isinstance(operation, HasNestedOps):
                for nested in operation.nested_op_lists():
                    visit(nested, resolver, unconditional=False)

    for iteration_resolver in resolvers:
        visit(operations, iteration_resolver, unconditional=True)

    for owner, indices in covered_indices.items():
        capacity = captured[owner]
        if (
            capacity.is_number
            and _is_concrete_integer(capacity)
            and int(capacity) >= 0
            and indices.issuperset(range(int(capacity)))
        ):
            fully_consumed.add(owner)
    consumed = {
        owner: (
            captured[owner] if owner in fully_consumed else sp.Integer(len(indices))
        )
        for owner, indices in covered_indices.items()
        if owner in fully_consumed or indices
    }
    consumed.update(
        {owner: captured[owner] for owner in fully_consumed if owner not in consumed}
    )
    uncertain_owners.difference_update(fully_consumed)
    return consumed, bool(uncertain_owners)


def _without_input_allocation_sites(
    sites: Mapping[str, ResourceExpr],
    operations: Sequence[Operation],
    initial_allocations: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Remove formal-input QInit declarations from allocation-site metadata.

    Args:
        sites (Mapping[str, ResourceExpr]): QInit sites collected while
            evaluating the operations.
        operations (Sequence[Operation]): Operations in the evaluated scope.
        initial_allocations (Mapping[str, ResourceExpr]): Caller-owned formal
            quantum wires keyed by logical ID.

    Returns:
        dict[str, ResourceExpr]: Allocation sites containing only body-owned
        qubits.
    """
    formal_site_ids = {
        operation.results[0].uuid
        for operation in operations
        if isinstance(operation, QInitOperation)
        and operation.results
        and operation.results[0].logical_id in initial_allocations
    }
    return {site: size for site, size in sites.items() if site not in formal_site_ids}


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


def _runtime_carrier_owner_sizes(
    value: Value,
    allocation_owners_by_uuid: Mapping[str, str],
) -> dict[str, ResourceExpr] | None:
    """Resolve tuple-style synthetic carriers to physical allocation owners.

    Args:
        value (Value): Candidate synthetic quantum array carrier.
        allocation_owners_by_uuid (Mapping[str, str]): QInit-result UUIDs
            mapped to root logical allocation IDs.

    Returns:
        dict[str, ResourceExpr] | None: Per-owner carrier counts, or ``None``
            when ``value`` has no element-identity runtime metadata.
    """
    runtime = value.metadata.array_runtime
    if runtime is None or not runtime.element_uuids or not runtime.element_logical_ids:
        return None
    parent_addresses = value.get_element_parent_addresses()
    owners: dict[str, ResourceExpr] = {}
    for index, logical_id in enumerate(runtime.element_logical_ids):
        address = parent_addresses[index] if index < len(parent_addresses) else None
        parent_uuid = (
            runtime.element_parent_uuids[index]
            if index < len(runtime.element_parent_uuids)
            else ""
        )
        owner = allocation_owners_by_uuid.get(
            address[0] if address is not None else parent_uuid,
            logical_id,
        )
        owners[owner] = owners.get(owner, _ZERO) + _ONE
    return owners


def _quantum_result_owner_sizes(
    results: Sequence[Value],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Aggregate live result width by root allocation identity.

    A nested qkernel can return several scalar elements from one freshly
    allocated array. Those elements share one owner but collectively keep
    more than one qubit live. Their returned widths are summed and capped by
    the root allocation size so duplicate or overlapping views remain
    conservative without exceeding the underlying allocation.

    Args:
        results (Sequence[Value]): Quantum and classical operation results.
        resolver (ExprResolver): Resolver for symbolic result dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            QInit-result UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Live returned width by allocation owner.
    """
    returned: dict[str, ResourceExpr] = {}
    capacities: dict[str, ResourceExpr] = {}
    for result in results:
        if not result.type.is_quantum():
            continue
        runtime_sizes = _runtime_carrier_owner_sizes(
            result,
            allocation_owners_by_uuid or {},
        )
        if runtime_sizes is not None:
            for owner, size in runtime_sizes.items():
                returned[owner] = returned.get(owner, _ZERO) + size
                capacities[owner] = returned[owner]
            continue
        owner = (allocation_owners_by_uuid or {}).get(
            result.uuid,
            _quantum_allocation_owner(result),
        )
        returned[owner] = returned.get(owner, _ZERO) + _qubit_value_size(
            result,
            resolver,
        )
        capacities[owner] = _quantum_owner_capacity(result, resolver)
    return {owner: sp.Min(size, capacities[owner]) for owner, size in returned.items()}


def _quantum_owner_capacities(
    values: Sequence[ValueBase],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Resolve complete root-allocation widths touched by quantum values.

    A scalar array element keeps its complete root allocation live even though
    the element itself has width one. Conditional merge inputs use this helper
    because branch bodies can be structurally empty while their merge values
    still retain an enclosing allocation.

    Args:
        values (Sequence[ValueBase]): Candidate quantum values.
        resolver (ExprResolver): Resolver for symbolic root dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional QInit
            result UUID to logical-owner map. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Complete capacity by root allocation owner.
    """
    capacities: dict[str, ResourceExpr] = {}
    owner_map = allocation_owners_by_uuid or {}
    for value in values:
        if not isinstance(value, Value) or not value.type.is_quantum():
            continue
        owner = owner_map.get(value.uuid, _quantum_allocation_owner(value))
        capacities[owner] = _resource_max(
            capacities.get(owner, _ZERO),
            _quantum_owner_capacity(value, resolver),
        )
    return capacities


def _destructive_input_owner_sizes(
    operation: Operation,
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str],
) -> dict[str, ResourceExpr]:
    """Resolve physical owners consumed by measurement-like operations.

    Tuple-form expectation values use a synthetic ``ArrayValue`` whose runtime
    metadata retains each original element and, for borrowed array elements,
    its root-allocation UUID. Liveness needs that metadata because the
    synthetic carrier itself is not an allocation owner.

    Args:
        operation (Operation): Destructive quantum observation.
        resolver (ExprResolver): Resolver for ordinary quantum operand widths.
        allocation_owners_by_uuid (Mapping[str, str]): QInit-result UUIDs
            mapped to their root logical allocation IDs.

    Returns:
        dict[str, ResourceExpr]: Consumed qubit count by live allocation owner.
    """
    quantum_inputs = [
        value
        for value in operation.all_input_values()
        if isinstance(value, Value) and value.type.is_quantum()
    ]
    return _quantum_result_owner_sizes(
        quantum_inputs,
        resolver,
        allocation_owners_by_uuid,
    )


def _quantum_owner_capacity(
    value: Value,
    resolver: ExprResolver,
) -> ResourceExpr:
    """Return the complete allocation width for a quantum value's owner.

    Args:
        value (Value): Quantum scalar, array, or array view.
        resolver (ExprResolver): Resolver for symbolic dimensions.

    Returns:
        ResourceExpr: Root array width or one independent scalar width.
    """
    root = _quantum_root_array(value)
    return (
        _qubit_value_size(root, resolver)
        if root is not None
        else _qubit_value_size(value, resolver)
    )


def _invoke_quantum_output_sizes(
    operation: InvokeOperation,
    body: Block,
    child_resolver: ExprResolver,
    caller_resolver: ExprResolver,
    *,
    body_external_control_qubits: int,
    body_final_live: Mapping[str, ResourceExpr],
    actual_operands: Sequence[ValueBase],
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> tuple[dict[str, ResourceExpr], dict[str, ResourceExpr], bool]:
    """Map body liveness onto caller input and output allocation owners.

    A callee branch may return arrays whose branches have different symbolic
    widths even though the caller-side IR result retains one representative
    static shape. The callee resolver contains the merged shape binding, so
    invocation liveness must carry that size across the call boundary. The
    body summary also identifies allocations that remain live without being
    returned. Such residual workspace is retained under a call-local owner;
    allocations consumed inside the body are absent from the summary and are
    therefore not resurrected.

    Args:
        operation (InvokeOperation): Caller-side invocation.
        body (Block): Selected implementation body.
        child_resolver (ExprResolver): Resolver after evaluating the body.
        caller_resolver (ExprResolver): Resolver for caller-side controls.
        body_external_control_qubits (int): Number of leading caller controls
            implemented outside the selected body's input/output contract.
        body_final_live (Mapping[str, ResourceExpr]): Authoritative live widths
            by callee allocation owner after body evaluation.
        actual_operands (Sequence[ValueBase]): Caller operands aligned to the
            selected body after wrapper-only controls are removed.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional known
            QInit UUID to logical-owner map. Defaults to ``None``.

    Returns:
        tuple[dict[str, ResourceExpr], dict[str, ResourceExpr], bool]: Input
        widths, output widths, and whether positional output mapping was
        complete.
    """
    owner_map = allocation_owners_by_uuid or {}
    input_sizes = _quantum_result_owner_sizes(
        [value for value in operation.operands if value.type.is_quantum()],
        caller_resolver,
        owner_map,
    )
    sources: list[tuple[ValueBase, ExprResolver, bool]] = []
    if body_external_control_qubits:
        sources.extend(
            (operand, caller_resolver, False)
            for operand in operation.operands[:body_external_control_qubits]
        )
    sources.extend((output, child_resolver, True) for output in body.output_values)
    if len(sources) != len(operation.results):
        return input_sizes, {}, False

    output_sizes: dict[str, ResourceExpr] = {}
    capacities: dict[str, ResourceExpr] = {}
    returned_by_body_owner: dict[str, ResourceExpr] = {}
    for result, (source, source_resolver, comes_from_body) in zip(
        operation.results,
        sources,
        strict=True,
    ):
        if (
            not isinstance(result, Value)
            or not isinstance(source, Value)
            or not result.type.is_quantum()
        ):
            continue
        owner = owner_map.get(result.uuid, _quantum_allocation_owner(result))
        source_size = _qubit_value_size(source, source_resolver)
        output_sizes[owner] = output_sizes.get(owner, _ZERO) + _qubit_value_size(
            source,
            source_resolver,
        )
        capacities[owner] = _quantum_owner_capacity(result, caller_resolver)
        if comes_from_body:
            source_owner = owner_map.get(
                source.uuid,
                _quantum_allocation_owner(source),
            )
            returned_by_body_owner[source_owner] = (
                returned_by_body_owner.get(source_owner, _ZERO) + source_size
            )
    output_sizes = {
        owner: sp.Min(size, capacities[owner]) for owner, size in output_sizes.items()
    }

    formal_to_actual_owner: dict[str, str] = {}
    for formal, actual in pair_block_operands(body, actual_operands):
        if (
            not isinstance(formal, Value)
            or not isinstance(actual, Value)
            or not formal.type.is_quantum()
            or not actual.type.is_quantum()
        ):
            continue
        formal_owner = owner_map.get(
            formal.uuid,
            _quantum_allocation_owner(formal),
        )
        actual_owner = owner_map.get(
            actual.uuid,
            _quantum_allocation_owner(actual),
        )
        formal_to_actual_owner[formal_owner] = actual_owner

    namespace = f"{type(operation).__name__}:{id(operation)}/live"
    for body_owner, live_size in body_final_live.items():
        residual = sp.Max(
            _ZERO,
            live_size - returned_by_body_owner.get(body_owner, _ZERO),
        )
        if residual == _ZERO:
            continue
        caller_owner = formal_to_actual_owner.get(
            body_owner,
            f"{namespace}/{body_owner}",
        )
        output_sizes[caller_owner] = output_sizes.get(caller_owner, _ZERO) + residual
    return input_sizes, output_sizes, True


@dataclasses.dataclass(frozen=True)
class _LivenessSummary:
    """Carry liveness width and the owners still live after a body.

    Args:
        width (WidthResources): Liveness-aware width of the evaluated body.
        final_live_by_owner (dict[str, ResourceExpr]): Remaining live qubits
            keyed by root allocation owner at the body boundary.
    """

    width: WidthResources
    final_live_by_owner: dict[str, ResourceExpr]


def _liveness_width(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    initial_allocations: Mapping[str, ResourceExpr],
    resolver: ExprResolver,
    *,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> _LivenessSummary:
    """Compute peak width from affine allocation and consumption lifetimes.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations and
            their nested width summaries in program order.
        initial_allocations (Mapping[str, ResourceExpr]): Live input wire sizes
            keyed by logical ID.
        resolver (ExprResolver): Resolver for symbolic result dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional known
            QInit UUID to logical-owner map inherited from enclosing scopes.
            Defaults to ``None``.

    Returns:
        _LivenessSummary: Width and authoritative final live-owner state.
    """
    live = dict(initial_allocations)
    baseline = sum(live.values(), _ZERO)
    current = baseline
    peak = current
    allocated = _ZERO
    clean = _ZERO
    dirty = _ZERO
    resolved_allocation_owners = dict(allocation_owners_by_uuid or {})
    for operation, estimate in scheduled:
        clean = _resource_max(clean, estimate.width.clean_ancilla_qubits)
        dirty = _resource_max(dirty, estimate.width.dirty_ancilla_qubits)
        if isinstance(operation, QInitOperation):
            result = operation.results[0]
            resolved_allocation_owners[result.uuid] = result.logical_id
            if result.logical_id in live:
                # Formal quantum inputs are represented by QInit declarations
                # in traced blocks but remain caller-owned wires.
                continue
            amount = estimate.width.allocated_qubits
            allocated += amount
            live[result.logical_id] = amount
            current += amount
            peak = sp.Max(peak, current)
        else:
            allocated += estimate.width.allocated_qubits
            peak = sp.Max(peak, current + estimate.width.peak_qubits)
        if isinstance(
            operation,
            (
                MeasureOperation,
                MeasureVectorOperation,
                MeasureQFixedOperation,
                ExpvalOp,
            ),
        ):
            for owner, measured in _destructive_input_owner_sizes(
                operation,
                resolver,
                resolved_allocation_owners,
            ).items():
                previous = live.get(owner, _ZERO)
                remaining = sp.Max(_ZERO, previous - measured)
                current += remaining - previous
                if remaining == _ZERO:
                    live.pop(owner, None)
                else:
                    live[owner] = remaining
        if not isinstance(operation, QInitOperation):
            input_values = [
                value
                for value in operation.all_input_values()
                if isinstance(value, Value) and value.type.is_quantum()
            ]
            result_values = [
                result
                for result in operation.results
                if isinstance(result, Value) and result.type.is_quantum()
            ]
            input_sizes = _quantum_result_owner_sizes(input_values, resolver)
            if estimate._has_output_summary:
                # Boundary summaries describe both sides of the liveness
                # transfer. An empty input map is authoritative and means the
                # operation creates returned owners from no caller-owned
                # quantum input; falling back to syntactic merge/call inputs
                # would consume those freshly returned owners immediately.
                input_sizes = estimate._input_sizes
            elif estimate._input_sizes:
                input_sizes = estimate._input_sizes
            result_sizes = (
                estimate._output_sizes
                if estimate._has_output_summary
                else _quantum_result_owner_sizes(result_values, resolver)
            )
            if isinstance(operation, InvokeOperation):
                capacities = {
                    _quantum_allocation_owner(value): _quantum_owner_capacity(
                        value,
                        resolver,
                    )
                    for value in (*input_values, *result_values)
                }
                for owner in input_sizes.keys() | result_sizes.keys():
                    previous = live.get(owner, _ZERO)
                    consumed = input_sizes.get(owner, _ZERO)
                    returned = result_sizes.get(owner, _ZERO)
                    if estimate._has_output_summary and consumed == returned:
                        # Affine calls that return the same aggregate owner
                        # width preserve liveness exactly. Avoid expanding the
                        # equivalent ``Max(0, previous-consumed)+returned``;
                        # SymPy cannot generally cancel it through Min/Piecewise
                        # array-width expressions.
                        updated = previous
                    else:
                        remaining = sp.Max(
                            _ZERO,
                            previous - consumed,
                        )
                        updated = (
                            remaining + returned
                            if estimate._has_output_summary
                            else sp.Min(capacities[owner], remaining + returned)
                        )
                    current += updated - previous
                    if updated == _ZERO:
                        live.pop(owner, None)
                    else:
                        live[owner] = updated
            elif estimate._has_output_summary:
                for owner in input_sizes.keys() | result_sizes.keys():
                    previous = live.get(owner, _ZERO)
                    consumed = input_sizes.get(owner, _ZERO)
                    returned = result_sizes.get(owner, _ZERO)
                    if consumed == returned:
                        continue
                    remaining = sp.Max(
                        _ZERO,
                        previous - consumed,
                    )
                    updated = remaining + returned
                    current += updated - previous
                    if updated == _ZERO:
                        live.pop(owner, None)
                    else:
                        live[owner] = updated
            else:
                for owner, amount in result_sizes.items():
                    if owner in live or owner in input_sizes:
                        continue
                    live[owner] = amount
                    current += amount
            peak = sp.Max(peak, current)
    relative_peak = sp.Max(_ZERO, peak - baseline)
    static_width = allocated + clean + dirty
    if _is_structurally_nonnegative(relative_peak - static_width):
        # Static circuit width is a hard upper bound. If liveness arithmetic
        # already reached that bound, retain the compact exact expression
        # instead of exposing an equivalent nest of Max/Piecewise nodes.
        relative_peak = static_width
    else:
        relative_peak = sp.Min(relative_peak, static_width)
    return _LivenessSummary(
        width=WidthResources(
            allocated_qubits=allocated,
            clean_ancilla_qubits=clean,
            dirty_ancilla_qubits=dirty,
            peak_qubits=relative_peak,
        ),
        final_live_by_owner=dict(live),
    )


def _merge_allocation_sites(
    left: Mapping[str, ResourceExpr],
    right: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Merge static QInit sites without counting one identity twice.

    A QInit operation in a loop body is emitted once and reset before each
    replayed iteration. The same result UUID therefore denotes one allocation
    site even when concrete interpretation visits it repeatedly. If a symbolic
    site size differs between visits, the largest size is retained safely.

    Args:
        left (Mapping[str, ResourceExpr]): Sites collected so far.
        right (Mapping[str, ResourceExpr]): Sites from another estimate.

    Returns:
        dict[str, ResourceExpr]: Union keyed by QInit result UUID.
    """
    merged = dict(left)
    for site, size in right.items():
        previous = merged.get(site)
        merged[site] = size if previous is None else sp.Max(previous, size)
    return merged


def _namespace_allocation_sites(
    estimate: ResourceEstimate,
    operation: Operation,
) -> ResourceEstimate:
    """Qualify nested QInit identities by their static call site.

    Callable implementation Blocks are shared recipes: two distinct Invoke,
    ControlledU, or Inverse operations may traverse the SAME inner QInit UUID.
    The emitter clones/allocates those call sites independently, while repeated
    evaluation of ONE operation in a concrete loop must still reuse its site.
    Prefixing with the stable in-memory operation identity provides exactly
    that scope for one estimation run; the map is internal and excluded from
    equality and serialization, so process-local identities never leak.

    Args:
        estimate (ResourceEstimate): Nested body estimate to qualify.
        operation (Operation): Static call-site operation owning the body.

    Returns:
        ResourceEstimate: Estimate with call-site-qualified allocation keys.
    """
    if not estimate._allocation_sites:
        return estimate
    namespace = f"{type(operation).__name__}:{id(operation)}"
    return dataclasses.replace(
        estimate,
        _allocation_sites={
            f"{namespace}/{site}": size
            for site, size in estimate._allocation_sites.items()
        },
    )


def _activate_allocation_sites(
    sites: Mapping[str, ResourceExpr],
    condition: sp.Basic,
) -> dict[str, ResourceExpr]:
    """Guard allocation-site sizes by whether a repeated body executes.

    Args:
        sites (Mapping[str, ResourceExpr]): Static QInit sites in the body.
        condition (sp.Basic): Boolean expression that is true when the body
            executes at least once.

    Returns:
        dict[str, ResourceExpr]: Site sizes that become zero on a zero-trip
        path.
    """
    return {
        site: _resource_expr(sp.Piecewise((size, condition), (_ZERO, True)))
        for site, size in sites.items()
    }


def _allocation_site_total(
    sites: Mapping[str, ResourceExpr],
) -> ResourceExpr:
    """Sum the sizes of distinct static QInit identities.

    Args:
        sites (Mapping[str, ResourceExpr]): QInit result UUIDs and sizes.

    Returns:
        ResourceExpr: Total qubits allocated by the distinct sites.
    """
    return sum(sites.values(), _ZERO)


def _anonymous_allocation_width(
    width: WidthResources,
    sites: Mapping[str, ResourceExpr],
) -> ResourceExpr:
    """Return allocated width not attributable to explicit QInit sites.

    Opaque cost models may report allocated qubits without exposing an IR
    QInit identity. Their contribution stays reusable across iterations and
    is tracked separately from the identity-aware site union.

    Args:
        width (WidthResources): Width summary containing total allocations.
        sites (Mapping[str, ResourceExpr]): Explicit QInit sites represented in
            that summary.

    Returns:
        ResourceExpr: Nonnegative anonymous allocation contribution.
    """
    return sp.Max(
        _ZERO,
        sp.simplify(width.allocated_qubits - _allocation_site_total(sites)),
    )


def _width_with_identity_aware_allocations(
    width: WidthResources,
    sites: Mapping[str, ResourceExpr],
    *,
    anonymous_allocated: ResourceExpr | None = None,
) -> WidthResources:
    """Replace only allocated width with a static-site-aware total.

    Peak width and ancilla fields retain their liveness/reuse semantics. Only
    ``allocated_qubits`` distinguishes repeated visits to one QInit identity
    from visits to different identities.

    Args:
        width (WidthResources): Reusable width summary to preserve otherwise.
        sites (Mapping[str, ResourceExpr]): Distinct explicit QInit sites.
        anonymous_allocated (ResourceExpr | None): Reusable allocation amount
            without explicit site identities. Defaults to the residual derived
            from ``width``.

    Returns:
        WidthResources: Width with identity-aware ``allocated_qubits``.
    """
    residual = (
        _anonymous_allocation_width(width, sites)
        if anonymous_allocated is None
        else anonymous_allocated
    )
    return dataclasses.replace(
        width,
        allocated_qubits=sp.simplify(_allocation_site_total(sites) + residual),
    )


def _branch_width_with_static_allocations(
    branch_width: WidthResources,
    left_width: WidthResources,
    left_sites: Mapping[str, ResourceExpr],
    right_width: WidthResources,
    right_sites: Mapping[str, ResourceExpr],
    merged_sites: Mapping[str, ResourceExpr],
) -> WidthResources:
    """Combine branch liveness with the union of static allocation sites.

    Runtime branches are mutually exclusive, so peak live width remains a
    maximum or Piecewise expression. Static circuit allocation is different:
    emitters reserve distinct QInit sites from both branches. Anonymous opaque
    allocations have no identity that proves reuse, so their branch residuals
    are conservatively added as well.

    Args:
        branch_width (WidthResources): Liveness-aware maximum or conditional
            branch width.
        left_width (WidthResources): True/left branch width.
        left_sites (Mapping[str, ResourceExpr]): Explicit allocation sites in
            the true/left branch.
        right_width (WidthResources): False/right branch width.
        right_sites (Mapping[str, ResourceExpr]): Explicit allocation sites in
            the false/right branch.
        merged_sites (Mapping[str, ResourceExpr]): Union of explicit sites from
            both branches.

    Returns:
        WidthResources: Branch width with conservative static allocations.
    """
    anonymous_allocated = _anonymous_allocation_width(
        left_width,
        left_sites,
    ) + _anonymous_allocation_width(
        right_width,
        right_sites,
    )
    return _width_with_identity_aware_allocations(
        branch_width,
        merged_sites,
        anonymous_allocated=anonymous_allocated,
    )


def _maximum_width_over_range(
    width: WidthResources,
    sites: Mapping[str, ResourceExpr],
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[WidthResources, dict[str, ResourceExpr], Boolean]:
    """Maximize reusable width across a symbolic loop range.

    Additive resources such as gates and depth are summed across loop
    iterations, but qubits can be reused. Explicit allocation sites are
    maximized individually because a static emitted circuit reserves every
    distinct site, even when different sizes occur on different iterations.

    Args:
        width (WidthResources): Width used by one symbolic loop iteration.
        sites (Mapping[str, ResourceExpr]): Explicit QInit sites in the body.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[WidthResources, dict[str, ResourceExpr], Boolean]: Maximum
        reusable width, maximized allocation sites, and the condition under
        which at least one conservative maximum may overestimate.
    """
    maximized_sites: dict[str, ResourceExpr] = {}
    conservative_guards: list[Boolean] = []
    for site, size in sites.items():
        maximum, conservative_when = _maximum_expr_over_range(
            size,
            loop_symbol,
            start,
            step,
            iterations,
        )
        maximized_sites[site] = maximum
        conservative_guards.append(conservative_when)

    anonymous = _anonymous_allocation_width(width, sites)
    maximum_anonymous, anonymous_conservative_when = _maximum_expr_over_range(
        anonymous,
        loop_symbol,
        start,
        step,
        iterations,
    )
    conservative_guards.append(anonymous_conservative_when)

    maxima: dict[str, ResourceExpr] = {}
    for field in dataclasses.fields(WidthResources):
        if field.name == "allocated_qubits":
            continue
        maximum, conservative_when = _maximum_expr_over_range(
            getattr(width, field.name),
            loop_symbol,
            start,
            step,
            iterations,
        )
        maxima[field.name] = maximum
        conservative_guards.append(conservative_when)
    maximized_width = WidthResources(
        allocated_qubits=_ZERO,
        **maxima,
    )
    return (
        _width_with_identity_aware_allocations(
            maximized_width,
            maximized_sites,
            anonymous_allocated=maximum_anonymous,
        ),
        maximized_sites,
        _boolean_condition(sp.Or(*conservative_guards)),
    )


def _maximum_live_owner_sizes(
    summaries: Sequence[Mapping[str, ResourceExpr]],
) -> tuple[dict[str, ResourceExpr], Boolean]:
    """Retain the largest observed live size for every loop-local owner.

    A later concrete iteration may report a smaller size for the same static
    allocation site. Without an explicit inter-iteration release proof, using
    only that final report could silently discard qubits that were live after
    an earlier iteration. Taking the owner-wise maximum is order-independent
    and conservative.

    Args:
        summaries (Sequence[Mapping[str, ResourceExpr]]): Live-owner summaries
            produced after each concrete iteration.

    Returns:
        tuple[dict[str, ResourceExpr], Boolean]: Owner-wise maximum sizes and
        the condition under which the maximum retains anything not present in
        the final summary.
    """
    if not summaries:
        return {}, sp.false
    owners = sorted({owner for summary in summaries for owner in summary})
    maxima = {
        owner: _resource_max_many([summary.get(owner, _ZERO) for summary in summaries])
        for owner in owners
    }
    final = summaries[-1]
    retains_prior_when = _boolean_condition(
        sp.Or(
            *(
                sp.Gt(maximum, final.get(owner, _ZERO))
                for owner, maximum in maxima.items()
            )
        )
    )
    return maxima, retains_prior_when


def _maximum_live_owner_sizes_over_range(
    sizes: Mapping[str, ResourceExpr],
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[dict[str, ResourceExpr], Boolean, Boolean]:
    """Maximize live loop-local owners across a symbolic iteration range.

    Args:
        sizes (Mapping[str, ResourceExpr]): Per-iteration live-owner sizes.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[dict[str, ResourceExpr], Boolean, Boolean]: Owner-wise maximum
        sizes, the condition under which a symbolic maximum may overestimate,
        and the condition under which a maximum retains owner width from
        before the final iteration.
    """
    maxima: dict[str, ResourceExpr] = {}
    conservative_guards: list[Boolean] = []
    retention_guards: list[Boolean] = []
    for owner in sorted(sizes):
        size = sizes[owner]
        maximum, conservative_when = _maximum_expr_over_range(
            size,
            loop_symbol,
            start,
            step,
            iterations,
        )
        maxima[owner] = maximum
        conservative_guards.append(conservative_when)
        final = _piecewise(
            cast(
                ResourceExpr,
                size.subs(loop_symbol, start + (iterations - _ONE) * step),
            ),
            _ZERO,
            sp.Gt(iterations, _ZERO),
        )
        maximum_matches_final = _safe_simplify(maximum - final) == _ZERO
        if (
            not maximum_matches_final
            and conservative_when is sp.false
            and loop_symbol in size.free_symbols
        ):
            index = sp.Dummy("live_owner_index", integer=True, nonnegative=True)
            transformed = cast(
                sp.Expr,
                size.subs(loop_symbol, start + step * index),
            )
            try:
                polynomial = sp.Poly(transformed, index)
            except sp.PolynomialError:
                polynomial = None
            maximum_matches_final = (
                polynomial is not None
                and polynomial.degree() <= 1
                and _is_structurally_nonnegative(polynomial.coeff_monomial(index))
            )
        retention_guards.append(
            sp.false
            if maximum_matches_final
            else _boolean_condition(sp.Gt(maximum, final))
        )
    return (
        maxima,
        _boolean_condition(sp.Or(*conservative_guards)),
        _boolean_condition(sp.Or(*retention_guards)),
    )


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
