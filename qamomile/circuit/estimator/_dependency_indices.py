"""Represent and compare scheduler wire indices."""

from __future__ import annotations

import dataclasses
import enum
import functools
from collections.abc import Mapping
from typing import (
    Any,
    cast,
)

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _expr,
    _is_structurally_less_equal,
    _safe_simplify,
)


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


_WIRE_RANGE_OFFSET = sp.Dummy(
    "wire_range_offset",
    integer=True,
    nonnegative=True,
)


_MAX_EXACT_LOOP_WIRE_EXPANSION = 256


_MAX_EXACT_LOOP_DISJOINTNESS_EXPANSION = 4096


_MAX_CONCRETE_VIEW_WIRE_EXPANSION = 4096


class _WireRelation(enum.Enum):
    """Classify whether two physical wire addresses overlap."""

    DEFINITE_OVERLAP = enum.auto()
    DISJOINT = enum.auto()
    POSSIBLE_OVERLAP = enum.auto()


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


def _range_positive_extent(iterations: ResourceExpr) -> ResourceExpr:
    """Return the branch used by a nonempty clipped iteration count.

    A Python-range trip count is commonly represented as ``Max(0, extent)``.
    Coverage is vacuous when that count is zero; when it is nonempty, the
    nonzero branch is the exact extent. Other expressions are left unchanged
    so callers fail closed unless their bounds are globally provable.

    Args:
        iterations (ResourceExpr): Nonnegative range trip-count expression.

    Returns:
        ResourceExpr: Exact extent on the range's nonempty branch.
    """
    if isinstance(iterations, sp.Max) and len(iterations.args) == 2:
        if iterations.args[0] == _ZERO:
            return cast(ResourceExpr, iterations.args[1])
        if iterations.args[1] == _ZERO:
            return cast(ResourceExpr, iterations.args[0])
    return iterations


def _wire_range_affine_signature(
    index: _WireRangeIndex,
) -> tuple[ResourceExpr, ResourceExpr] | None:
    """Recover one exact affine offset map from a symbolic wire range.

    Args:
        index (_WireRangeIndex): Symbolic range footprint to analyze.

    Returns:
        tuple[ResourceExpr, ResourceExpr] | None: Nonzero slope and start
            address, or ``None`` when exact affine reconstruction fails.
    """
    expression = index.index_at_offset
    slope = _safe_simplify(cast(ResourceExpr, sp.diff(expression, _WIRE_RANGE_OFFSET)))
    start = _safe_simplify(
        cast(
            ResourceExpr,
            expression.subs(_WIRE_RANGE_OFFSET, _ZERO, simultaneous=True),
        )
    )
    if (
        _WIRE_RANGE_OFFSET in slope.free_symbols
        or _WIRE_RANGE_OFFSET in start.free_symbols
        or slope.is_zero is not False
        or slope.is_integer is not True
        or slope.is_finite is not True
    ):
        return None
    reconstructed = _safe_simplify(
        cast(ResourceExpr, slope * _WIRE_RANGE_OFFSET + start)
    )
    if _safe_simplify(cast(ResourceExpr, expression - reconstructed)) != _ZERO:
        return None
    return slope, start


@functools.lru_cache(maxsize=4096)
def _wire_index_covers(covering: WireIndex, required: WireIndex) -> bool:
    """Prove that one owner-local footprint contains another footprint.

    The additional range proof is deliberately narrow: both address maps
    must be exact affine progressions, and every required address must map to
    an integer offset on the covering progression's lattice. This includes
    sparse and opposite-direction traversal of a denser covering range. A
    clipped ``Max(0, extent)`` is compared on its nonempty branch; its empty
    branch is covered vacuously.

    Args:
        covering (WireIndex): Candidate superset footprint.
        required (WireIndex): Footprint that must be fully contained.

    Returns:
        bool: Whether containment is proven for every admissible value.
    """
    if (
        covering is None
        or required is None
        or covering is _UNKNOWN_WIRE_INDEX
        or required is _UNKNOWN_WIRE_INDEX
    ):
        return False
    if covering == required:
        return True
    if not isinstance(covering, _WireRangeIndex) or not isinstance(
        required, _WireRangeIndex
    ):
        return False
    covering_signature = _wire_range_affine_signature(covering)
    required_signature = _wire_range_affine_signature(required)
    if covering_signature is None or required_signature is None:
        return False
    covering_slope, covering_start = covering_signature
    required_slope, required_start = required_signature
    stride_ratio_expression = _safe_simplify(
        cast(ResourceExpr, required_slope / covering_slope)
    )
    if stride_ratio_expression.is_number is not True or not _is_concrete_integer(
        stride_ratio_expression
    ):
        return False
    stride_ratio = int(stride_ratio_expression)
    offset_shift = _safe_simplify(
        cast(
            ResourceExpr,
            (required_start - covering_start) / covering_slope,
        )
    )
    if offset_shift.is_integer is not True:
        return False
    required_extent = _range_positive_extent(required.iterations)
    covering_extent = _range_positive_extent(covering.iterations)
    required_last_offset = _safe_simplify(
        cast(
            ResourceExpr,
            offset_shift + stride_ratio * (required_extent - _ONE),
        )
    )
    required_low, required_high = (
        (offset_shift, required_last_offset)
        if stride_ratio > 0
        else (required_last_offset, offset_shift)
    )
    return _is_structurally_less_equal(
        _ZERO, required_low
    ) and _is_structurally_less_equal(
        cast(ResourceExpr, required_high + _ONE),
        covering_extent,
    )


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


def _wire_range_scalar_membership_condition(
    wire_range: _WireRangeIndex,
    scalar: int | ResourceExpr,
) -> Boolean | None:
    """Return the exact condition under which a range contains one scalar.

    The proof is deliberately limited to an exact affine address progression
    whose scalar maps to an integer offset.  Returning ``None`` keeps every
    unsupported symbolic lattice fail-closed in the caller.

    Args:
        wire_range (_WireRangeIndex): Candidate affine range footprint.
        scalar (int | ResourceExpr): Candidate scalar address on the same
            allocation owner.

    Returns:
        Boolean | None: Exact membership condition, or ``None`` when integer
            membership cannot be proven symbolically.
    """
    signature = _wire_range_affine_signature(wire_range)
    if signature is None:
        return None
    slope, start = signature
    offset = _safe_simplify(cast(ResourceExpr, (_expr(scalar) - start) / slope))
    if offset.is_integer is not True or offset.is_finite is not True:
        return None
    return _boolean_condition(
        sp.And(
            sp.Ge(offset, _ZERO),
            sp.Lt(offset, _range_positive_extent(wire_range.iterations)),
        )
    )


def _wire_index_relation_under(
    left: WireIndex,
    right: WireIndex,
    active_when: Boolean,
) -> _WireRelation:
    """Classify two indices under a guarded operation-activity condition.

    The ordinary symmetric relation intentionally leaves every range/scalar
    pair as a possible alias.  A dependency scheduler has a directional
    activity guard, so it may additionally prove that an affine range either
    contains or excludes one scalar whenever that guard is active.  This does
    not change the public symmetric alias relation.

    Args:
        left (WireIndex): First owner-local wire index.
        right (WireIndex): Second owner-local wire index.
        active_when (Boolean): Condition under which the relation matters.

    Returns:
        _WireRelation: Guard-specialized overlap classification, falling back
            to ``POSSIBLE_OVERLAP`` whenever membership remains unresolved.
    """
    overlap = _wire_index_overlap_condition(left, right)
    if overlap is None:
        return _WireRelation.POSSIBLE_OVERLAP
    if (
        active_when is sp.false
        or overlap is sp.false
        or _boolean_guard_implies(
            active_when,
            sp.Not(overlap),
        )
    ):
        return _WireRelation.DISJOINT
    if overlap is sp.true or _boolean_guard_implies(active_when, overlap):
        return _WireRelation.DEFINITE_OVERLAP
    return _WireRelation.POSSIBLE_OVERLAP


def _boolean_guard_implies(active: Boolean, required: Boolean) -> bool:
    """Prove a narrow structural implication between Boolean guards.

    The helper recognizes conjunctions and monotone lower bounds sharing the
    exact same symbolic left-hand side. It deliberately avoids SymPy's general
    inequality solver, which can be both expensive and exception-prone for
    nested ``Min``/``Max`` resource expressions.

    Args:
        active (Boolean): Known operation-activity guard.
        required (Boolean): Predicate that must follow from ``active``.

    Returns:
        bool: Whether the limited structural rules prove the implication.
    """
    if active is sp.false or required is sp.true or active == required:
        return True
    if required is sp.false:
        return False
    if isinstance(required, sp.And):
        return all(
            _boolean_guard_implies(active, cast(Boolean, term))
            for term in required.args
        )
    active_terms = active.args if isinstance(active, sp.And) else (active,)
    if required in active_terms:
        return True
    if not getattr(required, "is_Relational", False):
        return False
    required_relation = cast(Any, required)
    if required_relation.rel_op in (">", ">="):
        required_lhs = required_relation.lhs
        required_rhs = required_relation.rhs
        required_strict = required_relation.rel_op == ">"
    elif required_relation.rel_op in ("<", "<="):
        required_lhs = required_relation.rhs
        required_rhs = required_relation.lhs
        required_strict = required_relation.rel_op == "<"
    else:
        return False
    for term in active_terms:
        if not getattr(term, "is_Relational", False):
            continue
        active_relation = cast(Any, term)
        if active_relation.rel_op in (">", ">="):
            active_lhs = active_relation.lhs
            active_rhs = active_relation.rhs
            active_strict = active_relation.rel_op == ">"
        elif active_relation.rel_op in ("<", "<="):
            active_lhs = active_relation.rhs
            active_rhs = active_relation.lhs
            active_strict = active_relation.rel_op == "<"
        else:
            continue
        if (
            isinstance(active_lhs, sp.Max)
            and _ZERO in active_lhs.args
            and len(active_lhs.args) == 2
            and (
                active_rhs.is_positive is True
                or (active_strict and active_rhs.is_nonnegative is True)
            )
        ):
            active_lhs = next(
                argument for argument in active_lhs.args if argument != _ZERO
            )
        if active_lhs != required_lhs:
            continue
        threshold_difference = _safe_simplify(
            cast(
                ResourceExpr,
                active_rhs - required_rhs,
            )
        )
        if threshold_difference.is_nonnegative is not True:
            continue
        if (
            not required_strict
            or active_strict
            or threshold_difference.is_positive is True
        ):
            return True
    return False


def _wire_index_overlap_condition(
    left: WireIndex,
    right: WireIndex,
) -> Boolean | None:
    """Return an exact symbolic overlap condition when one is available.

    Args:
        left (WireIndex): First owner-local scalar or range address.
        right (WireIndex): Second owner-local scalar or range address.

    Returns:
        Boolean | None: Exact overlap predicate, or ``None`` when the address
            forms require conservative possible-alias handling.
    """
    relation = _wire_index_relation(left, right)
    if relation is _WireRelation.DEFINITE_OVERLAP:
        return sp.true
    if relation is _WireRelation.DISJOINT:
        return sp.false
    if isinstance(left, _WireRangeIndex) and isinstance(right, (int, sp.Expr)):
        return _wire_range_scalar_membership_condition(left, right)
    if isinstance(right, _WireRangeIndex) and isinstance(left, (int, sp.Expr)):
        return _wire_range_scalar_membership_condition(right, left)
    if isinstance(left, (int, sp.Expr)) and isinstance(right, (int, sp.Expr)):
        return _boolean_condition(sp.Eq(_expr(left), _expr(right)))
    return None


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
