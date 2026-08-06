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

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
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
