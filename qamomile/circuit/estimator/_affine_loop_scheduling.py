"""Prove symbolic loop schedules from normalized quantum access maps."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_footprints import (
    _expand_dependency_owner_aliases,
)
from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    WireKey,
    _normalize_wire_index,
    _symbolic_wire_range_index,
    _wire_index_relation,
    _WireRelation,
)
from qamomile.circuit.estimator._quantum_values import (
    _quantum_allocation_owner,
    _quantum_element_index_expression,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_algebra import _conditional_depth
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import DepthResources
from qamomile.circuit.estimator._scopes import _LocalBlock
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value


class _LoopCompletionMode(enum.Enum):
    """Select how an exact symbolic loop proof exposes wire completion."""

    KEEP_SEQUENTIAL = enum.auto()
    AGGREGATE_SAFE = enum.auto()
    PROJECT_PARALLEL = enum.auto()


@dataclasses.dataclass(frozen=True)
class _LoopAxis:
    """Describe one symbolic Python-range axis.

    Args:
        symbol (sp.Symbol): Internal induction symbol.
        start (ResourceExpr): Inclusive range start.
        stop (ResourceExpr): Exclusive range stop.
        step (ResourceExpr): Signed range step.
        iterations (ResourceExpr): Exact number of executed iterations.
    """

    symbol: sp.Symbol
    start: ResourceExpr
    stop: ResourceExpr
    step: ResourceExpr
    iterations: ResourceExpr


@dataclasses.dataclass(frozen=True)
class _AffineIndex:
    """Store an exactly reconstructed affine scalar index.

    Args:
        expression (ResourceExpr): Original scalar index expression.
        coefficients (tuple[ResourceExpr, ...]): Coefficient for each loop
            symbol supplied to the analyzer.
        residual (ResourceExpr): Binder-independent affine residual.
    """

    expression: ResourceExpr
    coefficients: tuple[ResourceExpr, ...]
    residual: ResourceExpr


@dataclasses.dataclass(frozen=True)
class _WireAccess:
    """Describe one resolved physical scalar-wire access.

    Args:
        owner (str): Root allocation owner.
        index (ResourceExpr | None): Root scalar index, or ``None`` for one
            standalone scalar qubit.
        affine (_AffineIndex | None): Exact affine form when one can be proven.
    """

    owner: str
    index: ResourceExpr | None
    affine: _AffineIndex | None


@dataclasses.dataclass(frozen=True)
class _AccessSummary:
    """Collect normalized accesses for one loop-body operation list.

    Args:
        accesses (tuple[_WireAccess, ...]): Every resolved quantum input and
            result access in the body.
        gate_inputs (tuple[_WireAccess, _WireAccess] | None): Input addresses
            of an unconditional, address-preserving two-qubit gate when the
            body contains exactly one quantum operation.
    """

    accesses: tuple[_WireAccess, ...]
    gate_inputs: tuple[_WireAccess, _WireAccess] | None


@dataclasses.dataclass(frozen=True)
class _SymbolicLoopSchedule:
    """Carry one exact symbolic schedule certificate to the interpreter.

    Args:
        depth (DepthResources): Proven aggregate loop depth.
        completion_mode (_LoopCompletionMode): How caller-visible dependency
            completion must be formed.
        synchronized_entry_conditions (Mapping[WireKey, Boolean]): Guarded
            physical inputs that must share one entry layer for the aggregate
            formula to remain exact under composition.
        serial_frontier (frozenset[WireKey] | None): Exact physical operands
            of the first serial-chain gate. ``None`` means that the narrower
            frontier premise could not be represented safely.
        serial_active_when (Boolean): Guard under which the serial frontier
            certificate is required.
    """

    depth: DepthResources
    completion_mode: _LoopCompletionMode
    synchronized_entry_conditions: Mapping[WireKey, Boolean]
    serial_frontier: frozenset[WireKey] | None = None
    serial_active_when: Boolean = sp.false


def _align_affine_entry_conditions(
    conditions: Mapping[WireKey, Boolean],
    dependency_keys: frozenset[WireKey] | None,
) -> Mapping[WireKey, Boolean]:
    """Align affine entry requirements with already-projected dependencies.

    ``ResourceEstimate.sum_over`` may enumerate a bounded concrete range even
    when the affine certificate retains its compact symbolic range key. The
    enclosing scheduler must compare like representations so a uniform full
    write can prove that those inputs were re-synchronized.

    Args:
        conditions (Mapping[WireKey, Boolean]): Certificate-local guarded
            entry requirements.
        dependency_keys (frozenset[WireKey] | None): Caller-visible dependency
            footprint after ordinary range projection.

    Returns:
        Mapping[WireKey, Boolean]: Requirements expressed through matching
            caller-visible keys, or the original map when no precise
            projection is available.
    """
    if dependency_keys is None:
        return conditions
    aligned: dict[WireKey, Boolean] = {}
    matched_requirements: set[WireKey] = set()
    for dependency_key in dependency_keys:
        dependency_owner, dependency_index = dependency_key
        matching: list[Boolean] = []
        for requirement, condition in conditions.items():
            owner, index = requirement
            if (
                owner != dependency_owner
                or _wire_index_relation(index, dependency_index)
                is _WireRelation.DISJOINT
            ):
                continue
            matching.append(condition)
            matched_requirements.add(requirement)
        if matching:
            aligned[dependency_key] = _boolean_condition(sp.Or(*matching))
    aligned.update(
        {
            requirement: condition
            for requirement, condition in conditions.items()
            if requirement not in matched_requirements
        }
    )
    return aligned


def _exact_affine_index(
    expression: ResourceExpr,
    symbols: Sequence[sp.Symbol],
) -> _AffineIndex | None:
    """Reconstruct one scalar index as an exact affine expression.

    Differentiation alone is insufficient because a Piecewise expression can
    have a constant derivative while still changing its offset at a branch.
    This helper therefore verifies the coefficients, residual, and complete
    reconstruction independently of every loop-local binder.

    Args:
        expression (ResourceExpr): Candidate scalar index expression.
        symbols (Sequence[sp.Symbol]): Loop-local binders in stable order.

    Returns:
        _AffineIndex | None: Exact affine representation, or ``None`` when the
        expression remains nonlinear or conditionally binder-dependent.
    """
    coefficients: list[ResourceExpr] = []
    try:
        for symbol in symbols:
            coefficient = _safe_simplify(
                cast(ResourceExpr, sp.diff(expression, symbol))
            )
            if any(local in coefficient.free_symbols for local in symbols):
                return None
            coefficients.append(coefficient)
    except (TypeError, ValueError):
        return None
    residual = _safe_simplify(
        cast(
            ResourceExpr,
            expression
            - sum(
                (
                    coefficient * symbol
                    for coefficient, symbol in zip(
                        coefficients,
                        symbols,
                        strict=True,
                    )
                ),
                _ZERO,
            ),
        )
    )
    if any(local in residual.free_symbols for local in symbols):
        return None
    reconstructed = cast(
        ResourceExpr,
        residual
        + sum(
            (
                coefficient * symbol
                for coefficient, symbol in zip(
                    coefficients,
                    symbols,
                    strict=True,
                )
            ),
            _ZERO,
        ),
    )
    if _safe_simplify(cast(ResourceExpr, expression - reconstructed)) != _ZERO:
        return None
    return _AffineIndex(
        expression=expression,
        coefficients=tuple(coefficients),
        residual=residual,
    )


def _scalar_wire_access(
    value: Value,
    resolver: ExprResolver,
    *,
    symbols: Sequence[sp.Symbol],
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> _WireAccess | None:
    """Resolve one quantum scalar to a unique root address and affine form.

    Args:
        value (Value): Quantum scalar input or result.
        resolver (ExprResolver): Resolver scoped to the value's operations.
        symbols (Sequence[sp.Symbol]): Loop-local binders.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases. A nontrivial alias closure is rejected. Defaults to
            ``None``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        _WireAccess | None: Unique scalar address, or ``None`` when the value
        is a whole array, unresolved element, or conditional owner alias.
    """
    if isinstance(value, ArrayValue):
        return None
    owner = _quantum_allocation_owner(value)
    index: ResourceExpr | None
    if value.parent_array is None:
        index = None
    else:
        resolved_index = _quantum_element_index_expression(
            value,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        if resolved_index is None:
            return None
        normalized = _normalize_wire_index(resolved_index)
        if isinstance(normalized, int):
            index = sp.Integer(normalized)
        elif isinstance(normalized, sp.Expr):
            index = cast(ResourceExpr, normalized)
        else:
            return None
    expanded = _expand_dependency_owner_aliases({(owner, index)}, owner_aliases)
    if len(expanded) != 1:
        return None
    affine = None if index is None else _exact_affine_index(index, symbols)
    return _WireAccess(owner=owner, index=index, affine=affine)


def _analyze_accesses(
    operations: Sequence[Operation],
    resolver: ExprResolver,
    *,
    symbols: Sequence[sp.Symbol],
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> _AccessSummary | None:
    """Normalize every quantum access in a supported loop body.

    The general footprint accepts nested ``IfOperation`` regions so the
    disjoint proof keeps its existing coverage. A single-gate summary is
    exposed only when the top-level body consists of one two-qubit gate plus
    classical ``BinOp`` index preparation.

    Args:
        operations (Sequence[Operation]): Candidate body operations.
        resolver (ExprResolver): Resolver scoped to the body.
        symbols (Sequence[sp.Symbol]): Loop-local binders.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases. Defaults to ``None``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        _AccessSummary | None: Complete normalized access summary, or ``None``
        when any operation or quantum address cannot be represented safely.
    """
    ignored_operations = (
        QInitOperation,
        GlobalPhaseOperation,
        StoreArrayElementOperation,
        ReturnQuantumArrayElementOperation,
    )
    pending = list(reversed(operations))
    accesses: list[_WireAccess] = []
    top_level_gates = [item for item in operations if isinstance(item, GateOperation)]
    single_gate_shape = len(top_level_gates) == 1 and all(
        isinstance(item, (BinOp, GateOperation)) for item in operations
    )
    while pending:
        operation = pending.pop()
        if isinstance(operation, ignored_operations):
            continue
        if isinstance(operation, HasNestedOps):
            if not isinstance(operation, IfOperation):
                return None
            single_gate_shape = False
            for region in reversed(operation.nested_regions()):
                pending.extend(reversed(region.operations))
            continue
        for value in (*operation.all_input_values(), *operation.results):
            if not isinstance(value, Value) or not value.type.is_quantum():
                continue
            access = _scalar_wire_access(
                value,
                resolver,
                symbols=symbols,
                owner_aliases=owner_aliases,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            if access is None:
                return None
            accesses.append(access)
    gate_inputs: tuple[_WireAccess, _WireAccess] | None = None
    if single_gate_shape:
        gate = top_level_gates[0]
        if len(gate.qubit_operands) == 2 and len(gate.results) == 2:
            inputs = tuple(
                _scalar_wire_access(
                    value,
                    resolver,
                    symbols=symbols,
                    owner_aliases=owner_aliases,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
                for value in gate.qubit_operands
            )
            results = tuple(
                _scalar_wire_access(
                    value,
                    resolver,
                    symbols=symbols,
                    owner_aliases=owner_aliases,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
                for value in gate.results
            )
            if not any(access is None for access in (*inputs, *results)):
                typed_inputs = cast(tuple[_WireAccess, _WireAccess], inputs)
                typed_results = cast(tuple[_WireAccess, _WireAccess], results)
                if all(
                    before.owner == after.owner
                    and _wire_index_relation(before.index, after.index)
                    is _WireRelation.DEFINITE_OVERLAP
                    for before, after in zip(
                        typed_inputs,
                        typed_results,
                        strict=True,
                    )
                ):
                    gate_inputs = typed_inputs
    return _AccessSummary(accesses=tuple(accesses), gate_inputs=gate_inputs)


def _depth_is_one_gate_per_active_field(depth: DepthResources) -> bool:
    """Return whether every active depth field represents one gate layer.

    Args:
        depth (DepthResources): Candidate one-iteration depth profile.

    Returns:
        bool: Whether every field is zero or one and at least one is active.
    """
    values = tuple(
        cast(ResourceExpr, getattr(depth, field.name))
        for field in dataclasses.fields(DepthResources)
    )
    return any(value != _ZERO for value in values) and all(
        value in (_ZERO, _ONE) for value in values
    )


def _project_accesses(
    accesses: Sequence[_WireAccess],
    axis: _LoopAxis,
) -> frozenset[WireKey]:
    """Project flat body accesses over one symbolic range.

    Args:
        accesses (Sequence[_WireAccess]): Body-scoped physical accesses.
        axis (_LoopAxis): Range whose binder leaves scope.

    Returns:
        frozenset[WireKey]: Binder-independent scalar and symbolic-range keys.
    """
    concrete_values: tuple[sp.Integer, ...] | None = None
    bounds = tuple(
        _safe_simplify(value) for value in (axis.start, axis.step, axis.iterations)
    )
    if all(value.is_number and _is_concrete_integer(value) for value in bounds):
        concrete_start, concrete_step, concrete_iterations = (
            int(value) for value in bounds
        )
        if (
            concrete_step != 0
            and 0 <= concrete_iterations <= _MAX_EXACT_LOOP_WIRE_EXPANSION
        ):
            concrete_values = tuple(
                sp.Integer(concrete_start + concrete_step * offset)
                for offset in range(concrete_iterations)
            )
    projected: set[WireKey] = set()
    for access in accesses:
        if access.index is None:
            projected.add((access.owner, None))
            continue
        if axis.symbol not in access.index.free_symbols:
            projected.add((access.owner, _normalize_wire_index(access.index)))
            continue
        if concrete_values is not None:
            projected.update(
                (
                    access.owner,
                    _normalize_wire_index(
                        cast(
                            ResourceExpr,
                            access.index.subs(
                                axis.symbol,
                                value,
                                simultaneous=True,
                            ),
                        )
                    ),
                )
                for value in concrete_values
            )
            continue
        projected.add(
            (
                access.owner,
                _symbolic_wire_range_index(
                    access.index,
                    axis.symbol,
                    start=axis.start,
                    step=axis.step,
                    iterations=axis.iterations,
                ),
            )
        )
    return frozenset(projected)


def _entry_conditions(
    keys: frozenset[WireKey],
    active_when: Boolean,
) -> Mapping[WireKey, Boolean]:
    """Build one guarded synchronized-entry requirement map.

    Args:
        keys (frozenset[WireKey]): Physical addresses requiring one entry
            layer.
        active_when (Boolean): Condition under which synchronization matters.

    Returns:
        Mapping[WireKey, Boolean]: Per-address synchronization conditions.
    """
    return {key: active_when for key in keys}


def _prove_disjoint_schedule(
    summary: _AccessSummary,
    body_depth: DepthResources,
    axis: _LoopAxis,
) -> _SymbolicLoopSchedule | None:
    """Prove parallel depth for one injective affine footprint per owner.

    Args:
        summary (_AccessSummary): Normalized flat body accesses.
        body_depth (DepthResources): One-iteration depth profile.
        axis (_LoopAxis): Flat range axis.

    Returns:
        _SymbolicLoopSchedule | None: Parallel schedule certificate, or
        ``None`` when injectivity or depth invariance is not proven.
    """
    if any(
        axis.symbol in cast(ResourceExpr, getattr(body_depth, field.name)).free_symbols
        for field in dataclasses.fields(DepthResources)
    ):
        return None
    if not summary.accesses and any(
        getattr(body_depth, field.name) != _ZERO
        for field in dataclasses.fields(DepthResources)
    ):
        return None
    by_owner: dict[str, list[_WireAccess]] = {}
    for access in summary.accesses:
        by_owner.setdefault(access.owner, []).append(access)
    for accesses in by_owner.values():
        representative = accesses[0]
        if representative.affine is None:
            return None
        if any(
            access.affine is None
            or _wire_index_relation(access.index, representative.index)
            is not _WireRelation.DEFINITE_OVERLAP
            for access in accesses
        ):
            return None
        slope = representative.affine.coefficients[0]
        if slope.is_zero is not False:
            return None
    projected = _project_accesses(summary.accesses, axis)
    active = _boolean_condition(sp.Gt(axis.iterations, _ZERO))
    return _SymbolicLoopSchedule(
        depth=_conditional_depth(
            body_depth,
            DepthResources.zero(),
            active,
        ),
        completion_mode=_LoopCompletionMode.PROJECT_PARALLEL,
        synchronized_entry_conditions=_entry_conditions(projected, active),
    )


def _accesses_match_across_next_iteration(
    current: _WireAccess,
    following: _WireAccess,
    axis: _LoopAxis,
) -> bool:
    """Return whether two accesses identify one wire in adjacent iterations.

    Args:
        current (_WireAccess): Access evaluated at the current loop value.
        following (_WireAccess): Access evaluated at the next loop value.
        axis (_LoopAxis): Range defining that signed next value.

    Returns:
        bool: Whether ``current(i)`` equals ``following(i + step)`` for every
        valid adjacent iteration pair.
    """
    if current.owner != following.owner:
        return False
    if current.index is None or following.index is None:
        return current.index is None and following.index is None
    if current.affine is None or following.affine is None:
        return False
    shifted = cast(
        ResourceExpr,
        following.index.subs(
            axis.symbol,
            axis.symbol + axis.step,
            simultaneous=True,
        ),
    )
    return _safe_simplify(cast(ResourceExpr, current.index - shifted)) == _ZERO


def _first_iteration_gate_frontier(
    gate_inputs: tuple[_WireAccess, _WireAccess],
    axis: _LoopAxis,
) -> frozenset[WireKey] | None:
    """Project one serial gate's inputs at the first executed iteration.

    Args:
        gate_inputs (tuple[_WireAccess, _WireAccess]): Exact physical inputs
            of the address-preserving two-qubit gate in one loop iteration.
        axis (_LoopAxis): Range whose first value selects the frontier gate.

    Returns:
        frozenset[WireKey] | None: Two caller-visible scalar frontier keys, or
            ``None`` when either address remains loop-local or non-scalar.
    """
    frontier: set[WireKey] = set()
    for access in gate_inputs:
        if access.index is None:
            frontier.add((access.owner, None))
            continue
        first_index = _safe_simplify(
            cast(
                ResourceExpr,
                access.index.subs(
                    axis.symbol,
                    axis.start,
                    simultaneous=True,
                ),
            )
        )
        if axis.symbol in first_index.free_symbols:
            return None
        frontier.add((access.owner, _normalize_wire_index(first_index)))
    return frozenset(frontier)


def _prove_serial_chain_schedule(
    summary: _AccessSummary,
    body_depth: DepthResources,
    sequential_depth: DepthResources,
    axis: _LoopAxis,
) -> _SymbolicLoopSchedule | None:
    """Prove exact sequential depth from a consecutive-iteration dependency.

    Args:
        summary (_AccessSummary): Normalized flat body accesses.
        body_depth (DepthResources): One-iteration depth profile.
        sequential_depth (DepthResources): Existing exact scalar sum over the
            loop trip count.
        axis (_LoopAxis): Flat range axis.

    Returns:
        _SymbolicLoopSchedule | None: Serial-chain certificate, or ``None``
        when consecutive gates are not universally dependency-linked.
    """
    gate_inputs = summary.gate_inputs
    if gate_inputs is None or not _depth_is_one_gate_per_active_field(body_depth):
        return None
    if not any(
        _accesses_match_across_next_iteration(current, following, axis)
        for current in gate_inputs
        for following in gate_inputs
    ):
        return None
    active = _boolean_condition(sp.Gt(axis.iterations, _ONE))
    frontier = _first_iteration_gate_frontier(gate_inputs, axis)
    return _SymbolicLoopSchedule(
        depth=sequential_depth,
        completion_mode=_LoopCompletionMode.KEEP_SEQUENTIAL,
        synchronized_entry_conditions=_entry_conditions(
            _project_accesses(summary.accesses, axis),
            active,
        ),
        serial_frontier=frontier,
        serial_active_when=active,
    )


def _prove_triangular_schedule(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    outer: _LoopAxis,
    *,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> _SymbolicLoopSchedule | None:
    """Prove the critical path of a perfect triangular affine pair loop.

    Args:
        operation (ForOperation): Candidate outer range loop.
        resolver (ExprResolver): Resolver scoped to the outer body.
        body_depth (DepthResources): One outer iteration's depth profile.
        outer (_LoopAxis): Outer range axis.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases. Defaults to ``None``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        _SymbolicLoopSchedule | None: Triangular schedule certificate, or
        ``None`` when any domain, access, or depth premise fails.
    """
    if _safe_simplify(cast(ResourceExpr, outer.step - _ONE)) != _ZERO:
        return None
    nested = [item for item in operation.operations if isinstance(item, ForOperation)]
    if len(nested) != 1 or any(
        item is not nested[0] and not isinstance(item, BinOp)
        for item in operation.operations
    ):
        return None
    inner = nested[0]
    if (
        inner.loop_var_value is None
        or inner.region_args
        or inner.loop_carried_rebinds
        or len(inner.operands) not in (2, 3)
    ):
        return None
    inner_start = resolver.resolve(inner.operands[0])
    inner_stop = resolver.resolve(inner.operands[1])
    inner_step = (
        resolver.resolve(inner.operands[2]) if len(inner.operands) == 3 else _ONE
    )
    if any(
        _safe_simplify(cast(ResourceExpr, difference)) != _ZERO
        for difference in (
            inner_start - (outer.symbol + _ONE),
            inner_stop - outer.stop,
            inner_step - _ONE,
        )
    ):
        return None
    inner_symbol = sp.Dummy(
        "affine_inner",
        integer=True,
        nonnegative=True,
    )
    inner_resolver = resolver.child_scope(
        inner_block=_LocalBlock(inner.operations),
        extra_context={inner.loop_var_value.uuid: inner_symbol},
        extra_loop_vars={inner.loop_var: inner_symbol},
    )
    summary = _analyze_accesses(
        cast(Sequence[Operation], inner.operations),
        inner_resolver,
        symbols=(outer.symbol, inner_symbol),
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    gate_inputs = None if summary is None else summary.gate_inputs
    if gate_inputs is None or gate_inputs[0].owner != gate_inputs[1].owner:
        return None
    oriented: tuple[_WireAccess, _WireAccess] | None = None
    for outer_access, inner_access in (gate_inputs, tuple(reversed(gate_inputs))):
        outer_affine = outer_access.affine
        inner_affine = inner_access.affine
        if outer_affine is None or inner_affine is None:
            continue
        outer_coefficients = outer_affine.coefficients
        inner_coefficients = inner_affine.coefficients
        if (
            outer_coefficients[0].is_zero is not False
            or outer_coefficients[1] != _ZERO
            or inner_coefficients[0] != _ZERO
            or _safe_simplify(
                cast(ResourceExpr, outer_coefficients[0] - inner_coefficients[1])
            )
            != _ZERO
            or _safe_simplify(
                cast(ResourceExpr, outer_affine.residual - inner_affine.residual)
            )
            != _ZERO
        ):
            continue
        oriented = outer_access, inner_access
        break
    if oriented is None:
        return None
    inner_iterations = cast(ResourceExpr, sp.Max(_ZERO, inner_stop - inner_start))
    fields: dict[str, ResourceExpr] = {}
    active_fields = 0
    for field in dataclasses.fields(DepthResources):
        value = cast(ResourceExpr, getattr(body_depth, field.name))
        if value == _ZERO:
            fields[field.name] = _ZERO
            continue
        if _safe_simplify(cast(ResourceExpr, value - inner_iterations)) != _ZERO:
            return None
        active_fields += 1
        fields[field.name] = cast(
            ResourceExpr,
            sp.Max(_ZERO, 2 * outer.iterations - 3),
        )
    if active_fields == 0:
        return None
    outer_access, _inner_access = oriented
    if outer_access.index is None:
        return None
    entry_key: WireKey = (
        outer_access.owner,
        _symbolic_wire_range_index(
            outer_access.index,
            outer.symbol,
            start=outer.start,
            step=outer.step,
            iterations=outer.iterations,
        ),
    )
    active = _boolean_condition(sp.Gt(outer.iterations, 2))
    return _SymbolicLoopSchedule(
        depth=DepthResources(**fields),
        completion_mode=_LoopCompletionMode.AGGREGATE_SAFE,
        synchronized_entry_conditions=_entry_conditions(
            frozenset({entry_key}),
            active,
        ),
    )


def _symbolic_affine_loop_schedule(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    sequential_depth: DepthResources,
    *,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> _SymbolicLoopSchedule | None:
    """Prove one supported symbolic affine loop schedule.

    Args:
        operation (ForOperation): Candidate range loop.
        resolver (ExprResolver): Resolver scoped to the loop body.
        body_depth (DepthResources): One-iteration depth profile.
        sequential_depth (DepthResources): Existing scalar range sum.
        loop_symbol (sp.Symbol): Internal induction symbol.
        start (ResourceExpr): Inclusive range start.
        stop (ResourceExpr): Exclusive range stop.
        step (ResourceExpr): Signed range step.
        iterations (ResourceExpr): Exact trip count.
        allocated_qubits (ResourceExpr): Body-local allocation demand.
        clean_ancillas (ResourceExpr): Body clean-ancilla demand.
        dirty_ancillas (ResourceExpr): Body dirty-ancilla demand.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases. Defaults to ``None``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        _SymbolicLoopSchedule | None: Exact schedule certificate, or ``None``
        when the loop remains on the conservative general path.
    """
    if len(operation.operands) not in (2, 3) or any(
        demand != _ZERO for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
    ):
        return None
    axis = _LoopAxis(
        symbol=loop_symbol,
        start=start,
        stop=stop,
        step=step,
        iterations=iterations,
    )
    has_region_carry = bool(operation.region_args or operation.loop_carried_rebinds)
    if not has_region_carry:
        triangular = _prove_triangular_schedule(
            operation,
            resolver,
            body_depth,
            axis,
            owner_aliases=owner_aliases,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        if triangular is not None:
            return triangular
    if any(isinstance(item, HasNestedOps) for item in operation.operations):
        allowed_nested = all(
            not isinstance(item, HasNestedOps) or isinstance(item, IfOperation)
            for item in operation.operations
        )
        if not allowed_nested:
            return None
    summary = _analyze_accesses(
        cast(Sequence[Operation], operation.operations),
        resolver,
        symbols=(loop_symbol,),
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if summary is None:
        return None
    disjoint = _prove_disjoint_schedule(summary, body_depth, axis)
    if disjoint is not None:
        return disjoint
    if has_region_carry:
        return None
    return _prove_serial_chain_schedule(
        summary,
        body_depth,
        sequential_depth,
        axis,
    )
