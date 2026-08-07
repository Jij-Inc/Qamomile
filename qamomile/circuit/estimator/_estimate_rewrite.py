"""Rewrite symbolic expressions carried by resource estimates."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import TYPE_CHECKING, Any

import sympy as sp

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._dependency_metadata import (
    _map_dependency_completion,
    _map_dependency_keys,
)
from qamomile.circuit.estimator._resource_base import _is_concrete_integer
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _safe_constraint_substitute,
    _substitute_resource_expr,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    WidthResources,
)
from qamomile.circuit.estimator._symbolic import (
    _canonicalize_concrete_integer,
    _normalize_resource_scalar,
    _simplify_public_resource_expression,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _substitute_estimate(
    estimate: ResourceEstimate,
    values: dict[str, object],
) -> ResourceEstimate:
    """Substitute concrete values for public symbolic parameters.

    Args:
        estimate (ResourceEstimate): Estimate to specialize.
        values (dict[str, object]): Concrete values keyed by public parameter
            alias.
    Returns:
        ResourceEstimate: Estimate with substituted and simplified expressions.

    Raises:
        ValueError: If a name is not a parameter, or a supplied value violates
            an integer or nonnegative parameter domain.
        TypeError: If a supplied value is not a concrete numeric scalar.
    """
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for name, value in values.items():
        parameter = estimate.parameters.get(name)
        if parameter is None:
            available = ", ".join(estimate.parameters) or "(none)"
            raise ValueError(
                f"Unknown resource parameter '{name}'; available: {available}."
            )
        replacement = _normalize_resource_scalar(
            value,
            label=f"Resource parameter '{name}'",
            allow_symbolic=False,
            allow_bool=False,
        )
        if (
            replacement.is_number
            and parameter.is_integer is True
            and not _is_concrete_integer(replacement)
        ):
            raise ValueError(
                f"Cannot substitute non-integer value {value!r} for "
                f"integer resource parameter '{name}'."
            )
        if parameter.is_integer is True:
            replacement = _canonicalize_concrete_integer(replacement)
        if replacement.is_negative is True and parameter.is_nonnegative is True:
            raise ValueError(
                f"Cannot substitute negative value {value!r} for "
                f"nonnegative resource parameter '{name}'."
            )
        substitutions[parameter] = replacement
    substitute_expression = partial(
        _substitute_resource_expr,
        substitutions=substitutions,
    )
    mapped = _map_estimate_expressions(
        estimate,
        substitute_expression,
        constraint_fn=lambda expression: _safe_constraint_substitute(
            expression,
            substitutions,
        ),
        guard_fn=lambda expression: _safe_constraint_substitute(
            expression,
            substitutions,
        ),
        dependency_fn=substitute_expression,
    )
    # Substitution often collapses a previously large branch expression to a
    # small Piecewise/Min/Max form. Re-run the bounded public simplifier so
    # post-hoc specialization has the same canonical shape as direct input
    # specialization while respecting the global simplification node budget.
    return mapped.simplify()


def _simplify_estimate(estimate: ResourceEstimate) -> ResourceEstimate:
    """Simplify every public symbolic resource expression.

    Args:
        estimate (ResourceEstimate): Estimate to simplify.

    Returns:
        ResourceEstimate: Simplified estimate.
    """
    return _map_estimate_expressions(
        estimate,
        _simplify_public_resource_expression,
        # Guards are normalized while they are composed. Re-running SymPy's
        # global Boolean simplifier here is disproportionately expensive for
        # loop-derived predicates and can expose bound variables.
        guard_fn=lambda expression: expression,
        # Completion depths are private scheduling state already consumed at
        # call boundaries. Substitution still rewrites them, but root-level
        # simplification need not traverse the larger private expressions.
        dependency_fn=lambda expression: expression,
    )


def _map_estimate_expressions(
    estimate: ResourceEstimate,
    fn: Any,
    *,
    constraint_fn: Any | None = None,
    guard_fn: Any | None = None,
    dependency_fn: Any | None = None,
) -> ResourceEstimate:
    """Apply one rewrite consistently to all estimate expressions.

    Args:
        estimate (ResourceEstimate): Estimate whose expressions are rewritten.
        fn (Any): Callable rewriting public numeric resource expressions.
        constraint_fn (Any | None): Optional non-clamping rewrite for
            structural constraints. Defaults to ``fn``.
        guard_fn (Any | None): Optional rewrite for guarded provenance
            predicates. Defaults to ``constraint_fn``.
        dependency_fn (Any | None): Optional rewrite for private wire indices
            and completion depths. Defaults to ``constraint_fn``.

    Returns:
        ResourceEstimate: Rewritten estimate.
    """
    rewrite_constraint = fn if constraint_fn is None else constraint_fn
    rewrite_guard = rewrite_constraint if guard_fn is None else guard_fn
    rewrite_dependency = rewrite_constraint if dependency_fn is None else dependency_fn
    mapped_constraints = tuple(
        mapped
        for constraint in estimate._constraints
        if (mapped := constraint.mapped(rewrite_constraint)).active_when is not sp.false
    )
    mapped_assumptions = tuple(
        mapped
        for fact in (estimate._guarded_assumptions or ())
        if (mapped := fact.mapped(rewrite_guard)) is not None
    )
    mapped_derivations = tuple(
        mapped
        for fact in (estimate._guarded_derivations or ())
        if (mapped := fact.mapped(rewrite_guard)) is not None
    )
    mapped_qualities = tuple(
        mapped
        for fact in (estimate._guarded_qualities or ())
        if (mapped := fact.mapped(rewrite_guard)) is not None
    )
    mapped_approximations = tuple(
        mapped
        for fact in (estimate._guarded_approximations or ())
        if (mapped := fact.mapped(rewrite_guard)) is not None
    )
    mapped_calls = CallResources(
        calls_by_name={
            name: mapped
            for name, value in estimate.calls.calls_by_name.items()
            if (mapped := fn(value)) != _ZERO
        },
        queries_by_name={
            name: mapped
            for name, value in estimate.calls.queries_by_name.items()
            if (mapped := fn(value)) != _ZERO
        },
    )
    return dataclasses.replace(
        estimate.zero(),
        width=WidthResources(
            input_qubits=fn(estimate.width.input_qubits),
            allocated_qubits=fn(estimate.width.allocated_qubits),
            clean_ancilla_qubits=fn(estimate.width.clean_ancilla_qubits),
            dirty_ancilla_qubits=fn(estimate.width.dirty_ancilla_qubits),
            peak_qubits=fn(estimate.width.peak_qubits),
        ),
        gates=GateResources(
            total=fn(estimate.gates.total),
            single_qubit=fn(estimate.gates.single_qubit),
            two_qubit=fn(estimate.gates.two_qubit),
            multi_qubit=fn(estimate.gates.multi_qubit),
            clifford=fn(estimate.gates.clifford),
            rotation=fn(estimate.gates.rotation),
            t=fn(estimate.gates.t),
            toffoli=fn(estimate.gates.toffoli),
            non_clifford=fn(estimate.gates.non_clifford),
        ),
        measurements=MeasurementResources(total=fn(estimate.measurements.total)),
        resets=ResetResources(total=fn(estimate.resets.total)),
        depth=DepthResources(
            depth=fn(estimate.depth.depth),
            clifford_depth=fn(estimate.depth.clifford_depth),
            rotation_depth=fn(estimate.depth.rotation_depth),
            t_depth=fn(estimate.depth.t_depth),
            toffoli_depth=fn(estimate.depth.toffoli_depth),
            non_clifford_depth=fn(estimate.depth.non_clifford_depth),
            measurement_depth=fn(estimate.depth.measurement_depth),
            gate_depth=fn(estimate.depth.gate_depth),
            reset_depth=fn(estimate.depth.reset_depth),
        ),
        calls=mapped_calls,
        trace=(
            estimate.trace.mapped(rewrite_constraint)
            if estimate.trace is not None
            else None
        ),
        control_decomposition=estimate.control_decomposition,
        _allocation_sites={
            site: fn(size) for site, size in estimate._allocation_sites.items()
        },
        _constraints=mapped_constraints,
        _output_sizes={
            owner: fn(size) for owner, size in estimate._output_sizes.items()
        },
        _input_sizes={owner: fn(size) for owner, size in estimate._input_sizes.items()},
        _has_output_summary=estimate._has_output_summary,
        _dependency_keys=_map_dependency_keys(
            estimate._dependency_keys,
            rewrite_dependency,
        ),
        _dependency_reads=_map_dependency_keys(
            estimate._dependency_reads,
            rewrite_dependency,
        ),
        _dependency_writes=_map_dependency_keys(
            estimate._dependency_writes,
            rewrite_dependency,
        ),
        _dependency_completion=_map_dependency_completion(
            estimate._dependency_completion,
            rewrite_dependency,
        ),
        _dependency_completion_uniform=estimate._dependency_completion_uniform,
        _global_barrier_condition=_boolean_condition(
            rewrite_guard(estimate._global_barrier_condition)
        ),
        _measurement_taint_conditions={
            uuid: _boolean_condition(rewrite_guard(condition))
            for uuid, condition in estimate._measurement_taint_conditions.items()
        },
        _guarded_assumptions=mapped_assumptions,
        _guarded_derivations=mapped_derivations,
        _guarded_qualities=mapped_qualities,
        _guarded_approximations=mapped_approximations,
        _symbol_aliases=estimate._symbol_aliases,
    )


__all__: list[str] = []
