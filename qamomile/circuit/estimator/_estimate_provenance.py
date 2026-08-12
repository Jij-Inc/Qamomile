"""Normalize and compose resource-estimate provenance metadata."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING

import sympy as sp

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._estimate_domain import (
    _domain_assumptions,
    _validate_domain_rewrite_metrics,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
    _combine_approximation,
    _combine_derivation,
    _combine_quality,
)
from qamomile.circuit.estimator._resource_expressions import _boolean_condition
from qamomile.circuit.estimator._resource_types import (
    GateResources,
    ResourceAssumption,
    _active_approximation,
    _active_assumptions_with_quality_reasons,
    _active_derivation,
    _active_quality,
    _guarded_quality_with_reason,
    _GuardedApproximation,
    _GuardedAssumption,
    _GuardedDerivation,
    _GuardedQuality,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry
from qamomile.circuit.estimator._symbol_discovery import (
    _collect_parameters,
    _serialization_registry,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


_DEFER_RESOURCE_SYMBOL_METADATA = ContextVar(
    "qamomile_defer_resource_symbol_metadata",
    default=False,
)


def _quality_fallback_reason(quality: EstimateQuality) -> ResourceAssumption:
    """Return an honest generic reason for a non-exact quality.

    Args:
        quality (EstimateQuality): Non-exact quality lacking a more specific
            explanation.

    Returns:
        ResourceAssumption: Generic reason suitable for public reporting.

    Raises:
        ValueError: If ``quality`` is exact.
    """
    if quality is EstimateQuality.CONSERVATIVE:
        message = (
            "resource counts include a conservative estimate without a more "
            "specific reason"
        )
    elif quality is EstimateQuality.UNKNOWN:
        message = (
            "resource counts include an estimate of unknown quality without a "
            "more specific reason"
        )
    else:
        raise ValueError("exact resource quality does not require a reason")
    return ResourceAssumption(message, source="resource estimate quality")


def _quality_reason_from_metadata(
    quality: EstimateQuality,
    assumptions: Sequence[ResourceAssumption],
    explicit_reason: ResourceAssumption | None = None,
) -> ResourceAssumption:
    """Select a reason supplied with one non-exact quality update.

    Args:
        quality (EstimateQuality): Non-exact quality being recorded.
        assumptions (Sequence[ResourceAssumption]): Assumptions supplied by
            the same metadata update.
        explicit_reason (ResourceAssumption | None): Preferred explicit
            reason. Defaults to ``None``.

    Returns:
        ResourceAssumption: Explicit, simultaneous, or generic quality reason.

    Raises:
        TypeError: If an explicit reason is not a resource assumption or its
            message is not a string.
        ValueError: If ``quality`` is exact or the explicit reason is blank.
    """
    if quality is EstimateQuality.EXACT:
        raise ValueError("exact resource quality does not require a reason")
    if explicit_reason is not None:
        if not isinstance(explicit_reason, ResourceAssumption):
            raise TypeError("resource quality reason must be an assumption")
        if not isinstance(explicit_reason.message, str):
            raise TypeError("resource quality reason message must be a string")
        if not explicit_reason.message.strip():
            raise ValueError("resource quality reason must not be blank")
        return explicit_reason
    for assumption in assumptions:
        if (
            isinstance(assumption, ResourceAssumption)
            and isinstance(assumption.message, str)
            and assumption.message.strip()
        ):
            return assumption
    return _quality_fallback_reason(quality)


def _newly_supplied_assumptions(
    supplied: Sequence[ResourceAssumption],
    rendered: Sequence[ResourceAssumption],
) -> tuple[ResourceAssumption, ...]:
    """Separate assumptions appended to already rendered metadata.

    Identity-preserving prefix detection distinguishes an intentionally
    appended duplicate from the same reason already rendered by an existing
    quality fact. Non-prefix edits retain the historical additive behavior:
    values absent from canonical guarded provenance are appended as ordinary
    assumptions, while existing facts are never removed implicitly.

    Args:
        supplied (Sequence[ResourceAssumption]): Public assumptions entering
            estimate normalization.
        rendered (Sequence[ResourceAssumption]): Canonical assumptions already
            rendered by retained guarded provenance.

    Returns:
        tuple[ResourceAssumption, ...]: Newly supplied ordinary assumptions.
    """
    if len(supplied) >= len(rendered) and all(
        assumption is carried
        for assumption, carried in zip(supplied, rendered, strict=False)
    ):
        return tuple(supplied[len(rendered) :])
    return tuple(assumption for assumption in supplied if assumption not in rendered)


def _initialize_estimate_provenance(
    estimate: ResourceEstimate,
) -> None:
    """Normalize guarded metadata and derive public parameter metadata.

    Args:
        estimate (ResourceEstimate): Newly initialized estimate to normalize.

    Raises:
        RuntimeError: If retained input-domain rewrite evidence disagrees with
            public metrics, rendered assumptions, parameters, or symbol
            aliases.
    """
    _validate_domain_rewrite_metrics(estimate)
    estimate._constraints = tuple(
        constraint
        for constraint in estimate._constraints
        if constraint.active_when is not sp.false
    )
    estimate._global_barrier_condition = _boolean_condition(
        estimate._global_barrier_condition
    )
    prior_rendered = estimate._rendered_assumption_snapshot
    new_assumptions: tuple[ResourceAssumption, ...]
    if estimate._guarded_assumptions is None:
        new_assumptions = tuple(estimate.assumptions)
        estimate._guarded_assumptions = tuple(
            _GuardedAssumption(sp.true, assumption) for assumption in new_assumptions
        )
    elif prior_rendered is not None:
        if len(estimate.assumptions) < len(prior_rendered) or any(
            assumption is not carried
            for assumption, carried in zip(
                estimate.assumptions,
                prior_rendered,
                strict=False,
            )
        ):
            raise RuntimeError(
                "resource estimate assumptions no longer preserve the rendered "
                "domain snapshot prefix; append ordinary assumptions instead of "
                "inserting, removing, or reordering entries"
            )
        new_assumptions = tuple(estimate.assumptions[len(prior_rendered) :])
        estimate._guarded_assumptions = (
            *estimate._guarded_assumptions,
            *(
                _GuardedAssumption(sp.true, assumption)
                for assumption in new_assumptions
            ),
        )
    else:
        rendered_assumptions = _active_assumptions_with_quality_reasons(
            estimate._guarded_assumptions,
            estimate._guarded_qualities or (),
        )
        new_assumptions = _newly_supplied_assumptions(
            estimate.assumptions,
            rendered_assumptions,
        )
        estimate._guarded_assumptions = (
            *estimate._guarded_assumptions,
            *(
                _GuardedAssumption(sp.true, assumption)
                for assumption in new_assumptions
            ),
        )
    if estimate._guarded_derivations is None:
        estimate._guarded_derivations = (
            (_GuardedDerivation(sp.true, estimate.derivation),)
            if estimate.derivation is not EstimateDerivation.STRUCTURAL
            else ()
        )
    elif _combine_derivation(
        _active_derivation(estimate._guarded_derivations),
        estimate.derivation,
    ) is estimate.derivation and estimate.derivation is not _active_derivation(
        estimate._guarded_derivations
    ):
        estimate._guarded_derivations = (
            *estimate._guarded_derivations,
            _GuardedDerivation(sp.true, estimate.derivation),
        )
    estimate.derivation = _active_derivation(estimate._guarded_derivations)
    if estimate._guarded_qualities is None:
        estimate._guarded_qualities = (
            (
                _guarded_quality_with_reason(
                    sp.true,
                    estimate.quality,
                    _quality_reason_from_metadata(
                        estimate.quality,
                        new_assumptions,
                    ),
                ),
            )
            if estimate.quality is not EstimateQuality.EXACT
            else ()
        )
    elif _combine_quality(
        _active_quality(estimate._guarded_qualities),
        estimate.quality,
    ) is estimate.quality and estimate.quality is not _active_quality(
        estimate._guarded_qualities
    ):
        estimate._guarded_qualities = (
            *estimate._guarded_qualities,
            _guarded_quality_with_reason(
                sp.true,
                estimate.quality,
                _quality_reason_from_metadata(
                    estimate.quality,
                    new_assumptions,
                ),
            ),
        )
    estimate.quality = _active_quality(estimate._guarded_qualities)
    estimate.assumptions = _active_assumptions_with_quality_reasons(
        estimate._guarded_assumptions,
        estimate._guarded_qualities,
    )
    estimate._rendered_assumption_snapshot = (
        estimate.assumptions if estimate._domain_rewrite_state is not None else None
    )
    if estimate._guarded_approximations is None:
        estimate._guarded_approximations = (
            (_GuardedApproximation(sp.true, estimate.approximation),)
            if estimate.approximation is not ApproximationStatus.EXACT
            else ()
        )
    elif _combine_approximation(
        _active_approximation(estimate._guarded_approximations),
        estimate.approximation,
    ) is estimate.approximation and estimate.approximation is not _active_approximation(
        estimate._guarded_approximations
    ):
        estimate._guarded_approximations = (
            *estimate._guarded_approximations,
            _GuardedApproximation(sp.true, estimate.approximation),
        )
    estimate.approximation = _active_approximation(estimate._guarded_approximations)
    if not _DEFER_RESOURCE_SYMBOL_METADATA.get():
        _refresh_symbol_metadata(estimate)


def _refresh_symbol_metadata(
    estimate: ResourceEstimate,
    registry: SymbolRegistry | None = None,
) -> SymbolRegistry:
    """Derive stable public aliases and the parameter map.

    Args:
        estimate (ResourceEstimate): Estimate whose symbols should be refreshed.
        registry (SymbolRegistry | None): Precomputed registry for this exact
            estimate. Defaults to rebuilding one from all resource expressions
            and structural requirements.
    Returns:
        SymbolRegistry: Registry used to refresh the public metadata.
    """
    active_registry = registry or _serialization_registry(estimate)
    estimate._symbol_aliases = active_registry.aliases()
    estimate.parameters = _collect_parameters(estimate, active_registry)
    rendered = _active_assumptions_with_quality_reasons(
        estimate._guarded_assumptions or (),
        estimate._guarded_qualities or (),
    )
    estimate.assumptions = (*rendered, *_domain_assumptions(estimate, active_registry))
    estimate._rendered_assumption_snapshot = (
        estimate.assumptions if estimate._domain_rewrite_state is not None else None
    )
    return active_registry


def _with_estimate_metadata(
    estimate: ResourceEstimate,
    *,
    assumptions: Sequence[ResourceAssumption] = (),
    derivation: EstimateDerivation = EstimateDerivation.STRUCTURAL,
    quality: EstimateQuality = EstimateQuality.EXACT,
    quality_reason: ResourceAssumption | None = None,
    approximation: ApproximationStatus = ApproximationStatus.EXACT,
    active_when: sp.Basic = sp.true,
) -> ResourceEstimate:
    """Append guarded assumption, derivation, quality, and approximation.

    Args:
        estimate (ResourceEstimate): Estimate receiving the metadata.
        assumptions (Sequence[ResourceAssumption]): Assumptions to append.
            Defaults to none.
        derivation (EstimateDerivation): Derivation fact to append.
            ``STRUCTURAL`` adds no fact. Defaults to ``STRUCTURAL``.
        quality (EstimateQuality): Count quality to append. ``EXACT`` adds no
            fact. Defaults to ``EXACT``.
        quality_reason (ResourceAssumption | None): Explanation for a non-exact
            quality. Defaults to the first simultaneous nonblank assumption,
            then to an honest generic reason.
        approximation (ApproximationStatus): Mathematical approximation fact
            to append. ``EXACT`` adds no fact. Defaults to ``EXACT``.
        active_when (sp.Basic): Activation condition shared by the new facts.
            Defaults to true.

    Returns:
        ResourceEstimate: Copy with condition-aware metadata appended.

    Raises:
        ValueError: If an exact update supplies a quality reason, or an
            explicit quality reason has a blank message.
        TypeError: If an explicit quality reason is not a resource assumption.
    """
    condition = _boolean_condition(active_when)
    if quality is EstimateQuality.EXACT and quality_reason is not None:
        raise ValueError("exact resource quality does not accept a reason")
    guarded_quality_updates: tuple[_GuardedQuality, ...] = ()
    if quality is not EstimateQuality.EXACT:
        selected_quality_reason = _quality_reason_from_metadata(
            quality,
            assumptions,
            explicit_reason=quality_reason,
        )
        guarded_quality_updates = (
            _guarded_quality_with_reason(
                condition,
                quality,
                selected_quality_reason,
            ),
        )
    guarded_assumptions = estimate._guarded_assumptions or ()
    guarded_derivations = estimate._guarded_derivations or ()
    guarded_qualities = estimate._guarded_qualities or ()
    guarded_approximations = estimate._guarded_approximations or ()
    return dataclasses.replace(
        estimate,
        _guarded_assumptions=(
            *guarded_assumptions,
            *(_GuardedAssumption(condition, assumption) for assumption in assumptions),
        ),
        _guarded_derivations=(
            *guarded_derivations,
            *(
                (_GuardedDerivation(condition, derivation),)
                if derivation is not EstimateDerivation.STRUCTURAL
                else ()
            ),
        ),
        _guarded_qualities=(
            *guarded_qualities,
            *guarded_quality_updates,
        ),
        _guarded_approximations=(
            *guarded_approximations,
            *(
                (_GuardedApproximation(condition, approximation),)
                if approximation is not ApproximationStatus.EXACT
                else ()
            ),
        ),
    )


def _estimate_has_control_sensitive_resources(estimate: ResourceEstimate) -> bool:
    """Return whether an estimate contains control-model-dependent metrics.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        bool: Whether gates, decomposition ancillas, or gate depth are not
            structurally zero. Symbolic and unevaluated expressions are treated
            as control-sensitive conservatively.
    """
    expressions = [
        *(
            getattr(estimate.gates, field.name)
            for field in dataclasses.fields(GateResources)
        ),
        estimate.depth.gate_depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
    ]
    has_new_nonunitary_profile = any(
        expression != _ZERO
        for expression in (
            estimate.measurements.total,
            estimate.resets.total,
            estimate.depth.reset_depth,
        )
    )
    if not has_new_nonunitary_profile:
        expressions.append(estimate.depth.depth - estimate.depth.measurement_depth)
    return any(expression != _ZERO for expression in expressions)


def _merge_estimate_provenance(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> ControlDecomposition:
    """Merge compatible control-model provenance for resource algebra.

    Args:
        left (ResourceEstimate): Left operand.
        right (ResourceEstimate): Right operand.

    Returns:
        ControlDecomposition: Control decomposition for the result.

    Raises:
        ValueError: If control-model-sensitive estimates use incompatible
            provenance.
    """
    left_sensitive = _estimate_has_control_sensitive_resources(left)
    right_sensitive = _estimate_has_control_sensitive_resources(right)
    if left_sensitive and right_sensitive:
        if left.control_decomposition is not right.control_decomposition:
            raise ValueError(
                "Cannot compose resource estimates from different control "
                "decompositions: "
                f"{left.control_decomposition.value!r} and "
                f"{right.control_decomposition.value!r}."
            )
        return left.control_decomposition
    if left_sensitive:
        return left.control_decomposition
    if right_sensitive:
        return right.control_decomposition
    return left.control_decomposition


def _merge_symbol_aliases(
    *estimates: ResourceEstimate,
) -> dict[sp.Symbol, str]:
    """Merge stable public symbol aliases in operand order.

    Args:
        *estimates (ResourceEstimate): Estimates whose preferred aliases are
            merged from left to right.

    Returns:
        dict[sp.Symbol, str]: Left-biased, collision-free preferred aliases.
    """
    merged: dict[sp.Symbol, str] = {}
    claimed_aliases: set[str] = set()
    for estimate in estimates:
        for symbol, alias in estimate._symbol_aliases.items():
            if symbol in merged or alias in claimed_aliases:
                continue
            merged[symbol] = alias
            claimed_aliases.add(alias)
    return merged
