"""Manage input-domain rewrites of public resource metrics."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator._parameter_domain import (
    _ConsumedDomainRequirement,
    _DomainRewritePolicy,
    _prepare_domain_proof_environment,
    _rewrite_expression_in_domain,
)
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_constraints import _ConstraintOrigin
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    WidthResources,
    _active_approximation,
    _active_assumptions,
    _active_derivation,
    _active_quality,
)
from qamomile.circuit.estimator._serialization import (
    SymbolRegistry,
    stringify_expression,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


@dataclasses.dataclass(frozen=True)
class _PublicResourceSnapshot:
    """Store immutable scalar copies of every public resource metric.

    Args:
        width (tuple[ResourceExpr, ...]): Width fields in dataclass order.
        gates (tuple[ResourceExpr, ...]): Gate fields in dataclass order.
        measurements (tuple[ResourceExpr, ...]): Measurement fields.
        resets (tuple[ResourceExpr, ...]): Reset fields.
        depth (tuple[ResourceExpr, ...]): Depth fields in dataclass order.
        calls (tuple[tuple[str, ResourceExpr], ...]): Named callable counts.
        queries (tuple[tuple[str, ResourceExpr], ...]): Named query counts.
    """

    width: tuple[ResourceExpr, ...]
    gates: tuple[ResourceExpr, ...]
    measurements: tuple[ResourceExpr, ...]
    resets: tuple[ResourceExpr, ...]
    depth: tuple[ResourceExpr, ...]
    calls: tuple[tuple[str, ResourceExpr], ...]
    queries: tuple[tuple[str, ResourceExpr], ...]

    @classmethod
    def capture(cls, estimate: ResourceEstimate) -> _PublicResourceSnapshot:
        """Copy all public scalar metrics from an estimate.

        Args:
            estimate (ResourceEstimate): Estimate whose metrics are copied.

        Returns:
            _PublicResourceSnapshot: Immutable scalar-by-scalar snapshot.
        """
        return cls(
            width=_resource_record_values(estimate.width),
            gates=_resource_record_values(estimate.gates),
            measurements=_resource_record_values(estimate.measurements),
            resets=_resource_record_values(estimate.resets),
            depth=_resource_record_values(estimate.depth),
            calls=tuple(sorted(estimate.calls.calls_by_name.items())),
            queries=tuple(sorted(estimate.calls.queries_by_name.items())),
        )

    def mapped(
        self,
        fn: Callable[[ResourceExpr], ResourceExpr],
    ) -> _PublicResourceSnapshot:
        """Rewrite every expression stored in this snapshot.

        Args:
            fn (Callable[[ResourceExpr], ResourceExpr]): Scalar rewrite.

        Returns:
            _PublicResourceSnapshot: Rewritten immutable snapshot.
        """
        return dataclasses.replace(
            self,
            width=tuple(fn(value) for value in self.width),
            gates=tuple(fn(value) for value in self.gates),
            measurements=tuple(fn(value) for value in self.measurements),
            resets=tuple(fn(value) for value in self.resets),
            depth=tuple(fn(value) for value in self.depth),
            calls=tuple((name, fn(value)) for name, value in self.calls),
            queries=tuple((name, fn(value)) for name, value in self.queries),
        )

    def expressions(self) -> tuple[ResourceExpr, ...]:
        """Return every stored expression in deterministic order.

        Returns:
            tuple[ResourceExpr, ...]: Snapshot expressions.
        """
        return (
            *self.width,
            *self.gates,
            *self.measurements,
            *self.resets,
            *self.depth,
            *(value for _name, value in self.calls),
            *(value for _name, value in self.queries),
        )

    def width_resources(self) -> WidthResources:
        """Materialize the mutable width record.

        Returns:
            WidthResources: Fresh width resources.
        """
        return cast(
            WidthResources, _resource_record_from_values(WidthResources, self.width)
        )

    def gate_resources(self) -> GateResources:
        """Materialize the mutable gate record.

        Returns:
            GateResources: Fresh gate resources.
        """
        return cast(
            GateResources, _resource_record_from_values(GateResources, self.gates)
        )

    def measurement_resources(self) -> MeasurementResources:
        """Materialize the mutable measurement record.

        Returns:
            MeasurementResources: Fresh measurement resources.
        """
        return cast(
            MeasurementResources,
            _resource_record_from_values(MeasurementResources, self.measurements),
        )

    def reset_resources(self) -> ResetResources:
        """Materialize the mutable reset record.

        Returns:
            ResetResources: Fresh reset resources.
        """
        return cast(
            ResetResources, _resource_record_from_values(ResetResources, self.resets)
        )

    def depth_resources(self) -> DepthResources:
        """Materialize the mutable depth record.

        Returns:
            DepthResources: Fresh depth resources.
        """
        return cast(
            DepthResources, _resource_record_from_values(DepthResources, self.depth)
        )

    def call_resources(self) -> CallResources:
        """Materialize the mutable callable record.

        Returns:
            CallResources: Fresh callable and query resources.
        """
        return CallResources(
            calls_by_name=dict(self.calls),
            queries_by_name=dict(self.queries),
        )


@dataclasses.dataclass(frozen=True)
class _DomainRewriteState:
    """Retain the original formula and proof used by a domain rewrite.

    Args:
        original (_PublicResourceSnapshot): Domain-independent public metrics.
        rewritten (_PublicResourceSnapshot): Visible rewritten public metrics.
        requirements (tuple[_ConsumedDomainRequirement, ...]): Exact domain
            facts consumed by at least one successful rewrite.
    """

    original: _PublicResourceSnapshot
    rewritten: _PublicResourceSnapshot
    requirements: tuple[_ConsumedDomainRequirement, ...]


def _resource_record_values(record: Any) -> tuple[ResourceExpr, ...]:
    """Copy dataclass scalar fields without retaining mutable record aliases.

    Args:
        record (Any): Public resource dataclass instance.

    Returns:
        tuple[ResourceExpr, ...]: Scalar values in dataclass field order.
    """
    return tuple(
        cast(ResourceExpr, getattr(record, field.name))
        for field in dataclasses.fields(record)
    )


def _resource_record_from_values(
    record_type: Any,
    values: tuple[ResourceExpr, ...],
) -> Any:
    """Recreate one public resource record from scalar values.

    Args:
        record_type (Any): Public resource dataclass type.
        values (tuple[ResourceExpr, ...]): Values in dataclass field order.

    Returns:
        Any: Fresh public resource record.

    Raises:
        RuntimeError: If the snapshot no longer matches the public dataclass
            field schema.
    """
    fields = dataclasses.fields(record_type)
    if len(fields) != len(values):
        raise RuntimeError("resource snapshot schema no longer matches public fields")
    return record_type(
        **{field.name: value for field, value in zip(fields, values, strict=True)}
    )


def _merge_domain_rewrite_policy(
    left: _DomainRewritePolicy,
    right: _DomainRewritePolicy,
) -> _DomainRewritePolicy:
    """Merge binary domain-rewrite policies conservatively.

    Args:
        left (_DomainRewritePolicy): Left operand policy.
        right (_DomainRewritePolicy): Right operand policy.

    Returns:
        _DomainRewritePolicy: Policy for the composed estimate.
    """
    if _DomainRewritePolicy.DISABLED in (left, right):
        return _DomainRewritePolicy.DISABLED
    if _DomainRewritePolicy.ENABLED in (left, right):
        return _DomainRewritePolicy.ENABLED
    return _DomainRewritePolicy.INHERITED


def _validate_domain_rewrite_metrics(estimate: ResourceEstimate) -> None:
    """Reject public metric mutations that invalidate retained evidence.

    Args:
        estimate (ResourceEstimate): Estimate whose visible metrics are checked.

    Raises:
        RuntimeError: If visible metrics no longer match the rewrite snapshot.
    """
    state = estimate._domain_rewrite_state
    if (
        state is not None
        and _PublicResourceSnapshot.capture(estimate) != state.rewritten
    ):
        raise RuntimeError(
            "resource estimate domain rewrite state does not match visible metrics; "
            "use dataclasses.replace() on an estimate without rewrite state"
        )


def _validate_domain_rewrite_state(
    estimate: ResourceEstimate,
    *,
    validate_deferred_symbols: bool = False,
) -> None:
    """Reject public mutations that invalidate retained provenance.

    Args:
        estimate (ResourceEstimate): Estimate to validate.
        validate_deferred_symbols (bool): Whether to validate public symbol
            metadata even while intermediate metadata refresh is deferred.
            Defaults to ``False``.

    Raises:
        RuntimeError: If visible metrics, public provenance metadata,
            parameters, or symbol aliases were mutated directly.
    """
    _validate_domain_rewrite_metrics(estimate)
    rendered = estimate._rendered_assumption_snapshot
    if rendered is not None and (
        len(rendered) != len(estimate.assumptions)
        or any(
            old is not new
            for old, new in zip(rendered, estimate.assumptions, strict=True)
        )
    ):
        raise RuntimeError(
            "resource estimate assumptions were mutated without synchronizing "
            "domain rewrite provenance"
        )
    _validate_public_provenance_metadata(estimate)
    if estimate._domain_rewrite_state is not None:
        _validate_domain_symbol_metadata(
            estimate,
            validate_deferred=validate_deferred_symbols,
        )


def _validate_public_provenance_metadata(estimate: ResourceEstimate) -> None:
    """Reject public metadata that disagrees with guarded provenance.

    Args:
        estimate (ResourceEstimate): Estimate whose visible provenance fields
            are checked.

    Raises:
        RuntimeError: If assumptions, derivation, quality, or approximation
            were mutated without rebuilding their guarded provenance.
    """
    if estimate._domain_rewrite_state is None and estimate.assumptions != (
        _active_assumptions(estimate._guarded_assumptions or ())
    ):
        raise RuntimeError(
            "resource estimate assumptions do not match guarded provenance; "
            "use dataclasses.replace() instead of mutating public metadata"
        )
    provenance_values = (
        (
            "derivation",
            estimate.derivation,
            _active_derivation(estimate._guarded_derivations or ()),
        ),
        (
            "quality",
            estimate.quality,
            _active_quality(estimate._guarded_qualities or ()),
        ),
        (
            "approximation",
            estimate.approximation,
            _active_approximation(estimate._guarded_approximations or ()),
        ),
    )
    for field_name, visible, guarded in provenance_values:
        if visible is not guarded:
            raise RuntimeError(
                f"resource estimate {field_name} does not match guarded "
                "provenance; use dataclasses.replace() instead of mutating "
                "public metadata"
            )


def _validate_domain_symbol_metadata(
    estimate: ResourceEstimate,
    *,
    validate_deferred: bool = False,
) -> None:
    """Reject public symbol metadata that disagrees with retained formulas.

    Args:
        estimate (ResourceEstimate): Domain-rewritten estimate to validate.
        validate_deferred (bool): Whether to validate while intermediate
            metadata refresh is deferred. Defaults to ``False``.

    Raises:
        RuntimeError: If public parameters or retained aliases disagree with
            the canonical symbols in the estimate.
    """
    from qamomile.circuit.estimator._estimate_provenance import (
        _DEFER_RESOURCE_SYMBOL_METADATA,
    )
    from qamomile.circuit.estimator._symbol_discovery import (
        _collect_parameters,
        _serialization_registry,
    )

    if _DEFER_RESOURCE_SYMBOL_METADATA.get() and not validate_deferred:
        return
    registry = _serialization_registry(estimate)
    expected_aliases = registry.aliases()
    expected_parameters = _collect_parameters(estimate, registry)
    if (
        estimate._symbol_aliases != expected_aliases
        or estimate.parameters != expected_parameters
    ):
        raise RuntimeError(
            "resource estimate parameter metadata does not match its symbolic "
            "input-domain rewrite state"
        )


def _restore_domain_rewrite(estimate: ResourceEstimate) -> ResourceEstimate:
    """Restore domain-independent public formulas before resource algebra.

    Args:
        estimate (ResourceEstimate): Estimate entering an algebra boundary.

    Returns:
        ResourceEstimate: Estimate with original formulas and no consumed proof.

    Raises:
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    _validate_domain_rewrite_state(estimate)
    state = estimate._domain_rewrite_state
    if state is None:
        return estimate
    ordinary_assumptions = tuple(
        fact.assumption
        for fact in (estimate._guarded_assumptions or ())
        if fact.active_when is not sp.false
    )
    return dataclasses.replace(
        estimate,
        width=state.original.width_resources(),
        gates=state.original.gate_resources(),
        measurements=state.original.measurement_resources(),
        resets=state.original.reset_resources(),
        depth=state.original.depth_resources(),
        calls=state.original.call_resources(),
        assumptions=ordinary_assumptions,
        _domain_rewrite_state=None,
        _rendered_assumption_snapshot=None,
    )


def _apply_domain_rewrite(
    estimate: ResourceEstimate,
    *,
    policy: _DomainRewritePolicy | None = None,
) -> ResourceEstimate:
    """Rewrite public metrics from eligible qkernel input-domain facts.

    Args:
        estimate (ResourceEstimate): Domain-independent estimate to rewrite.
        policy (_DomainRewritePolicy | None): Optional policy override.
            Defaults to the estimate's current policy.

    Returns:
        ResourceEstimate: Canonical rewritten estimate or unchanged formulas.

    Raises:
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    restored = _restore_domain_rewrite(estimate)
    active_policy = policy or restored._domain_rewrite_policy
    restored = dataclasses.replace(
        restored,
        _domain_rewrite_policy=active_policy,
        _domain_rewrite_state=None,
        _rendered_assumption_snapshot=None,
    )
    if active_policy is not _DomainRewritePolicy.ENABLED:
        return restored

    original = _PublicResourceSnapshot.capture(restored)
    environment = _prepare_domain_proof_environment(restored._constraints)
    if environment is None:
        return restored
    requirements: list[_ConsumedDomainRequirement] = []
    rewrite_cache: dict[
        sp.Expr,
        tuple[sp.Expr, tuple[_ConsumedDomainRequirement, ...]],
    ] = {}

    def rewrite(value: ResourceExpr) -> ResourceExpr:
        """Rewrite one public scalar and collect exact proof evidence.

        Args:
            value (ResourceExpr): Public scalar expression.

        Returns:
            ResourceExpr: Rewritten or unchanged expression.
        """
        expression = cast(sp.Expr, sp.sympify(value))
        cached = rewrite_cache.get(expression)
        if cached is None:
            cached = _rewrite_expression_in_domain(
                expression,
                environment,
            )
            rewrite_cache[expression] = cached
        rewritten, consumed = cached
        for requirement in consumed:
            if requirement not in requirements:
                requirements.append(requirement)
        return cast(ResourceExpr, rewritten)

    rewritten_snapshot = original.mapped(rewrite)
    if environment.exhausted:
        return restored
    if not requirements:
        return restored
    rewritten = dataclasses.replace(
        restored,
        width=rewritten_snapshot.width_resources(),
        gates=rewritten_snapshot.gate_resources(),
        measurements=rewritten_snapshot.measurement_resources(),
        resets=rewritten_snapshot.reset_resources(),
        depth=rewritten_snapshot.depth_resources(),
        calls=rewritten_snapshot.call_resources(),
        _domain_rewrite_state=_DomainRewriteState(
            original=original,
            rewritten=rewritten_snapshot,
            requirements=tuple(requirements),
        ),
        _rendered_assumption_snapshot=None,
    )
    return rewritten


def _demote_opaque_domain(estimate: ResourceEstimate) -> ResourceEstimate:
    """Remove root-qkernel domain trust at an opaque cost boundary.

    Args:
        estimate (ResourceEstimate): Fixed or callback-produced definition cost.

    Returns:
        ResourceEstimate: Original formulas with validation-only model
        constraints and a disabled domain-rewrite policy.

    Raises:
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    _validate_domain_rewrite_state(
        estimate,
        validate_deferred_symbols=True,
    )
    restored = _restore_domain_rewrite(estimate)
    return dataclasses.replace(
        restored,
        _constraints=tuple(
            dataclasses.replace(
                constraint,
                provenance=dataclasses.replace(
                    constraint.provenance,
                    origin=_ConstraintOrigin.MODEL_CONTRACT,
                    root_formal_names=(),
                ),
            )
            for constraint in restored._constraints
        ),
        _domain_rewrite_policy=_DomainRewritePolicy.DISABLED,
        _domain_rewrite_state=None,
        _rendered_assumption_snapshot=None,
    )


def _domain_assumptions(
    estimate: ResourceEstimate,
    registry: SymbolRegistry,
) -> tuple[ResourceAssumption, ...]:
    """Render unresolved consumed predicates with stable public aliases.

    Args:
        estimate (ResourceEstimate): Estimate carrying structured evidence.
        registry (SymbolRegistry): Shared public symbol registry.

    Returns:
        tuple[ResourceAssumption, ...]: Unresolved domain requirements.

    Raises:
        ValueError: If a consumed predicate resolves false.
    """
    state = estimate._domain_rewrite_state
    if state is None:
        return ()
    rendered: list[ResourceAssumption] = []
    for requirement in state.requirements:
        predicate = cast(sp.Basic, requirement.predicate)
        if predicate is sp.true:
            continue
        if predicate is sp.false:
            labels = ", ".join(requirement.labels) or "qkernel input"
            raise ValueError(f"Input-domain requirement is violated: {labels}.")
        text = stringify_expression(predicate, registry)
        sources = ", ".join(requirement.source_formals)
        message = f"valid resource formula requires {text}"
        if sources:
            message += f" (from {sources})"
        assumption = ResourceAssumption(message, source="qkernel input domain")
        if assumption not in rendered:
            rendered.append(assumption)
    return tuple(rendered)


def _domain_state_expressions(estimate: ResourceEstimate) -> Iterable[sp.Basic]:
    """Yield expressions retained only by domain rewrite state.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        Iterable[sp.Basic]: Snapshot and structured predicate expressions.
    """
    state = estimate._domain_rewrite_state
    if state is None:
        return ()
    return (
        *(cast(sp.Basic, sp.sympify(value)) for value in state.original.expressions()),
        *(cast(sp.Basic, sp.sympify(value)) for value in state.rewritten.expressions()),
        *(cast(sp.Basic, requirement.predicate) for requirement in state.requirements),
    )


__all__: list[str] = []
