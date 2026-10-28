"""Compose resolved classical values with guarded dependencies."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._classical_expression import (
    _as_boolean,
)


@dataclasses.dataclass(frozen=True)
class _ResolvedClassicalFact:
    """Carry one resolved value and guarded scheduler dependencies.

    Dependencies are stored as a deterministic immutable tuple so facts can be
    compared structurally. Each pair maps one source token to the condition
    under which the resolved value depends on that source. Most tokens denote
    runtime observations; loop-array summaries also use internal readiness
    tokens that must not be mistaken for measurement feed-forward.

    Args:
        value (sp.Basic): Resolved symbolic or concrete classical value.
        source_guards (tuple[tuple[str, Boolean], ...]): Sorted source-token
            guards. False guards are omitted.
    """

    value: sp.Basic
    source_guards: tuple[tuple[str, Boolean], ...] = ()

    @classmethod
    def create(
        cls,
        value: sp.Basic | int | float | bool,
        source_guards: Mapping[str, sp.Basic] | None = None,
    ) -> _ResolvedClassicalFact:
        """Create a normalized immutable classical fact.

        Args:
            value (sp.Basic | int | float | bool): Resolved classical value.
            source_guards (Mapping[str, sp.Basic] | None): Optional source
                tokens and their activation guards. Defaults to ``None``.

        Returns:
            _ResolvedClassicalFact: Fact with deterministic nonfalse guards.
        """
        return cls(
            sp.sympify(value),
            _normalized_source_guards(source_guards or {}),
        )

    @property
    def dependencies(self) -> dict[str, Boolean]:
        """Return a mutable copy of the source-token guards.

        Returns:
            dict[str, Boolean]: Source tokens mapped to activation guards.
        """
        return dict(self.source_guards)


def _normalized_source_guards(
    source_guards: Mapping[str, sp.Basic],
) -> tuple[tuple[str, Boolean], ...]:
    """Normalize source guards into deterministic immutable storage.

    Args:
        source_guards (Mapping[str, sp.Basic]): Source-token activation guards.

    Returns:
        tuple[tuple[str, Boolean], ...]: Sorted nonfalse source guards.
    """
    normalized: list[tuple[str, Boolean]] = []
    for source, raw_guard in source_guards.items():
        guard = _as_boolean(raw_guard)
        if guard is not sp.false:
            normalized.append((source, guard))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _merge_source_guard_maps(
    *source_maps: Mapping[str, sp.Basic],
) -> dict[str, Boolean]:
    """Union guarded source dependencies by source-token identity.

    Args:
        *source_maps (Mapping[str, sp.Basic]): Source-token maps to combine.

    Returns:
        dict[str, Boolean]: Combined source guards using logical OR.
    """
    combined: dict[str, Boolean] = {}
    for source_map in source_maps:
        for source, raw_guard in source_map.items():
            guard = _as_boolean(raw_guard)
            combined[source] = _as_boolean(sp.Or(combined.get(source, sp.false), guard))
    return {
        source: guard for source, guard in combined.items() if guard is not sp.false
    }


def _guard_source_guards(
    source_guards: Mapping[str, sp.Basic],
    condition: sp.Basic,
) -> dict[str, Boolean]:
    """Conjoin one path condition with every source dependency.

    Args:
        source_guards (Mapping[str, sp.Basic]): Dependencies to guard.
        condition (sp.Basic): Path condition selecting those dependencies.

    Returns:
        dict[str, Boolean]: Dependencies active only under ``condition``.
    """
    predicate = _as_boolean(condition)
    return {
        source: _as_boolean(sp.And(predicate, _as_boolean(guard)))
        for source, guard in source_guards.items()
        if sp.And(predicate, _as_boolean(guard)) is not sp.false
    }


def _fact_with_dependencies(
    fact: _ResolvedClassicalFact,
    additions: Mapping[str, sp.Basic],
) -> _ResolvedClassicalFact:
    """Return one fact with additional source dependencies.

    Args:
        fact (_ResolvedClassicalFact): Existing resolved fact.
        additions (Mapping[str, sp.Basic]): Additional guarded sources.

    Returns:
        _ResolvedClassicalFact: Fact with merged immutable dependencies.
    """
    return _ResolvedClassicalFact.create(
        fact.value,
        _merge_source_guard_maps(fact.dependencies, additions),
    )


def _fact_from_expression(
    expression: sp.Basic,
    *sources: _ResolvedClassicalFact,
) -> _ResolvedClassicalFact:
    """Create a fact from sources that remain in the simplified expression.

    SymPy eagerly applies identities such as ``x & False == False`` and
    ``x * 0 == 0``. A source eliminated by such an identity no longer delays
    the result at runtime, so its readiness token must be removed as well.

    Args:
        expression (sp.Basic): Derived symbolic value.
        *sources (_ResolvedClassicalFact): Input facts used by the expression.

    Returns:
        _ResolvedClassicalFact: Expression with dependencies from inputs that
            still affect it.
    """
    expression_symbols = expression.free_symbols
    relevant_sources = (
        source
        for source in sources
        if source.value == expression
        or bool(source.value.free_symbols & expression_symbols)
    )
    return _ResolvedClassicalFact.create(
        expression,
        _merge_source_guard_maps(*(source.dependencies for source in relevant_sources)),
    )


def _coerce_classical_fact(
    value: sp.Basic | _ResolvedClassicalFact,
) -> _ResolvedClassicalFact:
    """Return an existing fact or wrap a dependency-free value.

    Args:
        value (sp.Basic | _ResolvedClassicalFact): Value or fact to normalize.

    Returns:
        _ResolvedClassicalFact: Normalized fact.
    """
    if isinstance(value, _ResolvedClassicalFact):
        return value
    return _ResolvedClassicalFact.create(value)


def _choice_classical_fact(
    when_true: _ResolvedClassicalFact,
    when_false: _ResolvedClassicalFact,
    selector: _ResolvedClassicalFact,
) -> _ResolvedClassicalFact:
    """Select between facts while preserving exact guarded dependencies.

    A selector does not affect the result when both projected facts are
    structurally identical. Otherwise branch dependencies are guarded by the
    selector and its negation, and the selector's own dependencies remain.

    Args:
        when_true (_ResolvedClassicalFact): Fact selected on a true predicate.
        when_false (_ResolvedClassicalFact): Fact selected on a false predicate.
        selector (_ResolvedClassicalFact): Predicate value and its sources.

    Returns:
        _ResolvedClassicalFact: Selected value with guarded dependencies.
    """
    if when_true == when_false:
        return when_true
    predicate = _as_boolean(selector.value)
    if predicate is sp.true:
        return _fact_with_dependencies(when_true, selector.dependencies)
    if predicate is sp.false:
        return _fact_with_dependencies(when_false, selector.dependencies)
    dependencies = _merge_source_guard_maps(
        _guard_source_guards(when_true.dependencies, predicate),
        _guard_source_guards(when_false.dependencies, sp.Not(predicate)),
        selector.dependencies,
    )
    value = sp.Piecewise(
        (when_true.value, predicate),
        (when_false.value, True),
    )
    return _ResolvedClassicalFact.create(value, dependencies)
