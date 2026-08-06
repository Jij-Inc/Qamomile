"""Define and transform immutable classical-array estimator states."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._classical_facts import (
    _ResolvedClassicalFact,
)
from qamomile.circuit.ir.value import ArrayValue


class _ArrayState:
    """Marker base for immutable classical-array estimator state."""


@dataclasses.dataclass(frozen=True)
class _ArrayReferenceState(_ArrayState):
    """Reference an IR array whose provenance remains in the current scope.

    Args:
        array (ArrayValue): IR array projected when an element is requested.
    """

    array: ArrayValue


@dataclasses.dataclass(frozen=True)
class _ArrayConstantState(_ArrayState):
    """Hold immutable nested constant-array contents.

    Args:
        contents (Any): Frozen nested array payload.
    """

    contents: Any


@dataclasses.dataclass(frozen=True)
class _ArrayStoreState(_ArrayState):
    """Represent one immutable element update.

    Args:
        previous (_ArrayState): State before the update.
        stored (_ResolvedClassicalFact): Call-scoped stored scalar fact.
        indices (tuple[_ResolvedClassicalFact, ...]): Call-scoped store-index
            facts.
    """

    previous: _ArrayState
    stored: _ResolvedClassicalFact
    indices: tuple[_ResolvedClassicalFact, ...]


@dataclasses.dataclass(frozen=True)
class _ArraySliceState(_ArrayState):
    """Represent a one-dimensional affine array view.

    Args:
        source (_ArrayState): State viewed by the slice.
        start (_ResolvedClassicalFact): Source-space first-index fact.
        step (_ResolvedClassicalFact): Source-space stride fact.
    """

    source: _ArrayState
    start: _ResolvedClassicalFact
    step: _ResolvedClassicalFact


@dataclasses.dataclass(frozen=True)
class _ArrayChoiceState(_ArrayState):
    """Represent a predicate-selected immutable array state.

    Args:
        when_true (_ArrayState): State selected when the predicate is true.
        when_false (_ArrayState): State selected when the predicate is false.
        condition (_ResolvedClassicalFact): Selection-predicate fact.
    """

    when_true: _ArrayState
    when_false: _ArrayState
    condition: _ResolvedClassicalFact


@dataclasses.dataclass(frozen=True)
class _ArrayUnknownLoopSummaryState(_ArrayState):
    """Summarize an array after iterations with unknown keys or values.

    Args:
        initial (_ArrayState): State selected when the loop is empty.
        iterations (sp.Expr): Number of loop iterations.
        fallback (sp.Symbol): Internal unresolved element value.
        uncertainty_token (str): Loop-boundary readiness token.
    """

    initial: _ArrayState
    iterations: sp.Expr
    fallback: sp.Symbol
    uncertainty_token: str


def _unknown_loop_array_summary_state(
    *,
    initial: _ArrayState,
    iterations: sp.Expr,
    fallback: sp.Symbol,
    uncertainty_token: str,
) -> _ArrayUnknownLoopSummaryState:
    """Create a conservative loop-exit state for unknown item updates.

    Args:
        initial (_ArrayState): State selected when the loop is empty.
        iterations (sp.Expr): Number of loop iterations.
        fallback (sp.Symbol): Internal unresolved element value.
        uncertainty_token (str): Loop-boundary readiness token.

    Returns:
        _ArrayUnknownLoopSummaryState: Immutable unknown loop summary.
    """
    return _ArrayUnknownLoopSummaryState(
        initial,
        iterations,
        fallback,
        uncertainty_token,
    )


def _constant_array_element(
    contents: Any,
    indices: tuple[sp.Expr, ...],
) -> Any | None:
    """Return a proven element from immutable constant-array contents.

    A symbolic index is accepted only when every candidate at that axis is
    equal. This proves the result without choosing or guessing an index.

    Args:
        contents (Any): Frozen nested array payload.
        indices (tuple[sp.Expr, ...]): Resolved local indices.

    Returns:
        Any | None: Proven scalar element, or ``None`` when a symbolic index
            can select distinct values or the payload is malformed.
    """
    current = contents
    for index in indices:
        if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
            return None
        if isinstance(index, sp.Integer):
            position = int(index)
            if position < 0 or position >= len(current):
                return None
            current = current[position]
            continue
        if not current:
            return None
        first = current[0]
        if any(candidate != first for candidate in current[1:]):
            return None
        current = first
    return current
