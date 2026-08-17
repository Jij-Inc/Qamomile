"""Project scalar facts from immutable classical-array states."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, cast

import sympy as sp

from qamomile.circuit.estimator._array_state import (
    _ArrayChoiceState,
    _ArrayConstantState,
    _ArrayReferenceState,
    _ArraySliceState,
    _ArrayState,
    _ArrayStoreState,
    _ArrayUnknownLoopSummaryState,
    _constant_array_element,
)
from qamomile.circuit.estimator._classical_expression import _as_boolean
from qamomile.circuit.estimator._classical_facts import (
    _choice_classical_fact,
    _fact_from_expression,
    _ResolvedClassicalFact,
)
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ArrayValue, Value

_ResolveFact = Callable[[Any, bool], _ResolvedClassicalFact]
_ArrayProducer = Callable[[ArrayValue], Operation | None]
_ArrayStateLookup = Callable[[str], _ArrayState | None]


@dataclasses.dataclass(frozen=True, slots=True)
class _ArrayProjector:
    """Project immutable array states without depending on ``ExprResolver``.

    Scalar expression resolution and IR producer lookup remain owned by the
    enclosing resolver. Explicit callbacks keep the recursive array algorithm
    cohesive while avoiding a module import cycle.

    Args:
        array_state (_ArrayStateLookup): Callback returning the current
            immutable state snapshot for an array-result UUID.
        resolve_fact (_ResolveFact): Callback resolving a scalar value and its
            guarded scheduler dependencies. The second argument selects
            concrete resolution.
        array_producer (_ArrayProducer): Callback returning the operation that
            produced an array SSA version.
    """

    array_state: _ArrayStateLookup
    resolve_fact: _ResolveFact
    array_producer: _ArrayProducer

    def resolve_element(
        self,
        value: Value,
        *,
        concrete: bool,
    ) -> sp.Expr | None:
        """Resolve one scalar element from its parent array state.

        Args:
            value (Value): Scalar value carrying ``parent_array`` and
                ``element_indices`` provenance.
            concrete (bool): Whether index and value resolution requires
                concrete expressions.

        Returns:
            sp.Expr | None: Stored or initialized element expression, or
                ``None`` when the array state cannot be proven.
        """
        fact = self.resolve_element_fact(value, concrete=concrete)
        return cast(sp.Expr, fact.value) if fact is not None else None

    def resolve_element_fact(
        self,
        value: Value,
        *,
        concrete: bool,
    ) -> _ResolvedClassicalFact | None:
        """Resolve one scalar array element as a provenance fact.

        Args:
            value (Value): Scalar value carrying parent-array provenance.
            concrete (bool): Whether index and value resolution is concrete.

        Returns:
            _ResolvedClassicalFact | None: Precise projected fact, or ``None``
                when the persistent array state cannot explain the element.
        """
        parent = value.parent_array
        if parent is None or not value.element_indices:
            return None
        indices = tuple(
            self.resolve_fact(index, concrete) for index in value.element_indices
        )
        return self._resolve_state_element_fact(
            parent,
            indices,
            concrete=concrete,
            visited=set(),
        )

    def _resolve_state_element_fact(
        self,
        array: ArrayValue,
        indices: tuple[_ResolvedClassicalFact, ...],
        *,
        concrete: bool,
        visited: set[str],
        ignore_selection: str | None = None,
    ) -> _ResolvedClassicalFact | None:
        """Project one provenance fact from an immutable array SSA state.

        Args:
            array (ArrayValue): Array version whose contents are inspected.
            indices (tuple[_ResolvedClassicalFact, ...]): Local element-index
                facts.
            concrete (bool): Whether nested scalar resolution is concrete.
            visited (set[str]): Array UUIDs already followed on this path.
            ignore_selection (str | None): Binding UUID to bypass once for a
                raw reference. Defaults to ``None``.

        Returns:
            _ResolvedClassicalFact | None: Projected fact, or ``None`` when no
                supported immutable source explains the element.
        """
        state = self.array_state(array.uuid)
        if state is not None and ignore_selection != array.uuid:
            return self._project_state_fact(
                state,
                indices,
                concrete=concrete,
                visited=visited,
            )

        if array.uuid in visited:
            return None
        visited.add(array.uuid)

        if array.is_slice():
            if (
                len(indices) != 1
                or array.slice_of is None
                or array.slice_start is None
                or array.slice_step is None
            ):
                return None
            start = self.resolve_fact(array.slice_start, concrete)
            step = self.resolve_fact(array.slice_step, concrete)
            root_index = _fact_from_expression(
                cast(sp.Expr, start.value)
                + cast(sp.Expr, step.value) * cast(sp.Expr, indices[0].value),
                start,
                step,
                indices[0],
            )
            return self._resolve_state_element_fact(
                array.slice_of,
                (root_index,),
                concrete=concrete,
                visited=visited,
            )

        producer = self.array_producer(array)
        if isinstance(producer, IfOperation):
            merge = next(
                (
                    candidate
                    for candidate in producer.iter_merges()
                    if candidate.result.uuid == array.uuid
                    and isinstance(candidate.true_value, ArrayValue)
                    and isinstance(candidate.false_value, ArrayValue)
                ),
                None,
            )
            if merge is None:
                return None
            predicate = self.resolve_fact(producer.condition, concrete)
            true_value = self._resolve_state_element_fact(
                cast(ArrayValue, merge.true_value),
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            false_value = self._resolve_state_element_fact(
                cast(ArrayValue, merge.false_value),
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            if true_value is None or false_value is None:
                return None
            return _choice_classical_fact(true_value, false_value, predicate)
        if isinstance(producer, StoreArrayElementOperation):
            return self._project_state_fact(
                _ArrayStoreState(
                    _ArrayReferenceState(producer.array),
                    self.resolve_fact(producer.stored_value, concrete),
                    tuple(
                        self.resolve_fact(index, concrete)
                        for index in producer.index_values
                    ),
                ),
                indices,
                concrete=concrete,
                visited=visited,
            )

        constant = array.get_const_array()
        if constant is None:
            return None
        constant_element = _constant_array_element(
            constant,
            tuple(cast(sp.Expr, index.value) for index in indices),
        )
        if constant_element is None:
            return None
        return self.resolve_fact(constant_element, concrete)

    def _project_state_fact(
        self,
        state: _ArrayState,
        indices: tuple[_ResolvedClassicalFact, ...],
        *,
        concrete: bool,
        visited: set[str],
    ) -> _ResolvedClassicalFact | None:
        """Project one scalar provenance fact from a persistent array state.

        Args:
            state (_ArrayState): Frozen state captured in its producing scope.
            indices (tuple[_ResolvedClassicalFact, ...]): Local element-index
                facts.
            concrete (bool): Whether nested scalar resolution is concrete.
            visited (set[str]): IR reference UUIDs already followed.

        Returns:
            _ResolvedClassicalFact | None: Projected fact, or ``None`` when a
                reference state cannot prove a value.
        """
        if isinstance(state, _ArrayReferenceState):
            return self._resolve_state_element_fact(
                state.array,
                indices,
                concrete=concrete,
                visited=visited,
                ignore_selection=state.array.uuid,
            )
        if isinstance(state, _ArrayConstantState):
            element = _constant_array_element(
                state.contents,
                tuple(cast(sp.Expr, index.value) for index in indices),
            )
            if element is None:
                return None
            return self.resolve_fact(element, concrete)
        if isinstance(state, _ArraySliceState):
            if len(indices) != 1:
                return None
            source_index = _fact_from_expression(
                cast(sp.Expr, state.start.value)
                + cast(sp.Expr, state.step.value) * cast(sp.Expr, indices[0].value),
                state.start,
                state.step,
                indices[0],
            )
            return self._project_state_fact(
                state.source,
                (source_index,),
                concrete=concrete,
                visited=visited,
            )
        if isinstance(state, _ArrayChoiceState):
            predicate = _as_boolean(state.condition.value)
            if predicate is sp.true:
                return self._project_state_fact(
                    state.when_true,
                    indices,
                    concrete=concrete,
                    visited=visited,
                )
            if predicate is sp.false:
                return self._project_state_fact(
                    state.when_false,
                    indices,
                    concrete=concrete,
                    visited=visited,
                )
            true_value = self._project_state_fact(
                state.when_true,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            false_value = self._project_state_fact(
                state.when_false,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            if true_value is None or false_value is None:
                return None
            return _choice_classical_fact(
                true_value,
                false_value,
                state.condition,
            )
        if isinstance(state, _ArrayUnknownLoopSummaryState):
            active = _as_boolean(sp.Gt(state.iterations, 0))
            initial = self._project_state_fact(
                state.initial,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            active_fact = _ResolvedClassicalFact.create(
                state.fallback,
                {state.uncertainty_token: active},
            )
            if initial is None:
                return active_fact
            return _choice_classical_fact(
                active_fact,
                initial,
                _ResolvedClassicalFact.create(active),
            )
        if not isinstance(state, _ArrayStoreState):
            return None
        if len(state.indices) != len(indices):
            return None
        guard = sp.And(
            *(
                sp.Eq(load_index.value, store_index.value)
                for load_index, store_index in zip(
                    indices,
                    state.indices,
                    strict=True,
                )
            )
        )
        if guard is sp.true:
            return state.stored
        previous = self._project_state_fact(
            state.previous,
            indices,
            concrete=concrete,
            visited=visited,
        )
        if guard is sp.false:
            return previous
        if previous is None:
            return None
        selector = _fact_from_expression(
            guard,
            *indices,
            *state.indices,
        )
        return _choice_classical_fact(state.stored, previous, selector)
