"""Own persistent classical-array state during resource interpretation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._array_state import (
    _ArrayChoiceState,
    _ArrayConstantState,
    _ArrayReferenceState,
    _ArraySliceState,
    _ArrayState,
    _ArrayStoreState,
    _ArrayUnknownLoopSummaryState,
)
from qamomile.circuit.estimator._classical_expression import _as_boolean
from qamomile.circuit.estimator._classical_facts import (
    _coerce_classical_fact,
    _guard_source_guards,
    _merge_source_guard_maps,
    _ResolvedClassicalFact,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ArrayValue

_ResolveFact = Callable[[Any], _ResolvedClassicalFact]
_ArrayProducer = Callable[[ArrayValue], Operation | None]


class _ArrayContext:
    """Own the UUID-to-immutable-state lifecycle for classical arrays.

    Every resolver owns a distinct context object. Ordinary child resolvers
    may initially share the same mutable UUID map, while :meth:`detach`
    replaces only the calling context's map. Scalar resolution and producer
    lookup are injected so this module does not depend on ``ExprResolver``.

    Args:
        resolve_fact (_ResolveFact): Callback resolving scalar values and
            their guarded scheduler dependencies.
        array_producer (_ArrayProducer): Callback returning the operation that
            produced an array SSA version.
        states (dict[str, _ArrayState] | None): Mutable state map to own or
            share. Defaults to ``None``, which creates an empty map.
    """

    __slots__ = ("_array_producer", "_resolve_fact", "_states")

    def __init__(
        self,
        resolve_fact: _ResolveFact,
        array_producer: _ArrayProducer,
        states: dict[str, _ArrayState] | None = None,
    ) -> None:
        """Initialize an array-state context.

        Args:
            resolve_fact (_ResolveFact): Callback resolving scalar values and
                their guarded scheduler dependencies.
            array_producer (_ArrayProducer): Callback returning the operation
                that produced an array SSA version.
            states (dict[str, _ArrayState] | None): Mutable state map to own or
                share. Defaults to ``None``, which creates an empty map.
        """
        self._resolve_fact = resolve_fact
        self._array_producer = array_producer
        self._states = states if states is not None else {}

    def shared_states(self) -> dict[str, _ArrayState]:
        """Return the map shared when constructing an ordinary child owner.

        The returned map is intentionally not copied. Each resolver wraps it
        in a separate :class:`_ArrayContext`, so detaching one owner cannot
        redirect any sibling or parent owner.

        Returns:
            dict[str, _ArrayState]: Mutable UUID-to-state map to share.
        """
        return self._states

    def get(self, uuid: str) -> _ArrayState | None:
        """Return the current immutable state bound to one array UUID.

        Args:
            uuid (str): Array SSA identity to look up.

        Returns:
            _ArrayState | None: Bound state, or ``None`` when unbound.
        """
        return self._states.get(uuid)

    def bind_selection(
        self,
        result: ArrayValue,
        when_true: ArrayValue,
        when_false: ArrayValue,
        condition: sp.Basic | _ResolvedClassicalFact,
    ) -> None:
        """Bind one result to branch-selected source array states.

        Args:
            result (ArrayValue): Array SSA version visible after selection.
            when_true (ArrayValue): Source selected when ``condition`` is true.
            when_false (ArrayValue): Source selected when ``condition`` is
                false.
            condition (sp.Basic | _ResolvedClassicalFact): Selection predicate,
                optionally with guarded source dependencies.
        """
        self._states[result.uuid] = _ArrayChoiceState(
            self.snapshot(
                when_true,
                ignore_binding=result.uuid if when_true.uuid == result.uuid else None,
            ),
            self.snapshot(
                when_false,
                ignore_binding=(
                    result.uuid if when_false.uuid == result.uuid else None
                ),
            ),
            _coerce_classical_fact(condition),
        )

    def bind_state(self, result: ArrayValue, state: _ArrayState) -> None:
        """Bind one caller-visible array result to an immutable snapshot.

        Args:
            result (ArrayValue): Array SSA value receiving the snapshot.
            state (_ArrayState): Frozen state resolved in the producing scope.
        """
        self._states[result.uuid] = state

    def bind_state_selection(
        self,
        result: ArrayValue,
        when_true: _ArrayState,
        when_false: _ArrayState,
        selector: _ResolvedClassicalFact,
    ) -> None:
        """Bind detached branch snapshots to one selected array result.

        Args:
            result (ArrayValue): Array SSA result receiving the selected state.
            when_true (_ArrayState): Detached true-branch snapshot.
            when_false (_ArrayState): Detached false-branch snapshot.
            selector (_ResolvedClassicalFact): Branch selector and its source
                dependencies.
        """
        state = (
            when_true
            if when_true == when_false
            else _ArrayChoiceState(when_true, when_false, selector)
        )
        self.bind_state(result, state)

    def detach(self) -> None:
        """Detach this owner from its currently shared mutable state map."""
        self._states = dict(self._states)

    def fork(self) -> dict[str, _ArrayState]:
        """Return a detached shallow copy of every array-state binding.

        Returns:
            dict[str, _ArrayState]: Detached UUID-to-state mapping.
        """
        return dict(self._states)

    def export(
        self,
        arrays: Sequence[ArrayValue] | None = None,
    ) -> dict[str, _ArrayState]:
        """Export all or selected persistent array-state bindings.

        Args:
            arrays (Sequence[ArrayValue] | None): Optional array SSA values to
                export. Defaults to ``None``, which exports every binding.

        Returns:
            dict[str, _ArrayState]: Detached mapping safe to import elsewhere.
        """
        if arrays is None:
            return self.fork()
        return {
            array.uuid: self._states[array.uuid]
            for array in arrays
            if array.uuid in self._states
        }

    def import_states(
        self,
        context: Mapping[str, _ArrayState],
        *,
        replace: bool = False,
    ) -> None:
        """Import persistent array-state bindings into this owner.

        Args:
            context (Mapping[str, _ArrayState]): Exported UUID-to-state map.
            replace (bool): Whether to replace every existing binding before
                importing. Defaults to ``False``, which overlays the supplied
                bindings.
        """
        imported = dict(context)
        if replace:
            self._states = imported
            return
        self._states = {**self._states, **imported}

    def bind_call_input(
        self,
        block: Block,
        formal: ArrayValue,
        state: _ArrayState,
    ) -> None:
        """Bind the entry lineage of one callable array formal.

        Callable tracing mutates ``Block.input_values`` to the latest SSA
        version. A helper that stores into an input can therefore expose its
        produced output version as the formal interface value. The formal is
        nevertheless the value visible at call entry, so bind it before body
        evaluation; also bind unproduced aliases in the same lineage without
        overwriting produced intermediate versions.

        Args:
            block (Block): Selected callable body.
            formal (ArrayValue): Array value paired with the call operand.
            state (_ArrayState): Caller state captured at invocation time.
        """
        produced = {
            result.uuid
            for operation in walk_operations(block.operations)
            for result in operation.results
            if isinstance(result, ArrayValue)
        }
        self._states[formal.uuid] = state
        candidates: list[ArrayValue] = []
        for operation in walk_operations(block.operations):
            candidates.extend(
                operand
                for operand in operation.operands
                if isinstance(operand, ArrayValue)
                and operand.logical_id == formal.logical_id
            )
        for candidate in candidates:
            if candidate.uuid != formal.uuid and candidate.uuid not in produced:
                self._states[candidate.uuid] = state

    def bind_loop_input(
        self,
        operations: Sequence[Operation],
        entry: ArrayValue,
        state: _ArrayState,
    ) -> None:
        """Bind one carried snapshot to a loop body's entry SSA values.

        A traced loop body reuses the same SSA graph for every iteration. The
        explicit entry always receives the previous iteration's snapshot,
        even when tracing reused its UUID for a later result. Other aliases are
        rebound only when unproduced, so within-iteration updates stay intact.

        Args:
            operations (Sequence[Operation]): Loop-body operations.
            entry (ArrayValue): Pre-loop array value naming the carried lineage.
            state (_ArrayState): Snapshot produced by the previous iteration.
        """
        produced = {
            result.uuid
            for operation in walk_operations(operations)
            for result in operation.results
            if isinstance(result, ArrayValue)
        }
        self._states[entry.uuid] = state
        candidates: list[ArrayValue] = []
        for operation in walk_operations(operations):
            candidates.extend(
                operand
                for operand in operation.operands
                if isinstance(operand, ArrayValue)
                and operand.logical_id == entry.logical_id
            )
        for candidate in candidates:
            if candidate.uuid != entry.uuid and candidate.uuid not in produced:
                self._states[candidate.uuid] = state

    def snapshot(
        self,
        array: ArrayValue,
        *,
        ignore_binding: str | None = None,
        visited: set[str] | None = None,
    ) -> _ArrayState:
        """Capture call-scoped immutable state for an array SSA value.

        Scalar store operands, indices, and selection predicates are resolved
        immediately, so a later invocation reusing the callee's formal UUIDs
        cannot change an earlier caller result.

        Args:
            array (ArrayValue): Array whose current state is captured.
            ignore_binding (str | None): Binding bypassed for one raw producer
                lookup. Defaults to ``None``.
            visited (set[str] | None): Array UUIDs already visited on this path.
                Defaults to ``None``.

        Returns:
            _ArrayState: Immutable state tree rooted at ``array``.
        """
        if visited is None:
            visited = set()
        binding = self._states.get(array.uuid)
        if binding is not None and ignore_binding != array.uuid:
            return binding
        if array.uuid in visited:
            return _ArrayReferenceState(array)
        visited.add(array.uuid)

        if (
            array.is_slice()
            and array.slice_of is not None
            and array.slice_start is not None
            and array.slice_step is not None
        ):
            return _ArraySliceState(
                self.snapshot(array.slice_of, visited=set(visited)),
                self._resolve_fact(array.slice_start),
                self._resolve_fact(array.slice_step),
            )
        producer = self._array_producer(array)
        if isinstance(producer, StoreArrayElementOperation):
            return _ArrayStoreState(
                self.snapshot(producer.array, visited=set(visited)),
                self._resolve_fact(producer.stored_value),
                tuple(self._resolve_fact(index) for index in producer.index_values),
            )
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
            if merge is not None:
                return _ArrayChoiceState(
                    self.snapshot(
                        cast(ArrayValue, merge.true_value),
                        visited=set(visited),
                    ),
                    self.snapshot(
                        cast(ArrayValue, merge.false_value),
                        visited=set(visited),
                    ),
                    self._resolve_fact(producer.condition),
                )
        constant = array.get_const_array()
        if constant is not None:
            return _ArrayConstantState(constant)
        return _ArrayReferenceState(array)

    def dependencies(self, array: ArrayValue) -> dict[str, Boolean]:
        """Return every guarded observation source retained by an array state.

        This whole-state summary is used only at aggregate boundaries such as
        loops and calls. Element reads remain precise through the array
        projector; publishing the union here ensures each retained element
        token becomes ready no earlier than the boundary.

        Args:
            array (ArrayValue): Array whose immutable state is summarized.

        Returns:
            dict[str, Boolean]: Retained source tokens and activation guards.
        """
        return self._state_dependencies(self.snapshot(array))

    def _state_dependencies(self, state: _ArrayState) -> dict[str, Boolean]:
        """Collect dependencies recursively from one persistent state node.

        Args:
            state (_ArrayState): State node to inspect.

        Returns:
            dict[str, Boolean]: Source guards reachable from the node.
        """
        if isinstance(state, (_ArrayReferenceState, _ArrayConstantState)):
            return {}
        if isinstance(state, _ArrayStoreState):
            return _merge_source_guard_maps(
                self._state_dependencies(state.previous),
                state.stored.dependencies,
                *(index.dependencies for index in state.indices),
            )
        if isinstance(state, _ArraySliceState):
            return _merge_source_guard_maps(
                self._state_dependencies(state.source),
                state.start.dependencies,
                state.step.dependencies,
            )
        if isinstance(state, _ArrayChoiceState):
            predicate = _as_boolean(state.condition.value)
            return _merge_source_guard_maps(
                state.condition.dependencies,
                _guard_source_guards(
                    self._state_dependencies(state.when_true), predicate
                ),
                _guard_source_guards(
                    self._state_dependencies(state.when_false), sp.Not(predicate)
                ),
            )
        if isinstance(state, _ArrayUnknownLoopSummaryState):
            active = _as_boolean(sp.Gt(state.iterations, 0))
            return _merge_source_guard_maps(
                _guard_source_guards(
                    self._state_dependencies(state.initial), sp.Not(active)
                ),
                {state.uncertainty_token: active},
            )
        return {}

    def guard_update(
        self,
        result: ArrayValue,
        previous: ArrayValue,
        condition: sp.Basic | _ResolvedClassicalFact,
    ) -> None:
        """Guard one body-local array update by an execution path.

        Nested loops encounter the same store result from the inside out. When
        an inner loop already guarded that result, conjoin the outer
        reachability condition instead of replacing the more specific inner
        condition.

        Args:
            result (ArrayValue): Array SSA version produced by the store.
            previous (ArrayValue): Array version read by the store.
            condition (sp.Basic | _ResolvedClassicalFact): Predicate that the
                enclosing region executes, optionally with provenance.
        """
        current = self._states.get(result.uuid)
        if current is None:
            current = self.snapshot(result, ignore_binding=result.uuid)
        self._states[result.uuid] = _ArrayChoiceState(
            current,
            self.snapshot(previous),
            _coerce_classical_fact(condition),
        )

    def record_store(self, operation: StoreArrayElementOperation) -> None:
        """Record one store result in program order.

        Inlining can intentionally reuse the actual operand UUID for a
        callee's returned array. A block-wide producer index then cannot
        distinguish the pre-call and post-call state. Recording each store as
        it executes preserves the earlier snapshot for untouched slots.

        Args:
            operation (StoreArrayElementOperation): Store just encountered by
                the estimator's sequential interpreter.
        """
        result = operation.results[0]
        if not isinstance(result, ArrayValue):
            return
        previous = self._states.get(operation.array.uuid)
        if previous is None:
            # InlinePass can retain the callee formal as the Store operand
            # while substituting the returned result UUID with the caller's
            # current array version. The result binding is then the precise
            # pre-call state that untouched slots must retain.
            previous = self._states.get(result.uuid)
        if previous is None and result.is_slice():
            # An inlined Store result can itself retain the caller view's
            # affine lineage even when no prior binding was materialized.
            previous = self.snapshot(result, ignore_binding=result.uuid)
        if previous is None:
            if operation.array.uuid == result.uuid:
                constant = operation.array.get_const_array()
                previous = (
                    _ArrayConstantState(constant)
                    if constant is not None
                    else _ArrayReferenceState(operation.array)
                )
            else:
                previous = self.snapshot(operation.array)
        self._states[result.uuid] = _ArrayStoreState(
            previous,
            self._resolve_fact(operation.stored_value),
            tuple(self._resolve_fact(index) for index in operation.index_values),
        )
