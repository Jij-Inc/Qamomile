"""Provide loop-state and dependency-scheduling services."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._array_state import (
    _ArrayState,
    _unknown_loop_array_summary_state,
)
from qamomile.circuit.estimator._classical_facts import (
    _ResolvedClassicalFact,
)
from qamomile.circuit.estimator._classical_provenance import (
    _loop_array_state_rebinds,
    _value_taint_condition,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._constraints import (
    _collect_array_value_constraints,
    _validated_unproven_array_constraints,
)
from qamomile.circuit.estimator._dependency_footprints import (
    _classical_dependency_key,
    _WireFootprint,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._interpreter_primitives import _PrimitiveInterpreter
from qamomile.circuit.estimator._loop_scheduling import (
    _concrete_loop_dependency_completion,
    _dependency_keys_depend_on_symbol,
    _disjoint_concrete_loop_depth,
    _loop_body_has_symbolic_quantum_index,
    _symbolic_disjoint_loop_depth,
    _uniform_parallel_loop_dependency_completion,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
    _boolean_condition,
    _expr,
    _piecewise,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    ResourceAssumption,
)
from qamomile.circuit.estimator._scheduling import (
    _aggregate_completion_overlap_condition,
    _dependency_depth,
    _estimate_has_nonzero_depth,
    _scheduled_depth_activity_conditions,
)
from qamomile.circuit.estimator._scopes import (
    _typed_value_symbol,
)
from qamomile.circuit.ir.dataflow import build_dependency_graph, walk_operations
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    LoopCarriedRebind,
    WhileOperation,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
)


def _loop_requires_hamiltonian_element_replay(operation: ForOperation) -> bool:
    """Return whether a concrete loop selects Hamiltonians by loop index.

    A Hamiltonian is an opaque Python object rather than a symbolic scalar.
    When an array-element access depends on the loop variable, one symbolic
    body evaluation cannot recover the different object selected by each
    iteration. Such a loop must inspect each concrete element, even when its
    iteration count exceeds the ordinary scalar-recurrence replay budget.

    Args:
        operation (ForOperation): Range loop to classify.

    Returns:
        bool: Whether a Pauli-evolution Hamiltonian index transitively depends
        on this loop's induction value.
    """
    loop_value = operation.loop_var_value
    if loop_value is None:
        return False
    graph = build_dependency_graph(operation.operations)
    for nested in walk_operations(operation.operations):
        if not isinstance(nested, PauliEvolveOp):
            continue
        observable = nested.observable
        if not observable.is_array_element():
            continue
        for index_value in observable.element_indices:
            pending = [index_value.uuid]
            visited: set[str] = set()
            while pending:
                current = pending.pop()
                if current == loop_value.uuid:
                    return True
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(graph.get(current, ()))
    return False


class _LoopSupportInterpreter(_PrimitiveInterpreter):
    """Add shared loop state, constraints, liveness, and scheduling."""

    def _initial_loop_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
    ) -> dict[LoopCarriedRebind, _ArrayState]:
        """Capture array states entering the first concrete loop iteration.

        Args:
            operation (ForOperation | ForItemsOperation): Loop whose explicit
                trace-time rebind records are inspected.
            resolver (ExprResolver): Resolver for the enclosing scope.

        Returns:
            dict[LoopCarriedRebind, _ArrayState]: Immutable entry snapshots for
            array rebind records only.
        """
        return {
            rebind: resolver.snapshot_array_state(cast(ArrayValue, rebind.before))
            for rebind in _loop_array_state_rebinds(operation)
        }

    def _bind_loop_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
        states: Mapping[LoopCarriedRebind, _ArrayState],
    ) -> None:
        """Bind prior array snapshots to one loop body's entry SSA values.

        Args:
            operation (ForOperation | ForItemsOperation): Loop being replayed.
            resolver (ExprResolver): Detached resolver for the next iteration.
            states (Mapping[LoopCarriedRebind, _ArrayState]): Prior exit state
                for every carried array lineage.
        """
        for rebind, state in states.items():
            assert isinstance(rebind.before, ArrayValue)
            resolver.bind_loop_array_input(
                operation.operations,
                rebind.before,
                state,
            )

    def _next_loop_array_states(
        self,
        resolver: ExprResolver,
        states: Mapping[LoopCarriedRebind, _ArrayState],
    ) -> dict[LoopCarriedRebind, _ArrayState]:
        """Capture array states produced by one concrete loop iteration.

        Args:
            resolver (ExprResolver): Evaluated iteration resolver.
            states (Mapping[LoopCarriedRebind, _ArrayState]): Carried records
                whose next snapshots are requested.

        Returns:
            dict[LoopCarriedRebind, _ArrayState]: Immutable exit snapshots.
        """
        return {
            rebind: resolver.snapshot_array_state(cast(ArrayValue, rebind.after))
            for rebind in states
        }

    def _unknown_loop_exit_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        initial_states: Mapping[LoopCarriedRebind, _ArrayState],
        *,
        iterations: ResourceExpr,
    ) -> dict[LoopCarriedRebind, _ArrayState]:
        """Build conservative caller-visible array states after a loop.

        Concrete loops selected for exact replay are handled before this
        helper is reached. A symbolic or otherwise unresolved loop instead
        exposes one internal fallback per carried array lineage so a body-local
        induction symbol cannot escape as a public resource parameter.

        Args:
            operation (ForOperation | ForItemsOperation): Loop owning the body
                transition.
            initial_states (Mapping[LoopCarriedRebind, _ArrayState]): Array
                snapshots before the first iteration.
            iterations (ResourceExpr): Number of range iterations.

        Returns:
            dict[LoopCarriedRebind, _ArrayState]: Unknown exit states keyed by
                carried array boundary.
        """
        exit_states: dict[LoopCarriedRebind, _ArrayState] = {}
        for rebind, initial_state in initial_states.items():
            assert isinstance(rebind.after, ArrayValue)
            fallback = _typed_value_symbol(
                rebind.after,
                f"{rebind.var_name}_after_loop_element",
                fresh=True,
            )
            self._run_state.unresolved_resource_symbols.add(fallback)
            if isinstance(rebind.after.type, BitType):
                self._run_state.runtime_value_domains[fallback] = (_ZERO, _ONE)
            uncertainty_token = f"$loop-array:{id(operation):x}:{rebind.after.uuid}"
            exit_states[rebind] = _unknown_loop_array_summary_state(
                initial=initial_state,
                iterations=cast(sp.Expr, iterations),
                fallback=fallback,
                uncertainty_token=uncertainty_token,
            )
        return exit_states

    def _conservative_loop_body_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        initial_states: Mapping[LoopCarriedRebind, _ArrayState],
        *,
        completed_iterations: sp.Expr,
    ) -> tuple[dict[LoopCarriedRebind, _ArrayState], frozenset[str]]:
        """Represent carried array elements conservatively inside a loop body.

        A single symbolic trace cannot know the value written by every prior
        iteration. The first iteration retains the initial state; later
        iterations project an internal fallback value with an explicit source
        token. Resource-affecting reads therefore select the safe runtime
        branch instead of reusing only the first iteration's value.

        Args:
            operation (ForOperation | ForItemsOperation): Loop owning the
                carried array lineages.
            initial_states (Mapping[LoopCarriedRebind, _ArrayState]): Array
                snapshots before the loop.
            completed_iterations (sp.Expr): Number of iterations preceding the
                body instance being summarized.

        Returns:
            tuple[dict[LoopCarriedRebind, _ArrayState], frozenset[str]]: Body
            entry states and the uncertainty source tokens they introduce.
        """
        states: dict[LoopCarriedRebind, _ArrayState] = {}
        source_tokens: set[str] = set()
        for rebind, initial_state in initial_states.items():
            assert isinstance(rebind.after, ArrayValue)
            fallback = _typed_value_symbol(
                rebind.after,
                f"{rebind.var_name}_loop_body_element",
                fresh=True,
            )
            self._run_state.unresolved_resource_symbols.add(fallback)
            if isinstance(rebind.after.type, BitType):
                self._run_state.runtime_value_domains[fallback] = (_ZERO, _ONE)
            source_token = f"$loop-array-body:{id(operation):x}:{rebind.after.uuid}"
            source_tokens.add(source_token)
            states[rebind] = _unknown_loop_array_summary_state(
                initial=initial_state,
                iterations=completed_iterations,
                fallback=fallback,
                uncertainty_token=source_token,
            )
        return states, frozenset(source_tokens)

    @staticmethod
    def _estimate_reads_source_tokens(
        estimate: ResourceEstimate,
        source_tokens: Iterable[str],
    ) -> bool:
        """Return whether quantum work reads any supplied classical source.

        Args:
            estimate (ResourceEstimate): Body estimate carrying directed
                scheduler accesses.
            source_tokens (Iterable[str]): Classical source-token identities to
                test.

        Returns:
            bool: Whether a token is read, or access metadata is unavailable
            and the safe answer is therefore unknown.
        """
        dependency_keys = frozenset(
            _classical_dependency_key(token) for token in source_tokens
        )
        if not dependency_keys:
            return False
        if estimate._dependency_reads is None:
            return True
        return not dependency_keys.isdisjoint(estimate._dependency_reads)

    def _apply_disjoint_loop_depth(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        body_resolver: ExprResolver,
        body: ResourceEstimate,
        sequential: ResourceEstimate,
        *,
        start: ResourceExpr,
        stop: ResourceExpr,
        step: ResourceExpr,
        loop_symbol: sp.Symbol,
        iterations: ResourceExpr,
        specialized_iterations: ResourceExpr,
        controls: ResourceExpr | int,
        has_cross_iteration_dependency: bool = False,
    ) -> ResourceEstimate:
        """Use parallel depth when distinct loop iterations touch distinct wires.

        The same projection applies whether or not the loop carries classical
        region arguments. Classical carries affect later values, but do not by
        themselves serialize quantum work on provably disjoint wires.

        Args:
            operation (ForOperation): Loop whose body is summarized.
            resolver (ExprResolver): Enclosing resolver for concrete wire
                projections.
            body_resolver (ExprResolver): Resolver used to evaluate one
                symbolic body iteration.
            body (ResourceEstimate): One-iteration resource estimate.
            sequential (ResourceEstimate): Sum of the body over all iterations.
            start (ResourceExpr): Inclusive range start.
            stop (ResourceExpr): Exclusive range stop.
            step (ResourceExpr): Range stride.
            loop_symbol (sp.Symbol): Symbol used for the body iteration.
            iterations (ResourceExpr): Unspecialized iteration count.
            specialized_iterations (ResourceExpr): Iteration count after
                supplied input values are applied.
            controls (ResourceExpr | int): Controls surrounding the loop.
            has_cross_iteration_dependency (bool): Whether classical state
                used by quantum work may flow from one iteration into the
                next. Defaults to ``False``.

        Returns:
            ResourceEstimate: Estimate with parallel depth when disjointness is
            proven, or the sequential estimate with conservative metadata when
            symbolic wire aliasing remains unresolved.
        """
        loop_barrier_condition = sequential._global_barrier_condition
        if specialized_iterations.is_zero is True:
            return sequential
        if has_cross_iteration_dependency:
            assumption = ResourceAssumption(
                "loop depth is sequential because classical state used by "
                "quantum work may flow between iterations",
                source="for",
            )
            return sequential._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(iterations, _ONE),
            )
        if _expr(controls) != _ZERO or loop_barrier_condition is sp.true:
            return sequential

        parallel_depth = _symbolic_disjoint_loop_depth(
            operation,
            body_resolver,
            body.depth,
            loop_symbol=loop_symbol,
            iterations=iterations,
            allocated_qubits=body.width.allocated_qubits,
            clean_ancillas=body.width.clean_ancilla_qubits,
            dirty_ancillas=body.width.dirty_ancilla_qubits,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        if parallel_depth is None:
            parallel_depth = _disjoint_concrete_loop_depth(
                operation,
                resolver,
                body.depth,
                body_dependency_keys=body._dependency_keys,
                start=start,
                stop=stop,
                step=step,
                loop_symbol=loop_symbol,
                allocated_qubits=body.width.allocated_qubits,
                clean_ancillas=body.width.clean_ancilla_qubits,
                dirty_ancillas=body.width.dirty_ancilla_qubits,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
        if parallel_depth is None:
            if _dependency_keys_depend_on_symbol(
                body._dependency_keys,
                loop_symbol,
            ) or _loop_body_has_symbolic_quantum_index(
                operation,
                body_resolver,
                loop_symbol,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            ):
                assumption = ResourceAssumption(
                    "symbolic loop depth is sequential because disjoint "
                    "iteration footprints could not be proven",
                    source="for",
                )
                return sequential._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=sp.And(
                        sp.Not(loop_barrier_condition),
                        sp.Gt(iterations, _ONE),
                    ),
                )
            return sequential

        parallel_completion = _concrete_loop_dependency_completion(
            body._dependency_completion,
            loop_symbol,
            start=start,
            stop=stop,
            step=step,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        if parallel_completion is None:
            parallel_completion = _uniform_parallel_loop_dependency_completion(
                body._dependency_completion,
                body_depth=body.depth.depth,
                projected_keys=sequential._dependency_keys,
                parallel_depth=parallel_depth.depth,
                loop_symbol=loop_symbol,
            )
        parallel_estimate = dataclasses.replace(
            sequential,
            depth=parallel_depth,
            _dependency_completion=(
                parallel_completion
                if parallel_completion is not None
                else sequential._dependency_completion
            ),
            _dependency_completion_uniform=(
                body._dependency_completion_uniform is True
                and all(
                    loop_symbol
                    not in cast(
                        ResourceExpr,
                        getattr(body.depth, field.name),
                    ).free_symbols
                    for field in dataclasses.fields(DepthResources)
                )
            ),
            _global_barrier_condition=sp.false,
        )
        if parallel_completion is None and parallel_estimate._dependency_keys:
            assumption = ResourceAssumption(
                "parallel loop uses aggregate completion latency because "
                "per-wire exit layers could not be projected exactly",
                source="for",
            )
            parallel_estimate = parallel_estimate._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(iterations, _ZERO),
            )
        return sequential.conditional(
            parallel_estimate,
            loop_barrier_condition,
        )

    def _schedule_concrete_loop_depth(
        self,
        operation: ForOperation | ForItemsOperation,
        entry_estimates: Sequence[ResourceEstimate],
        combined: ResourceEstimate,
    ) -> ResourceEstimate:
        """Schedule concrete loop iterations by their resolved wire use.

        Gate and call counts remain a sequential sum, but dictionary entries
        or range iterations acting on disjoint wires may occupy the same depth
        layers. A body-local allocation prevents this optimization because
        every iteration reuses the same allocation site.

        Args:
            operation (ForOperation | ForItemsOperation): Loop whose iterations
                were evaluated with concrete index/key/value bindings.
            entry_estimates (Sequence[ResourceEstimate]): Per-entry estimates
                before sequential depth offsets are applied.
            combined (ResourceEstimate): Sequentially composed estimate whose
                non-depth resources must be preserved.

        Returns:
            ResourceEstimate: Combined estimate with dependency-scheduled
                depth and caller-visible wire completion when available.
        """
        if not entry_estimates or any(
            estimate.width.allocated_qubits != _ZERO for estimate in entry_estimates
        ):
            return combined

        scheduled: list[tuple[Operation, ResourceEstimate]] = []
        footprints: list[_WireFootprint | None] = []
        for entry_estimate in entry_estimates:
            scheduled.append((operation, entry_estimate))
            if not _estimate_has_nonzero_depth(entry_estimate):
                footprints.append(None)
                continue
            keys = entry_estimate._dependency_keys
            if keys is None:
                assumption = ResourceAssumption(
                    "concrete loop iteration dependencies are unavailable; "
                    "depth remains sequential",
                    source="loop dependency scheduler",
                )
                return combined._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                )
            reads = (
                entry_estimate._dependency_reads
                if entry_estimate._dependency_reads is not None
                else keys
            )
            writes = (
                entry_estimate._dependency_writes
                if entry_estimate._dependency_writes is not None
                else keys
            )
            footprints.append((reads, writes))

        depth_activity_conditions = _scheduled_depth_activity_conditions(scheduled)
        (
            scheduled_depth,
            scheduled_completion,
            possible_alias_active,
            completion_is_uniform,
        ) = _dependency_depth(
            scheduled,
            footprints,
            activity_conditions=depth_activity_conditions,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        result = dataclasses.replace(
            combined,
            depth=scheduled_depth,
            _dependency_completion=scheduled_completion,
            _dependency_completion_uniform=completion_is_uniform,
        )
        if possible_alias_active is not sp.false:
            assumption = ResourceAssumption(
                "concrete loop quantum indices may alias and are "
                "scheduled conservatively",
                source="loop dependency scheduler",
            )
            result = result._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=possible_alias_active,
            )
        aggregate_completion_active = _aggregate_completion_overlap_condition(
            scheduled,
            footprints,
            activity_conditions=depth_activity_conditions,
        )
        if aggregate_completion_active is not sp.false:
            assumption = ResourceAssumption(
                "aggregate loop-iteration latency may over-serialize a "
                "later wire dependency",
                source="loop dependency scheduler",
            )
            result = result._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=aggregate_completion_active,
            )
        return result

    def _loop_initial_array_constraints(
        self,
        operation: ForOperation | ForItemsOperation | WhileOperation,
        resolver: ExprResolver,
    ) -> tuple[_ResourceConstraint, ...]:
        """Collect array requirements evaluated before entering a loop.

        Args:
            operation (ForOperation | ForItemsOperation | WhileOperation): Loop
                whose entry-side boundary is inspected.
            resolver (ExprResolver): Enclosing resolver for entry values.

        Returns:
            tuple[_ResourceConstraint, ...]: Validated entry requirements.
        """
        if isinstance(operation, WhileOperation):
            initial_values: list[ValueBase] = list(operation.operands[:1])
        else:
            initial_values = [cast(ValueBase, value) for value in operation.operands]
        for region_arg in operation.region_args:
            initial_values.append(region_arg.init)
        explicit_rebinds = tuple(
            rebind
            for rebind in operation.loop_carried_rebinds
            if not isinstance(rebind.before, ArrayValue)
            or not isinstance(rebind.after, ArrayValue)
        )
        for rebind in (*_loop_array_state_rebinds(operation), *explicit_rebinds):
            initial_values.append(rebind.before)
        constraints = [
            constraint.when(self._run_state.constraint_scope_condition)
            for constraint in _collect_array_value_constraints(
                initial_values,
                resolver,
            )
        ]
        return _validated_unproven_array_constraints(
            constraints,
            proven_cache=self._run_state.array_constraint_proven,
        )

    def _loop_iteration_array_constraints(
        self,
        operation: ForOperation | ForItemsOperation | WhileOperation,
        body_resolver: ExprResolver,
    ) -> tuple[_ResourceConstraint, ...]:
        """Collect array requirements evaluated by one loop iteration.

        These constraints must be attached to the one-iteration estimate before
        a symbolic range is summed. That lets ``ResourceEstimate._sum_over``
        quantify induction variables and closed-form carried values instead of
        leaking them as public parameters or validating only one representative
        iteration.

        Args:
            operation (ForOperation | ForItemsOperation | WhileOperation): Loop
                whose body-side boundary is inspected.
            body_resolver (ExprResolver): Resolver for one body iteration.

        Returns:
            tuple[_ResourceConstraint, ...]: Validated per-iteration
            requirements under the current constraint scope.
        """
        body_values: list[ValueBase] = []
        if isinstance(operation, WhileOperation):
            body_values.extend(operation.operands[1:])
        body_values.extend(arg.yielded for arg in operation.region_args)
        body_values.extend(rebind.after for rebind in operation.loop_carried_rebinds)
        constraints = [
            constraint.when(self._run_state.constraint_scope_condition)
            for constraint in _collect_array_value_constraints(
                body_values,
                body_resolver,
            )
        ]
        return _validated_unproven_array_constraints(
            constraints,
            proven_cache=self._run_state.array_constraint_proven,
        )

    def _publish_loop_rebind_results(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
        body_resolver: ExprResolver,
        estimate: ResourceEstimate,
        *,
        active_when: sp.Basic,
        array_states: Mapping[LoopCarriedRebind, _ArrayState] | None = None,
    ) -> ResourceEstimate:
        """Publish loop-exit values and guarded observation provenance.

        A classical Bit-array store currently remains a body-local SSA rewrite:
        the traced store reads the pre-loop array version on every iteration,
        and the frontend exposes its result after the loop without a RegionArg.
        Bind that result to the source array on a zero-trip path while retaining
        the one traced body result when at least one iteration executes.

        Args:
            operation (ForOperation | ForItemsOperation): Loop containing
                trace-time rebound records.
            resolver (ExprResolver): Enclosing resolver to update.
            body_resolver (ExprResolver): Resolver for one body execution.
            estimate (ResourceEstimate): Loop estimate carrying body taint.
            active_when (sp.Basic): Predicate that at least one iteration runs.
            array_states (Mapping[LoopCarriedRebind, _ArrayState] | None):
                Optional already-folded loop-exit snapshots. Defaults to
                ``None``, which snapshots the evaluated body resolver.

        Returns:
            ResourceEstimate: Estimate with rebound-result provenance.
        """
        active = _boolean_condition(active_when)
        selector_fact = _ResolvedClassicalFact.create(active)
        explicit_nonarray_rebinds = tuple(
            rebind
            for rebind in operation.loop_carried_rebinds
            if not (
                isinstance(rebind.before, ArrayValue)
                and isinstance(rebind.after, ArrayValue)
            )
        )
        rebinds = (
            *_loop_array_state_rebinds(operation),
            *explicit_nonarray_rebinds,
        )
        array_before_states = {
            rebind: resolver.snapshot_array_state(cast(ArrayValue, rebind.before))
            for rebind in rebinds
            if isinstance(rebind.before, ArrayValue)
            and isinstance(rebind.after, ArrayValue)
        }
        array_before_facts = {
            rebind: resolver.resolve_classical_fact(rebind.before)
            for rebind in array_before_states
        }
        result_taint = self._guard_array_updates(
            operation.operations,
            resolver,
            estimate,
            active_when=active,
        )
        for rebind in rebinds:
            if isinstance(rebind.before, ArrayValue) or isinstance(
                rebind.after,
                ArrayValue,
            ):
                if not isinstance(rebind.before, ArrayValue) or not isinstance(
                    rebind.after,
                    ArrayValue,
                ):
                    continue
                before_state = array_before_states[rebind]
                after_state = (
                    array_states[rebind]
                    if array_states is not None and rebind in array_states
                    else body_resolver.snapshot_array_state(rebind.after)
                )
                if active is sp.true:
                    resolver.bind_array_state(rebind.after, after_state)
                elif active is sp.false:
                    resolver.bind_array_state(rebind.after, before_state)
                else:
                    resolver.bind_array_state_selection(
                        rebind.after,
                        after_state,
                        before_state,
                        selector_fact,
                    )
                resolver.bind_classical_selection(
                    rebind.after,
                    body_resolver.resolve_classical_fact(rebind.after),
                    array_before_facts[rebind],
                    selector_fact,
                )
                before_taint = _value_taint_condition(
                    rebind.before,
                    self._run_state.measurement_taint_conditions,
                )
                after_taint = _value_taint_condition(
                    rebind.after,
                    estimate._measurement_taint_conditions,
                )
                condition = _boolean_condition(
                    sp.Or(
                        sp.And(active, after_taint),
                        sp.And(sp.Not(active), before_taint),
                    )
                )
                if condition is not sp.false:
                    result_taint[rebind.after.uuid] = condition
                continue
            before_fact = resolver.resolve_classical_fact(rebind.before)
            after_fact = (
                before_fact
                if active is sp.false
                else body_resolver.resolve_classical_fact(rebind.after)
            )
            resolver.bind_classical_selection(
                cast(Value, rebind.after),
                after_fact,
                before_fact,
                selector_fact,
                value_override=_piecewise(
                    cast(ResourceExpr, after_fact.value),
                    cast(ResourceExpr, before_fact.value),
                    active,
                ),
            )
            before_taint = _value_taint_condition(
                rebind.before,
                self._run_state.measurement_taint_conditions,
            )
            after_taint = _value_taint_condition(
                rebind.after,
                estimate._measurement_taint_conditions,
            )
            condition = _boolean_condition(
                sp.Or(
                    sp.And(active, after_taint),
                    sp.And(sp.Not(active), before_taint),
                )
            )
            if condition is not sp.false:
                result_taint[rebind.after.uuid] = condition
        if not result_taint:
            return estimate
        return dataclasses.replace(
            estimate,
            _measurement_taint_conditions={
                **estimate._measurement_taint_conditions,
                **result_taint,
            },
        )

    def _guard_array_updates(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
        estimate: ResourceEstimate | None = None,
        *,
        active_when: sp.Basic,
    ) -> dict[str, Boolean]:
        """Guard nested classical-array stores by one region's reachability.

        Array-state bindings are shared by related resolver scopes. Walking the
        complete nested operation tree therefore composes loop and branch
        reachability from the innermost region outward while keeping each
        Store's input as its pre-region array version.

        Args:
            operations (Sequence[Operation]): Region operations whose nested
                stores are guarded.
            resolver (ExprResolver): Resolver owning the shared array-state
                bindings.
            estimate (ResourceEstimate | None): Region estimate carrying
                observation provenance. Defaults to ``None`` when the caller
                publishes provenance separately.
            active_when (sp.Basic): Predicate that the region executes.

        Returns:
            dict[str, Boolean]: Store-result observation provenance guarded by
                the region reachability condition.
        """
        active = _boolean_condition(active_when)
        taint: dict[str, Boolean] = {}
        for operation in walk_operations(operations):
            if not isinstance(operation, StoreArrayElementOperation):
                continue
            if not operation.results or not isinstance(
                operation.results[0],
                ArrayValue,
            ):
                continue
            result = operation.results[0]
            resolver.guard_array_update(result, operation.array, active)
            if estimate is None:
                continue
            condition = _value_taint_condition(
                result,
                estimate._measurement_taint_conditions,
            )
            guarded = _and_conditions(active, condition)
            if guarded is not sp.false:
                taint[result.uuid] = guarded
        return taint
