"""Interpret range loops with carried region values."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._allocation_width import (
    _anonymous_allocation_width,
    _width_with_identity_aware_allocations,
)
from qamomile.circuit.estimator._call_liveness import (
    _captured_quantum_allocations,
)
from qamomile.circuit.estimator._classical_facts import (
    _ResolvedClassicalFact,
)
from qamomile.circuit.estimator._classical_provenance import (
    _loop_may_taint,
    _merge_classical_source_conditions,
    _value_taint_condition,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_indices import (
    _specialize_dependency_expression,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._estimate_composition import (
    _SequentialEstimateComposer,
)
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._interpreter_core import (
    _CONCRETE_REGION_REPLAY_LIMIT,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _with_conservative_loop_output_liveness,
)
from qamomile.circuit.estimator._interpreter_loop_support import (
    _loop_requires_hamiltonian_element_replay,
    _LoopSupportInterpreter,
)
from qamomile.circuit.estimator._liveness import (
    _maximum_live_owner_sizes,
    _maximum_live_owner_sizes_over_range,
)
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._measurement_provenance import (
    _merge_measurement_taint_conditions,
)
from qamomile.circuit.estimator._opaque import _estimate_uses_unresolved_functions
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_algebra import (
    _max_width,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
    _boolean_condition,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
    WidthResources,
)
from qamomile.circuit.estimator._scopes import (
    _invariant_identity_branch_guard,
    _LocalBlock,
    _loop_invariant_symbols,
    _solve_affine_recurrence,
    _typed_value_symbol,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
)


class _ForRegionInterpreter(_LoopSupportInterpreter):
    """Add concrete and symbolic carried-region range evaluation."""

    def _eval_region_for(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        *,
        start: ResourceExpr,
        stop: ResourceExpr,
        step: ResourceExpr,
        loop_symbol: sp.Symbol,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Evaluate a for loop with explicit loop-carried values.

        Concrete bounds are interpreted iteration by iteration with the same
        ``init -> block_arg -> yielded -> result`` rule as execution. Symbolic
        bounds use a closed form for independent affine recurrences; unsupported
        coupled or nonlinear recurrences remain explicit for symbolic bounds.
        Concrete unsupported recurrences fall back to exact replay before their
        final values are exposed to later operations.

        Args:
            operation (ForOperation): Loop carrying region arguments.
            resolver (ExprResolver): Enclosing symbolic environment.
            start (ResourceExpr): Inclusive loop start.
            stop (ResourceExpr): Exclusive loop stop.
            step (ResourceExpr): Python-range step.
            loop_symbol (sp.Symbol): Symbol representing the loop variable.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Exact concrete or closed-form symbolic estimate.

        Raises:
            ValueError: If a concrete loop has a zero step.
        """
        specialized_bounds = tuple(
            self._apply_condition_values(bound) for bound in (start, stop, step)
        )
        concrete_bounds = tuple(
            self._concrete_scalar(bound) for bound in specialized_bounds
        )
        if all(bound is not None for bound in concrete_bounds):
            concrete_start, concrete_stop, concrete_step = cast(
                tuple[int, int, int], concrete_bounds
            )
            if concrete_step == 0:
                raise ValueError(
                    "Resource estimation cannot evaluate a zero-step loop."
                )
            concrete_range = range(
                concrete_start,
                concrete_stop,
                concrete_step,
            )
            if (
                _loop_requires_hamiltonian_element_replay(operation)
                or len(concrete_range[: _CONCRETE_REGION_REPLAY_LIMIT + 1])
                <= _CONCRETE_REGION_REPLAY_LIMIT
            ):
                return self._eval_concrete_region_for(
                    operation,
                    resolver,
                    concrete_range,
                    controls=controls,
                )
        return self._eval_symbolic_region_for(
            operation,
            resolver,
            start=start,
            stop=stop,
            step=step,
            loop_symbol=loop_symbol,
            controls=controls,
        )

    def _eval_concrete_region_for(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        iterations: range,
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Interpret a concrete region-argument loop iteration by iteration.

        Args:
            operation (ForOperation): Loop carrying region arguments.
            resolver (ExprResolver): Enclosing symbolic environment.
            iterations (range): Concrete Python iteration values.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Sequential work with per-iteration peak width
                and identity-deduplicated static allocations.
        """
        carried = {
            arg.block_arg.uuid: self._apply_condition_values(resolver.resolve(arg.init))
            for arg in operation.region_args
        }
        carried_facts = {
            arg.block_arg.uuid: resolver.resolve_classical_fact(arg.init)
            for arg in operation.region_args
        }
        carried_taint = {
            arg.block_arg.uuid: condition
            for arg in operation.region_args
            if (
                condition := _value_taint_condition(
                    arg.init,
                    self._run_state.measurement_taint_conditions,
                )
            )
            is not sp.false
        }
        composer = _SequentialEstimateComposer(ResourceEstimate.zero())
        iteration_estimates: list[ResourceEstimate] = []
        iteration_width = WidthResources.zero()
        anonymous_allocated = _ZERO
        body = _LocalBlock(operation.operations)
        last_child = resolver
        array_states = self._initial_loop_array_states(operation, resolver)
        for ordinal, loop_value in enumerate(iterations):
            loop_expr = sp.Integer(loop_value)
            context = dict(carried)
            if operation.loop_var_value is not None:
                context[operation.loop_var_value.uuid] = loop_expr
            child = resolver.child_scope(
                inner_block=body,
                extra_context=context,
                extra_loop_vars={operation.loop_var: loop_expr},
            )
            child.copy_array_context()
            self._bind_loop_array_states(operation, child, array_states)
            last_child = child
            for arg in operation.region_args:
                fact = carried_facts[arg.block_arg.uuid]
                child.bind_classical_fact(
                    arg.block_arg,
                    _ResolvedClassicalFact.create(
                        carried[arg.block_arg.uuid],
                        fact.dependencies,
                    ),
                )
            with self._observation_occurrence_scope("range", operation, ordinal):
                with self._measurement_taint_scope(carried_taint):
                    iteration_estimate = self.eval_operations(
                        operation.operations,
                        child,
                        controls=controls,
                        initial_allocations=_captured_quantum_allocations(
                            operation.operations,
                            child,
                            self._run_state.allocation_owners_by_uuid,
                        ),
                    )
                    iteration_estimate = _with_constraints(
                        iteration_estimate,
                        *self._loop_iteration_array_constraints(
                            operation,
                            child,
                        ),
                    )
            composer.append(iteration_estimate)
            iteration_estimates.append(iteration_estimate)
            iteration_width = _max_width(iteration_width, iteration_estimate.width)
            anonymous_allocated = sp.Max(
                anonymous_allocated,
                _anonymous_allocation_width(
                    iteration_estimate.width,
                    iteration_estimate._allocation_sites,
                ),
            )
            carried = {
                arg.block_arg.uuid: self._apply_condition_values(
                    child.resolve(arg.yielded)
                )
                for arg in operation.region_args
            }
            carried_facts = {
                arg.block_arg.uuid: child.resolve_classical_fact(arg.yielded)
                for arg in operation.region_args
            }
            carried_taint = {
                arg.block_arg.uuid: condition
                for arg in operation.region_args
                if (
                    condition := _value_taint_condition(
                        arg.yielded,
                        iteration_estimate._measurement_taint_conditions,
                    )
                )
                is not sp.false
            }
            array_states = self._next_loop_array_states(child, array_states)
        for arg in operation.region_args:
            fact = carried_facts[arg.block_arg.uuid]
            resolver.bind_classical_fact(
                arg.result,
                _ResolvedClassicalFact.create(
                    carried[arg.block_arg.uuid],
                    fact.dependencies,
                ),
            )
        estimate = composer.finish()
        estimate = dataclasses.replace(
            estimate,
            width=_width_with_identity_aware_allocations(
                iteration_width,
                estimate._allocation_sites,
                anonymous_allocated=anonymous_allocated,
            ),
        )
        scheduled = self._schedule_concrete_loop_depth(
            operation,
            iteration_estimates,
            estimate,
        )
        result_taint = {
            arg.result.uuid: carried_taint[arg.block_arg.uuid]
            for arg in operation.region_args
            if arg.block_arg.uuid in carried_taint
        }
        scheduled = dataclasses.replace(
            scheduled,
            _measurement_taint_conditions=_merge_measurement_taint_conditions(
                scheduled._measurement_taint_conditions,
                result_taint,
            ),
        )
        scheduled = self._publish_loop_rebind_results(
            operation,
            resolver,
            last_child,
            scheduled,
            active_when=sp.true if iteration_estimates else sp.false,
            array_states=array_states,
        )
        if not iteration_estimates:
            return scheduled
        output_sizes, retains_prior_when = _maximum_live_owner_sizes(
            [iteration._output_sizes for iteration in iteration_estimates]
        )
        scheduled = dataclasses.replace(
            scheduled,
            _output_sizes=output_sizes,
            _has_output_summary=all(
                iteration._has_output_summary for iteration in iteration_estimates
            ),
        )
        if retains_prior_when is not sp.false:
            scheduled = _with_conservative_loop_output_liveness(
                scheduled,
                active_when=retains_prior_when,
                source="for liveness",
            )
        return scheduled

    def _eval_symbolic_region_for(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        *,
        start: ResourceExpr,
        stop: ResourceExpr,
        step: ResourceExpr,
        loop_symbol: sp.Symbol,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Summarize a symbolic or large region loop.

        Independent affine carries stay compact. A concrete loop whose carry
        cannot be solved that way is replayed exactly so its final values are
        safe for operations following the loop.

        Args:
            operation (ForOperation): Loop carrying region arguments.
            resolver (ExprResolver): Enclosing symbolic environment.
            start (ResourceExpr): Inclusive loop start.
            stop (ResourceExpr): Exclusive loop stop.
            step (ResourceExpr): Python-range step.
            loop_symbol (sp.Symbol): Symbol representing the loop variable.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Symbolic loop estimate and recurrence assumptions.

        Raises:
            ValueError: If input specialization produces a concrete zero-step
                range.
            NotImplementedError: If quantum resource use depends on an
                unsupported symbolic loop-carried recurrence.
        """
        carry_symbols = {
            arg.block_arg.uuid: _typed_value_symbol(
                arg.block_arg,
                f"{arg.var_name}_carry",
                fresh=True,
            )
            for arg in operation.region_args
        }
        initial_carry_facts = {
            arg.block_arg.uuid: resolver.resolve_classical_fact(arg.init)
            for arg in operation.region_args
        }
        initial_array_states = self._initial_loop_array_states(operation, resolver)
        completed_iterations = cast(
            sp.Expr,
            sp.simplify((loop_symbol - start) / step),
        )
        body_array_states, body_array_source_tokens = (
            self._conservative_loop_body_array_states(
                operation,
                initial_array_states,
                completed_iterations=completed_iterations,
            )
        )
        context: dict[str, sp.Expr] = dict(carry_symbols)
        if operation.loop_var_value is not None:
            context[operation.loop_var_value.uuid] = loop_symbol
        initial_carry_taint = {
            arg.block_arg.uuid: condition
            for arg in operation.region_args
            if (
                condition := _value_taint_condition(
                    arg.init,
                    self._run_state.measurement_taint_conditions,
                )
            )
            is not sp.false
        }
        probe = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=context,
            extra_loop_vars={operation.loop_var: loop_symbol},
        )
        probe.copy_array_context()
        self._bind_loop_array_states(operation, probe, body_array_states)
        for arg in operation.region_args:
            initial_fact = initial_carry_facts[arg.block_arg.uuid]
            probe.bind_classical_fact(
                arg.block_arg,
                _ResolvedClassicalFact.create(
                    carry_symbols[arg.block_arg.uuid],
                    initial_fact.dependencies,
                ),
            )
        # Evaluate once so branch phi results become available to the resolver
        # before recurrence expressions are inspected. The estimate itself is
        # discarded and recomputed with the closed-form carry-at-iteration values.
        with self._observation_occurrence_scope("range-family", operation, -1):
            with self._isolated_loop_taint_probe_state():
                with self._measurement_taint_scope(initial_carry_taint):
                    probe_estimate = self.eval_operations(
                        operation.operations,
                        probe,
                        controls=controls,
                        initial_allocations=_captured_quantum_allocations(
                            operation.operations,
                            probe,
                            self._run_state.allocation_owners_by_uuid,
                        ),
                    )
        loop_taint = _loop_may_taint(
            operation,
            initial_carry_taint,
            probe_estimate,
        )
        loop_source_conditions: dict[str, Boolean] = {}
        for fact in initial_carry_facts.values():
            _merge_classical_source_conditions(
                loop_source_conditions,
                fact.dependencies,
            )
        for arg in operation.region_args:
            _merge_classical_source_conditions(
                loop_source_conditions,
                probe.resolve_classical_fact(arg.yielded).dependencies,
            )

        iterations = symbolic_iterations(start, stop, step)
        specialized_bounds = tuple(
            self._apply_condition_values(bound, record_usage=False)
            for bound in (start, stop, step)
        )
        specialized_iterations = symbolic_iterations(*specialized_bounds)
        concrete_specialized_bounds = tuple(
            self._concrete_scalar(bound) for bound in specialized_bounds
        )
        concrete_bounds = (
            cast(tuple[int, int, int], concrete_specialized_bounds)
            if all(bound is not None for bound in concrete_specialized_bounds)
            else None
        )
        concrete_replay: range | None = None
        if concrete_bounds is not None:
            concrete_start, concrete_stop, concrete_step = concrete_bounds
            if concrete_step == 0:
                raise ValueError(
                    "Resource estimation cannot evaluate a zero-step loop."
                )
            candidate = range(concrete_start, concrete_stop, concrete_step)
            # Keep the concrete range as a correctness fallback after the
            # compact affine solver has had an opportunity to handle the loop.
            concrete_replay = candidate
        at_iteration: dict[str, sp.Expr] = {}
        final_values: dict[str, sp.Expr] = {}
        unresolved_iteration_functions: list[Any] = []
        guarded_assumptions: list[tuple[ResourceAssumption, Boolean]] = []
        all_carry_symbols = set(carry_symbols.values())
        for arg in operation.region_args:
            init = resolver.resolve(arg.init)
            yielded = probe.resolve(arg.yielded)
            carry_symbol = carry_symbols[arg.block_arg.uuid]
            recurrence = _solve_affine_recurrence(
                yielded=yielded,
                carry_symbol=carry_symbol,
                other_carry_symbols=all_carry_symbols - {carry_symbol},
                loop_symbol=loop_symbol,
                start=start,
                step=step,
                iterations=iterations,
                init=init,
            )
            if recurrence is None:
                if concrete_replay is not None:
                    return self._eval_concrete_region_for(
                        operation,
                        resolver,
                        concrete_replay,
                        controls=controls,
                    )
                carry_function = sp.Function(f"{arg.var_name}_carry")
                at_value = carry_function(loop_symbol)
                unresolved_iteration_functions.append(carry_function)
                unknown_final_value = _typed_value_symbol(
                    arg.result,
                    f"{arg.var_name}_after_loop",
                    fresh=True,
                )
                self._run_state.unresolved_resource_symbols.add(unknown_final_value)
                identity_guard = _invariant_identity_branch_guard(
                    yielded,
                    carry_symbol=carry_symbol,
                    invariant_symbols=(
                        _loop_invariant_symbols(
                            resolver,
                            operation.captures,
                            bound_expressions=(start, stop, step),
                        )
                        - self._run_state.runtime_observation_symbols
                        - all_carry_symbols
                        - {loop_symbol}
                    ),
                )
                final_value = cast(
                    sp.Expr,
                    sp.Piecewise(
                        (
                            init,
                            sp.Or(sp.Eq(iterations, _ZERO), identity_guard),
                        ),
                        (unknown_final_value, True),
                    ),
                )
                guarded_assumptions.append(
                    (
                        ResourceAssumption(
                            "loop-carried recurrence could not be reduced to an "
                            "independent affine closed form; its final value "
                            "remains symbolic",
                            source=arg.var_name,
                        ),
                        _and_conditions(
                            sp.Gt(iterations, _ZERO),
                            cast(Boolean, sp.Not(identity_guard)),
                        ),
                    )
                )
            else:
                at_value, final_value = recurrence
            at_iteration[arg.block_arg.uuid] = cast(sp.Expr, at_value)
            final_values[arg.result.uuid] = cast(sp.Expr, final_value)

        body_context = dict(at_iteration)
        if operation.loop_var_value is not None:
            body_context[operation.loop_var_value.uuid] = loop_symbol
        child = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=body_context,
            extra_loop_vars={operation.loop_var: loop_symbol},
        )
        child.copy_array_context()
        self._bind_loop_array_states(operation, child, body_array_states)
        for arg in operation.region_args:
            sources = (
                loop_source_conditions
                if arg.block_arg.uuid in loop_taint.at_iteration
                else initial_carry_facts[arg.block_arg.uuid].dependencies
            )
            child.bind_classical_fact(
                arg.block_arg,
                _ResolvedClassicalFact.create(
                    at_iteration[arg.block_arg.uuid],
                    sources,
                ),
            )
        with self._observation_occurrence_scope("range-family", operation, -1):
            with self._guarded_constraint_scope(sp.Gt(iterations, _ZERO)):
                with self._measurement_taint_scope(loop_taint.at_iteration):
                    inner = self.eval_operations(
                        operation.operations,
                        child,
                        controls=controls,
                        initial_allocations=_captured_quantum_allocations(
                            operation.operations,
                            child,
                            self._run_state.allocation_owners_by_uuid,
                        ),
                    )
                    inner = _with_constraints(
                        inner,
                        *self._loop_iteration_array_constraints(
                            operation,
                            child,
                        ),
                    )
        loop_body_reads_carried_array = self._estimate_reads_source_tokens(
            inner,
            body_array_source_tokens,
        )
        if _estimate_uses_unresolved_functions(
            inner,
            unresolved_iteration_functions,
        ):
            raise NotImplementedError(
                "Resource estimation cannot keep a symbolic loop compact when "
                "its quantum resource use depends on an unsupported symbolic "
                "loop-carried recurrence. Use a supported affine or fixed-point "
                "carry, or supply concrete loop bounds so the loop can be "
                "replayed."
            )
        dependency_start, dependency_stop, dependency_step = (
            _specialize_dependency_expression(
                bound,
                self._run_state.condition_values,
                self._run_state.branch_condition_names,
            )
            for bound in (start, stop, step)
        )
        estimate = inner._sum_over(
            loop_symbol,
            start,
            stop,
            step,
            dependency_start=dependency_start,
            dependency_stop=dependency_stop,
            dependency_step=dependency_step,
        )
        estimate = self._apply_disjoint_loop_depth(
            operation,
            resolver,
            child,
            inner,
            estimate,
            start=start,
            stop=stop,
            step=step,
            loop_symbol=loop_symbol,
            iterations=iterations,
            specialized_iterations=specialized_iterations,
            controls=controls,
            has_cross_iteration_dependency=(
                bool(loop_taint.at_iteration) or loop_body_reads_carried_array
            ),
        )
        active_iterations = _boolean_condition(sp.Gt(iterations, _ZERO))
        zero_iterations = _boolean_condition(sp.Eq(iterations, _ZERO))
        for arg in operation.region_args:
            final_sources: dict[str, Boolean] = {}
            _merge_classical_source_conditions(
                final_sources,
                {
                    source: _and_conditions(zero_iterations, guard)
                    for source, guard in initial_carry_facts[
                        arg.block_arg.uuid
                    ].dependencies.items()
                },
            )
            if arg.block_arg.uuid in loop_taint.final:
                _merge_classical_source_conditions(
                    final_sources,
                    {
                        source: _and_conditions(active_iterations, guard)
                        for source, guard in loop_source_conditions.items()
                    },
                )
            resolver.bind_classical_fact(
                arg.result,
                _ResolvedClassicalFact.create(
                    final_values[arg.result.uuid],
                    final_sources,
                ),
            )
        for assumption, active_when in guarded_assumptions:
            estimate = estimate._with_metadata(
                assumptions=(assumption,),
                active_when=active_when,
            )
        output_sizes, maximum_conservative_when, retains_prior_when = (
            _maximum_live_owner_sizes_over_range(
                inner._output_sizes,
                loop_symbol,
                start,
                step,
                iterations,
            )
        )
        estimate = dataclasses.replace(
            estimate,
            _output_sizes=output_sizes,
            _has_output_summary=inner._has_output_summary,
        )
        result_taint = {
            arg.result.uuid: loop_taint.final[arg.block_arg.uuid]
            for arg in operation.region_args
            if arg.block_arg.uuid in loop_taint.final
        }
        estimate = dataclasses.replace(
            estimate,
            _measurement_taint_conditions=_merge_measurement_taint_conditions(
                estimate._measurement_taint_conditions,
                result_taint,
            ),
        )
        if maximum_conservative_when is not sp.false:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=maximum_conservative_when,
                source="for liveness",
            )
        if retains_prior_when is not sp.false:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=_and_conditions(
                    sp.Gt(iterations, _ONE),
                    retains_prior_when,
                ),
                source="for liveness",
            )
        exit_array_states = self._unknown_loop_exit_array_states(
            operation,
            initial_array_states,
            iterations=iterations,
        )
        if loop_body_reads_carried_array:
            estimate = estimate._with_metadata(
                assumptions=(
                    ResourceAssumption(
                        "loop body resources use a conservative summary of "
                        "carried classical array state",
                        source="for classical state",
                    ),
                ),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(iterations, _ONE),
            )
        if exit_array_states:
            estimate = estimate._with_metadata(
                assumptions=(
                    ResourceAssumption(
                        "loop-exit classical array readiness is summarized at "
                        "the loop completion boundary",
                        source="for classical state",
                    ),
                ),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(iterations, _ZERO),
            )
        return self._publish_loop_rebind_results(
            operation,
            resolver,
            child,
            estimate,
            active_when=sp.Gt(iterations, _ZERO),
            array_states=exit_array_states,
        )
