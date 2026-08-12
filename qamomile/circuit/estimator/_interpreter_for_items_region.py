"""Interpret dictionary-items loops with carried region values."""

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
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._estimate_composition import (
    _SequentialEstimateComposer,
)
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._interpreter_dataflow import (
    _with_conservative_loop_output_liveness,
)
from qamomile.circuit.estimator._interpreter_for_items_context import (
    _ForItemsContextInterpreter,
)
from qamomile.circuit.estimator._liveness import (
    _maximum_live_owner_sizes,
    _maximum_live_owner_sizes_over_range,
)
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
    ForItemsOperation,
)


class _ForItemsRegionInterpreter(_ForItemsContextInterpreter):
    """Add symbolic and concrete carried-region items evaluation."""

    def _eval_region_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        *,
        cardinality: ResourceExpr,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Evaluate a dictionary-items loop with loop-carried values.

        Args:
            operation (ForItemsOperation): Items loop carrying region arguments.
            resolver (ExprResolver): Enclosing symbolic environment.
            cardinality (ResourceExpr): Symbolic item count.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Per-entry estimate for bound dictionaries,
            otherwise a cardinality-based symbolic estimate.

        Raises:
            NotImplementedError: If an unbound loop-carried value has a
                recurrence that depends on the current item key or value, or
                if quantum resource use depends on an unsupported symbolic
                recurrence.
        """
        entries = self._for_items_entries(operation)
        if entries is not None:
            return self._eval_concrete_region_for_items(
                operation,
                resolver,
                entries,
                controls=controls,
            )

        item_symbol = sp.Dummy("item_index", integer=True, nonnegative=True)
        context, item_symbols = self._symbolic_for_items_context(operation)
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
        body_array_states, body_array_source_tokens = (
            self._conservative_loop_body_array_states(
                operation,
                initial_array_states,
                completed_iterations=item_symbol,
            )
        )
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
        context.update(carry_symbols)
        probe = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=context,
        )
        probe.copy_array_context()
        self._bind_loop_array_states(
            operation,
            probe,
            body_array_states,
        )
        for arg in operation.region_args:
            initial_fact = initial_carry_facts[arg.block_arg.uuid]
            probe.bind_classical_fact(
                arg.block_arg,
                _ResolvedClassicalFact.create(
                    carry_symbols[arg.block_arg.uuid],
                    initial_fact.dependencies,
                ),
            )
        with self._observation_occurrence_scope("items-family", operation, -1):
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
        self._ensure_for_items_resource_independent(probe_estimate, item_symbols)

        at_iteration: dict[str, sp.Expr] = {}
        final_values: dict[str, sp.Expr] = {}
        unresolved_iteration_functions: list[Any] = []
        guarded_assumptions: list[tuple[ResourceAssumption, Boolean]] = []
        all_carry_symbols = set(carry_symbols.values())
        for arg in operation.region_args:
            carry_symbol = carry_symbols[arg.block_arg.uuid]
            yielded = probe.resolve(arg.yielded)
            if yielded.free_symbols & item_symbols:
                raise NotImplementedError(
                    "Resource estimation does not support a symbolic "
                    "ForItemsOperation carry whose recurrence depends on the "
                    "current item key or value. Supply a concrete dictionary "
                    "or make the recurrence entry-independent."
                )
            recurrence = _solve_affine_recurrence(
                yielded=yielded,
                carry_symbol=carry_symbol,
                other_carry_symbols=all_carry_symbols - {carry_symbol},
                loop_symbol=item_symbol,
                start=_ZERO,
                step=_ONE,
                iterations=cardinality,
                init=resolver.resolve(arg.init),
            )
            if recurrence is None:
                carry_function = sp.Function(f"{arg.var_name}_carry")
                at_value = carry_function(item_symbol)
                unresolved_iteration_functions.append(carry_function)
                unknown_final_value = _typed_value_symbol(
                    arg.result,
                    f"{arg.var_name}_after_items",
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
                            bound_expressions=(cardinality,),
                        )
                        - self._run_state.runtime_observation_symbols
                        - all_carry_symbols
                        - item_symbols
                        - {item_symbol}
                    ),
                )
                final_value = cast(
                    sp.Expr,
                    sp.Piecewise(
                        (
                            resolver.resolve(arg.init),
                            sp.Or(sp.Eq(cardinality, _ZERO), identity_guard),
                        ),
                        (unknown_final_value, True),
                    ),
                )
                guarded_assumptions.append(
                    (
                        ResourceAssumption(
                            "items-loop carry could not be reduced to an "
                            "independent affine closed form; its final value "
                            "remains symbolic",
                            source=arg.var_name,
                        ),
                        _and_conditions(
                            sp.Gt(cardinality, _ZERO),
                            cast(Boolean, sp.Not(identity_guard)),
                        ),
                    )
                )
            else:
                at_value, final_value = recurrence
            at_iteration[arg.block_arg.uuid] = cast(sp.Expr, at_value)
            final_values[arg.result.uuid] = cast(sp.Expr, final_value)

        body_context = {**context, **at_iteration}
        child = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=body_context,
        )
        child.copy_array_context()
        self._bind_loop_array_states(
            operation,
            child,
            body_array_states,
        )
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
        with self._observation_occurrence_scope("items-family", operation, -1):
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
        self._ensure_for_items_resource_independent(inner, item_symbols)
        if _estimate_uses_unresolved_functions(
            inner,
            unresolved_iteration_functions,
        ):
            raise NotImplementedError(
                "Resource estimation cannot keep a symbolic items loop compact "
                "when its quantum resource use depends on an unsupported "
                "symbolic loop-carried recurrence. Supply a concrete dictionary "
                "or use a supported affine or fixed-point carry."
            )
        estimate = inner.sum_over(item_symbol, _ZERO, cardinality, _ONE)
        if loop_body_reads_carried_array:
            estimate = estimate._with_metadata(
                assumptions=(
                    ResourceAssumption(
                        "items-loop body resources use a conservative summary "
                        "of carried classical array state",
                        source="items classical state",
                    ),
                ),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(cardinality, _ONE),
            )
        active_entries = _boolean_condition(sp.Gt(cardinality, _ZERO))
        zero_entries = _boolean_condition(sp.Eq(cardinality, _ZERO))
        for arg in operation.region_args:
            final_sources: dict[str, Boolean] = {}
            _merge_classical_source_conditions(
                final_sources,
                {
                    source: _and_conditions(zero_entries, guard)
                    for source, guard in initial_carry_facts[
                        arg.block_arg.uuid
                    ].dependencies.items()
                },
            )
            if arg.block_arg.uuid in loop_taint.final:
                _merge_classical_source_conditions(
                    final_sources,
                    {
                        source: _and_conditions(active_entries, guard)
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
        exit_array_states = self._unknown_loop_exit_array_states(
            operation,
            initial_array_states,
            iterations=cardinality,
        )
        if exit_array_states:
            estimate = estimate._with_metadata(
                assumptions=(
                    ResourceAssumption(
                        "items-loop classical array state is unknown after one "
                        "or more unbound entries",
                        source="items classical state",
                    ),
                ),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(cardinality, _ZERO),
            )
        estimate = self._publish_loop_rebind_results(
            operation,
            resolver,
            child,
            estimate,
            active_when=sp.Gt(cardinality, _ZERO),
            array_states=exit_array_states,
        )
        for assumption, active_when in guarded_assumptions:
            estimate = estimate._with_metadata(
                assumptions=(assumption,),
                active_when=active_when,
            )
        output_sizes, maximum_conservative_when, retains_prior_when = (
            _maximum_live_owner_sizes_over_range(
                inner._output_sizes,
                item_symbol,
                _ZERO,
                _ONE,
                cardinality,
            )
        )
        estimate = dataclasses.replace(
            estimate,
            _output_sizes=output_sizes,
            _has_output_summary=inner._has_output_summary,
        )
        if maximum_conservative_when is not sp.false:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=maximum_conservative_when,
                source="items liveness",
            )
        if retains_prior_when is not sp.false:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=_and_conditions(
                    sp.Gt(cardinality, _ONE),
                    retains_prior_when,
                ),
                source="items liveness",
            )
        return estimate

    def _eval_concrete_region_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        entries: tuple[tuple[Any, Any], ...],
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Interpret a bound items loop with carried values per entry.

        Args:
            operation (ForItemsOperation): Items loop carrying region arguments.
            resolver (ExprResolver): Enclosing symbolic environment.
            entries (tuple[tuple[Any, Any], ...]): Bound key-value entries.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Summed work with dependency-scheduled depth,
                per-entry peak width, and identity-deduplicated allocations.
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
        entry_estimates: list[ResourceEstimate] = []
        iteration_width = WidthResources.zero()
        anonymous_allocated = _ZERO
        last_child = resolver
        array_states = self._initial_loop_array_states(operation, resolver)
        for ordinal, (key, value) in enumerate(entries):
            context = {
                **carried,
                **self._concrete_for_items_context(operation, key, value),
            }
            child = resolver.child_scope(
                inner_block=_LocalBlock(operation.operations),
                extra_context=context,
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
            with self._observation_occurrence_scope("items", operation, ordinal):
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
            entry_estimates.append(iteration_estimate)
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
            entry_estimates,
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
            active_when=sp.true if entries else sp.false,
            array_states=array_states,
        )
        if not entry_estimates:
            return scheduled
        output_sizes, retains_prior_when = _maximum_live_owner_sizes(
            [entry._output_sizes for entry in entry_estimates]
        )
        scheduled = dataclasses.replace(
            scheduled,
            _output_sizes=output_sizes,
            _has_output_summary=all(
                entry._has_output_summary for entry in entry_estimates
            ),
        )
        if retains_prior_when is not sp.false:
            scheduled = _with_conservative_loop_output_liveness(
                scheduled,
                active_when=retains_prior_when,
                source="items liveness",
            )
        return scheduled
