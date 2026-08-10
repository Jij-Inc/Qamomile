"""Interpret dictionary-items loops for resource estimation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._allocation_width import (
    _anonymous_allocation_width,
    _width_with_identity_aware_allocations,
)
from qamomile.circuit.estimator._call_liveness import (
    _captured_quantum_allocations,
    _loop_body_has_destructive_observation,
    _loop_captured_observation_consumption,
    _with_operation_output_summary,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_composition import (
    _SequentialEstimateComposer,
)
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._interpreter_dataflow import (
    _require_uncontrolled_operation,
    _with_conservative_loop_output_liveness,
)
from qamomile.circuit.estimator._interpreter_for_items_region import (
    _ForItemsRegionInterpreter,
)
from qamomile.circuit.estimator._liveness import (
    _maximum_live_owner_sizes,
)
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
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
    WidthResources,
)
from qamomile.circuit.estimator._scopes import (
    _LocalBlock,
    resolve_for_items_cardinality,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
)


class _ForItemsInterpreter(_ForItemsRegionInterpreter):
    """Dispatch dictionary-items loop resource evaluation."""

    def eval_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a dictionary-items loop.

        Args:
            operation (ForItemsOperation): For-items operation.
            resolver (ExprResolver): Resolver for the outer scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Repeated body estimate.

        Raises:
            ValueError: If the items loop is nested under coherent quantum
                control.
            NotImplementedError: If an unbound dictionary loop has a body
                whose resource use depends on the current key or value, or if
                manually constructed IR carries a quantum region argument.
        """
        _require_uncontrolled_operation(operation, controls)
        if any(arg.result.type.is_quantum() for arg in operation.region_args):
            raise NotImplementedError(
                "Resource estimation does not support quantum "
                "ForItemsOperation region arguments. Keep quantum values as "
                "explicit loop captures or lower the loop before estimation."
            )
        cardinality = resolve_for_items_cardinality(operation)
        initial_array_states = self._initial_loop_array_states(operation, resolver)
        entries = self._for_items_entries(operation)
        symbolic_item_context, _item_symbols = self._symbolic_for_items_context(
            operation
        )
        symbolic_item_resolver = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=symbolic_item_context,
        )
        symbolic_item_resolver.copy_array_context()
        self._bind_loop_array_states(
            operation,
            symbolic_item_resolver,
            initial_array_states,
        )
        captured_allocations = _captured_quantum_allocations(
            operation.operations,
            symbolic_item_resolver,
            self._run_state.allocation_owners_by_uuid,
        )
        consumption_resolvers: tuple[ExprResolver, ...]
        consumption_iterations_are_definite = False
        if not _loop_body_has_destructive_observation(operation.operations):
            consumption_resolvers = ()
        elif entries is not None and len(entries) <= _MAX_EXACT_LOOP_WIRE_EXPANSION:
            consumption_resolvers = tuple(
                resolver.child_scope(
                    inner_block=_LocalBlock(operation.operations),
                    extra_context=self._concrete_for_items_context(
                        operation,
                        key,
                        value,
                    ),
                )
                for key, value in entries
            )
            consumption_iterations_are_definite = True
        else:
            consumption_resolvers = (symbolic_item_resolver,)
        body_output_sizes: Mapping[str, ResourceExpr] = {}
        if operation.region_args:
            with self._guarded_constraint_scope(sp.Gt(cardinality, _ZERO)):
                estimate = self._eval_region_for_items(
                    operation,
                    resolver,
                    cardinality=cardinality,
                    controls=controls,
                )
            body_output_sizes = estimate._output_sizes
        else:
            if entries is not None:
                estimate = self._eval_concrete_for_items(
                    operation,
                    resolver,
                    entries,
                    controls=controls,
                )
                body_output_sizes = estimate._output_sizes
            else:
                context, item_symbols = self._symbolic_for_items_context(operation)
                item_ordinal = sp.Dummy(
                    "item_index",
                    integer=True,
                    nonnegative=True,
                )
                child = resolver.child_scope(
                    inner_block=_LocalBlock(operation.operations),
                    extra_context=context,
                )
                child.copy_array_context()
                body_array_states, body_array_source_tokens = (
                    self._conservative_loop_body_array_states(
                        operation,
                        initial_array_states,
                        completed_iterations=item_ordinal,
                    )
                )
                self._bind_loop_array_states(
                    operation,
                    child,
                    body_array_states,
                )
                with self._guarded_constraint_scope(sp.Gt(cardinality, _ZERO)):
                    with self._observation_occurrence_scope(
                        "items-family",
                        operation,
                        -1,
                    ):
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
                self._ensure_for_items_resource_independent(inner, item_symbols)
                loop_body_reads_carried_array = self._estimate_reads_source_tokens(
                    inner,
                    body_array_source_tokens,
                )
                body_output_sizes = inner._output_sizes
                estimate = inner.sum_over(
                    item_ordinal,
                    _ZERO,
                    cardinality,
                    _ONE,
                )
                exit_array_states = self._unknown_loop_exit_array_states(
                    operation,
                    initial_array_states,
                    iterations=cardinality,
                )
                if loop_body_reads_carried_array:
                    estimate = estimate._with_metadata(
                        assumptions=(
                            ResourceAssumption(
                                "items-loop body resources use a conservative "
                                "summary of carried classical array state",
                                source="items classical state",
                            ),
                        ),
                        quality=EstimateQuality.CONSERVATIVE,
                        active_when=sp.Gt(cardinality, _ONE),
                    )
                if exit_array_states:
                    estimate = estimate._with_metadata(
                        assumptions=(
                            ResourceAssumption(
                                "items-loop classical array state is unknown "
                                "after one or more unbound entries",
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
        additional_consumed, retained_consumption_is_conservative = (
            _loop_captured_observation_consumption(
                operation.operations,
                captured_allocations,
                consumption_resolvers,
                definite_iterations=consumption_iterations_are_definite,
                allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
            )
        )
        estimate = _with_operation_output_summary(
            estimate,
            operation,
            resolver,
            active_when=sp.Gt(cardinality, _ZERO),
            body_output_sizes=body_output_sizes,
            additional_consumed_allocations=additional_consumed,
            allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
        )
        if retained_consumption_is_conservative:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=sp.Gt(cardinality, _ZERO),
                source="items liveness",
            )
        return _with_constraints(
            estimate,
            *self._loop_initial_array_constraints(operation, resolver),
        )

    def _eval_concrete_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        entries: tuple[tuple[Any, Any], ...],
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Interpret a bound items loop without carried values per entry.

        Args:
            operation (ForItemsOperation): Bound dictionary loop to evaluate.
            resolver (ExprResolver): Enclosing symbolic environment.
            entries (tuple[tuple[Any, Any], ...]): Concrete key-value entries
                in insertion order.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Per-entry composition. Gates and calls add,
                depth follows resolved wire dependencies, peak and ancilla
                width use the per-entry maximum, and allocated width is the
                union of distinct QInit identities.
        """
        composer = _SequentialEstimateComposer(ResourceEstimate.zero())
        entry_estimates: list[ResourceEstimate] = []
        iteration_width = WidthResources.zero()
        anonymous_allocated = _ZERO
        last_child = resolver
        array_states = self._initial_loop_array_states(operation, resolver)
        for ordinal, (key, value) in enumerate(entries):
            child = resolver.child_scope(
                inner_block=_LocalBlock(operation.operations),
                extra_context=self._concrete_for_items_context(
                    operation,
                    key,
                    value,
                ),
            )
            child.copy_array_context()
            self._bind_loop_array_states(operation, child, array_states)
            last_child = child
            with self._observation_occurrence_scope("items", operation, ordinal):
                entry_estimate = self.eval_operations(
                    operation.operations,
                    child,
                    controls=controls,
                    initial_allocations=_captured_quantum_allocations(
                        operation.operations,
                        child,
                        self._run_state.allocation_owners_by_uuid,
                    ),
                )
                entry_estimate = _with_constraints(
                    entry_estimate,
                    *self._loop_iteration_array_constraints(
                        operation,
                        child,
                    ),
                )
            composer.append(entry_estimate)
            entry_estimates.append(entry_estimate)
            iteration_width = _max_width(iteration_width, entry_estimate.width)
            anonymous_allocated = sp.Max(
                anonymous_allocated,
                _anonymous_allocation_width(
                    entry_estimate.width,
                    entry_estimate._allocation_sites,
                ),
            )
            array_states = self._next_loop_array_states(child, array_states)
        # ``seq`` counts every concrete visit to a QInit. Keep sequential
        # gate/depth/call totals, take reusable width fields per-entry, and
        # replace only allocated_qubits with the distinct static-site union.
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
