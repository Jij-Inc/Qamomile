"""Interpret ordinary range loops for resource estimation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._array_state import (
    _ArrayState,
)
from qamomile.circuit.estimator._call_liveness import (
    _captured_quantum_allocations,
    _loop_captured_observation_consumption,
    _with_operation_output_summary,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    _specialize_dependency_expression,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._interpreter_core import (
    _CONCRETE_REGION_REPLAY_LIMIT,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _with_conservative_loop_output_liveness,
)
from qamomile.circuit.estimator._interpreter_for_region import _ForRegionInterpreter
from qamomile.circuit.estimator._liveness import (
    _maximum_live_owner_sizes_over_range,
)
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.estimator._scopes import (
    _LocalBlock,
    build_for_loop_scope,
)
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    LoopCarriedRebind,
)


class _ForInterpreter(_ForRegionInterpreter):
    """Dispatch ordinary range-loop resource evaluation."""

    def eval_for(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a for loop.

        Args:
            operation (ForOperation): Loop operation.
            resolver (ExprResolver): Resolver for the outer scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Loop resource estimate.

        Raises:
            NotImplementedError: If manually constructed IR carries a quantum
                value through a loop region argument.
        """
        if len(operation.operands) < 2:
            return ResourceEstimate.zero("empty_for")
        if any(arg.result.type.is_quantum() for arg in operation.region_args):
            raise NotImplementedError(
                "Resource estimation does not support quantum ForOperation "
                "region arguments. Keep quantum values as explicit loop "
                "captures or lower the loop before estimation."
            )
        child, start, stop, step, loop_symbol = build_for_loop_scope(
            operation,
            resolver,
        )
        iterations = symbolic_iterations(start, stop, step)
        initial_array_states = self._initial_loop_array_states(operation, resolver)
        specialized_bounds = tuple(
            self._apply_condition_values(bound, record_usage=False)
            for bound in (start, stop, step)
        )
        specialized_iterations = symbolic_iterations(*specialized_bounds)
        captured_allocations = _captured_quantum_allocations(
            operation.operations,
            child,
            self._run_state.allocation_owners_by_uuid,
        )
        consumption_resolvers: tuple[ExprResolver, ...]
        consumption_iterations_are_definite = False
        concrete_iteration_range: range | None = None
        concrete_specialized_bounds = tuple(
            self._concrete_scalar(bound) for bound in specialized_bounds
        )
        if all(bound is not None for bound in concrete_specialized_bounds):
            concrete_start, concrete_stop, concrete_step = cast(
                tuple[int, int, int],
                concrete_specialized_bounds,
            )
            concrete_range = range(concrete_start, concrete_stop, concrete_step)
            concrete_iteration_range = concrete_range
            if (
                operation.loop_var_value is not None
                and len(concrete_range[: _MAX_EXACT_LOOP_WIRE_EXPANSION + 1])
                <= _MAX_EXACT_LOOP_WIRE_EXPANSION
            ):
                consumption_resolvers = tuple(
                    child.child_scope(
                        _LocalBlock(operation.operations),
                        extra_context={
                            operation.loop_var_value.uuid: sp.Integer(iteration)
                        },
                        extra_loop_vars={operation.loop_var: sp.Integer(iteration)},
                    )
                    for iteration in concrete_range
                )
                consumption_iterations_are_definite = True
            elif not concrete_range:
                consumption_resolvers = ()
                consumption_iterations_are_definite = True
            else:
                consumption_resolvers = (child,)
        elif specialized_iterations.is_zero is True:
            consumption_resolvers = ()
            consumption_iterations_are_definite = True
        else:
            consumption_resolvers = (child,)
        dependency_start, dependency_stop, dependency_step = (
            _specialize_dependency_expression(
                bound,
                self._run_state.condition_values,
                self._run_state.branch_condition_names,
            )
            for bound in (start, stop, step)
        )
        body_output_sizes: Mapping[str, ResourceExpr] = {}
        output_maximum_conservative_when: Boolean = sp.false
        output_retains_prior_when: Boolean = sp.false
        replayed_concrete_body = False
        loop_exit_array_states: dict[LoopCarriedRebind, _ArrayState] | None = None
        loop_body_reads_carried_array = False
        if operation.region_args:
            with self._guarded_constraint_scope(sp.Gt(iterations, _ZERO)):
                estimate = self._eval_region_for(
                    operation,
                    resolver,
                    start=start,
                    stop=stop,
                    step=step,
                    loop_symbol=loop_symbol,
                    controls=controls,
                )
            body_output_sizes = estimate._output_sizes
            replayed_concrete_body = True
        else:
            inner: ResourceEstimate | None = None
            if specialized_iterations.is_zero is True:
                estimate = ResourceEstimate.zero("empty_for")
                loop_exit_array_states = dict(initial_array_states)
            elif (
                concrete_iteration_range is not None
                and (
                    bool(operation.loop_carried_rebinds)
                    or any(
                        isinstance(nested, StoreArrayElementOperation)
                        for nested in walk_operations(operation.operations)
                    )
                )
                and len(concrete_iteration_range[: _CONCRETE_REGION_REPLAY_LIMIT + 1])
                <= _CONCRETE_REGION_REPLAY_LIMIT
            ):
                with self._guarded_constraint_scope(sp.Gt(iterations, _ZERO)):
                    estimate = self._eval_concrete_region_for(
                        operation,
                        resolver,
                        concrete_iteration_range,
                        controls=controls,
                    )
                body_output_sizes = estimate._output_sizes
                replayed_concrete_body = True
            else:
                child.copy_array_context()
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
                self._bind_loop_array_states(
                    operation,
                    child,
                    body_array_states,
                )
                with self._guarded_constraint_scope(sp.Gt(iterations, _ZERO)):
                    inner = self.eval_operations(
                        operation.operations,
                        child,
                        controls=controls,
                        initial_allocations=captured_allocations,
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
                (
                    body_output_sizes,
                    output_maximum_conservative_when,
                    output_retains_prior_when,
                ) = _maximum_live_owner_sizes_over_range(
                    inner._output_sizes,
                    loop_symbol,
                    start,
                    step,
                    iterations,
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
                loop_exit_array_states = self._unknown_loop_exit_array_states(
                    operation,
                    initial_array_states,
                    iterations=iterations,
                )
            if not replayed_concrete_body and inner is not None:
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
                    has_cross_iteration_dependency=(loop_body_reads_carried_array),
                )
        if output_maximum_conservative_when is not sp.false:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=output_maximum_conservative_when,
                source="for liveness",
            )
        if output_retains_prior_when is not sp.false:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=_and_conditions(
                    sp.Gt(iterations, _ONE),
                    output_retains_prior_when,
                ),
                source="for liveness",
            )
        if not replayed_concrete_body:
            if loop_body_reads_carried_array:
                estimate = estimate._with_metadata(
                    assumptions=(
                        ResourceAssumption(
                            "loop body resources use a conservative summary "
                            "of carried classical array state",
                            source="for classical state",
                        ),
                    ),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=sp.Gt(iterations, _ONE),
                )
            if loop_exit_array_states:
                estimate = estimate._with_metadata(
                    assumptions=(
                        ResourceAssumption(
                            "loop-exit classical array readiness is summarized "
                            "at the loop completion boundary",
                            source="for classical state",
                        ),
                    ),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=sp.Gt(iterations, _ZERO),
                )
            estimate = self._publish_loop_rebind_results(
                operation,
                resolver,
                child,
                estimate,
                active_when=sp.Gt(iterations, _ZERO),
                array_states=loop_exit_array_states,
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
            active_when=sp.Gt(iterations, _ZERO),
            body_output_sizes=body_output_sizes,
            additional_consumed_allocations=additional_consumed,
            allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
        )
        if retained_consumption_is_conservative:
            estimate = _with_conservative_loop_output_liveness(
                estimate,
                active_when=sp.Gt(iterations, _ZERO),
                source="for liveness",
            )
        return _with_constraints(
            estimate,
            *self._loop_initial_array_constraints(
                operation,
                resolver,
            ),
        )
