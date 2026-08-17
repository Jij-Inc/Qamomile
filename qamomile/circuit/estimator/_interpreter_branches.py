"""Interpret conditional branches for resource estimation."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._call_liveness import (
    _captured_quantum_allocations,
    _definitely_consumed_captured_allocations,
)
from qamomile.circuit.estimator._classical_provenance import (
    _classical_fact_runtime_condition,
    _classical_fact_uncertainty_condition,
)
from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._interpreter_dataflow import (
    _conditional_resource_map,
    _if_merge_captured_allocations,
    _refine_boolean_under_assumption,
    _require_uncontrolled_operation,
)
from qamomile.circuit.estimator._interpreter_if_merge import _IfMergeInterpreter
from qamomile.circuit.estimator._liveness import _branch_owner_sizes
from qamomile.circuit.estimator._measurement_provenance import (
    _merge_measurement_taint_conditions,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_algebra import (
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.estimator._scopes import (
    build_if_scopes,
)
from qamomile.circuit.ir.operation.control_flow import (
    IfOperation,
)


class _BranchInterpreter(_IfMergeInterpreter):
    """Add conditional-branch selection and orchestration."""

    def eval_if(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a conditional, specializing decidable compile-time branches.

        When the condition is a compile-time constant (from ``bindings``) or is
        resolvable from a supplied classical parameter value (from
        ``inputs``), only the taken branch is counted. A measurement-backed
        or otherwise undecidable condition falls back to the conservative maximum
        of both branches.

        Args:
            operation (IfOperation): If operation.
            resolver (ExprResolver): Resolver for the outer scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Taken-branch estimate when decidable, otherwise the
            maximum of the true and false branches.
        """
        condition_fact = resolver.resolve_classical_fact(operation.condition)
        resolved_condition = cast(sp.Expr, condition_fact.value)
        predicate = _boolean_condition(resolved_condition)
        runtime_guard = _classical_fact_runtime_condition(condition_fact)
        uncertainty_guard = _classical_fact_uncertainty_condition(condition_fact)
        conservative_guard = _boolean_condition(sp.Or(runtime_guard, uncertainty_guard))
        compile_predicate = (
            _refine_boolean_under_assumption(
                predicate,
                cast(Boolean, sp.Not(conservative_guard)),
            )
            if conservative_guard is not sp.false
            else predicate
        )
        internal_choice_symbols = {
            symbol
            for symbol in predicate.free_symbols
            if isinstance(symbol, sp.Symbol)
            and (
                symbol
                in (
                    self._run_state.runtime_observation_symbols
                    | self._run_state.unresolved_resource_symbols
                )
                or any(
                    symbol.name.endswith(uuid)
                    for uuid in self._run_state.measurement_taint_conditions
                )
            )
        }
        if conservative_guard is not sp.false and internal_choice_symbols:
            # A source guard proves that internal values are semantic
            # don't-cares on its complement. Give those roots one arbitrary
            # representative before retrying the bounded projection so nested
            # Piecewise carry formulas cannot retain a dead internal token.
            representative = _boolean_condition(
                cast(
                    sp.Basic,
                    predicate.xreplace(
                        {symbol: _ZERO for symbol in internal_choice_symbols}
                    ),
                )
            )
            compile_predicate = _refine_boolean_under_assumption(
                representative,
                cast(Boolean, sp.Not(conservative_guard)),
            )
        taken, note = self._decide_branch(resolved_condition)
        if conservative_guard is not sp.false:
            # Under this guard the branch is selected by a shot-dependent
            # observation or unresolved loop state, even if the compile-time
            # projection happens to simplify elsewhere.
            taken = None
        true_child, false_child = build_if_scopes(operation, resolver)
        true_inputs = _captured_quantum_allocations(
            operation.true_operations,
            true_child,
            self._run_state.allocation_owners_by_uuid,
        )
        false_inputs = _captured_quantum_allocations(
            operation.false_operations,
            false_child,
            self._run_state.allocation_owners_by_uuid,
        )
        true_inputs.update(
            _if_merge_captured_allocations(
                operation.true_operations,
                [
                    merge.true_value
                    for merge in operation.iter_merges()
                    if merge.result.type.is_quantum()
                ],
                true_child,
                self._run_state.allocation_owners_by_uuid,
            )
        )
        false_inputs.update(
            _if_merge_captured_allocations(
                operation.false_operations,
                [
                    merge.false_value
                    for merge in operation.iter_merges()
                    if merge.result.type.is_quantum()
                ],
                false_child,
                self._run_state.allocation_owners_by_uuid,
            )
        )
        true_consumed = _definitely_consumed_captured_allocations(
            operation.true_operations,
            true_inputs,
            true_child,
            self._run_state.allocation_owners_by_uuid,
        )
        false_consumed = _definitely_consumed_captured_allocations(
            operation.false_operations,
            false_inputs,
            false_child,
            self._run_state.allocation_owners_by_uuid,
        )
        if taken is not None:
            branch_ops = (
                operation.true_operations if taken else operation.false_operations
            )
            branch_child = true_child if taken else false_child
            estimate = self.eval_operations(
                branch_ops,
                branch_child,
                controls=controls,
                initial_allocations=true_inputs if taken else false_inputs,
            )
            self._publish_if_results(
                operation,
                resolver,
                true_child,
                false_child,
                taken=taken,
                runtime_condition=sp.false,
            )
            result_taint = self._if_merge_taint_conditions(
                operation,
                true_estimate=estimate if taken else None,
                false_estimate=estimate if not taken else None,
                predicate=predicate,
                runtime_guard=sp.false,
                conservative_guard=sp.false,
                taken=taken,
            )
            estimate = self._with_if_dependency_outputs(
                operation,
                resolver,
                true_estimate=estimate if taken else None,
                false_estimate=estimate if not taken else None,
                combined=estimate,
                taken=taken,
                runtime_condition=False,
                condition=predicate,
            )
            output_sizes = self._if_output_sizes(
                operation,
                resolver,
                true_child,
                false_child,
                true_estimate=estimate if taken else None,
                false_estimate=estimate if not taken else None,
                true_inputs=true_inputs,
                false_inputs=false_inputs,
                taken=taken,
                runtime_condition=False,
                condition=predicate,
                true_consumed=true_consumed,
                false_consumed=false_consumed,
            )
            estimate = dataclasses.replace(
                estimate,
                trace=_wrap_trace(
                    f"if[{'true' if taken else 'false'} branch]",
                    estimate.trace,
                ),
                _output_sizes=output_sizes,
                _input_sizes=true_inputs if taken else false_inputs,
                _has_output_summary=True,
                _measurement_taint_conditions=_merge_measurement_taint_conditions(
                    estimate._measurement_taint_conditions,
                    result_taint,
                ),
            )
            return _with_constraints(
                estimate,
                *self._if_boundary_array_constraints(
                    operation,
                    resolver,
                    true_child,
                    false_child,
                    taken=taken,
                    predicate=predicate,
                    conservative_guard=sp.false,
                ),
            )
        if runtime_guard is not sp.false:
            _require_uncontrolled_operation(operation, controls)
        conservative_control = conservative_guard is not sp.false
        true_active = (
            sp.true if conservative_control else _boolean_condition(compile_predicate)
        )
        false_active = (
            sp.true
            if conservative_control
            else _boolean_condition(sp.Not(compile_predicate))
        )
        with self._guarded_constraint_scope(true_active):
            true_estimate = self.eval_operations(
                operation.true_operations,
                true_child,
                controls=controls,
                initial_allocations=true_inputs,
            )
        with self._guarded_constraint_scope(false_active):
            false_estimate = self.eval_operations(
                operation.false_operations,
                false_child,
                controls=controls,
                initial_allocations=false_inputs,
            )
        self._publish_if_results(
            operation,
            resolver,
            true_child,
            false_child,
            taken=None,
            runtime_condition=runtime_guard,
            conservative_condition=uncertainty_guard,
        )
        result_taint = self._if_merge_taint_conditions(
            operation,
            true_estimate=true_estimate,
            false_estimate=false_estimate,
            predicate=compile_predicate,
            runtime_guard=runtime_guard,
            conservative_guard=uncertainty_guard,
            taken=None,
        )

        compile_combined = true_estimate.conditional(
            false_estimate,
            compile_predicate,
        )
        compile_combined = self._with_if_dependency_outputs(
            operation,
            resolver,
            true_estimate=true_estimate,
            false_estimate=false_estimate,
            combined=compile_combined,
            taken=None,
            runtime_condition=False,
            condition=compile_predicate,
        )
        compile_output_sizes = self._if_output_sizes(
            operation,
            resolver,
            true_child,
            false_child,
            true_estimate=true_estimate,
            false_estimate=false_estimate,
            true_inputs=true_inputs,
            false_inputs=false_inputs,
            taken=None,
            runtime_condition=False,
            condition=compile_predicate,
            true_consumed=true_consumed,
            false_consumed=false_consumed,
        )
        compile_input_sizes = _branch_owner_sizes(
            true_inputs,
            false_inputs,
            condition=compile_predicate,
            runtime_condition=False,
        )
        compile_combined = dataclasses.replace(
            compile_combined,
            _output_sizes=compile_output_sizes,
            _input_sizes=compile_input_sizes,
            _has_output_summary=True,
        )
        if note is not None:
            trace = compile_combined.trace
            if trace is not None:
                trace = dataclasses.replace(
                    trace,
                    assumptions=(*trace.assumptions, note),
                )
            compile_combined = dataclasses.replace(compile_combined, trace=trace)
            compile_combined = compile_combined._with_metadata(assumptions=(note,))

        conservative_combined = true_estimate.choice(false_estimate)
        conservative_combined = self._with_if_dependency_outputs(
            operation,
            resolver,
            true_estimate=true_estimate,
            false_estimate=false_estimate,
            combined=conservative_combined,
            taken=None,
            runtime_condition=True,
            condition=compile_predicate,
        )
        conservative_output_sizes = self._if_output_sizes(
            operation,
            resolver,
            true_child,
            false_child,
            true_estimate=true_estimate,
            false_estimate=false_estimate,
            true_inputs=true_inputs,
            false_inputs=false_inputs,
            taken=None,
            runtime_condition=True,
            condition=compile_predicate,
            true_consumed=true_consumed,
            false_consumed=false_consumed,
        )
        conservative_input_sizes = _branch_owner_sizes(
            true_inputs,
            false_inputs,
            condition=compile_predicate,
            runtime_condition=True,
        )
        conservative_reason = (
            "measurement-derived"
            if uncertainty_guard is sp.false
            else (
                "unresolved loop-state"
                if runtime_guard is sp.false
                else "measurement-derived or unresolved loop-state"
            )
        )
        conservative_assumption = ResourceAssumption(
            f"{conservative_reason} conditional resources are combined field "
            "by field across all possible branches",
            source="if",
        )
        conservative_combined = dataclasses.replace(
            conservative_combined,
            _output_sizes=conservative_output_sizes,
            _input_sizes=conservative_input_sizes,
            _has_output_summary=True,
        )
        conservative_combined = conservative_combined._with_metadata(
            assumptions=(conservative_assumption,),
            quality=EstimateQuality.CONSERVATIVE,
        )

        if conservative_guard is sp.true:
            conservative_combined = dataclasses.replace(
                conservative_combined,
                _measurement_taint_conditions=_merge_measurement_taint_conditions(
                    conservative_combined._measurement_taint_conditions,
                    result_taint,
                ),
            )
            return _with_constraints(
                conservative_combined,
                *self._if_boundary_array_constraints(
                    operation,
                    resolver,
                    true_child,
                    false_child,
                    taken=None,
                    predicate=compile_predicate,
                    conservative_guard=conservative_guard,
                ),
            )

        combined = conservative_combined.conditional(
            compile_combined,
            conservative_guard,
        )
        combined = dataclasses.replace(
            combined,
            _output_sizes=_conditional_resource_map(
                conservative_output_sizes,
                compile_output_sizes,
                conservative_guard,
            ),
            _input_sizes=_conditional_resource_map(
                conservative_input_sizes,
                compile_input_sizes,
                conservative_guard,
            ),
            _has_output_summary=True,
            _measurement_taint_conditions=_merge_measurement_taint_conditions(
                combined._measurement_taint_conditions,
                result_taint,
            ),
        )
        return _with_constraints(
            combined,
            *self._if_boundary_array_constraints(
                operation,
                resolver,
                true_child,
                false_child,
                taken=None,
                predicate=compile_predicate,
                conservative_guard=conservative_guard,
            ),
        )

    def _decide_branch(
        self, condition: sp.Basic
    ) -> tuple[bool | None, ResourceAssumption | None]:
        """Decide a branch condition from constants and supplied values.

        Substitutes any known classical parameter values into the resolved
        condition, then tests it for a definite truth value (nonzero is true).
        Records the names that participated in the predicate so downstream
        substitution reporting does not misfile them as no-ops, and, when the
        branch stays undecidable despite a supplied value, produces an
        assumption naming the unresolved symbols.

        Args:
            condition (sp.Basic): Resolved condition expression. May be a numeric
                ``Expr`` or a ``BooleanAtom`` (from a comparison predicate).

        Returns:
            tuple[bool | None, ResourceAssumption | None]: The branch decision
            (``True`` / ``False``, or ``None`` when undecidable) and an optional
            undecidable-branch assumption (only when a supplied value touched an
            undecidable condition).
        """
        original = condition
        used: set[str] = set()
        if self._run_state.condition_values and condition.free_symbols:
            subs: dict[Any, Any] = {}
            for symbol in condition.free_symbols:
                if isinstance(symbol, sp.Dummy):
                    continue
                name = _symbol_display_name(symbol)
                if name in self._run_state.condition_values:
                    subs[symbol] = self._run_state.condition_values[name]
                    used.add(name)
            if subs:
                condition = condition.subs(subs, simultaneous=True)
        if isinstance(condition, sp.logic.boolalg.BooleanAtom):
            decision: bool | None = bool(condition)
        elif condition.is_number:
            decision = bool(condition != 0)
        else:
            decision = None
        self._run_state.branch_condition_names |= used
        if decision is not None or not used:
            return decision, None
        separator: str = ", "
        unresolved = separator.join(
            sorted(_symbol_display_name(symbol) for symbol in condition.free_symbols)
        )
        message = (
            f"branch condition '{original}' is undecidable from the supplied "
            "values; conservative maximum of both branches used; "
            f"unresolved: {unresolved}"
        )
        if message in self._run_state.reported_undecidable:
            return None, None
        self._run_state.reported_undecidable.add(message)
        return None, ResourceAssumption(message, source="if")
