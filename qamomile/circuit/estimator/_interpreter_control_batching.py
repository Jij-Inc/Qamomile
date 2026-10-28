"""Profile controlled regions for resource estimation."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._aggregate_control_profile import (
    _clean_ancilla_aggregate_control_profile,
)
from qamomile.circuit.estimator._call_liveness import (
    _captured_quantum_allocations,
)
from qamomile.circuit.estimator._classical_provenance import (
    _classical_fact_uncertainty_condition,
    _loop_array_state_rebinds,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._control_decomposition import (
    static_clean_ancilla_batch_profile,
)
from qamomile.circuit.estimator._control_model import (
    _clean_ancilla_shared_ladder_condition,
    _EstimatorControlBatchProfile,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._gate_models import (
    _estimate_named_gate,
)
from qamomile.circuit.estimator._interpreter_core import (
    _CONCRETE_REGION_REPLAY_LIMIT,
    _InterpreterCore,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _refine_boolean_under_assumption,
    _ResourceInlineBoundaryOperation,
)
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    ControlDecomposition,
    EstimateQuality,
    ResourceExpr,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_conditions import _PhaseIdentity
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _ConditionIndicator,
    _expr,
    _resource_activity_condition,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.estimator._scopes import (
    _controlled_u_child_resolver,
    _inverse_block_child_resolver,
    _LocalBlock,
    _scalar_target_broadcast_factor,
    _select_case_child_resolver,
    _solve_affine_recurrence,
    _typed_value_symbol,
    build_for_loop_scope,
    build_if_scopes,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    InvokeOperation,
)
from qamomile.circuit.ir.operation.classical_ops import (
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder,
)


class _ControlBatchingInterpreter(_InterpreterCore):
    """Add shared-control-ladder profiling and projection."""

    @staticmethod
    def _control_batch_profile_requires_state(
        operations: Sequence[Operation],
    ) -> bool:
        """Return whether batching depends on sequential classical state.

        Most controlled bodies contain only quantum leaves and pure scalar
        expressions that the resolver can trace lazily. Array stores, inlined
        call boundaries, and classical control-flow results instead require
        program-order state publication before a later activity predicate can
        be classified.

        Args:
            operations (Sequence[Operation]): One sequential body to profile.

        Returns:
            bool: Whether a resource-neutral state prepass is required.
        """
        for operation in operations:
            if isinstance(operation, StoreArrayElementOperation):
                return True
            if (
                isinstance(operation, _ResourceInlineBoundaryOperation)
                and operation.array_state_bindings
            ):
                return True
            if isinstance(operation, IfOperation) and any(
                not merge.result.type.is_quantum() for merge in operation.iter_merges()
            ):
                return True
            if isinstance(operation, (ForOperation, ForItemsOperation)) and (
                operation.region_args
                or operation.loop_carried_rebinds
                or _loop_array_state_rebinds(operation)
            ):
                return True
            if isinstance(operation, WhileOperation) and (
                operation.region_args or operation.loop_carried_rebinds
            ):
                return True
        return False

    def _control_batch_profile_resolver(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
    ) -> ExprResolver:
        """Prepare resolved sequential state for control-work profiling.

        The normal interpreter is the sole owner of array, branch-merge, and
        loop-exit semantics. Reusing it as a zero-control prepass avoids a
        second, drifting state machine in the batching walker. The detached
        resolver preserves the caller state and all user-visible estimator
        metadata; opaque definition costs remain safely memoized.

        Args:
            operations (Sequence[Operation]): Sequential controlled body.
            resolver (ExprResolver): Resolver at the body entry.

        Returns:
            ExprResolver: Original resolver for stateless bodies, otherwise a
            detached resolver containing every program-order classical result.
        """
        if not self._control_batch_profile_requires_state(operations):
            return resolver
        body = _LocalBlock(list(operations))
        prepared = resolver.child_scope(inner_block=body)
        prepared.copy_array_context()
        with self._isolated_loop_taint_probe_state():
            self.eval_operations(
                list(operations),
                prepared,
                controls=_ZERO,
                initial_allocations=_captured_quantum_allocations(
                    operations,
                    prepared,
                    self._run_state.allocation_owners_by_uuid,
                ),
                allow_control_batching=False,
            )
        return prepared

    def _controlled_operation_batch_profile(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> _EstimatorControlBatchProfile:
        """Return the symbolic control-batching profile of one operation.

        The profile retains parameter-dependent activity rather than choosing
        a decomposition before public inputs are substituted. This keeps
        symbolic specialization and direct input specialization on the same
        fixed clean-ancilla resource-model path.

        Args:
            operation (Operation): Operation inside a controlled body.
            resolver (ExprResolver): Resolver for compile-time structure.

        Returns:
            _EstimatorControlBatchProfile: Capped symbolic work.
        """
        if isinstance(operation, _ResourceInlineBoundaryOperation):
            self._bind_resource_inline_array_states(operation, resolver)
            return _EstimatorControlBatchProfile()
        if isinstance(operation, StoreArrayElementOperation):
            # Classical array stores update the estimator's resolver state but
            # are not coherent quantum work.  They intentionally remain an
            # emitter error if compile-time lowering fails to remove them;
            # resource estimation runs earlier on the semantic IR.
            return _EstimatorControlBatchProfile()
        static_profile = static_clean_ancilla_batch_profile(operation)
        if static_profile is not None:
            return _EstimatorControlBatchProfile(work=static_profile.work)
        if isinstance(operation, GlobalPhaseOperation):
            phase = self._apply_condition_values(
                resolver.resolve(operation.phase),
                record_usage=False,
            )
            return _EstimatorControlBatchProfile(work=1).when(
                sp.Eq(_PhaseIdentity(phase), _ZERO)
            )
        if isinstance(operation, PauliEvolveOp):
            gamma = self._apply_condition_values(
                resolver.resolve(operation.gamma),
                record_usage=False,
            )
            return _EstimatorControlBatchProfile(work=2).when(
                sp.Ne(gamma, _ZERO),
            )
        if isinstance(operation, ControlledUOperation):
            power = self._apply_condition_values(
                resolver.resolve(operation.power),
                record_usage=False,
            )
            active = _boolean_condition(sp.Gt(power, _ZERO))
            if not isinstance(operation.block, Block):
                return _EstimatorControlBatchProfile()
            body_operands = operation.body_operands
            broadcast = self._apply_condition_values(
                _scalar_target_broadcast_factor(
                    operation.block,
                    [operand for operand in body_operands if operand.type.is_quantum()],
                    resolver,
                ),
                record_usage=False,
            )
            active = sp.And(active, sp.Gt(broadcast, _ZERO))
            if _boolean_condition(active) is sp.false:
                return _EstimatorControlBatchProfile()
            child = _controlled_u_child_resolver(operation, resolver)
            body_profile = self._controlled_body_batch_profile(
                operation.block.operations,
                child,
            )
            # Every valid ControlledU owns at least one local control. Once
            # the body is active, composing that control with an outer carrier
            # is itself enough to justify the shared path.
            return body_profile.when(active).as_shared_leaf()
        if isinstance(operation, ForOperation):
            if len(operation.operands) < 2:
                return _EstimatorControlBatchProfile()
            child, start, stop, step, loop_symbol = build_for_loop_scope(
                operation,
                resolver,
            )
            child.copy_array_context()
            start = self._apply_condition_values(start, record_usage=False)
            stop = self._apply_condition_values(stop, record_usage=False)
            step = self._apply_condition_values(step, record_usage=False)
            iterations = symbolic_iterations(start, stop, step)
            if iterations.is_zero is True:
                return _EstimatorControlBatchProfile()
            if operation.region_args:
                return self._controlled_region_for_batch_profile(
                    operation,
                    resolver,
                    start=start,
                    step=step,
                    iterations=iterations,
                    loop_symbol=loop_symbol,
                )
            return self._controlled_body_batch_profile(
                operation.operations,
                child,
            ).sum_over(
                loop_symbol,
                start,
                step,
                iterations,
            )
        if isinstance(operation, IfOperation):
            condition_fact = resolver.resolve_classical_fact(operation.condition)
            condition = self._apply_condition_values(
                cast(sp.Expr, condition_fact.value),
                record_usage=False,
            )
            true_child, false_child = build_if_scopes(operation, resolver)
            decision = self._peek_branch_decision(condition)
            if decision is True:
                return self._controlled_body_batch_profile(
                    operation.true_operations,
                    true_child,
                )
            if decision is False:
                return self._controlled_body_batch_profile(
                    operation.false_operations,
                    false_child,
                )
            true_profile = self._controlled_body_batch_profile(
                operation.true_operations,
                true_child,
            )
            false_profile = self._controlled_body_batch_profile(
                operation.false_operations,
                false_child,
            )
            uncertainty_guard = _classical_fact_uncertainty_condition(condition_fact)
            if uncertainty_guard is sp.false:
                return true_profile.conditional(
                    false_profile,
                    _boolean_condition(condition),
                )
            choice_profile = true_profile.choice(false_profile)
            if uncertainty_guard is sp.true:
                return choice_profile
            internal_symbols = (
                condition.free_symbols & self._run_state.unresolved_resource_symbols
            )
            representative = cast(
                sp.Expr,
                condition.xreplace({symbol: _ZERO for symbol in internal_symbols}),
            )
            compile_condition = _refine_boolean_under_assumption(
                _boolean_condition(representative),
                cast(Boolean, sp.Not(uncertainty_guard)),
            )
            compile_profile = true_profile.conditional(
                false_profile,
                compile_condition,
            )
            return choice_profile.conditional(
                compile_profile,
                uncertainty_guard,
            )
        if isinstance(operation, InvokeOperation):
            strategy = self._strategy_for(operation)
            selection = operation.select_body(strategy=strategy)
            body = selection.body
            if not isinstance(body, Block):
                opaque_cost = (
                    operation.definition.opaque_cost
                    if operation.definition is not None
                    else None
                )
                if opaque_cost is None:
                    return _EstimatorControlBatchProfile()
                context, transform = self._opaque_cost_context(
                    operation,
                    resolver,
                    controls=_ZERO,
                    strategy=strategy,
                )
                base_cost = self._resolve_opaque_definition_cost(
                    operation,
                    opaque_cost,
                    context,
                )
                profile = _clean_ancilla_aggregate_control_profile(base_cost)
                if transform.added_controls:
                    return profile.as_shared_leaf()
                return profile
            body_implements_controls = selection.implements_controls
            child = resolver.call_child_scope(
                operation,
                called_block=body,
                body_implements_transform=body_implements_controls,
                actual_operands=selection.operands,
            )
            body_identity = id(body)
            call_state = tuple(
                self._apply_condition_values(
                    child.resolve(formal),
                    record_usage=False,
                )
                for formal in body.input_values
                if not formal.type.is_quantum()
            )
            active_states = self._run_state.active_batch_profile_states.setdefault(
                body_identity,
                [],
            )
            repeated_state = call_state in active_states
            concrete_progress = bool(active_states) and any(
                current != previous and current.is_number and previous.is_number
                for current, previous in zip(
                    call_state,
                    active_states[-1],
                    strict=True,
                )
            )
            if repeated_state or (active_states and not concrete_progress):
                return _EstimatorControlBatchProfile(work=2)
            active_states.append(call_state)
            try:
                body_profile = self._controlled_body_batch_profile(
                    body.operations,
                    child,
                )
            finally:
                active_states.pop()
                if not active_states:
                    self._run_state.active_batch_profile_states.pop(body_identity, None)
            own_controls = len(operation.operands) - len(selection.operands)
            if own_controls:
                return body_profile.as_shared_leaf()
            return body_profile
        if isinstance(operation, SelectOperation):
            case_profiles: list[_EstimatorControlBatchProfile] = []
            for case_block in operation.case_blocks:
                child = _select_case_child_resolver(
                    operation,
                    case_block,
                    resolver,
                )
                broadcast = self._apply_condition_values(
                    _scalar_target_broadcast_factor(
                        case_block,
                        operation.target_operands,
                        resolver,
                    ),
                    record_usage=False,
                )
                case_active = _boolean_condition(sp.Gt(broadcast, _ZERO))
                if case_active is sp.false:
                    continue
                case_profiles.append(
                    self._controlled_body_batch_profile(
                        case_block.operations,
                        child,
                    ).when(case_active)
                )
            # SELECT contributes at least one index control to every active
            # case, so one active leaf already benefits from composing that
            # control with the outer carrier.
            return _EstimatorControlBatchProfile.combine(case_profiles).as_shared_leaf()
        if isinstance(operation, InverseBlockOperation):
            if not isinstance(operation.implementation_block, Block):
                return _EstimatorControlBatchProfile()
            child = _inverse_block_child_resolver(operation, resolver)
            body_profile = self._controlled_body_batch_profile(
                operation.implementation_block.operations,
                child,
            )
            if operation.num_control_qubits:
                return body_profile.as_shared_leaf()
            return body_profile
        return _EstimatorControlBatchProfile(work=1)

    def _controlled_region_for_batch_profile(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        *,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
        loop_symbol: sp.Symbol,
    ) -> _EstimatorControlBatchProfile:
        """Build a batching profile with loop-carried values in scope.

        Args:
            operation (ForOperation): Loop carrying explicit region values.
            resolver (ExprResolver): Enclosing expression resolver.
            start (ResourceExpr): First Python-range value.
            step (ResourceExpr): Python-range step.
            iterations (ResourceExpr): Number of executed iterations.
            loop_symbol (sp.Symbol): Symbolic loop induction variable.

        Returns:
            _EstimatorControlBatchProfile: Exact affine-carry profile, or a
            conservative nonempty-loop profile for unsupported recurrences.
        """
        concrete_profile = self._concrete_region_batch_profile(
            operation,
            resolver,
            start=start,
            step=step,
            iterations=iterations,
            maximum_iterations=_CONCRETE_REGION_REPLAY_LIMIT,
        )
        if concrete_profile is not None:
            return concrete_profile
        carry_symbols = {
            arg.block_arg.uuid: _typed_value_symbol(
                arg.block_arg,
                f"{arg.var_name}_batch_carry",
                fresh=True,
            )
            for arg in operation.region_args
        }
        probe_context: dict[str, sp.Expr] = dict(carry_symbols)
        if operation.loop_var_value is not None:
            probe_context[operation.loop_var_value.uuid] = loop_symbol
        body = _LocalBlock(operation.operations)
        probe = resolver.child_scope(
            inner_block=body,
            extra_context=probe_context,
            extra_loop_vars={operation.loop_var: loop_symbol},
        )
        probe.copy_array_context()
        with self._isolated_loop_taint_probe_state():
            self.eval_operations(
                operation.operations,
                probe,
                controls=_ZERO,
                initial_allocations=_captured_quantum_allocations(
                    operation.operations,
                    probe,
                    self._run_state.allocation_owners_by_uuid,
                ),
                allow_control_batching=False,
            )

        at_iteration: dict[str, sp.Expr] = {}
        all_carry_symbols = set(carry_symbols.values())
        for arg in operation.region_args:
            init = self._apply_condition_values(
                resolver.resolve(arg.init),
                record_usage=False,
            )
            yielded = self._apply_condition_values(
                probe.resolve(arg.yielded),
                record_usage=False,
            )
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
                return _EstimatorControlBatchProfile(work=2).when(
                    sp.Gt(iterations, _ZERO)
                )
            at_iteration[arg.block_arg.uuid] = cast(sp.Expr, recurrence[0])

        body_context = dict(at_iteration)
        if operation.loop_var_value is not None:
            body_context[operation.loop_var_value.uuid] = loop_symbol
        child = resolver.child_scope(
            inner_block=body,
            extra_context=body_context,
            extra_loop_vars={operation.loop_var: loop_symbol},
        )
        child.copy_array_context()
        return self._controlled_body_batch_profile(
            operation.operations,
            child,
        ).sum_over(
            loop_symbol,
            start,
            step,
            iterations,
        )

    def _concrete_region_batch_profile(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        *,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
        maximum_iterations: int,
    ) -> _EstimatorControlBatchProfile | None:
        """Replay a concrete loop carry for the control-batching decision.

        Args:
            operation (ForOperation): Loop carrying explicit region values.
            resolver (ExprResolver): Enclosing expression resolver.
            start (ResourceExpr): First Python-range value.
            step (ResourceExpr): Python-range step.
            iterations (ResourceExpr): Number of executed iterations.
            maximum_iterations (int): Largest concrete range to replay.

        Returns:
            _EstimatorControlBatchProfile | None: Exact combined profile, or
            ``None`` while any range value remains symbolic or exceeds the
            requested replay limit.
        """
        concrete_values = tuple(
            self._concrete_scalar(value) for value in (start, step, iterations)
        )
        if any(value is None for value in concrete_values):
            return None
        concrete_start, concrete_step, iteration_count = cast(
            tuple[int, int, int],
            concrete_values,
        )
        if concrete_step == 0 or iteration_count < 0:
            return None
        if iteration_count > maximum_iterations:
            return None
        carried = {
            arg.block_arg.uuid: self._apply_condition_values(
                resolver.resolve(arg.init),
                record_usage=False,
            )
            for arg in operation.region_args
        }
        profiles: list[_EstimatorControlBatchProfile] = []
        body = _LocalBlock(operation.operations)
        profile_parent = resolver.child_scope(inner_block=body)
        profile_parent.copy_array_context()
        for offset in range(iteration_count):
            loop_value = sp.Integer(concrete_start + concrete_step * offset)
            context = dict(carried)
            if operation.loop_var_value is not None:
                context[operation.loop_var_value.uuid] = loop_value
            child = profile_parent.child_scope(
                inner_block=body,
                extra_context=context,
                extra_loop_vars={operation.loop_var: loop_value},
            )
            with self._isolated_loop_taint_probe_state():
                self.eval_operations(
                    operation.operations,
                    child,
                    controls=_ZERO,
                    initial_allocations=_captured_quantum_allocations(
                        operation.operations,
                        child,
                        self._run_state.allocation_owners_by_uuid,
                    ),
                    allow_control_batching=False,
                )
            profiles.append(
                self._controlled_body_batch_profile(
                    operation.operations,
                    child,
                )
            )
            combined = _EstimatorControlBatchProfile.combine(profiles)
            if combined.has_multiple_work is sp.true:
                return combined
            carried = {
                arg.block_arg.uuid: self._apply_condition_values(
                    child.resolve(arg.yielded),
                    record_usage=False,
                )
                for arg in operation.region_args
            }
        return _EstimatorControlBatchProfile.combine(profiles)

    def _controlled_body_batch_profile(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
    ) -> _EstimatorControlBatchProfile:
        """Return a controlled body's recursively resolved batching profile.

        Args:
            operations (Sequence[Operation]): Controlled body operations.
            resolver (ExprResolver): Resolver for compile-time structure.

        Returns:
            _EstimatorControlBatchProfile: Capped symbolic work.
        """
        resolver = self._control_batch_profile_resolver(operations, resolver)
        return _EstimatorControlBatchProfile.combine(
            self._controlled_operation_batch_profile(operation, resolver)
            for operation in operations
        )

    def _controlled_body_batch_condition(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
        controls: ResourceExpr,
    ) -> sp.Basic:
        """Return when the selected control model uses one shared ladder.

        Args:
            operations (Sequence[Operation]): Controlled body operations.
            resolver (ExprResolver): Resolver for compile-time structure.
            controls (ResourceExpr): Number of surrounding controls.

        Returns:
            sp.Basic: Symbolic predicate defined by the selected fixed
                resource model after concrete specialization.
        """
        if (
            self.config.control_decomposition
            is not ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
        ):
            return sp.false
        control_count = self._apply_condition_values(
            _expr(controls),
            record_usage=False,
        )
        if control_count.is_zero is True or control_count == _ONE:
            return sp.false
        profile = self._controlled_body_batch_profile(operations, resolver)
        return _clean_ancilla_shared_ladder_condition(
            control_count,
            profile,
        )

    def _peek_branch_decision(self, condition: sp.Basic) -> bool | None:
        """Inspect a compile-time branch without mutating estimator metadata.

        Args:
            condition (sp.Basic): Resolved condition expression.

        Returns:
            bool | None: Definite branch value, or ``None`` when unresolved.
        """
        if self._run_state.condition_values and condition.free_symbols:
            substitutions: dict[Any, Any] = {
                symbol: self._run_state.condition_values[_symbol_display_name(symbol)]
                for symbol in condition.free_symbols
                if not isinstance(symbol, sp.Dummy)
                and _symbol_display_name(symbol) in self._run_state.condition_values
            }
            if substitutions:
                condition = condition.subs(substitutions, simultaneous=True)
        if isinstance(condition, sp.logic.boolalg.BooleanAtom):
            return bool(condition)
        if condition.is_number:
            return bool(condition != 0)
        return None

    def _with_shared_control_ladder(
        self,
        body: ResourceEstimate,
        controls: ResourceExpr,
    ) -> ResourceEstimate:
        """Wrap a singly controlled body in one compute/uncompute ladder.

        Args:
            body (ResourceEstimate): Body already estimated under one control.
            controls (ResourceExpr): Original concrete control count.

        Returns:
            ResourceEstimate: Clean-ancilla Toffoli cost with concurrently
            held clean ancillas.
        """
        recipe = clean_ancilla_toffoli_ladder(controls)
        outer_clean_ancillas = recipe.clean_ancillas
        toffoli = _estimate_named_gate(
            "toffoli",
            _ZERO,
            control_decomposition=self.config.control_decomposition,
        )
        ladder = toffoli.repeat(recipe.total_toffolis)
        compute_depth = recipe.compute_toffolis * toffoli.depth.depth
        estimate = ladder.seq(body)
        trace_children = tuple(
            node
            for node in (toffoli.trace, body.trace, toffoli.trace)
            if node is not None
        )
        reason = ResourceAssumption(
            message=(
                "clean-ancilla controlled region uses one shared conjunction "
                "ladder across the modeled body; a different decomposition "
                "may use fewer gates, depth, or ancillas"
            ),
            source="clean-ancilla shared control ladder",
        )
        return dataclasses.replace(
            estimate,
            width=dataclasses.replace(
                body.width,
                clean_ancilla_qubits=(
                    body.width.clean_ancilla_qubits + outer_clean_ancillas
                ),
                peak_qubits=body.width.peak_qubits + outer_clean_ancillas,
            ),
            trace=(
                ResourceTraceNode(
                    name=f"shared_control_ladder({controls})",
                    source_kind="clean_ancilla_toffoli",
                    summary=(
                        f"toffoli_steps={recipe.total_toffolis}, "
                        f"clean_ancillas={outer_clean_ancillas}"
                    ),
                    children=trace_children,
                )
                if self.config.trace
                else None
            ),
            _output_sizes=body._output_sizes,
            _input_sizes=body._input_sizes,
            _has_output_summary=body._has_output_summary,
            _dependency_keys=body._dependency_keys,
            _dependency_reads=body._dependency_reads,
            _dependency_writes=body._dependency_writes,
            _dependency_completion=(
                {
                    key: cast(
                        ResourceExpr,
                        _ConditionIndicator(_resource_activity_condition(completion))
                        * (compute_depth + completion),
                    )
                    for key, completion in (body._dependency_completion or {}).items()
                }
                if body._dependency_completion is not None
                else None
            ),
        )._with_metadata(
            assumptions=(reason,),
            quality=EstimateQuality.CONSERVATIVE,
        )
