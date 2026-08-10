"""Interpret callable and opaque resource operations."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._aggregate_control_profile import (
    _aggregate_zero_gate_residual_condition,
)
from qamomile.circuit.estimator._allocation_width import (
    _namespace_allocation_sites,
)
from qamomile.circuit.estimator._call_liveness import (
    _block_input_allocations,
    _invoke_quantum_output_sizes,
)
from qamomile.circuit.estimator._classical_provenance import (
    _value_taint_condition,
)
from qamomile.circuit.estimator._config import (
    UnknownResourcePolicy,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._constraints import (
    _block_output_constraints,
    _quantum_operand_width_constraints,
)
from qamomile.circuit.estimator._dependency_call_mapping import (
    _map_body_dependency_completion,
    _map_body_dependency_keys,
)
from qamomile.circuit.estimator._dependency_footprints import _wire_keys_for_values
from qamomile.circuit.estimator._dependency_metadata import (
    _complete_dependency_completion,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._estimate_provenance import (
    _DEFER_RESOURCE_SYMBOL_METADATA,
)
from qamomile.circuit.estimator._estimate_validation import (
    _estimate_activity,
    _require_unitary_resource_estimate,
    _with_constraints,
)
from qamomile.circuit.estimator._input_contract import (
    _opaque_cost_target_shapes,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _publish_invoke_classical_results,
)
from qamomile.circuit.estimator._interpreter_for_items import (
    _ForItemsInterpreter,
)
from qamomile.circuit.estimator._measurement_provenance import (
    _merge_measurement_taint_conditions,
)
from qamomile.circuit.estimator._opaque import (
    OpaqueCostContext,
    _opaque_call_relative_width,
    _OpaqueInvocationTransform,
    _validate_opaque_cost_provenance,
    _zero_control_count,
)
from qamomile.circuit.estimator._product_formula import (
    _apply_product_formula_contract,
    _require_concrete_product_formula_structure,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resolver_indices import (
    _resolve_concrete_array_payload,
)
from qamomile.circuit.estimator._resource_algebra import (
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _ConditionIndicator,
    _expr,
    _resource_activity_condition,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.estimator._runtime_observation import (
    _block_runtime_observation_summary,
)
from qamomile.circuit.estimator._scheduling import (
    _estimate_has_nonzero_depth,
    _with_body_boundary_depth_metadata,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableBodySelection,
    InvokeOperation,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    ValueBase,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


class _CallInterpreter(_ForItemsInterpreter):
    """Add callable-body, opaque-cost, and invocation handlers."""

    def _eval_call_body(
        self,
        block: Block,
        child: ExprResolver,
        actual_operands: Sequence[ValueBase],
        *,
        controls: ResourceExpr | int,
        callable_attrs: Mapping[str, Any] | None = None,
        contract_operands: Sequence[ValueBase] | None = None,
        contract_resolver: ExprResolver | None = None,
        source: str | None = None,
    ) -> ResourceEstimate:
        """Evaluate a body after remapping caller measurement provenance.

        Every body-backed call boundary uses this helper so a classical actual
        derived from measurement taints the corresponding callee formal UUID.
        This makes nested runtime conditionals retain their feed-forward
        semantics instead of being mistaken for compile-time symbolic choices.

        Args:
            block (Block): Callee implementation block to evaluate.
            child (ExprResolver): Resolver with formal values bound to the
                call-site actual operands.
            actual_operands (Sequence[ValueBase]): Call-site operands after
                wrapper-only coherent controls have been removed, grouped by
                quantum operands followed by classical/object operands.
            controls (ResourceExpr | int): Coherent controls surrounding every
                quantum operation in the body.
            callable_attrs (Mapping[str, Any] | None): Resource metadata for
                the callable boundary. Defaults to ``None``.
            contract_operands (Sequence[ValueBase] | None): Caller-scope ABI
                operands referenced by ``callable_attrs``. Defaults to the
                body operands.
            contract_resolver (ExprResolver | None): Caller-scope resolver for
                computed structural operands. Defaults to ``None``.
            source (str | None): Callable name used in contract diagnostics.
                Defaults to the block name.

        Returns:
            ResourceEstimate: Body estimate with measurement provenance
            preserved across the call boundary.

        Raises:
            ValueError: If recursive expansion repeats a resolved call state,
                changes only symbolically, exhausts Python's call stack before
                reaching a base case, or a product-formula structure operand
                remains unresolved.
            NotImplementedError: If the selected body contains legacy scalar
                Bit state that cannot flow between loop iterations.
        """
        active_attrs = callable_attrs or {}
        active_contract_operands = (
            contract_operands if contract_operands is not None else actual_operands
        )
        if active_attrs:
            _require_concrete_product_formula_structure(
                active_attrs,
                active_contract_operands,
                bindings={
                    **self.bindings,
                    **self._run_state.condition_values,
                },
                resolver=contract_resolver,
                specialize=lambda expression: self._apply_condition_values(
                    expression,
                    record_usage=False,
                ),
                source=source or block.name or "qkernel",
            )
        block_identity = id(block)
        resolved_classical_inputs = tuple(
            (
                formal,
                self._apply_condition_values(
                    child.resolve(formal),
                    record_usage=False,
                ),
            )
            for formal in block.input_values
            if not formal.type.is_quantum()
        )
        call_state = tuple(
            [value for _formal, value in resolved_classical_inputs] + [_expr(controls)]
        )
        local_bindings: dict[str, Any] = {}
        for formal, value in resolved_classical_inputs:
            if value not in (sp.true, sp.false) and not value.is_number:
                continue
            local_bindings[formal.name] = value
            parameter_name = formal.parameter_name()
            if parameter_name is not None:
                local_bindings[parameter_name] = value
        self._validate_legacy_scalar_bit_rebinds(
            block,
            local_bindings=local_bindings,
        )
        active_states = self._run_state.active_call_states.setdefault(
            block_identity, []
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
            name = block.name or "qkernel"
            raise ValueError(
                f"Recursive resource estimation for '{name}' did not reach "
                "a base case. Supply a concrete recursion-driving value in "
                "inputs, or replace the recursion with a bounded loop."
            )
        previous_taint = self._run_state.measurement_taint_conditions
        previous_bindings = self.bindings
        active_states.append(call_state)
        try:
            tainted_formals = {
                formal.uuid: condition
                for formal, actual in pair_block_operands(block, actual_operands)
                if (
                    condition := _value_taint_condition(
                        actual,
                        self._run_state.measurement_taint_conditions,
                    )
                )
                is not sp.false
            }
            parameter_operands: list[Any] = []
            for operand in actual_operands:
                if not (operand.type.is_classical() or operand.type.is_object()):
                    continue
                resolved_operand: Any = operand
                if isinstance(operand, ArrayValue):
                    payload = _resolve_concrete_array_payload(
                        operand,
                        {
                            **previous_bindings,
                            **self._run_state.condition_values,
                        },
                        resolve_expression=(
                            contract_resolver.resolve
                            if contract_resolver is not None
                            else None
                        ),
                        specialize=lambda expression: self._apply_condition_values(
                            expression,
                            record_usage=False,
                        ),
                        source=source or block.name or "qkernel call",
                    )
                    if payload is not None:
                        resolved_operand = payload
                parameter_operands.append(resolved_operand)
            self.bindings = ValueResolver().bind_block_params(
                block,
                parameter_operands,
                dict(previous_bindings),
            )
            self._run_state.measurement_taint_conditions = (
                _merge_measurement_taint_conditions(
                    previous_taint,
                    tainted_formals,
                )
            )
            estimate = self.eval_operations(
                block.operations,
                child,
                controls=controls,
                initial_allocations=_block_input_allocations(block, child),
            )
            return _with_constraints(
                estimate,
                *_block_output_constraints(
                    block,
                    child,
                    active_when=self._run_state.constraint_scope_condition,
                    proven_cache=self._run_state.array_constraint_proven,
                ),
            )
        except RecursionError as error:
            name = block.name or "qkernel"
            raise ValueError(
                f"Recursive resource estimation for '{name}' exhausted the "
                "Python call stack before reaching a base case. Supply a "
                "concrete recursion-driving value in inputs, or replace the "
                "recursion with a bounded loop."
            ) from error
        finally:
            self.bindings = previous_bindings
            self._run_state.measurement_taint_conditions = previous_taint
            active_states.pop()
            if not active_states:
                self._run_state.active_call_states.pop(block_identity, None)

    def eval_invoke(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a callable invocation from its selected implementation.

        Args:
            operation (InvokeOperation): Callable invocation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Invocation resource estimate.

        Raises:
            ValueError: If the invocation has neither an implementation body nor
                an explicit opaque cost.
        """
        callable_attrs = {
            **(operation.definition.attrs if operation.definition is not None else {}),
            **operation.attrs,
        }
        width_constraints = _quantum_operand_width_constraints(
            callable_attrs,
            operation.target_qubits,
            resolver,
            source=operation.custom_name,
        )
        strategy = self._strategy_for(operation)
        selection = operation.select_body(strategy=strategy)
        if isinstance(selection.body, Block):
            return _with_constraints(
                self._estimate_invoke_body(
                    operation,
                    selection,
                    resolver,
                    controls,
                ),
                *width_constraints,
            )
        cost_context, invocation_transform = self._opaque_cost_context(
            operation,
            resolver,
            controls=_expr(controls),
            strategy=strategy,
        )
        opaque_cost = (
            operation.definition.opaque_cost
            if operation.definition is not None
            else None
        )
        if opaque_cost is not None:
            return _with_constraints(
                self._estimate_opaque_cost(
                    operation,
                    opaque_cost,
                    cost_context,
                    invocation_transform,
                ),
                *width_constraints,
            )
        return _with_constraints(
            self._handle_unknown_invoke(operation, invocation_transform),
            *width_constraints,
        )

    def _strategy_for(self, operation: InvokeOperation) -> str | None:
        """Return the selected strategy for an invocation.

        Args:
            operation (InvokeOperation): Invocation to inspect.

        Returns:
            str | None: Strategy name from call attrs or estimator config.
        """
        names = (
            operation.custom_name,
            operation.target.name,
            operation.gate_type.value,
        )
        for name in names:
            if name in self.config.strategies:
                return self.config.strategies[name]
        return operation.strategy_name

    def _opaque_cost_context(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr,
        strategy: str | None,
    ) -> tuple[OpaqueCostContext, _OpaqueInvocationTransform]:
        """Build definition-cost and private transform contexts.

        Args:
            operation (InvokeOperation): Invocation operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr): Surrounding control count.
            strategy (str | None): Selected strategy.

        Returns:
            tuple[OpaqueCostContext, _OpaqueInvocationTransform]: Public
            definition-level callback inputs and private call-site transforms.
        """
        is_oracle = operation.attrs.get("kind") == "oracle"
        declared_controls = operation.num_declared_control_qubits if is_oracle else 0
        added_controls = (
            operation.num_added_control_qubits
            if is_oracle
            else operation.num_control_qubits
            if operation.transform.is_controlled
            else 0
        )
        cost_context = OpaqueCostContext(
            callable_name=operation.custom_name,
            target_shapes=_opaque_cost_target_shapes(
                operation,
                resolver,
                added_controls=added_controls,
                declared_controls=declared_controls,
            ),
            definition_control_qubits=declared_controls,
            strategy=strategy,
            control_decomposition=self.config.control_decomposition,
        )
        invocation_transform = _OpaqueInvocationTransform(
            declared_controls=declared_controls,
            added_controls=added_controls,
            inherited_controls=controls,
            inverse=operation.transform.is_inverse,
            control_value=operation.control_value,
        )
        return cost_context, invocation_transform

    def _estimate_opaque_cost(
        self,
        operation: InvokeOperation,
        cost: Any,
        context: OpaqueCostContext,
        transform: _OpaqueInvocationTransform,
    ) -> ResourceEstimate:
        """Evaluate an explicit cost attached to a bodyless callable.

        Fixed ``ResourceEstimate`` costs and callbacks share one contract:
        each describes one ordinary application of the callable definition.
        For an Oracle, that base cost includes its declared controls. The
        cost author also includes any phase-relevant work that must remain
        visible under later coherent controls. The estimator then applies
        inversion, controls added with ``qmc.control``, and controls inherited
        from an outer controlled qkernel without adding hidden global-phase
        overhead.

        Args:
            operation (InvokeOperation): Invocation operation.
            cost (Any): ``ResourceEstimate`` or callable accepting
                ``context``.
            context (OpaqueCostContext): Definition-level callback inputs.
            transform (_OpaqueInvocationTransform): Private call-site
                transforms applied after the base cost is obtained.

        Returns:
            ResourceEstimate: Explicit opaque estimate.

        Raises:
            TypeError: If ``cost`` is neither a ``ResourceEstimate`` nor a
                callable returning one.
            ValueError: If the opaque cost reports control-sensitive resources
                for a different control decomposition, or if a non-unitary
                cost is controlled or inverted.
        """
        estimate = self._resolve_opaque_definition_cost(
            operation,
            cost,
            context,
        )
        # Root input width and caller-scoped dependency, liveness, and
        # observation maps belong to the qkernel that produced an aggregate
        # cost, not to this opaque call site. The caller derives that boundary
        # information from this InvokeOperation instead.
        relative_width, anonymous_workspace = _opaque_call_relative_width(
            estimate.width
        )
        estimate = _namespace_allocation_sites(
            dataclasses.replace(
                estimate,
                width=relative_width,
                _output_sizes={},
                _input_sizes={},
                _has_output_summary=False,
                _dependency_keys=None,
                _dependency_reads=None,
                _dependency_writes=None,
                _measurement_taint_conditions={},
            ),
            operation,
        )
        if anonymous_workspace != _ZERO:
            assumption = ResourceAssumption(
                "opaque peak width exceeds its categorized workspace; the "
                "residual is treated as anonymous allocated workspace",
                source=operation.custom_name,
            )
            estimate = estimate._with_metadata(
                assumptions=(assumption,),
                active_when=sp.Gt(anonymous_workspace, _ZERO),
            )
        if transform.declared_controls:
            _require_unitary_resource_estimate(
                estimate,
                transform="use as a coherently controlled Oracle",
            )
        if transform.inverse:
            estimate = estimate.inverse()
        if transform.external_controls != _ZERO:
            estimate = estimate.controlled(transform.external_controls)
        if transform.local_controls:
            bracket_activity = _safe_simplify(
                _estimate_activity(estimate)
                + cast(
                    ResourceExpr,
                    _ConditionIndicator(
                        _aggregate_zero_gate_residual_condition(estimate)
                    ),
                )
            )
            estimate = self._with_zero_control_bracket(
                estimate,
                zero_controls=transform.zero_controls,
                active_when=bracket_activity,
            )
        measurement_condition = _boolean_condition(
            sp.Or(
                _resource_activity_condition(estimate.measurements.total),
                _resource_activity_condition(estimate.depth.measurement_depth),
            )
        )
        # An aggregate cost cannot say which classical result carries an
        # observed value. Conservatively taint every classical result while
        # retaining the call's ordinary operand-local depth footprint.
        measurement_outputs = {
            result.uuid: measurement_condition
            for result in operation.results
            if not result.type.is_quantum() and measurement_condition is not sp.false
        }
        estimate = dataclasses.replace(
            estimate,
            _measurement_taint_conditions=measurement_outputs,
        )
        has_quantum_endpoint = any(
            isinstance(value, ValueBase) and value.type.is_quantum()
            for value in (*operation.all_input_values(), *operation.results)
        )
        if _estimate_has_nonzero_depth(estimate) and not has_quantum_endpoint:
            raise ValueError(
                f"nonzero-depth opaque callable '{operation.custom_name}' has "
                "no quantum operand on which to place its scheduling dependency"
            )
        estimate = dataclasses.replace(
            estimate,
            trace=_wrap_trace(
                operation.custom_name,
                estimate.trace,
                source_kind="opaque_cost",
                strategy=context.strategy,
            ),
        )
        return estimate._with_metadata(derivation=EstimateDerivation.MODELED)

    def _resolve_opaque_definition_cost(
        self,
        operation: InvokeOperation,
        cost: Any,
        context: OpaqueCostContext,
    ) -> ResourceEstimate:
        """Resolve and cache one opaque callable's definition-level cost.

        Args:
            operation (InvokeOperation): Invocation whose definition is priced.
            cost (Any): Fixed estimate or callback accepting the context.
            context (OpaqueCostContext): Definition-level callback inputs.

        Returns:
            ResourceEstimate: Validated and normalized base estimate before
                inverse or added/inherited controls are applied.

        Raises:
            TypeError: If cost is neither a ResourceEstimate nor a callback
                returning one.
            ValueError: If model provenance is incompatible with the active
                estimator configuration.
        """
        cache_key = (
            id(operation),
            context.callable_name,
            tuple(context.target_shapes.items()),
            context.definition_control_qubits,
            context.strategy,
            context.control_decomposition,
        )
        cached = self._run_state.opaque_definition_cost_cache.get(cache_key)
        if cached is not None and cached[0] is operation:
            return cached[1]
        if isinstance(cost, ResourceEstimate):
            estimate = cost
        elif callable(cost):
            defer_token = _DEFER_RESOURCE_SYMBOL_METADATA.set(False)
            try:
                estimate = cost(context)
            finally:
                _DEFER_RESOURCE_SYMBOL_METADATA.reset(defer_token)
            if not isinstance(estimate, ResourceEstimate):
                raise TypeError(
                    f"Opaque cost for '{operation.custom_name}' must return "
                    "ResourceEstimate."
                )
        else:
            raise TypeError(
                f"Opaque cost for '{operation.custom_name}' must be a "
                "ResourceEstimate or callable."
            )
        _validate_opaque_cost_provenance(
            estimate,
            name=operation.custom_name,
            control_decomposition=self.config.control_decomposition,
        )
        # Public resource dataclasses accept ordinary Python numeric values.
        # Compose one base application through the resource algebra so every
        # field is normalized before dependency scheduling inspects SymPy
        # predicates.
        normalized = estimate.repeat(_ONE)
        self._run_state.opaque_definition_cost_cache[cache_key] = (
            operation,
            normalized,
        )
        return normalized

    def _estimate_invoke_body(
        self,
        operation: InvokeOperation,
        selection: CallableBodySelection,
        resolver: ExprResolver,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Estimate an invocation by traversing its body.

        For an ordinary body selected by a controlled invocation, add the
        invocation's controls to the surrounding controls before traversing
        its primitives. A transform-specific implementation already contains
        those controls, but a non-default activation value still contributes
        the invocation-level X bracket emitted around that body.

        Args:
            operation (InvokeOperation): Invocation operation.
            selection (CallableBodySelection): Validated callable body and
                call-site values aligned to its ABI.
            resolver (ExprResolver): Call-site resolver.
            controls (ResourceExpr | int): Surrounding control count.

        Returns:
            ResourceEstimate: Body-derived estimate.

        Raises:
            TypeError: If ``selection`` does not contain an IR body.
        """
        body = selection.body
        if not isinstance(body, Block):
            raise TypeError("_estimate_invoke_body requires a selected IR body.")
        realized_transform = selection.realized_transform
        body_implements_controls = selection.implements_controls
        body_implements_inverse = realized_transform.is_inverse
        child = resolver.call_child_scope(
            operation,
            called_block=body,
            body_implements_transform=body_implements_controls,
            actual_operands=selection.operands,
        )
        local_controls = (
            operation.num_control_qubits if operation.transform.is_controlled else 0
        )
        body_added_controls = len(operation.operands) - len(selection.operands)
        total_controls = _expr(controls) + body_added_controls
        actual_operands = selection.operands
        callable_attrs = {
            **(operation.definition.attrs if operation.definition is not None else {}),
            **operation.attrs,
        }
        body_estimate = self._eval_call_body(
            body,
            child,
            actual_operands,
            controls=total_controls,
            callable_attrs=callable_attrs,
            contract_operands=operation.operands[
                operation.num_body_external_control_qubits :
            ],
            contract_resolver=resolver,
            source=operation.custom_name,
        )
        body_estimate = _apply_product_formula_contract(
            body_estimate,
            callable_attrs,
            operation.operands[operation.num_body_external_control_qubits :],
            resolver,
            bindings=self.bindings,
            specialize=lambda expression: self._apply_condition_values(
                expression,
                record_usage=False,
            ),
            source=operation.custom_name,
        )
        _publish_invoke_classical_results(
            body.output_values,
            selection.results,
            child,
            resolver,
        )
        body_dependency_estimate = body_estimate
        if operation.transform.is_inverse and not body_implements_inverse:
            body_estimate = body_estimate.inverse()
        zero_controls = _ZERO
        if local_controls:
            zero_controls = _zero_control_count(
                local_controls,
                operation.control_value,
            )
            body_estimate = self._with_zero_control_bracket(
                body_estimate,
                zero_controls=zero_controls,
                active_when=_estimate_activity(body_estimate),
            )
        caller_results: Sequence[ValueBase] = selection.results
        dependency_keys = _map_body_dependency_keys(
            body,
            body_dependency_estimate,
            actual_operands,
            caller_results,
            resolver,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        dependency_completion = _map_body_dependency_completion(
            body,
            body_estimate,
            actual_operands,
            caller_results,
            resolver,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        if dependency_keys is not None:
            mapped_keys = set(dependency_keys)
            if local_controls and _estimate_has_nonzero_depth(body_estimate):
                mapped_keys.update(
                    _wire_keys_for_values(
                        operation.operands[: operation.num_control_qubits],
                        resolver,
                        scalar_values=self._run_state.condition_values,
                        used_names=self._run_state.branch_condition_names,
                    )
                )
            body_estimate = dataclasses.replace(
                body_estimate,
                _dependency_keys=frozenset(mapped_keys),
                _dependency_reads=frozenset(mapped_keys),
                _dependency_writes=frozenset(mapped_keys),
                _dependency_completion=_complete_dependency_completion(
                    dependency_completion,
                    mapped_keys,
                    fallback_depth=body_estimate.depth.depth,
                ),
            )
        input_sizes, output_sizes, has_output_summary = _invoke_quantum_output_sizes(
            operation,
            body,
            child,
            resolver,
            body_external_control_qubits=(
                len(operation.results) - len(selection.results)
            ),
            body_final_live=body_dependency_estimate._output_sizes,
            actual_operands=actual_operands,
            allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
        )
        estimate = _namespace_allocation_sites(
            dataclasses.replace(
                body_estimate,
                trace=_wrap_trace(
                    operation.custom_name,
                    body_estimate.trace,
                    source_kind="body",
                ),
                _output_sizes=output_sizes,
                _input_sizes=input_sizes,
                _has_output_summary=has_output_summary,
            ),
            operation,
        )
        estimate = _with_body_boundary_depth_metadata(
            estimate,
            estimate,
            source=operation.custom_name,
            zero_controls=zero_controls,
        )
        observation_outputs, _has_runtime_observation = (
            _block_runtime_observation_summary(
                body,
                strategy_for=self._strategy_for,
                cache=self._run_state.runtime_observation_cache,
            )
        )
        caller_taint: dict[str, Boolean] = {}
        for body_index, output in enumerate(body.output_values):
            condition = _value_taint_condition(
                output,
                body_estimate._measurement_taint_conditions,
            )
            if condition is sp.false and body_index in observation_outputs:
                condition = sp.true
            if condition is sp.false:
                continue
            for caller_index in selection.map_result_indices(
                (body_index,),
                operation.results,
            ):
                caller_taint[operation.results[caller_index].uuid] = condition
        if caller_taint:
            estimate = dataclasses.replace(
                estimate,
                _measurement_taint_conditions=caller_taint,
            )
        return estimate

    def _handle_unknown_invoke(
        self,
        operation: InvokeOperation,
        transform: _OpaqueInvocationTransform,
    ) -> ResourceEstimate:
        """Handle an invocation without a body or opaque cost.

        Args:
            operation (InvokeOperation): Invocation operation.
            transform (_OpaqueInvocationTransform): Private call-site
                transforms.

        Returns:
            ResourceEstimate: Opaque or zero estimate when policy permits.

        Raises:
            ValueError: If the configured unknown policy is ``ERROR``.
        """
        name = operation.custom_name
        if self.config.unknown_policy is UnknownResourcePolicy.OPAQUE_CALL:
            estimate = ResourceEstimate(
                calls=CallResources(
                    calls_by_name={name: _ONE},
                    queries_by_name={name: _ONE},
                ),
                trace=ResourceTraceNode(name, "opaque"),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
            )
        elif self.config.unknown_policy is UnknownResourcePolicy.ZERO_WITH_WARNING:
            assumption = ResourceAssumption(
                "unknown callable counted as zero resources",
                source=name,
            )
            estimate = ResourceEstimate(
                assumptions=(assumption,),
                trace=ResourceTraceNode(name, "opaque", assumptions=(assumption,)),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
            )
        else:
            raise ValueError(
                f"Cannot estimate resources for callable '{name}': no body or "
                "opaque cost is available."
            )
        if transform.local_controls:
            estimate = self._with_zero_control_bracket(
                estimate,
                zero_controls=transform.zero_controls,
            )
        return estimate
