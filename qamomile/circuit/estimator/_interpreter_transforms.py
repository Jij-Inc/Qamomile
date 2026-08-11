"""Interpret transformed callable resource operations."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, cast

import sympy as sp

from qamomile.circuit.estimator._allocation_width import (
    _namespace_allocation_sites,
)
from qamomile.circuit.estimator._config import (
    UnknownResourcePolicy,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._constraints import (
    _quantum_operand_width_constraints,
)
from qamomile.circuit.estimator._dependency_call_mapping import (
    _map_body_dependency_completion,
    _map_body_dependency_keys,
    _map_body_synchronized_entry_conditions,
)
from qamomile.circuit.estimator._dependency_footprints import (
    _controlled_u_control_wire_keys,
    _wire_keys_for_values,
)
from qamomile.circuit.estimator._dependency_indices import _UNKNOWN_WIRE_INDEX
from qamomile.circuit.estimator._dependency_metadata import (
    _complete_dependency_completion,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import (
    _estimate_activity,
    _with_constraints,
)
from qamomile.circuit.estimator._interpreter_calls import (
    _CallInterpreter,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _publish_invoke_classical_results,
)
from qamomile.circuit.estimator._opaque import (
    _zero_control_count,
)
from qamomile.circuit.estimator._product_formula import (
    _apply_product_formula_contract,
)
from qamomile.circuit.estimator._quantum_values import _qubit_value_size
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_algebra import (
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    GateResources,
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.estimator._scheduling import (
    _estimate_has_nonzero_depth,
    _with_body_boundary_depth_metadata,
)
from qamomile.circuit.estimator._scopes import (
    _controlled_u_child_resolver,
    _inverse_block_child_resolver,
    _resolve_controlled_u,
    _scalar_target_broadcast_factor,
    _select_case_child_resolver,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import (
    Value,
)


class _TransformedCallInterpreter(_CallInterpreter):
    """Add controlled, SELECT, and inverse callable handlers."""

    def eval_controlled_u(
        self,
        operation: ControlledUOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a controlled-U operation with power-aware semantics.

        Args:
            operation (ControlledUOperation): Controlled unitary operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Controlled unitary estimate.

        Raises:
            ValueError: If the controlled callable has no body and the unknown
                resource policy is ``ERROR``, or if a structural width or
                control requirement is invalid.
        """
        local_controls, _num_targets = _resolve_controlled_u(operation, resolver)
        local_controls = self._apply_condition_values(
            _expr(local_controls),
            record_usage=False,
        )
        total_controls = self._apply_condition_values(
            _expr(controls) + local_controls,
            record_usage=False,
        )
        power = self._apply_condition_values(resolver.resolve(operation.power))
        control_pool_width = sum(
            (
                _qubit_value_size(value, resolver)
                for value in operation.control_operands
            ),
            _ZERO,
        )
        control_indices = getattr(operation, "control_indices", None)
        resolved_indices = (
            tuple(
                self._apply_condition_values(resolver.resolve(index))
                for index in control_indices
            )
            if control_indices is not None
            else ()
        )
        selected_control_keys = _controlled_u_control_wire_keys(
            operation,
            resolved_indices,
            resolver,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        unresolved_control_selection = control_indices is not None and any(
            index is None or index is _UNKNOWN_WIRE_INDEX
            for _owner, index in selected_control_keys
        )
        callable_name = (
            operation.callable_ref.name
            if operation.callable_ref is not None
            else "controlled_u"
        )

        def apply_product_formula(estimate: ResourceEstimate) -> ResourceEstimate:
            """Apply callable-level product-formula metadata before repetition.

            Args:
                estimate (ResourceEstimate): One target-body application.

            Returns:
                ResourceEstimate: Estimate with callable-level approximation
                and validity metadata.
            """
            return _apply_product_formula_contract(
                estimate,
                operation.callable_attrs,
                operation.body_operands,
                resolver,
                bindings=self.bindings,
                specialize=lambda expression: self._apply_condition_values(
                    expression,
                    record_usage=False,
                ),
                source=callable_name,
            )

        structural_constraints = [
            *_quantum_operand_width_constraints(
                operation.callable_attrs,
                operation.body_operands,
                resolver,
                source=callable_name,
            ),
            _ResourceConstraint(
                expression=local_controls,
                minimum=1,
                label="Controlled operation control count",
                unit="control qubit",
            ),
            _ResourceConstraint(
                expression=(
                    sp.Integer(len(resolved_indices))
                    if control_indices is not None
                    else control_pool_width
                ),
                minimum=None,
                expected=local_controls,
                label="Controlled operation control operand width",
                unit="control qubit",
            ),
            _ResourceConstraint(
                expression=power,
                minimum=0,
                label="Controlled operation power",
                unit="application",
            ),
        ]
        for position, index in enumerate(resolved_indices):
            structural_constraints.append(
                _ResourceConstraint(
                    expression=index,
                    minimum=0,
                    label=(
                        f"Controlled operation control index {position} lower bound"
                    ),
                    unit="control position",
                )
            )
            structural_constraints.append(
                _ResourceConstraint(
                    expression=control_pool_width - index,
                    minimum=1,
                    label=(
                        f"Controlled operation control index {position} upper bound"
                    ),
                    unit="available control position",
                )
            )
        for left, right in itertools.combinations(resolved_indices, 2):
            structural_constraints.append(
                _ResourceConstraint(
                    expression=(left - right) ** 2,
                    minimum=1,
                    label="Controlled operation control indices uniqueness",
                )
            )
        for constraint in structural_constraints:
            constraint.validate()
        if power == _ZERO:
            return _with_constraints(
                ResourceEstimate.zero(f"{callable_name}^0"),
                *structural_constraints,
            )
        if isinstance(operation.block, Block):
            actual_operands = operation.body_operands
            broadcast = self._apply_condition_values(
                _scalar_target_broadcast_factor(
                    operation.block,
                    [
                        operand
                        for operand in actual_operands
                        if operand.type.is_quantum()
                    ],
                    resolver,
                ),
                record_usage=False,
            )
            if broadcast.is_zero is True:
                return _with_constraints(
                    ResourceEstimate.zero(f"{callable_name}[empty broadcast]"),
                    *structural_constraints,
                )
            child = _controlled_u_child_resolver(operation, resolver)
            body = self._eval_call_body(
                operation.block,
                child,
                actual_operands,
                controls=total_controls,
                callable_attrs=operation.callable_attrs,
                contract_operands=operation.body_operands,
                contract_resolver=resolver,
                source=callable_name,
            )
            body = apply_product_formula(body)
            body_dependency_estimate = body
            repetitions = power * broadcast
            estimate = body.repeat(repetitions)
            zero_controls = _ZERO
            if hasattr(operation, "control_value"):
                zero_controls = _zero_control_count(
                    int(local_controls),
                    cast(Any, operation).control_value,
                )
                estimate = self._with_zero_control_bracket(
                    estimate,
                    zero_controls=zero_controls,
                    active_when=_estimate_activity(estimate),
                )
            dependency_keys = _map_body_dependency_keys(
                operation.block,
                body_dependency_estimate,
                actual_operands,
                operation.results[len(operation.control_operands) :],
                resolver,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            dependency_completion = _map_body_dependency_completion(
                operation.block,
                estimate,
                actual_operands,
                operation.results[len(operation.control_operands) :],
                resolver,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            synchronized_entry_conditions = _map_body_synchronized_entry_conditions(
                operation.block,
                estimate,
                actual_operands,
                resolver,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            if dependency_keys is not None:
                mapped_keys = set(dependency_keys)
                if _estimate_has_nonzero_depth(estimate):
                    mapped_keys.update(selected_control_keys)
                estimate = dataclasses.replace(
                    estimate,
                    _dependency_keys=frozenset(mapped_keys),
                    _dependency_reads=frozenset(mapped_keys),
                    _dependency_writes=frozenset(mapped_keys),
                    _dependency_completion=_complete_dependency_completion(
                        dependency_completion,
                        mapped_keys,
                        fallback_depth=estimate.depth.depth,
                    ),
                    _dependency_synchronized_entry_conditions=(
                        synchronized_entry_conditions
                    ),
                )
            estimate = _with_constraints(
                _namespace_allocation_sites(estimate, operation),
                *structural_constraints,
            )
            if unresolved_control_selection:
                assumption = ResourceAssumption(
                    "control selection could not be resolved to scalar "
                    "control-pool dependency addresses",
                    source=callable_name,
                )
                estimate = estimate._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                )
            return _with_body_boundary_depth_metadata(
                estimate,
                estimate,
                source=callable_name,
                zero_controls=zero_controls,
                scalar_broadcast=broadcast,
            )

        name = callable_name
        assumption = ResourceAssumption(
            "controlled callable has no implementation body or explicit cost",
            source=name,
        )
        if self.config.unknown_policy is UnknownResourcePolicy.ERROR:
            raise ValueError(
                f"Cannot estimate resources for controlled callable '{name}': "
                "no body or opaque cost is available."
            )
        if self.config.unknown_policy is UnknownResourcePolicy.ZERO_WITH_WARNING:
            estimate = ResourceEstimate(
                assumptions=(assumption,),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
                trace=ResourceTraceNode(
                    name,
                    "opaque",
                    assumptions=(assumption,),
                ),
            )
            if hasattr(operation, "control_value"):
                estimate = self._with_zero_control_bracket(
                    estimate,
                    zero_controls=_zero_control_count(
                        int(local_controls),
                        cast(Any, operation).control_value,
                    ),
                    active_when=power,
                )
            estimate = apply_product_formula(estimate)
            return _with_constraints(estimate, *structural_constraints)
        gates = GateResources(
            total=_ZERO,
        )
        calls = CallResources(
            calls_by_name={name: power},
            queries_by_name={name: power},
        )
        estimate = ResourceEstimate(
            gates=gates,
            calls=calls,
            assumptions=(assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            trace=ResourceTraceNode(
                name,
                "opaque",
                summary=f"controlled power={power}",
                assumptions=(assumption,),
            ),
        )
        if hasattr(operation, "control_value"):
            estimate = self._with_zero_control_bracket(
                estimate,
                zero_controls=_zero_control_count(
                    int(local_controls),
                    cast(Any, operation).control_value,
                ),
                active_when=power,
            )
        estimate = apply_product_formula(estimate)
        return _with_constraints(estimate, *structural_constraints)

    def eval_select(
        self,
        operation: SelectOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate every controlled case body of a SELECT operation.

        SELECT lowering emits one controlled case for every addressable case,
        so logical resource estimation composes all case bodies sequentially.
        The index register contributes one coherent control per index qubit,
        in addition to controls surrounding the SELECT itself. Control-value
        LSB-first zero-valued case bits add X brackets around each nonempty
        case. Under an outer controlled region those bracket gates inherit the
        outer controls, matching the clean-ancilla SELECT decomposition.

        Args:
            operation (SelectOperation): SELECT operation to evaluate.
            resolver (ExprResolver): Resolver for the SELECT call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Sequential estimate of all controlled case
            bodies, including scalar-case broadcast over vector targets.

        Raises:
            ValueError: If the SELECT index width, flattened index operand
                width, target operand width, or a case resource contract is
                invalid.
        """
        index_controls = self._apply_condition_values(
            (
                resolver.resolve(operation.num_index_qubits)
                if isinstance(operation.num_index_qubits, Value)
                else _expr(operation.num_index_qubits)
            ),
            record_usage=False,
        )
        surrounding_controls = self._apply_condition_values(
            _expr(controls),
            record_usage=False,
        )
        total_controls = surrounding_controls + index_controls
        minimum_width = (len(operation.case_blocks) - 1).bit_length()
        index_constraint = _ResourceConstraint(
            expression=index_controls,
            minimum=minimum_width,
            label=f"SELECT index width for {len(operation.case_blocks)} cases",
            unit="qubit",
        )
        index_operand_constraint = _ResourceConstraint(
            expression=sum(
                (
                    _qubit_value_size(value, resolver)
                    for value in operation.index_operands
                ),
                _ZERO,
            ),
            minimum=None,
            expected=index_controls,
            label="SELECT index operand width",
            unit="qubit",
        )
        target_operand_constraint = _ResourceConstraint(
            expression=sum(
                (
                    _qubit_value_size(value, resolver)
                    for value in operation.target_operands
                ),
                _ZERO,
            ),
            minimum=1,
            label="SELECT target operand width",
            unit="qubit",
        )
        index_constraint.validate()
        index_operand_constraint.validate()
        target_operand_constraint.validate()
        case_width_constraints = tuple(
            constraint
            for case_index, attrs in enumerate(operation.case_callable_attrs)
            for constraint in _quantum_operand_width_constraints(
                attrs,
                operation.target_operands,
                resolver,
                source=f"SELECT case {case_index}",
            )
        )
        for constraint in case_width_constraints:
            constraint.validate()
        case_estimates: list[ResourceEstimate] = []
        for case_index, case_block in enumerate(operation.case_blocks):
            case_attrs = (
                operation.case_callable_attrs[case_index]
                if operation.case_callable_attrs
                else {}
            )
            child = _select_case_child_resolver(operation, case_block, resolver)
            actual_operands = [
                *operation.target_operands,
                *operation.param_operands,
            ]
            broadcast = self._apply_condition_values(
                _scalar_target_broadcast_factor(
                    case_block,
                    operation.target_operands,
                    resolver,
                ),
                record_usage=False,
            )
            case_body = self._eval_call_body(
                case_block,
                child,
                actual_operands,
                controls=total_controls,
                callable_attrs=case_attrs,
                contract_operands=actual_operands,
                contract_resolver=resolver,
                source=f"SELECT case {case_index}",
            )
            case_body = _apply_product_formula_contract(
                case_body,
                case_attrs,
                actual_operands,
                resolver,
                bindings=self.bindings,
                specialize=lambda expression: self._apply_condition_values(
                    expression,
                    record_usage=False,
                ),
                source=f"SELECT case {case_index}",
            )
            case_estimate = case_body.repeat(broadcast)
            activity = _estimate_activity(case_estimate)
            zero_controls = index_controls - case_index.bit_count()
            case_estimate = self._with_zero_control_bracket(
                case_estimate,
                zero_controls=zero_controls,
                surrounding_controls=surrounding_controls,
                active_when=activity,
            )
            dependency_keys = _map_body_dependency_keys(
                case_block,
                case_body,
                actual_operands,
                operation.results[operation.num_index_args :],
                resolver,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            dependency_completion = _map_body_dependency_completion(
                case_block,
                case_estimate,
                actual_operands,
                operation.results[operation.num_index_args :],
                resolver,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            synchronized_entry_conditions = _map_body_synchronized_entry_conditions(
                case_block,
                case_estimate,
                actual_operands,
                resolver,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            if dependency_keys is not None:
                mapped_keys = set(dependency_keys)
                if _estimate_has_nonzero_depth(case_estimate):
                    mapped_keys.update(
                        _wire_keys_for_values(
                            operation.index_operands,
                            resolver,
                            scalar_values=self._run_state.condition_values,
                            used_names=self._run_state.branch_condition_names,
                        )
                    )
                case_estimate = dataclasses.replace(
                    case_estimate,
                    _dependency_keys=frozenset(mapped_keys),
                    _dependency_reads=frozenset(mapped_keys),
                    _dependency_writes=frozenset(mapped_keys),
                    _dependency_completion=_complete_dependency_completion(
                        dependency_completion,
                        mapped_keys,
                        fallback_depth=case_estimate.depth.depth,
                    ),
                    _dependency_synchronized_entry_conditions=(
                        synchronized_entry_conditions
                    ),
                )
            case_estimate = _with_body_boundary_depth_metadata(
                case_estimate,
                case_estimate,
                source=f"select[{case_index}]",
                zero_controls=zero_controls,
                scalar_broadcast=broadcast,
            )
            case_estimate = dataclasses.replace(
                case_estimate,
                trace=_wrap_trace(
                    f"select[{case_index}]",
                    case_estimate.trace,
                    source_kind="body",
                ),
            )
            case_estimates.append(case_estimate)
        estimate = ResourceEstimate.seq_all(case_estimates)
        return _with_constraints(
            _namespace_allocation_sites(
                dataclasses.replace(
                    estimate,
                    trace=_wrap_trace(
                        "select",
                        estimate.trace,
                        source_kind="body",
                    ),
                ),
                operation,
            ),
            index_constraint,
            index_operand_constraint,
            target_operand_constraint,
            *case_width_constraints,
        )

    def eval_inverse_block(
        self,
        operation: InverseBlockOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate an inverse block through its implementation body.

        Args:
            operation (InverseBlockOperation): Inverse operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Inverse implementation estimate.

        Raises:
            ValueError: If the inverse callable has no implementation body and
                the unknown resource policy is ``ERROR``, or if a structural
                width requirement is invalid.
        """
        name = operation.name or "inverse_block"
        width_constraints = _quantum_operand_width_constraints(
            operation.callable_attrs,
            operation.target_qubits,
            resolver,
            source=name,
        )
        if not isinstance(operation.implementation_block, Block):
            assumption = ResourceAssumption(
                "inverse callable has no implementation body or explicit cost",
                source=name,
            )
            if self.config.unknown_policy is UnknownResourcePolicy.ERROR:
                raise ValueError(
                    f"Cannot estimate resources for inverse callable '{name}': "
                    "no implementation body is available."
                )
            if self.config.unknown_policy is UnknownResourcePolicy.OPAQUE_CALL:
                estimate = ResourceEstimate(
                    calls=CallResources(
                        calls_by_name={name: _ONE},
                        queries_by_name={name: _ONE},
                    ),
                    assumptions=(assumption,),
                    derivation=EstimateDerivation.MODELED,
                    quality=EstimateQuality.UNKNOWN,
                    trace=ResourceTraceNode(
                        name,
                        "opaque",
                        assumptions=(assumption,),
                    ),
                )
            else:
                estimate = ResourceEstimate(
                    assumptions=(assumption,),
                    derivation=EstimateDerivation.MODELED,
                    quality=EstimateQuality.UNKNOWN,
                    trace=ResourceTraceNode(
                        name,
                        "opaque",
                        assumptions=(assumption,),
                    ),
                )
            estimate = self._with_zero_control_bracket(
                estimate.inverse(),
                zero_controls=_zero_control_count(
                    operation.num_control_qubits,
                    operation.control_value,
                ),
            )
            return _with_constraints(
                _namespace_allocation_sites(estimate, operation),
                *width_constraints,
            )
        child = _inverse_block_child_resolver(operation, resolver)
        actual_operands = [*operation.target_qubits, *operation.parameters]
        body_estimate = self._eval_call_body(
            operation.implementation_block,
            child,
            actual_operands,
            controls=_expr(controls) + operation.num_control_qubits,
            callable_attrs=operation.callable_attrs,
            contract_operands=actual_operands,
            contract_resolver=resolver,
            source=name,
        )
        body_estimate = _apply_product_formula_contract(
            body_estimate,
            operation.callable_attrs,
            actual_operands,
            resolver,
            bindings=self.bindings,
            specialize=lambda expression: self._apply_condition_values(
                expression,
                record_usage=False,
            ),
            source=name,
        )
        _publish_invoke_classical_results(
            operation.implementation_block.output_values,
            operation.results[operation.num_control_qubits :],
            child,
            resolver,
        )
        # ``implementation_block`` is already the gate-by-gate inverse
        # fallback. Applying ``ResourceEstimate.inverse()`` here would
        # transform an opaque callback result twice and incorrectly reject an
        # authoritative measurement-assisted inverse implementation.
        estimate = dataclasses.replace(
            body_estimate,
            trace=_wrap_trace("inverse", body_estimate.trace),
        )
        zero_controls = _zero_control_count(
            operation.num_control_qubits,
            operation.control_value,
        )
        estimate = self._with_zero_control_bracket(
            estimate,
            zero_controls=zero_controls,
            active_when=_estimate_activity(estimate),
        )
        dependency_keys = _map_body_dependency_keys(
            operation.implementation_block,
            body_estimate,
            actual_operands,
            operation.results[operation.num_control_qubits :],
            resolver,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        dependency_completion = _map_body_dependency_completion(
            operation.implementation_block,
            estimate,
            actual_operands,
            operation.results[operation.num_control_qubits :],
            resolver,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        synchronized_entry_conditions = _map_body_synchronized_entry_conditions(
            operation.implementation_block,
            estimate,
            actual_operands,
            resolver,
            scalar_values=self._run_state.condition_values,
            used_names=self._run_state.branch_condition_names,
        )
        if dependency_keys is not None:
            mapped_keys = set(dependency_keys)
            if operation.num_control_qubits and _estimate_has_nonzero_depth(estimate):
                mapped_keys.update(
                    _wire_keys_for_values(
                        operation.control_qubits,
                        resolver,
                        scalar_values=self._run_state.condition_values,
                        used_names=self._run_state.branch_condition_names,
                    )
                )
            estimate = dataclasses.replace(
                estimate,
                _dependency_keys=frozenset(mapped_keys),
                _dependency_reads=frozenset(mapped_keys),
                _dependency_writes=frozenset(mapped_keys),
                _dependency_completion=_complete_dependency_completion(
                    dependency_completion,
                    mapped_keys,
                    fallback_depth=estimate.depth.depth,
                ),
                _dependency_synchronized_entry_conditions=(
                    synchronized_entry_conditions
                ),
            )
        estimate = _with_constraints(
            _namespace_allocation_sites(estimate, operation),
            *width_constraints,
        )
        return _with_body_boundary_depth_metadata(
            estimate,
            estimate,
            source=name,
            zero_controls=zero_controls,
        )
