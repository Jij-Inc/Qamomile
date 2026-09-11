"""Merge conditional-branch resource state."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._call_liveness import (
    _quantum_result_owner_sizes,
)
from qamomile.circuit.estimator._classical_provenance import (
    _value_taint_condition,
)
from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._constraints import (
    _collect_array_value_constraints,
    _validated_unproven_array_constraints,
)
from qamomile.circuit.estimator._dependency_call_mapping import (
    _map_value_dependency_keys,
)
from qamomile.circuit.estimator._dependency_indices import WireKey
from qamomile.circuit.estimator._dependency_metadata import (
    _normalized_dependency_completion,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._interpreter_while import _WhileInterpreter
from qamomile.circuit.estimator._quantum_values import (
    _quantum_allocation_owner,
    _qubit_value_size,
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
    _piecewise,
    _resource_max,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.estimator._scopes import (
    _typed_value_symbol,
)
from qamomile.circuit.ir.operation.control_flow import (
    IfOperation,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
)


class _IfMergeInterpreter(_WhileInterpreter):
    """Add conditional result, dependency, and liveness merging."""

    def _publish_if_results(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        true_resolver: ExprResolver,
        false_resolver: ExprResolver,
        *,
        taken: bool | None,
        runtime_condition: Boolean = sp.false,
        conservative_condition: Boolean = sp.false,
    ) -> None:
        """Publish branch-merge results into the enclosing symbolic environment.

        Args:
            operation (IfOperation): Conditional carrying the merge records.
            resolver (ExprResolver): Enclosing resolver to update.
            true_resolver (ExprResolver): Resolver for the true branch.
            false_resolver (ExprResolver): Resolver for the false branch.
            taken (bool | None): Decided branch, or ``None`` when the condition
                remains symbolic or runtime-dependent.
            runtime_condition (Boolean): Condition under which the branch is
                selected from an observation at runtime. Defaults to false.
            conservative_condition (Boolean): Condition under which an
                unresolved loop state requires a branch-wise conservative
                choice. Defaults to false.
        """
        condition_fact = resolver.resolve_classical_fact(operation.condition)
        predicate = _boolean_condition(condition_fact.value)
        nondeterministic_condition = _boolean_condition(
            sp.Or(runtime_condition, conservative_condition)
        )
        for merge in operation.iter_merges():
            if (
                all(
                    isinstance(value, ArrayValue)
                    for value in (
                        merge.true_value,
                        merge.false_value,
                        merge.result,
                    )
                )
                and not merge.result.type.is_quantum()
            ):
                true_array = cast(ArrayValue, merge.true_value)
                false_array = cast(ArrayValue, merge.false_value)
                result_array = cast(ArrayValue, merge.result)
                true_state = true_resolver.snapshot_array_state(true_array)
                false_state = false_resolver.snapshot_array_state(false_array)
                if taken is True:
                    resolver.bind_array_state(result_array, true_state)
                elif taken is False:
                    resolver.bind_array_state(result_array, false_state)
                else:
                    resolver.bind_array_state_selection(
                        result_array,
                        true_state,
                        false_state,
                        condition_fact,
                    )
                for true_dim, false_dim, result_dim in zip(
                    true_array.shape,
                    false_array.shape,
                    result_array.shape,
                    strict=True,
                ):
                    true_size = true_resolver.resolve(true_dim)
                    false_size = false_resolver.resolve(false_dim)
                    if taken is True:
                        merged_size = true_size
                    elif taken is False:
                        merged_size = false_size
                    else:
                        compile_size = _piecewise(
                            true_size,
                            false_size,
                            predicate,
                        )
                        merged_size = _piecewise(
                            sp.Max(true_size, false_size),
                            compile_size,
                            nondeterministic_condition,
                        )
                    true_fact = true_resolver.resolve_classical_fact(true_dim)
                    false_fact = false_resolver.resolve_classical_fact(false_dim)
                    if taken is True:
                        resolver.bind_classical_fact(result_dim, true_fact)
                    elif taken is False:
                        resolver.bind_classical_fact(result_dim, false_fact)
                    else:
                        resolver.bind_classical_selection(
                            result_dim,
                            true_fact,
                            false_fact,
                            condition_fact,
                            value_override=merged_size,
                        )
                true_array_fact = true_resolver.resolve_classical_fact(true_array)
                false_array_fact = false_resolver.resolve_classical_fact(false_array)
                if taken is True:
                    resolver.bind_classical_fact(result_array, true_array_fact)
                elif taken is False:
                    resolver.bind_classical_fact(result_array, false_array_fact)
                else:
                    resolver.bind_classical_selection(
                        result_array,
                        true_array_fact,
                        false_array_fact,
                        condition_fact,
                    )
                continue
            true_value = true_resolver.resolve(merge.true_value)
            false_value = false_resolver.resolve(merge.false_value)
            if taken is True:
                merged = true_value
            elif taken is False:
                merged = false_value
            else:
                # A runtime measurement or unresolved loop state chooses one
                # merge value. Keeping that choice as a nested Piecewise makes
                # feed-forward-heavy circuits grow exponentially even though
                # resource counting already combines both branches above. A
                # fresh typed symbol preserves the unknown value without
                # coupling later estimates to the complete selection history.
                runtime_value = _typed_value_symbol(
                    merge.result,
                    merge.result.name,
                    fresh=True,
                )
                runtime_domain = self._register_runtime_value_domain(
                    runtime_value,
                    cast(sp.Expr, true_value),
                    cast(sp.Expr, false_value),
                )
                if runtime_domain is not None and len(runtime_domain) == 1:
                    runtime_branch_value = runtime_domain[0]
                    self._run_state.runtime_value_domains.pop(runtime_value, None)
                else:
                    runtime_branch_value = runtime_value
                    if runtime_condition is not sp.false:
                        self._run_state.runtime_observation_symbols.add(runtime_value)
                    else:
                        self._run_state.unresolved_resource_symbols.add(runtime_value)
                compile_value = _piecewise(
                    cast(ResourceExpr, true_value),
                    cast(ResourceExpr, false_value),
                    predicate,
                )
                merged = _piecewise(
                    cast(ResourceExpr, runtime_branch_value),
                    cast(ResourceExpr, compile_value),
                    nondeterministic_condition,
                )
            if merge.result.type.is_quantum():
                true_size = _qubit_value_size(merge.true_value, true_resolver)
                false_size = _qubit_value_size(merge.false_value, false_resolver)
                if taken is True:
                    selected_size = true_size
                elif taken is False:
                    selected_size = false_size
                else:
                    selected_size = _piecewise(
                        sp.Max(true_size, false_size),
                        _piecewise(true_size, false_size, predicate),
                        nondeterministic_condition,
                    )
                resolver.bind_quantum_size(merge.result, selected_size)
                resolver.bind(merge.result, cast(sp.Expr, merged))
                selected_values = (
                    (merge.true_value,)
                    if taken is True
                    else (
                        (merge.false_value,)
                        if taken is False
                        else (merge.true_value, merge.false_value)
                    )
                )
                owners = {
                    self._run_state.allocation_owners_by_uuid.get(
                        value.uuid,
                        _quantum_allocation_owner(value),
                    )
                    for value in selected_values
                    if isinstance(value, Value) and value.type.is_quantum()
                }
                if len(owners) == 1:
                    self._run_state.allocation_owners_by_uuid[merge.result.uuid] = next(
                        iter(owners)
                    )
                result_owner = _quantum_allocation_owner(merge.result)
                source_owners = {
                    _quantum_allocation_owner(value)
                    for value in selected_values
                    if isinstance(value, Value) and value.type.is_quantum()
                }
                source_owners.discard(result_owner)
                if source_owners:
                    self._run_state.dependency_owner_aliases[result_owner] = frozenset(
                        source_owners
                    )
            else:
                true_fact = true_resolver.resolve_classical_fact(merge.true_value)
                false_fact = false_resolver.resolve_classical_fact(merge.false_value)
                if taken is True:
                    resolver.bind_classical_fact(merge.result, true_fact)
                elif taken is False:
                    resolver.bind_classical_fact(merge.result, false_fact)
                else:
                    resolver.bind_classical_selection(
                        merge.result,
                        true_fact,
                        false_fact,
                        condition_fact,
                        value_override=merged,
                    )

    def _if_merge_taint_conditions(
        self,
        operation: IfOperation,
        *,
        true_estimate: ResourceEstimate | None,
        false_estimate: ResourceEstimate | None,
        predicate: Boolean,
        runtime_guard: Boolean,
        conservative_guard: Boolean,
        taken: bool | None,
    ) -> dict[str, Boolean]:
        """Map guarded observation provenance onto conditional results.

        A runtime-selected merge is observation-derived regardless of the
        selected source. Outside that guard, provenance follows the ordinary
        compile-time predicate to the corresponding branch source.

        Args:
            operation (IfOperation): Conditional carrying merge records.
            true_estimate (ResourceEstimate | None): Evaluated true branch, or
                ``None`` when it was not selected.
            false_estimate (ResourceEstimate | None): Evaluated false branch,
                or ``None`` when it was not selected.
            predicate (Boolean): Compile-time branch predicate.
            runtime_guard (Boolean): Condition selecting runtime semantics.
            conservative_guard (Boolean): Condition under which unresolved
                loop state can select either branch.
            taken (bool | None): Statically selected branch, if any.

        Returns:
            dict[str, Boolean]: Observation provenance keyed by merge-result
            UUID.
        """
        result: dict[str, Boolean] = {}
        for merge in operation.iter_merges():
            if merge.result.type.is_quantum():
                continue
            true_taint = (
                _value_taint_condition(
                    merge.true_value,
                    true_estimate._measurement_taint_conditions,
                )
                if true_estimate is not None
                else sp.false
            )
            false_taint = (
                _value_taint_condition(
                    merge.false_value,
                    false_estimate._measurement_taint_conditions,
                )
                if false_estimate is not None
                else sp.false
            )
            if taken is True:
                condition = true_taint
            elif taken is False:
                condition = false_taint
            else:
                condition = _boolean_condition(
                    sp.Or(
                        runtime_guard,
                        sp.And(
                            conservative_guard,
                            sp.Or(true_taint, false_taint),
                        ),
                        sp.And(
                            sp.Not(conservative_guard),
                            predicate,
                            true_taint,
                        ),
                        sp.And(
                            sp.Not(conservative_guard),
                            sp.Not(predicate),
                            false_taint,
                        ),
                    )
                )
            if condition is not sp.false:
                result[merge.result.uuid] = condition
        return result

    def _if_boundary_array_constraints(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        true_resolver: ExprResolver,
        false_resolver: ExprResolver,
        *,
        taken: bool | None,
        predicate: Boolean,
        conservative_guard: Boolean,
    ) -> tuple[_ResourceConstraint, ...]:
        """Collect condition and merge-only array constraints with guards.

        Branch operations own their ordinary array accesses. Merge records can
        reference an array element without emitting any operation, so those
        boundary-only values are collected here under the branch condition
        that can actually select them.

        Args:
            operation (IfOperation): Conditional carrying merge records.
            resolver (ExprResolver): Enclosing resolver for the condition.
            true_resolver (ExprResolver): True-branch resolver.
            false_resolver (ExprResolver): False-branch resolver.
            taken (bool | None): Statically selected branch, if any.
            predicate (Boolean): Compile-time branch predicate.
            conservative_guard (Boolean): Condition requiring both branch
                constraints to remain active.

        Returns:
            tuple[_ResourceConstraint, ...]: Validated guarded requirements.
        """
        outer = self._run_state.constraint_scope_condition
        constraints = [
            constraint.when(outer)
            for constraint in _collect_array_value_constraints(
                (operation.condition,),
                resolver,
            )
        ]
        if taken is True:
            true_active = outer
            false_active = sp.false
        elif taken is False:
            true_active = sp.false
            false_active = outer
        else:
            true_active = _and_conditions(
                outer,
                sp.Or(conservative_guard, predicate),
            )
            false_active = _and_conditions(
                outer,
                sp.Or(conservative_guard, sp.Not(predicate)),
            )
        true_values = tuple(merge.true_value for merge in operation.iter_merges())
        false_values = tuple(merge.false_value for merge in operation.iter_merges())
        constraints.extend(
            constraint.when(true_active)
            for constraint in _collect_array_value_constraints(
                true_values,
                true_resolver,
            )
        )
        constraints.extend(
            constraint.when(false_active)
            for constraint in _collect_array_value_constraints(
                false_values,
                false_resolver,
            )
        )
        return _validated_unproven_array_constraints(
            constraints,
            proven_cache=self._run_state.array_constraint_proven,
        )

    def _with_if_dependency_outputs(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        *,
        true_estimate: ResourceEstimate | None,
        false_estimate: ResourceEstimate | None,
        combined: ResourceEstimate,
        taken: bool | None,
        runtime_condition: bool,
        condition: Boolean | None = None,
    ) -> ResourceEstimate:
        """Publish branch completion depths on conditional result owners.

        Branch work is scheduled on each source allocation, whereas operations
        after the conditional read its fresh merge-result owner. Copying the
        per-element completion depth to that result connects both sides of the
        boundary and also lets an enclosing callable map the returned value to
        its caller.

        Args:
            operation (IfOperation): Conditional carrying merge records.
            resolver (ExprResolver): Enclosing resolver after result shapes
                have been published.
            true_estimate (ResourceEstimate | None): Evaluated true branch, or
                ``None`` when it was not selected.
            false_estimate (ResourceEstimate | None): Evaluated false branch,
                or ``None`` when it was not selected.
            combined (ResourceEstimate): Selected or combined branch estimate.
            taken (bool | None): Statically selected branch, if any.
            runtime_condition (bool): Whether a measurement selects the branch
                at runtime.
            condition (Boolean | None): Compile-time predicate already refined
                for the active provenance branch. Defaults to resolving the
                operation condition in the caller scope.

        Returns:
            ResourceEstimate: Estimate with merge-result dependency metadata.
        """

        def mapped_completion(
            source: Value,
            estimate: ResourceEstimate | None,
        ) -> dict[WireKey, ResourceExpr]:
            """Map one branch source's completion onto its merge result.

            Args:
                source (Value): Quantum value yielded by the branch.
                estimate (ResourceEstimate | None): Corresponding branch
                    estimate, or ``None`` when that branch was not evaluated.

            Returns:
                dict[WireKey, ResourceExpr]: Result-owner completion depths.
            """
            if estimate is None:
                return {}
            source_completion = _normalized_dependency_completion(estimate)
            if source_completion is None:
                return {}
            source_owner = _quantum_allocation_owner(source)
            mapped: dict[WireKey, ResourceExpr] = {}
            for key, depth in source_completion.items():
                if key[0] != source_owner:
                    continue
                for result_key in _map_value_dependency_keys(
                    source,
                    merge.result,
                    frozenset((key,)),
                    resolver,
                    scalar_values=self._run_state.condition_values,
                    used_names=self._run_state.branch_condition_names,
                ):
                    mapped[result_key] = _resource_max(
                        mapped.get(result_key, _ZERO),
                        depth,
                    )
            return mapped

        keys = set(combined._dependency_keys or ())
        completion = dict(_normalized_dependency_completion(combined) or {})
        selected_condition = (
            condition
            if condition is not None
            else _boolean_condition(resolver.resolve(operation.condition))
        )
        has_ambiguous_alias = False
        for merge in operation.iter_merges():
            if not merge.result.type.is_quantum():
                continue
            true_completion = mapped_completion(
                merge.true_value,
                true_estimate,
            )
            false_completion = mapped_completion(
                merge.false_value,
                false_estimate,
            )
            for key in true_completion.keys() | false_completion.keys():
                if taken is True:
                    depth = true_completion.get(key, _ZERO)
                elif taken is False:
                    depth = false_completion.get(key, _ZERO)
                elif runtime_condition:
                    depth = _resource_max(
                        true_completion.get(key, _ZERO),
                        false_completion.get(key, _ZERO),
                    )
                else:
                    depth = _piecewise(
                        true_completion.get(key, _ZERO),
                        false_completion.get(key, _ZERO),
                        selected_condition,
                    )
                keys.add(key)
                completion[key] = _resource_max(
                    completion.get(key, _ZERO),
                    depth,
                )
            result_owner = _quantum_allocation_owner(merge.result)
            has_ambiguous_alias |= (
                len(
                    self._run_state.dependency_owner_aliases.get(
                        result_owner, frozenset()
                    )
                )
                > 1
            )
        result = dataclasses.replace(
            combined,
            _dependency_keys=frozenset(keys),
            _dependency_reads=frozenset(keys),
            _dependency_writes=frozenset(keys),
            _dependency_completion=completion,
            _dependency_completion_uniform=combined._dependency_completion_uniform,
        )
        if taken is not None or not has_ambiguous_alias:
            return result
        assumption = ResourceAssumption(
            "an unresolved conditional quantum result may alias either branch "
            "source and is scheduled conservatively",
            source="if dependency scheduler",
        )
        return result._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
        )

    def _if_output_sizes(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        true_resolver: ExprResolver,
        false_resolver: ExprResolver,
        *,
        true_estimate: ResourceEstimate | None,
        false_estimate: ResourceEstimate | None,
        true_inputs: Mapping[str, ResourceExpr],
        false_inputs: Mapping[str, ResourceExpr],
        taken: bool | None,
        runtime_condition: bool,
        condition: Boolean | None = None,
        true_consumed: Mapping[str, ResourceExpr],
        false_consumed: Mapping[str, ResourceExpr],
    ) -> dict[str, ResourceExpr]:
        """Resolve live quantum merge widths across an if operation.

        IR merge results retain one representative static shape, which can be
        wrong when branches produce differently sized arrays. This summary is
        consumed by outer liveness instead of re-reading that representative
        shape.

        Args:
            operation (IfOperation): Conditional carrying merge records.
            resolver (ExprResolver): Resolver for the branch condition.
            true_resolver (ExprResolver): True-branch resolver.
            false_resolver (ExprResolver): False-branch resolver.
            true_estimate (ResourceEstimate | None): Evaluated true-branch
                summary when that branch was visited.
            false_estimate (ResourceEstimate | None): Evaluated false-branch
                summary when that branch was visited.
            true_inputs (Mapping[str, ResourceExpr]): Outer allocations live at
                true-branch entry.
            false_inputs (Mapping[str, ResourceExpr]): Outer allocations live
                at false-branch entry.
            taken (bool | None): Statically selected branch, if any.
            runtime_condition (bool): Whether a measurement selects the branch
                at runtime.
            condition (Boolean | None): Compile-time predicate already refined
                for the active provenance branch. Defaults to resolving the
                operation condition in the caller scope.
            true_consumed (Mapping[str, ResourceExpr]): Captured owner widths
                destroyed by the true branch.
            false_consumed (Mapping[str, ResourceExpr]): Captured owner widths
                destroyed by the false branch.

        Returns:
            dict[str, ResourceExpr]: Live merged output width by root owner.
        """
        selected_condition = (
            condition
            if condition is not None
            else _boolean_condition(resolver.resolve(operation.condition))
        )
        true_authoritative = bool(
            true_estimate is not None and true_estimate._has_output_summary
        )
        false_authoritative = bool(
            false_estimate is not None and false_estimate._has_output_summary
        )
        true_sizes = {
            owner: (
                true_estimate._output_sizes.get(owner, _ZERO)
                if true_authoritative and true_estimate is not None
                else sp.Max(
                    _ZERO,
                    size - true_consumed.get(owner, _ZERO),
                )
            )
            for owner, size in true_inputs.items()
        }
        false_sizes = {
            owner: (
                false_estimate._output_sizes.get(owner, _ZERO)
                if false_authoritative and false_estimate is not None
                else sp.Max(
                    _ZERO,
                    size - false_consumed.get(owner, _ZERO),
                )
            )
            for owner, size in false_inputs.items()
        }

        def local_residual_width(
            estimate: ResourceEstimate | None,
            inputs: Mapping[str, ResourceExpr],
            returned_values: Sequence[Value],
            branch_resolver: ExprResolver,
        ) -> ResourceExpr:
            """Count branch-local live qubits not exposed by merge results.

            Args:
                estimate (ResourceEstimate | None): Evaluated branch summary.
                inputs (Mapping[str, ResourceExpr]): Caller-owned allocations
                    live at branch entry.
                returned_values (Sequence[Value]): Quantum values selected by
                    the branch's merge records.
                branch_resolver (ExprResolver): Resolver for branch-local
                    result widths.

            Returns:
                ResourceExpr: Live branch-local width inaccessible through a
                merge result.
            """
            if estimate is None or not estimate._has_output_summary:
                return _ZERO
            returned_by_owner = _quantum_result_owner_sizes(
                returned_values,
                branch_resolver,
                self._run_state.allocation_owners_by_uuid,
            )
            return sum(
                (
                    sp.Max(
                        _ZERO,
                        size - returned_by_owner.get(owner, _ZERO),
                    )
                    for owner, size in estimate._output_sizes.items()
                    if owner not in inputs
                ),
                _ZERO,
            )

        quantum_merges = tuple(
            merge for merge in operation.iter_merges() if merge.result.type.is_quantum()
        )
        true_residual = local_residual_width(
            true_estimate,
            true_inputs,
            [merge.true_value for merge in quantum_merges],
            true_resolver,
        )
        false_residual = local_residual_width(
            false_estimate,
            false_inputs,
            [merge.false_value for merge in quantum_merges],
            false_resolver,
        )
        output_sizes: dict[str, ResourceExpr] = {}
        for owner in true_sizes.keys() | false_sizes.keys():
            true_size = true_sizes.get(owner, _ZERO)
            false_size = false_sizes.get(owner, _ZERO)
            if taken is True:
                size = true_size
            elif taken is False:
                size = false_size
            elif runtime_condition:
                size = sp.Max(true_size, false_size)
            else:
                size = _piecewise(true_size, false_size, selected_condition)
            if size != _ZERO:
                output_sizes[owner] = size
        owner_map = self._run_state.allocation_owners_by_uuid
        for merge in quantum_merges:
            true_owner = owner_map.get(
                merge.true_value.uuid,
                _quantum_allocation_owner(merge.true_value),
            )
            false_owner = owner_map.get(
                merge.false_value.uuid,
                _quantum_allocation_owner(merge.false_value),
            )
            true_returned = (
                _ZERO
                if true_owner in true_inputs
                else _qubit_value_size(merge.true_value, true_resolver)
            )
            false_returned = (
                _ZERO
                if false_owner in false_inputs
                else _qubit_value_size(merge.false_value, false_resolver)
            )
            if taken is True:
                returned = true_returned
            elif taken is False:
                returned = false_returned
            elif runtime_condition:
                returned = _resource_max(true_returned, false_returned)
            else:
                returned = _piecewise(
                    true_returned,
                    false_returned,
                    selected_condition,
                )
            if returned == _ZERO:
                continue
            result_owner = owner_map.get(
                merge.result.uuid,
                _quantum_allocation_owner(merge.result),
            )
            output_sizes[result_owner] = (
                output_sizes.get(result_owner, _ZERO) + returned
            )
        if taken is True:
            residual = true_residual
        elif taken is False:
            residual = false_residual
        elif runtime_condition:
            residual = _resource_max(true_residual, false_residual)
        else:
            residual = _piecewise(
                true_residual,
                false_residual,
                selected_condition,
            )
        if residual != _ZERO:
            output_sizes[f"IfOperation:{operation.condition.uuid}/live"] = residual
        return output_sizes
