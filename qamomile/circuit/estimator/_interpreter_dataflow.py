"""Analyze resource-estimation dataflow without interpreter state."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._call_liveness import (
    _quantum_owner_capacities,
)
from qamomile.circuit.estimator._constants import (
    _ZERO,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._quantum_values import _quantum_allocation_owner
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _expr,
    _piecewise,
)
from qamomile.circuit.estimator._resource_scalars import _safe_piecewise_fold
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
    Signature,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
)


@dataclasses.dataclass
class _ResourceInlineBoundaryOperation(Operation):
    """Retain resource-only call state after a body is inlined.

    The operation is an estimator-only, zero-work marker. It validates an
    optional quantum-width contract and snapshots caller classical arrays into
    cloned callee entry values at the original call position. Referenced values
    deliberately do not appear in :attr:`Operation.operands`, so dependency
    scheduling and liveness do not mistake the boundary for executable work.

    Args:
        constraint_operands (tuple[Value, ...]): Caller-scoped quantum target
            operands whose widths must satisfy the callable contract.
        resource_operands (tuple[ValueBase, ...]): Caller-scoped operands in
            the callable's untransformed ABI order. They remain outside
            :attr:`Operation.operands` because the marker performs no work.
        callable_attrs (Mapping[str, Any]): Merged callable definition and
            invocation attributes that carry resource declarations.
        source (str): Callable name used in diagnostics.
        array_state_bindings (tuple[tuple[ArrayValue, tuple[ArrayValue, ...]],
            ...]): Caller arrays paired with cloned callee entry values that
            receive simultaneous call-time snapshots.
    """

    constraint_operands: tuple[Value, ...] = ()
    resource_operands: tuple[ValueBase, ...] = ()
    callable_attrs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    source: str = "callable"
    array_state_bindings: tuple[
        tuple[ArrayValue, tuple[ArrayValue, ...]],
        ...,
    ] = ()

    @property
    def signature(self) -> Signature:
        """Return an empty signature for the zero-work marker.

        Returns:
            Signature: Empty operation signature.
        """
        return Signature()

    @property
    def operation_kind(self) -> OperationKind:
        """Classify the marker as classical zero work.

        Returns:
            OperationKind: ``CLASSICAL`` so quantum scheduling ignores it.
        """
        return OperationKind.CLASSICAL


def _conditional_resource_map(
    when_true: Mapping[str, ResourceExpr],
    when_false: Mapping[str, ResourceExpr],
    condition: sp.Basic,
) -> dict[str, ResourceExpr]:
    """Select resource-map values field by field under one condition.

    Args:
        when_true (Mapping[str, ResourceExpr]): Values used when the condition
            holds.
        when_false (Mapping[str, ResourceExpr]): Values used otherwise.
        condition (sp.Basic): Boolean selection predicate.

    Returns:
        dict[str, ResourceExpr]: Conditional values for the union of keys.
    """
    predicate = _boolean_condition(condition)
    return {
        key: _piecewise(
            when_true.get(key, _ZERO),
            when_false.get(key, _ZERO),
            predicate,
        )
        for key in when_true.keys() | when_false.keys()
    }


def _zero_symbols_implied_by_assumption(
    assumption: Boolean,
) -> set[sp.Symbol]:
    """Return symbols that a Boolean assumption proves equal to zero.

    Equalities in a conjunction are positive facts, while an equality below a
    negation is not. For a disjunction, a zero value is implied only when every
    alternative proves it. This structural proof deliberately ignores other
    Boolean forms instead of guessing their polarity.

    Args:
        assumption (Boolean): Branch fact whose positive implications should
            be inspected.

    Returns:
        set[sp.Symbol]: Symbols proven to equal zero throughout the assumed
        branch.
    """
    if isinstance(assumption, sp.And):
        implied: set[sp.Symbol] = set()
        for argument in assumption.args:
            implied.update(_zero_symbols_implied_by_assumption(cast(Boolean, argument)))
        return implied
    if isinstance(assumption, sp.Or):
        alternatives = [
            _zero_symbols_implied_by_assumption(cast(Boolean, argument))
            for argument in assumption.args
        ]
        if not alternatives:
            return set()
        return set.intersection(*alternatives)
    if isinstance(assumption, sp.Equality):
        if isinstance(assumption.lhs, sp.Symbol) and assumption.rhs == _ZERO:
            return {assumption.lhs}
        if isinstance(assumption.rhs, sp.Symbol) and assumption.lhs == _ZERO:
            return {assumption.rhs}
        return set()
    if (
        isinstance(assumption, sp.LessThan)
        and isinstance(assumption.lhs, sp.Symbol)
        and assumption.lhs.is_nonnegative is True
        and assumption.rhs == _ZERO
    ):
        return {assumption.lhs}
    return set()


def _refine_boolean_under_assumption(
    condition: Boolean,
    assumption: Boolean,
) -> Boolean:
    """Refine a Boolean expression within a known symbolic branch.

    Runtime-observation values can appear in a merged classical expression only
    on the branch where their provenance guard is active. Refining the ordinary
    compile-time predicate under the complement removes those unreachable
    symbols before resource formulas and structural constraints are composed.

    Args:
        condition (Boolean): Predicate to simplify.
        assumption (Boolean): Branch fact known to hold.

    Returns:
        Boolean: Refined predicate, or the original condition if SymPy cannot
        safely refine it.
    """
    zero_substitutions = {
        symbol: _ZERO for symbol in _zero_symbols_implied_by_assumption(assumption)
    }
    narrowed = cast(Boolean, condition.xreplace(zero_substitutions))
    folded = cast(Boolean, _safe_piecewise_fold(narrowed))

    # ``assumption`` identifies the domain where this projection is used.
    # Its complement is therefore a genuine don't-care set.  In particular,
    # a measurement-selected merge is represented as ``Piecewise(runtime,
    # guard, compile_value)``; ordinary ``refine`` does not reliably remove
    # the runtime-only symbol when ``guard`` is a nested conjunction.
    try:
        projected = cast(
            Boolean,
            sp.simplify_logic(
                folded,
                dontcare=sp.Not(assumption),
                force=False,
            ),
        )
    except (RecursionError, TypeError, ValueError):
        projected = folded
    try:
        refined = sp.refine(projected, assumption)
    except (RecursionError, TypeError, ValueError):
        refined = projected
    return _boolean_condition(cast(sp.Basic, refined))


def _publish_invoke_classical_results(
    body_outputs: Sequence[ValueLike],
    caller_outputs: Sequence[ValueBase],
    body_resolver: ExprResolver,
    caller_resolver: ExprResolver,
) -> None:
    """Publish selected-body classical results into the caller resolver.

    A call is a dataflow boundary, so the caller resolver cannot discover a
    scalar result by scanning the callee block. Explicitly carrying the selected
    body's expression across that boundary keeps later compile-time branches,
    loop bounds, and array dimensions equivalent to an inlined body.

    Args:
        body_outputs (Sequence[ValueLike]): Selected body outputs in ABI order.
        caller_outputs (Sequence[ValueBase]): Aligned call-site results.
        body_resolver (ExprResolver): Resolver containing callee expressions.
        caller_resolver (ExprResolver): Resolver to update for later caller work.

    Raises:
        ValueError: If the already-validated selected ABI has inconsistent
            aggregate or array-shape arity.
    """

    def needs_publication(value: ValueBase) -> bool:
        """Return whether one caller output carries classical resolver state.

        Args:
            value (ValueBase): Caller-side output to inspect.

        Returns:
            bool: Whether scalar contents or array dimensions must be copied.
        """
        if isinstance(value, TupleValue):
            return any(needs_publication(element) for element in value.elements)
        if isinstance(value, DictValue):
            return any(
                needs_publication(key) or needs_publication(entry)
                for key, entry in value.entries
            )
        if isinstance(value, ArrayValue):
            return True
        return isinstance(value, Value) and not value.type.is_quantum()

    def publish(body_value: ValueLike, caller_value: ValueBase) -> None:
        """Publish one recursively aligned output value.

        Args:
            body_value (ValueLike): Callee-side output value.
            caller_value (ValueBase): Caller-side aligned result.

        Raises:
            ValueError: If aggregate or shape arity is inconsistent.
        """
        if isinstance(body_value, TupleValue):
            if not isinstance(caller_value, TupleValue) or len(
                body_value.elements
            ) != len(caller_value.elements):
                raise ValueError(
                    "Selected callable tuple output arity is inconsistent."
                )
            for nested_body, nested_caller in zip(
                body_value.elements,
                caller_value.elements,
                strict=True,
            ):
                publish(nested_body, nested_caller)
            return
        if isinstance(body_value, DictValue):
            if not isinstance(caller_value, DictValue) or len(
                body_value.entries
            ) != len(caller_value.entries):
                raise ValueError(
                    "Selected callable dictionary output arity is inconsistent."
                )
            for (body_key, body_entry), (caller_key, caller_entry) in zip(
                body_value.entries,
                caller_value.entries,
                strict=True,
            ):
                publish(body_key, caller_key)
                publish(body_entry, caller_entry)
            return
        if not isinstance(body_value, Value) or not isinstance(caller_value, Value):
            raise ValueError("Selected callable output value kinds are inconsistent.")
        if isinstance(body_value, ArrayValue):
            if not isinstance(caller_value, ArrayValue) or len(body_value.shape) != len(
                caller_value.shape
            ):
                raise ValueError("Selected callable array output rank is inconsistent.")
            for body_dimension, caller_dimension in zip(
                body_value.shape,
                caller_value.shape,
                strict=True,
            ):
                caller_resolver.bind_classical_fact(
                    caller_dimension,
                    body_resolver.resolve_classical_fact(body_dimension),
                )
            if not body_value.type.is_quantum():
                caller_resolver.bind_array_state(
                    caller_value,
                    body_resolver.snapshot_array_state(body_value),
                )
                caller_resolver.bind_classical_fact(
                    caller_value,
                    body_resolver.resolve_classical_fact(body_value),
                )
            return
        if not body_value.type.is_quantum():
            caller_resolver.bind_classical_fact(
                caller_value,
                body_resolver.resolve_classical_fact(body_value),
            )

    if not any(needs_publication(output) for output in caller_outputs):
        return
    if len(body_outputs) != len(caller_outputs):
        raise ValueError("Selected callable output arity is inconsistent.")
    for body_output, caller_output in zip(
        body_outputs,
        caller_outputs,
        strict=True,
    ):
        publish(body_output, caller_output)


def _if_merge_captured_allocations(
    operations: Sequence[Operation],
    merge_values: Sequence[ValueBase],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str],
) -> dict[str, ResourceExpr]:
    """Return outer allocations referenced only by branch merge records.

    A branch can be structurally empty while its merge record selects an
    element of an outer array. Such a value is not visible to ordinary capture
    analysis, but its complete root owner must remain live. Values produced by
    the branch are deliberately excluded: local QInit sites are counted by the
    branch estimate and must never be reclassified as caller-owned inputs.

    Args:
        operations (Sequence[Operation]): Operations in one conditional branch.
        merge_values (Sequence[ValueBase]): Quantum merge sources selected from
            that branch.
        resolver (ExprResolver): Resolver for symbolic root dimensions.
        allocation_owners_by_uuid (Mapping[str, str]): Known QInit result UUIDs
            mapped to root allocation owners.

    Returns:
        dict[str, ResourceExpr]: Captured outer owner capacities omitted from
        the branch operation list.
    """
    produced = {
        result.uuid
        for operation in operations
        for result in operation.results
        if isinstance(result, Value)
    }
    local_owners = {
        _quantum_allocation_owner(result)
        for operation in operations
        if isinstance(operation, QInitOperation)
        for result in operation.results
        if isinstance(result, Value) and result.type.is_quantum()
    }
    candidates = [
        value
        for value in merge_values
        if isinstance(value, Value)
        and value.type.is_quantum()
        and value.uuid not in produced
        and allocation_owners_by_uuid.get(
            value.uuid,
            _quantum_allocation_owner(value),
        )
        not in local_owners
    ]
    return _quantum_owner_capacities(
        candidates,
        resolver,
        allocation_owners_by_uuid,
    )


def _with_conservative_loop_output_liveness(
    estimate: ResourceEstimate,
    *,
    active_when: sp.Basic,
    source: str,
) -> ResourceEstimate:
    """Mark a loop output summary that retains a conservative owner maximum.

    Args:
        estimate (ResourceEstimate): Loop estimate to annotate.
        active_when (sp.Basic): Condition under which the maximum can exceed
            the exact post-loop live state.
        source (str): Assumption source label.

    Returns:
        ResourceEstimate: Estimate with guarded conservative-quality metadata.
    """
    assumption = ResourceAssumption(
        "post-loop qubit liveness retains a conservative owner width because "
        "an inter-iteration release could not be proven",
        source=source,
    )
    return estimate._with_metadata(
        assumptions=(assumption,),
        quality=EstimateQuality.CONSERVATIVE,
        active_when=active_when,
    )


def _require_uncontrolled_operation(
    operation: Operation,
    controls: ResourceExpr | int,
) -> None:
    """Reject an operation unsupported inside a controlled unitary.

    Args:
        operation (Operation): Operation reached under coherent controls.
        controls (ResourceExpr | int): Surrounding coherent-control count.

    Raises:
        ValueError: If one or more surrounding controls may be active.
    """
    control_count = _expr(controls)
    if control_count != _ZERO:
        raise ValueError(
            f"Cannot estimate controlled {type(operation).__name__}: the "
            "shared controlled decomposition does not support this operation inside a "
            "controlled unitary."
        )
