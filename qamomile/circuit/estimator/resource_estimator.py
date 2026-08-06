"""Estimate algorithmic resources by abstractly interpreting qkernel IR."""

from __future__ import annotations

import dataclasses
import enum
import itertools
import math
import numbers
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.logic.boolalg import Boolean

from qamomile.circuit._array_shape import (
    _ARRAY_PROTOCOL_ERRORS,
    _rectangular_array_shape,
)
from qamomile.circuit.estimator._control_decomposition import (
    CLEAN_ANCILLA_BATCH_MIN_WORK,
    static_clean_ancilla_batch_profile,
)
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._metrics import (
    _SYMPY_SIMPLIFICATION_ERRORS,
    ApproximationStatus,
    CallResources,
    ControlDecomposition,
    DepthResources,
    EstimateDerivation,
    EstimateQuality,
    GateBasis,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceExpr,
    ResourceTraceNode,
    WidthResources,
    _activation_over_range,
    _active_approximation,
    _active_assumptions,
    _active_derivation,
    _active_quality,
    _add_calls,
    _add_depth,
    _add_gates,
    _add_measurements,
    _add_resets,
    _and_conditions,
    _boolean_condition,
    _combine_approximation,
    _combine_derivation,
    _combine_quality,
    _conditional_calls,
    _conditional_depth,
    _conditional_gates,
    _conditional_measurements,
    _conditional_resets,
    _conditional_trace,
    _conditional_width,
    _ConditionIndicator,
    _depth_from_gate_resources,
    _expr,
    _GuardedApproximation,
    _GuardedAssumption,
    _GuardedDerivation,
    _GuardedQuality,
    _has_large_concrete_sum,
    _is_concrete_integer,
    _max_calls,
    _max_depth,
    _max_gates,
    _max_measurements,
    _max_resets,
    _max_width,
    _merge_trace,
    _parallel_width,
    _piecewise,
    _resource_activity_condition,
    _resource_expr,
    _resource_max,
    _ResourceConstraint,
    _safe_constraint_substitute,
    _safe_simplify,
    _scale_calls,
    _scale_depth,
    _scale_gates,
    _scale_measurements,
    _scale_resets,
    _seq_width,
    _substitute_resource_expr,
    _sum_calls,
    _sum_depth,
    _sum_expr,
    _sum_gates,
    _sum_measurements,
    _sum_resets,
    _symbol_display_name,
    _unresolved_condition_guard,
    _wrap_trace,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
    UnresolvedValueError,
    _ArrayState,
    _fact_from_expression,
    _ResolvedClassicalFact,
    input_shape_dimension_aliases,
)
from qamomile.circuit.estimator._scheduling import (
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    _UNKNOWN_WIRE_INDEX,
    WireKey,
    _activate_allocation_sites,
    _aggregate_completion_overlap_condition,
    _anonymous_allocation_width,
    _block_input_allocations,
    _branch_owner_sizes,
    _branch_width_with_static_allocations,
    _captured_quantum_allocations,
    _classical_dependency_footprint,
    _classical_dependency_key,
    _concrete_loop_dependency_completion,
    _controlled_u_control_wire_keys,
    _count_qinit,
    _definitely_consumed_captured_allocations,
    _dependency_depth,
    _dependency_keys_depend_on_symbol,
    _disjoint_concrete_loop_depth,
    _estimate_has_nonzero_depth,
    _expand_dependency_owner_aliases,
    _invoke_quantum_output_sizes,
    _liveness_width,
    _LocalBlock,
    _loop_body_has_symbolic_quantum_index,
    _loop_captured_observation_consumption,
    _map_body_dependency_completion,
    _map_body_dependency_keys,
    _map_value_dependency_keys,
    _maximum_live_owner_sizes,
    _maximum_live_owner_sizes_over_range,
    _maximum_width_over_range,
    _merge_allocation_sites,
    _merge_dependency_keys,
    _namespace_allocation_sites,
    _normalize_wire_index,
    _operation_has_uniform_intrinsic_completion,
    _operation_has_unresolved_quantum_index,
    _quantum_allocation_owner,
    _quantum_owner_capacities,
    _quantum_result_owner_sizes,
    _quantum_wire_keys,
    _qubit_value_size,
    _root_callable_resource_attrs,
    _root_callable_shape_inputs,
    _scheduled_depth_activity_conditions,
    _specialize_dependency_expression,
    _symbolic_disjoint_loop_depth,
    _symbolic_wire_range_index,
    _uniform_parallel_loop_dependency_completion,
    _width_with_identity_aware_allocations,
    _wire_keys_for_values,
    _WireFootprint,
    _WireRangeIndex,
    _with_body_boundary_depth_metadata,
    _with_operation_output_summary,
    _without_input_allocation_sites,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.dataflow import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
    has_legacy_scalar_bit_rebinds,
    walk_operations,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.callable import (
    CallableBodySelection,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    LoopCarriedRebind,
    WhileOperation,
)
from qamomile.circuit.ir.operation.control_work import (
    ControlWorkKind,
    classify_control_work,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
    Signature,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    collect_value_like_uuids,
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes.analyze import (
    reject_loop_carried_classical_rebinds,
)
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    lower_compile_time_ifs_preserving_loop_conditions,
)
from qamomile.circuit.transpiler.passes.control_flow_reachability import (
    static_for_items_entries,
)
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder,
    clean_ancilla_toffoli_ladder_or_empty,
)
from qamomile.circuit.transpiler.passes.inline import (
    InlinePass,
    count_inline_invokes,
)

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel

_ZERO = sp.Integer(0)
_ONE = sp.Integer(1)
_OBSERVATION_SOURCE_PREFIX = "$observation"
_LOOP_ARRAY_SOURCE_PREFIX = "$loop-array"


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
        callable_attrs (Mapping[str, Any]): Merged callable definition and
            invocation attributes that carry the width declaration.
        source (str): Callable name used in diagnostics.
        array_state_bindings (tuple[tuple[ArrayValue, tuple[ArrayValue, ...]],
            ...]): Caller arrays paired with cloned callee entry values that
            receive simultaneous call-time snapshots.
    """

    constraint_operands: tuple[Value, ...] = ()
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


_DEFAULT_GATE_BASIS = GateBasis.LOGICAL
_DEFAULT_CONTROL_DECOMPOSITION = ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
_DEFAULT_ROTATION_SYNTHESIS_PRECISION = 1e-10
# Circuit transpilation performs one initial inline pass before its 64-round
# recursion loop. The validation copy starts from the original Block, so it
# needs one additional round to cover the same supported concrete depth.
_MAX_LOOP_VALIDATION_INLINE_DEPTH = 65
_DEFER_RESOURCE_SYMBOL_METADATA = ContextVar(
    "qamomile_defer_resource_symbol_metadata",
    default=False,
)
_PHASE_CLASS_CODES = {
    None: 0,
    "z": 1,
    "s": 2,
    "t": 3,
    "sdg": 4,
    "tdg": 5,
    "p": -1,
}


_CONCRETE_REGION_REPLAY_LIMIT = 64
_MAX_RUNTIME_DOMAIN_ALTERNATIVES = 16
# Global SymPy simplification becomes superlinear on nested branch extrema;
# resource composition has already normalized larger expressions structurally.
_PUBLIC_RESOURCE_SIMPLIFY_NODE_LIMIT = 32


class _CappedRangeSum(sp.Function):
    """Represent a loop-local batching sum with its induction variable bound.

    The control batching policy only needs to distinguish zero, one, and at
    least two units of work. Keeping the summand inside a ``Lambda`` prevents
    SymPy from releasing a ``Sum`` dummy into public parameters when a
    surrounding ``Piecewise`` is simplified.
    """

    nargs = 4
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(
        cls,
        summand: sp.Basic,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
    ) -> sp.Expr | None:
        """Evaluate a capped sum once its loop range is concrete.

        Args:
            summand (sp.Basic): One-argument Lambda for per-iteration work.
            start (sp.Expr): First Python-range value.
            step (sp.Expr): Python-range step.
            iterations (sp.Expr): Number of executed iterations.

        Returns:
            sp.Expr | None: Integer in ``[0, 2]`` when decidable, otherwise
            ``None`` to retain the bound symbolic node.
        """
        if not isinstance(summand, sp.Lambda) or len(summand.variables) != 1:
            return None
        if iterations.is_zero is True:
            return _ZERO
        loop_symbol = summand.variables[0]
        expression = cast(sp.Expr, summand.expr)
        if loop_symbol not in expression.free_symbols and expression.is_number:
            return cast(sp.Expr, sp.Min(2, expression * iterations))
        if not all(
            value.is_number and _is_concrete_integer(value)
            for value in (start, step, iterations)
        ):
            return None
        count = int(iterations)
        if count <= 0:
            return _ZERO
        offset = sp.Dummy("batch_offset", integer=True, nonnegative=True)
        transformed = expression.subs(
            loop_symbol,
            start + step * offset,
        )
        capped = _capped_nonnegative_integer_sum(
            transformed,
            offset,
            count,
        )
        if capped is not None:
            return sp.Integer(capped)
        expanded = expression.replace(
            lambda node: isinstance(node, _ConditionIndicator),
            lambda node: sp.Piecewise(
                (_ONE, cast(sp.Basic, node.args[0])),
                (_ZERO, True),
            ),
        )
        transformed = expanded.subs(
            loop_symbol,
            start + step * offset,
        )
        evaluated = cast(
            sp.Expr,
            sp.Sum(transformed, (offset, 0, count - 1)).doit(),
        )
        if evaluated.is_integer is True and evaluated.is_number:
            return sp.Integer(min(2, max(0, int(evaluated))))
        return None


def _simplify_public_resource_expression(
    expression: ResourceExpr,
) -> ResourceExpr:
    """Simplify a public metric unless it contains branching structure.

    ``_CappedRangeSum`` deliberately keeps a loop induction variable inside a
    ``Lambda``, while ``_ConditionIndicator`` keeps a Boolean predicate opaque
    to ``Piecewise`` rewriting. Resource expressions also contain nested
    ``Min``, ``Max``, and ``Piecewise`` nodes that are already structurally
    normalized while they are composed. SymPy's global simplifier spends
    substantial time exploring those nodes, usually returning an expression
    of the same shape and occasionally attempting to release a bound variable.
    Concrete substitution evaluates them directly, so retaining the symbolic
    form is both faster and safer.

    Args:
        expression (ResourceExpr): Public resource expression to normalize.

    Returns:
        ResourceExpr: Safely simplified expression, or the original
            structurally normalized expression when global simplification is
            unsafe or disproportionately expensive.
    """
    normalized = _expr(expression)
    if _has_large_concrete_sum(normalized):
        return normalized
    has_branching_structure = False
    for node_count, node in enumerate(
        sp.preorder_traversal(normalized),
        start=1,
    ):
        if isinstance(node, (_CappedRangeSum, _ConditionIndicator)):
            return normalized
        if isinstance(node, (sp.Max, sp.Min, sp.Piecewise)):
            has_branching_structure = True
        if (
            has_branching_structure
            and node_count > _PUBLIC_RESOURCE_SIMPLIFY_NODE_LIMIT
        ):
            return normalized
    return _safe_simplify(normalized)


def _capped_nonnegative_integer_sum(
    expression: sp.Expr,
    symbol: sp.Symbol,
    count: int,
) -> int | None:
    """Return a nonnegative integer range sum capped at two.

    Args:
        expression (sp.Expr): Per-position nonnegative integer expression.
        symbol (sp.Symbol): Zero-based range-position symbol.
        count (int): Number of positions.

    Returns:
        int | None: Exact capped sum, or ``None`` when the expression grammar
        cannot be decided without expanding the range.
    """
    if count <= 0 or expression == _ZERO:
        return 0
    if symbol not in expression.free_symbols and expression.is_number:
        return min(2, max(0, int(expression) * count))
    if isinstance(expression, _ConditionIndicator):
        return _capped_integer_condition_count(
            cast(sp.Basic, expression.args[0]),
            symbol,
            count,
        )
    coefficient, remainder = expression.as_coeff_Mul()
    if (
        coefficient.is_integer is True
        and coefficient.is_nonnegative is True
        and isinstance(remainder, _ConditionIndicator)
    ):
        active_count = _capped_integer_condition_count(
            cast(sp.Basic, remainder.args[0]),
            symbol,
            count,
        )
        if active_count is None:
            return None
        return min(2, int(coefficient) * active_count)
    if isinstance(expression, sp.Add):
        total = 0
        for term in expression.args:
            term_sum = _capped_nonnegative_integer_sum(
                cast(sp.Expr, term),
                symbol,
                count,
            )
            if term_sum is None:
                return None
            total = min(2, total + term_sum)
            if total == 2:
                return 2
        return total
    if isinstance(expression, sp.Min) and _expr(2) in expression.args:
        remaining = [arg for arg in expression.args if arg != _expr(2)]
        if len(remaining) == 1:
            return _capped_nonnegative_integer_sum(
                cast(sp.Expr, remaining[0]),
                symbol,
                count,
            )
    return None


def _capped_integer_condition_count(
    condition: sp.Basic,
    symbol: sp.Symbol,
    count: int,
) -> int | None:
    """Count satisfying integer range positions, capped at two.

    Args:
        condition (sp.Basic): Boolean condition over ``symbol``.
        symbol (sp.Symbol): Zero-based integer position.
        count (int): Exclusive upper bound.

    Returns:
        int | None: Exact capped cardinality, or ``None`` for an unsupported
        symbolic set.
    """
    normalized = _boolean_condition(condition)
    if normalized is sp.false or count <= 0:
        return 0
    if normalized is sp.true:
        return min(2, count)
    if normalized.free_symbols - {symbol}:
        return None
    try:
        satisfying = sp.Intersection(
            normalized.as_set(),
            sp.Range(0, count),
        )
    except (
        ArithmeticError,
        AttributeError,
        NotImplementedError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return None
    return _capped_integer_set_cardinality(satisfying)


def _capped_integer_set_cardinality(values: sp.Set) -> int | None:
    """Return an integer set's cardinality capped at two.

    Args:
        values (sp.Set): Integer-valued set.

    Returns:
        int | None: Exact capped cardinality, or ``None`` when unavailable.
    """
    if values is sp.S.EmptySet or values.is_empty is True:
        return 0
    if isinstance(values, sp.FiniteSet):
        return min(2, len(values))
    if isinstance(values, sp.Range):
        size = values.size
        if size.is_integer is True and size.is_number:
            return min(2, max(0, int(size)))
        return None
    if isinstance(values, sp.Union):
        total = 0
        for subset in values.args:
            subset_count = _capped_integer_set_cardinality(cast(sp.Set, subset))
            if subset_count is None:
                return None
            total = min(2, total + subset_count)
            if total == 2:
                return 2
        return total
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class _EstimatorControlBatchProfile:
    """Track symbolic work needed to choose a shared control carrier.

    ``work`` is capped at two because the fixed resource model only
    distinguishes empty, singleton, and multi-operation bodies.

    Args:
        work (ResourceExpr | int): Symbolic controlled work, capped at two.
    """

    work: ResourceExpr | int = _ZERO

    def __post_init__(self) -> None:
        """Normalize the work expression."""
        object.__setattr__(self, "work", _expr(self.work))

    @property
    def active(self) -> Boolean:
        """Return whether this profile contains any controlled work.

        Returns:
            sp.Basic: Symbolic nonempty-work predicate.
        """
        work = cast(ResourceExpr, self.work)
        if work == _ZERO:
            return sp.false
        if work.is_positive is True:
            return sp.true
        if isinstance(work, _ConditionIndicator):
            return _boolean_condition(work.args[0])
        coefficient, remainder = work.as_coeff_Mul()
        if coefficient.is_positive and isinstance(remainder, _ConditionIndicator):
            return _boolean_condition(remainder.args[0])
        return _boolean_condition(sp.Gt(work, _ZERO))

    @property
    def has_multiple_work(self) -> Boolean:
        """Return whether the profile contains at least two work units.

        Returns:
            sp.Basic: Symbolic batching-threshold predicate.
        """
        work = cast(ResourceExpr, self.work)
        if work.is_number:
            return sp.true if work >= 2 else sp.false
        coefficient, remainder = work.as_coeff_Mul()
        if coefficient >= 2 and isinstance(remainder, _ConditionIndicator):
            return _boolean_condition(remainder.args[0])
        if isinstance(work, sp.Min) and _expr(2) in work.args:
            remaining = [arg for arg in work.args if arg != _expr(2)]
            if len(remaining) == 1:
                return _boolean_condition(sp.Ge(remaining[0], 2))
        return _boolean_condition(sp.Ge(work, 2))

    def when(self, condition: sp.Basic) -> _EstimatorControlBatchProfile:
        """Guard this profile by one symbolic activation condition.

        Args:
            condition (sp.Basic): Condition under which the work executes.

        Returns:
            _EstimatorControlBatchProfile: Conditionally active profile.
        """
        predicate = _boolean_condition(condition)
        indicator = cast(ResourceExpr, _ConditionIndicator(predicate))
        return _EstimatorControlBatchProfile(
            work=cast(ResourceExpr, self.work) * indicator,
        )

    def as_shared_leaf(self) -> _EstimatorControlBatchProfile:
        """Promote any active work to one independently batch-worthy leaf.

        Nested controlled calls, SELECT cases, and Pauli evolution already
        contain an intrinsic control boundary. One active instance therefore
        justifies composing that boundary with an outer shared carrier.

        Returns:
            _EstimatorControlBatchProfile: Heavy profile guarded by activity.
        """
        active = self.active
        indicator = cast(ResourceExpr, _ConditionIndicator(active))
        return _EstimatorControlBatchProfile(
            work=2 * indicator,
        )

    def conditional(
        self,
        other: _EstimatorControlBatchProfile,
        condition: sp.Basic,
    ) -> _EstimatorControlBatchProfile:
        """Select between two profiles with a symbolic condition.

        Args:
            other (_EstimatorControlBatchProfile): False-branch profile.
            condition (sp.Basic): Predicate selecting ``self`` when true.

        Returns:
            _EstimatorControlBatchProfile: Exact symbolic branch profile.
        """
        predicate = _boolean_condition(condition)
        true_indicator = cast(ResourceExpr, _ConditionIndicator(predicate))
        false_indicator = cast(
            ResourceExpr,
            _ConditionIndicator(sp.Not(predicate)),
        )
        return _EstimatorControlBatchProfile(
            work=(
                cast(ResourceExpr, self.work) * true_indicator
                + cast(ResourceExpr, other.work) * false_indicator
            ),
        )

    def choice(
        self,
        other: _EstimatorControlBatchProfile,
    ) -> _EstimatorControlBatchProfile:
        """Take the conservative work maximum of two possible branches.

        Args:
            other (_EstimatorControlBatchProfile): Alternative branch profile.

        Returns:
            _EstimatorControlBatchProfile: Field-wise branch maximum, capped
                at the shared-ladder threshold.
        """
        return _EstimatorControlBatchProfile(
            work=sp.Min(2, sp.Max(self.work, other.work)),
        )

    def sum_over(
        self,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
    ) -> _EstimatorControlBatchProfile:
        """Accumulate a profile over Python ``range`` semantics.

        Args:
            loop_symbol (sp.Symbol): Symbolic loop induction variable.
            start (ResourceExpr): First loop value.
            step (ResourceExpr): Loop step.
            iterations (ResourceExpr): Number of executed iterations.

        Returns:
            _EstimatorControlBatchProfile: Capped aggregate loop profile.
        """
        work = _CappedRangeSum(
            sp.Lambda(loop_symbol, cast(ResourceExpr, self.work)),
            start,
            step,
            iterations,
        )
        return _EstimatorControlBatchProfile(work=work)

    @classmethod
    def combine(
        cls,
        profiles: Iterable[_EstimatorControlBatchProfile],
    ) -> _EstimatorControlBatchProfile:
        """Combine sequential operation profiles with capped work.

        Args:
            profiles (Iterable[_EstimatorControlBatchProfile]): Profiles in
                program order.

        Returns:
            _EstimatorControlBatchProfile: Combined symbolic body profile.
        """
        work_items: list[ResourceExpr] = []
        for profile in profiles:
            profile_work = cast(ResourceExpr, profile.work)
            if profile_work != _ZERO:
                work_items.append(profile_work)
        if not work_items:
            work = _ZERO
        elif len(work_items) == 1:
            work = work_items[0]
        else:
            work = cast(ResourceExpr, sp.Min(2, sp.Add(*work_items)))
        return cls(work=work)


def _clean_ancilla_shared_ladder_condition(
    controls: ResourceExpr,
    profile: _EstimatorControlBatchProfile,
) -> sp.Basic:
    """Return when the fixed model shares one control-condition ladder.

    Both body-backed calls and aggregate opaque costs must use this exact
    threshold so an invocation boundary cannot change the selected resource
    recipe.

    Args:
        controls (ResourceExpr): Number of surrounding coherent controls.
        profile (_EstimatorControlBatchProfile): Capped controlled work.

    Returns:
        sp.Basic: Symbolic shared-ladder selection predicate.
    """
    return _boolean_condition(
        sp.And(
            sp.Ge(controls, CLEAN_ANCILLA_BATCH_MIN_WORK),
            profile.has_multiple_work,
        )
    )


def _clean_ancilla_aggregate_control_profile(
    estimate: ResourceEstimate,
) -> _EstimatorControlBatchProfile:
    """Return controlled work declared by one aggregate opaque cost.

    The profile is shared by the enclosing-body preflight and the aggregate
    projection itself. This ensures both stages choose the same per-primitive
    or shared-ladder recipe after symbolic specialization.

    Args:
        estimate (ResourceEstimate): Definition-level aggregate cost.

    Returns:
        _EstimatorControlBatchProfile: Total modeled work, capped at the
            sharing threshold, when known one- or two-qubit work makes the
            aggregate projection eligible.
    """
    if _aggregate_arity_projection_reason(estimate) is not None:
        return _EstimatorControlBatchProfile()
    known_arity_work = estimate.gates.single_qubit + estimate.gates.two_qubit
    return _EstimatorControlBatchProfile(
        work=sp.Min(
            CLEAN_ANCILLA_BATCH_MIN_WORK,
            estimate.gates.total,
        )
    ).when(sp.Gt(known_arity_work, _ZERO))


class _CanonicalPhaseClass(sp.Function):
    """Classify a phase after numeric substitution without erasing small phases."""

    nargs = 1

    @classmethod
    def eval(cls, phase: sp.Expr) -> sp.Integer | None:
        """Evaluate a concrete phase to its canonical gate-class code.

        Args:
            phase (sp.Expr): Phase angle in radians.

        Returns:
            sp.Integer | None: Canonical class for a numeric phase, or
            ``None`` to retain a symbolic function application.
        """
        normalized = cast(sp.Expr, sp.sympify(phase))
        if not normalized.is_number:
            return None
        gate_name = _canonical_phase_gate_name(normalized)
        return sp.Integer(_PHASE_CLASS_CODES[gate_name])


def _normalize_resource_scalar(
    value: object,
    *,
    label: str,
    allow_symbolic: bool,
    allow_bool: bool,
) -> sp.Expr:
    """Normalize a public resource scalar and reject invalid number domains.

    Args:
        value (object): Candidate Python or SymPy scalar.
        label (str): User-facing parameter label for diagnostics.
        allow_symbolic (bool): Whether a non-numeric SymPy expression may
            remain in the estimate.
        allow_bool (bool): Whether a Boolean may be normalized to zero or one.

    Returns:
        sp.Expr: Normalized finite-real scalar or permitted symbolic
            expression.

    Raises:
        TypeError: If the value is Boolean when disallowed, is a string, or is
            not a supported numeric/SymPy scalar.
        ValueError: If a concrete number is non-real or non-finite, or a
            symbolic expression is provably non-real or non-finite.
    """
    expected = (
        "a numeric scalar or explicit SymPy expression"
        if allow_symbolic
        else "a concrete numeric scalar"
    )
    if isinstance(value, bool):
        if allow_bool:
            return sp.Integer(int(value))
        raise TypeError(f"{label} requires {expected}, got bool ({value!r}).")
    if isinstance(value, (str, bytes)) or not (
        isinstance(value, numbers.Complex) or isinstance(value, sp.Expr)
    ):
        raise TypeError(
            f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
        )
    scalar = value.item() if hasattr(value, "item") else value
    try:
        normalized = sp.sympify(scalar)
    except (TypeError, ValueError, sp.SympifyError) as error:
        raise TypeError(
            f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
        ) from error
    if not isinstance(normalized, sp.Expr):
        raise TypeError(
            f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
        )
    if normalized.is_number is not True:
        if not allow_symbolic:
            raise TypeError(
                f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
            )
        if normalized.is_real is False or normalized.is_finite is False:
            raise ValueError(f"{label} must be finite and real, got {value!r}.")
        return cast(sp.Expr, normalized)
    if normalized.is_real is not True or normalized.is_finite is not True:
        raise ValueError(f"{label} must be finite and real, got {value!r}.")
    return cast(sp.Expr, normalized)


def _canonicalize_concrete_integer(value: sp.Expr) -> sp.Expr:
    """Return an exact SymPy integer for an integer-valued concrete number.

    Python and NumPy integer-valued floats are accepted for integer resource
    parameters. Canonicalizing them at the public boundary prevents SymPy's
    undecided ``Float.is_integer`` property from retaining internal loop nodes
    after all user parameters have been substituted.

    Args:
        value (sp.Expr): Concrete or symbolic resource scalar.

    Returns:
        sp.Expr: Exact integer for a concrete integral value, otherwise the
            original expression.
    """
    if value.is_number and _is_concrete_integer(value):
        return sp.Integer(int(value))
    return value


def _typed_value_symbol(
    value: ValueBase,
    name: str,
    *,
    fresh: bool = False,
) -> sp.Symbol:
    """Create a symbolic scalar matching an IR value's domain.

    Args:
        value (ValueBase): IR value whose type supplies the symbol
            assumptions.
        name (str): Human-readable symbol name.
        fresh (bool): Whether to create an identity-distinct ``Dummy``
            instead of a same-name ``Symbol``. Defaults to False.

    Returns:
        sp.Symbol: A real symbol for Float, a nonnegative integer for
            UInt/Bit, or an unconstrained symbol for other value types.
    """
    factory = sp.Dummy if fresh else sp.Symbol
    if isinstance(value.type, FloatType):
        return factory(name, real=True)
    if isinstance(value.type, (BitType, UIntType)):
        return factory(name, integer=True, nonnegative=True)
    return factory(name)


class UnknownResourcePolicy(enum.StrEnum):
    """Control how the estimator handles bodyless unknown callables.

    Values:
        ERROR: Raise when a callable has neither a body nor an opaque cost.
        OPAQUE_CALL: Count one opaque call/query and continue.
        ZERO_WITH_WARNING: Record an assumption and continue with zero cost.
    """

    ERROR = "error"
    OPAQUE_CALL = "opaque_call"
    ZERO_WITH_WARNING = "zero_with_warning"


@dataclasses.dataclass
class ResourceEstimate:
    """Carry the full algorithmic resource estimate for a qkernel or block.

    Args:
        width (WidthResources): Logical width and ancilla estimate.
        gates (GateResources): Logical gate-resource estimate.
        depth (DepthResources): Logical depth-resource estimate.
        calls (CallResources): Callable/query-resource estimate.
        measurements (MeasurementResources): Per-qubit measurement resources.
        resets (ResetResources): Per-qubit reset resources.
        assumptions (tuple[ResourceAssumption, ...]): Modeling assumptions.
        trace (ResourceTraceNode | None): Explanation tree root. Defaults to
            ``None``.
        parameters (dict[str, sp.Symbol]): Symbols present in the estimate,
            keyed by unique public aliases. Defaults to an empty dict.
        derivation (EstimateDerivation): Whether counts are derived from
            visible structure or use a resource model. Defaults to
            ``STRUCTURAL``.
        quality (EstimateQuality): Relationship between reported counts
            and the selected circuit cost. Defaults to ``EXACT``.
        approximation (ApproximationStatus): Whether the selected circuit
            approximates an ideal mathematical operation. Defaults to
            ``EXACT``.
        basis (GateBasis): Gate basis used for the estimate. Defaults to the
            logical algorithmic basis.
        control_decomposition (ControlDecomposition): Coherent-control
            decomposition used for the estimate. Defaults to the clean-ancilla
            Toffoli model.
        precision (float | None): Rotation-synthesis precision when the basis
            uses approximate synthesis. Defaults to ``None``.
        _allocation_sites (dict[str, ResourceExpr]): Internal QInit-site sizes
            keyed by stable operation-result UUID. Concrete loop evaluation
            uses this identity map to count one static allocation site once
            even when the body is replayed across multiple iterations.
        _constraints (tuple[_ResourceConstraint, ...]): Internal structural
            requirements retained across symbolic substitution.
        _output_sizes (dict[str, ResourceExpr]): Internal live quantum output
            widths keyed by root allocation owner for nested control-flow
            operations.
        _input_sizes (dict[str, ResourceExpr]): Internal captured quantum input
            widths consumed or replaced by nested control-flow operations.
        _has_output_summary (bool): Whether ``_output_sizes`` is authoritative,
            including when a nested operation has no live quantum outputs.
        _dependency_keys (frozenset[WireKey] | None): Internal caller-scoped
            quantum wires that contribute nonzero depth. ``None`` requests the
            enclosing operation's conservative ordinary footprint.
        _dependency_reads (frozenset[WireKey] | None): Internal scheduler
            inputs for rescheduling an already interpreted body, including
            immutable classical observation tokens. ``None`` means that only
            ``_dependency_keys`` is available.
        _dependency_writes (frozenset[WireKey] | None): Internal scheduler
            outputs for rescheduling an already interpreted body, including
            newly published observation tokens. ``None`` means that only
            ``_dependency_keys`` is available.
        _dependency_completion (dict[WireKey, ResourceExpr] | None): Internal
            caller-visible completion depth for each dependency wire.
            ``None`` requests conservative reconstruction from
            ``_dependency_keys`` or the enclosing operation footprint.
        _dependency_completion_uniform (bool | None): Whether every
            caller-visible wire is proven to complete at the aggregate peak
            of every depth field. ``None`` means that field-wise uniformity
            was not proven.
        _global_barrier_condition (Boolean): Condition under which an opaque,
            nested non-unitary, or runtime-control boundary lacks enough
            wire-level provenance for exact dependency scheduling.
        _measurement_taint_conditions (dict[str, Boolean]): Estimator-local
            conditions under which classical SSA values derive from runtime
            quantum observations. This state is used only while recursively
            interpreting a body and is not a public resource metric.
        _guarded_assumptions (tuple[_GuardedAssumption, ...] | None): Internal
            condition-aware assumption provenance. ``None`` initializes facts
            from the public ``assumptions`` tuple.
        _guarded_derivations (tuple[_GuardedDerivation, ...] | None): Internal
            condition-aware modeled-derivation provenance. ``None`` initializes
            a fact from the public ``derivation`` value.
        _guarded_qualities (tuple[_GuardedQuality, ...] | None): Internal
            condition-aware non-exact count qualities. ``None`` initializes a
            fact from the public ``quality`` value.
        _guarded_approximations (tuple[_GuardedApproximation, ...] | None):
            Internal condition-aware mathematical approximation provenance.
            ``None`` initializes a fact from the public ``approximation``
            value.
        _symbol_aliases (dict[sp.Symbol, str]): Internal stable public aliases
            retained across expression rewrites and partial substitution.
    """

    width: WidthResources = dataclasses.field(default_factory=WidthResources.zero)
    gates: GateResources = dataclasses.field(default_factory=GateResources.zero)
    depth: DepthResources = dataclasses.field(default_factory=DepthResources.zero)
    calls: CallResources = dataclasses.field(default_factory=CallResources.zero)
    assumptions: tuple[ResourceAssumption, ...] = ()
    trace: ResourceTraceNode | None = None
    parameters: dict[str, sp.Symbol] = dataclasses.field(default_factory=dict)
    derivation: EstimateDerivation = EstimateDerivation.STRUCTURAL
    quality: EstimateQuality = EstimateQuality.EXACT
    approximation: ApproximationStatus = ApproximationStatus.EXACT
    basis: GateBasis = _DEFAULT_GATE_BASIS
    control_decomposition: ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION
    precision: float | None = None
    measurements: MeasurementResources = dataclasses.field(
        default_factory=MeasurementResources.zero
    )
    resets: ResetResources = dataclasses.field(default_factory=ResetResources.zero)
    _allocation_sites: dict[str, ResourceExpr] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _constraints: tuple[_ResourceConstraint, ...] = dataclasses.field(
        default_factory=tuple,
        repr=False,
    )
    _output_sizes: dict[str, ResourceExpr] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _input_sizes: dict[str, ResourceExpr] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _has_output_summary: bool = dataclasses.field(
        default=False,
        repr=False,
        compare=False,
    )
    _dependency_keys: frozenset[WireKey] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _dependency_reads: frozenset[WireKey] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _dependency_writes: frozenset[WireKey] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _dependency_completion: dict[WireKey, ResourceExpr] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _dependency_completion_uniform: bool | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _global_barrier_condition: Boolean = dataclasses.field(
        default=sp.false,
        repr=False,
        compare=False,
    )
    _measurement_taint_conditions: dict[str, Boolean] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _guarded_assumptions: tuple[_GuardedAssumption, ...] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _guarded_derivations: tuple[_GuardedDerivation, ...] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _guarded_qualities: tuple[_GuardedQuality, ...] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _guarded_approximations: tuple[_GuardedApproximation, ...] | None = (
        dataclasses.field(
            default=None,
            repr=False,
            compare=False,
        )
    )
    _symbol_aliases: dict[sp.Symbol, str] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalize guarded metadata and derive the public parameter map."""
        self._constraints = tuple(
            constraint
            for constraint in self._constraints
            if constraint.active_when is not sp.false
        )
        self._global_barrier_condition = _boolean_condition(
            self._global_barrier_condition
        )
        if self._guarded_assumptions is None:
            self._guarded_assumptions = tuple(
                _GuardedAssumption(sp.true, assumption)
                for assumption in self.assumptions
            )
        else:
            active_assumptions = _active_assumptions(self._guarded_assumptions)
            self._guarded_assumptions = (
                *self._guarded_assumptions,
                *(
                    _GuardedAssumption(sp.true, assumption)
                    for assumption in self.assumptions
                    if assumption not in active_assumptions
                ),
            )
        if self._guarded_derivations is None:
            self._guarded_derivations = (
                (_GuardedDerivation(sp.true, self.derivation),)
                if self.derivation is not EstimateDerivation.STRUCTURAL
                else ()
            )
        elif _combine_derivation(
            _active_derivation(self._guarded_derivations),
            self.derivation,
        ) is self.derivation and self.derivation is not _active_derivation(
            self._guarded_derivations
        ):
            self._guarded_derivations = (
                *self._guarded_derivations,
                _GuardedDerivation(sp.true, self.derivation),
            )
        self.assumptions = _active_assumptions(self._guarded_assumptions)
        self.derivation = _active_derivation(self._guarded_derivations)
        if self._guarded_qualities is None:
            self._guarded_qualities = (
                (_GuardedQuality(sp.true, self.quality),)
                if self.quality is not EstimateQuality.EXACT
                else ()
            )
        elif _combine_quality(
            _active_quality(self._guarded_qualities),
            self.quality,
        ) is self.quality and self.quality is not _active_quality(
            self._guarded_qualities
        ):
            self._guarded_qualities = (
                *self._guarded_qualities,
                _GuardedQuality(sp.true, self.quality),
            )
        self.quality = _active_quality(self._guarded_qualities)
        if self._guarded_approximations is None:
            self._guarded_approximations = (
                (_GuardedApproximation(sp.true, self.approximation),)
                if self.approximation is not ApproximationStatus.EXACT
                else ()
            )
        elif _combine_approximation(
            _active_approximation(self._guarded_approximations),
            self.approximation,
        ) is self.approximation and self.approximation is not _active_approximation(
            self._guarded_approximations
        ):
            self._guarded_approximations = (
                *self._guarded_approximations,
                _GuardedApproximation(sp.true, self.approximation),
            )
        self.approximation = _active_approximation(self._guarded_approximations)
        if not _DEFER_RESOURCE_SYMBOL_METADATA.get():
            self._refresh_symbol_metadata()

    def _refresh_symbol_metadata(
        self,
        registry: SymbolRegistry | None = None,
    ) -> SymbolRegistry:
        """Derive stable public aliases and the parameter map.

        Args:
            registry (SymbolRegistry | None): Precomputed registry for this
                exact estimate. Defaults to rebuilding one from all resource
                expressions and structural requirements.

        Returns:
            SymbolRegistry: Registry used to refresh the public metadata.
        """
        active_registry = registry or _serialization_registry(self)
        self._symbol_aliases = active_registry.aliases()
        self.parameters = _collect_parameters(self, active_registry)
        return active_registry

    def _with_metadata(
        self,
        *,
        assumptions: Sequence[ResourceAssumption] = (),
        derivation: EstimateDerivation = EstimateDerivation.STRUCTURAL,
        quality: EstimateQuality = EstimateQuality.EXACT,
        approximation: ApproximationStatus = ApproximationStatus.EXACT,
        active_when: sp.Basic = sp.true,
    ) -> ResourceEstimate:
        """Append guarded assumption, derivation, quality, and approximation.

        Args:
            assumptions (Sequence[ResourceAssumption]): Assumptions to append.
                Defaults to none.
            derivation (EstimateDerivation): Derivation fact to append.
                ``STRUCTURAL`` adds no fact. Defaults to ``STRUCTURAL``.
            quality (EstimateQuality): Count quality to append. ``EXACT``
                adds no fact. Defaults to ``EXACT``.
            approximation (ApproximationStatus): Mathematical approximation
                fact to append. ``EXACT`` adds no fact. Defaults to ``EXACT``.
            active_when (sp.Basic): Activation condition shared by the new
                facts. Defaults to true.

        Returns:
            ResourceEstimate: Copy with condition-aware metadata appended.
        """
        condition = _boolean_condition(active_when)
        guarded_assumptions = self._guarded_assumptions or ()
        guarded_derivations = self._guarded_derivations or ()
        guarded_qualities = self._guarded_qualities or ()
        guarded_approximations = self._guarded_approximations or ()
        return dataclasses.replace(
            self,
            _guarded_assumptions=(
                *guarded_assumptions,
                *(
                    _GuardedAssumption(condition, assumption)
                    for assumption in assumptions
                ),
            ),
            _guarded_derivations=(
                *guarded_derivations,
                *(
                    (_GuardedDerivation(condition, derivation),)
                    if derivation is not EstimateDerivation.STRUCTURAL
                    else ()
                ),
            ),
            _guarded_qualities=(
                *guarded_qualities,
                *(
                    (_GuardedQuality(condition, quality),)
                    if quality is not EstimateQuality.EXACT
                    else ()
                ),
            ),
            _guarded_approximations=(
                *guarded_approximations,
                *(
                    (_GuardedApproximation(condition, approximation),)
                    if approximation is not ApproximationStatus.EXACT
                    else ()
                ),
            ),
        )

    @property
    def qubits(self) -> ResourceExpr:
        """Return the peak logical qubit alias.

        Returns:
            ResourceExpr: Same value as ``width.peak_qubits``.
        """
        return self.width.peak_qubits

    @property
    def circuit_qubits(self) -> ResourceExpr:
        """Return the conservative static circuit-width alias.

        Returns:
            ResourceExpr: Same value as ``width.circuit_qubits``.
        """
        return self.width.circuit_qubits

    @staticmethod
    def zero(trace_name: str | None = None) -> ResourceEstimate:
        """Return an empty resource estimate.

        Args:
            trace_name (str | None): Optional trace-node name for the empty
                estimate. Defaults to ``None``.

        Returns:
            ResourceEstimate: Zero-valued estimate.
        """
        trace = (
            ResourceTraceNode(trace_name, "body", summary="0")
            if trace_name is not None
            else None
        )
        return ResourceEstimate(trace=trace)

    @staticmethod
    def primitive(
        name: str,
        gates: GateResources | None = None,
        *,
        width: WidthResources | None = None,
        depth: DepthResources | None = None,
    ) -> ResourceEstimate:
        """Create an estimate for one primitive operation.

        Args:
            name (str): Primitive operation name.
            gates (GateResources | None): Gate resources. Defaults to zero.
            width (WidthResources | None): Width resources. Defaults to zero.
            depth (DepthResources | None): Depth resources. Defaults to one
                layer when gates are non-zero, otherwise zero.

        Returns:
            ResourceEstimate: Primitive estimate with a trace node.
        """
        gate_resources = gates or GateResources.zero()
        if depth is None:
            depth = _depth_from_gate_resources(gate_resources)
        return ResourceEstimate(
            width=width or WidthResources.zero(),
            gates=gate_resources,
            depth=depth,
            trace=ResourceTraceNode(
                name=name,
                source_kind="primitive",
                summary=f"gates={gate_resources.total}",
            ),
        )

    def seq(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose this estimate before another estimate.

        Args:
            other (ResourceEstimate): Estimate that runs after this one.

        Returns:
            ResourceEstimate: Sequentially composed estimate.
        """
        basis, control_decomposition, precision = _merge_estimate_provenance(
            self,
            other,
        )
        return ResourceEstimate(
            width=_seq_width(self.width, other.width),
            gates=_add_gates(self.gates, other.gates),
            depth=_add_depth(self.depth, other.depth),
            calls=_add_calls(self.calls, other.calls),
            measurements=_add_measurements(self.measurements, other.measurements),
            resets=_add_resets(self.resets, other.resets),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("seq", self.trace, other.trace),
            derivation=_combine_derivation(self.derivation, other.derivation),
            quality=_combine_quality(self.quality, other.quality),
            approximation=_combine_approximation(
                self.approximation,
                other.approximation,
            ),
            basis=basis,
            control_decomposition=control_decomposition,
            precision=precision,
            _allocation_sites=_merge_allocation_sites(
                self._allocation_sites,
                other._allocation_sites,
            ),
            _constraints=(*self._constraints, *other._constraints),
            _dependency_keys=_merge_dependency_keys(self, other),
            _dependency_reads=_merge_dependency_accesses(
                self,
                other,
                writes=False,
            ),
            _dependency_writes=_merge_dependency_accesses(
                self,
                other,
                writes=True,
            ),
            _dependency_completion=_seq_dependency_completion(self, other),
            _global_barrier_condition=sp.Or(
                self._global_barrier_condition,
                other._global_barrier_condition,
            ),
            _measurement_taint_conditions=_merge_measurement_taint_conditions(
                self._measurement_taint_conditions,
                other._measurement_taint_conditions,
            ),
            _guarded_assumptions=(
                *(self._guarded_assumptions or ()),
                *(other._guarded_assumptions or ()),
            ),
            _guarded_derivations=(
                *(self._guarded_derivations or ()),
                *(other._guarded_derivations or ()),
            ),
            _guarded_qualities=(
                *(self._guarded_qualities or ()),
                *(other._guarded_qualities or ()),
            ),
            _guarded_approximations=(
                *(self._guarded_approximations or ()),
                *(other._guarded_approximations or ()),
            ),
            _symbol_aliases=_merge_symbol_aliases(self, other),
        )

    @staticmethod
    def seq_all(estimates: Iterable[ResourceEstimate]) -> ResourceEstimate:
        """Compose estimates with a streaming, order-preserving reduction.

        Repeated left-folding copies accumulated guarded metadata at every
        step. The binary-counter reduction preserves :meth:`seq` semantics,
        avoids quadratic copy growth, and retains only logarithmically many
        intermediate estimates while consuming an iterable.

        Args:
            estimates (Iterable[ResourceEstimate]): Estimates in execution
                order.

        Returns:
            ResourceEstimate: Sequential composition, or an exact zero
            estimate for an empty sequence.

        Raises:
            ValueError: If the estimates use incompatible basis or precision
                provenance.
        """
        composer = _SequentialEstimateComposer()
        for estimate in estimates:
            composer.append(estimate)
        return composer.finish()

    def parallel(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose this estimate in parallel with another estimate.

        Args:
            other (ResourceEstimate): Estimate that runs concurrently.

        Returns:
            ResourceEstimate: Parallel composition.
        """
        basis, control_decomposition, precision = _merge_estimate_provenance(
            self,
            other,
        )
        return ResourceEstimate(
            width=_parallel_width(self.width, other.width),
            gates=_add_gates(self.gates, other.gates),
            depth=_max_depth(self.depth, other.depth),
            calls=_add_calls(self.calls, other.calls),
            measurements=_add_measurements(self.measurements, other.measurements),
            resets=_add_resets(self.resets, other.resets),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("parallel", self.trace, other.trace),
            derivation=_combine_derivation(self.derivation, other.derivation),
            quality=_combine_quality(self.quality, other.quality),
            approximation=_combine_approximation(
                self.approximation,
                other.approximation,
            ),
            basis=basis,
            control_decomposition=control_decomposition,
            precision=precision,
            _allocation_sites=_merge_allocation_sites(
                self._allocation_sites,
                other._allocation_sites,
            ),
            _constraints=(*self._constraints, *other._constraints),
            _dependency_keys=_merge_dependency_keys(self, other),
            _dependency_reads=_merge_dependency_accesses(
                self,
                other,
                writes=False,
            ),
            _dependency_writes=_merge_dependency_accesses(
                self,
                other,
                writes=True,
            ),
            _dependency_completion=_max_dependency_completion(self, other),
            _global_barrier_condition=sp.Or(
                self._global_barrier_condition,
                other._global_barrier_condition,
            ),
            _measurement_taint_conditions=_merge_measurement_taint_conditions(
                self._measurement_taint_conditions,
                other._measurement_taint_conditions,
            ),
            _guarded_assumptions=(
                *(self._guarded_assumptions or ()),
                *(other._guarded_assumptions or ()),
            ),
            _guarded_derivations=(
                *(self._guarded_derivations or ()),
                *(other._guarded_derivations or ()),
            ),
            _guarded_qualities=(
                *(self._guarded_qualities or ()),
                *(other._guarded_qualities or ()),
            ),
            _guarded_approximations=(
                *(self._guarded_approximations or ()),
                *(other._guarded_approximations or ()),
            ),
            _symbol_aliases=_merge_symbol_aliases(self, other),
        )

    def choice(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose a conservative branch choice.

        Args:
            other (ResourceEstimate): Alternative branch estimate.

        Returns:
            ResourceEstimate: Element-wise maximum of both branches.
        """
        basis, control_decomposition, precision = _merge_estimate_provenance(
            self,
            other,
        )
        allocation_sites = _merge_allocation_sites(
            self._allocation_sites,
            other._allocation_sites,
        )
        return ResourceEstimate(
            width=_branch_width_with_static_allocations(
                _max_width(self.width, other.width),
                self.width,
                self._allocation_sites,
                other.width,
                other._allocation_sites,
                allocation_sites,
            ),
            gates=_max_gates(self.gates, other.gates),
            depth=_max_depth(self.depth, other.depth),
            calls=_max_calls(self.calls, other.calls),
            measurements=_max_measurements(self.measurements, other.measurements),
            resets=_max_resets(self.resets, other.resets),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("choice", self.trace, other.trace),
            derivation=_combine_derivation(self.derivation, other.derivation),
            quality=_combine_quality(
                EstimateQuality.CONSERVATIVE,
                _combine_quality(self.quality, other.quality),
            ),
            approximation=_combine_approximation(
                self.approximation,
                other.approximation,
            ),
            basis=basis,
            control_decomposition=control_decomposition,
            precision=precision,
            _allocation_sites=allocation_sites,
            _constraints=(*self._constraints, *other._constraints),
            _dependency_keys=_merge_dependency_keys(self, other),
            _dependency_reads=_merge_dependency_accesses(
                self,
                other,
                writes=False,
            ),
            _dependency_writes=_merge_dependency_accesses(
                self,
                other,
                writes=True,
            ),
            _dependency_completion=_max_dependency_completion(self, other),
            _global_barrier_condition=sp.Or(
                self._global_barrier_condition,
                other._global_barrier_condition,
            ),
            _measurement_taint_conditions=_merge_measurement_taint_conditions(
                self._measurement_taint_conditions,
                other._measurement_taint_conditions,
            ),
            _guarded_assumptions=(
                *(self._guarded_assumptions or ()),
                *(other._guarded_assumptions or ()),
            ),
            _guarded_derivations=(
                *(self._guarded_derivations or ()),
                *(other._guarded_derivations or ()),
            ),
            _guarded_qualities=(
                *(self._guarded_qualities or ()),
                *(other._guarded_qualities or ()),
                _GuardedQuality(sp.true, EstimateQuality.CONSERVATIVE),
            ),
            _guarded_approximations=(
                *(self._guarded_approximations or ()),
                *(other._guarded_approximations or ()),
            ),
            _symbol_aliases=_merge_symbol_aliases(self, other),
        )

    def conditional(
        self,
        other: ResourceEstimate,
        condition: sp.Basic,
    ) -> ResourceEstimate:
        """Select this estimate or another with a symbolic condition.

        Args:
            other (ResourceEstimate): Estimate for the false branch.
            condition (sp.Basic): SymPy Boolean selecting this estimate when
                true and ``other`` when false.

        Returns:
            ResourceEstimate: Field-wise exact piecewise branch estimate.
        """
        condition = _boolean_condition(condition)
        if condition is sp.true:
            return self
        if condition is sp.false:
            return other
        basis, control_decomposition, precision = _merge_estimate_provenance(
            self,
            other,
        )
        allocation_sites = _merge_allocation_sites(
            _activate_allocation_sites(self._allocation_sites, condition),
            _activate_allocation_sites(
                other._allocation_sites,
                sp.Not(condition),
            ),
        )
        conditional_width = _conditional_width(self.width, other.width, condition)
        anonymous_allocated = _piecewise(
            _anonymous_allocation_width(self.width, self._allocation_sites),
            _anonymous_allocation_width(other.width, other._allocation_sites),
            condition,
        )
        dependency_keys = _merge_dependency_keys(self, other)
        estimate = ResourceEstimate(
            width=_width_with_identity_aware_allocations(
                conditional_width,
                allocation_sites,
                anonymous_allocated=anonymous_allocated,
            ),
            gates=_conditional_gates(self.gates, other.gates, condition),
            depth=_conditional_depth(self.depth, other.depth, condition),
            calls=_conditional_calls(self.calls, other.calls, condition),
            measurements=_conditional_measurements(
                self.measurements,
                other.measurements,
                condition,
            ),
            resets=_conditional_resets(self.resets, other.resets, condition),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_conditional_trace(condition, self.trace, other.trace),
            derivation=_combine_derivation(self.derivation, other.derivation),
            quality=_combine_quality(self.quality, other.quality),
            approximation=_combine_approximation(
                self.approximation,
                other.approximation,
            ),
            basis=basis,
            control_decomposition=control_decomposition,
            precision=precision,
            _allocation_sites=allocation_sites,
            _constraints=(
                *(constraint.when(condition) for constraint in self._constraints),
                *(
                    constraint.when(sp.Not(condition))
                    for constraint in other._constraints
                ),
            ),
            _dependency_keys=dependency_keys,
            _dependency_reads=_merge_dependency_accesses(
                self,
                other,
                writes=False,
            ),
            _dependency_writes=_merge_dependency_accesses(
                self,
                other,
                writes=True,
            ),
            _dependency_completion=_conditional_dependency_completion(
                self,
                other,
                condition,
            ),
            _global_barrier_condition=sp.Or(
                sp.And(condition, self._global_barrier_condition),
                sp.And(sp.Not(condition), other._global_barrier_condition),
            ),
            _measurement_taint_conditions=(
                _conditional_measurement_taint_conditions(
                    self._measurement_taint_conditions,
                    other._measurement_taint_conditions,
                    condition,
                )
            ),
            _guarded_assumptions=(
                *(fact.when(condition) for fact in (self._guarded_assumptions or ())),
                *(
                    fact.when(sp.Not(condition))
                    for fact in (other._guarded_assumptions or ())
                ),
            ),
            _guarded_derivations=(
                *(fact.when(condition) for fact in (self._guarded_derivations or ())),
                *(
                    fact.when(sp.Not(condition))
                    for fact in (other._guarded_derivations or ())
                ),
            ),
            _guarded_qualities=(
                *(fact.when(condition) for fact in (self._guarded_qualities or ())),
                *(
                    fact.when(sp.Not(condition))
                    for fact in (other._guarded_qualities or ())
                ),
            ),
            _guarded_approximations=(
                *(
                    fact.when(condition)
                    for fact in (self._guarded_approximations or ())
                ),
                *(
                    fact.when(sp.Not(condition))
                    for fact in (other._guarded_approximations or ())
                ),
            ),
            _symbol_aliases=_merge_symbol_aliases(self, other),
        )
        masks_differ = self._dependency_keys != other._dependency_keys
        if masks_differ and (dependency_keys is None or len(dependency_keys) > 1):
            assumption = ResourceAssumption(
                "symbolic branch depth uses the union of branch-specific "
                "wire dependencies",
                source=str(condition),
            )
            estimate = estimate._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=_unresolved_condition_guard(condition),
            )
        return estimate

    def repeat(self, factor: ResourceExpr | int) -> ResourceEstimate:
        """Repeat this estimate with reusable width.

        Args:
            factor (ResourceExpr | int): Iteration or power factor.

        Returns:
            ResourceEstimate: Repeated estimate.

        Raises:
            ValueError: If a concrete factor is negative or non-integral.
        """
        f = _expr(factor)
        factor_constraint = _ResourceConstraint(
            expression=f,
            minimum=0,
            label="Resource repetition factor",
        )
        factor_constraint.validate()
        active_width = _conditional_width(
            self.width,
            WidthResources.zero(),
            sp.Gt(f, _ZERO),
        )
        active_sites = _activate_allocation_sites(
            self._allocation_sites,
            sp.Gt(f, _ZERO),
        )
        active_when = sp.Gt(f, _ZERO)
        estimate = ResourceEstimate(
            width=active_width,
            gates=_scale_gates(self.gates, f),
            depth=_scale_depth(self.depth, f),
            calls=_scale_calls(self.calls, f),
            measurements=_scale_measurements(self.measurements, f),
            resets=_scale_resets(self.resets, f),
            assumptions=(),
            trace=_wrap_trace(
                f"repeat({f})",
                self.trace.when(active_when) if self.trace is not None else None,
            ),
            derivation=EstimateDerivation.STRUCTURAL,
            quality=EstimateQuality.EXACT,
            approximation=ApproximationStatus.EXACT,
            basis=self.basis,
            control_decomposition=self.control_decomposition,
            precision=self.precision,
            _allocation_sites=active_sites,
            _constraints=tuple(
                constraint.when(sp.Gt(f, _ZERO)) for constraint in self._constraints
            )
            + ((factor_constraint,) if not f.is_number else ()),
            _dependency_keys=(frozenset() if f == _ZERO else self._dependency_keys),
            _dependency_reads=(frozenset() if f == _ZERO else self._dependency_reads),
            _dependency_writes=(frozenset() if f == _ZERO else self._dependency_writes),
            _dependency_completion=(
                {
                    key: cast(
                        ResourceExpr,
                        _ConditionIndicator(
                            sp.And(
                                active_when,
                                _resource_activity_condition(completion),
                            )
                        )
                        * ((f - _ONE) * self.depth.depth + completion),
                    )
                    for key, completion in self._dependency_completion.items()
                }
                if self._dependency_completion is not None
                else None
            ),
            _dependency_completion_uniform=self._dependency_completion_uniform,
            _global_barrier_condition=sp.And(
                self._global_barrier_condition,
                active_when,
            ),
            _measurement_taint_conditions=_guard_measurement_taint_conditions(
                self._measurement_taint_conditions,
                active_when,
            ),
            _guarded_assumptions=tuple(
                fact.when(active_when) for fact in (self._guarded_assumptions or ())
            ),
            _guarded_derivations=tuple(
                fact.when(active_when) for fact in (self._guarded_derivations or ())
            ),
            _guarded_qualities=tuple(
                fact.when(active_when) for fact in (self._guarded_qualities or ())
            ),
            _guarded_approximations=tuple(
                fact.when(active_when) for fact in (self._guarded_approximations or ())
            ),
            _symbol_aliases=self._symbol_aliases,
        )
        if (
            f.is_positive is not True
            and self._dependency_keys is not None
            and len(self._dependency_keys) > 1
            and f != _ZERO
        ):
            assumption = ResourceAssumption(
                "a possibly zero repetition uses a conservative shared "
                "wire-dependency summary",
                source=str(f),
            )
            estimate = estimate._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.And(
                    active_when,
                    _resource_activity_condition(self.depth.depth),
                ),
            )
        return estimate

    def controlled(self, num_controls: ResourceExpr | int) -> ResourceEstimate:
        """Estimate controls on an aggregate cost from its known arity profile.

        Under ``ABSTRACT``, every source primitive remains one logical
        operation, declared arity buckets shift by the control count, and
        shared controls serialize aggregate gate depth. Under
        ``CLEAN_ANCILLA_TOFFOLI``, declared one- or two-qubit gates are
        projected through a fixed aggregate-level batching model. Two or more
        controls around at least two modeled operations share one body-wide
        control ladder; smaller cases retain per-primitive lowering. Gate
        names and scheduling are unavailable, so gate-family fields use
        independent upper bounds over the supported logical primitive
        families. A complete arity profile therefore produces a conservative
        estimate, while a profile with unclassified or undecomposed gates
        remains directionally unknown. Explicit costs are complete contracts:
        their authors must include any phase-relevant work that later controls
        need as a declared logical primitive, and the estimator does not add
        hidden global-phase overhead. Angle-specific phase classification
        requires a body-backed global-phase operation; a declared one-qubit
        phase entry is an upper-bound representative for a target-free phase.
        Unsupported gate bases and aggregate measurement or reset costs fail
        closed. Body-backed qkernels are controlled by the estimator
        interpreter instead.

        Args:
            num_controls (ResourceExpr | int): Number of active controls.

        Returns:
            ResourceEstimate: Estimate with a recorded controlled assumption.

        Raises:
            ValueError: If a concrete control count or projected gate count
                is negative or non-integral, if the selected gate basis has no
                aggregate controlled projection, or if the estimate contains
                measurement or reset resources.
        """
        controls = _expr(num_controls)
        control_constraint = _ResourceConstraint(
            expression=controls,
            minimum=0,
            label="Aggregate controlled resource count",
            unit="control qubit",
        )
        control_constraint.validate()
        if controls == _ZERO:
            return self
        _require_unitary_resource_estimate(
            self,
            transform="coherently control",
        )
        profile_constraints = (
            *_aggregate_resource_profile_constraints(self, controls),
            *_aggregate_arity_projection_constraints(
                self,
                controls,
                model_label="Controlled",
            ),
        )
        if _safe_simplify(
            _estimate_activity(self)
        ) == _ZERO and not _estimate_has_basis_sensitive_resources(self):
            if controls.is_number:
                return self
            return dataclasses.replace(
                self,
                _constraints=(
                    *self._constraints,
                    *profile_constraints,
                    control_constraint,
                ),
            )
        if not _estimate_has_basis_sensitive_resources(self):
            projected = None
            reason = _aggregate_arity_profile_reason(self) or (
                "the aggregate has no gate-model-sensitive resource profile"
            )
        elif self.basis is not GateBasis.LOGICAL:
            raise ValueError(
                "Controlled aggregate resource projection is not defined for "
                f"gate basis {self.basis.value!r} with control decomposition "
                f"{self.control_decomposition.value!r}. Use the logical basis "
                "or a body-backed callable with a defined controlled lowering."
            )
        elif (
            self.basis is GateBasis.LOGICAL
            and self.control_decomposition is ControlDecomposition.ABSTRACT
        ):
            projected, reason = _project_abstract_aggregate_controlled_cost(
                self,
                controls,
            )
        else:
            projected, reason = _project_clean_ancilla_aggregate_controlled_cost(
                self,
                controls,
            )
        if projected is not None:
            projected = dataclasses.replace(
                projected,
                _constraints=(*projected._constraints, *profile_constraints),
            )
            if not controls.is_number:
                projected = self.conditional(
                    projected,
                    sp.Eq(controls, _ZERO),
                )
                projected = dataclasses.replace(
                    projected,
                    _constraints=(*projected._constraints, control_constraint),
                    _output_sizes=self._output_sizes,
                    _input_sizes=self._input_sizes,
                    _has_output_summary=self._has_output_summary,
                    _symbol_aliases=self._symbol_aliases,
                )
            return projected
        assumption = _unprojected_aggregate_control_assumption(
            reason,
            controls,
        )
        controlled_estimate = ResourceEstimate(
            width=self.width,
            gates=self.gates,
            depth=self.depth,
            calls=self.calls,
            measurements=self.measurements,
            resets=self.resets,
            assumptions=self.assumptions,
            trace=_wrap_trace(f"controlled({controls})", self.trace),
            derivation=self.derivation,
            quality=self.quality,
            approximation=self.approximation,
            basis=self.basis,
            control_decomposition=self.control_decomposition,
            precision=self.precision,
            _allocation_sites=self._allocation_sites,
            _constraints=(
                *self._constraints,
                *profile_constraints,
                *((control_constraint,) if not controls.is_number else ()),
            ),
            _output_sizes=self._output_sizes,
            _input_sizes=self._input_sizes,
            _has_output_summary=self._has_output_summary,
            _dependency_keys=self._dependency_keys,
            _dependency_reads=self._dependency_reads,
            _dependency_writes=self._dependency_writes,
            _dependency_completion=self._dependency_completion,
            _dependency_completion_uniform=self._dependency_completion_uniform,
            _global_barrier_condition=self._global_barrier_condition,
            _measurement_taint_conditions=self._measurement_taint_conditions,
            _guarded_assumptions=self._guarded_assumptions,
            _guarded_derivations=self._guarded_derivations,
            _guarded_qualities=self._guarded_qualities,
            _guarded_approximations=self._guarded_approximations,
            _symbol_aliases=self._symbol_aliases,
        )
        return controlled_estimate._with_metadata(
            assumptions=(assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=_simplify_aggregate_metadata_guard(
                sp.And(
                    sp.Gt(controls, _ZERO),
                    sp.Or(
                        sp.Gt(self.gates.total, _ZERO),
                        _aggregate_zero_gate_residual_condition(self),
                    ),
                )
            ),
        )

    def inverse(self) -> ResourceEstimate:
        """Apply an inverse transform.

        Returns:
            ResourceEstimate: Estimate with identical logical resources.

        Raises:
            ValueError: If the estimate contains measurement or reset
                resources and is therefore not unitary.
        """
        _require_unitary_resource_estimate(
            self,
            transform="invert",
        )
        return ResourceEstimate(
            width=self.width,
            gates=self.gates,
            depth=self.depth,
            calls=self.calls,
            measurements=self.measurements,
            resets=self.resets,
            assumptions=self.assumptions,
            trace=_wrap_trace("inverse", self.trace),
            parameters=self.parameters,
            derivation=self.derivation,
            quality=self.quality,
            approximation=self.approximation,
            basis=self.basis,
            control_decomposition=self.control_decomposition,
            precision=self.precision,
            _allocation_sites=self._allocation_sites,
            _constraints=self._constraints,
            _output_sizes=self._output_sizes,
            _input_sizes=self._input_sizes,
            _has_output_summary=self._has_output_summary,
            _dependency_keys=self._dependency_keys,
            _dependency_reads=self._dependency_reads,
            _dependency_writes=self._dependency_writes,
            # Reversing a multi-wire aggregate preserves its touched wires but
            # can change which wire finishes first. Only a gate-by-gate inverse
            # can reconstruct exact caller-visible completion layers.
            _dependency_completion=None,
            _dependency_completion_uniform=None,
            _global_barrier_condition=self._global_barrier_condition,
            _measurement_taint_conditions=self._measurement_taint_conditions,
            _guarded_assumptions=self._guarded_assumptions,
            _guarded_derivations=self._guarded_derivations,
            _guarded_qualities=self._guarded_qualities,
            _guarded_approximations=self._guarded_approximations,
            _symbol_aliases=self._symbol_aliases,
        )

    def sum_over(
        self,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        stop: ResourceExpr,
        step: ResourceExpr = _ONE,
    ) -> ResourceEstimate:
        """Sum loop-dependent resources over Python ``range`` semantics.

        Args:
            loop_symbol (sp.Symbol): Symbol used for the loop variable.
            start (ResourceExpr): Inclusive start bound.
            stop (ResourceExpr): Exclusive stop bound.
            step (ResourceExpr): Loop step. Defaults to one.

        Returns:
            ResourceEstimate: Estimate with additive metrics summed over the
            loop and width kept reusable.
        """
        return self._sum_over(
            loop_symbol,
            start,
            stop,
            step,
            dependency_start=start,
            dependency_stop=stop,
            dependency_step=step,
        )

    def _sum_over(
        self,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        stop: ResourceExpr,
        step: ResourceExpr,
        *,
        dependency_start: ResourceExpr,
        dependency_stop: ResourceExpr,
        dependency_step: ResourceExpr,
    ) -> ResourceEstimate:
        """Sum resources while using specialized bounds for wire projection.

        Args:
            loop_symbol (sp.Symbol): Symbol used for the loop variable.
            start (ResourceExpr): Inclusive resource-expression start bound.
            stop (ResourceExpr): Exclusive resource-expression stop bound.
            step (ResourceExpr): Resource-expression loop step.
            dependency_start (ResourceExpr): Start bound specialized only for
                caller-visible wire projection.
            dependency_stop (ResourceExpr): Stop bound specialized only for
                caller-visible wire projection.
            dependency_step (ResourceExpr): Step specialized only for
                caller-visible wire projection.

        Returns:
            ResourceEstimate: Estimate with additive metrics summed over the
                loop and width kept reusable.
        """
        step_constraint = _ResourceConstraint(
            expression=sp.Abs(step),
            minimum=1,
            label="Loop step magnitude",
        )
        step_constraint.validate()
        iterations = symbolic_iterations(start, stop, step)
        projected_iterations = symbolic_iterations(
            dependency_start,
            dependency_stop,
            dependency_step,
        )
        if loop_symbol not in _free_symbols(self):
            return _project_dependency_metadata_over_symbol(
                _with_constraints(
                    self.repeat(iterations),
                    step_constraint,
                ),
                loop_symbol,
                start=dependency_start,
                step=dependency_step,
                iterations=projected_iterations,
            )
        width, allocation_sites, width_conservative_when = _maximum_width_over_range(
            self.width,
            self._allocation_sites,
            loop_symbol,
            start,
            step,
            iterations,
        )
        estimate = ResourceEstimate(
            width=width,
            gates=_sum_gates(self.gates, loop_symbol, start, step, iterations),
            depth=_sum_depth(self.depth, loop_symbol, start, step, iterations),
            calls=_sum_calls(self.calls, loop_symbol, start, step, iterations),
            measurements=_sum_measurements(
                self.measurements,
                loop_symbol,
                start,
                step,
                iterations,
            ),
            resets=_sum_resets(
                self.resets,
                loop_symbol,
                start,
                step,
                iterations,
            ),
            trace=_wrap_trace(
                f"sum({loop_symbol}={start}..{stop})",
                self.trace.when(sp.Gt(iterations, _ZERO))
                if self.trace is not None
                else None,
            ),
            basis=self.basis,
            control_decomposition=self.control_decomposition,
            precision=self.precision,
            _allocation_sites=allocation_sites,
            _constraints=(
                step_constraint,
                *(
                    constraint.bound_over(
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    ).when(sp.Gt(iterations, _ZERO))
                    for constraint in self._constraints
                ),
            ),
            _dependency_keys=self._dependency_keys,
            _dependency_reads=self._dependency_reads,
            _dependency_writes=self._dependency_writes,
            _dependency_completion=(
                {
                    key: cast(
                        ResourceExpr,
                        _ConditionIndicator(
                            _activation_over_range(
                                _resource_activity_condition(completion),
                                loop_symbol,
                                start,
                                step,
                                iterations,
                            )
                        )
                        * _sum_expr(
                            self.depth.depth,
                            loop_symbol,
                            start,
                            step,
                            iterations,
                        ),
                    )
                    for key, completion in self._dependency_completion.items()
                }
                if self._dependency_completion is not None
                else None
            ),
            _global_barrier_condition=_boolean_condition(
                _activation_over_range(
                    self._global_barrier_condition,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                )
            ),
            _measurement_taint_conditions={
                uuid: _boolean_condition(
                    _activation_over_range(
                        condition,
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    )
                )
                for uuid, condition in self._measurement_taint_conditions.items()
            },
            _guarded_assumptions=tuple(
                dataclasses.replace(
                    fact,
                    active_when=_activation_over_range(
                        fact.active_when,
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    ),
                )
                for fact in (self._guarded_assumptions or ())
            ),
            _guarded_derivations=tuple(
                dataclasses.replace(
                    fact,
                    active_when=_activation_over_range(
                        fact.active_when,
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    ),
                )
                for fact in (self._guarded_derivations or ())
            ),
            _guarded_qualities=tuple(
                dataclasses.replace(
                    fact,
                    active_when=_activation_over_range(
                        fact.active_when,
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    ),
                )
                for fact in (self._guarded_qualities or ())
            ),
            _guarded_approximations=tuple(
                dataclasses.replace(
                    fact,
                    active_when=_activation_over_range(
                        fact.active_when,
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    ),
                )
                for fact in (self._guarded_approximations or ())
            ),
            _symbol_aliases=self._symbol_aliases,
        )
        if width_conservative_when is not sp.false:
            estimate = estimate._with_metadata(
                assumptions=(
                    ResourceAssumption(
                        "symbolic loop width uses a conservative excess-sum "
                        "bound because its maximum could not be proven",
                        source=str(loop_symbol),
                    ),
                ),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=width_conservative_when,
            )
        return _project_dependency_metadata_over_symbol(
            estimate,
            loop_symbol,
            start=dependency_start,
            step=dependency_step,
            iterations=projected_iterations,
        )

    def substitute(self, **values: object) -> ResourceEstimate:
        """Substitute concrete values for symbolic parameters.

        Args:
            **values (object): Mapping from parameter name to a concrete
                numeric scalar.

        Returns:
            ResourceEstimate: Estimate with substituted expressions.

        Raises:
            ValueError: If a name is not a parameter, or a supplied value
                violates an integer or nonnegative parameter domain.
            TypeError: If a supplied value is not a concrete numeric scalar.
        """
        subs: dict[sp.Symbol, sp.Expr] = {}
        for name, value in values.items():
            parameter = self.parameters.get(name)
            if parameter is None:
                available = ", ".join(self.parameters) or "(none)"
                raise ValueError(
                    f"Unknown resource parameter '{name}'; available: {available}."
                )
            replacement = _normalize_resource_scalar(
                value,
                label=f"Resource parameter '{name}'",
                allow_symbolic=False,
                allow_bool=False,
            )
            if (
                replacement.is_number
                and parameter.is_integer is True
                and not _is_concrete_integer(replacement)
            ):
                raise ValueError(
                    f"Cannot substitute non-integer value {value!r} for "
                    f"integer resource parameter '{name}'."
                )
            if parameter.is_integer is True:
                replacement = _canonicalize_concrete_integer(replacement)
            if replacement.is_negative is True and parameter.is_nonnegative is True:
                raise ValueError(
                    f"Cannot substitute negative value {value!r} for "
                    f"nonnegative resource parameter '{name}'."
                )
            subs[parameter] = replacement
        substitute_expression = partial(
            _substitute_resource_expr,
            substitutions=subs,
        )
        mapped = self._map_expr(
            substitute_expression,
            constraint_fn=lambda expr: _safe_constraint_substitute(expr, subs),
            guard_fn=lambda expr: _safe_constraint_substitute(expr, subs),
            dependency_fn=substitute_expression,
        )
        # Substitution often collapses a previously large branch expression to
        # a small Piecewise/Min/Max form. Re-run the bounded public simplifier
        # so post-hoc specialization has the same canonical shape as direct
        # input specialization, while still skipping expressions above the
        # global-simplification node budget.
        return mapped.simplify()

    def simplify(self) -> ResourceEstimate:
        """Simplify all symbolic expressions.

        Returns:
            ResourceEstimate: Simplified estimate.
        """
        return self._map_expr(
            _simplify_public_resource_expression,
            # Guards are already normalized while they are composed.
            # Re-running SymPy's global Boolean simplifier here is
            # disproportionately expensive for loop-derived predicates and
            # can expose bound variables. Concrete substitution still
            # rewrites and resolves every guard through the default path.
            guard_fn=lambda expression: expression,
            # Dependency completion is private scheduling state. All call
            # boundary decisions have already consumed it by the time the
            # root estimate is simplified, and recursively simplifying these
            # substantially larger expressions does not change public
            # metrics. Substitution still rewrites this state through the
            # default path so a returned estimate remains composable.
            dependency_fn=lambda expression: expression,
        )

    def explain(self, metric: str | None = None) -> str:
        """Render the resource-estimation trace.

        Args:
            metric (str | None): Optional metric name to mention in the
                heading. Filtering is reserved for a later pass. Defaults to
                ``None``.

        Returns:
            str: Human-readable explanation tree.
        """
        heading = "Resource estimate"
        if metric is not None:
            heading = f"{heading} for {metric}"
        if self.trace is None:
            return heading
        registry = _serialization_registry(self)
        return f"{heading}\n{self.trace.render(2, registry)}"

    def to_dict(self) -> dict[str, Any]:
        """Convert this estimate to a JSON-friendly report snapshot.

        Symbolic fields are display strings, not a round-trip expression
        format. They can contain Qamomile-specific symbolic nodes and must not
        be evaluated with :func:`sympy.sympify`. To produce a concrete report,
        specialize the original estimate with :meth:`substitute` before
        calling this method.

        Returns:
            dict[str, Any]: Report fields with stringified resource
                expressions.
        """
        registry = _serialization_registry(self)
        serialize = registry.stringify
        return {
            "width": {
                "input_qubits": serialize(self.width.input_qubits),
                "allocated_qubits": serialize(self.width.allocated_qubits),
                "clean_ancilla_qubits": serialize(self.width.clean_ancilla_qubits),
                "dirty_ancilla_qubits": serialize(self.width.dirty_ancilla_qubits),
                "peak_qubits": serialize(self.width.peak_qubits),
                "circuit_qubits": serialize(self.width.circuit_qubits),
            },
            "gates": {
                "total": serialize(self.gates.total),
                "single_qubit": serialize(self.gates.single_qubit),
                "two_qubit": serialize(self.gates.two_qubit),
                "multi_qubit": serialize(self.gates.multi_qubit),
                "clifford": serialize(self.gates.clifford),
                "rotation": serialize(self.gates.rotation),
                "t": serialize(self.gates.t),
                "toffoli": serialize(self.gates.toffoli),
                "non_clifford": serialize(self.gates.non_clifford),
            },
            "measurements": {
                "total": serialize(self.measurements.total),
            },
            "resets": {
                "total": serialize(self.resets.total),
            },
            "depth": {
                "depth": serialize(self.depth.depth),
                "clifford_depth": serialize(self.depth.clifford_depth),
                "rotation_depth": serialize(self.depth.rotation_depth),
                "t_depth": serialize(self.depth.t_depth),
                "toffoli_depth": serialize(self.depth.toffoli_depth),
                "non_clifford_depth": serialize(self.depth.non_clifford_depth),
                "measurement_depth": serialize(self.depth.measurement_depth),
                "gate_depth": serialize(self.depth.gate_depth),
                "reset_depth": serialize(self.depth.reset_depth),
            },
            "calls": {
                "calls_by_name": {
                    name: serialize(value)
                    for name, value in self.calls.calls_by_name.items()
                },
                "queries_by_name": {
                    name: serialize(value)
                    for name, value in self.calls.queries_by_name.items()
                },
            },
            "assumptions": [
                {"message": assumption.message, "source": assumption.source}
                for assumption in self.assumptions
            ],
            "parameters": {
                name: registry.name(symbol) for name, symbol in self.parameters.items()
            },
            "derivation": self.derivation.value,
            "quality": self.quality.value,
            "approximation": self.approximation.value,
            "basis": self.basis.value,
            "control_decomposition": self.control_decomposition.value,
            "precision": self.precision,
            "requirements": [
                {
                    "expression": serialize(constraint.expression),
                    "active_when": serialize(constraint.active_when),
                    "minimum": constraint.minimum,
                    "minimum_inclusive": constraint.minimum_inclusive,
                    "finite": constraint.finite,
                    "expected": (
                        serialize(constraint.expected)
                        if constraint.expected is not None
                        else None
                    ),
                    "integer": constraint.integer,
                    "label": constraint.label,
                    "unit": constraint.unit,
                    "ranges": [
                        {
                            "symbol": registry.name(loop_range.symbol),
                            "start": serialize(loop_range.start),
                            "step": serialize(loop_range.step),
                            "iterations": serialize(loop_range.iterations),
                        }
                        for loop_range in constraint.ranges
                    ],
                }
                for constraint in self._constraints
            ],
        }

    def _map_expr(
        self,
        fn: Any,
        *,
        constraint_fn: Any | None = None,
        guard_fn: Any | None = None,
        dependency_fn: Any | None = None,
    ) -> ResourceEstimate:
        """Apply a function to every symbolic expression.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.
            constraint_fn (Any | None): Optional non-clamping rewrite for
                structural constraints. Defaults to ``fn``.
            guard_fn (Any | None): Optional rewrite for guarded assumption,
                derivation, quality, and approximation predicates. Defaults
                to ``constraint_fn``.
            dependency_fn (Any | None): Optional rewrite for private wire-key
                indices and per-wire completion depths. Defaults to
                ``constraint_fn``.

        Returns:
            ResourceEstimate: Rewritten estimate.
        """
        rewrite_constraint = fn if constraint_fn is None else constraint_fn
        rewrite_guard = rewrite_constraint if guard_fn is None else guard_fn
        rewrite_dependency = (
            rewrite_constraint if dependency_fn is None else dependency_fn
        )
        mapped_constraints = tuple(
            mapped
            for constraint in self._constraints
            if (mapped := constraint.mapped(rewrite_constraint)).active_when
            is not sp.false
        )
        mapped_assumptions = tuple(
            mapped
            for fact in (self._guarded_assumptions or ())
            if (mapped := fact.mapped(rewrite_guard)) is not None
        )
        mapped_derivations = tuple(
            mapped
            for fact in (self._guarded_derivations or ())
            if (mapped := fact.mapped(rewrite_guard)) is not None
        )
        mapped_qualities = tuple(
            mapped
            for fact in (self._guarded_qualities or ())
            if (mapped := fact.mapped(rewrite_guard)) is not None
        )
        mapped_approximations = tuple(
            mapped
            for fact in (self._guarded_approximations or ())
            if (mapped := fact.mapped(rewrite_guard)) is not None
        )
        mapped_calls = CallResources(
            calls_by_name={
                name: mapped
                for name, value in self.calls.calls_by_name.items()
                if (mapped := fn(value)) != _ZERO
            },
            queries_by_name={
                name: mapped
                for name, value in self.calls.queries_by_name.items()
                if (mapped := fn(value)) != _ZERO
            },
        )
        mapped = ResourceEstimate(
            width=WidthResources(
                input_qubits=fn(self.width.input_qubits),
                allocated_qubits=fn(self.width.allocated_qubits),
                clean_ancilla_qubits=fn(self.width.clean_ancilla_qubits),
                dirty_ancilla_qubits=fn(self.width.dirty_ancilla_qubits),
                peak_qubits=fn(self.width.peak_qubits),
            ),
            gates=GateResources(
                total=fn(self.gates.total),
                single_qubit=fn(self.gates.single_qubit),
                two_qubit=fn(self.gates.two_qubit),
                multi_qubit=fn(self.gates.multi_qubit),
                clifford=fn(self.gates.clifford),
                rotation=fn(self.gates.rotation),
                t=fn(self.gates.t),
                toffoli=fn(self.gates.toffoli),
                non_clifford=fn(self.gates.non_clifford),
            ),
            measurements=MeasurementResources(
                total=fn(self.measurements.total),
            ),
            resets=ResetResources(
                total=fn(self.resets.total),
            ),
            depth=DepthResources(
                depth=fn(self.depth.depth),
                clifford_depth=fn(self.depth.clifford_depth),
                rotation_depth=fn(self.depth.rotation_depth),
                t_depth=fn(self.depth.t_depth),
                toffoli_depth=fn(self.depth.toffoli_depth),
                non_clifford_depth=fn(self.depth.non_clifford_depth),
                measurement_depth=fn(self.depth.measurement_depth),
                gate_depth=fn(self.depth.gate_depth),
                reset_depth=fn(self.depth.reset_depth),
            ),
            calls=mapped_calls,
            trace=(
                self.trace.mapped(rewrite_constraint)
                if self.trace is not None
                else None
            ),
            basis=self.basis,
            control_decomposition=self.control_decomposition,
            precision=self.precision,
            _allocation_sites={
                site: fn(size) for site, size in self._allocation_sites.items()
            },
            _constraints=mapped_constraints,
            _output_sizes={
                owner: fn(size) for owner, size in self._output_sizes.items()
            },
            _input_sizes={owner: fn(size) for owner, size in self._input_sizes.items()},
            _has_output_summary=self._has_output_summary,
            _dependency_keys=_map_dependency_keys(
                self._dependency_keys,
                rewrite_dependency,
            ),
            _dependency_reads=_map_dependency_keys(
                self._dependency_reads,
                rewrite_dependency,
            ),
            _dependency_writes=_map_dependency_keys(
                self._dependency_writes,
                rewrite_dependency,
            ),
            _dependency_completion=_map_dependency_completion(
                self._dependency_completion,
                rewrite_dependency,
            ),
            _dependency_completion_uniform=self._dependency_completion_uniform,
            _global_barrier_condition=_boolean_condition(
                rewrite_guard(self._global_barrier_condition)
            ),
            _measurement_taint_conditions={
                uuid: _boolean_condition(rewrite_guard(condition))
                for uuid, condition in self._measurement_taint_conditions.items()
            },
            _guarded_assumptions=mapped_assumptions,
            _guarded_derivations=mapped_derivations,
            _guarded_qualities=mapped_qualities,
            _guarded_approximations=mapped_approximations,
            _symbol_aliases=self._symbol_aliases,
        )
        return mapped


class _SequentialEstimateComposer:
    """Compose a stream of estimates with logarithmic intermediate storage."""

    def __init__(self) -> None:
        """Initialize an empty binary-counter reduction."""
        self._levels: list[ResourceEstimate | None] = []

    def append(self, estimate: ResourceEstimate) -> None:
        """Append one estimate after all previously supplied estimates.

        Args:
            estimate (ResourceEstimate): Next estimate in execution order.

        Raises:
            ValueError: If composition encounters incompatible basis or
                precision provenance.
        """
        carry = estimate
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            earlier = self._levels[level]
            assert earlier is not None
            carry = earlier.seq(carry)
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(carry)
        else:
            self._levels[level] = carry

    def finish(self) -> ResourceEstimate:
        """Return the order-preserving composition accumulated so far.

        Returns:
            ResourceEstimate: Sequential composition, or exact zero when no
            estimate was appended.

        Raises:
            ValueError: If composition encounters incompatible basis or
                precision provenance.
        """
        result: ResourceEstimate | None = None
        for estimate in reversed(self._levels):
            if estimate is None:
                continue
            result = estimate if result is None else result.seq(estimate)
        return result if result is not None else ResourceEstimate.zero()


def _estimate_has_basis_sensitive_resources(estimate: ResourceEstimate) -> bool:
    """Return whether an estimate contains basis-dependent metrics.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        bool: Whether gates, decomposition ancillas, or gate depth are not
        structurally zero. Symbolic and unevaluated expressions are treated as
        basis-sensitive conservatively.
    """
    expressions = [
        *(
            getattr(estimate.gates, field.name)
            for field in dataclasses.fields(GateResources)
        ),
        estimate.depth.gate_depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
    ]
    has_new_nonunitary_profile = any(
        expression != _ZERO
        for expression in (
            estimate.measurements.total,
            estimate.resets.total,
            estimate.depth.reset_depth,
        )
    )
    if not has_new_nonunitary_profile:
        # Before ``gate_depth`` and reset resources existed, an aggregate cost
        # could distinguish measurement depth only. Preserve that contract:
        # any remaining legacy depth is gate-sensitive. New measurement/reset
        # profiles use their explicit counts/categories and ``gate_depth``.
        expressions.append(estimate.depth.depth - estimate.depth.measurement_depth)
    # This predicate runs for every resource-algebra composition. Calling
    # ``simplify`` here makes a linear fold repeatedly traverse the complete
    # accumulated expression and can turn ordinary algorithm estimates into
    # quadratic (or worse) work. Resource constructors preserve exact zeros,
    # so structural equality is sufficient for the neutral fast path. An
    # expression whose zero-ness is merely unproven remains basis-sensitive,
    # which is the safe provenance decision.
    return any(expression != _ZERO for expression in expressions)


def _require_unitary_resource_estimate(
    estimate: ResourceEstimate,
    *,
    transform: str,
) -> None:
    """Reject an aggregate cost that declares non-unitary resources.

    Aggregate measurement and reset costs cannot be coherently transformed
    without an implementation-specific model. Counts and depth categories are
    checked independently so legacy costs that expose only measurement depth
    also fail closed.

    Args:
        estimate (ResourceEstimate): Aggregate cost to validate.
        transform (str): Infinitive phrase naming the requested transform for
            the diagnostic, such as ``"invert"``.

    Raises:
        ValueError: If measurement or reset resources may be nonzero.
    """
    nonunitary = tuple(
        label
        for label, expression in (
            ("measurements.total", estimate.measurements.total),
            ("resets.total", estimate.resets.total),
            ("depth.measurement_depth", estimate.depth.measurement_depth),
            ("depth.reset_depth", estimate.depth.reset_depth),
        )
        if _safe_simplify(expression) != _ZERO
    )
    if nonunitary:
        fields = ", ".join(nonunitary)
        raise ValueError(
            f"Cannot {transform} a resource estimate with non-unitary "
            f"resources ({fields}). Move measurement and reset outside the "
            "coherent transform, or describe a unitary base implementation."
        )


def _precisions_match(left: float | None, right: float | None) -> bool:
    """Compare optional synthesis precisions without direct float equality.

    Args:
        left (float | None): First precision value.
        right (float | None): Second precision value.

    Returns:
        bool: Whether both values are absent or exactly the same finite float.
    """
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)


def _merge_estimate_provenance(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> tuple[GateBasis, ControlDecomposition, float | None]:
    """Merge compatible gate-model provenance for resource algebra.

    Gate-model-neutral width and call estimates may compose with any basis and
    control decomposition. Two estimates carrying gate-model-sensitive metrics
    must agree on the basis, control decomposition, and, for Clifford+T,
    synthesis precision.

    Args:
        left (ResourceEstimate): Left operand.
        right (ResourceEstimate): Right operand.

    Returns:
        tuple[GateBasis, ControlDecomposition, float | None]: Basis, control
            decomposition, and precision for the result.

    Raises:
        ValueError: If gate-model-sensitive estimates use incompatible
            provenance.
    """
    left_sensitive = _estimate_has_basis_sensitive_resources(left)
    right_sensitive = _estimate_has_basis_sensitive_resources(right)
    if left_sensitive and right_sensitive:
        if left.basis is not right.basis:
            raise ValueError(
                "Cannot compose resource estimates from different gate "
                f"bases: {left.basis.value!r} and {right.basis.value!r}."
            )
        if left.control_decomposition is not right.control_decomposition:
            raise ValueError(
                "Cannot compose resource estimates from different control "
                "decompositions: "
                f"{left.control_decomposition.value!r} and "
                f"{right.control_decomposition.value!r}."
            )
        if left.basis is GateBasis.CLIFFORD_T and not _precisions_match(
            left.precision,
            right.precision,
        ):
            raise ValueError(
                "Cannot compose Clifford+T estimates with different "
                f"precisions: {left.precision!r} and {right.precision!r}."
            )
        return left.basis, left.control_decomposition, left.precision
    if left_sensitive:
        return left.basis, left.control_decomposition, left.precision
    if right_sensitive:
        return right.basis, right.control_decomposition, right.precision
    return left.basis, left.control_decomposition, left.precision


def _merge_symbol_aliases(
    *estimates: ResourceEstimate,
) -> dict[sp.Symbol, str]:
    """Merge stable public symbol aliases in operand order.

    The first estimate claiming a public alias retains it. When another
    symbol claims the same alias, omitting that later preference lets
    ``SymbolRegistry`` assign a collision-free suffix while protecting the
    earlier symbol's published name.

    Args:
        *estimates (ResourceEstimate): Estimates whose preferred aliases are
            merged from left to right.

    Returns:
        dict[sp.Symbol, str]: Left-biased, collision-free preferred aliases.
    """
    merged: dict[sp.Symbol, str] = {}
    claimed_aliases: set[str] = set()
    for estimate in estimates:
        for symbol, alias in estimate._symbol_aliases.items():
            if symbol in merged or alias in claimed_aliases:
                continue
            merged[symbol] = alias
            claimed_aliases.add(alias)
    return merged


def _merge_measurement_taint_conditions(
    *mappings: Mapping[str, Boolean],
) -> dict[str, Boolean]:
    """Merge guarded measurement provenance by SSA value identity.

    Args:
        *mappings (Mapping[str, Boolean]): UUID-keyed provenance conditions.

    Returns:
        dict[str, Boolean]: Conditions combined with Boolean OR, excluding
            values whose merged condition is definitely false.
    """
    merged: dict[str, Boolean] = {}
    for mapping in mappings:
        for uuid, condition in mapping.items():
            combined = _boolean_condition(sp.Or(merged.get(uuid, sp.false), condition))
            if combined is sp.false:
                merged.pop(uuid, None)
            else:
                merged[uuid] = combined
    return merged


def _guard_measurement_taint_conditions(
    mapping: Mapping[str, Boolean],
    condition: sp.Basic,
) -> dict[str, Boolean]:
    """Conjoin one activation condition with every taint entry.

    Args:
        mapping (Mapping[str, Boolean]): UUID-keyed provenance conditions.
        condition (sp.Basic): Additional activation predicate.

    Returns:
        dict[str, Boolean]: Guarded non-false provenance entries.
    """
    active = _boolean_condition(condition)
    return {
        uuid: guarded
        for uuid, value in mapping.items()
        if (guarded := _boolean_condition(sp.And(active, value))) is not sp.false
    }


def _conditional_measurement_taint_conditions(
    when_true: Mapping[str, Boolean],
    when_false: Mapping[str, Boolean],
    condition: sp.Basic,
) -> dict[str, Boolean]:
    """Select guarded provenance from two compile-time branches.

    Args:
        when_true (Mapping[str, Boolean]): True-branch provenance.
        when_false (Mapping[str, Boolean]): False-branch provenance.
        condition (sp.Basic): Branch selection predicate.

    Returns:
        dict[str, Boolean]: Branch-guarded union of both mappings.
    """
    predicate = _boolean_condition(condition)
    return _merge_measurement_taint_conditions(
        _guard_measurement_taint_conditions(when_true, predicate),
        _guard_measurement_taint_conditions(when_false, sp.Not(predicate)),
    )


def _estimate_uses_unresolved_functions(
    estimate: ResourceEstimate,
    functions: Iterable[Any],
) -> bool:
    """Return whether resource algebra retains an unresolved carry function.

    Comparing one concrete function application is insufficient after loop
    summation because SymPy can rewrite ``carry(k)`` to ``carry(0)`` or another
    application of the same undefined function. Matching the function identity
    catches every such rewrite without confusing independent same-name carries.

    Args:
        estimate (ResourceEstimate): Estimate whose serialized algebra is
            inspected.
        functions (Iterable[Any]): Identity-distinct undefined SymPy functions
            created for unsupported loop-carried recurrences.

    Returns:
        bool: Whether any resource or guarded metadata expression contains an
            application of one of the supplied functions.
    """
    unresolved = tuple(functions)
    return bool(unresolved) and any(
        expression.has(*unresolved)
        for expression in _serialization_expressions(estimate)
        if isinstance(expression, sp.Basic)
    )


def _validate_opaque_cost_provenance(
    estimate: ResourceEstimate,
    *,
    name: str,
    basis: GateBasis,
    control_decomposition: ControlDecomposition,
    precision: float,
) -> None:
    """Validate a fixed or callback-produced opaque cost model.

    Args:
        estimate (ResourceEstimate): Opaque cost result to validate.
        name (str): User-facing callable name for diagnostics.
        basis (GateBasis): Basis requested by the active estimator.
        control_decomposition (ControlDecomposition): Coherent-control
            decomposition requested by the active estimator.
        precision (float): Clifford+T synthesis precision requested by the
            active estimator.

    Raises:
        ValueError: If basis-sensitive cost metrics use incompatible basis or
            precision provenance.
    """
    if not _estimate_has_basis_sensitive_resources(estimate):
        return
    if estimate.basis is not basis:
        raise ValueError(
            f"Opaque cost for '{name}' uses basis {estimate.basis.value!r}, "
            f"but the estimator uses {basis.value!r}."
        )
    if estimate.control_decomposition is not control_decomposition:
        raise ValueError(
            f"Opaque cost for '{name}' uses control decomposition "
            f"{estimate.control_decomposition.value!r}, but the estimator "
            f"uses {control_decomposition.value!r}."
        )
    if basis is GateBasis.CLIFFORD_T and not _precisions_match(
        estimate.precision,
        precision,
    ):
        raise ValueError(
            f"Opaque cost for '{name}' uses precision "
            f"{estimate.precision!r}, but the estimator uses {precision!r}."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class OpaqueCostContext:
    """Describe the base Oracle definition requested from a cost callback.

    The callback models one ordinary application of the definition. Its
    result therefore includes ``definition_control_qubits`` but never controls
    added by a later ``qmc.control`` call or inherited from an enclosing
    controlled qkernel. The estimator applies those call-site transforms after
    the callback returns. The callback result is a complete definition-level
    contract and must include any phase-relevant work that those later
    coherent controls need to transform.

    Args:
        callable_name (str): Human-readable callable name.
        target_shapes (Mapping[str, tuple[ResourceExpr, ...]]): Definition
            target shapes keyed by formal operand name. A scalar qubit has an
            empty shape tuple.
        definition_control_qubits (int): Controls declared by the Oracle
            definition and already included in the callback's base cost.
        strategy (str | None): Selected base resource strategy. Defaults to
            ``None``.
        basis (GateBasis): Requested output gate basis. Defaults to
            ``LOGICAL``.
        control_decomposition (ControlDecomposition): Requested coherent
            control model. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.
        precision (float | None): Clifford+T synthesis precision, or ``None``
            for other bases. Defaults to ``None``.
    """

    callable_name: str
    target_shapes: Mapping[str, tuple[ResourceExpr, ...]]
    definition_control_qubits: int
    strategy: str | None = None
    basis: GateBasis = _DEFAULT_GATE_BASIS
    control_decomposition: ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION
    precision: float | None = None

    def __post_init__(self) -> None:
        """Validate and freeze definition-level callback inputs.

        Raises:
            TypeError: If the callable name, target names, target shapes, or
                definition control count have invalid Python types.
            ValueError: If the definition control count is negative.
        """
        if not isinstance(self.callable_name, str):
            raise TypeError("callable_name must be a string.")
        if isinstance(self.definition_control_qubits, bool) or not isinstance(
            self.definition_control_qubits, int
        ):
            raise TypeError("definition_control_qubits must be a plain Python int.")
        if self.definition_control_qubits < 0:
            raise ValueError("definition_control_qubits must be nonnegative.")
        normalized_shapes: dict[str, tuple[ResourceExpr, ...]] = {}
        for name, shape in self.target_shapes.items():
            if not isinstance(name, str):
                raise TypeError("target shape names must be strings.")
            if not isinstance(shape, tuple):
                raise TypeError(f"target shape for {name!r} must be a tuple.")
            normalized_shapes[name] = tuple(_expr(dimension) for dimension in shape)
        object.__setattr__(
            self,
            "target_shapes",
            MappingProxyType(normalized_shapes),
        )

    @property
    def target_qubits(self) -> ResourceExpr:
        """Return the flattened width of all definition targets.

        Returns:
            ResourceExpr: Sum of scalar targets and products of array
            dimensions.
        """
        return _safe_simplify(
            _expr(
                sum(
                    (
                        sp.prod(shape) if shape else _ONE
                        for shape in self.target_shapes.values()
                    ),
                    _ZERO,
                )
            )
        )


def _opaque_call_relative_width(
    width: WidthResources,
) -> tuple[WidthResources, ResourceExpr]:
    """Remove a standalone estimate's caller-owned width baseline.

    Explicit opaque costs may be copied from a root qkernel estimate, whose
    ``peak_qubits`` includes ``input_qubits``. At an Invoke boundary those
    operands are already live in the caller, so only the peak above that
    baseline belongs to the call body. Any residual peak not categorized as
    allocation or clean/dirty ancilla is retained as anonymous allocation. A
    partial declaration that omits ``peak_qubits`` is normalized so its peak
    cannot be smaller than its declared allocation and ancilla workspace.

    Args:
        width (WidthResources): Definition-level standalone width.

    Returns:
        tuple[WidthResources, ResourceExpr]: Body-relative width for one opaque
            invocation and the anonymous workspace added to its allocation
            count.
    """
    categorized_workspace = (
        width.allocated_qubits + width.clean_ancilla_qubits + width.dirty_ancilla_qubits
    )
    relative_peak = _safe_simplify(
        sp.Max(
            _ZERO,
            width.peak_qubits - width.input_qubits,
            categorized_workspace,
        )
    )
    anonymous_workspace = _safe_simplify(
        sp.Max(_ZERO, relative_peak - categorized_workspace)
    )
    relative_width = dataclasses.replace(
        width,
        input_qubits=_ZERO,
        allocated_qubits=width.allocated_qubits + anonymous_workspace,
        peak_qubits=relative_peak,
    )
    return relative_width, anonymous_workspace


@dataclasses.dataclass(frozen=True, slots=True)
class _OpaqueInvocationTransform:
    """Carry call-site transforms kept out of ``OpaqueCostContext``.

    Args:
        declared_controls (int): Controls represented by the base definition.
        added_controls (int): Controls prepended by a frontend transform.
        inherited_controls (ResourceExpr): Controls inherited from enclosing
            controlled bodies.
        inverse (bool): Whether to invert the base definition.
        control_value (int | None): Combined LSB-first activation value for
            added then declared local controls.
    """

    declared_controls: int
    added_controls: int
    inherited_controls: ResourceExpr
    inverse: bool
    control_value: int | None

    @property
    def external_controls(self) -> ResourceExpr:
        """Return controls the estimator applies to the base cost.

        Returns:
            ResourceExpr: Added plus inherited controls.
        """
        return self.inherited_controls + self.added_controls

    @property
    def local_controls(self) -> int:
        """Return the controls normalized by this Oracle invocation.

        Returns:
            int: Added plus definition-declared controls.
        """
        return self.added_controls + self.declared_controls

    @property
    def zero_controls(self) -> int:
        """Return zero-valued controls in the combined local condition.

        Returns:
            int: Number of local controls activated by zero.
        """
        return int(
            _zero_control_count(
                self.local_controls,
                self.control_value,
            )
        )


@dataclasses.dataclass
class _ResourceEstimatorConfig:
    """Configure ``ResourceEstimator`` behavior.

    Args:
        strategies (dict[str, str]): Strategy overrides by callable name.
        trace (bool): Whether estimates should carry trace nodes.
        simplify (bool): Whether to simplify the final estimate.
        unknown_policy (UnknownResourcePolicy): Handling for unknown opaque
            callables.
        basis (GateBasis): Output gate basis. Defaults to ``LOGICAL``.
        control_decomposition (ControlDecomposition): Coherent-control
            decomposition. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.
        precision (float): Approximation precision for rotation synthesis in
            ``CLIFFORD_T`` basis. Defaults to ``1e-10``.
    """

    strategies: dict[str, str] = dataclasses.field(default_factory=dict)
    trace: bool = False
    simplify: bool = True
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR
    basis: GateBasis = _DEFAULT_GATE_BASIS
    control_decomposition: ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION
    precision: float = _DEFAULT_ROTATION_SYNTHESIS_PRECISION


class ResourceEstimator:
    """Estimate algorithmic resources for qkernels and IR blocks."""

    def __init__(
        self,
        *,
        strategies: dict[str, str] | None = None,
        trace: bool = False,
        simplify: bool = True,
        unknown_policy: str | UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
        basis: str | GateBasis = _DEFAULT_GATE_BASIS,
        control_decomposition: str
        | ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION,
        precision: float = _DEFAULT_ROTATION_SYNTHESIS_PRECISION,
    ) -> None:
        """Initialize a resource estimator.

        Args:
            strategies (dict[str, str] | None): Strategy overrides by callable
                name. Defaults to ``None``.
            trace (bool): Whether to keep explanation traces. Defaults to
                ``False``.
            simplify (bool): Whether to simplify the final estimate. Defaults
                to ``True``.
            unknown_policy (str | UnknownResourcePolicy): Handling for unknown
                bodyless callables. Defaults to ``ERROR``.
            basis (str | GateBasis): Output gate basis. Defaults to
                ``LOGICAL``.
            control_decomposition (str | ControlDecomposition):
                Coherent-control decomposition. Defaults to
                ``CLEAN_ANCILLA_TOFFOLI``.
            precision (float): Rotation-synthesis precision for
                ``CLIFFORD_T`` basis. Defaults to ``1e-10``.

        Raises:
            ValueError: If ``unknown_policy``, ``basis``, or
                ``control_decomposition`` is unknown, or ``precision`` is
                outside ``(0, 1)``.
        """
        if not 0 < precision < 1:
            raise ValueError("precision must satisfy 0 < precision < 1.")
        try:
            normalized_unknown_policy = UnknownResourcePolicy(unknown_policy)
        except ValueError as error:
            valid = ", ".join(member.value for member in UnknownResourcePolicy)
            raise ValueError(
                f"unknown resource policy {unknown_policy!r}; expected one of: {valid}"
            ) from error
        try:
            normalized_basis = GateBasis(basis)
        except ValueError as error:
            valid = ", ".join(member.value for member in GateBasis)
            raise ValueError(
                f"unknown gate basis {basis!r}; expected one of: {valid}"
            ) from error
        try:
            normalized_control_decomposition = ControlDecomposition(
                control_decomposition
            )
        except ValueError as error:
            valid = ", ".join(member.value for member in ControlDecomposition)
            raise ValueError(
                "unknown control decomposition "
                f"{control_decomposition!r}; expected one of: {valid}"
            ) from error
        self.config = _ResourceEstimatorConfig(
            strategies=dict(strategies or {}),
            trace=trace,
            simplify=simplify,
            unknown_policy=normalized_unknown_policy,
            basis=normalized_basis,
            control_decomposition=normalized_control_decomposition,
            precision=precision,
        )

    def estimate(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        *,
        inputs: dict[str, Any] | None = None,
        strategies: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        """Estimate algorithmic resources for a qkernel, block, or operations.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Object to
                estimate. QKernel-like objects are built before traversal.
            inputs (dict[str, Any] | None): QKernel input values used to
                specialize the symbolic estimate without constructing a
                problem-sized circuit. Exact one-dimensional root quantum-port
                widths declared by callable resource metadata are inferred
                when omitted. Defaults to ``None``.
            strategies (dict[str, str] | None): Per-call override merged over
                estimator-level strategies. Defaults to ``None``.

        Returns:
            ResourceEstimate: Algorithmic resource estimate.

        Raises:
            ValueError: If an input name is unknown, a callable resource
                contract is malformed or violated, or a structural resource
                requirement fails.
            TypeError: If ``kernel`` is not a supported estimator input.
            NotImplementedError: If the input IR contains a construct not
                supported by resource estimation.
        """
        defer_token = _DEFER_RESOURCE_SYMBOL_METADATA.set(True)
        try:
            estimate = self._estimate_deferred(
                kernel,
                inputs=inputs,
                strategies=strategies,
            )
        finally:
            _DEFER_RESOURCE_SYMBOL_METADATA.reset(defer_token)
        estimate._refresh_symbol_metadata()
        return estimate

    def _estimate_deferred(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        *,
        inputs: dict[str, Any] | None = None,
        strategies: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        """Estimate resources while deferring public symbol derivation.

        The public :meth:`estimate` wrapper activates the deferral context and
        refreshes aliases and parameters exactly once on the final result.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Object to
                estimate. QKernel-like objects are built before traversal.
            inputs (dict[str, Any] | None): Values used to specialize symbolic
                qkernel inputs. Defaults to ``None``.
            strategies (dict[str, str] | None): Per-call strategy overrides.
                Defaults to ``None``.

        Returns:
            ResourceEstimate: Estimate awaiting one final public-symbol
            metadata refresh.

        Raises:
            ValueError: If an input, callable resource contract, or structural
                requirement is invalid.
            TypeError: If ``kernel`` is not a supported estimator input.
            NotImplementedError: If the input contains an unsupported
                construct.
        """
        explicit_inputs = dict(inputs or {})
        root_callable_attrs = _root_callable_resource_attrs(kernel)
        build_inputs, estimation_inputs = _partition_estimation_inputs(
            kernel,
            explicit_inputs,
        )
        block_or_ops = self._coerce_input(
            kernel,
            build_inputs,
            estimation_inputs,
        )
        config = dataclasses.replace(
            self.config,
            strategies={**self.config.strategies, **dict(strategies or {})},
        )
        interpreter = ResourceInterpreter(
            config=config,
            bindings=build_inputs,
            condition_values=_scalar_values({**build_inputs, **estimation_inputs}),
        )
        estimate = interpreter.estimate(block_or_ops)
        # Boundary liveness is interpreter-local metadata. A root estimate can
        # later be reused as an opaque definition cost, where caller owner
        # identities would be meaningless and unserializable. Drop it before
        # public symbol discovery so private size expressions cannot introduce
        # public parameters either.
        estimate = dataclasses.replace(
            estimate,
            _output_sizes={},
            _input_sizes={},
            _has_output_summary=False,
        )
        root_source = getattr(kernel, "name", None) or (
            block_or_ops.name if isinstance(block_or_ops, Block) else "qkernel"
        )
        if isinstance(block_or_ops, Block):
            estimate = _with_constraints(
                estimate,
                *_quantum_operand_width_constraints(
                    root_callable_attrs,
                    block_or_ops.input_values,
                    ExprResolver(
                        block=block_or_ops,
                        context=_root_input_binding_context(
                            block_or_ops,
                            build_inputs,
                        ),
                    ),
                    source=root_source,
                ),
            )
        # Runtime-domain alternatives may contain ordinary qkernel symbols.
        # Expand them before applying user inputs so the normal substitution
        # and constraint validation path specializes every alternative rather
        # than reintroducing an already supplied symbol afterward.
        estimate = interpreter.resolve_finite_runtime_constraints(estimate)
        if build_inputs:
            estimate._refresh_symbol_metadata()
            estimate = _substitute_bindings(estimate, build_inputs)
        inferred_shape_inputs = _root_callable_shape_inputs(
            root_callable_attrs,
            block_or_ops,
            explicit_inputs,
            source=root_source,
        )
        effective_estimation_inputs = {
            **inferred_shape_inputs,
            **estimation_inputs,
        }
        expanded_estimation_inputs, shape_input_names = _expand_array_shape_inputs(
            block_or_ops,
            effective_estimation_inputs,
        )
        if effective_estimation_inputs:
            estimate = _apply_inputs(
                estimate,
                expanded_estimation_inputs,
                contract_names=_contract_names(block_or_ops),
                input_types=_scalar_input_types(block_or_ops),
                branch_condition_names=interpreter.branch_condition_names,
                consumed_input_names=shape_input_names,
            )
        if not config.trace:
            # Large recursive call bodies can produce an explanation tree
            # deeper than Python's recursion limit.  A disabled trace is not
            # observable, so discard it before symbolic mapping/simplification.
            estimate = dataclasses.replace(estimate, trace=None)
        if config.simplify:
            estimate = estimate.simplify()
        estimate = dataclasses.replace(
            estimate,
            basis=config.basis,
            control_decomposition=config.control_decomposition,
            precision=(
                config.precision if config.basis is GateBasis.CLIFFORD_T else None
            ),
        )
        interpreter.validate_no_internal_resource_symbols(estimate)
        return estimate

    def _coerce_input(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        build_inputs: dict[str, Any],
        estimation_inputs: dict[str, Any],
    ) -> Block | Sequence[Operation]:
        """Coerce a supported input into an IR block or operation list.

        Parameterizable inputs stay symbolic while non-parameterizable inputs
        are supplied during tracing. This keeps problem sizes scalable without
        requiring users to distinguish build-time from estimation-time inputs.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Input
                object.
            build_inputs (dict[str, Any]): Structural values supplied while
                tracing the qkernel.
            estimation_inputs (dict[str, Any]): Parameterizable values kept
                symbolic until after interpretation.

        Returns:
            Block | Sequence[Operation]: IR object ready for interpretation.
        """
        if isinstance(kernel, Block):
            return kernel
        if isinstance(kernel, Sequence):
            return kernel
        build = getattr(kernel, "build", None)
        if callable(build):
            parameters = _estimator_parameters(kernel, build_inputs)
            return build(parameters=parameters, **build_inputs)
        block = getattr(kernel, "block", None)
        if isinstance(block, Block):
            return block
        raise TypeError(
            "ResourceEstimator.estimate() expects a QKernel, Block, or "
            "sequence of Operation objects."
        )


def _find_runtime_observation_results(
    operations: Sequence[Operation],
) -> set[str]:
    """Return classical results that must remain runtime-dependent.

    Expectation values behave like observations for estimator scheduling and
    branch specialization, but they deliberately are not sample-only
    ``KernelEffect.MEASUREMENT`` effects. Keeping the additional seeds local to
    the estimator avoids changing compiler effect validation. Invoke results
    are deliberately excluded because each caller adds provenance from the
    implementation selected by the current resource strategy.

    Args:
        operations (Sequence[Operation]): Operation tree to inspect.

    Returns:
        set[str]: Measurement, projection-bit, and expectation-result UUIDs.
    """
    results = find_measurement_results(operations)
    for operation in walk_operations(operations):
        if isinstance(operation, ExpvalOp):
            results.update(result.uuid for result in operation.results)
    return results


def _operation_input_taint_condition(
    operation: Operation,
    taint_conditions: Mapping[str, Boolean],
) -> Boolean:
    """Return when an operation consumes observation-derived classical data.

    Args:
        operation (Operation): Operation whose inputs should be inspected.
        taint_conditions (Mapping[str, Boolean]): UUID-keyed observation
            provenance in the current evaluation scope.

    Returns:
        Boolean: Union of active provenance conditions on classical inputs.
    """
    conditions = [
        _value_taint_condition(value, taint_conditions)
        for value in operation.all_input_values()
        if isinstance(value, ValueBase) and not value.type.is_quantum()
    ]
    return _boolean_condition(sp.Or(*conditions)) if conditions else sp.false


def _value_taint_condition(
    value: ValueBase,
    taint_conditions: Mapping[str, Boolean],
) -> Boolean:
    """Return guarded observation provenance through value ancestry.

    Args:
        value (ValueBase): Classical scalar, container, or array element.
        taint_conditions (Mapping[str, Boolean]): UUID-keyed provenance map.

    Returns:
        Boolean: Union of provenance on the value and structural ancestors.
    """
    conditions = [
        taint_conditions[uuid]
        for uuid in collect_value_like_uuids(cast(ValueLike, value))
        if uuid in taint_conditions
    ]
    return _boolean_condition(sp.Or(*conditions)) if conditions else sp.false


def _merge_classical_source_conditions(
    target: dict[str, Boolean],
    additions: Mapping[str, sp.Basic],
) -> None:
    """Merge guarded observation-source conditions into one mapping.

    Args:
        target (dict[str, Boolean]): Mutable destination keyed by source token.
        additions (Mapping[str, sp.Basic]): Source guards to union into the
            destination.
    """
    for source, raw_condition in additions.items():
        condition = _boolean_condition(raw_condition)
        merged = _boolean_condition(sp.Or(target.get(source, sp.false), condition))
        if merged is sp.false:
            target.pop(source, None)
        else:
            target[source] = merged


def _resolved_classical_source_conditions(
    values: Iterable[ValueBase],
    resolver: ExprResolver,
    *,
    ignored_uuids: frozenset[str] = frozenset(),
) -> dict[str, Boolean]:
    """Resolve semantic classical dependencies without broadening elements.

    A classical array element is resolved as one fact. Its parent array is not
    traversed again, because doing so would reintroduce a coarse whole-array
    dependency after the persistent array state proved the selected element
    clean. Quantum values contribute only classical address metadata.

    Args:
        values (Iterable[ValueBase]): Values whose semantic dependencies are
            collected.
        resolver (ExprResolver): Resolver carrying classical facts.
        ignored_uuids (frozenset[str]): Structural inputs that are not consumed
            by the operation. Defaults to an empty set.

    Returns:
        dict[str, Boolean]: Classical source tokens and activation guards.
    """
    source_conditions: dict[str, Boolean] = {}

    def visit(value: ValueBase) -> None:
        """Collect dependencies from one aggregate or scalar value.

        Args:
            value (ValueBase): Value to inspect.
        """
        if isinstance(value, TupleValue):
            for element in value.elements:
                visit(element)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit(key)
                visit(entry_value)
            return
        if not isinstance(value, Value) or value.uuid in ignored_uuids:
            return
        if value.type.is_quantum():
            metadata: list[Value] = []
            if isinstance(value, ArrayValue):
                metadata.extend(value.shape)
                if value.slice_start is not None:
                    metadata.append(value.slice_start)
                if value.slice_step is not None:
                    metadata.append(value.slice_step)
            else:
                metadata.extend(value.element_indices)
            for item in metadata:
                _merge_classical_source_conditions(
                    source_conditions,
                    resolver.resolve_classical_fact(item).dependencies,
                )
            return
        if isinstance(value, ArrayValue):
            _merge_classical_source_conditions(
                source_conditions,
                resolver.array_state_dependencies(value),
            )
            for item in (
                *value.shape,
                *((value.slice_start,) if value.slice_start is not None else ()),
                *((value.slice_step,) if value.slice_step is not None else ()),
            ):
                _merge_classical_source_conditions(
                    source_conditions,
                    resolver.resolve_classical_fact(item).dependencies,
                )
            return
        _merge_classical_source_conditions(
            source_conditions,
            resolver.resolve_classical_fact(value).dependencies,
        )

    for root in values:
        visit(root)
    return source_conditions


def _resolved_classical_facts(
    values: Iterable[ValueBase],
    resolver: ExprResolver,
) -> tuple[_ResolvedClassicalFact, ...]:
    """Resolve semantic scalar or aggregate facts without broad ancestry.

    Args:
        values (Iterable[ValueBase]): Classical values read by one operation.
        resolver (ExprResolver): Resolver carrying value and array state.

    Returns:
        tuple[_ResolvedClassicalFact, ...]: Facts associated with the supplied
            semantic inputs.
    """
    facts: list[_ResolvedClassicalFact] = []

    def visit(value: ValueBase) -> None:
        """Append facts represented by one structured value.

        Args:
            value (ValueBase): Value to inspect.
        """
        if isinstance(value, TupleValue):
            for element in value.elements:
                visit(element)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit(key)
                visit(entry_value)
            return
        if isinstance(value, Value) and not value.type.is_quantum():
            facts.append(resolver.resolve_classical_fact(value))

    for root in values:
        visit(root)
    return tuple(facts)


def _classical_fact_runtime_condition(
    fact: _ResolvedClassicalFact,
) -> Boolean:
    """Return when a resolved classical fact depends on an observation.

    Args:
        fact (_ResolvedClassicalFact): Resolved value and guarded source tokens.

    Returns:
        Boolean: Union of all active source guards, or false when independent.
    """
    dependencies = {
        source: guard
        for source, guard in fact.dependencies.items()
        if source.startswith(_OBSERVATION_SOURCE_PREFIX)
    }
    return (
        _boolean_condition(sp.Or(*dependencies.values())) if dependencies else sp.false
    )


def _classical_fact_uncertainty_condition(
    fact: _ResolvedClassicalFact,
) -> Boolean:
    """Return when a fact depends on a conservative loop-array fallback.

    Loop-array source tokens are scheduler readiness edges, not runtime
    observations. Keeping this condition separate prevents an unresolved
    symbolic loop state from being mistaken for measurement feed-forward.

    Args:
        fact (_ResolvedClassicalFact): Resolved value and guarded source tokens.

    Returns:
        Boolean: Union of active loop-array fallback guards, or false.
    """
    dependencies = {
        source: guard
        for source, guard in fact.dependencies.items()
        if source.startswith(_LOOP_ARRAY_SOURCE_PREFIX)
    }
    return (
        _boolean_condition(sp.Or(*dependencies.values())) if dependencies else sp.false
    )


def _operation_classical_dependency_inputs(
    operation: Operation,
) -> tuple[ValueBase, ...]:
    """Return values semantically read by one scheduling boundary.

    Structured ``if`` and loop regions require guarded traversal and are
    handled by ``_scheduling_classical_input_sources``. Other operations retain
    their ordinary IR input list.

    Args:
        operation (Operation): Operation whose semantic reads are requested.

    Returns:
        tuple[ValueBase, ...]: Values whose classical readiness can delay the
            operation.
    """
    return tuple(operation.all_input_values())


def _operation_classical_dependency_outputs(
    operation: Operation,
) -> tuple[ValueBase, ...]:
    """Return classical boundary values whose source tokens become ready.

    Loop-carried array rewrites are represented by ``LoopCarriedRebind``
    records rather than ordinary operation results. Their exposed ``after``
    values must still publish accumulated observation readiness at the loop
    boundary so later feed-forward work cannot start at loop entry.

    Args:
        operation (Operation): Operation whose boundary outputs are requested.

    Returns:
        tuple[ValueBase, ...]: Ordinary results plus loop-rebound values.
    """
    outputs: list[ValueBase] = list(operation.results)
    if isinstance(operation, (ForOperation, ForItemsOperation, WhileOperation)):
        outputs.extend(rebind.after for rebind in _loop_array_state_rebinds(operation))
        outputs.extend(
            rebind.after
            for rebind in operation.loop_carried_rebinds
            if not isinstance(rebind.after, ArrayValue)
        )
    return tuple(outputs)


def _loop_array_state_rebinds(
    operation: ForOperation | ForItemsOperation | WhileOperation,
) -> tuple[LoopCarriedRebind, ...]:
    """Return explicit and inferred classical-array loop state boundaries.

    A direct element Store can expose its produced array SSA value after a loop
    without creating a frontend ``LoopCarriedRebind`` record. The estimator
    infers that boundary from each logical lineage's unproduced entry operand
    and last produced result so detached iteration scopes retain the same
    semantics as the shared traced graph.

    Args:
        operation (ForOperation | ForItemsOperation | WhileOperation): Loop
            whose classical-array state boundaries are requested.

    Returns:
        tuple[LoopCarriedRebind, ...]: Explicit records followed by inferred
        classical-array records in body order.
    """
    explicit = tuple(
        rebind
        for rebind in operation.loop_carried_rebinds
        if isinstance(rebind.before, ArrayValue)
        and isinstance(rebind.after, ArrayValue)
        and not rebind.before.type.is_quantum()
        and not rebind.after.type.is_quantum()
    )
    covered_lineages = {
        cast(ArrayValue, rebind.before).logical_id for rebind in explicit
    }
    nested = tuple(walk_operations(operation.operations))
    produced = {
        result.uuid
        for nested_operation in nested
        for result in nested_operation.results
        if isinstance(result, ArrayValue)
    }
    entries: dict[str, ArrayValue] = {}
    exits: dict[str, ArrayValue] = {}
    for nested_operation in nested:
        for operand in nested_operation.operands:
            if (
                isinstance(operand, ArrayValue)
                and not operand.type.is_quantum()
                and operand.uuid not in produced
            ):
                entries.setdefault(operand.logical_id, operand)
        for result in nested_operation.results:
            if isinstance(result, ArrayValue) and not result.type.is_quantum():
                exits[result.logical_id] = result
    inferred = tuple(
        LoopCarriedRebind(
            var_name=entry.name,
            before=entry,
            after=exits[logical_id],
        )
        for logical_id, entry in entries.items()
        if logical_id in exits and logical_id not in covered_lineages
    )
    return (*explicit, *inferred)


def _structural_value_ancestry(value: ValueBase) -> tuple[ValueBase, ...]:
    """Return one value and recursively embedded structural values.

    Args:
        value (ValueBase): Scalar, array, tuple, or dictionary IR value.

    Returns:
        tuple[ValueBase, ...]: Identity-deduplicated ancestry in visit order.
    """
    ordered: list[ValueBase] = []
    visited: set[str] = set()

    def visit(current: ValueBase) -> None:
        """Visit one structural value.

        Args:
            current (ValueBase): Value whose embedded ancestry is traversed.
        """
        if current.uuid in visited:
            return
        visited.add(current.uuid)
        ordered.append(current)
        if isinstance(current, TupleValue):
            for element in current.elements:
                visit(element)
        elif isinstance(current, DictValue):
            for key, entry_value in current.entries:
                visit(key)
                visit(entry_value)
        elif isinstance(current, ArrayValue):
            for dimension in current.shape:
                visit(dimension)
            if current.slice_of is not None:
                visit(current.slice_of)
            if current.slice_start is not None:
                visit(current.slice_start)
            if current.slice_step is not None:
                visit(current.slice_step)
        elif isinstance(current, Value):
            if current.parent_array is not None:
                visit(current.parent_array)
            for index in current.element_indices:
                visit(index)

    visit(value)
    return tuple(ordered)


def _direct_observation_result_uuids(operation: Operation) -> tuple[str, ...]:
    """Return result UUIDs directly produced by a runtime observation.

    Args:
        operation (Operation): Operation whose direct results should be
            classified.

    Returns:
        tuple[str, ...]: Direct measurement or expectation-value result UUIDs.
    """
    if isinstance(
        operation,
        (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation, ExpvalOp),
    ):
        return tuple(result.uuid for result in operation.results)
    if isinstance(operation, ProjectOperation) and len(operation.results) > 1:
        return (operation.results[1].uuid,)
    return ()


def _direct_observation_source_conditions(
    operation: Operation,
    resolver: ExprResolver,
) -> dict[str, Boolean]:
    """Return source tokens created by one direct observation operation.

    Array-state projection deliberately avoids treating a whole array as one
    element dependency. A vector observation is the exception at its producer
    boundary: its aggregate result fact is where the scheduler first learns
    that the source token becomes ready. Element reads remain precise after
    this publication.

    Args:
        operation (Operation): Operation whose direct observation results are
            inspected.
        resolver (ExprResolver): Resolver containing the newly published
            result facts.

    Returns:
        dict[str, Boolean]: Newly created source tokens and activation guards.
    """
    direct_results = frozenset(_direct_observation_result_uuids(operation))
    source_conditions: dict[str, Boolean] = {}
    for result in operation.results:
        if not isinstance(result, Value) or result.uuid not in direct_results:
            continue
        _merge_classical_source_conditions(
            source_conditions,
            resolver.resolve_classical_fact(result).dependencies,
        )
    return source_conditions


def _propagate_operation_measurement_taint(
    operation: Operation,
    estimate: ResourceEstimate,
    inherited: Mapping[str, Boolean],
) -> dict[str, Boolean]:
    """Propagate guarded observation provenance across one operation.

    Structured operations publish branch- and loop-aware result conditions in
    their estimate. Ordinary operations retain the existing conservative
    all-input-to-all-result dataflow rule without flattening nested region
    edges into the enclosing scope.

    Args:
        operation (Operation): Evaluated operation.
        estimate (ResourceEstimate): Operation estimate carrying any nested
            result provenance.
        inherited (Mapping[str, Boolean]): Provenance before the operation.

    Returns:
        dict[str, Boolean]: Updated provenance for the current scope.
    """
    updated = _merge_measurement_taint_conditions(
        inherited,
        estimate._measurement_taint_conditions,
    )
    if not isinstance(operation, HasNestedOps):
        input_condition = _operation_input_taint_condition(operation, updated)
        if input_condition is not sp.false:
            updated = _merge_measurement_taint_conditions(
                updated,
                {result.uuid: input_condition for result in operation.results},
            )
    direct = _direct_observation_result_uuids(operation)
    if direct:
        updated = _merge_measurement_taint_conditions(
            updated,
            {uuid: sp.true for uuid in direct},
        )
    return updated


@dataclasses.dataclass(frozen=True)
class _LoopMayTaint:
    """Summarize path-insensitive observation provenance for a loop.

    Args:
        at_iteration (dict[str, Boolean]): Carried block arguments that may be
            observation-derived in at least one loop iteration.
        final (dict[str, Boolean]): Carried block arguments whose corresponding
            loop results may be observation-derived after the loop.
    """

    at_iteration: dict[str, Boolean]
    final: dict[str, Boolean]


def _loop_may_taint(
    operation: ForOperation | ForItemsOperation,
    initial: Mapping[str, Boolean],
    probe_estimate: ResourceEstimate,
) -> _LoopMayTaint:
    """Compute a two-point may-be-runtime fixed point for a symbolic loop.

    The analysis intentionally discards path and iteration correlations.  A
    source that can reach a carried value in any iteration marks that value as
    runtime-derived for resource decisions.  This is the requested
    conservative policy and avoids constructing a symbolic Boolean recurrence
    whose exactness is not useful once runtime branches are combined by their
    field-wise maxima.

    Args:
        operation (ForOperation | ForItemsOperation): Symbolic region loop.
        initial (Mapping[str, Boolean]): Observation provenance on entry.
        probe_estimate (ResourceEstimate): One body probe carrying selected
            callable observation results and ordinary dataflow propagation.

    Returns:
        _LoopMayTaint: May-provenance at an arbitrary iteration and loop exit.
    """
    graph = build_dependency_graph(operation.operations)
    for arg in operation.region_args:
        # The next iteration's block argument receives this iteration's yield.
        graph.setdefault(arg.block_arg.uuid, set()).add(arg.yielded.uuid)
    seeds = {uuid for uuid, condition in initial.items() if condition is not sp.false}
    seeds.update(
        uuid
        for uuid, condition in probe_estimate._measurement_taint_conditions.items()
        if condition is not sp.false
    )
    # A yielded array element can depend on an observation through its index
    # or view metadata without having a producer edge of its own. Seed that
    # synthetic leaf explicitly so the loop backedge cannot turn it into an
    # apparently clean public carry.
    seeds.update(
        arg.yielded.uuid
        for arg in operation.region_args
        if _value_taint_condition(
            arg.yielded,
            probe_estimate._measurement_taint_conditions,
        )
        is not sp.false
    )
    derived = find_measurement_derived_values(graph, seeds)
    at_iteration = {
        arg.block_arg.uuid: sp.true
        for arg in operation.region_args
        if arg.block_arg.uuid in derived
        or initial.get(arg.block_arg.uuid, sp.false) is not sp.false
    }
    final = {
        arg.block_arg.uuid: sp.true
        for arg in operation.region_args
        if initial.get(arg.block_arg.uuid, sp.false) is not sp.false
        or arg.yielded.uuid in derived
    }
    return _LoopMayTaint(at_iteration=at_iteration, final=final)


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
    free_assumption_symbols = assumption.free_symbols
    zero_substitutions: dict[sp.Symbol, sp.Integer] = {}
    for atom in sp.preorder_traversal(assumption):
        if isinstance(atom, sp.Equality):
            if (
                isinstance(atom.lhs, sp.Symbol)
                and atom.lhs in free_assumption_symbols
                and atom.rhs == _ZERO
            ):
                zero_substitutions[atom.lhs] = _ZERO
            elif (
                isinstance(atom.rhs, sp.Symbol)
                and atom.rhs in free_assumption_symbols
                and atom.lhs == _ZERO
            ):
                zero_substitutions[atom.rhs] = _ZERO
        elif isinstance(atom, sp.LessThan):
            if (
                isinstance(atom.lhs, sp.Symbol)
                and atom.lhs in free_assumption_symbols
                and atom.lhs.is_nonnegative is True
                and atom.rhs == _ZERO
            ):
                zero_substitutions[atom.lhs] = _ZERO
    narrowed = cast(Boolean, condition.xreplace(zero_substitutions))
    try:
        folded = cast(Boolean, sp.piecewise_fold(narrowed))
    except (RecursionError, TypeError, ValueError):
        folded = narrowed

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


class ResourceInterpreter:
    """Abstractly interpret IR operations into resource algebra values."""

    def __init__(
        self,
        *,
        config: _ResourceEstimatorConfig,
        bindings: Mapping[str, Any],
        condition_values: Mapping[str, sp.Expr] | None = None,
    ) -> None:
        """Initialize an interpreter.

        Args:
            config (_ResourceEstimatorConfig): Estimator configuration.
            bindings (Mapping[str, Any]): Concrete user bindings.
            condition_values (Mapping[str, sp.Expr] | None): Numeric scalar
                input values used to decide compile-time ``if`` branches and
                resolve physical dependency indices. Defaults to ``None``.
        """
        self.config = config
        self.bindings = bindings
        self.condition_values: Mapping[str, sp.Expr] = condition_values or {}
        # Names of classical parameters used during interpretation for branch
        # decisions or physical dependency scheduling, so downstream input
        # reporting does not misfile them as resource-irrelevant.
        self.branch_condition_names: set[str] = set()
        # Undecidable-branch messages already reported, so an IfOperation
        # duplicated at trace time (e.g. a Python-level loop) yields one
        # assumption, not one per copy.
        self._reported_undecidable: set[str] = set()
        self._measurement_taint_conditions: dict[str, Boolean] = {}
        # Direct observation result UUIDs identify traced SSA definitions, not
        # repeated runtime occurrences. Concrete loop replay extends this path
        # per iteration so independent observations receive independent
        # scheduler readiness tokens. Symbolic loops use one family scope.
        self._observation_occurrence_path: tuple[tuple[str, int, int], ...] = ()
        # Fresh symbols created for shot-dependent branch merges are semantic
        # placeholders, not user-substitutable resource parameters. Keep their
        # identities separate from the equally fresh placeholders that model
        # unsupported but compile-time loop recurrences.
        self._runtime_observation_symbols: set[sp.Symbol] = set()
        self._runtime_value_domains: dict[
            sp.Symbol,
            tuple[sp.Expr, ...] | None,
        ] = {}
        # Symbols used only as internal placeholders for an unsupported
        # loop-carried value must never become user-substitutable resource
        # parameters. Finalization rejects an estimate if one reaches a
        # public metric, constraint, guard, or trace expression.
        self._unresolved_resource_symbols: set[sp.Symbol] = set()
        # Structural constraints discovered while interpreting an inactive
        # branch or zero-trip loop must remain guarded until specialization.
        self._constraint_scope_condition: Boolean = sp.true
        # Array legality is a pure function of the fully resolved structural
        # constraint. Repeated body invocations commonly rediscover the same
        # element bounds, so retain whether each one was already proven.
        self._array_constraint_proven: dict[_ResourceConstraint, bool] = {}
        # Selected callable bodies can expose runtime observations that are
        # not represented by KernelEffect, notably expectation values. Cache
        # their output indices and whether the body contains any observation,
        # retaining the Block to guard against id reuse.
        self._runtime_observation_cache: dict[
            int,
            tuple[Block, frozenset[int], bool],
        ] = {}
        # Synthetic tuple carriers retain physical parent UUIDs rather than
        # Value ancestry. Keep the corresponding allocation-owner identity
        # across nested control-flow and callable evaluation scopes.
        self._allocation_owners_by_uuid: dict[str, str] = {}
        # Conditional merge results have fresh SSA owners while still
        # referring to one of their branch-source allocations. Dependency
        # scheduling retains those possible physical aliases independently of
        # liveness owner resolution so post-merge work cannot run beside its
        # selected producer.
        self._dependency_owner_aliases: dict[str, frozenset[str]] = {}
        # Recursive kernels are valid when concrete inputs reach a base case.
        # Track resolved call states so only cycles or symbolically changing
        # recurrences fail early; a terminating concrete recursion has no
        # estimator-specific depth ceiling.
        self._active_call_states: dict[int, list[tuple[sp.Expr, ...]]] = {}
        # Batch-profile discovery recursively follows callable bodies before
        # their resource evaluation. Track resolved call states independently
        # so concrete recursion may reach its base case while symbolic cycles
        # stop without exhausting Python's call stack.
        self._active_batch_profile_states: dict[
            int,
            list[tuple[sp.Expr, ...]],
        ] = {}
        # Opaque callbacks describe definition-level costs and may be inspected
        # once for control batching before normal operation evaluation. Cache
        # that base result so profiling never executes user code twice for one
        # call context. The value keeps the operation strongly referenced and
        # identity-checked, matching the other id-keyed interpreter caches.
        self._opaque_definition_cost_cache: dict[
            tuple[Any, ...],
            tuple[InvokeOperation, ResourceEstimate],
        ] = {}
        # While loops have no source-level induction variable, so expose a
        # deterministic lexical-order name for each independent trip count.
        # Strongly reference every operation used by the id-keyed map so probe
        # evaluation and skipped branches cannot consume or reassign a name.
        self._while_trip_count_names: dict[
            tuple[tuple[tuple[int, int], ...], int],
            tuple[WhileOperation, str],
        ] = {}
        self._while_name_scan_operations: dict[
            tuple[tuple[tuple[int, int], ...], int],
            Operation,
        ] = {}

    @contextmanager
    def _guarded_constraint_scope(
        self,
        active_when: sp.Basic,
    ) -> Generator[None, None, None]:
        """Conjoin one activation guard while nested constraints are built.

        Args:
            active_when (sp.Basic): Predicate under which the nested scope can
                execute.

        Yields:
            None: Control returns to the caller while the guard is active.
        """
        previous = self._constraint_scope_condition
        self._constraint_scope_condition = _and_conditions(previous, active_when)
        try:
            yield
        finally:
            self._constraint_scope_condition = previous

    @contextmanager
    def _measurement_taint_scope(
        self,
        additions: Mapping[str, Boolean],
    ) -> Generator[None, None, None]:
        """Expose temporary guarded observation provenance in a nested scope.

        Args:
            additions (Mapping[str, Boolean]): UUID-keyed provenance to merge
                for the nested evaluation.

        Yields:
            None: Control returns while the temporary mapping is visible.
        """
        previous = self._measurement_taint_conditions
        self._measurement_taint_conditions = _merge_measurement_taint_conditions(
            previous,
            additions,
        )
        try:
            yield
        finally:
            self._measurement_taint_conditions = previous

    @contextmanager
    def _observation_occurrence_scope(
        self,
        kind: str,
        operation: Operation,
        ordinal: int,
    ) -> Generator[None, None, None]:
        """Qualify direct observation tokens within one repeated body visit.

        Args:
            kind (str): Stable loop-family label such as ``"range"``.
            operation (Operation): Repeated operation owning the body.
            ordinal (int): Concrete visit ordinal, or ``-1`` for one symbolic
                occurrence family.

        Yields:
            None: Control returns while newly published observation tokens are
            qualified by this occurrence path.
        """
        previous = self._observation_occurrence_path
        self._observation_occurrence_path = (
            *previous,
            (kind, id(operation), ordinal),
        )
        try:
            yield
        finally:
            self._observation_occurrence_path = previous

    def _observation_source_token(
        self,
        operation: Operation,
        result_uuid: str,
        result_slot: int,
        resolver: ExprResolver,
    ) -> str:
        """Return one identity-qualified scheduler token for an observation.

        Args:
            operation (Operation): Direct observation operation.
            result_uuid (str): UUID of the observed classical result.
            result_slot (int): Position in the operation's direct results.
            resolver (ExprResolver): Resolver carrying the callable path.

        Returns:
            str: Token unique to the call path, repeated occurrence, and slot.
        """
        call_scope = "/".join(
            f"{call_id:x}:{body_id:x}" for call_id, body_id in resolver.structural_scope
        )
        occurrence = "/".join(
            f"{kind}:{owner_id:x}:{ordinal}"
            for kind, owner_id, ordinal in self._observation_occurrence_path
        )
        scope = "/".join(part for part in (call_scope, occurrence) if part)
        identity = (
            f"{result_uuid}@{scope}#{result_slot}"
            if scope
            else f"{result_uuid}#{result_slot}"
        )
        return f"{_OBSERVATION_SOURCE_PREFIX}:{identity}"

    @contextmanager
    def _isolated_loop_taint_probe_state(self) -> Generator[None, None, None]:
        """Prevent discarded loop probes from mutating interpretation results.

        Definition-cost and observation-summary caches remain shared because
        they are identity-keyed pure memoization and prevent user callbacks from
        being executed repeatedly. Runtime-observation symbol identities also
        remain shared because the probe resolver expressions are reused by the
        recurrence analysis after this scope exits. Reporting, constraint, and
        owner state is restored so extra provenance probes cannot suppress
        assumptions, claim input usage, or change final scheduling metadata.

        Yields:
            None: Control returns while disposable probe mutations are isolated.
        """
        branch_condition_names = set(self.branch_condition_names)
        reported_undecidable = set(self._reported_undecidable)
        constraint_scope_condition = self._constraint_scope_condition
        array_constraint_proven = dict(self._array_constraint_proven)
        allocation_owners_by_uuid = dict(self._allocation_owners_by_uuid)
        dependency_owner_aliases = dict(self._dependency_owner_aliases)
        try:
            yield
        finally:
            self.branch_condition_names.clear()
            self.branch_condition_names.update(branch_condition_names)
            self._reported_undecidable.clear()
            self._reported_undecidable.update(reported_undecidable)
            self._constraint_scope_condition = constraint_scope_condition
            self._array_constraint_proven.clear()
            self._array_constraint_proven.update(array_constraint_proven)
            self._allocation_owners_by_uuid.clear()
            self._allocation_owners_by_uuid.update(allocation_owners_by_uuid)
            self._dependency_owner_aliases.clear()
            self._dependency_owner_aliases.update(dependency_owner_aliases)

    def _record_runtime_value_symbols(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> None:
        """Classify resolver fallbacks for observation-derived IR values.

        Ordinary expressions can mix runtime state with public inputs, for
        example ``measured_choice + n``.  Their free symbols must therefore
        remain classified independently.  Only the resolver-owned fallback
        for a value that cannot otherwise be expressed is newly internalized.
        This covers direct observations and array contents selected through a
        runtime-derived index or view without mistaking ``n`` for an outcome.

        Args:
            operation (Operation): Recently evaluated operation and its inputs.
            resolver (ExprResolver): Resolver that names scalar values.
        """
        candidates = (
            value
            for root in (*operation.all_input_values(), *operation.results)
            for value in _structural_value_ancestry(root)
            if isinstance(value, Value) and not value.type.is_quantum()
        )
        for value in candidates:
            if (
                _value_taint_condition(
                    value,
                    self._measurement_taint_conditions,
                )
                is sp.false
            ):
                continue
            symbol = resolver.unresolved_fallback_symbol(value)
            if symbol is None:
                continue
            self._runtime_observation_symbols.add(symbol)
            if symbol in self._runtime_value_domains:
                continue
            self._runtime_value_domains[symbol] = (
                (_ZERO, _ONE) if isinstance(value.type, BitType) else None
            )

    def _publish_operation_classical_facts(
        self,
        operation: Operation,
        resolver: ExprResolver,
        input_sources: Mapping[str, Boolean],
    ) -> None:
        """Publish one operation's classical values into the shared fact model.

        Direct observations create readiness-source tokens. Ordinary
        classical results retain the guarded sources of their semantic inputs.
        Nested operations publish their own boundary results because their
        branch, loop, or callable structure determines more precise facts.

        Args:
            operation (Operation): Operation whose results were just evaluated.
            resolver (ExprResolver): Resolver updated for subsequent work.
            input_sources (Mapping[str, Boolean]): Observation sources consumed
                by the operation before it was evaluated.
        """
        direct_sources = _direct_observation_result_uuids(operation)
        if direct_sources:
            direct_slots = {
                result_uuid: slot for slot, result_uuid in enumerate(direct_sources)
            }
            for result in operation.results:
                if not isinstance(result, Value) or result.uuid not in direct_slots:
                    continue
                source_token = self._observation_source_token(
                    operation,
                    result.uuid,
                    direct_slots[result.uuid],
                    resolver,
                )
                resolver.bind_classical_fact(
                    result,
                    _ResolvedClassicalFact.create(
                        resolver.resolve(result),
                        {source_token: sp.true},
                    ),
                )
            return
        if isinstance(operation, (HasNestedOps, InvokeOperation)):
            return
        source_facts = _resolved_classical_facts(
            _operation_classical_dependency_inputs(operation),
            resolver,
        )
        for result in operation.results:
            if not isinstance(result, Value) or result.type.is_quantum():
                continue
            resolved = resolver.resolve(result)
            if resolver.unresolved_fallback_symbol(result) is not None:
                fact = _ResolvedClassicalFact.create(resolved, input_sources)
            else:
                fact = _fact_from_expression(resolved, *source_facts)
            resolver.bind_classical_fact(
                result,
                fact,
            )

    def _scheduling_classical_input_sources(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> dict[str, Boolean]:
        """Resolve guarded classical reads for one scheduling boundary.

        A structured operation contributes only the reads of regions that can
        execute. Compile-time branch predicates therefore guard reads inside
        their respective regions instead of turning every captured source into
        an unconditional dependency. Range-local predicates are projected over
        the finite iteration domain, so a read is retained exactly when at
        least one reachable iteration can execute it.

        Args:
            operation (Operation): Operation whose scheduling reads are needed.
            resolver (ExprResolver): Resolver for the enclosing scope.

        Returns:
            dict[str, Boolean]: Classical source tokens and their execution
            guards at the operation boundary.
        """
        collected: dict[str, Boolean] = {}

        def project_over_ranges(
            condition: Boolean,
            ranges: tuple[
                tuple[sp.Symbol, ResourceExpr, ResourceExpr, ResourceExpr],
                ...,
            ],
        ) -> Boolean:
            """Project a per-iteration guard to the enclosing boundary.

            Args:
                condition (Boolean): Guard expressed in nested loop scopes.
                ranges (tuple[tuple[sp.Symbol, ResourceExpr, ResourceExpr,
                    ResourceExpr], ...]): Enclosing range domains from outer
                    to inner.

            Returns:
                Boolean: Existential guard with every range induction symbol
                bound inside a finite-domain predicate.
            """
            projected: sp.Basic = condition
            for loop_symbol, start, step, iterations in reversed(ranges):
                projected = _activation_over_range(
                    projected,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                )
            return _boolean_condition(projected)

        def add_values(
            values: Iterable[ValueBase],
            scope: ExprResolver,
            active_when: Boolean,
            ranges: tuple[
                tuple[sp.Symbol, ResourceExpr, ResourceExpr, ResourceExpr],
                ...,
            ],
        ) -> None:
            """Add guarded source tokens from semantic input values.

            Args:
                values (Iterable[ValueBase]): Values read by the region.
                scope (ExprResolver): Resolver for the values' region.
                active_when (Boolean): Guard under which the read executes.
                ranges (tuple[tuple[sp.Symbol, ResourceExpr, ResourceExpr,
                    ResourceExpr], ...]): Enclosing finite range domains.
            """
            if active_when is sp.false:
                return
            additions = _resolved_classical_source_conditions(values, scope)
            _merge_classical_source_conditions(
                collected,
                {
                    source: project_over_ranges(
                        _and_conditions(active_when, guard),
                        ranges,
                    )
                    for source, guard in additions.items()
                },
            )

        def branch_conditions(
            predicate: Boolean,
            active_when: Boolean,
            unprojectable_symbols: frozenset[sp.Symbol],
        ) -> tuple[Boolean, Boolean]:
            """Return guarded true/false activity for one nested branch.

            Args:
                predicate (Boolean): Resolved branch predicate.
                active_when (Boolean): Activity inherited from parent regions.
                unprojectable_symbols (frozenset[sp.Symbol]): Loop-local
                    symbols whose finite values are unavailable at this
                    boundary, such as dictionary keys.

            Returns:
                tuple[Boolean, Boolean]: True- and false-region guards. When
                an unprojectable local symbol participates, both retain the
                parent guard as a safe envelope.
            """
            if predicate.free_symbols & unprojectable_symbols:
                return active_when, active_when
            return (
                _and_conditions(active_when, predicate),
                _and_conditions(
                    active_when,
                    cast(Boolean, sp.Not(predicate)),
                ),
            )

        def visit_operations(
            operations: Iterable[Operation],
            scope: ExprResolver,
            active_when: Boolean,
            ranges: tuple[
                tuple[sp.Symbol, ResourceExpr, ResourceExpr, ResourceExpr],
                ...,
            ],
            unprojectable_symbols: frozenset[sp.Symbol],
        ) -> None:
            """Visit one structured region in program order.

            Args:
                operations (Iterable[Operation]): Region operations to visit.
                scope (ExprResolver): Resolver for the region.
                active_when (Boolean): Guard under which the region executes.
                ranges (tuple[tuple[sp.Symbol, ResourceExpr, ResourceExpr,
                    ResourceExpr], ...]): Enclosing finite range domains.
                unprojectable_symbols (frozenset[sp.Symbol]): Loop-local
                    symbols without an enumerable range.
            """
            for nested in operations:
                visit(
                    nested,
                    scope,
                    active_when,
                    ranges,
                    unprojectable_symbols,
                )

        def visit(
            current: Operation,
            scope: ExprResolver,
            active_when: Boolean,
            ranges: tuple[
                tuple[sp.Symbol, ResourceExpr, ResourceExpr, ResourceExpr],
                ...,
            ],
            unprojectable_symbols: frozenset[sp.Symbol],
        ) -> None:
            """Collect source reads from one atomic or structured operation.

            Args:
                current (Operation): Operation to inspect.
                scope (ExprResolver): Resolver for the operation's region.
                active_when (Boolean): Guard under which it executes.
                ranges (tuple[tuple[sp.Symbol, ResourceExpr, ResourceExpr,
                    ResourceExpr], ...]): Enclosing finite range domains.
                unprojectable_symbols (frozenset[sp.Symbol]): Loop-local
                    symbols without an enumerable range.
            """
            if active_when is sp.false:
                return
            if isinstance(current, IfOperation):
                add_values((current.condition,), scope, active_when, ranges)
                predicate = _boolean_condition(
                    scope.resolve_classical_fact(current.condition).value
                )
                true_active, false_active = branch_conditions(
                    predicate,
                    active_when,
                    unprojectable_symbols,
                )
                true_child, false_child = build_if_scopes(current, scope)
                visit_operations(
                    current.true_operations,
                    true_child,
                    true_active,
                    ranges,
                    unprojectable_symbols,
                )
                visit_operations(
                    current.false_operations,
                    false_child,
                    false_active,
                    ranges,
                    unprojectable_symbols,
                )
                return
            if isinstance(current, ForOperation) and len(current.operands) >= 3:
                add_values(current.operands[:3], scope, active_when, ranges)
                child, start, stop, step, loop_symbol = build_for_loop_scope(
                    current,
                    scope,
                )
                iterations = symbolic_iterations(start, stop, step)
                visit_operations(
                    current.operations,
                    child,
                    active_when,
                    (*ranges, (loop_symbol, start, step, iterations)),
                    unprojectable_symbols,
                )
                return
            if isinstance(current, ForItemsOperation) and current.operands:
                loop_active = _and_conditions(
                    active_when,
                    cast(Boolean, sp.Gt(resolve_for_items_cardinality(current), _ZERO)),
                )
                # Dictionary values are not structural bounds. Guard their
                # readiness by nonempty iteration before visiting the body.
                add_values(current.operands, scope, loop_active, ranges)
                child = build_for_items_scope(current, scope)
                local_values = (
                    *(current.key_var_values or ()),
                    *((current.value_var_value,) if current.value_var_value else ()),
                )
                local_symbols = {
                    symbol
                    for value in local_values
                    for symbol in child.resolve(value).free_symbols
                    if isinstance(symbol, sp.Symbol)
                }
                visit_operations(
                    current.operations,
                    child,
                    loop_active,
                    ranges,
                    unprojectable_symbols | frozenset(local_symbols),
                )
                return
            if isinstance(current, WhileOperation):
                if current.operands:
                    add_values(current.operands[:1], scope, active_when, ranges)
                operation_key = (scope.structural_scope, id(current))
                cached_name = self._while_trip_count_names.get(operation_key)
                trip_count_name = (
                    cached_name[1]
                    if cached_name is not None and cached_name[0] is current
                    else "|while|"
                )
                child, trip_count = build_while_scope(
                    current,
                    scope,
                    trip_count_name=trip_count_name,
                )
                visit_operations(
                    current.operations,
                    child,
                    _and_conditions(
                        active_when,
                        cast(Boolean, sp.Gt(trip_count, _ZERO)),
                    ),
                    ranges,
                    unprojectable_symbols,
                )
                return
            add_values(
                _operation_classical_dependency_inputs(current),
                scope,
                active_when,
                ranges,
            )

        visit(operation, resolver, sp.true, (), frozenset())
        return collected

    def _finite_runtime_alternatives(
        self,
        expression: sp.Expr,
    ) -> tuple[sp.Expr, ...] | None:
        """Expand an expression over registered finite runtime domains.

        Args:
            expression (sp.Expr): Runtime or compile-time scalar expression.

        Returns:
            tuple[sp.Expr, ...] | None: Structural alternatives, or ``None``
                when a participating runtime value is unbounded or the fixed
                expansion limit would be exceeded.
        """
        runtime_symbols = sorted(
            (
                symbol
                for symbol in expression.free_symbols
                if isinstance(symbol, sp.Symbol)
                and symbol in self._runtime_value_domains
            ),
            key=sp.default_sort_key,
        )
        if not runtime_symbols:
            return (expression,)
        domains = [self._runtime_value_domains[symbol] for symbol in runtime_symbols]
        if any(domain is None for domain in domains):
            return None
        finite_domains = cast(list[tuple[sp.Expr, ...]], domains)
        alternative_count = math.prod(len(domain) for domain in finite_domains)
        if alternative_count > _MAX_RUNTIME_DOMAIN_ALTERNATIVES:
            return None
        alternatives: list[sp.Expr] = []
        for values in itertools.product(*finite_domains):
            replacement: dict[sp.Symbol, sp.Expr] = dict(
                zip(runtime_symbols, values, strict=True)
            )
            candidate = cast(
                sp.Expr,
                expression.subs(list(replacement.items()), simultaneous=True),
            )
            if candidate not in alternatives:
                alternatives.append(candidate)
        return tuple(alternatives)

    def _register_runtime_value_domain(
        self,
        symbol: sp.Symbol,
        *sources: sp.Expr,
    ) -> tuple[sp.Expr, ...] | None:
        """Register the finite union of runtime branch-source alternatives.

        Args:
            symbol (sp.Symbol): Fresh internal runtime merge symbol.
            *sources (sp.Expr): Values selected by the runtime branches.

        Returns:
            tuple[sp.Expr, ...] | None: Registered finite alternatives, or
                ``None`` when the runtime domain cannot be bounded safely.
        """
        alternatives: list[sp.Expr] = []
        for source in sources:
            expanded = self._finite_runtime_alternatives(source)
            if expanded is None:
                self._runtime_value_domains[symbol] = None
                return None
            for candidate in expanded:
                if candidate not in alternatives:
                    alternatives.append(candidate)
            if len(alternatives) > _MAX_RUNTIME_DOMAIN_ALTERNATIVES:
                self._runtime_value_domains[symbol] = None
                return None
        domain = tuple(alternatives)
        self._runtime_value_domains[symbol] = domain
        return domain

    def estimate(self, block_or_ops: Block | Sequence[Operation]) -> ResourceEstimate:
        """Estimate resources for a block or operation sequence.

        Args:
            block_or_ops (Block | Sequence[Operation]): IR block or operations.

        Returns:
            ResourceEstimate: Estimated logical resources.

        Raises:
            QubitConsumedError: If compiler-equivalent inlining finds duplicate
                quantum operands at one call site.
            ValueError: If a selected inline body violates its invocation
                contract.
            NotImplementedError: If the IR contains legacy scalar Bit state
                that cannot flow correctly between loop iterations.
        """
        if isinstance(block_or_ops, Block):
            self._validate_legacy_scalar_bit_rebinds(block_or_ops)
            block_or_ops = self._resource_estimation_view(block_or_ops)
            # Only genuine classical parameters may decide a branch — a
            # measurement bit is never a param slot, so this prevents a runtime
            # ``if bit:`` from being specialized by a same-named value.
            if self.condition_values:
                slot_names = {slot.name for slot in block_or_ops.param_slots}
                self.condition_values = {
                    name: value
                    for name, value in self.condition_values.items()
                    if name in slot_names
                }
            resolver = ExprResolver(
                block=block_or_ops,
                context=_root_input_binding_context(block_or_ops, self.bindings),
            )
            input_allocations = _block_input_allocations(block_or_ops, resolver)
            body = self.eval_operations(
                block_or_ops.operations,
                resolver,
                initial_allocations=input_allocations,
            )
            body = _with_constraints(
                body,
                *_block_input_constraints(block_or_ops, resolver),
                *_block_output_constraints(
                    block_or_ops,
                    resolver,
                    proven_cache=self._array_constraint_proven,
                ),
            )
            input_qubits = sum(input_allocations.values(), _ZERO)
            width = dataclasses.replace(
                body.width,
                input_qubits=input_qubits,
                peak_qubits=input_qubits + body.width.peak_qubits,
            )
            return dataclasses.replace(
                body,
                width=width,
                trace=(
                    _wrap_trace(block_or_ops.name or "qkernel", body.trace)
                    if self.config.trace
                    else None
                ),
            )
        sequence_block = Block(
            name="operation_sequence",
            operations=list(block_or_ops),
            kind=BlockKind.HIERARCHICAL,
        )
        self._validate_legacy_scalar_bit_rebinds(sequence_block)
        sequence_view = self._resource_estimation_view(sequence_block)
        resolver = ExprResolver(
            context=_root_input_binding_context(block_or_ops, self.bindings)
        )
        return self.eval_operations(sequence_view.operations, resolver)

    def _resource_estimation_view(self, block: Block) -> Block:
        """Build the compiler-equivalent block used for interpretation.

        Ordinary direct qkernel calls are compiler-level organization rather
        than resource boundaries. Expanding them before scheduling preserves
        the readiness of individual measurement results and quantum wires, so
        extracting code into a helper does not change the estimate. Explicit
        quantum-width contracts remain as zero-work validation markers at the
        original call sites.

        Args:
            block (Block): Hierarchical root block to interpret.

        Returns:
            Block: Non-mutating view with eligible direct calls inlined.

        Raises:
            QubitConsumedError: If inlining detects duplicate quantum actuals.
            ValueError: If a selected body violates the invocation contract.
        """
        return InlinePass(
            body_selector=self._resource_estimation_inline_body,
            inline_prefix_factory=self._resource_estimation_inline_prefix,
            preserve_bound_quantum_identity=True,
            preserve_bound_array_identity=True,
        ).run(block)

    def _resource_estimation_inline_body(
        self,
        operation: InvokeOperation,
    ) -> Block | None:
        """Select the strategy-specific body for resource inlining.

        Args:
            operation (InvokeOperation): Direct inline-policy invocation being
                considered for expansion.

        Returns:
            Block | None: Strategy-selected body, or ``None`` when no inline
                body is available.

        Raises:
            ValueError: If the selected body violates the invocation contract.
        """
        return self._loop_validation_inline_body(operation)

    def _resource_estimation_inline_prefix(
        self,
        operation: InvokeOperation,
        array_state_bindings: tuple[
            tuple[ArrayValue, tuple[ArrayValue, ...]],
            ...,
        ],
    ) -> tuple[Operation, ...]:
        """Preserve resource-only state while dissolving a call boundary.

        Args:
            operation (InvokeOperation): Caller-substituted invocation whose
                selected body is about to be inlined.
            array_state_bindings (tuple[tuple[ArrayValue,
                tuple[ArrayValue, ...]], ...]): Caller arrays paired with
                cloned callee entries that need call-time snapshots.

        Returns:
            tuple[Operation, ...]: One zero-work boundary marker when the call
                declares a quantum width or carries array state, otherwise an
                empty tuple.
        """
        callable_attrs = {
            **(operation.definition.attrs if operation.definition is not None else {}),
            **operation.attrs,
        }
        legacy_width = callable_attrs.get("num_target_qubits")
        has_width_contract = callable_attrs.get("resource_contract") is not None or (
            type(legacy_width) is int and legacy_width > 0
        )
        if not has_width_contract and not array_state_bindings:
            return ()
        return (
            _ResourceInlineBoundaryOperation(
                constraint_operands=tuple(operation.target_qubits),
                callable_attrs=callable_attrs,
                source=operation.custom_name,
                array_state_bindings=array_state_bindings,
            ),
        )

    def resolve_finite_runtime_constraints(
        self,
        estimate: ResourceEstimate,
    ) -> ResourceEstimate:
        """Expand structural constraints over finite runtime alternatives.

        Runtime indices can be scheduled conservatively on their whole owner
        while still having a small, known value domain such as a measured Bit.
        Every alternative must satisfy the structural constraint, after which
        no internal outcome symbol needs to remain in the public payload.

        Args:
            estimate (ResourceEstimate): Specialized estimate to rewrite.

        Returns:
            ResourceEstimate: Estimate whose finitely bounded runtime
                constraints have been expanded and validated.

        Raises:
            ValueError: If any runtime alternative violates a constraint.
        """
        expanded_constraints: list[_ResourceConstraint] = []
        for constraint in estimate._constraints:
            expressions: list[sp.Basic] = [
                sp.sympify(constraint.expression),
                sp.sympify(constraint.active_when),
            ]
            if constraint.expected is not None:
                expressions.append(sp.sympify(constraint.expected))
            for loop_range in constraint.ranges:
                expressions.extend(
                    sp.sympify(value)
                    for value in (
                        loop_range.start,
                        loop_range.step,
                        loop_range.iterations,
                    )
                )
            symbols = set().union(
                *(expression.free_symbols for expression in expressions)
            )
            runtime_symbols = sorted(
                (
                    symbol
                    for symbol in symbols
                    if isinstance(symbol, sp.Symbol)
                    and symbol in self._runtime_value_domains
                ),
                key=sp.default_sort_key,
            )
            if not runtime_symbols:
                expanded_constraints.append(constraint)
                continue
            domains = [
                self._runtime_value_domains[symbol] for symbol in runtime_symbols
            ]
            if any(domain is None for domain in domains):
                expanded_constraints.append(constraint)
                continue
            finite_domains = cast(list[tuple[sp.Expr, ...]], domains)
            if (
                math.prod(len(domain) for domain in finite_domains)
                > _MAX_RUNTIME_DOMAIN_ALTERNATIVES
            ):
                expanded_constraints.append(constraint)
                continue
            for values in itertools.product(*finite_domains):
                replacements: dict[sp.Symbol, sp.Expr] = dict(
                    zip(runtime_symbols, values, strict=True)
                )
                expanded_constraints.append(
                    constraint.mapped(
                        lambda expression, replacements=replacements: cast(
                            sp.Expr,
                            sp.sympify(expression).subs(
                                list(replacements.items()),
                                simultaneous=True,
                            ),
                        )
                    )
                )
        return dataclasses.replace(
            estimate,
            _constraints=tuple(expanded_constraints),
        )

    def validate_no_internal_resource_symbols(
        self,
        estimate: ResourceEstimate,
    ) -> None:
        """Reject internal runtime values that escaped into public algebra.

        Internal observation outcomes and unresolved loop carries are semantic
        execution state, not user inputs. Hiding them from ``parameters``
        would leave an unsubstitutable expression, while publishing them would
        let a user choose an outcome and silently underestimate a runtime
        worst case. The estimator therefore fails closed when such a value
        affects any serialized metric, constraint, metadata guard, or trace.

        Args:
            estimate (ResourceEstimate): Final specialized estimate to check.

        Raises:
            NotImplementedError: If an internal runtime or unresolved carry
                symbol remains in the public estimate payload.
        """
        internal_symbols = (
            self._runtime_observation_symbols | self._unresolved_resource_symbols
        )
        escaped = (
            _free_symbols(estimate) | _trace_guard_free_symbols(estimate.trace)
        ) & internal_symbols
        if not escaped:
            return
        names = ", ".join(sorted(_symbol_display_name(symbol) for symbol in escaped))
        raise NotImplementedError(
            "Resource estimation cannot bound a runtime-derived or unresolved "
            f"loop-carried value used by resource-sensitive structure ({names}). "
            "Provide concrete structural inputs or rewrite the continuation so "
            "its resource cost does not depend on that runtime value."
        )

    def _validate_legacy_scalar_bit_rebinds(
        self,
        block_or_operations: Block | Sequence[Operation],
        *,
        output_values: Sequence[ValueLike] | None = None,
        local_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        """Reuse compiler loop-state validation before interpreting a body.

        Args:
            block_or_operations (Block | Sequence[Operation]): Semantic body
                to validate. A Block is inlined into a temporary affine view
                so only callee inputs that the selected body actually reads
                count as loop back-edge reads.
            output_values (Sequence[ValueLike] | None): Body outputs used for
                post-loop liveness when a bare operation sequence is passed.
                A Block always supplies its own outputs. Defaults to ``None``.
            local_bindings (Mapping[str, Any] | None): Resolved callee-formal
                values added to root resource inputs. Defaults to ``None``.

        Raises:
            NotImplementedError: If a legacy scalar Bit rebind cannot be
                represented across the loop boundary.
        """
        operations = (
            block_or_operations.operations
            if isinstance(block_or_operations, Block)
            else block_or_operations
        )
        validation_bindings = {
            name: self._compiler_validation_binding(value)
            for name, value in self.bindings.items()
        }
        validation_bindings.update(
            {
                name: self._compiler_validation_binding(value)
                for name, value in self.condition_values.items()
            }
        )
        if local_bindings is not None:
            validation_bindings.update(
                {
                    name: self._compiler_validation_binding(value)
                    for name, value in local_bindings.items()
                }
            )
        validation_outputs = output_values
        if isinstance(block_or_operations, Block):
            validation_block = self._loop_validation_view(
                block_or_operations,
                validation_bindings,
            )
            operations = validation_block.operations
            validation_outputs = validation_block.output_values
        if not has_legacy_scalar_bit_rebinds(operations):
            return
        try:
            reject_loop_carried_classical_rebinds(
                list(operations),
                bindings=validation_bindings,
                output_values=(
                    list(validation_outputs) if validation_outputs is not None else None
                ),
            )
        except ValidationError as exc:
            raise NotImplementedError(str(exc)) from exc

    def _loop_validation_view(
        self,
        block: Block,
        bindings: dict[str, Any],
    ) -> Block:
        """Build the compiler-equivalent view used by loop-state validation.

        Resource interpretation intentionally preserves callable boundaries,
        but the compiler validates loop-carried state after inline-policy calls
        have been expanded. This temporary view follows the same selected-body
        and compile-time specialization rules without changing the Block that
        resource evaluation traverses. Repeating inline then specialization
        also handles concrete self-recursion up to the compiler's supported
        unroll depth.

        Args:
            block (Block): Hierarchical semantic body to validate.
            bindings (dict[str, Any]): Concrete compiler-domain bindings used
                for branch specialization.

        Returns:
            Block: Non-mutating validation view. A recursive call that does not
                converge within the supported depth remains boxed and is
                handled conservatively by the shared validator.

        Raises:
            ValueError: If a strategy-selected body violates the invocation
                input or output contract.
            QubitConsumedError: If inlining detects duplicate quantum actuals.
            ValidationError: If compile-time specialization encounters invalid
                IR.
        """
        inline = InlinePass(body_selector=self._loop_validation_inline_body)
        view = block
        for _ in range(_MAX_LOOP_VALIDATION_INLINE_DEPTH):
            view = inline.run(view)
            if view.kind is not BlockKind.HIERARCHICAL:
                return view
            view = lower_compile_time_ifs_preserving_loop_conditions(
                view,
                bindings,
            )
            if count_inline_invokes(view.operations) == 0:
                # Lowering the final base-case branch can remove the last
                # invocation while the copied block still carries its stale
                # HIERARCHICAL kind. Refresh it exactly as recursion unrolling
                # does so the supported depth boundary stays identical.
                return inline.run(view)
        name = block.name or "qkernel"
        raise ValueError(
            f"Recursive resource validation for '{name}' did not reach a "
            "base case within the supported inline depth. Supply a concrete "
            "recursion-driving value in inputs, or replace the "
            "recursion with a bounded loop."
        )

    def _loop_validation_inline_body(self, operation: InvokeOperation) -> Block | None:
        """Select the body that resource interpretation would evaluate.

        Args:
            operation (InvokeOperation): Inline-policy invocation being copied
                into the validation view.

        Returns:
            Block | None: Strategy-selected body, or None when unavailable.

        Raises:
            ValueError: If the selected body violates the invocation contract.
        """
        return operation.select_body(strategy=self._strategy_for(operation)).body

    @staticmethod
    def _compiler_validation_binding(value: Any) -> Any:
        """Convert a concrete SymPy scalar to the compiler binding domain.

        Args:
            value (Any): Resource-estimator input or resolved callee value.

        Returns:
            Any: Equivalent Python scalar when concrete, otherwise the
                original binding object.
        """
        if not isinstance(value, sp.Expr) or not value.is_number:
            return value
        if _is_concrete_integer(value):
            return int(value)
        if value.is_real is True:
            return float(value)
        return value

    def _reserve_while_trip_count_names(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
    ) -> None:
        """Reserve deterministic names for every structurally reachable while.

        The interpreter may evaluate a loop body first as a symbolic probe and
        again with resolved carry values, or skip that body after input
        specialization. Preorder reservation makes both paths use the same
        public trip-count symbol names.

        Args:
            operations (Sequence[Operation]): Root operations whose nested
                control-flow and callable bodies should be indexed.
            resolver (ExprResolver): Resolver carrying the current structural
                call-site path.
        """

        def visit(
            body_operations: Sequence[Operation],
            structural_scope: tuple[tuple[int, int], ...],
            active_call_bodies: frozenset[int],
        ) -> None:
            """Visit one operation sequence in deterministic lexical order.

            Args:
                body_operations (Sequence[Operation]): Operations to scan.
                structural_scope (tuple[tuple[int, int], ...]): Callable path
                    containing this operation sequence.
                active_call_bodies (frozenset[int]): Callable block identities
                    already entered on this lexical path.
            """
            for operation in body_operations:
                operation_id = id(operation)
                operation_key = (structural_scope, operation_id)
                cached = self._while_name_scan_operations.get(operation_key)
                if cached is operation:
                    continue
                self._while_name_scan_operations[operation_key] = operation
                if isinstance(operation, WhileOperation):
                    ordinal = len(self._while_trip_count_names) + 1
                    name = "|while|" if ordinal == 1 else f"|while[{ordinal}]|"
                    self._while_trip_count_names[operation_key] = (operation, name)

                if isinstance(operation, InvokeOperation):
                    body, _realized_transform = operation.body_for_transform(
                        strategy=self._strategy_for(operation)
                    )
                    if isinstance(body, Block) and id(body) not in active_call_bodies:
                        visit(
                            body.operations,
                            (*structural_scope, (operation_id, id(body))),
                            active_call_bodies | {id(body)},
                        )
                elif isinstance(operation, ControlledUOperation):
                    if (
                        isinstance(operation.block, Block)
                        and id(operation.block) not in active_call_bodies
                    ):
                        visit(
                            operation.block.operations,
                            (
                                *structural_scope,
                                (operation_id, id(operation.block)),
                            ),
                            active_call_bodies | {id(operation.block)},
                        )
                elif isinstance(operation, SelectOperation):
                    for case in operation.case_blocks:
                        if id(case) in active_call_bodies:
                            continue
                        visit(
                            case.operations,
                            (*structural_scope, (operation_id, id(case))),
                            active_call_bodies | {id(case)},
                        )
                elif isinstance(operation, InverseBlockOperation):
                    if (
                        isinstance(operation.implementation_block, Block)
                        and id(operation.implementation_block) not in active_call_bodies
                    ):
                        visit(
                            operation.implementation_block.operations,
                            (
                                *structural_scope,
                                (operation_id, id(operation.implementation_block)),
                            ),
                            active_call_bodies | {id(operation.implementation_block)},
                        )
                elif isinstance(operation, HasNestedOps):
                    for nested in operation.nested_op_lists():
                        visit(nested, structural_scope, active_call_bodies)

        visit(operations, resolver.structural_scope, frozenset())

    def eval_block(self, block: Block, resolver: ExprResolver) -> ResourceEstimate:
        """Evaluate a block body.

        Args:
            block (Block): Block to evaluate.
            resolver (ExprResolver): Resolver scoped to ``block``.

        Returns:
            ResourceEstimate: Estimated resources for the body.
        """
        estimate = self.eval_operations(
            block.operations,
            resolver,
            initial_allocations=_block_input_allocations(block, resolver),
        )
        return _with_constraints(
            estimate,
            *_block_input_constraints(block, resolver),
        )

    @staticmethod
    def _bind_resource_inline_array_states(
        operation: _ResourceInlineBoundaryOperation,
        resolver: ExprResolver,
    ) -> None:
        """Snapshot caller arrays into one inlined callable's entry values.

        Every caller state is captured before any callee entry is rebound.
        This preserves simultaneous call-argument semantics even when future
        IR permits two classical formals to alias one array lineage.

        Args:
            operation (_ResourceInlineBoundaryOperation): Zero-work call
                boundary carrying caller-to-callee array bindings.
            resolver (ExprResolver): Resolver at the original call position.
        """
        snapshots = tuple(
            (entries, resolver.snapshot_array_state(actual))
            for actual, entries in operation.array_state_bindings
        )
        for entries, state in snapshots:
            for entry in entries:
                resolver.bind_array_state(entry, state)

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
                    self._allocation_owners_by_uuid,
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
                sp.Ne(
                    _CanonicalPhaseClass(phase),
                    _PHASE_CLASS_CODES[None],
                )
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
                condition.free_symbols & self._unresolved_resource_symbols
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
            active_states = self._active_batch_profile_states.setdefault(
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
                    self._active_batch_profile_states.pop(body_identity, None)
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
                    self._allocation_owners_by_uuid,
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
                        self._allocation_owners_by_uuid,
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
        if self.condition_values and condition.free_symbols:
            substitutions: dict[Any, Any] = {
                symbol: self.condition_values[_symbol_display_name(symbol)]
                for symbol in condition.free_symbols
                if not isinstance(symbol, sp.Dummy)
                and _symbol_display_name(symbol) in self.condition_values
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
        toffoli = _estimate_named_gate_in_basis(
            "toffoli",
            _ZERO,
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=self.config.precision,
        )
        ladder = toffoli.repeat(recipe.total_toffolis)
        compute_depth = recipe.compute_toffolis * toffoli.depth.depth
        estimate = ladder.seq(body)
        trace_children = tuple(
            node
            for node in (toffoli.trace, body.trace, toffoli.trace)
            if node is not None
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
        )._with_metadata(quality=EstimateQuality.CONSERVATIVE)

    def _block_runtime_observation_summary(
        self,
        block: Block,
    ) -> tuple[frozenset[int], bool]:
        """Summarize selected runtime observations as a least fixed point.

        Measurement provenance is cached in the IR, while expectation values
        intentionally are not a ``KernelEffect``. Resource scheduling needs
        both, so this estimator-local summary recursively follows the same
        selected Invoke bodies used for resource evaluation. Recursive
        callable graphs start at the empty summary and iterate until no output
        provenance or observation flag grows, so a cycle never caches a
        partial result based on access order.

        Args:
            block (Block): Selected callable body to inspect.

        Returns:
            tuple[frozenset[int], bool]: Body output indices derived from a
            runtime observation and whether any observation occurs in the body.
        """
        cached = self._runtime_observation_cache.get(id(block))
        if cached is not None and cached[0] is block:
            return cached[1], cached[2]

        reachable = self._selected_runtime_observation_blocks(block)
        summaries: dict[int, tuple[frozenset[int], bool]] = {
            id(candidate): (frozenset(), False) for candidate in reachable
        }
        while True:
            changed = False
            updated: dict[int, tuple[frozenset[int], bool]] = {}
            for candidate in reachable:
                output_indices, has_observation = self._runtime_observation_equation(
                    candidate, summaries
                )
                previous_indices, previous_observation = summaries[id(candidate)]
                summary = (
                    previous_indices | output_indices,
                    previous_observation or has_observation,
                )
                updated[id(candidate)] = summary
                changed = changed or summary != summaries[id(candidate)]
            summaries = updated
            if not changed:
                break

        for candidate in reachable:
            output_indices, has_observation = summaries[id(candidate)]
            self._runtime_observation_cache[id(candidate)] = (
                candidate,
                output_indices,
                has_observation,
            )
        result = summaries[id(block)]
        return result

    def _selected_runtime_observation_blocks(
        self,
        root: Block,
    ) -> tuple[Block, ...]:
        """Collect blocks reachable through selected Invoke implementations.

        Args:
            root (Block): Selected body at the root of the callable graph.

        Returns:
            tuple[Block, ...]: Strongly referenced blocks in deterministic
                preorder.
        """
        blocks: dict[int, Block] = {}
        pending = [root]
        while pending:
            block = pending.pop()
            identity = id(block)
            if identity in blocks:
                continue
            blocks[identity] = block
            children: list[Block] = []
            for operation in walk_operations(block.operations):
                if not isinstance(operation, InvokeOperation):
                    continue
                selection = operation.select_body(
                    strategy=self._strategy_for(operation)
                )
                if isinstance(selection.body, Block):
                    children.append(selection.body)
            pending.extend(reversed(children))
        return tuple(blocks.values())

    def _runtime_observation_equation(
        self,
        block: Block,
        summaries: Mapping[int, tuple[frozenset[int], bool]],
    ) -> tuple[frozenset[int], bool]:
        """Apply one selected-observation equation to a block.

        Args:
            block (Block): Body whose local equation should be evaluated.
            summaries (Mapping[int, tuple[frozenset[int], bool]]): Previous
                fixed-point approximation for every reachable selected body.

        Returns:
            tuple[frozenset[int], bool]: Output provenance and observation flag
                derived in this iteration.
        """
        roots = _find_runtime_observation_results(block.operations)
        has_observation = bool(roots)
        for operation in walk_operations(block.operations):
            if not isinstance(operation, InvokeOperation):
                continue
            selection = operation.select_body(strategy=self._strategy_for(operation))
            if not isinstance(selection.body, Block):
                continue
            body_indices, nested_has_observation = summaries[id(selection.body)]
            mapped_indices = selection.map_result_indices(
                body_indices,
                operation.results,
            )
            roots.update(
                operation.results[index].uuid
                for index in mapped_indices
                if index < len(operation.results)
            )
            has_observation = has_observation or nested_has_observation

        derived = find_measurement_derived_values(
            build_dependency_graph(block.operations),
            roots,
        )
        derived.update(roots)
        return (
            frozenset(
                index
                for index, output in enumerate(block.output_values)
                if output.uuid in derived
            ),
            has_observation,
        )

    def _invoke_runtime_observation_summary(
        self,
        operation: InvokeOperation,
    ) -> tuple[frozenset[int], bool]:
        """Map a selected body's runtime observations to Invoke results.

        Args:
            operation (InvokeOperation): Callable invocation to inspect.

        Returns:
            tuple[frozenset[int], bool]: Caller result indices derived from an
            observation and whether the selected body contains an observation.
        """
        strategy = self._strategy_for(operation)
        selection = operation.select_body(strategy=strategy)
        body = selection.body
        if not isinstance(body, Block):
            indices = operation.measurement_result_indices_for(strategy=strategy)
            return indices, bool(indices)
        body_indices, has_observation = self._block_runtime_observation_summary(
            body,
        )
        return (
            selection.map_result_indices(body_indices, operation.results),
            has_observation,
        )

    def eval_operations(
        self,
        operations: list[Operation],
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
        initial_allocations: Mapping[str, ResourceExpr] | None = None,
        allow_control_batching: bool = True,
    ) -> ResourceEstimate:
        """Evaluate a list of operations sequentially.

        Args:
            operations (list[Operation]): Operations to evaluate.
            resolver (ExprResolver): Value resolver for this scope.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.
            initial_allocations (Mapping[str, ResourceExpr] | None): Live
                quantum inputs keyed by logical wire ID. Defaults to ``None``.
            allow_control_batching (bool): Whether this body boundary may
                choose a shared control ladder. Recursive call boundaries keep
                their own independent choice. Defaults to ``True``.

        Returns:
            ResourceEstimate: Sequential composition of operation resources.
        """
        self._reserve_while_trip_count_names(operations, resolver)
        control_count = self._apply_condition_values(
            _expr(controls),
            record_usage=False,
        )
        batch_condition = (
            self._controlled_body_batch_condition(
                operations,
                resolver,
                control_count,
            )
            if allow_control_batching
            else sp.false
        )
        if batch_condition is not sp.false:
            body = self.eval_operations(
                operations,
                resolver,
                controls=_ONE,
                initial_allocations=initial_allocations,
            )
            active = _resource_activity_condition(_estimate_activity(body))
            shared = self._with_shared_control_ladder(
                body,
                control_count,
            ).conditional(body, active)
            if batch_condition is sp.true:
                return shared
            direct = self.eval_operations(
                operations,
                resolver,
                controls=control_count,
                initial_allocations=initial_allocations,
                allow_control_batching=False,
            )
            batch_dependency_keys = _merge_dependency_keys(shared, direct)
            shared = dataclasses.replace(
                shared,
                _dependency_keys=batch_dependency_keys,
            )
            direct = dataclasses.replace(
                direct,
                _dependency_keys=batch_dependency_keys,
            )
            return shared.conditional(direct, batch_condition)

        previous_taint = self._measurement_taint_conditions
        self._measurement_taint_conditions = dict(previous_taint)
        try:
            scheduled: list[tuple[Operation, ResourceEstimate]] = []
            scheduled_classical_sources: list[
                tuple[dict[str, Boolean], dict[str, Boolean]]
            ] = []
            known_classical_sources: set[str] = set()
            seen_array_constraints: set[_ResourceConstraint] = set()
            for operation in operations:
                structural_input_sources = _resolved_classical_source_conditions(
                    operation.all_input_values(),
                    resolver,
                )
                known_classical_sources.update(structural_input_sources)
                classical_input_sources = self._scheduling_classical_input_sources(
                    operation,
                    resolver,
                )
                operation_estimate = self.eval_operation(
                    operation,
                    resolver,
                    controls=control_count,
                )
                if isinstance(operation, StoreArrayElementOperation):
                    resolver.record_array_store(operation)
                self._measurement_taint_conditions = (
                    _propagate_operation_measurement_taint(
                        operation,
                        operation_estimate,
                        self._measurement_taint_conditions,
                    )
                )
                self._record_runtime_value_symbols(operation, resolver)
                self._publish_operation_classical_facts(
                    operation,
                    resolver,
                    classical_input_sources,
                )
                classical_output_sources = _resolved_classical_source_conditions(
                    _operation_classical_dependency_outputs(operation),
                    resolver,
                )
                _merge_classical_source_conditions(
                    classical_output_sources,
                    _direct_observation_source_conditions(operation, resolver),
                )
                published_classical_sources = {
                    source: guard
                    for source, guard in classical_output_sources.items()
                    if source not in known_classical_sources
                }
                known_classical_sources.update(classical_output_sources)
                known_classical_sources.update(published_classical_sources)
                if _estimate_has_nonzero_depth(
                    operation_estimate
                ) and _operation_has_unresolved_quantum_index(
                    operation,
                    resolver,
                    scalar_values=self.condition_values,
                    used_names=self.branch_condition_names,
                ):
                    assumption = ResourceAssumption(
                        "unresolved quantum index may alias any scalar of its "
                        "allocation",
                        source=type(operation).__name__,
                    )
                    operation_estimate = operation_estimate._with_metadata(
                        assumptions=(assumption,),
                        quality=EstimateQuality.CONSERVATIVE,
                    )
                if not self.config.trace:
                    # Every operation estimate is freshly produced for this
                    # traversal. Dropping its optional explanation before the
                    # sequential fold avoids constructing a deep tree that the
                    # public estimator would discard at the end anyway.
                    operation_estimate.trace = None
                discovered_constraints = (
                    ()
                    if isinstance(operation, HasNestedOps)
                    else _operation_array_constraints(
                        operation,
                        resolver,
                        active_when=self._constraint_scope_condition,
                        proven_cache=self._array_constraint_proven,
                    )
                )
                array_constraints = tuple(
                    constraint
                    for constraint in discovered_constraints
                    if constraint not in seen_array_constraints
                    and constraint not in operation_estimate._constraints
                )
                seen_array_constraints.update(array_constraints)
                operation_estimate = _with_constraints(
                    operation_estimate,
                    *array_constraints,
                )
                operation_estimate = dataclasses.replace(
                    operation_estimate,
                    basis=self.config.basis,
                    control_decomposition=self.config.control_decomposition,
                    precision=(
                        self.config.precision
                        if self.config.basis is GateBasis.CLIFFORD_T
                        else None
                    ),
                )
                scheduled.append((operation, operation_estimate))
                scheduled_classical_sources.append(
                    (classical_input_sources, published_classical_sources)
                )
            dependency_keys: set[WireKey] = set()
            wire_footprints: list[_WireFootprint | None] = []
            classical_dependency_conditions: list[Boolean] = []
            classical_read_conditions: list[dict[WireKey, Boolean]] = []
            scheduled_with_dependencies: list[tuple[Operation, ResourceEstimate]] = []
            for (
                operation,
                operation_estimate,
            ), (
                classical_input_sources,
                classical_output_sources,
            ) in zip(
                scheduled,
                scheduled_classical_sources,
                strict=True,
            ):
                classical_reads, classical_writes, classical_active = (
                    _classical_dependency_footprint(
                        classical_input_sources,
                        classical_output_sources,
                    )
                )
                classical_dependency_conditions.append(classical_active)
                classical_read_conditions.append(
                    {
                        _classical_dependency_key(source): condition
                        for source, raw_condition in classical_input_sources.items()
                        if (condition := _boolean_condition(raw_condition))
                        is not sp.false
                    }
                )
                if isinstance(operation, (IfOperation, WhileOperation)):
                    condition_value = (
                        operation.condition
                        if isinstance(operation, IfOperation)
                        else operation.operands[0]
                    )
                    condition_runtime = _classical_fact_runtime_condition(
                        resolver.resolve_classical_fact(condition_value),
                    )
                    if condition_runtime is not sp.false and not classical_reads:
                        raise AssertionError(
                            "A runtime control-flow operation must read its "
                            "observation dependency token."
                        )
                if (
                    not _estimate_has_nonzero_depth(operation_estimate)
                    and not classical_reads
                    and not classical_writes
                ):
                    wire_footprints.append(None)
                    scheduled_with_dependencies.append(
                        (
                            operation,
                            dataclasses.replace(
                                operation_estimate,
                                _dependency_keys=frozenset(),
                                _dependency_completion={},
                                _dependency_completion_uniform=True,
                            ),
                        )
                    )
                    continue
                if operation_estimate._dependency_keys is not None:
                    footprint_keys = frozenset(
                        _expand_dependency_owner_aliases(
                            set(operation_estimate._dependency_keys),
                            self._dependency_owner_aliases,
                        )
                    )
                    reads = set(footprint_keys)
                    writes = set(footprint_keys)
                    completion_uniform = (
                        operation_estimate._dependency_completion_uniform
                    )
                else:
                    reads, writes = _quantum_wire_keys(
                        operation,
                        resolver,
                        scalar_values=self.condition_values,
                        used_names=self.branch_condition_names,
                        owner_aliases=self._dependency_owner_aliases,
                        allocation_owners_by_uuid=self._allocation_owners_by_uuid,
                    )
                    footprint_keys = frozenset(reads | writes)
                    completion_uniform = _operation_has_uniform_intrinsic_completion(
                        operation,
                        operation_estimate,
                        footprint_keys,
                        surrounding_controls=_expr(control_count),
                    )
                wire_footprints.append(
                    (
                        frozenset((*reads, *classical_reads)),
                        frozenset((*writes, *classical_writes)),
                    )
                )
                dependency_keys.update(footprint_keys)
                operation_completion = _normalized_dependency_completion(
                    operation_estimate
                )
                if operation_completion is not None and self._dependency_owner_aliases:
                    expanded_completion: dict[WireKey, ResourceExpr] = {}
                    for key, completion in operation_completion.items():
                        for expanded_key in _expand_dependency_owner_aliases(
                            {key},
                            self._dependency_owner_aliases,
                        ):
                            expanded_completion[expanded_key] = _resource_max(
                                expanded_completion.get(expanded_key, _ZERO),
                                completion,
                            )
                    operation_completion = expanded_completion
                if operation_completion is None:
                    operation_completion = {
                        key: operation_estimate.depth.depth for key in footprint_keys
                    }
                operation_estimate = dataclasses.replace(
                    operation_estimate,
                    _dependency_keys=footprint_keys,
                    _dependency_completion=operation_completion,
                    _dependency_completion_uniform=completion_uniform,
                )
                scheduled_with_dependencies.append((operation, operation_estimate))
            scheduled = scheduled_with_dependencies
            estimate = ResourceEstimate.seq_all(
                operation_estimate for _, operation_estimate in scheduled
            )
            depth_footprints = wire_footprints
            if _expr(controls) != _ZERO:
                control_carrier = ("$resource_control_carrier", None)
                depth_footprints = [
                    (
                        (
                            frozenset((*footprint[0], control_carrier)),
                            frozenset((*footprint[1], control_carrier)),
                        )
                        if footprint is not None
                        else None
                    )
                    for footprint in wire_footprints
                ]
            depth_activity_conditions = _scheduled_depth_activity_conditions(scheduled)
            depth_activity_conditions = tuple(
                _boolean_condition(sp.Or(depth_active, classical_active))
                for depth_active, classical_active in zip(
                    depth_activity_conditions,
                    classical_dependency_conditions,
                    strict=True,
                )
            )
            (
                scheduled_depth,
                scheduled_completion,
                possible_alias_active,
                completion_is_uniform,
            ) = _dependency_depth(
                scheduled,
                depth_footprints,
                activity_conditions=depth_activity_conditions,
                read_conditions=classical_read_conditions,
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            )
            aggregate_completion_active = _aggregate_completion_overlap_condition(
                scheduled,
                depth_footprints,
                activity_conditions=depth_activity_conditions,
            )
            liveness = _liveness_width(
                scheduled,
                initial_allocations or {},
                resolver,
                allocation_owners_by_uuid=self._allocation_owners_by_uuid,
            )
            result = dataclasses.replace(
                estimate,
                depth=scheduled_depth,
                width=liveness.width,
                _allocation_sites=_without_input_allocation_sites(
                    estimate._allocation_sites,
                    operations,
                    initial_allocations or {},
                ),
                _dependency_keys=frozenset(dependency_keys),
                _dependency_reads=frozenset(
                    key
                    for (_operation, operation_estimate), footprint in zip(
                        scheduled,
                        wire_footprints,
                        strict=True,
                    )
                    if footprint is not None
                    and _estimate_has_nonzero_depth(operation_estimate)
                    for key in footprint[0]
                ),
                _dependency_writes=frozenset(
                    key
                    for (_operation, operation_estimate), footprint in zip(
                        scheduled,
                        wire_footprints,
                        strict=True,
                    )
                    if footprint is not None
                    and _estimate_has_nonzero_depth(operation_estimate)
                    for key in footprint[1]
                ),
                _dependency_completion={
                    key: completion
                    for key, completion in scheduled_completion.items()
                    if key in dependency_keys
                },
                _dependency_completion_uniform=completion_is_uniform,
                _output_sizes=liveness.final_live_by_owner,
                _input_sizes=dict(initial_allocations or {}),
                _has_output_summary=True,
                _measurement_taint_conditions=dict(self._measurement_taint_conditions),
            )
            if possible_alias_active is not sp.false:
                assumption = ResourceAssumption(
                    "symbolic quantum indices may alias and are scheduled "
                    "conservatively",
                    source="dependency scheduler",
                )
                result = result._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=possible_alias_active,
                )
            if aggregate_completion_active is not sp.false:
                assumption = ResourceAssumption(
                    "aggregate latency may over-serialize a later wire dependency "
                    "and therefore overestimate depth",
                    source="dependency scheduler",
                )
                result = result._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=aggregate_completion_active,
                )
            return result
        finally:
            self._measurement_taint_conditions = previous_taint

    def eval_operation(
        self,
        operation: Operation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate one operation.

        Args:
            operation (Operation): Operation to evaluate.
            resolver (ExprResolver): Resolver for operation operands.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Operation resource estimate.

        Raises:
            ValueError: If an estimator-only call-site width contract is
                malformed or violated.
            NotImplementedError: If the operation kind is not supported by
                resource estimation.
        """
        match operation:
            case _ResourceInlineBoundaryOperation():
                self._bind_resource_inline_array_states(operation, resolver)
                return _with_constraints(
                    ResourceEstimate.zero(),
                    *_quantum_operand_width_constraints(
                        operation.callable_attrs,
                        operation.constraint_operands,
                        resolver,
                        source=operation.source,
                    ),
                )
            case GateOperation():
                return self.eval_gate(operation, controls=controls)
            case QInitOperation():
                return self.eval_qinit(operation, resolver)
            case (
                MeasureOperation() | MeasureVectorOperation() | MeasureQFixedOperation()
            ):
                _require_uncontrolled_operation(operation, controls)
                return self.eval_measure(operation, resolver)
            case ExpvalOp():
                _require_uncontrolled_operation(operation, controls)
                return self.eval_expval(operation, resolver)
            case ProjectOperation():
                _require_uncontrolled_operation(operation, controls)
                return self.eval_project(operation)
            case ResetOperation():
                _require_uncontrolled_operation(operation, controls)
                return self.eval_reset(operation)
            case ForOperation():
                return self.eval_for(operation, resolver, controls=controls)
            case WhileOperation():
                return self.eval_while(operation, resolver, controls=controls)
            case IfOperation():
                return self.eval_if(operation, resolver, controls=controls)
            case ForItemsOperation():
                return self.eval_for_items(operation, resolver, controls=controls)
            case InvokeOperation():
                return self.eval_invoke(operation, resolver, controls=controls)
            case SelectOperation():
                return self.eval_select(operation, resolver, controls=controls)
            case GlobalPhaseOperation():
                return self.eval_global_phase(
                    operation,
                    resolver,
                    controls=controls,
                )
            case ControlledUOperation():
                return self.eval_controlled_u(operation, resolver, controls=controls)
            case InverseBlockOperation():
                return self.eval_inverse_block(operation, resolver, controls=controls)
            case PauliEvolveOp():
                return self.eval_pauli_evolve(
                    operation,
                    resolver,
                    controls=controls,
                )
            case UnaryMathOp():
                if classify_control_work(operation) is ControlWorkKind.UNSUPPORTED:
                    _require_uncontrolled_operation(operation, controls)
                return self.eval_unary_math(operation, resolver)
            case CastOperation() | ReturnQuantumArrayElementOperation():
                return ResourceEstimate.zero()
            case StoreArrayElementOperation():
                # The enclosing sequential interpreter publishes the updated
                # array state after this zero-resource semantic operation.
                return ResourceEstimate.zero()
            case SliceArrayOperation() | ReleaseSliceViewOperation():
                # Estimation runs before the transpiler strips slice-lifetime
                # markers. They are valid zero-work structure here even though
                # either marker reaching the later emit walker is an error.
                return ResourceEstimate.zero()
            case HasNestedOps():
                raise NotImplementedError(
                    "Resource estimation does not support nested operation "
                    f"{type(operation).__name__}."
                )
            case _:
                if classify_control_work(operation) is ControlWorkKind.UNSUPPORTED:
                    _require_uncontrolled_operation(operation, controls)
                if operation.operation_kind is OperationKind.CLASSICAL:
                    return ResourceEstimate.zero()
                raise NotImplementedError(
                    "Resource estimation does not support non-classical "
                    f"operation {type(operation).__name__} "
                    f"({operation.operation_kind.value}); refusing to report "
                    "it as an exact zero-cost operation."
                )

    def eval_unary_math(
        self,
        operation: UnaryMathOp,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate structural requirements of one unary math expression.

        Classical arithmetic has no quantum gate cost, but unary math domains
        must remain valid after resource-input substitution. Retaining those
        domains here produces source-level diagnostics before downstream width
        expressions turn into infinities or invalid unsigned sizes.

        Args:
            operation (UnaryMathOp): Unary mathematical IR operation.
            resolver (ExprResolver): Resolver for the mathematical operand.

        Returns:
            ResourceEstimate: Zero quantum cost with any retained structural
                input-domain requirements.

        Raises:
            ValueError: If the operation types are malformed or a concrete
                input violates the unary operation's numeric domain.
            NotImplementedError: If the unary operation kind is unknown.
        """
        kind = operation.kind
        if kind is None:
            raise ValueError("Cannot estimate unary math operation without a kind.")
        expected_output_type = FloatType if kind is UnaryMathOpKind.LOG2 else UIntType
        if not isinstance(
            operation.input.type, (UIntType, FloatType)
        ) or not isinstance(
            operation.output.type,
            expected_output_type,
        ):
            raise ValueError(
                f"Cannot estimate malformed {kind.name} operation: "
                "expected UInt or Float input and "
                f"{expected_output_type.__name__} output."
            )
        match kind:
            case UnaryMathOpKind.LOG2:
                is_uint = isinstance(operation.input.type, UIntType)
                constraint = _ResourceConstraint(
                    expression=resolver.resolve(operation.input),
                    minimum=1 if is_uint else 0,
                    label="log2 input",
                    integer=is_uint,
                    minimum_inclusive=is_uint,
                    finite=not is_uint,
                )
                return _with_constraints(
                    ResourceEstimate.zero("log2"),
                    constraint,
                )
            case UnaryMathOpKind.CEIL:
                if isinstance(operation.input.type, UIntType):
                    return ResourceEstimate.zero("ceil")
                constraint = _ResourceConstraint(
                    expression=resolver.resolve(operation.input),
                    minimum=-1,
                    label="ceil input",
                    integer=False,
                    minimum_inclusive=False,
                    finite=True,
                )
                return _with_constraints(
                    ResourceEstimate.zero("ceil"),
                    constraint,
                )
            case _:
                raise NotImplementedError(
                    "Resource estimation does not support unary math kind "
                    f"{operation.kind!r}."
                )

    def _eval_call_body(
        self,
        block: Block,
        child: ExprResolver,
        actual_operands: Sequence[ValueBase],
        *,
        controls: ResourceExpr | int,
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

        Returns:
            ResourceEstimate: Body estimate with measurement provenance
            preserved across the call boundary.

        Raises:
            ValueError: If recursive expansion repeats a resolved call state,
                changes only symbolically, or exhausts Python's call stack
                before reaching a base case.
            NotImplementedError: If the selected body contains legacy scalar
                Bit state that cannot flow between loop iterations.
        """
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
        local_bindings: dict[str, sp.Expr] = {}
        for formal, value in resolved_classical_inputs:
            if not value.is_number:
                continue
            local_bindings[formal.name] = value
            parameter_name = formal.parameter_name()
            if parameter_name is not None:
                local_bindings[parameter_name] = value
        self._validate_legacy_scalar_bit_rebinds(
            block,
            local_bindings=local_bindings,
        )
        active_states = self._active_call_states.setdefault(block_identity, [])
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
        active_states.append(call_state)
        tainted_formals = {
            formal.uuid: condition
            for formal, actual in pair_block_operands(block, actual_operands)
            if (
                condition := _value_taint_condition(
                    actual,
                    self._measurement_taint_conditions,
                )
            )
            is not sp.false
        }
        previous_taint = self._measurement_taint_conditions
        self._measurement_taint_conditions = _merge_measurement_taint_conditions(
            previous_taint,
            tainted_formals,
        )
        try:
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
                    active_when=self._constraint_scope_condition,
                    proven_cache=self._array_constraint_proven,
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
            self._measurement_taint_conditions = previous_taint
            active_states.pop()
            if not active_states:
                self._active_call_states.pop(block_identity, None)

    def eval_gate(
        self,
        operation: GateOperation,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a primitive gate operation.

        Args:
            operation (GateOperation): Gate operation.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Primitive gate resources.

        Raises:
            ValueError: If abstract controls are requested in the Clifford+T
                basis or the selected basis lacks a controlled lowering.
            NotImplementedError: If the clean-ancilla model has no registered
                lowering for the primitive.
        """
        if (
            self.config.basis is GateBasis.LOGICAL
            and self.config.control_decomposition
            is ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
        ):
            return _estimate_clean_ancilla_gate(operation, _expr(controls))
        if (
            self.config.basis is GateBasis.CLIFFORD_T
            and self.config.control_decomposition is ControlDecomposition.ABSTRACT
            and _expr(controls) != _ZERO
        ):
            raise ValueError(
                "Clifford+T estimation cannot preserve a controlled primitive "
                "as abstract. Select the clean-ancilla Toffoli control "
                "decomposition."
            )

        gates = _classify_gate(
            operation,
            num_controls=controls,
            basis=self.config.basis,
            precision=self.config.precision,
        )
        name = operation.gate_type.name.lower() if operation.gate_type else "gate"
        depth = (
            _clifford_t_gate_depth(operation, _expr(controls), self.config.precision)
            if self.config.basis is GateBasis.CLIFFORD_T
            else None
        )
        estimate = ResourceEstimate.primitive(name, gates, depth=depth)
        estimate = dataclasses.replace(
            estimate,
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=(
                self.config.precision
                if self.config.basis is GateBasis.CLIFFORD_T
                else None
            ),
        )
        if self.config.basis is GateBasis.CLIFFORD_T:
            clean_ancillas = _clifford_t_clean_ancillas(operation, _expr(controls))
            if clean_ancillas != _ZERO:
                estimate = dataclasses.replace(
                    estimate,
                    width=WidthResources(
                        clean_ancilla_qubits=clean_ancillas,
                        peak_qubits=clean_ancillas,
                    ),
                )
        if self.config.basis is GateBasis.CLIFFORD_T:
            conservative_when = _clifford_t_conservative_condition(
                name,
                _expr(controls),
            )
            if conservative_when is not sp.false:
                estimate = estimate._with_metadata(
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=conservative_when,
                )
            if _gate_has_rotation(operation):
                estimate = estimate._with_metadata(
                    quality=EstimateQuality.UNKNOWN,
                    approximation=ApproximationStatus.APPROXIMATE,
                )
        return estimate

    def eval_global_phase(
        self,
        operation: GlobalPhaseOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a zero-qubit phase under its surrounding controls.

        The target-neutral estimator assigns a standalone global phase no
        logical gate cost. A target materializer may still synthesize it with
        gates or a clean carrier. With one or more coherent controls it
        becomes a phase gate on one control, with the remaining controls
        guarding that gate.

        Args:
            operation (GlobalPhaseOperation): Phase operation to estimate.
            resolver (ExprResolver): Resolver for the phase operand.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Zero when uncontrolled, otherwise one logical
            phase-gate contribution with the correct arity.
        """
        phase = self._apply_condition_values(resolver.resolve(operation.phase))
        return self._estimate_global_phase_expression(phase, controls=controls)

    def _estimate_global_phase_expression(
        self,
        phase: sp.Expr,
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Estimate a resolved global-phase expression under controls.

        Args:
            phase (sp.Expr): Resolved phase angle in radians.
            controls (ResourceExpr | int): Number of coherent controls.

        Returns:
            ResourceEstimate: Zero for an unobservable global phase or an
            estimate for the relative phase induced by coherent controls.
        """
        control_count = _expr(controls)
        if control_count == _ZERO:
            return ResourceEstimate.zero("global_phase")
        phase_class = _CanonicalPhaseClass(phase)
        zero = ResourceEstimate.zero("global_phase")

        def gate_estimate(gate_name: str) -> ResourceEstimate:
            """Estimate one canonical relative-phase gate.

            Args:
                gate_name (str): Canonical phase gate name.

            Returns:
                ResourceEstimate: Gate estimate under all but one control.
            """
            estimate = _estimate_named_gate_in_basis(
                gate_name,
                control_count - _ONE,
                basis=self.config.basis,
                control_decomposition=self.config.control_decomposition,
                precision=self.config.precision,
            )
            return dataclasses.replace(
                estimate,
                trace=_wrap_trace(
                    "controlled_global_phase",
                    estimate.trace,
                    source_kind="decomposition",
                ),
            )

        if phase.is_number:
            gate_name = _canonical_phase_gate_name(phase)
            if gate_name is None:
                phase_estimate = zero
            else:
                if self.config.basis is not GateBasis.CLIFFORD_T:
                    # The shared control implementation preserves every nontrivial
                    # angle as P(theta), including special Clifford angles.
                    gate_name = "p"
                phase_estimate = gate_estimate(gate_name)
        elif self.config.basis is not GateBasis.CLIFFORD_T:
            phase_estimate = zero.conditional(
                gate_estimate("p"),
                sp.Eq(phase_class, _PHASE_CLASS_CODES[None]),
            )
        else:
            phase_estimate = gate_estimate("p")
            for gate_name in ("tdg", "sdg", "t", "s", "z"):
                phase_estimate = gate_estimate(gate_name).conditional(
                    phase_estimate,
                    sp.Eq(phase_class, _PHASE_CLASS_CODES[gate_name]),
                )
            phase_estimate = zero.conditional(
                phase_estimate,
                sp.Eq(phase_class, _PHASE_CLASS_CODES[None]),
            )

        if control_count.is_number and control_count.is_integer:
            return phase_estimate
        return zero.conditional(
            phase_estimate,
            sp.Eq(control_count, _ZERO),
        )

    def _with_zero_control_bracket(
        self,
        estimate: ResourceEstimate,
        *,
        zero_controls: ResourceExpr | int,
        surrounding_controls: ResourceExpr | int = 0,
        active_when: ResourceExpr | int | None = None,
    ) -> ResourceEstimate:
        """Bracket an estimate with X gates for zero-valued controls.

        One X is applied before and after the controlled region for every
        zero-valued control. ``surrounding_controls`` applies only to those X
        gates, as happens for SELECT nested under an outer controlled region.
        Brackets that normalize an operation's own controls are unconditional
        and therefore leave this argument at zero.

        Args:
            estimate (ResourceEstimate): Controlled-region estimate.
            zero_controls (ResourceExpr | int): Number of controls whose
                activation bit is zero.
            surrounding_controls (ResourceExpr | int): Controls inherited by
                the bracket X gates. Defaults to zero.
            active_when (ResourceExpr | int | None): Optional expression that
                must be positive for the region to emit. Defaults to ``None``.

        Returns:
            ResourceEstimate: Bracketed resource estimate.
        """
        zeros = _expr(zero_controls)
        if zeros == _ZERO:
            return estimate
        bracket_gate = _estimate_named_gate_in_basis(
            "x",
            _expr(surrounding_controls),
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=self.config.precision,
        )
        side = bracket_gate.repeat(zeros)
        if _expr(surrounding_controls) == _ZERO:
            side = dataclasses.replace(
                side,
                depth=_conditional_depth(
                    bracket_gate.depth,
                    DepthResources.zero(),
                    sp.Gt(zeros, _ZERO),
                ),
            )
        bracketed = side.seq(estimate).seq(side)
        if active_when is None:
            return bracketed
        active = _expr(active_when)
        if active.is_number:
            return bracketed if active > 0 else estimate
        return bracketed.conditional(
            estimate,
            _resource_activity_condition(active),
        )

    def eval_qinit(
        self,
        operation: QInitOperation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate a qubit allocation.

        Args:
            operation (QInitOperation): Qubit initialization operation.
            resolver (ExprResolver): Resolver for symbolic array shapes.

        Returns:
            ResourceEstimate: Width-only allocation estimate.
        """
        count = _count_qinit(operation, resolver)
        result = operation.results[0]
        self._allocation_owners_by_uuid[result.uuid] = result.logical_id
        dimension_constraints = (
            tuple(
                _ResourceConstraint(
                    expression=resolver.resolve(dimension),
                    minimum=0,
                    label=f"Qubit allocation dimension {position}",
                    unit="element",
                )
                for position, dimension in enumerate(result.shape)
            )
            if isinstance(result, ArrayValue)
            else ()
        )
        for constraint in dimension_constraints:
            constraint.validate()
        width = WidthResources(
            allocated_qubits=count,
            peak_qubits=count,
        )
        return _with_constraints(
            ResourceEstimate(
                width=width,
                trace=ResourceTraceNode(
                    "qinit",
                    "primitive",
                    summary=f"qubits={count}",
                ),
                _allocation_sites={operation.results[0].uuid: count},
            ),
            *dimension_constraints,
        )

    def eval_measure(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate a measurement operation.

        Args:
            operation (Operation): Measurement-like operation.
            resolver (ExprResolver): Resolver for vector operand dimensions.

        Returns:
            ResourceEstimate: Measurement depth estimate.

        Raises:
            TypeError: If ``operation`` is not a supported measurement IR
                operation.
        """
        if isinstance(operation, MeasureOperation):
            measured_qubits = _ONE
        elif isinstance(operation, MeasureVectorOperation):
            measured_qubits = (
                _qubit_value_size(operation.operands[0], resolver)
                if operation.operands
                else _ZERO
            )
        elif isinstance(operation, MeasureQFixedOperation):
            type_width = (
                _qubit_value_size(operation.operands[0], resolver)
                if operation.operands
                else _ZERO
            )
            measured_qubits = (
                type_width if type_width != _ZERO else sp.Integer(operation.num_bits)
            )
        else:  # pragma: no cover - dispatch admits only measurement operations.
            raise TypeError(
                f"Unsupported measurement operation {type(operation).__name__}."
            )
        layer = _piecewise(
            _ONE,
            _ZERO,
            sp.Gt(measured_qubits, _ZERO),
        )
        return ResourceEstimate(
            measurements=MeasurementResources(total=measured_qubits),
            depth=DepthResources(depth=layer, measurement_depth=layer),
            trace=ResourceTraceNode(type(operation).__name__, "primitive"),
        )

    def eval_expval(
        self,
        operation: ExpvalOp,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate an abstract expectation-value measurement.

        The semantic operation consumes the input state, but its concrete gate
        and shot cost depends on observable grouping, basis rotations, and the
        executor's sampling policy. The estimator therefore records one
        abstract query and one measurement layer without inventing a backend
        decomposition.

        Args:
            operation (ExpvalOp): Expectation-value operation.
            resolver (ExprResolver): Resolver for the quantum operand width.

        Returns:
            ResourceEstimate: Modeled abstract query and measurement depth.
        """
        measured_qubits = (
            _qubit_value_size(operation.qubits, resolver)
            if operation.operands
            else _ZERO
        )
        layer = _piecewise(
            _ONE,
            _ZERO,
            sp.Gt(measured_qubits, _ZERO),
        )
        assumption = ResourceAssumption(
            "expectation-value gate and shot costs depend on observable "
            "grouping, basis rotations, and executor sampling policy",
            source="ExpvalOp",
        )
        return ResourceEstimate(
            depth=DepthResources(depth=layer, measurement_depth=layer),
            calls=CallResources(
                calls_by_name={"expval": _ONE},
                queries_by_name={"expval": _ONE},
            ),
            assumptions=(assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            trace=ResourceTraceNode(
                "expval",
                "opaque",
                summary="one abstract expectation query",
                assumptions=(assumption,),
            ),
        )

    def eval_project(self, operation: ProjectOperation) -> ResourceEstimate:
        """Evaluate a projective measurement operation.

        The public X/Y projection helpers normally lower to basis-change gates
        around a Z projection before this interpreter runs. Hand-built or
        deserialized IR can still carry an X/Y axis, so estimate those semantic
        basis changes here as well instead of silently treating every axis as
        a bare measurement.

        Args:
            operation (ProjectOperation): Projection operation.

        Returns:
            ResourceEstimate: Measurement-like resource estimate.
        """
        measurement = ResourceEstimate(
            measurements=MeasurementResources(total=_ONE),
            depth=DepthResources(depth=_ONE, measurement_depth=_ONE),
            trace=ResourceTraceNode("project_z", "primitive"),
        )
        basis_changes = {
            "x": (("h",), ("h",)),
            "y": (("sdg", "h"), ("h", "s")),
            "z": ((), ()),
        }
        before, after = basis_changes[operation.axis]
        estimate = ResourceEstimate.zero()
        for gate_name in before:
            estimate = estimate.seq(
                _estimate_named_gate_in_basis(
                    gate_name,
                    _ZERO,
                    basis=self.config.basis,
                    control_decomposition=self.config.control_decomposition,
                    precision=self.config.precision,
                )
            )
        estimate = estimate.seq(measurement)
        for gate_name in after:
            estimate = estimate.seq(
                _estimate_named_gate_in_basis(
                    gate_name,
                    _ZERO,
                    basis=self.config.basis,
                    control_decomposition=self.config.control_decomposition,
                    precision=self.config.precision,
                )
            )
        return dataclasses.replace(
            estimate,
            trace=_wrap_trace(
                f"project_{operation.axis}",
                estimate.trace,
                source_kind="decomposition",
            ),
        )

    def eval_reset(self, operation: ResetOperation) -> ResourceEstimate:
        """Evaluate a reset operation.

        Args:
            operation (ResetOperation): Reset operation.

        Returns:
            ResourceEstimate: Reset primitive resource estimate.
        """
        return ResourceEstimate(
            resets=ResetResources(total=_ONE),
            depth=DepthResources(depth=_ONE, reset_depth=_ONE),
            trace=ResourceTraceNode(
                name="reset",
                source_kind="primitive",
                summary="resets=1",
            ),
        )

    def _initial_loop_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
    ) -> dict[LoopCarriedRebind, _ArrayState]:
        """Capture array states entering the first concrete loop iteration.

        Args:
            operation (ForOperation | ForItemsOperation): Loop whose explicit
                trace-time rebind records are inspected.
            resolver (ExprResolver): Resolver for the enclosing scope.

        Returns:
            dict[LoopCarriedRebind, _ArrayState]: Immutable entry snapshots for
            array rebind records only.
        """
        return {
            rebind: resolver.snapshot_array_state(cast(ArrayValue, rebind.before))
            for rebind in _loop_array_state_rebinds(operation)
        }

    def _bind_loop_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
        states: Mapping[LoopCarriedRebind, _ArrayState],
    ) -> None:
        """Bind prior array snapshots to one loop body's entry SSA values.

        Args:
            operation (ForOperation | ForItemsOperation): Loop being replayed.
            resolver (ExprResolver): Detached resolver for the next iteration.
            states (Mapping[LoopCarriedRebind, _ArrayState]): Prior exit state
                for every carried array lineage.
        """
        for rebind, state in states.items():
            assert isinstance(rebind.before, ArrayValue)
            resolver.bind_loop_array_input(
                operation.operations,
                rebind.before,
                state,
            )

    def _next_loop_array_states(
        self,
        resolver: ExprResolver,
        states: Mapping[LoopCarriedRebind, _ArrayState],
    ) -> dict[LoopCarriedRebind, _ArrayState]:
        """Capture array states produced by one concrete loop iteration.

        Args:
            resolver (ExprResolver): Evaluated iteration resolver.
            states (Mapping[LoopCarriedRebind, _ArrayState]): Carried records
                whose next snapshots are requested.

        Returns:
            dict[LoopCarriedRebind, _ArrayState]: Immutable exit snapshots.
        """
        return {
            rebind: resolver.snapshot_array_state(cast(ArrayValue, rebind.after))
            for rebind in states
        }

    def _summarize_for_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
        body_resolver: ExprResolver,
        initial_states: Mapping[LoopCarriedRebind, _ArrayState],
        *,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
        concrete_iterations: range | None,
        updates_are_unresolved: bool = False,
    ) -> dict[LoopCarriedRebind, _ArrayState]:
        """Build caller-visible array states after a range loop.

        One traced body is a persistent state transition. Concrete large loops
        instantiate and fold that transition in execution order; unresolved
        ranges retain a quantified summary so body-local induction symbols do
        not escape as public resource parameters.

        Args:
            operation (ForOperation | ForItemsOperation): Loop owning the body
                transition.
            resolver (ExprResolver): Enclosing resolver used to instantiate
                persistent states.
            body_resolver (ExprResolver): Evaluated one-iteration resolver.
            initial_states (Mapping[LoopCarriedRebind, _ArrayState]): Array
                snapshots before the first iteration.
            loop_symbol (sp.Symbol): Symbolic induction variable.
            start (ResourceExpr): First range value.
            step (ResourceExpr): Range stride.
            iterations (ResourceExpr): Number of range iterations.
            concrete_iterations (range | None): Concrete range when every
                bound is known, otherwise ``None``.
            updates_are_unresolved (bool): Whether symbolic iteration keys or
                values prevent an element-specific update summary. Defaults
                to ``False``.

        Returns:
            dict[LoopCarriedRebind, _ArrayState]: Folded or quantified exit
                states keyed by carried array boundary.
        """
        if not initial_states:
            return {}
        if concrete_iterations is None and updates_are_unresolved:
            exit_states: dict[LoopCarriedRebind, _ArrayState] = {}
            for rebind, initial_state in initial_states.items():
                assert isinstance(rebind.after, ArrayValue)
                fallback = _typed_value_symbol(
                    rebind.after,
                    f"{rebind.var_name}_after_loop_element",
                    fresh=True,
                )
                self._unresolved_resource_symbols.add(fallback)
                if isinstance(rebind.after.type, BitType):
                    self._runtime_value_domains[fallback] = (_ZERO, _ONE)
                uncertainty_token = f"$loop-array:{id(operation):x}:{rebind.after.uuid}"
                exit_states[rebind] = resolver.unknown_loop_array_summary_state(
                    initial=initial_state,
                    iterations=cast(sp.Expr, iterations),
                    fallback=fallback,
                    uncertainty_token=uncertainty_token,
                )
            return exit_states
        transition_states = self._next_loop_array_states(
            body_resolver,
            initial_states,
        )
        if concrete_iterations is not None:
            exit_states = dict(initial_states)
            for loop_value in concrete_iterations:
                replacements = tuple(
                    (initial_states[rebind], exit_states[rebind])
                    for rebind in initial_states
                )
                exit_states = {
                    rebind: resolver.instantiate_array_state(
                        transition_states[rebind],
                        state_replacements=replacements,
                        substitutions={loop_symbol: sp.Integer(loop_value)},
                    )
                    for rebind in initial_states
                }
            return exit_states

        exit_states: dict[LoopCarriedRebind, _ArrayState] = {}
        for rebind, initial_state in initial_states.items():
            assert isinstance(rebind.after, ArrayValue)
            fallback = _typed_value_symbol(
                rebind.after,
                f"{rebind.var_name}_after_loop_element",
                fresh=True,
            )
            self._unresolved_resource_symbols.add(fallback)
            if isinstance(rebind.after.type, BitType):
                self._runtime_value_domains[fallback] = (_ZERO, _ONE)
            uncertainty_token = f"$loop-array:{id(operation):x}:{rebind.after.uuid}"
            exit_states[rebind] = resolver.loop_array_summary_state(
                initial=initial_state,
                iteration=transition_states[rebind],
                loop_symbol=loop_symbol,
                start=cast(sp.Expr, start),
                step=cast(sp.Expr, step),
                iterations=cast(sp.Expr, iterations),
                fallback=fallback,
                uncertainty_token=uncertainty_token,
            )
        return exit_states

    def _conservative_loop_body_array_states(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
        initial_states: Mapping[LoopCarriedRebind, _ArrayState],
        *,
        completed_iterations: sp.Expr,
    ) -> tuple[dict[LoopCarriedRebind, _ArrayState], frozenset[str]]:
        """Represent carried array elements conservatively inside a loop body.

        A single symbolic trace cannot know the value written by every prior
        iteration. The first iteration retains the initial state; later
        iterations project an internal fallback value with an explicit source
        token. Resource-affecting reads therefore select the safe runtime
        branch instead of reusing only the first iteration's value.

        Args:
            operation (ForOperation | ForItemsOperation): Loop owning the
                carried array lineages.
            resolver (ExprResolver): Resolver used to create immutable summary
                states.
            initial_states (Mapping[LoopCarriedRebind, _ArrayState]): Array
                snapshots before the loop.
            completed_iterations (sp.Expr): Number of iterations preceding the
                body instance being summarized.

        Returns:
            tuple[dict[LoopCarriedRebind, _ArrayState], frozenset[str]]: Body
            entry states and the uncertainty source tokens they introduce.
        """
        states: dict[LoopCarriedRebind, _ArrayState] = {}
        source_tokens: set[str] = set()
        for rebind, initial_state in initial_states.items():
            assert isinstance(rebind.after, ArrayValue)
            fallback = _typed_value_symbol(
                rebind.after,
                f"{rebind.var_name}_loop_body_element",
                fresh=True,
            )
            self._unresolved_resource_symbols.add(fallback)
            if isinstance(rebind.after.type, BitType):
                self._runtime_value_domains[fallback] = (_ZERO, _ONE)
            source_token = f"$loop-array-body:{id(operation):x}:{rebind.after.uuid}"
            source_tokens.add(source_token)
            states[rebind] = resolver.unknown_loop_array_summary_state(
                initial=initial_state,
                iterations=completed_iterations,
                fallback=fallback,
                uncertainty_token=source_token,
            )
        return states, frozenset(source_tokens)

    @staticmethod
    def _estimate_reads_source_tokens(
        estimate: ResourceEstimate,
        source_tokens: Iterable[str],
    ) -> bool:
        """Return whether quantum work reads any supplied classical source.

        Args:
            estimate (ResourceEstimate): Body estimate carrying directed
                scheduler accesses.
            source_tokens (Iterable[str]): Classical source-token identities to
                test.

        Returns:
            bool: Whether a token is read, or access metadata is unavailable
            and the safe answer is therefore unknown.
        """
        dependency_keys = frozenset(
            _classical_dependency_key(token) for token in source_tokens
        )
        if not dependency_keys:
            return False
        if estimate._dependency_reads is None:
            return True
        return not dependency_keys.isdisjoint(estimate._dependency_reads)

    def _apply_disjoint_loop_depth(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        body_resolver: ExprResolver,
        body: ResourceEstimate,
        sequential: ResourceEstimate,
        *,
        start: ResourceExpr,
        stop: ResourceExpr,
        step: ResourceExpr,
        loop_symbol: sp.Symbol,
        iterations: ResourceExpr,
        specialized_iterations: ResourceExpr,
        controls: ResourceExpr | int,
        has_cross_iteration_dependency: bool = False,
    ) -> ResourceEstimate:
        """Use parallel depth when distinct loop iterations touch distinct wires.

        The same projection applies whether or not the loop carries classical
        region arguments. Classical carries affect later values, but do not by
        themselves serialize quantum work on provably disjoint wires.

        Args:
            operation (ForOperation): Loop whose body is summarized.
            resolver (ExprResolver): Enclosing resolver for concrete wire
                projections.
            body_resolver (ExprResolver): Resolver used to evaluate one
                symbolic body iteration.
            body (ResourceEstimate): One-iteration resource estimate.
            sequential (ResourceEstimate): Sum of the body over all iterations.
            start (ResourceExpr): Inclusive range start.
            stop (ResourceExpr): Exclusive range stop.
            step (ResourceExpr): Range stride.
            loop_symbol (sp.Symbol): Symbol used for the body iteration.
            iterations (ResourceExpr): Unspecialized iteration count.
            specialized_iterations (ResourceExpr): Iteration count after
                supplied input values are applied.
            controls (ResourceExpr | int): Controls surrounding the loop.
            has_cross_iteration_dependency (bool): Whether classical state
                used by quantum work may flow from one iteration into the
                next. Defaults to ``False``.

        Returns:
            ResourceEstimate: Estimate with parallel depth when disjointness is
            proven, or the sequential estimate with conservative metadata when
            symbolic wire aliasing remains unresolved.
        """
        loop_barrier_condition = sequential._global_barrier_condition
        if specialized_iterations.is_zero is True:
            return sequential
        if has_cross_iteration_dependency:
            assumption = ResourceAssumption(
                "loop depth is sequential because classical state used by "
                "quantum work may flow between iterations",
                source="for",
            )
            return sequential._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(iterations, _ONE),
            )
        if _expr(controls) != _ZERO or loop_barrier_condition is sp.true:
            return sequential

        parallel_depth = _symbolic_disjoint_loop_depth(
            operation,
            body_resolver,
            body.depth,
            loop_symbol=loop_symbol,
            iterations=iterations,
            allocated_qubits=body.width.allocated_qubits,
            clean_ancillas=body.width.clean_ancilla_qubits,
            dirty_ancillas=body.width.dirty_ancilla_qubits,
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        if parallel_depth is None:
            parallel_depth = _disjoint_concrete_loop_depth(
                operation,
                resolver,
                body.depth,
                body_dependency_keys=body._dependency_keys,
                start=start,
                stop=stop,
                step=step,
                loop_symbol=loop_symbol,
                allocated_qubits=body.width.allocated_qubits,
                clean_ancillas=body.width.clean_ancilla_qubits,
                dirty_ancillas=body.width.dirty_ancilla_qubits,
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            )
        if parallel_depth is None:
            if _dependency_keys_depend_on_symbol(
                body._dependency_keys,
                loop_symbol,
            ) or _loop_body_has_symbolic_quantum_index(
                operation,
                body_resolver,
                loop_symbol,
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            ):
                assumption = ResourceAssumption(
                    "symbolic loop depth is sequential because disjoint "
                    "iteration footprints could not be proven",
                    source="for",
                )
                return sequential._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=sp.And(
                        sp.Not(loop_barrier_condition),
                        sp.Gt(iterations, _ONE),
                    ),
                )
            return sequential

        parallel_completion = _concrete_loop_dependency_completion(
            body._dependency_completion,
            loop_symbol,
            start=start,
            stop=stop,
            step=step,
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        if parallel_completion is None:
            parallel_completion = _uniform_parallel_loop_dependency_completion(
                body._dependency_completion,
                body_depth=body.depth.depth,
                projected_keys=sequential._dependency_keys,
                parallel_depth=parallel_depth.depth,
                loop_symbol=loop_symbol,
            )
        parallel_estimate = dataclasses.replace(
            sequential,
            depth=parallel_depth,
            _dependency_completion=(
                parallel_completion
                if parallel_completion is not None
                else sequential._dependency_completion
            ),
            _dependency_completion_uniform=(
                body._dependency_completion_uniform is True
                and all(
                    loop_symbol
                    not in cast(
                        ResourceExpr,
                        getattr(body.depth, field.name),
                    ).free_symbols
                    for field in dataclasses.fields(DepthResources)
                )
            ),
            _global_barrier_condition=sp.false,
        )
        if parallel_completion is None and parallel_estimate._dependency_keys:
            assumption = ResourceAssumption(
                "parallel loop uses aggregate completion latency because "
                "per-wire exit layers could not be projected exactly",
                source="for",
            )
            parallel_estimate = parallel_estimate._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=sp.Gt(iterations, _ZERO),
            )
        return sequential.conditional(
            parallel_estimate,
            loop_barrier_condition,
        )

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
            self._allocation_owners_by_uuid,
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
                self.condition_values,
                self.branch_condition_names,
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
                        resolver,
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
                loop_exit_array_states = self._summarize_for_array_states(
                    operation,
                    resolver,
                    child,
                    initial_array_states,
                    loop_symbol=loop_symbol,
                    start=start,
                    step=step,
                    iterations=iterations,
                    concrete_iterations=(
                        None if body_array_source_tokens else concrete_iteration_range
                    ),
                    updates_are_unresolved=bool(body_array_source_tokens),
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
                allocation_owners_by_uuid=self._allocation_owners_by_uuid,
            )
        )
        estimate = _with_operation_output_summary(
            estimate,
            operation,
            resolver,
            active_when=sp.Gt(iterations, _ZERO),
            body_output_sizes=body_output_sizes,
            additional_consumed_allocations=additional_consumed,
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
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
                len(concrete_range[: _CONCRETE_REGION_REPLAY_LIMIT + 1])
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
                    self._measurement_taint_conditions,
                )
            )
            is not sp.false
        }
        composer = _SequentialEstimateComposer()
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
                            self._allocation_owners_by_uuid,
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
        context: dict[str, sp.Expr] = dict(carry_symbols)
        if operation.loop_var_value is not None:
            context[operation.loop_var_value.uuid] = loop_symbol
        initial_carry_taint = {
            arg.block_arg.uuid: condition
            for arg in operation.region_args
            if (
                condition := _value_taint_condition(
                    arg.init,
                    self._measurement_taint_conditions,
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
        self._bind_loop_array_states(operation, probe, initial_array_states)
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
                            self._allocation_owners_by_uuid,
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
                self._unresolved_resource_symbols.add(unknown_final_value)
                identity_guard = _invariant_identity_branch_guard(
                    yielded,
                    carry_symbol=carry_symbol,
                    invariant_symbols=(
                        _loop_invariant_symbols(
                            resolver,
                            operation.captures,
                            bound_expressions=(start, stop, step),
                        )
                        - self._runtime_observation_symbols
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
        completed_iterations = cast(
            sp.Expr,
            sp.simplify((loop_symbol - start) / step),
        )
        body_array_states, body_array_source_tokens = (
            self._conservative_loop_body_array_states(
                operation,
                resolver,
                initial_array_states,
                completed_iterations=completed_iterations,
            )
        )
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
                            self._allocation_owners_by_uuid,
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
                self.condition_values,
                self.branch_condition_names,
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
        exit_array_states = self._summarize_for_array_states(
            operation,
            resolver,
            child,
            initial_array_states,
            loop_symbol=loop_symbol,
            start=start,
            step=step,
            iterations=iterations,
            concrete_iterations=(None if body_array_source_tokens else concrete_replay),
            updates_are_unresolved=bool(body_array_source_tokens),
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

    def _apply_condition_values(
        self,
        expression: sp.Expr,
        *,
        record_usage: bool = True,
    ) -> sp.Expr:
        """Substitute supplied scalar values into an expression.

        Args:
            expression (sp.Expr): Expression to specialize.
            record_usage (bool): Whether matching input names should be marked
                as consumed by a scheduling or branch decision. Defaults to
                ``True``.

        Returns:
            sp.Expr: Specialized expression.
        """
        condition_inputs = {
            symbol: self.condition_values[_symbol_display_name(symbol)]
            for symbol in expression.free_symbols
            if not isinstance(symbol, sp.Dummy)
            and _symbol_display_name(symbol) in self.condition_values
        }
        if not condition_inputs:
            return expression
        if record_usage:
            self.branch_condition_names.update(
                _symbol_display_name(symbol) for symbol in condition_inputs
            )
        return cast(
            sp.Expr,
            expression.subs(list(condition_inputs.items()), simultaneous=True).doit(),
        )

    def _concrete_scalar(self, expression: sp.Expr) -> int | None:
        """Resolve an integer expression after any requested specialization.

        Args:
            expression (sp.Expr): Symbolic scalar expression.

        Returns:
            int | None: Concrete integer, or ``None`` when unresolved.
        """
        if expression.is_number and _is_concrete_integer(expression):
            return int(expression)
        return None

    def eval_while(
        self,
        operation: WhileOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a while loop using its declared or symbolic trip count.

        Args:
            operation (WhileOperation): While-loop operation.
            resolver (ExprResolver): Resolver for the outer scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Repeated loop-body estimate.

        Raises:
            ValueError: If the loop is nested under coherent quantum control.
            NotImplementedError: If the loop carries a rebound value whose
                recurrence the symbolic while-loop model cannot represent.
            AssertionError: If deterministic trip-count name reservation is
                unexpectedly missing for the operation.
        """
        _require_uncontrolled_operation(operation, controls)
        if operation.loop_carried_rebinds or operation.region_args:
            names = sorted(
                {rebind.var_name for rebind in operation.loop_carried_rebinds}
                | {arg.var_name for arg in operation.region_args}
            )
            variables = ", ".join(names) or "(unnamed)"
            raise NotImplementedError(
                "Resource estimation does not support loop-carried values in "
                f"WhileOperation ({variables}). A symbolic trip count alone "
                "cannot determine the carried recurrence."
            )
        self._reserve_while_trip_count_names((operation,), resolver)
        operation_key = (resolver.structural_scope, id(operation))
        cached_name = self._while_trip_count_names.get(operation_key)
        if cached_name is None or cached_name[0] is not operation:
            raise AssertionError("WhileOperation trip-count name was not reserved.")
        trip_count_name = cached_name[1]
        child, trip_count = build_while_scope(
            operation,
            resolver,
            trip_count_name=trip_count_name,
        )
        with self._guarded_constraint_scope(sp.Gt(trip_count, _ZERO)):
            inner = self.eval_operations(
                operation.operations,
                child,
                controls=controls,
                initial_allocations=_captured_quantum_allocations(
                    operation.operations,
                    child,
                    self._allocation_owners_by_uuid,
                ),
            )
            inner = _with_constraints(
                inner,
                *self._loop_iteration_array_constraints(
                    operation,
                    child,
                ),
            )
        estimate = _with_operation_output_summary(
            inner.repeat(trip_count),
            operation,
            resolver,
            active_when=sp.Gt(trip_count, _ZERO),
            body_output_sizes=inner._output_sizes,
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
        )
        assumption = ResourceAssumption(
            "runtime while resources use the declared trip count and a "
            "wire-local dependency envelope",
            source="while",
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
        )
        array_taint = self._guard_array_updates(
            operation.operations,
            resolver,
            estimate,
            active_when=sp.Gt(trip_count, _ZERO),
        )
        estimate = dataclasses.replace(
            estimate,
            _measurement_taint_conditions={
                **estimate._measurement_taint_conditions,
                **array_taint,
            },
        )
        return _with_constraints(
            estimate,
            *self._loop_initial_array_constraints(
                operation,
                resolver,
            ),
        )

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
                    self._runtime_observation_symbols
                    | self._unresolved_resource_symbols
                )
                or any(
                    symbol.name.endswith(uuid)
                    for uuid in self._measurement_taint_conditions
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
            self._allocation_owners_by_uuid,
        )
        false_inputs = _captured_quantum_allocations(
            operation.false_operations,
            false_child,
            self._allocation_owners_by_uuid,
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
                self._allocation_owners_by_uuid,
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
                self._allocation_owners_by_uuid,
            )
        )
        true_consumed = _definitely_consumed_captured_allocations(
            operation.true_operations,
            true_inputs,
            true_child,
            self._allocation_owners_by_uuid,
        )
        false_consumed = _definitely_consumed_captured_allocations(
            operation.false_operations,
            false_inputs,
            false_child,
            self._allocation_owners_by_uuid,
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
                    self._runtime_value_domains.pop(runtime_value, None)
                else:
                    runtime_branch_value = runtime_value
                    if runtime_condition is not sp.false:
                        self._runtime_observation_symbols.add(runtime_value)
                    else:
                        self._unresolved_resource_symbols.add(runtime_value)
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
                    self._allocation_owners_by_uuid.get(
                        value.uuid,
                        _quantum_allocation_owner(value),
                    )
                    for value in selected_values
                    if isinstance(value, Value) and value.type.is_quantum()
                }
                if len(owners) == 1:
                    self._allocation_owners_by_uuid[merge.result.uuid] = next(
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
                    self._dependency_owner_aliases[result_owner] = frozenset(
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
        outer = self._constraint_scope_condition
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
            proven_cache=self._array_constraint_proven,
        )

    def _loop_initial_array_constraints(
        self,
        operation: ForOperation | ForItemsOperation | WhileOperation,
        resolver: ExprResolver,
    ) -> tuple[_ResourceConstraint, ...]:
        """Collect array requirements evaluated before entering a loop.

        Args:
            operation (ForOperation | ForItemsOperation | WhileOperation): Loop
                whose entry-side boundary is inspected.
            resolver (ExprResolver): Enclosing resolver for entry values.

        Returns:
            tuple[_ResourceConstraint, ...]: Validated entry requirements.
        """
        if isinstance(operation, WhileOperation):
            initial_values: list[ValueBase] = list(operation.operands[:1])
        else:
            initial_values = [cast(ValueBase, value) for value in operation.operands]
        for region_arg in operation.region_args:
            initial_values.append(region_arg.init)
        explicit_rebinds = tuple(
            rebind
            for rebind in operation.loop_carried_rebinds
            if not isinstance(rebind.before, ArrayValue)
            or not isinstance(rebind.after, ArrayValue)
        )
        for rebind in (*_loop_array_state_rebinds(operation), *explicit_rebinds):
            initial_values.append(rebind.before)
        constraints = [
            constraint.when(self._constraint_scope_condition)
            for constraint in _collect_array_value_constraints(
                initial_values,
                resolver,
            )
        ]
        return _validated_unproven_array_constraints(
            constraints,
            proven_cache=self._array_constraint_proven,
        )

    def _loop_iteration_array_constraints(
        self,
        operation: ForOperation | ForItemsOperation | WhileOperation,
        body_resolver: ExprResolver,
    ) -> tuple[_ResourceConstraint, ...]:
        """Collect array requirements evaluated by one loop iteration.

        These constraints must be attached to the one-iteration estimate before
        a symbolic range is summed. That lets ``ResourceEstimate._sum_over``
        quantify induction variables and closed-form carried values instead of
        leaking them as public parameters or validating only one representative
        iteration.

        Args:
            operation (ForOperation | ForItemsOperation | WhileOperation): Loop
                whose body-side boundary is inspected.
            body_resolver (ExprResolver): Resolver for one body iteration.

        Returns:
            tuple[_ResourceConstraint, ...]: Validated per-iteration
            requirements under the current constraint scope.
        """
        body_values: list[ValueBase] = []
        if isinstance(operation, WhileOperation):
            body_values.extend(operation.operands[1:])
        body_values.extend(arg.yielded for arg in operation.region_args)
        body_values.extend(rebind.after for rebind in operation.loop_carried_rebinds)
        constraints = [
            constraint.when(self._constraint_scope_condition)
            for constraint in _collect_array_value_constraints(
                body_values,
                body_resolver,
            )
        ]
        return _validated_unproven_array_constraints(
            constraints,
            proven_cache=self._array_constraint_proven,
        )

    def _publish_loop_rebind_results(
        self,
        operation: ForOperation | ForItemsOperation,
        resolver: ExprResolver,
        body_resolver: ExprResolver,
        estimate: ResourceEstimate,
        *,
        active_when: sp.Basic,
        array_states: Mapping[LoopCarriedRebind, _ArrayState] | None = None,
    ) -> ResourceEstimate:
        """Publish loop-exit values and guarded observation provenance.

        A classical Bit-array store currently remains a body-local SSA rewrite:
        the traced store reads the pre-loop array version on every iteration,
        and the frontend exposes its result after the loop without a RegionArg.
        Bind that result to the source array on a zero-trip path while retaining
        the one traced body result when at least one iteration executes.

        Args:
            operation (ForOperation | ForItemsOperation): Loop containing
                trace-time rebound records.
            resolver (ExprResolver): Enclosing resolver to update.
            body_resolver (ExprResolver): Resolver for one body execution.
            estimate (ResourceEstimate): Loop estimate carrying body taint.
            active_when (sp.Basic): Predicate that at least one iteration runs.
            array_states (Mapping[LoopCarriedRebind, _ArrayState] | None):
                Optional already-folded loop-exit snapshots. Defaults to
                ``None``, which snapshots the evaluated body resolver.

        Returns:
            ResourceEstimate: Estimate with rebound-result provenance.
        """
        active = _boolean_condition(active_when)
        selector_fact = _ResolvedClassicalFact.create(active)
        explicit_nonarray_rebinds = tuple(
            rebind
            for rebind in operation.loop_carried_rebinds
            if not (
                isinstance(rebind.before, ArrayValue)
                and isinstance(rebind.after, ArrayValue)
            )
        )
        rebinds = (
            *_loop_array_state_rebinds(operation),
            *explicit_nonarray_rebinds,
        )
        array_before_states = {
            rebind: resolver.snapshot_array_state(cast(ArrayValue, rebind.before))
            for rebind in rebinds
            if isinstance(rebind.before, ArrayValue)
            and isinstance(rebind.after, ArrayValue)
        }
        array_before_facts = {
            rebind: resolver.resolve_classical_fact(rebind.before)
            for rebind in array_before_states
        }
        result_taint = self._guard_array_updates(
            operation.operations,
            resolver,
            estimate,
            active_when=active,
        )
        for rebind in rebinds:
            if isinstance(rebind.before, ArrayValue) or isinstance(
                rebind.after,
                ArrayValue,
            ):
                if not isinstance(rebind.before, ArrayValue) or not isinstance(
                    rebind.after,
                    ArrayValue,
                ):
                    continue
                before_state = array_before_states[rebind]
                after_state = (
                    array_states[rebind]
                    if array_states is not None and rebind in array_states
                    else body_resolver.snapshot_array_state(rebind.after)
                )
                if active is sp.true:
                    resolver.bind_array_state(rebind.after, after_state)
                elif active is sp.false:
                    resolver.bind_array_state(rebind.after, before_state)
                else:
                    resolver.bind_array_state_selection(
                        rebind.after,
                        after_state,
                        before_state,
                        selector_fact,
                    )
                resolver.bind_classical_selection(
                    rebind.after,
                    body_resolver.resolve_classical_fact(rebind.after),
                    array_before_facts[rebind],
                    selector_fact,
                )
                before_taint = _value_taint_condition(
                    rebind.before,
                    self._measurement_taint_conditions,
                )
                after_taint = _value_taint_condition(
                    rebind.after,
                    estimate._measurement_taint_conditions,
                )
                condition = _boolean_condition(
                    sp.Or(
                        sp.And(active, after_taint),
                        sp.And(sp.Not(active), before_taint),
                    )
                )
                if condition is not sp.false:
                    result_taint[rebind.after.uuid] = condition
                continue
            before_fact = resolver.resolve_classical_fact(rebind.before)
            after_fact = (
                before_fact
                if active is sp.false
                else body_resolver.resolve_classical_fact(rebind.after)
            )
            resolver.bind_classical_selection(
                cast(Value, rebind.after),
                after_fact,
                before_fact,
                selector_fact,
                value_override=_piecewise(
                    cast(ResourceExpr, after_fact.value),
                    cast(ResourceExpr, before_fact.value),
                    active,
                ),
            )
            before_taint = _value_taint_condition(
                rebind.before,
                self._measurement_taint_conditions,
            )
            after_taint = _value_taint_condition(
                rebind.after,
                estimate._measurement_taint_conditions,
            )
            condition = _boolean_condition(
                sp.Or(
                    sp.And(active, after_taint),
                    sp.And(sp.Not(active), before_taint),
                )
            )
            if condition is not sp.false:
                result_taint[rebind.after.uuid] = condition
        if not result_taint:
            return estimate
        return dataclasses.replace(
            estimate,
            _measurement_taint_conditions={
                **estimate._measurement_taint_conditions,
                **result_taint,
            },
        )

    def _guard_array_updates(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
        estimate: ResourceEstimate | None = None,
        *,
        active_when: sp.Basic,
    ) -> dict[str, Boolean]:
        """Guard nested classical-array stores by one region's reachability.

        Array-state bindings are shared by related resolver scopes. Walking the
        complete nested operation tree therefore composes loop and branch
        reachability from the innermost region outward while keeping each
        Store's input as its pre-region array version.

        Args:
            operations (Sequence[Operation]): Region operations whose nested
                stores are guarded.
            resolver (ExprResolver): Resolver owning the shared array-state
                bindings.
            estimate (ResourceEstimate | None): Region estimate carrying
                observation provenance. Defaults to ``None`` when the caller
                publishes provenance separately.
            active_when (sp.Basic): Predicate that the region executes.

        Returns:
            dict[str, Boolean]: Store-result observation provenance guarded by
                the region reachability condition.
        """
        active = _boolean_condition(active_when)
        taint: dict[str, Boolean] = {}
        for operation in walk_operations(operations):
            if not isinstance(operation, StoreArrayElementOperation):
                continue
            if not operation.results or not isinstance(
                operation.results[0],
                ArrayValue,
            ):
                continue
            result = operation.results[0]
            resolver.guard_array_update(result, operation.array, active)
            if estimate is None:
                continue
            condition = _value_taint_condition(
                result,
                estimate._measurement_taint_conditions,
            )
            guarded = _and_conditions(active, condition)
            if guarded is not sp.false:
                taint[result.uuid] = guarded
        return taint

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
                    scalar_values=self.condition_values,
                    used_names=self.branch_condition_names,
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
                len(self._dependency_owner_aliases.get(result_owner, frozenset())) > 1
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
                self._allocation_owners_by_uuid,
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
        owner_map = self._allocation_owners_by_uuid
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
        if self.condition_values and condition.free_symbols:
            subs: dict[Any, Any] = {}
            for symbol in condition.free_symbols:
                if isinstance(symbol, sp.Dummy):
                    continue
                name = _symbol_display_name(symbol)
                if name in self.condition_values:
                    subs[symbol] = self.condition_values[name]
                    used.add(name)
            if subs:
                condition = condition.subs(subs, simultaneous=True)
        if isinstance(condition, sp.logic.boolalg.BooleanAtom):
            decision: bool | None = bool(condition)
        elif condition.is_number:
            decision = bool(condition != 0)
        else:
            decision = None
        self.branch_condition_names |= used
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
        if message in self._reported_undecidable:
            return None, None
        self._reported_undecidable.add(message)
        return None, ResourceAssumption(message, source="if")

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
            self._allocation_owners_by_uuid,
        )
        consumption_iterations_are_definite = False
        if entries is not None and len(entries) <= _MAX_EXACT_LOOP_WIRE_EXPANSION:
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
        elif entries == ():
            consumption_resolvers = ()
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
                        resolver,
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
                                self._allocation_owners_by_uuid,
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
                exit_array_states = self._summarize_for_array_states(
                    operation,
                    resolver,
                    child,
                    initial_array_states,
                    loop_symbol=item_ordinal,
                    start=_ZERO,
                    step=_ONE,
                    iterations=cardinality,
                    concrete_iterations=None,
                    updates_are_unresolved=True,
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
                allocation_owners_by_uuid=self._allocation_owners_by_uuid,
            )
        )
        estimate = _with_operation_output_summary(
            estimate,
            operation,
            resolver,
            active_when=sp.Gt(cardinality, _ZERO),
            body_output_sizes=body_output_sizes,
            additional_consumed_allocations=additional_consumed,
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
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
        initial_carry_taint = {
            arg.block_arg.uuid: condition
            for arg in operation.region_args
            if (
                condition := _value_taint_condition(
                    arg.init,
                    self._measurement_taint_conditions,
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
            initial_array_states,
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
                            self._allocation_owners_by_uuid,
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
                self._unresolved_resource_symbols.add(unknown_final_value)
                identity_guard = _invariant_identity_branch_guard(
                    yielded,
                    carry_symbol=carry_symbol,
                    invariant_symbols=(
                        _loop_invariant_symbols(
                            resolver,
                            operation.captures,
                            bound_expressions=(cardinality,),
                        )
                        - self._runtime_observation_symbols
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
        body_array_states, body_array_source_tokens = (
            self._conservative_loop_body_array_states(
                operation,
                resolver,
                initial_array_states,
                completed_iterations=item_symbol,
            )
        )
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
                        self._allocation_owners_by_uuid,
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
        exit_array_states = self._summarize_for_array_states(
            operation,
            resolver,
            child,
            initial_array_states,
            loop_symbol=item_symbol,
            start=_ZERO,
            step=_ONE,
            iterations=cardinality,
            concrete_iterations=None,
            updates_are_unresolved=True,
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
        composer = _SequentialEstimateComposer()
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
                        self._allocation_owners_by_uuid,
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
                    self._measurement_taint_conditions,
                )
            )
            is not sp.false
        }
        composer = _SequentialEstimateComposer()
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
                            self._allocation_owners_by_uuid,
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

    def _schedule_concrete_loop_depth(
        self,
        operation: ForOperation | ForItemsOperation,
        entry_estimates: Sequence[ResourceEstimate],
        combined: ResourceEstimate,
    ) -> ResourceEstimate:
        """Schedule concrete loop iterations by their resolved wire use.

        Gate and call counts remain a sequential sum, but dictionary entries
        or range iterations acting on disjoint wires may occupy the same depth
        layers. A body-local allocation prevents this optimization because
        every iteration reuses the same allocation site.

        Args:
            operation (ForOperation | ForItemsOperation): Loop whose iterations
                were evaluated with concrete index/key/value bindings.
            entry_estimates (Sequence[ResourceEstimate]): Per-entry estimates
                before sequential depth offsets are applied.
            combined (ResourceEstimate): Sequentially composed estimate whose
                non-depth resources must be preserved.

        Returns:
            ResourceEstimate: Combined estimate with dependency-scheduled
                depth and caller-visible wire completion when available.
        """
        if not entry_estimates or any(
            estimate.width.allocated_qubits != _ZERO for estimate in entry_estimates
        ):
            return combined

        scheduled: list[tuple[Operation, ResourceEstimate]] = []
        footprints: list[_WireFootprint | None] = []
        for entry_estimate in entry_estimates:
            scheduled.append((operation, entry_estimate))
            if not _estimate_has_nonzero_depth(entry_estimate):
                footprints.append(None)
                continue
            keys = entry_estimate._dependency_keys
            if keys is None:
                assumption = ResourceAssumption(
                    "concrete loop iteration dependencies are unavailable; "
                    "depth remains sequential",
                    source="loop dependency scheduler",
                )
                return combined._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                )
            reads = (
                entry_estimate._dependency_reads
                if entry_estimate._dependency_reads is not None
                else keys
            )
            writes = (
                entry_estimate._dependency_writes
                if entry_estimate._dependency_writes is not None
                else keys
            )
            footprints.append((reads, writes))

        depth_activity_conditions = _scheduled_depth_activity_conditions(scheduled)
        (
            scheduled_depth,
            scheduled_completion,
            possible_alias_active,
            completion_is_uniform,
        ) = _dependency_depth(
            scheduled,
            footprints,
            activity_conditions=depth_activity_conditions,
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        result = dataclasses.replace(
            combined,
            depth=scheduled_depth,
            _dependency_completion=scheduled_completion,
            _dependency_completion_uniform=completion_is_uniform,
        )
        if possible_alias_active is not sp.false:
            assumption = ResourceAssumption(
                "concrete loop quantum indices may alias and are "
                "scheduled conservatively",
                source="loop dependency scheduler",
            )
            result = result._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=possible_alias_active,
            )
        aggregate_completion_active = _aggregate_completion_overlap_condition(
            scheduled,
            footprints,
            activity_conditions=depth_activity_conditions,
        )
        if aggregate_completion_active is not sp.false:
            assumption = ResourceAssumption(
                "aggregate loop-iteration latency may over-serialize a "
                "later wire dependency",
                source="loop dependency scheduler",
            )
            result = result._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.CONSERVATIVE,
                active_when=aggregate_completion_active,
            )
        return result

    def _for_items_entries(
        self,
        operation: ForItemsOperation,
    ) -> tuple[tuple[Any, Any], ...] | None:
        """Return concrete entries available to an items loop.

        Args:
            operation (ForItemsOperation): Items loop to inspect.

        Returns:
            tuple[tuple[Any, Any], ...] | None: Bound entries, or ``None`` when
            the dictionary remains symbolic.
        """
        if not operation.operands:
            return ()
        operand = operation.operands[0]
        parameter_name = getattr(operand, "parameter_name", lambda: None)()
        bound = self.bindings.get(parameter_name) if parameter_name else None
        if isinstance(bound, Mapping):
            return tuple(bound.items())
        return static_for_items_entries(operation)

    def _symbolic_for_items_context(
        self,
        operation: ForItemsOperation,
    ) -> tuple[dict[str, sp.Expr], frozenset[sp.Symbol]]:
        """Build identity-bearing symbolic bindings for an items loop.

        Args:
            operation (ForItemsOperation): Items loop to bind.

        Returns:
            tuple[dict[str, sp.Expr], frozenset[sp.Symbol]]: UUID-keyed
                symbolic context and every symbol derived from the current
                key/value entry.

        Raises:
            UnresolvedValueError: If key or value formal identities are absent.
        """
        if operation.key_var_values is None:
            raise UnresolvedValueError(
                "?",
                "ForItemsOperation is missing key_var_values identities.",
            )
        if operation.value_var_value is None:
            raise UnresolvedValueError(
                "?",
                "ForItemsOperation is missing value_var_value identity.",
            )

        context: dict[str, sp.Expr] = {}
        iteration_symbols: set[sp.Symbol] = set()
        array_symbols: dict[str, sp.Symbol] = {}

        for index, key_value in enumerate(operation.key_var_values):
            name = (
                operation.key_vars[index] if index < len(operation.key_vars) else "key"
            )
            symbol = _typed_value_symbol(key_value, name, fresh=True)
            context[key_value.uuid] = symbol
            iteration_symbols.add(symbol)
            if isinstance(key_value, ArrayValue):
                array_symbols[key_value.uuid] = symbol
                for axis, dimension in enumerate(key_value.shape):
                    dimension_symbol = sp.Dummy(
                        f"{name}_dim{axis}",
                        integer=True,
                        nonnegative=True,
                    )
                    context[dimension.uuid] = dimension_symbol
                    iteration_symbols.add(dimension_symbol)

        value_symbol = _typed_value_symbol(
            operation.value_var_value,
            operation.value_var,
            fresh=True,
        )
        context[operation.value_var_value.uuid] = value_symbol
        iteration_symbols.add(value_symbol)

        def array_iteration_symbol(array: ArrayValue) -> sp.Symbol | None:
            """Find the item symbol behind an array value or view.

            Args:
                array (ArrayValue): Array whose slice ancestry is inspected.

            Returns:
                sp.Symbol | None: Matching item symbol, if any.
            """
            current: ArrayValue | None = array
            visited: set[str] = set()
            while current is not None and current.uuid not in visited:
                visited.add(current.uuid)
                if current.uuid in array_symbols:
                    return array_symbols[current.uuid]
                current = current.slice_of
            return None

        def register_array_alias(value: ValueBase) -> None:
            """Map vector-key elements and views to their item dependency.

            Args:
                value (ValueBase): Referenced IR value to inspect.
            """
            if isinstance(value, Value) and value.parent_array is not None:
                symbol = array_iteration_symbol(value.parent_array)
                if symbol is not None:
                    context[value.uuid] = symbol
            if isinstance(value, ArrayValue):
                for dimension in value.shape:
                    register_array_alias(dimension)
                if value.slice_start is not None:
                    register_array_alias(value.slice_start)
                if value.slice_step is not None:
                    register_array_alias(value.slice_step)

        def walk(operations: list[Operation]) -> None:
            """Register vector-key aliases throughout nested body operations.

            Args:
                operations (list[Operation]): Operations in the current scope.
            """
            for nested_operation in operations:
                for value in (
                    *nested_operation.all_input_values(),
                    *nested_operation.results,
                ):
                    if isinstance(value, ValueBase):
                        register_array_alias(value)
                if isinstance(nested_operation, HasNestedOps):
                    for nested in nested_operation.nested_op_lists():
                        walk(nested)

        walk(operation.operations)
        return context, frozenset(iteration_symbols)

    def _ensure_for_items_resource_independent(
        self,
        estimate: ResourceEstimate,
        iteration_symbols: frozenset[sp.Symbol],
    ) -> None:
        """Reject symbolic item loops whose resources or constraints vary.

        Args:
            estimate (ResourceEstimate): One symbolic body evaluation.
            iteration_symbols (frozenset[sp.Symbol]): Identity-bearing symbols
                assigned to current key/value formals and their derived array
                values.

        Raises:
            NotImplementedError: If any resource metric or structural
                requirement depends on a current dictionary key or value.
        """
        if _free_symbols(estimate) & iteration_symbols:
            raise NotImplementedError(
                "Resource estimation does not support a symbolic "
                "ForItemsOperation body whose resource use or structural "
                "requirements depend on the current item key or value. "
                "Supply a concrete dictionary or make the body "
                "entry-independent."
            )

    def _concrete_for_items_context(
        self,
        operation: ForItemsOperation,
        key: Any,
        value: Any,
    ) -> dict[str, sp.Expr]:
        """Build concrete key and value bindings for one item iteration.

        Args:
            operation (ForItemsOperation): Items loop to bind.
            key (Any): Current dictionary key.
            value (Any): Current dictionary value.

        Returns:
            dict[str, sp.Expr]: UUID-keyed scalar iteration context.

        Raises:
            ValueError: If a tuple-key value does not match the declared key
                arity.
        """
        context: dict[str, sp.Expr] = {}
        key_values = list(operation.key_var_values or ())
        if (
            operation.key_is_vector
            and len(key_values) == 1
            and isinstance(key_values[0], ArrayValue)
        ):
            self._bind_concrete_vector_key(
                operation,
                key_values[0],
                key,
                context,
            )
        elif len(key_values) > 1:
            if (
                not isinstance(key, Sequence)
                or isinstance(key, (str, bytes))
                or len(key) != len(key_values)
            ):
                raise ValueError(
                    "ForItems tuple key must contain exactly "
                    f"{len(key_values)} element(s), got {key!r}."
                )
            for ir_value, concrete in zip(key_values, key, strict=True):
                context[ir_value.uuid] = _sympify_resource_value(
                    concrete, ir_value.name
                )
        elif key_values:
            context[key_values[0].uuid] = _sympify_resource_value(
                key,
                key_values[0].name,
            )
        if operation.value_var_value is not None:
            context[operation.value_var_value.uuid] = _sympify_resource_value(
                value,
                operation.value_var,
            )
        return context

    def _bind_concrete_vector_key(
        self,
        operation: ForItemsOperation,
        formal: ArrayValue,
        key: Any,
        context: dict[str, sp.Expr],
    ) -> None:
        """Bind a concrete Vector dictionary key into body Value identities.

        Args:
            operation (ForItemsOperation): Items loop whose body references the
                vector key.
            formal (ArrayValue): Vector-key region formal.
            key (Any): Concrete dictionary key for one iteration.
            context (dict[str, sp.Expr]): UUID-keyed context updated in place.

        Raises:
            ValueError: If the concrete key is not a sequence or a constant
                element access is out of bounds.
            NotImplementedError: If the body dynamically indexes the current
                vector key; exact per-entry resource evaluation for that shape
                is not implemented.
        """
        if not isinstance(key, Sequence) or isinstance(key, (str, bytes)):
            raise ValueError(
                "A concrete Dict[Vector, ...] key must be a sequence of scalar values."
            )
        elements = tuple(key)
        if formal.shape:
            context[formal.shape[0].uuid] = sp.Integer(len(elements))

        seen: set[int] = set()

        def belongs_to_formal(array: ArrayValue) -> bool:
            """Return whether an array is the formal or one of its views.

            Args:
                array (ArrayValue): Candidate key array or view.

            Returns:
                bool: True when its slice ancestry reaches the vector formal.
            """
            current: ArrayValue | None = array
            visited: set[int] = set()
            while current is not None and id(current) not in visited:
                visited.add(id(current))
                if current.logical_id == formal.logical_id:
                    return True
                current = current.slice_of
            return False

        def register_value(value: ValueBase) -> None:
            """Bind concrete key elements reachable through one IR value.

            Args:
                value (ValueBase): Referenced body value to inspect
                    recursively.

            Raises:
                ValueError: If a constant key index is out of bounds.
                NotImplementedError: If a current-key element has a dynamic
                    index or unresolved view mapping.
            """
            if id(value) in seen:
                return
            seen.add(id(value))

            for index in getattr(value, "element_indices", ()):
                register_value(index)
            parent = getattr(value, "parent_array", None)
            if isinstance(parent, ArrayValue):
                register_value(parent)

            if (
                isinstance(value, Value)
                and isinstance(parent, ArrayValue)
                and belongs_to_formal(parent)
            ):
                if len(value.element_indices) != 1:
                    raise NotImplementedError(
                        "Resource estimation supports only one-dimensional "
                        "constant indexing of a concrete ForItems Vector key."
                    )
                index_value = value.element_indices[0]
                if not index_value.is_constant():
                    raise NotImplementedError(
                        "Resource estimation cannot exactly evaluate a "
                        "dynamically indexed concrete ForItems Vector key."
                    )
                local_index = index_value.get_const()
                if isinstance(local_index, bool) or not isinstance(local_index, int):
                    raise NotImplementedError(
                        "Resource estimation requires integer indexing for a "
                        "concrete ForItems Vector key."
                    )
                resolved = resolve_root_array_index(parent, local_index)
                if resolved is None or resolved[0].logical_id != formal.logical_id:
                    raise NotImplementedError(
                        "Resource estimation could not resolve a concrete "
                        "ForItems Vector-key view to its formal index space."
                    )
                root_index = resolved[1]
                if not 0 <= root_index < len(elements):
                    raise ValueError(
                        f"ForItems Vector-key index {root_index} is out of "
                        f"bounds for a key of length {len(elements)}."
                    )
                context[value.uuid] = _sympify_resource_value(
                    elements[root_index],
                    value.name,
                )

            if isinstance(value, ArrayValue):
                for dimension in value.shape:
                    register_value(dimension)
                if value.slice_start is not None:
                    register_value(value.slice_start)
                if value.slice_step is not None:
                    register_value(value.slice_step)

        def walk(operations: list[Operation]) -> None:
            """Visit nested loop-body operations for key references.

            Args:
                operations (list[Operation]): Operations to inspect.
            """
            for body_operation in operations:
                for value in (
                    *body_operation.all_input_values(),
                    *body_operation.results,
                ):
                    if isinstance(value, ValueBase):
                        register_value(value)
                if isinstance(body_operation, HasNestedOps):
                    for nested in body_operation.nested_op_lists():
                        walk(nested)

        walk(operation.operations)

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
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
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
            )
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
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            )
            dependency_completion = _map_body_dependency_completion(
                operation.block,
                estimate,
                actual_operands,
                operation.results[len(operation.control_operands) :],
                resolver,
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
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
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            )
            dependency_completion = _map_body_dependency_completion(
                case_block,
                case_estimate,
                actual_operands,
                operation.results[operation.num_index_args :],
                resolver,
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            )
            if dependency_keys is not None:
                mapped_keys = set(dependency_keys)
                if _estimate_has_nonzero_depth(case_estimate):
                    mapped_keys.update(
                        _wire_keys_for_values(
                            operation.index_operands,
                            resolver,
                            scalar_values=self.condition_values,
                            used_names=self.branch_condition_names,
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
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        dependency_completion = _map_body_dependency_completion(
            operation.implementation_block,
            estimate,
            actual_operands,
            operation.results[operation.num_control_qubits :],
            resolver,
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        if dependency_keys is not None:
            mapped_keys = set(dependency_keys)
            if operation.num_control_qubits and _estimate_has_nonzero_depth(estimate):
                mapped_keys.update(
                    _wire_keys_for_values(
                        operation.control_qubits,
                        resolver,
                        scalar_values=self.condition_values,
                        used_names=self.branch_condition_names,
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

    def eval_pauli_evolve(
        self,
        operation: PauliEvolveOp,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a Pauli-gadget decomposition for a bound Hamiltonian.

        Basis changes and parity ladders stay uncontrolled under an enclosing
        controlled evolution; only the axial rotation is controlled. A
        Hamiltonian constant contributes a controlled relative global phase.

        Args:
            operation (PauliEvolveOp): Pauli evolution operation.
            resolver (ExprResolver): Resolver for observable and time operands.
            controls (ResourceExpr | int): Surrounding coherent controls.
                Defaults to zero.

        Returns:
            ResourceEstimate: Clean-ancilla Pauli-gadget resources, or a modeled
            opaque/zero estimate when the configured unknown policy permits
            an unbound Hamiltonian.

        Raises:
            ValueError: If the Hamiltonian is unbound under ``ERROR`` policy,
                is non-Hermitian, or requires more qubits than its target
                register provides.
        """
        import qamomile.observable as qm_o
        from qamomile.observable.hamiltonian import (
            HERMITIAN_IMAG_ATOL,
            PAULI_TERM_ZERO_ATOL,
            _pauli_strings_anticommute,
        )

        hamiltonian = self._resolve_hamiltonian_binding(operation, resolver)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            assumption = ResourceAssumption(
                "PauliEvolveOp requires a bound Hamiltonian for gate resources.",
                source="PauliEvolveOp",
            )
            if self.config.unknown_policy is UnknownResourcePolicy.ERROR:
                raise ValueError(
                    "Cannot estimate PauliEvolveOp without a bound "
                    "Hamiltonian; supply the observable through inputs or "
                    "bindings, or select a non-error unknown resource policy."
                )
            calls = CallResources.zero()
            if self.config.unknown_policy is UnknownResourcePolicy.OPAQUE_CALL:
                calls = CallResources(
                    calls_by_name={"pauli_evolve": _ONE},
                    queries_by_name={"pauli_evolve": _ONE},
                )
            return ResourceEstimate(
                calls=calls,
                assumptions=(assumption,),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
                trace=ResourceTraceNode(
                    "pauli_evolve",
                    "modeled",
                    assumptions=(assumption,),
                ),
            )

        register_constraint = _ResourceConstraint(
            expression=_qubit_value_size(operation.qubits, resolver),
            minimum=hamiltonian.num_qubits,
            label=(
                "Pauli evolution register width for a "
                f"{hamiltonian.num_qubits}-qubit Hamiltonian"
            ),
            unit="qubit",
        )
        register_constraint.validate()

        gamma = self._apply_condition_values(resolver.resolve(operation.gamma))
        constant = complex(hamiltonian.constant)
        if not math.isfinite(constant.real) or not math.isfinite(constant.imag):
            raise ValueError(
                "PauliEvolveOp requires finite Hamiltonian coefficients, but "
                f"the constant is {hamiltonian.constant}."
            )
        if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
            raise ValueError(
                "PauliEvolveOp requires a Hermitian Hamiltonian (real "
                "constant), but the constant has a nonzero imaginary part."
            )
        for _operators, coefficient in hamiltonian:
            numeric_coefficient = complex(coefficient)
            if not math.isfinite(numeric_coefficient.real) or not math.isfinite(
                numeric_coefficient.imag
            ):
                raise ValueError(
                    "PauliEvolveOp requires finite Hamiltonian coefficients, "
                    f"but a term coefficient is {coefficient}."
                )
            if abs(numeric_coefficient.imag) > HERMITIAN_IMAG_ATOL:
                raise ValueError(
                    "PauliEvolveOp requires a Hermitian Hamiltonian (real "
                    "Pauli coefficients), but a term has a nonzero imaginary "
                    "part."
                )
        if gamma.is_zero is True:
            return _with_constraints(
                ResourceEstimate.zero("pauli_evolve"),
                register_constraint,
            )

        term_estimates: list[ResourceEstimate] = []
        active_pauli_terms: list[tuple[qm_o.PauliOperator, ...]] = []
        for operators, coefficient in hamiltonian:
            resolved_coefficient = complex(coefficient)
            if abs(resolved_coefficient) < PAULI_TERM_ZERO_ATOL or not operators:
                continue
            active_pauli_terms.append(operators)
            x_count = sum(operator.pauli == qm_o.Pauli.X for operator in operators)
            y_count = sum(operator.pauli == qm_o.Pauli.Y for operator in operators)
            basis_h_gate = _estimate_named_gate_in_basis(
                "h",
                _ZERO,
                basis=self.config.basis,
                control_decomposition=self.config.control_decomposition,
                precision=self.config.precision,
            )
            basis_h = basis_h_gate.repeat(2 * (x_count + y_count))
            basis_s_gate = _estimate_named_gate_in_basis(
                "s",
                _ZERO,
                basis=self.config.basis,
                control_decomposition=self.config.control_decomposition,
                precision=self.config.precision,
            )
            basis_s = basis_s_gate.repeat(2 * y_count)
            parity_gate = _estimate_named_gate_in_basis(
                "cx",
                _ZERO,
                basis=self.config.basis,
                control_decomposition=self.config.control_decomposition,
                precision=self.config.precision,
            )
            parity = parity_gate.repeat(2 * max(0, len(operators) - 1))
            rotation = _estimate_named_gate_in_basis(
                "rz",
                _expr(controls),
                basis=self.config.basis,
                control_decomposition=self.config.control_decomposition,
                precision=self.config.precision,
            )
            term = basis_h.seq(basis_s).seq(parity).seq(rotation)
            term = dataclasses.replace(
                term,
                depth=_classify_pauli_evolve_depth(
                    operators,
                    basis_h_layer=basis_h_gate.depth,
                    basis_s_layer=basis_s_gate.depth,
                    parity_layer=parity_gate.depth,
                    rotation=rotation.depth,
                ),
            )
            term_estimates.append(term)

        if abs(constant) >= PAULI_TERM_ZERO_ATOL:
            phase = -gamma * sp.Float(constant.real)
            term_estimates.append(
                self._estimate_global_phase_expression(
                    cast(sp.Expr, phase),
                    controls=controls,
                )
            )
        estimate = ResourceEstimate.seq_all(term_estimates)
        trotter_assumption = None
        if not _pauli_terms_share_local_basis(active_pauli_terms) and any(
            _pauli_strings_anticommute(left, right)
            for index, left in enumerate(active_pauli_terms)
            for right in active_pauli_terms[index + 1 :]
        ):
            trotter_assumption = ResourceAssumption(
                "Noncommuting Pauli terms are counted as one first-order "
                "Lie-Trotter product-formula step in Hamiltonian term order.",
                source="PauliEvolveOp",
            )
        trace = _wrap_trace(
            "pauli_evolve",
            estimate.trace,
            source_kind="body",
        )
        if trotter_assumption is not None:
            trace = dataclasses.replace(
                trace,
                assumptions=(*trace.assumptions, trotter_assumption),
            )
        active_estimate = dataclasses.replace(
            estimate,
            trace=trace,
        )
        if trotter_assumption is not None:
            active_estimate = active_estimate._with_metadata(
                assumptions=(trotter_assumption,),
                approximation=ApproximationStatus.APPROXIMATE,
            )
        if gamma.is_zero is not False:
            active_estimate = ResourceEstimate.zero("pauli_evolve").conditional(
                active_estimate,
                sp.Eq(gamma, _ZERO),
            )
        return _with_constraints(active_estimate, register_constraint)

    def _resolve_hamiltonian_binding(
        self,
        operation: PauliEvolveOp,
        resolver: ExprResolver,
    ) -> Any:
        """Resolve a Pauli evolution's observable through nested call scopes.

        Args:
            operation (PauliEvolveOp): Pauli evolution operation.
            resolver (ExprResolver): Resolver for the current callable scope.

        Returns:
            Any: Bound Hamiltonian value, or ``None`` when no binding matches.
        """
        observable = operation.observable
        candidates = [observable.name, observable.uuid]
        resolved = resolver.resolve(observable)
        if isinstance(resolved, sp.Symbol):
            candidates.append(_symbol_display_name(resolved))
        for candidate in candidates:
            if candidate in self.bindings:
                return self.bindings[candidate]
        return None

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
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=(
                self.config.precision
                if self.config.basis is GateBasis.CLIFFORD_T
                else None
            ),
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
            ValueError: If the opaque cost reports gate-model-sensitive
                resources for a different basis, control decomposition, or
                synthesis precision, or if a non-unitary cost is controlled or
                inverted.
        """
        estimate = self._resolve_opaque_definition_cost(
            operation,
            cost,
            context,
        )
        # Root input width and caller-scoped dependency/liveness maps belong to
        # the qkernel that produced an aggregate cost, not to this opaque call
        # site. The caller scheduler derives that boundary information from
        # this InvokeOperation instead.
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
            context.basis,
            context.control_decomposition,
            context.precision,
        )
        cached = self._opaque_definition_cost_cache.get(cache_key)
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
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=self.config.precision,
        )
        # Public resource dataclasses accept ordinary Python numeric values.
        # Compose one base application through the resource algebra so every
        # field is normalized before dependency scheduling inspects SymPy
        # predicates.
        normalized = estimate.repeat(_ONE)
        self._opaque_definition_cost_cache[cache_key] = (operation, normalized)
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
        body_estimate = self._eval_call_body(
            body,
            child,
            actual_operands,
            controls=total_controls,
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
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        dependency_completion = _map_body_dependency_completion(
            body,
            body_estimate,
            actual_operands,
            caller_results,
            resolver,
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        if dependency_keys is not None:
            mapped_keys = set(dependency_keys)
            if local_controls and _estimate_has_nonzero_depth(body_estimate):
                mapped_keys.update(
                    _wire_keys_for_values(
                        operation.operands[: operation.num_control_qubits],
                        resolver,
                        scalar_values=self.condition_values,
                        used_names=self.branch_condition_names,
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
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
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
            self._block_runtime_observation_summary(body)
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


def estimate_resources(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
    *,
    inputs: dict[str, Any] | None = None,
    strategies: dict[str, str] | None = None,
    trace: bool = False,
    unknown_policy: str | UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
    basis: str | GateBasis = _DEFAULT_GATE_BASIS,
    control_decomposition: str | ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION,
    precision: float = _DEFAULT_ROTATION_SYNTHESIS_PRECISION,
) -> ResourceEstimate:
    """Estimate algorithmic resources using the default estimator facade.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): QKernel,
            block, or operation sequence to estimate.
        inputs (dict[str, Any] | None): QKernel input values used to specialize
            the symbolic estimate without building a problem-sized circuit.
            Exact one-dimensional root quantum-port widths declared by
            callable resource metadata are inferred when omitted. Defaults to
            ``None``.
        strategies (dict[str, str] | None): Strategy overrides by callable
            name. Defaults to ``None``.
        trace (bool): Whether to retain the explanation tree. Defaults to
            ``False``.
        unknown_policy (str | UnknownResourcePolicy): Unknown callable
            handling. Defaults to ``ERROR``.
        basis (str | GateBasis): Output gate basis. Defaults to ``LOGICAL``.
        control_decomposition (str | ControlDecomposition): Coherent-control
            decomposition. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.
        precision (float): Rotation-synthesis precision for ``CLIFFORD_T``.
            Defaults to ``1e-10``.

    Returns:
        ResourceEstimate: Algorithmic resource estimate.

    Raises:
        ValueError: If the basis, precision, input specialization, callable
            resource contract, or structural requirements are invalid.
        TypeError: If ``kernel`` is not a supported estimator input.
        NotImplementedError: If the input IR contains a construct not
            supported by resource estimation.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def repeated_h(n: qmc.UInt) -> qmc.Qubit:
        ...     q = qmc.qubit("q")
        ...     for _ in qmc.range(n):
        ...         q = qmc.h(q)
        ...     return q
        >>> symbolic = estimate_resources(repeated_h)
        >>> str(symbolic.gates.total)
        'n'
        >>> estimate_resources(repeated_h, inputs={"n": 8}).gates.total
        8
    """
    estimator = ResourceEstimator(
        strategies=strategies,
        trace=trace,
        unknown_policy=unknown_policy,
        basis=basis,
        control_decomposition=control_decomposition,
        precision=precision,
    )
    return estimator.estimate(
        kernel,
        inputs=inputs,
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


def _with_constraints(
    estimate: ResourceEstimate,
    *constraints: _ResourceConstraint,
) -> ResourceEstimate:
    """Attach validated structural requirements to an estimate.

    Args:
        estimate (ResourceEstimate): Estimate that owns the requirements.
        *constraints (_ResourceConstraint): Requirements to retain.

    Returns:
        ResourceEstimate: Estimate carrying the appended constraints.

    Raises:
        ValueError: If a constraint is already concretely invalid.
    """
    for constraint in constraints:
        constraint.validate()
    return dataclasses.replace(
        estimate,
        _constraints=(*constraints, *estimate._constraints),
    )


def _operation_array_constraints(
    operation: Operation,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic = sp.true,
    proven_cache: dict[_ResourceConstraint, bool] | None = None,
) -> tuple[_ResourceConstraint, ...]:
    """Collect unresolved array-access requirements for one operation.

    Embedded element provenance is carried by ``Value.parent_array`` and
    ``Value.element_indices`` rather than by a dedicated load operation. This
    traversal follows those references, tuple/dict members, and array-view
    ancestry so every operation kind receives the same bounds checks. Classical
    stores and deferred quantum returns encode their indices as separate
    operands and are paired with their arrays explicitly below.

    Args:
        operation (Operation): IR operation whose direct values are inspected.
            Nested operation lists are evaluated in their own resolver scopes
            and are therefore not recursively walked here.
        resolver (ExprResolver): Resolver for array dimensions, element
            indices, and view affine-map expressions in the operation's scope.
        active_when (sp.Basic): Predicate under which the operation executes.
            Defaults to true.
        proven_cache (dict[_ResourceConstraint, bool] | None): Optional cache
            of previously validated requirements and whether their symbolic
            assumptions prove them. Defaults to ``None``.

    Returns:
        tuple[_ResourceConstraint, ...]: Deduplicated requirements that are not
            already provable from symbolic type assumptions.

    Raises:
        ValueError: If an access has the wrong number of indices, a view has
            malformed rank/affine metadata, or a concrete access is out of
            bounds.
    """
    constraints = list(
        _collect_array_value_constraints(
            (*operation.all_input_values(), *operation.results),
            resolver,
        )
    )

    if isinstance(operation, StoreArrayElementOperation):
        constraints.extend(
            _array_index_constraints(
                operation.array,
                operation.index_values,
                resolver,
                access_kind="store",
            )
        )
    elif isinstance(operation, ReturnQuantumArrayElementOperation):
        constraints.extend(
            _array_index_constraints(
                operation.array,
                operation.target_indices,
                resolver,
                access_kind="return target",
            )
        )
        constraints.extend(
            _array_index_constraints(
                operation.array,
                operation.source_indices,
                resolver,
                access_kind="return source",
            )
        )

    return _validated_unproven_array_constraints(
        tuple(constraint.when(active_when) for constraint in constraints),
        proven_cache=proven_cache,
    )


def _collect_array_value_constraints(
    values: Iterable[ValueBase],
    resolver: ExprResolver,
) -> tuple[_ResourceConstraint, ...]:
    """Collect unvalidated array requirements embedded in values.

    Args:
        values (Iterable[ValueBase]): Values whose element and view ancestry
            should be inspected.
        resolver (ExprResolver): Resolver for dimensions, indices, and affine
            view metadata.

    Returns:
        tuple[_ResourceConstraint, ...]: Raw, possibly duplicated structural
        requirements.
    """
    constraints: list[_ResourceConstraint] = []
    visited: set[str] = set()

    def visit(value: ValueBase) -> None:
        """Visit one value and its embedded array dependencies.

        Args:
            value (ValueBase): Value, array, tuple, or dictionary reference to
                inspect recursively.
        """
        if value.uuid in visited:
            return
        visited.add(value.uuid)

        if isinstance(value, TupleValue):
            for element in value.elements:
                visit(element)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit(key)
                visit(entry_value)
            return
        if isinstance(value, ArrayValue):
            constraints.extend(_array_view_constraints(value, resolver))
            for dimension in value.shape:
                visit(dimension)
            if value.slice_of is not None:
                visit(value.slice_of)
            if value.slice_start is not None:
                visit(value.slice_start)
            if value.slice_step is not None:
                visit(value.slice_step)
            return
        if isinstance(value, Value):
            if value.parent_array is not None:
                constraints.extend(
                    _array_index_constraints(
                        value.parent_array,
                        value.element_indices,
                        resolver,
                        access_kind="element",
                    )
                )
                visit(value.parent_array)
            for index in value.element_indices:
                visit(index)

    for value in values:
        visit(value)
    return tuple(constraints)


def _array_index_constraints(
    array: ArrayValue,
    indices: Sequence[Value],
    resolver: ExprResolver,
    *,
    access_kind: str,
) -> tuple[_ResourceConstraint, ...]:
    """Build per-axis lower and upper bounds for one array access.

    Args:
        array (ArrayValue): Array whose local coordinate space is addressed.
        indices (Sequence[Value]): One index expression per array dimension.
        resolver (ExprResolver): Resolver for the index and shape expressions.
        access_kind (str): User-facing access role such as ``"element"`` or
            ``"store"`` used in diagnostic labels.

    Returns:
        tuple[_ResourceConstraint, ...]: Two constraints per array dimension.

    Raises:
        ValueError: If the number of indices differs from the array rank.
    """
    if len(indices) != len(array.shape):
        display_name = array.name or "array"
        raise ValueError(
            f"Array '{display_name}' {access_kind} access requires "
            f"{len(array.shape)} indices; got {len(indices)}."
        )

    display_name = array.name or "array"
    constraints: list[_ResourceConstraint] = []
    for axis, (index, dimension) in enumerate(zip(indices, array.shape, strict=True)):
        index_expression = resolver.resolve(index)
        dimension_expression = resolver.resolve(dimension)
        label = f"Array '{display_name}' {access_kind} axis {axis} index"
        constraints.extend(
            (
                _ResourceConstraint(
                    expression=index_expression,
                    minimum=0,
                    label=f"{label} lower bound",
                ),
                _ResourceConstraint(
                    expression=dimension_expression - index_expression,
                    minimum=1,
                    label=(
                        f"Array '{display_name}' {access_kind} axis {axis} "
                        "in-bounds margin (dimension - index)"
                    ),
                    unit="element",
                ),
            )
        )
    return tuple(constraints)


def _array_view_constraints(
    view: ArrayValue,
    resolver: ExprResolver,
) -> tuple[_ResourceConstraint, ...]:
    """Build affine-map and full-coverage requirements for one array view.

    A local element bound is insufficient for a view whose declared length
    extends beyond its parent. Requiring the final covered parent coordinate to
    remain in bounds also protects whole-view operations such as vector
    measurement, where no scalar element ``Value`` exists in the IR. Empty
    views make the coverage requirement vacuous because they address no parent
    slot.

    Args:
        view (ArrayValue): Array value that may be a strided one-dimensional
            view over another array.
        resolver (ExprResolver): Resolver for shape, start, and step values.

    Returns:
        tuple[_ResourceConstraint, ...]: Empty for a root array, otherwise the
            view length/start/step and nonempty coverage requirements.

    Raises:
        ValueError: If a sliced array lacks one-dimensional shape metadata or
            either affine-map operand.
    """
    if view.slice_of is None:
        return ()
    if (
        len(view.shape) != 1
        or len(view.slice_of.shape) != 1
        or view.slice_start is None
        or view.slice_step is None
    ):
        raise ValueError(
            f"Array view '{view.name or 'array'}' must have one-dimensional "
            "shape plus slice_start and slice_step metadata."
        )

    length = resolver.resolve(view.shape[0])
    parent_length = resolver.resolve(view.slice_of.shape[0])
    start = resolver.resolve(view.slice_start)
    step = resolver.resolve(view.slice_step)
    final_parent_index = start + step * (length - _ONE)
    coverage_slack = _piecewise(
        parent_length - final_parent_index,
        _ONE,
        sp.Gt(length, _ZERO),
    )
    display_name = view.name or "array"
    return (
        _ResourceConstraint(
            expression=length,
            minimum=0,
            label=f"Array view '{display_name}' length",
            unit="element",
        ),
        _ResourceConstraint(
            expression=start,
            minimum=0,
            label=f"Array view '{display_name}' start",
        ),
        _ResourceConstraint(
            expression=step,
            minimum=1,
            label=f"Array view '{display_name}' step",
        ),
        _ResourceConstraint(
            expression=coverage_slack,
            minimum=1,
            label=f"Array view '{display_name}' parent coverage",
            unit="element",
        ),
    )


def _validated_unproven_array_constraints(
    constraints: Sequence[_ResourceConstraint],
    *,
    proven_cache: dict[_ResourceConstraint, bool] | None = None,
) -> tuple[_ResourceConstraint, ...]:
    """Validate, prune, and deduplicate array structural requirements.

    Args:
        constraints (Sequence[_ResourceConstraint]): Fresh requirements from
            one operation's directly referenced values.
        proven_cache (dict[_ResourceConstraint, bool] | None): Optional cache
            of prior validation and proof results. Defaults to ``None``.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements that may still fail after
            later input substitution, preserving first-seen order.

    Raises:
        ValueError: If a requirement is already concretely invalid.
    """
    retained: list[_ResourceConstraint] = []
    seen: set[_ResourceConstraint] = set()
    for constraint in constraints:
        proven = proven_cache.get(constraint) if proven_cache is not None else None
        if proven is None:
            constraint.validate()
            proven = _array_constraint_is_proven(constraint)
            if proven_cache is not None:
                proven_cache[constraint] = proven
        if proven or constraint in seen:
            continue
        seen.add(constraint)
        retained.append(constraint)
    return tuple(retained)


def _array_constraint_is_proven(constraint: _ResourceConstraint) -> bool:
    """Return whether symbolic assumptions already prove a requirement.

    Args:
        constraint (_ResourceConstraint): Requirement to inspect after concrete
            validation.

    Returns:
        bool: Whether integrality, equality, and lower bounds are all proven
            without retaining the requirement for later substitution.
    """
    if constraint.ranges:
        return False
    expression = _safe_simplify(constraint.expression)
    integer_proven = (
        not constraint.integer
        or expression.is_integer is True
        or (expression.is_number and _is_concrete_integer(expression))
    )
    if not integer_proven:
        return False
    if constraint.finite and expression.is_finite is not True:
        return False
    if constraint.expected is not None:
        expected = _safe_simplify(constraint.expected)
        if _safe_simplify(expression - expected) != _ZERO:
            return False
    if constraint.minimum is not None:
        return constraint._minimum_relation(expression) is sp.true
    return True


def _zero_control_count(
    num_controls: int,
    control_value: int | None,
) -> int:
    """Return the number of zero bits in a concrete control pattern.

    Args:
        num_controls (int): Width of the control register.
        control_value (int | None): LSB-first activation value, or ``None``
            for the ordinary all-ones pattern.

    Returns:
        int: Controls that require X bracketing.
    """
    if control_value is None:
        return 0
    return num_controls - control_value.bit_count()


def _estimate_activity(estimate: ResourceEstimate) -> ResourceExpr:
    """Return an expression that is zero only for an empty operation estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        ResourceExpr: Gate, measurement, reset, and callable activity.
    """
    call_activity = sum(estimate.calls.calls_by_name.values(), _ZERO)
    query_activity = sum(estimate.calls.queries_by_name.values(), _ZERO)
    return (
        estimate.gates.total
        + estimate.measurements.total
        + estimate.resets.total
        + call_activity
        + query_activity
    )


def _estimate_nonunitary_boundary_condition(
    estimate: ResourceEstimate,
) -> Boolean:
    """Return when an estimate declares measurement or reset work.

    Bodyless opaque calls cannot expose output-level observation provenance,
    but their explicit resource contract can still prove that the call is a
    non-unitary scheduling boundary.

    Args:
        estimate (ResourceEstimate): Aggregate invocation estimate to inspect.

    Returns:
        Boolean: Condition under which a measurement/reset count or
            corresponding depth may be nonzero.
    """
    conditions = tuple(
        _resource_activity_condition(value)
        for value in (
            estimate.measurements.total,
            estimate.resets.total,
            estimate.depth.measurement_depth,
            estimate.depth.reset_depth,
        )
    )
    return _boolean_condition(sp.Or(*conditions))


def _normalized_dependency_completion(
    estimate: ResourceEstimate,
) -> dict[WireKey, ResourceExpr] | None:
    """Return an explicit per-wire completion map when one can be recovered.

    Args:
        estimate (ResourceEstimate): Estimate carrying dependency metadata.

    Returns:
        dict[WireKey, ResourceExpr] | None: Caller-visible local completion
        depths, an empty map for a proven empty estimate, or ``None`` for an
        unknown footprint.
    """
    if estimate._dependency_completion is not None:
        return dict(estimate._dependency_completion)
    keys = estimate._dependency_keys
    if keys is None:
        if not _estimate_has_nonzero_depth(estimate):
            return {}
        return None
    return {key: estimate.depth.depth for key in keys}


def _merge_dependency_accesses(
    left: ResourceEstimate,
    right: ResourceEstimate,
    *,
    writes: bool,
) -> frozenset[WireKey] | None:
    """Merge body-level scheduler reads or writes across two estimates.

    Args:
        left (ResourceEstimate): Left composition operand.
        right (ResourceEstimate): Right composition operand.
        writes (bool): Whether to merge write sets instead of read sets.

    Returns:
        frozenset[WireKey] | None: Union of known access sets, or ``None``
            when a nonempty operand has no recoverable footprint.
    """

    def normalized(estimate: ResourceEstimate) -> frozenset[WireKey] | None:
        """Recover one access set from explicit or symmetric metadata.

        Args:
            estimate (ResourceEstimate): Estimate whose access set is needed.

        Returns:
            frozenset[WireKey] | None: Explicit access set, symmetric fallback,
                empty set for zero depth, or ``None`` when unknown.
        """
        accesses = estimate._dependency_writes if writes else estimate._dependency_reads
        if accesses is not None:
            return accesses
        if estimate._dependency_keys is not None:
            return estimate._dependency_keys
        if not _estimate_has_nonzero_depth(estimate):
            return frozenset()
        return None

    left_accesses = normalized(left)
    right_accesses = normalized(right)
    if left_accesses is None or right_accesses is None:
        return None
    return left_accesses | right_accesses


def _project_dependency_metadata_over_symbol(
    estimate: ResourceEstimate,
    symbol: sp.Symbol,
    *,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> ResourceEstimate:
    """Project loop-local scalar wire keys into the enclosing scope.

    A loop induction symbol is local to one iteration. Once the loop is
    summarized, an address such as ``register[i]`` represents a set of
    caller-visible wires rather than one scalar wire. Concrete bounded loops
    enumerate those addresses exactly. A symbolic or large one-dimensional
    loop retains a canonical range descriptor, while a nested range that
    cannot be represented without leaking an outer binder falls back to an
    unknown-scalar marker. Neither representation pretends that the loop
    definitely touched the whole allocation.

    Args:
        estimate (ResourceEstimate): Estimate whose dependency metadata may
            contain the bound symbol.
        symbol (sp.Symbol): Bound symbol leaving scope.
        start (ResourceExpr): First Python-range value.
        step (ResourceExpr): Python-range step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        ResourceEstimate: Estimate with concrete loop addresses enumerated and
            unresolved ranges represented without leaking local symbols.
    """

    start_expr = _expr(start)
    step_expr = _expr(step)
    iterations_expr = _expr(iterations)
    concrete_values: tuple[sp.Integer, ...] | None = None
    if all(
        bound.is_number and _is_concrete_integer(bound)
        for bound in (start_expr, step_expr, iterations_expr)
    ):
        concrete_start = int(start_expr)
        concrete_step = int(step_expr)
        concrete_iterations = int(iterations_expr)
        if (
            concrete_step != 0
            and 0 <= concrete_iterations <= _MAX_EXACT_LOOP_WIRE_EXPANSION
        ):
            concrete_values = tuple(
                sp.Integer(concrete_start + concrete_step * offset)
                for offset in range(concrete_iterations)
            )

    def project(key: WireKey) -> tuple[WireKey, ...]:
        """Project one symbol-dependent scalar address.

        Args:
            key (WireKey): Allocation-owner and scalar-index address.

        Returns:
            tuple[WireKey, ...]: Concrete projected addresses, one unknown
                scalar address, or the unchanged address.
        """
        owner, index = key
        if isinstance(index, _WireRangeIndex):
            range_symbols = (
                index.index_at_offset.free_symbols | index.iterations.free_symbols
            )
            if symbol not in range_symbols:
                return (key,)
            if concrete_values is None:
                return ((owner, _UNKNOWN_WIRE_INDEX),)
            return tuple(
                (
                    owner,
                    index.mapped(
                        lambda expression: expression.subs(
                            symbol,
                            value,
                            simultaneous=True,
                        )
                    ),
                )
                for value in concrete_values
            )
        if not isinstance(index, sp.Expr) or symbol not in index.free_symbols:
            return (key,)
        if concrete_values is None:
            return (
                (
                    owner,
                    _symbolic_wire_range_index(
                        index,
                        symbol,
                        start=start_expr,
                        step=step_expr,
                        iterations=iterations_expr,
                    ),
                ),
            )
        return tuple(
            (
                owner,
                _normalize_wire_index(
                    cast(
                        ResourceExpr,
                        index.subs(symbol, value, simultaneous=True),
                    )
                ),
            )
            for value in concrete_values
        )

    keys = estimate._dependency_keys
    projected_keys = (
        frozenset(projected for key in keys for projected in project(key))
        if keys is not None
        else None
    )
    reads = estimate._dependency_reads
    projected_reads = (
        frozenset(projected for key in reads for projected in project(key))
        if reads is not None
        else None
    )
    writes = estimate._dependency_writes
    projected_writes = (
        frozenset(projected for key in writes for projected in project(key))
        if writes is not None
        else None
    )
    completion = estimate._dependency_completion
    projected_completion: dict[WireKey, ResourceExpr] | None
    if completion is None:
        projected_completion = None
    else:
        projected_completion = {}
        for key, depth in completion.items():
            for projected in project(key):
                projected_completion[projected] = _resource_max(
                    projected_completion.get(projected, _ZERO),
                    depth,
                )
    return dataclasses.replace(
        estimate,
        _dependency_keys=projected_keys,
        _dependency_reads=projected_reads,
        _dependency_writes=projected_writes,
        _dependency_completion=projected_completion,
    )


def _seq_dependency_completion(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> dict[WireKey, ResourceExpr] | None:
    """Compose per-wire completion depths sequentially.

    Args:
        left (ResourceEstimate): First composition operand.
        right (ResourceEstimate): Second composition operand.

    Returns:
        dict[WireKey, ResourceExpr] | None: Sequential completion depths, or
        ``None`` if either nonempty footprint is unknown.
    """
    left_completion = _normalized_dependency_completion(left)
    right_completion = _normalized_dependency_completion(right)
    if left_completion is None or right_completion is None:
        return None
    merged = dict(left_completion)
    for key, completion in right_completion.items():
        previous = merged.get(key, _ZERO)
        active = _ConditionIndicator(_resource_activity_condition(completion))
        merged[key] = cast(
            ResourceExpr,
            previous + active * (left.depth.depth + completion - previous),
        )
    return merged


def _max_dependency_completion(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> dict[WireKey, ResourceExpr] | None:
    """Merge per-wire completion depths by their maximum.

    Args:
        left (ResourceEstimate): First composition operand.
        right (ResourceEstimate): Second composition operand.

    Returns:
        dict[WireKey, ResourceExpr] | None: Per-wire maximums, or ``None`` if
        either nonempty footprint is unknown.
    """
    left_completion = _normalized_dependency_completion(left)
    right_completion = _normalized_dependency_completion(right)
    if left_completion is None or right_completion is None:
        return None
    return {
        key: _resource_max(
            left_completion.get(key, _ZERO),
            right_completion.get(key, _ZERO),
        )
        for key in left_completion.keys() | right_completion.keys()
    }


def _conditional_dependency_completion(
    true_estimate: ResourceEstimate,
    false_estimate: ResourceEstimate,
    condition: sp.Basic,
) -> dict[WireKey, ResourceExpr] | None:
    """Select per-wire completion between two symbolic branches.

    Args:
        true_estimate (ResourceEstimate): Estimate selected when true.
        false_estimate (ResourceEstimate): Estimate selected when false.
        condition (sp.Basic): Branch predicate.

    Returns:
        dict[WireKey, ResourceExpr] | None: Exact branch-selected completion
        depths, or ``None`` when either nonempty branch is unknown.
    """
    true_completion = _normalized_dependency_completion(true_estimate)
    false_completion = _normalized_dependency_completion(false_estimate)
    if true_completion is None or false_completion is None:
        return None
    return {
        key: cast(
            ResourceExpr,
            false_completion.get(key, _ZERO)
            + _ConditionIndicator(condition)
            * (true_completion.get(key, _ZERO) - false_completion.get(key, _ZERO)),
        )
        for key in true_completion.keys() | false_completion.keys()
    }


def _map_dependency_key(
    key: WireKey,
    fn: Any,
) -> WireKey:
    """Rewrite the symbolic index of one private dependency key.

    Args:
        key (WireKey): Allocation-owner and optional scalar-index address.
        fn (Any): Symbolic expression rewrite callable.

    Returns:
        WireKey: Address with its symbolic scalar index rewritten and
            normalized.
    """
    owner, index = key
    if isinstance(index, _WireRangeIndex):
        return owner, index.mapped(fn)
    if not isinstance(index, sp.Expr):
        return key
    return owner, _normalize_wire_index(cast(ResourceExpr, fn(index)))


def _map_dependency_keys(
    keys: frozenset[WireKey] | None,
    fn: Any,
) -> frozenset[WireKey] | None:
    """Rewrite every symbolic private dependency key.

    Args:
        keys (frozenset[WireKey] | None): Dependency addresses, or ``None``
            when the footprint is unavailable.
        fn (Any): Symbolic expression rewrite callable.

    Returns:
        frozenset[WireKey] | None: Rewritten dependency addresses.
    """
    if keys is None:
        return None
    return frozenset(_map_dependency_key(key, fn) for key in keys)


def _map_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    fn: Any,
) -> dict[WireKey, ResourceExpr] | None:
    """Rewrite per-wire addresses and completion depths.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): Completion depths
            to rewrite.
        fn (Any): Symbolic expression rewrite callable.

    Returns:
        dict[WireKey, ResourceExpr] | None: Rewritten nonzero completion
            depths with colliding addresses merged.
    """
    if completion is None:
        return None
    mapped: dict[WireKey, ResourceExpr] = {}
    for key, value in completion.items():
        rewritten = cast(ResourceExpr, fn(value))
        if rewritten != _ZERO:
            mapped_key = _map_dependency_key(key, fn)
            mapped[mapped_key] = _resource_max(
                mapped.get(mapped_key, _ZERO),
                rewritten,
            )
    return mapped


def _complete_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    keys: Iterable[WireKey],
    *,
    fallback_depth: ResourceExpr,
) -> dict[WireKey, ResourceExpr]:
    """Fill missing caller-wire completion with an aggregate depth.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): Precisely mapped
            completion depths when available.
        keys (Iterable[WireKey]): Complete caller-visible dependency keys.
        fallback_depth (ResourceExpr): Aggregate completion used for keys
            whose individual latency is unavailable.

    Returns:
        dict[WireKey, ResourceExpr]: Complete per-wire completion map.
    """
    completed = dict(completion or {})
    for key in keys:
        completed.setdefault(key, fallback_depth)
    return completed


def _canonical_phase_gate_name(phase: sp.Expr) -> str | None:
    """Return the simplest fixed phase gate for a resolved angle.

    Numeric multiples of pi/4 are recognized modulo 2*pi. Other numeric and
    symbolic phases retain the arbitrary phase gate ``p``.

    Args:
        phase (sp.Expr): Resolved real phase angle in radians.

    Returns:
        str | None: ``None`` for an identity phase, a fixed gate name for a
        recognized Clifford+T phase, or ``"p"`` otherwise.
    """
    if not phase.is_number or phase.is_real is False:
        return "p"
    pi_quarters = sp.simplify(4 * phase / sp.pi)
    if pi_quarters.is_integer is True and pi_quarters.is_number:
        exact_class = int(pi_quarters) % 8
        return {
            0: None,
            1: "t",
            2: "s",
            4: "z",
            6: "sdg",
            7: "tdg",
        }.get(exact_class, "p")

    numeric_value = float(sp.N(phase))
    if not math.isfinite(numeric_value):
        return "p"
    value = math.fmod(numeric_value, math.tau)
    if value < 0:
        value += math.tau
    # Match only a literal floating-point representation of the identity.
    # A fixed absolute tolerance here would silently erase small but nonzero
    # phases and could undercount below the requested synthesis precision.
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=0.0):
        return None
    fixed = (
        (math.pi / 4, "t"),
        (math.pi / 2, "s"),
        (math.pi, "z"),
        (3 * math.pi / 2, "sdg"),
        (7 * math.pi / 4, "tdg"),
    )
    for angle, gate_name in fixed:
        floating_error = 4 * max(math.ulp(value), math.ulp(angle))
        if math.isclose(value, angle, rel_tol=0.0, abs_tol=floating_error):
            return gate_name
    return "p"


def _sympify_resource_value(value: Any, fallback_name: str) -> sp.Expr:
    """Convert a bound loop item to a scalar resource expression.

    Args:
        value (Any): Bound dictionary key or value.
        fallback_name (str): Symbol name used for a non-scalar value.

    Returns:
        sp.Expr: SymPy scalar, or a symbolic placeholder for structural data.
    """
    try:
        expression = sp.sympify(value)
    except (TypeError, ValueError, sp.SympifyError):
        return sp.Symbol(fallback_name)
    if isinstance(expression, sp.Expr):
        return expression
    return sp.Symbol(fallback_name)


def build_for_loop_scope(
    operation: ForOperation,
    resolver: ExprResolver,
) -> tuple[ExprResolver, ResourceExpr, ResourceExpr, ResourceExpr, sp.Symbol]:
    """Build resolver and symbolic bounds for a for loop.

    Args:
        operation (ForOperation): Loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, ResourceExpr, ResourceExpr, ResourceExpr, sp.Symbol]:
        Child resolver, start, stop, step, and loop-variable symbol.

    Raises:
        UnresolvedValueError: If the loop has no UUID-bearing loop-variable
            value and would require unsafe display-name resolution.
    """
    if operation.loop_var_value is None:
        raise UnresolvedValueError(
            "?",
            "ForOperation is missing loop_var_value identity; display-name "
            "resolution is unsafe for nested loops.",
        )
    loop_symbol = sp.Dummy(operation.loop_var, integer=True)
    child = resolver.child_scope(
        inner_block=_LocalBlock(operation.operations),
        extra_context={operation.loop_var_value.uuid: loop_symbol},
        # Opaque resource models expose loop variables by their source-level
        # names. Expression resolution itself remains UUID-based above.
        extra_loop_vars={operation.loop_var: loop_symbol},
    )
    start = child.resolve(operation.operands[0])
    stop = child.resolve(operation.operands[1])
    step = (
        child.resolve(operation.operands[2]) if len(operation.operands) >= 3 else _ONE
    )
    return child, start, stop, step, loop_symbol


def _solve_affine_recurrence(
    *,
    yielded: sp.Expr,
    carry_symbol: sp.Symbol,
    other_carry_symbols: set[sp.Symbol],
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
    init: sp.Expr,
) -> tuple[sp.Expr, sp.Expr] | None:
    """Solve one independent affine loop-carried recurrence.

    The supported recurrence is ``x[k + 1] = a*x[k] + b(k)`` where ``a``
    does not depend on the loop index or another carry. This covers counters,
    arithmetic accumulators, and geometric updates without requiring users to
    write separate resource equations for ordinary classical qkernel code.

    Args:
        yielded (sp.Expr): Expression yielded by one body iteration.
        carry_symbol (sp.Symbol): Symbol representing the incoming carry.
        other_carry_symbols (set[sp.Symbol]): Symbols for simultaneously carried
            values, which are rejected as coupled recurrences.
        loop_symbol (sp.Symbol): Symbol representing the Python loop value.
        start (ResourceExpr): Inclusive Python-range start.
        step (ResourceExpr): Python-range step.
        iterations (ResourceExpr): Symbolic number of loop iterations.
        init (sp.Expr): Carry value before iteration zero.

    Returns:
        tuple[sp.Expr, sp.Expr] | None: Carry at the current loop iteration and
        final carry after all iterations, or ``None`` when the recurrence is
        nonlinear, coupled, or has an index-dependent multiplier.
    """
    try:
        fixed_point_residual = sp.simplify(yielded.subs(carry_symbol, init) - init)
        coefficient = sp.simplify(sp.diff(yielded, carry_symbol))
        remainder = sp.simplify(yielded - coefficient * carry_symbol)
    except _SYMPY_SIMPLIFICATION_ERRORS:
        # Nested symbolic sums and piecewise observation values can exceed
        # SymPy's polynomial reduction domain. Such a failure means only that
        # this optional compact solver cannot prove an affine closed form; the
        # caller already has a conservative symbolic or concrete-replay path.
        return None
    if fixed_point_residual == _ZERO:
        return init, init
    if (
        carry_symbol in coefficient.free_symbols
        or carry_symbol in remainder.free_symbols
    ):
        return None
    if (coefficient.free_symbols | remainder.free_symbols) & other_carry_symbols:
        return None
    if loop_symbol in coefficient.free_symbols:
        return None

    summation_index = sp.Symbol(
        f"{loop_symbol}_previous", integer=True, nonnegative=True
    )
    previous_loop_value = start + summation_index * step
    previous_remainder = remainder.subs(loop_symbol, previous_loop_value)

    def value_after(count: sp.Expr) -> sp.Expr:
        """Return the recurrence value after ``count`` iterations.

        Args:
            count (sp.Expr): Number of completed iterations.

        Returns:
            sp.Expr: Closed-form recurrence value.
        """
        if coefficient == 1:
            accumulated = sp.Sum(
                previous_remainder,
                (summation_index, 0, count - 1),
            ).doit()
            return cast(sp.Expr, sp.simplify(init + accumulated))
        if coefficient == 0:
            last_remainder = remainder.subs(
                loop_symbol,
                start + (count - 1) * step,
            )
            return cast(
                sp.Expr,
                sp.Piecewise((init, sp.Eq(count, 0)), (last_remainder, True)),
            )
        accumulated = sp.Sum(
            coefficient ** (count - 1 - summation_index) * previous_remainder,
            (summation_index, 0, count - 1),
        ).doit()
        return cast(
            sp.Expr,
            sp.simplify(coefficient**count * init + accumulated),
        )

    try:
        completed_at_loop_value = sp.simplify((loop_symbol - start) / step)
        at_iteration = value_after(completed_at_loop_value)
        final = value_after(iterations)
    except _SYMPY_SIMPLIFICATION_ERRORS:
        return None
    nonfinite_atoms = (sp.nan, sp.zoo, sp.oo, -sp.oo)
    if at_iteration.has(*nonfinite_atoms) or final.has(*nonfinite_atoms):
        return None
    return at_iteration, final


def _invariant_identity_branch_guard(
    yielded: sp.Expr,
    *,
    carry_symbol: sp.Symbol,
    invariant_symbols: set[sp.Symbol],
) -> Boolean:
    """Return where an unsupported recurrence provably preserves its carry.

    Only ordered ``Piecewise`` branches whose value is exactly the incoming
    carry are considered. Every symbol in an effective guard must come from an
    enclosing capture or loop bound that the caller proved invariant. This
    positive proof avoids mistaking an unresolved body-local fallback symbol
    for an immutable input. Unresolved functions are rejected as well.

    Args:
        yielded (sp.Expr): Expression yielded by one loop iteration.
        carry_symbol (sp.Symbol): Symbol representing the incoming carry.
        invariant_symbols (set[sp.Symbol]): Symbols proven to come from
            immutable enclosing captures or loop bounds.

    Returns:
        Boolean: Ordered condition under which every iteration is an identity,
        or ``False`` when no such invariant branch can be proved.
    """
    if not isinstance(yielded, sp.Piecewise):
        return sp.false
    branches = cast(tuple[Any, ...], yielded.args)
    if not branches or branches[-1][1] is not sp.true:
        return sp.false
    remaining: Boolean = sp.true
    identity_guards: list[Boolean] = []
    for value, raw_guard in branches:
        branch_guard = _boolean_condition(cast(sp.Basic, raw_guard))
        effective_guard = _and_conditions(remaining, branch_guard)
        guard_symbols = cast(set[sp.Symbol], effective_guard.free_symbols)
        guard_is_invariant = not (
            not guard_symbols <= invariant_symbols
            or any(
                isinstance(node, AppliedUndef)
                for node in sp.preorder_traversal(effective_guard)
            )
        )
        preserves_carry = value == carry_symbol
        if guard_is_invariant and preserves_carry:
            identity_guards.append(effective_guard)
        remaining = _and_conditions(
            remaining,
            cast(Boolean, sp.Not(branch_guard)),
        )
        if remaining is sp.false:
            break
    return cast(Boolean, sp.Or(*identity_guards))


def _loop_invariant_symbols(
    resolver: ExprResolver,
    captures: Iterable[ValueBase],
    *,
    bound_expressions: Iterable[sp.Expr],
) -> set[sp.Symbol]:
    """Collect symbols proven invariant for one loop invocation.

    Region captures are explicit IR references to values defined outside the
    loop body. Scalar captures and captured array dimensions therefore remain
    fixed throughout that invocation, as do the already-resolved loop bounds.
    Container contents are deliberately not guessed from their printed form.

    Args:
        resolver (ExprResolver): Resolver for the enclosing loop scope.
        captures (Iterable[ValueBase]): Explicit outer-scope region captures.
        bound_expressions (Iterable[sp.Expr]): Resolved bounds or cardinality
            expressions fixed before the loop begins.

    Returns:
        set[sp.Symbol]: SymPy identities with an IR-backed invariance proof.
    """
    invariant: set[sp.Symbol] = set()

    def add_expression(expression: sp.Expr) -> None:
        """Add every scalar symbol in one proven-invariant expression.

        Args:
            expression (sp.Expr): Resolved enclosing expression.
        """
        invariant.update(
            symbol
            for symbol in expression.free_symbols
            if isinstance(symbol, sp.Symbol)
        )

    for expression in bound_expressions:
        add_expression(expression)
    for capture in captures:
        if isinstance(capture, ArrayValue):
            for dimension in capture.shape:
                add_expression(resolver.resolve(dimension))
        elif isinstance(capture, Value) and not capture.type.is_quantum():
            add_expression(resolver.resolve(capture))
    return invariant


def build_while_scope(
    operation: WhileOperation,
    resolver: ExprResolver,
    *,
    trip_count_name: str = "|while|",
) -> tuple[ExprResolver, sp.Symbol]:
    """Build resolver and symbolic trip count for a while loop.

    Args:
        operation (WhileOperation): While-loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.
        trip_count_name (str): Public symbolic trip-count name. Defaults to
            ``"|while|"`` for backward compatibility.

    Returns:
        tuple[ExprResolver, sp.Symbol]: Child resolver and ``|while|`` symbol.
    """
    child = resolver.child_scope(inner_block=_LocalBlock(operation.operations))
    return child, sp.Symbol(trip_count_name, integer=True, nonnegative=True)


def build_if_scopes(
    operation: IfOperation,
    resolver: ExprResolver,
) -> tuple[ExprResolver, ExprResolver]:
    """Build child resolvers for both branches of an if operation.

    Args:
        operation (IfOperation): Conditional operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, ExprResolver]: True-branch and false-branch
        resolvers.
    """
    true_child = resolver.child_scope(
        inner_block=_LocalBlock(operation.true_operations)
    )
    false_child = resolver.child_scope(
        inner_block=_LocalBlock(operation.false_operations)
    )
    true_child.copy_array_context()
    false_child.copy_array_context()
    return true_child, false_child


def build_for_items_scope(
    operation: ForItemsOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build a child resolver for a for-items loop.

    Args:
        operation (ForItemsOperation): For-items operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        ExprResolver: Child resolver for the loop body.
    """
    return resolver.child_scope(inner_block=_LocalBlock(operation.operations))


def resolve_for_items_cardinality(operation: ForItemsOperation) -> ResourceExpr:
    """Return the symbolic cardinality of a for-items input dictionary.

    Args:
        operation (ForItemsOperation): For-items operation.

    Returns:
        ResourceExpr: Symbol of the form ``|dict_name|``.
    """
    dict_operand = operation.operands[0]
    if hasattr(dict_operand, "is_parameter") and dict_operand.is_parameter():
        dict_name = dict_operand.parameter_name() or dict_operand.name
    else:
        dict_name = dict_operand.name
    return sp.Symbol(f"|{dict_name}|", integer=True, nonnegative=True)


def _resolve_controlled_u(
    operation: ControlledUOperation,
    resolver: ExprResolver,
) -> tuple[ResourceExpr, int]:
    """Resolve controlled-U control and target arity.

    Args:
        operation (ControlledUOperation): Controlled-U operation.
        resolver (ExprResolver): Resolver for symbolic control counts.

    Returns:
        tuple[ResourceExpr, int]: Number of controls and concrete number of
        target values.
    """
    if operation.is_symbolic_num_controls:
        controls = resolver.resolve(operation.num_controls)
    else:
        controls = _expr(cast(int, operation.num_controls))

    if isinstance(operation.block, Block):
        targets = sum(
            1 for value in operation.block.input_values if value.type.is_quantum()
        )
    else:
        target_operands = getattr(operation, "target_operands", [])
        targets = len(target_operands) if target_operands else 1
    return controls, targets


_IR_SINGLE_QUBIT_GATE_TYPES = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.S,
        GateOperationType.SDG,
        GateOperationType.T,
        GateOperationType.TDG,
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.P,
    }
)
_IR_TWO_QUBIT_GATE_TYPES = frozenset(
    {
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
        GateOperationType.CP,
        GateOperationType.RZZ,
    }
)
_IR_MULTI_QUBIT_GATE_TYPES = frozenset({GateOperationType.TOFFOLI})
_CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES = frozenset(
    {
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
        GateOperationType.CP,
        GateOperationType.RZZ,
        GateOperationType.TOFFOLI,
    }
)


def _validate_ir_gate_arity_profiles() -> None:
    """Require one and only one arity profile for every IR gate type.

    Raises:
        RuntimeError: If a ``GateOperationType`` is missing from the resource
            arity profile or appears in more than one arity class.
    """
    groups = (
        _IR_SINGLE_QUBIT_GATE_TYPES,
        _IR_TWO_QUBIT_GATE_TYPES,
        _IR_MULTI_QUBIT_GATE_TYPES,
    )
    covered = frozenset().union(*groups)
    duplicated = {
        gate_type
        for gate_type in covered
        if sum(gate_type in group for group in groups) != 1
    }
    missing = set(GateOperationType) - covered
    multi_target = _IR_TWO_QUBIT_GATE_TYPES | _IR_MULTI_QUBIT_GATE_TYPES
    missing_clean_ancilla = (
        multi_target - _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES
    )
    extraneous_clean_ancilla = (
        _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES - multi_target
    )
    if missing or duplicated or missing_clean_ancilla or extraneous_clean_ancilla:
        missing_names = sorted(gate_type.name for gate_type in missing)
        duplicated_names = sorted(gate_type.name for gate_type in duplicated)
        missing_clean_ancilla_names = sorted(
            gate_type.name for gate_type in missing_clean_ancilla
        )
        extraneous_clean_ancilla_names = sorted(
            gate_type.name for gate_type in extraneous_clean_ancilla
        )
        raise RuntimeError(
            "Resource estimation requires an exhaustive, disjoint IR gate "
            "arity profile; "
            f"missing={missing_names}, duplicated={duplicated_names}, "
            f"missing_clean_ancilla_multi_target={missing_clean_ancilla_names}, "
            f"extraneous_clean_ancilla_multi_target={extraneous_clean_ancilla_names}."
        )


_validate_ir_gate_arity_profiles()

_GATE_OPERATION_ARITY: dict[GateOperationType, int] = {
    **{gate_type: 1 for gate_type in _IR_SINGLE_QUBIT_GATE_TYPES},
    **{gate_type: 2 for gate_type in _IR_TWO_QUBIT_GATE_TYPES},
    **{gate_type: 3 for gate_type in _IR_MULTI_QUBIT_GATE_TYPES},
}

_CLIFFORD_GATE_TYPES = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.S,
        GateOperationType.SDG,
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
    }
)
_T_GATE_TYPES = frozenset({GateOperationType.T, GateOperationType.TDG})
_ROTATION_GATE_TYPES = frozenset(
    {
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.P,
        GateOperationType.CP,
        GateOperationType.RZZ,
    }
)
_CONTROLLED_CLIFFORD_GATE_TYPES = frozenset(
    {
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
    }
)


def _gate_type_names(gate_types: Iterable[GateOperationType]) -> set[str]:
    """Return canonical lowercase names for IR gate types.

    Args:
        gate_types (Iterable[GateOperationType]): IR gate types to name.

    Returns:
        set[str]: Canonical lowercase enum names.
    """
    return {gate_type.name.lower() for gate_type in gate_types}


_SYNTHETIC_SINGLE_QUBIT_GATE_NAMES = {"u", "u1", "u2", "u3"}
_SYNTHETIC_MULTI_QUBIT_GATE_NAMES = {"ccx"}
_CLIFFORD_GATES = _gate_type_names(_CLIFFORD_GATE_TYPES)
_T_GATES = _gate_type_names(_T_GATE_TYPES)
_SINGLE_QUBIT_GATES = (
    _gate_type_names(_IR_SINGLE_QUBIT_GATE_TYPES) | _SYNTHETIC_SINGLE_QUBIT_GATE_NAMES
)
_TWO_QUBIT_GATES = _gate_type_names(_IR_TWO_QUBIT_GATE_TYPES)
_ROTATION_GATES = _gate_type_names(_ROTATION_GATE_TYPES)
_MULTI_QUBIT_GATES = (
    _gate_type_names(_IR_MULTI_QUBIT_GATE_TYPES) | _SYNTHETIC_MULTI_QUBIT_GATE_NAMES
)
_GATE_BASE_QUBITS: dict[str, int] = {
    **{
        gate_type.name.lower(): arity
        for gate_type, arity in _GATE_OPERATION_ARITY.items()
    },
    "ccx": 3,
}
_CONTROLLED_CLIFFORD_GATES = _gate_type_names(_CONTROLLED_CLIFFORD_GATE_TYPES)


def _serial_depth_from_gate_resources(gates: GateResources) -> DepthResources:
    """Return a conservative serial depth for a logical gate sequence.

    Args:
        gates (GateResources): Aggregate resources for a sequence whose
            decomposition steps share controls or targets.

    Returns:
        DepthResources: Gate-family counts interpreted as serial depths.
    """
    return DepthResources(
        depth=gates.total,
        clifford_depth=gates.clifford,
        rotation_depth=gates.rotation,
        t_depth=gates.t,
        toffoli_depth=gates.toffoli,
        non_clifford_depth=gates.non_clifford,
        gate_depth=gates.total,
    )


def _clean_ancilla_sequence_estimate(
    name: str,
    gates: GateResources,
    *,
    clean_ancillas: ResourceExpr = _ZERO,
    quality: EstimateQuality = EstimateQuality.EXACT,
) -> ResourceEstimate:
    """Build one clean-ancilla Toffoli decomposition estimate.

    Args:
        name (str): Human-readable gate or decomposition name.
        gates (GateResources): Aggregate logical gate resources.
        clean_ancillas (ResourceExpr): Reusable clean-ancilla demand.
            Defaults to zero.
        quality (EstimateQuality): Count quality. Defaults to ``EXACT``.

    Returns:
        ResourceEstimate: Serial gate/depth and reusable-width estimate.
    """
    width = WidthResources(
        clean_ancilla_qubits=clean_ancillas,
        peak_qubits=clean_ancillas,
    )
    return ResourceEstimate(
        width=width,
        gates=gates,
        depth=_serial_depth_from_gate_resources(gates),
        quality=quality,
        trace=ResourceTraceNode(
            name=name,
            source_kind="clean_ancilla_toffoli",
            summary=f"gates={gates.total}, clean_ancillas={clean_ancillas}",
        ),
    )


def _clean_ancilla_primitive_estimate(name: str) -> ResourceEstimate:
    """Return one uncontrolled logical primitive estimate.

    Args:
        name (str): Lowercase primitive gate name.

    Returns:
        ResourceEstimate: One-gate logical estimate.
    """
    return _clean_ancilla_sequence_estimate(name, _classify_uncontrolled_gate(name))


def _clean_ancilla_single_control_estimate(name: str) -> ResourceEstimate:
    """Return one singly controlled logical primitive estimate.

    The selected recipe represents fixed S/T-family controls with a controlled
    phase gate.

    Args:
        name (str): Lowercase base gate name.

    Returns:
        ResourceEstimate: Singly controlled logical estimate.
    """
    controlled_name = "p" if name in {"s", "sdg", "t", "tdg"} else name
    return _clean_ancilla_sequence_estimate(
        f"c-{name}",
        _classify_controlled_gate(controlled_name, _ONE),
    )


def _clean_ancilla_generic_multi_control_estimate(
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Lower a generic gate with the clean-ancilla Toffoli recipe.

    The recipe computes the AND of ``n`` controls with ``n - 1`` clean
    ancillas, applies one singly controlled primitive, then uncomputes the
    ladder. Other decompositions or whole-body ladder sharing may be cheaper,
    so the result is an upper bound.

    Args:
        name (str): Lowercase single-target base gate name.
        controls (ResourceExpr): Number of controls, interpreted on the branch
            where it is at least two.

    Returns:
        ResourceEstimate: Clean-ancilla Toffoli estimate.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    ladder = _scale_gates(
        _classify_uncontrolled_gate("toffoli"),
        recipe.total_toffolis,
    )
    central = _clean_ancilla_single_control_estimate(name).gates
    return _clean_ancilla_sequence_estimate(
        f"mc-{name}",
        _add_gates(ladder, central),
        clean_ancillas=recipe.clean_ancillas,
        quality=EstimateQuality.CONSERVATIVE,
    )


def _select_control_count_estimate(
    controls: ResourceExpr,
    *,
    zero: ResourceEstimate,
    one: ResourceEstimate,
    two: ResourceEstimate,
    many: ResourceEstimate,
) -> ResourceEstimate:
    """Select a decomposition by a concrete or symbolic control count.

    Args:
        controls (ResourceExpr): Nonnegative control-count expression.
        zero (ResourceEstimate): Uncontrolled estimate.
        one (ResourceEstimate): Single-control estimate.
        two (ResourceEstimate): Two-control estimate.
        many (ResourceEstimate): Estimate for three or more controls.

    Returns:
        ResourceEstimate: Concrete branch or field-wise symbolic piecewise
        estimate.

    Raises:
        ValueError: If ``controls`` is a concrete negative integer.
    """
    if controls.is_number and controls.is_integer:
        value = int(controls)
        if value < 0:
            raise ValueError(f"control count must be nonnegative, got {value}.")
        if value == 0:
            return zero
        if value == 1:
            return one
        if value == 2:
            return two
        return many
    estimate = many
    estimate = two.conditional(estimate, sp.Eq(controls, 2))
    estimate = one.conditional(estimate, sp.Eq(controls, 1))
    return zero.conditional(estimate, sp.Eq(controls, 0))


def _clean_ancilla_single_target_estimate(
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a controlled primitive with a clean-ancilla Toffoli ladder.

    Args:
        name (str): Lowercase base gate name.
        controls (ResourceExpr): Surrounding control count.

    Returns:
        ResourceEstimate: Logical decomposition estimate.
    """
    zero = _clean_ancilla_primitive_estimate(name)
    one = _clean_ancilla_single_control_estimate(name)
    generic = _clean_ancilla_generic_multi_control_estimate(name, controls)
    return _select_control_count_estimate(
        controls,
        zero=zero,
        one=one,
        two=generic,
        many=generic,
    )


def _clean_ancilla_x_estimate(controls: ResourceExpr) -> ResourceEstimate:
    """Estimate an X gate with the shared multi-control fast paths.

    Args:
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceEstimate: Clean-ancilla X-family estimate.
    """
    return _select_control_count_estimate(
        controls,
        zero=_clean_ancilla_primitive_estimate("x"),
        one=_clean_ancilla_single_control_estimate("x"),
        two=_clean_ancilla_primitive_estimate("toffoli"),
        many=_clean_ancilla_generic_multi_control_estimate("x", controls),
    )


def _clean_ancilla_z_estimate(controls: ResourceExpr) -> ResourceEstimate:
    """Estimate a Z gate with the shared two-control fast path.

    Args:
        controls (ResourceExpr): Number of controls on the Z target.

    Returns:
        ResourceEstimate: Clean-ancilla Z-family estimate.
    """
    two = (
        _clean_ancilla_primitive_estimate("h")
        .seq(_clean_ancilla_primitive_estimate("toffoli"))
        .seq(_clean_ancilla_primitive_estimate("h"))
    )
    return _select_control_count_estimate(
        controls,
        zero=_clean_ancilla_primitive_estimate("z"),
        one=_clean_ancilla_single_control_estimate("z"),
        two=two,
        many=_clean_ancilla_generic_multi_control_estimate("z", controls),
    )


def _clean_ancilla_y_estimate(controls: ResourceExpr) -> ResourceEstimate:
    """Estimate a Y gate with the exact two-control conjugation.

    Args:
        controls (ResourceExpr): Number of controls on the Y target.

    Returns:
        ResourceEstimate: Clean-ancilla Y-family estimate.
    """
    two = (
        _clean_ancilla_primitive_estimate("sdg")
        .seq(_clean_ancilla_primitive_estimate("toffoli"))
        .seq(_clean_ancilla_primitive_estimate("s"))
    )
    return _select_control_count_estimate(
        controls,
        zero=_clean_ancilla_primitive_estimate("y"),
        one=_clean_ancilla_single_control_estimate("y"),
        two=two,
        many=_clean_ancilla_generic_multi_control_estimate("y", controls),
    )


def _clean_ancilla_controlled_branch(
    controls: ResourceExpr,
    uncontrolled: ResourceEstimate,
    controlled: ResourceEstimate,
) -> ResourceEstimate:
    """Select an uncontrolled estimate only when the control count is zero.

    Args:
        controls (ResourceExpr): Surrounding control count.
        uncontrolled (ResourceEstimate): Zero-control estimate.
        controlled (ResourceEstimate): Positive-control estimate.

    Returns:
        ResourceEstimate: Selected or symbolic piecewise estimate.
    """
    if controls.is_number and controls.is_integer:
        return uncontrolled if int(controls) == 0 else controlled
    return uncontrolled.conditional(controlled, sp.Eq(controls, 0))


def _estimate_clean_ancilla_named_gate(
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate one named gate after clean-ancilla controlled lowering.

    Args:
        name (str): Lowercase Qamomile gate name.
        controls (ResourceExpr): Additional surrounding controls.

    Returns:
        ResourceEstimate: Logical decomposition estimate.
    """
    if name == "ccx":
        name = "toffoli"
    if name == "x":
        return _clean_ancilla_x_estimate(controls)
    if name == "z":
        return _clean_ancilla_z_estimate(controls)
    if name == "y":
        return _clean_ancilla_y_estimate(controls)
    if name == "cx":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate("cx"),
            _clean_ancilla_x_estimate(controls + _ONE),
        )
    if name == "cz":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate("cz"),
            _clean_ancilla_z_estimate(controls + _ONE),
        )
    if name == "cp":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate("cp"),
            _clean_ancilla_single_target_estimate("p", controls + _ONE),
        )
    if name == "toffoli":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate("toffoli"),
            _clean_ancilla_x_estimate(controls + 2),
        )
    if name == "swap":
        middle = _clean_ancilla_x_estimate(controls + _ONE)
        controlled = (
            _clean_ancilla_primitive_estimate("cx")
            .seq(middle)
            .seq(_clean_ancilla_primitive_estimate("cx"))
        )
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate("swap"),
            controlled,
        )
    if name == "rzz":
        controlled = (
            _clean_ancilla_primitive_estimate("cx")
            .seq(_clean_ancilla_single_target_estimate("rz", controls))
            .seq(_clean_ancilla_primitive_estimate("cx"))
        )
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate("rzz"),
            controlled,
        )
    return _clean_ancilla_single_target_estimate(name, controls)


def _clean_ancilla_controlled_arity_envelope(
    num_qubits: int,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Bound one unknown primitive from its operand arity.

    Every resource field is maximized independently over the supported gate
    kinds of the requested arity. The resulting fields need not describe one
    common gate decomposition, but each is a valid upper bound when the gate
    name is unavailable.

    Args:
        num_qubits (int): Primitive operand arity. Supported values are one
            and two.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        ResourceEstimate: Field-wise upper bound for one primitive.

    Raises:
        ValueError: If ``num_qubits`` is not one or two.
    """
    if num_qubits == 1:
        gate_names = sorted(_SINGLE_QUBIT_GATES)
    elif num_qubits == 2:
        gate_names = sorted(_TWO_QUBIT_GATES)
    else:
        raise ValueError(
            "Clean-ancilla aggregate control projection supports only one- and "
            f"two-qubit primitives; got arity {num_qubits}."
        )

    envelope = _estimate_clean_ancilla_named_gate(gate_names[0], controls)
    for gate_name in gate_names[1:]:
        envelope = envelope.choice(
            _estimate_clean_ancilla_named_gate(gate_name, controls)
        )
    return dataclasses.replace(
        envelope,
        trace=ResourceTraceNode(
            name=f"unknown_{num_qubits}q",
            source_kind="clean_ancilla_arity_upper_bound",
            summary=f"controls={controls}",
            children=(envelope.trace,) if envelope.trace is not None else (),
        ),
    )


def _aggregate_zero_gate_residual_condition(
    estimate: ResourceEstimate,
) -> Boolean:
    """Return whether non-total transform-sensitive resources are active.

    A symbolic complete gate profile may later specialize to zero. Calls,
    queries, gate-depth declarations, or decomposition workspace can still
    make that zero-gate specialization different from an explicit empty cost.
    Every non-total gate field is included so a symbolic ``gates.total`` that
    later becomes zero cannot erase a separately declared arity or family
    profile before its guarded constraints are checked.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        Boolean: Condition under which a zero total still has declared
            transform-sensitive resources.
    """
    expressions = (
        *estimate.calls.calls_by_name.values(),
        *estimate.calls.queries_by_name.values(),
        *(
            getattr(estimate.gates, field.name)
            for field in dataclasses.fields(GateResources)
            if field.name != "total"
        ),
        estimate.depth.depth,
        estimate.depth.gate_depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
    )
    return sp.Or(*(_resource_activity_condition(_expr(value)) for value in expressions))


def _aggregate_resource_profile_constraints(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[_ResourceConstraint, ...]:
    """Retain the basic domain requirements of an aggregate resource profile.

    Every resource metric is a nonnegative integer. Gate-family fields are
    independent bounds and are not cross-classified here. Arity fields,
    however, partition ``total`` for controlled aggregate projection and
    therefore cannot exceed it.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to validate.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements guarded by a positive
            control count.

    Raises:
        ValueError: If active concrete values violate a requirement.
    """
    requirements: list[_ResourceConstraint] = []
    resources: list[tuple[str, ResourceExpr]] = [
        *(
            (f"gates.{field.name}", getattr(estimate.gates, field.name))
            for field in dataclasses.fields(GateResources)
        ),
        ("measurements.total", estimate.measurements.total),
        ("resets.total", estimate.resets.total),
        *(
            (f"depth.{field.name}", getattr(estimate.depth, field.name))
            for field in dataclasses.fields(DepthResources)
        ),
        *(
            (f"width.{field.name}", getattr(estimate.width, field.name))
            for field in dataclasses.fields(WidthResources)
        ),
        *(
            (f"calls.calls_by_name[{name!r}]", count)
            for name, count in estimate.calls.calls_by_name.items()
        ),
        *(
            (f"calls.queries_by_name[{name!r}]", count)
            for name, count in estimate.calls.queries_by_name.items()
        ),
    ]
    for label, count in resources:
        simplified = _safe_simplify(count)
        if simplified.is_nonnegative is not True or simplified.is_integer is not True:
            requirements.append(
                _ResourceConstraint(
                    expression=count,
                    minimum=0,
                    label=f"Aggregate controlled {label}",
                )
            )
    gates = estimate.gates
    arity_remainder = _safe_simplify(
        gates.total - gates.single_qubit - gates.two_qubit - gates.multi_qubit
    )
    if arity_remainder.is_nonnegative is not True:
        requirements.append(
            _ResourceConstraint(
                expression=arity_remainder,
                minimum=0,
                label="Aggregate controlled unclassified arity remainder",
                unit="gate",
            )
        )
    active_when = sp.Gt(controls, _ZERO)
    guarded = tuple(requirement.when(active_when) for requirement in requirements)
    for requirement in guarded:
        requirement.validate()
    return guarded


def _simplify_aggregate_metadata_guard(condition: sp.Basic) -> Boolean:
    """Simplify a small aggregate-profile metadata condition.

    Args:
        condition (sp.Basic): Boolean condition built from aggregate counts.

    Returns:
        Boolean: Simplified guard with contradictory zero/nonzero predicates
            removed.
    """
    simplified = _safe_simplify(cast(ResourceExpr, condition))
    return _boolean_condition(cast(sp.Basic, simplified))


def _aggregate_arity_profile_reason(
    estimate: ResourceEstimate,
) -> str | None:
    """Return why an aggregate gate profile cannot be transformed safely.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        str | None: Invalid or unavailable profile reason, or ``None`` when
            the declared aggregate fields satisfy the common transform
            contract.
    """
    gates = estimate.gates
    if _safe_simplify(gates.total) == _ZERO:
        return "the zero gate profile has no declared primitive arity to project"
    for label, count in (
        ("total", gates.total),
        ("one-qubit", gates.single_qubit),
        ("two-qubit", gates.two_qubit),
        ("three-or-more-qubit", gates.multi_qubit),
        ("Clifford", gates.clifford),
        ("rotation", gates.rotation),
        ("T", gates.t),
        ("Toffoli", gates.toffoli),
        ("non-Clifford", gates.non_clifford),
    ):
        if _safe_simplify(count).is_negative is True:
            return f"the {label} count is negative"
    arity_remainder = _safe_simplify(
        gates.total - gates.single_qubit - gates.two_qubit - gates.multi_qubit
    )
    if arity_remainder.is_negative is True:
        return "the declared arity gate counts exceed the total gate count"
    for label, count, upper_label, upper in (
        ("Clifford", gates.clifford, "total", gates.total),
        ("rotation", gates.rotation, "total", gates.total),
        (
            "T",
            gates.t,
            "possible one-qubit",
            gates.total - gates.two_qubit - gates.multi_qubit,
        ),
        (
            "Toffoli",
            gates.toffoli,
            "possible three-or-more-qubit",
            gates.total - gates.single_qubit - gates.two_qubit,
        ),
        ("non-Clifford", gates.non_clifford, "total", gates.total),
    ):
        excess = _safe_simplify(count - upper)
        if excess.is_positive is True:
            return f"the {label} count exceeds the declared {upper_label} gate count"
    if any(
        _safe_simplify(expression) != _ZERO
        for expression in (
            estimate.measurements.total,
            estimate.resets.total,
            estimate.depth.measurement_depth,
            estimate.depth.reset_depth,
        )
    ):
        return "controlled opaque costs cannot contain measurement or reset resources"
    return None


def _aggregate_arity_projection_reason(
    estimate: ResourceEstimate,
) -> str | None:
    """Return why clean-ancilla aggregate projection is unavailable.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        str | None: Ineligibility reason, or ``None`` when at least one
            one- or two-qubit gate can use the selected control recipe.
    """
    if estimate.basis is not GateBasis.LOGICAL:
        return (
            "automatic arity projection is available only in the logical "
            f"basis, not {estimate.basis.value}"
        )
    if estimate.control_decomposition is not ControlDecomposition.CLEAN_ANCILLA_TOFFOLI:
        return (
            "automatic arity projection is available only with the "
            "clean-ancilla Toffoli control decomposition, not "
            f"{estimate.control_decomposition.value}"
        )
    reason = _aggregate_arity_profile_reason(estimate)
    if reason is not None:
        return reason
    gates = estimate.gates
    if _safe_simplify(gates.single_qubit + gates.two_qubit) == _ZERO:
        return "the aggregate has no declared one- or two-qubit gate profile"
    return None


def _unprojected_aggregate_control_assumption(
    reason: str,
    controls: ResourceExpr,
) -> ResourceAssumption:
    """Describe why an aggregate controlled cost remains unchanged.

    Args:
        reason (str): Missing information or invalid profile description.
        controls (ResourceExpr): Number of requested coherent controls.

    Returns:
        ResourceAssumption: User-facing modeled-cost assumption.
    """
    return ResourceAssumption(
        message=(
            "aggregate controlled cost is unchanged because "
            f"{reason}; no primitive body is available under the active "
            "coherent controls"
        )
    )


def _aggregate_arity_projection_constraints(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
    *,
    model_label: str,
) -> tuple[_ResourceConstraint, ...]:
    """Retain unresolved gate-family compatibility requirements.

    Args:
        estimate (ResourceEstimate): Eligible aggregate estimate.
        controls (ResourceExpr): Number of added coherent controls.
        model_label (str): Human-readable projection model used in
            requirement diagnostics.

    Returns:
        tuple[_ResourceConstraint, ...]: Family requirements that become
            active only when one or more controls use the projection.
    """
    gates = estimate.gates
    requirements: list[_ResourceConstraint] = []
    for label, count, upper_label, upper in (
        ("Clifford", gates.clifford, "total", gates.total),
        ("rotation", gates.rotation, "total", gates.total),
        (
            "T",
            gates.t,
            "possible one-qubit",
            gates.total - gates.two_qubit - gates.multi_qubit,
        ),
        (
            "Toffoli",
            gates.toffoli,
            "possible three-or-more-qubit",
            gates.total - gates.single_qubit - gates.two_qubit,
        ),
        ("non-Clifford", gates.non_clifford, "total", gates.total),
    ):
        if _safe_simplify(count) == _ZERO:
            continue
        if _safe_simplify(count - upper).is_nonpositive is not True:
            requirements.append(
                _ResourceConstraint(
                    expression=upper - count,
                    minimum=0,
                    label=(
                        f"{model_label} aggregate {label} count within "
                        f"{upper_label} gate count"
                    ),
                    unit="gate",
                )
            )
    active_when = sp.Gt(controls, _ZERO)
    guarded = tuple(requirement.when(active_when) for requirement in requirements)
    for requirement in guarded:
        requirement.validate()
    return guarded


def _project_abstract_aggregate_controlled_cost(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[ResourceEstimate | None, str]:
    """Project an aggregate profile to abstract controlled primitives.

    Each source primitive remains one logical operation. Declared arity
    buckets shift by the added control count, while an undeclared arity
    remainder stays unclassified. Because aggregate costs omit gate names,
    gate-family fields use independent upper bounds over Qamomile's supported
    logical primitive families. Every projected primitive shares the coherent
    controls, so its aggregate gate depth is serial.

    Args:
        estimate (ResourceEstimate): Aggregate logical cost before adding
            coherent controls.
        controls (ResourceExpr): Positive or symbolically nonnegative number
            of added coherent controls.

    Returns:
        tuple[ResourceEstimate | None, str]: Projected abstract estimate and an
            empty reason, or ``None`` with the reason the profile cannot be
            transformed safely.
    """
    if estimate.basis is not GateBasis.LOGICAL:
        return (
            None,
            "abstract controlled primitives are available only in the logical "
            f"basis, not {estimate.basis.value}",
        )
    if estimate.control_decomposition is not ControlDecomposition.ABSTRACT:
        return (
            None,
            "abstract aggregate projection requires the abstract control "
            f"decomposition, not {estimate.control_decomposition.value}",
        )
    reason = _aggregate_arity_profile_reason(estimate)
    if reason is not None:
        return None, reason

    gates = estimate.gates
    known_arity_count = _safe_simplify(
        gates.single_qubit + gates.two_qubit + gates.multi_qubit
    )
    unclassified_count = _safe_simplify(gates.total - known_arity_count)
    one_control = sp.Eq(controls, _ONE)
    two_controls = sp.Eq(controls, sp.Integer(2))

    projected_two_qubit = _piecewise(
        gates.single_qubit,
        _ZERO,
        one_control,
    )
    projected_multi_qubit = _safe_simplify(known_arity_count - projected_two_qubit)

    # Aggregate profiles do not identify X/Y/Z, CX, or parametric gate names.
    # Bound every family independently from the largest compatible arity
    # bucket instead of pretending the fields describe one concrete gate mix.
    possible_single_qubit = _safe_simplify(
        gates.total - gates.two_qubit - gates.multi_qubit
    )
    possible_two_qubit = _safe_simplify(
        gates.total - gates.single_qubit - gates.multi_qubit
    )
    possible_one_or_two_qubit = _safe_simplify(gates.total - gates.multi_qubit)
    projected_clifford = _piecewise(
        possible_single_qubit,
        _ZERO,
        one_control,
    )
    projected_toffoli = _piecewise(
        possible_two_qubit,
        _piecewise(
            possible_single_qubit,
            _ZERO,
            two_controls,
        ),
        one_control,
    )
    projected_gates = GateResources(
        total=gates.total,
        single_qubit=_ZERO,
        two_qubit=projected_two_qubit,
        multi_qubit=projected_multi_qubit,
        clifford=projected_clifford,
        rotation=possible_one_or_two_qubit,
        t=_ZERO,
        toffoli=projected_toffoli,
        non_clifford=gates.total,
    )
    serial_depth = DepthResources(
        depth=sp.Max(estimate.depth.depth, gates.total),
        clifford_depth=projected_clifford,
        rotation_depth=possible_one_or_two_qubit,
        t_depth=_ZERO,
        toffoli_depth=projected_toffoli,
        non_clifford_depth=gates.total,
        measurement_depth=_ZERO,
        gate_depth=sp.Max(estimate.depth.gate_depth, gates.total),
        reset_depth=_ZERO,
    )
    controlled = ResourceEstimate(
        width=estimate.width,
        gates=projected_gates,
        depth=serial_depth,
        calls=estimate.calls,
        measurements=estimate.measurements,
        resets=estimate.resets,
        assumptions=estimate.assumptions,
        trace=_merge_trace(
            f"controlled_abstract_projection({controls})",
            estimate.trace,
            ResourceTraceNode(
                name="abstract_controlled_aggregate",
                source_kind="abstract_control",
                summary=f"gates={gates.total}, controls={controls}",
            ),
        ),
        derivation=estimate.derivation,
        quality=estimate.quality,
        approximation=estimate.approximation,
        basis=estimate.basis,
        control_decomposition=estimate.control_decomposition,
        precision=estimate.precision,
        _allocation_sites=estimate._allocation_sites,
        _constraints=estimate._constraints,
        _output_sizes=estimate._output_sizes,
        _input_sizes=estimate._input_sizes,
        _has_output_summary=estimate._has_output_summary,
        _dependency_keys=estimate._dependency_keys,
        _dependency_reads=estimate._dependency_reads,
        _dependency_writes=estimate._dependency_writes,
        _global_barrier_condition=estimate._global_barrier_condition,
        _measurement_taint_conditions=estimate._measurement_taint_conditions,
        _guarded_assumptions=estimate._guarded_assumptions,
        _guarded_derivations=estimate._guarded_derivations,
        _guarded_qualities=estimate._guarded_qualities,
        _guarded_approximations=estimate._guarded_approximations,
        _symbol_aliases=estimate._symbol_aliases,
    )
    common_message = (
        "controlled aggregate cost uses one abstract operation per source "
        "primitive and serializes gate depth because every operation shares "
        "the coherent controls. Gate names and original scheduling are "
        "unavailable. Gate-family fields are independent field-wise upper "
        "bounds and may not sum to total; controlled primitives are not "
        "reported as T gates"
    )
    complete_assumption = ResourceAssumption(
        message=(
            f"{common_message}. Every declared arity bucket is shifted by the "
            "added control count. The explicit aggregate cost is treated as a "
            "complete contract, including any phase-relevant work required "
            "under later coherent controls"
        )
    )
    partial_assumption = ResourceAssumption(
        message=(
            f"{common_message}; one or more source gates have unclassified "
            "arity and remain outside single_qubit, two_qubit, and multi_qubit. "
            "The transformed arity fields therefore may not sum to total"
        )
    )
    active_controls = sp.Gt(controls, _ZERO)
    active_projection = sp.And(active_controls, sp.Gt(gates.total, _ZERO))
    if unclassified_count == _ZERO:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=active_projection,
        )
    elif unclassified_count.is_positive is True:
        controlled = controlled._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=active_controls,
        )
    else:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=sp.And(
                active_projection,
                sp.Eq(unclassified_count, _ZERO),
            ),
        )._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=sp.And(
                active_controls,
                sp.Gt(unclassified_count, _ZERO),
            ),
        )
    missing_gate_profile_active = _simplify_aggregate_metadata_guard(
        sp.And(
            active_controls,
            sp.Eq(gates.total, _ZERO),
            _aggregate_zero_gate_residual_condition(estimate),
        )
    )
    controlled = controlled._with_metadata(
        assumptions=(
            _unprojected_aggregate_control_assumption(
                "the zero gate profile has no declared primitive arity to project",
                controls,
            ),
        ),
        derivation=EstimateDerivation.MODELED,
        quality=EstimateQuality.UNKNOWN,
        active_when=missing_gate_profile_active,
    )
    return controlled, ""


def _clean_ancilla_shared_aggregate_control_ladder(
    body: ResourceEstimate,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Wrap a singly controlled aggregate body in one shared AND ladder.

    The caller supplies the body cost after every modeled primitive has been
    reduced to one effective control. The outer ladder remains live while that
    body runs, so its clean workspace is added rather than reused.

    Args:
        body (ResourceEstimate): Aggregate body projected beneath one effective
            coherent control.
        controls (ResourceExpr): Original positive control count. A symbolic
            zero branch is restored by :meth:`ResourceEstimate.controlled`
            after this helper returns.

    Returns:
        ResourceEstimate: Body-wide clean-ancilla control projection with one
        compute/uncompute ladder and concurrently held clean ancillas.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    outer_clean_ancillas = recipe.clean_ancillas
    toffoli = _clean_ancilla_primitive_estimate("toffoli")
    compute = toffoli.repeat(recipe.compute_toffolis)
    ladder = toffoli.repeat(recipe.total_toffolis)
    estimate = ladder.seq(body)
    trace_children = tuple(
        node for node in (compute.trace, body.trace, compute.trace) if node is not None
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
        trace=ResourceTraceNode(
            name=f"shared_control_ladder({controls})",
            source_kind="clean_ancilla_toffoli",
            summary=(
                f"toffoli_steps={recipe.total_toffolis}, "
                f"clean_ancillas={outer_clean_ancillas}"
            ),
            children=trace_children,
        ),
        _output_sizes=body._output_sizes,
        _input_sizes=body._input_sizes,
        _has_output_summary=body._has_output_summary,
        _dependency_keys=body._dependency_keys,
        _dependency_reads=body._dependency_reads,
        _dependency_writes=body._dependency_writes,
        _symbol_aliases=body._symbol_aliases,
    )._with_metadata(quality=EstimateQuality.CONSERVATIVE)


def _project_clean_ancilla_aggregate_controlled_cost(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[ResourceEstimate | None, str]:
    """Project the known part of an aggregate arity profile through controls.

    The projection uses one shared conjunction when at least two controls
    surround at least two modeled operations. Smaller cases retain the
    per-primitive projection. Gate-name uncertainty is represented by
    field-wise one- and two-qubit envelopes. Any remaining gate count stays as
    a unit-cost opaque serial placeholder without being assigned a false
    arity. Decomposition ancillas for the known portion are added to any
    scratch width already declared by the opaque cost.

    Args:
        estimate (ResourceEstimate): Aggregate cost before adding controls.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        tuple[ResourceEstimate | None, str]: Partially projected estimate and
            an empty reason, or ``None`` with the reason projection was
            unavailable.
    """
    reason = _aggregate_arity_projection_reason(estimate)
    if reason is not None:
        return None, reason
    single = _clean_ancilla_controlled_arity_envelope(1, _ONE).repeat(
        estimate.gates.single_qubit
    )
    two = _clean_ancilla_controlled_arity_envelope(2, _ONE).repeat(
        estimate.gates.two_qubit
    )
    unresolved_count = _safe_simplify(
        estimate.gates.total - estimate.gates.single_qubit - estimate.gates.two_qubit
    )
    unresolved = ResourceEstimate(
        gates=GateResources(
            total=unresolved_count,
            multi_qubit=estimate.gates.multi_qubit,
        ),
        depth=DepthResources(
            depth=unresolved_count,
            gate_depth=unresolved_count,
        ),
        trace=ResourceTraceNode(
            name="unresolved_controlled_aggregate",
            source_kind="opaque_arity_remainder",
            summary=f"gates={unresolved_count}",
        ),
        derivation=EstimateDerivation.MODELED,
        basis=estimate.basis,
        control_decomposition=estimate.control_decomposition,
        precision=estimate.precision,
    )
    projected_body = single.seq(two).seq(unresolved)
    shared_projection = _clean_ancilla_shared_aggregate_control_ladder(
        projected_body,
        controls,
    )
    per_primitive_projection = (
        _clean_ancilla_controlled_arity_envelope(1, controls)
        .repeat(estimate.gates.single_qubit)
        .seq(
            _clean_ancilla_controlled_arity_envelope(2, controls).repeat(
                estimate.gates.two_qubit
            )
        )
        .seq(unresolved)
    )
    aggregate_profile = _clean_ancilla_aggregate_control_profile(estimate)
    use_shared_ladder = _clean_ancilla_shared_ladder_condition(
        controls,
        aggregate_profile,
    )
    projected = shared_projection.conditional(
        per_primitive_projection,
        use_shared_ladder,
    )
    has_unresolved_gates = sp.Gt(unresolved_count, _ZERO)
    projected_gates = dataclasses.replace(
        projected.gates,
        clifford=sp.Max(
            projected.gates.clifford,
            _piecewise(
                estimate.gates.clifford,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        rotation=sp.Max(
            projected.gates.rotation,
            _piecewise(
                estimate.gates.rotation,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        t=sp.Max(
            projected.gates.t,
            _piecewise(
                estimate.gates.t,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        toffoli=sp.Max(
            projected.gates.toffoli,
            _piecewise(
                estimate.gates.toffoli,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        non_clifford=sp.Max(
            projected.gates.non_clifford,
            _piecewise(
                estimate.gates.non_clifford,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
    )
    added_clean_ancillas = projected.width.clean_ancilla_qubits
    width = dataclasses.replace(
        estimate.width,
        clean_ancilla_qubits=(
            estimate.width.clean_ancilla_qubits + added_clean_ancillas
        ),
        peak_qubits=estimate.width.peak_qubits + added_clean_ancillas,
    )
    complete_assumption = ResourceAssumption(
        message=(
            "controlled aggregate cost uses the aggregate clean-ancilla "
            "Toffoli "
            "batching model: at least two modeled operations under at least "
            "two active controls share one computed AND ladder and use "
            "conservative single-control arity upper bounds. Smaller cases "
            "use conservative per-primitive arity upper bounds. "
            "Arity fields are independent field-wise upper bounds and may not "
            "sum to total. "
            "The bounds range over Qamomile's supported logical primitive "
            "families. Gate kinds and original scheduling are unavailable, so "
            "this fixed aggregate recipe may differ from a concrete engine's "
            "emission choices. Decomposition clean ancillas are counted in "
            "addition to declared opaque workspace. The explicit aggregate "
            "cost is treated as a complete contract, including any "
            "phase-relevant work required under later coherent controls"
        )
    )
    partial_assumption = ResourceAssumption(
        message=(
            "controlled aggregate cost uses the aggregate clean-ancilla "
            "Toffoli "
            "batching model: at least two modeled operations under at least "
            "two active controls share one computed AND ladder and use "
            "conservative single-control arity upper bounds. Smaller cases "
            "use conservative per-primitive arity upper bounds; "
            "one or more gates outside the supported one- and two-qubit "
            "buckets remain one modeled operation each in total and serial "
            "depth. This unresolved portion may include gates with "
            "unclassified arity; declared multi_qubit gates remain classified "
            "but are not decomposed. "
            "Their controlled decomposition and additional clean ancillas "
            "are unavailable, unclassified gates are not reported as "
            "multi_qubit. Arity fields are independent field-wise upper bounds "
            "and may not sum to total. The supported arity bounds range over "
            "Qamomile's logical primitive families, but the overall result is "
            "a model rather than a full upper bound because the remainder "
            "decomposition is unavailable. This fixed aggregate recipe may "
            "differ from a concrete engine's emission choices. Declared "
            "gate-family counts are retained only as field-wise floors"
        )
    )
    controlled = ResourceEstimate(
        width=width,
        gates=projected_gates,
        depth=_max_depth(estimate.depth, projected.depth),
        calls=estimate.calls,
        measurements=estimate.measurements,
        resets=estimate.resets,
        assumptions=estimate.assumptions,
        trace=_merge_trace(
            f"controlled_arity_projection({controls})",
            estimate.trace,
            projected.trace,
        ),
        derivation=estimate.derivation,
        quality=estimate.quality,
        approximation=estimate.approximation,
        basis=estimate.basis,
        control_decomposition=estimate.control_decomposition,
        precision=estimate.precision,
        _allocation_sites=estimate._allocation_sites,
        _constraints=(
            *estimate._constraints,
            *projected._constraints,
        ),
        _output_sizes=estimate._output_sizes,
        _input_sizes=estimate._input_sizes,
        _has_output_summary=estimate._has_output_summary,
        _dependency_keys=estimate._dependency_keys,
        _dependency_reads=estimate._dependency_reads,
        _dependency_writes=estimate._dependency_writes,
        _global_barrier_condition=estimate._global_barrier_condition,
        _measurement_taint_conditions=estimate._measurement_taint_conditions,
        _guarded_assumptions=estimate._guarded_assumptions,
        _guarded_derivations=estimate._guarded_derivations,
        _guarded_qualities=estimate._guarded_qualities,
        _guarded_approximations=estimate._guarded_approximations,
        _symbol_aliases=estimate._symbol_aliases,
    )
    active_controls = sp.Gt(controls, _ZERO)
    active_projection = sp.And(
        active_controls,
        sp.Gt(estimate.gates.total, _ZERO),
    )
    if unresolved_count == _ZERO:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=active_projection,
        )
    elif unresolved_count.is_positive is True:
        controlled = controlled._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=active_controls,
        )
    else:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=sp.And(
                active_projection,
                sp.Eq(unresolved_count, _ZERO),
            ),
        )._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=sp.And(
                active_controls,
                sp.Gt(unresolved_count, _ZERO),
            ),
        )
    known_arity_count = _safe_simplify(
        estimate.gates.single_qubit + estimate.gates.two_qubit
    )
    if known_arity_count.is_positive is not True:
        positive_total_active = _simplify_aggregate_metadata_guard(
            sp.And(
                active_controls,
                sp.Eq(known_arity_count, _ZERO),
                sp.Gt(estimate.gates.total, _ZERO),
            )
        )
        zero_gate_residual_active = _simplify_aggregate_metadata_guard(
            sp.And(
                active_controls,
                sp.Eq(known_arity_count, _ZERO),
                sp.Eq(estimate.gates.total, _ZERO),
                _aggregate_zero_gate_residual_condition(estimate),
            )
        )
        retained = (
            dataclasses.replace(
                estimate,
                trace=_wrap_trace(f"controlled({controls})", estimate.trace),
            )
            ._with_metadata(
                assumptions=(
                    _unprojected_aggregate_control_assumption(
                        "the aggregate has no declared one- or two-qubit gate profile",
                        controls,
                    ),
                ),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
                active_when=positive_total_active,
            )
            ._with_metadata(
                assumptions=(
                    _unprojected_aggregate_control_assumption(
                        "the zero gate profile has no declared primitive arity to project",
                        controls,
                    ),
                ),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
                active_when=zero_gate_residual_active,
            )
        )
        controlled = retained.conditional(
            controlled,
            sp.Eq(known_arity_count, _ZERO),
        )
        controlled = dataclasses.replace(
            controlled,
            _allocation_sites=estimate._allocation_sites,
            _output_sizes=estimate._output_sizes,
            _input_sizes=estimate._input_sizes,
            _has_output_summary=estimate._has_output_summary,
            _dependency_keys=estimate._dependency_keys,
            _dependency_reads=estimate._dependency_reads,
            _dependency_writes=estimate._dependency_writes,
            _symbol_aliases=estimate._symbol_aliases,
        )
    return controlled, ""


def _estimate_clean_ancilla_gate(
    operation: GateOperation,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a primitive through the clean-ancilla Toffoli model.

    Args:
        operation (GateOperation): Primitive gate operation.
        controls (ResourceExpr): Surrounding control count.

    Returns:
        ResourceEstimate: Logical decomposition estimate.

    Raises:
        NotImplementedError: If the operation has no registered IR arity or a
            multi-target gate lacks an explicit clean-ancilla lowering.
    """
    gate_type = operation.gate_type
    if (
        not isinstance(gate_type, GateOperationType)
        or gate_type not in _GATE_OPERATION_ARITY
    ):
        raise NotImplementedError(
            "Clean-ancilla Toffoli control decomposition is not defined for "
            f"IR gate {gate_type!r}."
        )
    arity = _GATE_OPERATION_ARITY[gate_type]
    if arity > 1 and gate_type not in _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES:
        raise NotImplementedError(
            "Clean-ancilla Toffoli control decomposition requires an explicit "
            "multi-target "
            f"lowering for IR gate {gate_type.name}."
        )
    name = gate_type.name.lower()
    return _estimate_clean_ancilla_named_gate(name, controls)


def _estimate_named_gate_in_basis(
    name: str,
    controls: ResourceExpr,
    *,
    basis: GateBasis,
    control_decomposition: ControlDecomposition,
    precision: float,
) -> ResourceEstimate:
    """Estimate a named primitive in one public gate model.

    This helper is used for decomposition gates introduced by the estimator
    itself, such as control-value brackets and Pauli-gadget basis changes.

    Args:
        name (str): Lowercase gate name.
        controls (ResourceExpr): Number of additional coherent controls.
        basis (GateBasis): Requested output basis.
        control_decomposition (ControlDecomposition): Requested coherent-control
            representation.
        precision (float): Rotation-synthesis precision for ``CLIFFORD_T``.

    Returns:
        ResourceEstimate: Gate, depth, and decomposition-ancilla resources.

    Raises:
        ValueError: If the Clifford+T basis is asked to preserve a controlled
            primitive abstractly or lacks a lowering for the named gate.
    """
    if (
        basis is GateBasis.LOGICAL
        and control_decomposition is ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
    ):
        return _estimate_clean_ancilla_named_gate(name, controls)

    normalized_name = "toffoli" if name == "ccx" else name
    if basis is GateBasis.CLIFFORD_T:
        if control_decomposition is ControlDecomposition.ABSTRACT and controls != _ZERO:
            raise ValueError(
                "Clifford+T estimation cannot preserve a controlled primitive "
                "as abstract. Select the clean-ancilla Toffoli control "
                "decomposition."
            )
        gates = _classify_clifford_t_gate(
            normalized_name,
            controls,
            precision,
        )
        clean_ancillas = _named_clifford_t_clean_ancillas(
            normalized_name,
            controls,
        )
        estimate = ResourceEstimate(
            width=WidthResources(
                clean_ancilla_qubits=clean_ancillas,
                peak_qubits=clean_ancillas,
            ),
            gates=gates,
            depth=_named_clifford_t_depth(
                normalized_name,
                controls,
                gates,
                precision,
            ),
            trace=ResourceTraceNode(
                normalized_name,
                "clifford_t_decomposition",
                summary=f"gates={gates.total}",
            ),
        )
        estimate = dataclasses.replace(
            estimate,
            basis=basis,
            control_decomposition=control_decomposition,
            precision=precision,
        )
        conservative_when = _clifford_t_conservative_condition(
            normalized_name,
            controls,
        )
        if conservative_when is not sp.false:
            estimate = estimate._with_metadata(
                quality=EstimateQuality.CONSERVATIVE,
                active_when=conservative_when,
            )
        if normalized_name in _ROTATION_GATES:
            estimate = estimate._with_metadata(
                quality=EstimateQuality.UNKNOWN,
                approximation=ApproximationStatus.APPROXIMATE,
            )
        return estimate

    gates = (
        _classify_uncontrolled_gate(normalized_name)
        if controls == _ZERO
        else _classify_controlled_gate(normalized_name, controls)
    )
    return dataclasses.replace(
        ResourceEstimate.primitive(normalized_name, gates),
        basis=basis,
        control_decomposition=control_decomposition,
        precision=None,
    )


def _named_clifford_t_clean_ancillas(
    name: str,
    surrounding_controls: ResourceExpr,
) -> ResourceExpr:
    """Return Clifford+T ancillas for an estimator-introduced named gate.

    Args:
        name (str): Lowercase gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.

    Returns:
        ResourceExpr: Peak clean-ancilla demand.
    """
    return _clifford_t_clean_ancillas_for_name(name, surrounding_controls)


def _clifford_t_clean_ancillas_for_name(
    name: str,
    surrounding_controls: ResourceExpr,
) -> ResourceExpr:
    """Return clean ancillas for one canonical Clifford+T lowering.

    Args:
        name (str): Lowercase logical gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.

    Returns:
        ResourceExpr: Reusable clean-ancilla demand.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        controls = surrounding_controls + inherent_controls[name]
        return _clifford_t_mcx_clean_ancillas(controls)
    if name in {"z", "y"}:
        return _clifford_t_mcx_clean_ancillas(surrounding_controls)
    if name == "cz":
        return _clifford_t_mcx_clean_ancillas(surrounding_controls + _ONE)
    if name == "swap":
        return _clifford_t_mcx_clean_ancillas(surrounding_controls + _ONE)
    if name == "p":
        return clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls
        ).clean_ancillas
    if name == "cp":
        recipe = clean_ancilla_toffoli_ladder_or_empty(surrounding_controls + _ONE)
        return _piecewise(
            _ZERO,
            recipe.clean_ancillas,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"s", "sdg"}:
        return clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls
        ).clean_ancillas
    if name in {"t", "tdg"}:
        return clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls + _ONE
        ).clean_ancillas
    return clean_ancilla_toffoli_ladder_or_empty(surrounding_controls).clean_ancillas


def _clifford_t_mcx_clean_ancillas(controls: ResourceExpr) -> ResourceExpr:
    """Return workspace for the clean-ancilla MCX recipe.

    Args:
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceExpr: Clean ancillas required by the logical control recipe
            after lowering its Toffoli gates to Clifford+T.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    return _resource_expr(
        sp.Piecewise(
            (_ZERO, controls <= 2),
            (recipe.clean_ancillas, True),
        )
    )


def _clifford_t_mcx_toffoli_count(controls: ResourceExpr) -> ResourceExpr:
    """Return the Toffoli count for the clean-ancilla MCX recipe.

    Args:
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceExpr: Toffoli gates in the selected logical control recipe.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    return _resource_expr(
        sp.Piecewise(
            (_ZERO, controls <= 1),
            (_ONE, sp.Eq(controls, 2)),
            (recipe.total_toffolis, True),
        )
    )


def _named_clifford_t_depth(
    name: str,
    surrounding_controls: ResourceExpr,
    gates: GateResources,
    precision: float,
) -> DepthResources:
    """Return canonical depth for an estimator-introduced Clifford+T gate.

    Args:
        name (str): Lowercase gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.
        gates (GateResources): Already-classified aggregate gate counts.
        precision (float): Rotation-synthesis precision used to classify
            ``gates``.

    Returns:
        DepthResources: Canonical critical-path depth where available,
        otherwise a conservative serial depth.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        return _clifford_t_gate_depth_for_mcx(
            surrounding_controls + inherent_controls[name]
        )
    if name == "swap":
        if surrounding_controls == _ZERO:
            return DepthResources(
                depth=sp.Integer(3),
                clifford_depth=sp.Integer(3),
                gate_depth=sp.Integer(3),
            )
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls + _ONE)
        return dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
            gate_depth=middle.gate_depth + 2,
        )
    if name in {"z", "y"}:
        if surrounding_controls == _ZERO:
            return _serial_depth_from_gate_resources(gates)
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls)
        controlled = dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
            gate_depth=middle.gate_depth + 2,
        )
        if surrounding_controls.is_number:
            return controlled
        return _conditional_depth(
            DepthResources(
                depth=_ONE,
                clifford_depth=_ONE,
                gate_depth=_ONE,
            ),
            controlled,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name == "cz":
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls + _ONE)
        return dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
            gate_depth=middle.gate_depth + 2,
        )
    if name in {"p", "cp"}:
        effective_controls = surrounding_controls + (_ONE if name == "cp" else _ZERO)
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            effective_controls
        ).total_toffolis
        rotation_t = _classify_uncontrolled_clifford_t_gate("p", precision).t
        controlled = _clifford_t_cp_depth(
            ladder_steps,
            rotation_t,
        )
        if name == "cp":
            return controlled
        uncontrolled = _serial_depth_from_gate_resources(
            _classify_uncontrolled_clifford_t_gate("p", precision)
        )
        if surrounding_controls == _ZERO:
            return uncontrolled
        if surrounding_controls.is_number:
            return controlled
        return _conditional_depth(
            uncontrolled,
            controlled,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"s", "sdg"}:
        if surrounding_controls == _ZERO:
            return _serial_depth_from_gate_resources(gates)
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls
        ).total_toffolis
        controlled = DepthResources(
            depth=15 * ladder_steps + 4,
            clifford_depth=8 * ladder_steps + 2,
            t_depth=3 * ladder_steps + 2,
            non_clifford_depth=3 * ladder_steps + 2,
            gate_depth=15 * ladder_steps + 4,
        )
        if surrounding_controls.is_number:
            return controlled
        return _conditional_depth(
            DepthResources(
                depth=_ONE,
                clifford_depth=_ONE,
                gate_depth=_ONE,
            ),
            controlled,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"t", "tdg"}:
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls + _ONE
        ).total_toffolis
        return DepthResources(
            depth=15 * ladder_steps + 1,
            clifford_depth=8 * ladder_steps,
            t_depth=3 * ladder_steps + 1,
            non_clifford_depth=3 * ladder_steps + 1,
            gate_depth=15 * ladder_steps + 1,
        )
    return _serial_depth_from_gate_resources(gates)


def _clifford_t_cp_depth(
    ladder_steps: ResourceExpr,
    rotation_t: ResourceExpr,
) -> DepthResources:
    """Return depth for a Toffoli ladder around one synthesized CP gate.

    A CP decomposition uses three axial rotations and two CX gates. The two
    same-sign rotations can occupy one parallel layer, so the central CP has
    T-depth ``2 * rotation_t`` rather than ``3 * rotation_t``.

    Args:
        ladder_steps (ResourceExpr): Number of serial Toffoli gates used to
            compute and uncompute the surrounding control conjunction.
        rotation_t (ResourceExpr): T count and T-depth of one synthesized
            axial rotation.

    Returns:
        DepthResources: Aggregate and category depth of the complete
            controlled-phase recipe.
    """
    central_depth = 2 * rotation_t + 2
    central_t_depth = 2 * rotation_t
    return DepthResources(
        depth=15 * ladder_steps + central_depth,
        clifford_depth=8 * ladder_steps + 2,
        t_depth=3 * ladder_steps + central_t_depth,
        non_clifford_depth=3 * ladder_steps + central_t_depth,
        gate_depth=15 * ladder_steps + central_depth,
    )


def _classify_gate(
    operation: GateOperation,
    *,
    num_controls: ResourceExpr | int = 0,
    basis: GateBasis = _DEFAULT_GATE_BASIS,
    precision: float = _DEFAULT_ROTATION_SYNTHESIS_PRECISION,
) -> GateResources:
    """Classify one primitive gate into logical gate resources.

    Args:
        operation (GateOperation): Primitive gate operation.
        num_controls (ResourceExpr | int): Surrounding controls. Defaults to
            zero.
        basis (GateBasis): Gate basis to report. Defaults to ``LOGICAL``.
        precision (float): Rotation-synthesis precision in ``CLIFFORD_T``
            basis. Defaults to ``1e-10``.

    Returns:
        GateResources: Resource contribution of the primitive gate.
    """
    gate_name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if gate_name == "ccx":
        gate_name = "toffoli"
    if basis is GateBasis.CLIFFORD_T:
        return _classify_clifford_t_gate(
            gate_name,
            _expr(num_controls),
            precision,
        )
    if _expr(num_controls) == 0:
        return _classify_uncontrolled_gate(gate_name)
    return _classify_controlled_gate(gate_name, _expr(num_controls))


def _gate_has_rotation(operation: GateOperation) -> bool:
    """Return whether a primitive gate carries an arbitrary rotation.

    Args:
        operation (GateOperation): Primitive gate operation.

    Returns:
        bool: Whether the gate requires approximate Clifford+T synthesis.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    return name in _ROTATION_GATES


def _clifford_t_conservative_condition(
    gate_name: str,
    num_controls: ResourceExpr,
) -> sp.Basic:
    """Return when a Clifford+T gate uses a conservative exact decomposition.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Number of surrounding coherent controls.

    Returns:
        sp.Basic: Boolean activation condition for conservative provenance.
    """
    if gate_name in _ROTATION_GATES:
        return sp.false
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if gate_name in inherent_controls:
        return _boolean_condition(sp.Gt(num_controls + inherent_controls[gate_name], 2))
    mcx_wrapper_controls = {"z": 0, "y": 0, "cz": 1, "swap": 1}
    if gate_name in mcx_wrapper_controls:
        return _boolean_condition(
            sp.Gt(num_controls + mcx_wrapper_controls[gate_name], 2)
        )
    if gate_name in {"s", "sdg"}:
        return _boolean_condition(sp.Gt(num_controls, _ONE))
    exact_controlled = {
        "t",
        "tdg",
    }
    if gate_name in exact_controlled:
        return sp.false
    return _boolean_condition(sp.Gt(num_controls, _ZERO))


def _classify_clifford_t_gate(
    gate_name: str,
    num_controls: ResourceExpr,
    precision: float,
) -> GateResources:
    """Lower one logical primitive to aggregate Clifford+T resources.

    Exact canonical decompositions are used for X-family, Pauli, SWAP, and
    fixed phases. Arbitrary uncontrolled axial rotations use the
    Ross-Selinger asymptotic cost model
    ``ceil(3 log2(1 / precision))`` T gates. This scalar formula is not a
    field-wise upper bound for a concrete synthesized sequence. A controlled primitive is
    rejected unless this estimator defines an explicit Clifford+T lowering.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Number of surrounding controls.
        precision (float): Rotation-synthesis precision.

    Returns:
        GateResources: Clifford+T aggregate counts.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if gate_name in inherent_controls:
        return _multi_controlled_x_clifford_t(
            num_controls + inherent_controls[gate_name]
        )
    uncontrolled = _classify_uncontrolled_clifford_t_gate(gate_name, precision)
    if num_controls.is_number and num_controls.is_integer and int(num_controls) == 0:
        return uncontrolled
    controlled = _classify_controlled_clifford_t_gate(
        gate_name,
        num_controls,
        precision,
    )
    if num_controls.is_number and num_controls.is_integer:
        return controlled
    return _conditional_gates(
        uncontrolled,
        controlled,
        sp.Eq(num_controls, _ZERO),
    )


def _classify_uncontrolled_clifford_t_gate(
    gate_name: str,
    precision: float,
) -> GateResources:
    """Lower an uncontrolled logical primitive to Clifford+T resources.

    Args:
        gate_name (str): Lowercase logical gate name.
        precision (float): Rotation-synthesis precision.

    Returns:
        GateResources: Aggregate resources for the canonical decomposition.
    """
    if gate_name == "swap":
        return GateResources(
            total=sp.Integer(3),
            two_qubit=sp.Integer(3),
            clifford=sp.Integer(3),
        )
    if gate_name == "cz":
        # Use the same H-CX-H canonical Clifford lowering as controlled Z.
        return GateResources(
            total=sp.Integer(3),
            single_qubit=sp.Integer(2),
            two_qubit=sp.Integer(1),
            clifford=sp.Integer(3),
        )
    rotation_t = sp.Integer(math.ceil(3 * math.log2(1 / precision)))
    rotation_multiplicity = 0
    extra_single_clifford = 0
    extra_two_clifford = 0
    if gate_name in {"rz", "p"}:
        rotation_multiplicity = 1
    elif gate_name == "rx":
        rotation_multiplicity = 1
        extra_single_clifford = 2
    elif gate_name == "ry":
        rotation_multiplicity = 1
        extra_single_clifford = 4
    elif gate_name == "cp":
        rotation_multiplicity = 3
        extra_two_clifford = 2
    elif gate_name == "rzz":
        rotation_multiplicity = 1
        extra_two_clifford = 2
    if rotation_multiplicity:
        t_count = rotation_multiplicity * rotation_t
        extra_clifford = extra_single_clifford + extra_two_clifford
        return GateResources(
            total=t_count + extra_clifford,
            single_qubit=t_count + extra_single_clifford,
            two_qubit=sp.Integer(extra_two_clifford),
            clifford=sp.Integer(extra_clifford),
            t=t_count,
            non_clifford=t_count,
        )
    return _classify_uncontrolled_gate(gate_name)


def _classify_controlled_clifford_t_gate(
    gate_name: str,
    num_controls: ResourceExpr,
    precision: float,
) -> GateResources:
    """Return the defined Clifford+T lowering for a controlled primitive.

    Multi-control lowering uses the clean-ancilla Toffoli model: clean ancillas
    compute a reusable conjunction, one singly controlled primitive is
    applied, and the conjunction is uncomputed. Specialized Pauli, SWAP, and
    phase paths avoid treating an arbitrary control arity as one opaque gate.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Positive surrounding-control count.
        precision (float): Rotation-synthesis precision.

    Returns:
        GateResources: Aggregate decomposition resources.

    Raises:
        ValueError: If no explicit controlled Clifford+T lowering is defined.
    """
    if gate_name == "swap":
        middle = _multi_controlled_x_clifford_t(num_controls + _ONE)
        return _add_gates(
            middle,
            GateResources(
                total=sp.Integer(2),
                two_qubit=sp.Integer(2),
                clifford=sp.Integer(2),
            ),
        )
    if gate_name in {"z", "y"}:
        middle = _multi_controlled_x_clifford_t(num_controls)
        return _add_gates(
            middle,
            GateResources(
                total=sp.Integer(2),
                single_qubit=sp.Integer(2),
                clifford=sp.Integer(2),
            ),
        )
    if gate_name == "cz":
        middle = _multi_controlled_x_clifford_t(num_controls + _ONE)
        return _add_gates(
            middle,
            GateResources(
                total=sp.Integer(2),
                single_qubit=sp.Integer(2),
                clifford=sp.Integer(2),
            ),
        )
    if gate_name in {"p", "cp"}:
        effective_controls = num_controls + (_ONE if gate_name == "cp" else _ZERO)
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            effective_controls
        ).total_toffolis
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            ladder_steps,
        )
        central = _classify_uncontrolled_clifford_t_gate("cp", precision)
        return _add_gates(ladder, central)
    if gate_name in {"s", "sdg"}:
        # CS = (T x T) - CX - Tdg(target) - CX. The first two T gates
        # share a layer, so this exact phase-polynomial circuit uses three
        # T gates at T-depth two and needs no clean carrier.
        effective_ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            num_controls
        ).total_toffolis
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            effective_ladder_steps,
        )
        central = GateResources(
            total=sp.Integer(5),
            single_qubit=sp.Integer(3),
            two_qubit=sp.Integer(2),
            clifford=sp.Integer(2),
            t=sp.Integer(3),
            non_clifford=sp.Integer(3),
        )
        return _add_gates(ladder, central)
    if gate_name in {"t", "tdg"}:
        # Compute the conjunction of every control and the target, phase one
        # clean carrier exactly, then uncompute it. This avoids assuming a
        # controlled-T primitive is itself a Clifford+T gate.
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            clean_ancilla_toffoli_ladder_or_empty(num_controls + _ONE).total_toffolis,
        )
        return _add_gates(ladder, _classify_uncontrolled_gate(gate_name))
    raise ValueError(
        "Clifford+T lowering is not defined for controlled gate "
        f"'{gate_name}'. Use the logical basis or provide an explicit cost "
        "model."
    )


def _multi_controlled_x_clifford_t(
    num_controls: ResourceExpr,
) -> GateResources:
    """Lower a multi-controlled X using a clean-ancilla Toffoli ladder.

    Args:
        num_controls (ResourceExpr): Number of controls on the X target.

    Returns:
        GateResources: Aggregate Clifford+T counts for the selected logical
            clean-ancilla recipe.
    """
    toffolis = _clifford_t_mcx_toffoli_count(num_controls)
    return GateResources(
        total=_resource_expr(
            sp.Piecewise(
                (_ONE, num_controls <= 1),
                (sp.Integer(15), sp.Eq(num_controls, 2)),
                (15 * toffolis + _ONE, True),
            )
        ),
        single_qubit=_resource_expr(
            sp.Piecewise(
                (_ONE, sp.Eq(num_controls, 0)),
                (_ZERO, sp.Eq(num_controls, 1)),
                (9 * toffolis, sp.Eq(num_controls, 2)),
                (9 * toffolis, True),
            )
        ),
        two_qubit=_resource_expr(
            sp.Piecewise(
                (_ZERO, sp.Eq(num_controls, 0)),
                (_ONE, sp.Eq(num_controls, 1)),
                (6 * toffolis, sp.Eq(num_controls, 2)),
                (6 * toffolis + _ONE, True),
            )
        ),
        clifford=_resource_expr(
            sp.Piecewise(
                (_ONE, num_controls <= 1),
                (8 * toffolis, sp.Eq(num_controls, 2)),
                (8 * toffolis + _ONE, True),
            )
        ),
        t=_resource_expr(
            sp.Piecewise(
                (_ZERO, num_controls <= 1),
                (7 * toffolis, True),
            )
        ),
        non_clifford=_resource_expr(
            sp.Piecewise(
                (_ZERO, num_controls <= 1),
                (7 * toffolis, True),
            )
        ),
    )


def _clifford_t_clean_ancillas(
    operation: GateOperation,
    surrounding_controls: ResourceExpr,
) -> ResourceExpr:
    """Return clean ancillas required by the selected basis decomposition.

    Args:
        operation (GateOperation): Logical primitive being lowered.
        surrounding_controls (ResourceExpr): Additional enclosing controls.

    Returns:
        ResourceExpr: Peak clean-ancilla requirement.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if name == "ccx":
        name = "toffoli"
    return _clifford_t_clean_ancillas_for_name(name, surrounding_controls)


def _clifford_t_gate_depth(
    operation: GateOperation,
    surrounding_controls: ResourceExpr,
    precision: float,
) -> DepthResources:
    """Return depth for the selected canonical Clifford+T decomposition.

    Args:
        operation (GateOperation): Logical primitive being lowered.
        surrounding_controls (ResourceExpr): Additional enclosing controls.
        precision (float): Rotation-synthesis precision.

    Returns:
        DepthResources: Conservative decomposition critical path.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if name == "ccx":
        name = "toffoli"
    gates = _classify_clifford_t_gate(name, surrounding_controls, precision)
    return _named_clifford_t_depth(
        name,
        surrounding_controls,
        gates,
        precision,
    )


def _clifford_t_gate_depth_for_mcx(
    controls: ResourceExpr,
) -> DepthResources:
    """Return the Toffoli-ladder depth for a multi-controlled X.

    Args:
        controls (ResourceExpr): Number of controls.

    Returns:
        DepthResources: Canonical ladder depth.
    """
    toffolis = _clifford_t_mcx_toffoli_count(controls)
    return DepthResources(
        depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (sp.Integer(15), sp.Eq(controls, 2)),
                (15 * toffolis + _ONE, True),
            )
        ),
        clifford_depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (sp.Integer(8), sp.Eq(controls, 2)),
                (8 * toffolis + _ONE, True),
            )
        ),
        t_depth=_resource_expr(
            sp.Piecewise(
                (_ZERO, controls <= 1),
                (3 * toffolis, True),
            )
        ),
        non_clifford_depth=_resource_expr(
            sp.Piecewise(
                (_ZERO, controls <= 1),
                (3 * toffolis, True),
            )
        ),
        gate_depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (sp.Integer(15), sp.Eq(controls, 2)),
                (15 * toffolis + _ONE, True),
            )
        ),
    )


def _classify_uncontrolled_gate(gate_name: str) -> GateResources:
    """Classify an uncontrolled primitive gate.

    Args:
        gate_name (str): Lowercase gate name.

    Returns:
        GateResources: Resource contribution of the gate.
    """
    clifford = _ONE if gate_name in _CLIFFORD_GATES else _ZERO
    rotation = _ONE if gate_name in _ROTATION_GATES else _ZERO
    t_count = _ONE if gate_name in _T_GATES else _ZERO
    multi = _ONE if gate_name in _MULTI_QUBIT_GATES else _ZERO
    return GateResources(
        total=_ONE,
        single_qubit=_ONE if gate_name in _SINGLE_QUBIT_GATES else _ZERO,
        two_qubit=_ONE if gate_name in _TWO_QUBIT_GATES else _ZERO,
        multi_qubit=multi,
        clifford=clifford,
        rotation=rotation,
        t=t_count,
        toffoli=_ONE if gate_name == "toffoli" else _ZERO,
        non_clifford=_ONE - clifford,
    )


def _classify_controlled_gate(
    gate_name: str,
    num_controls: ResourceExpr,
) -> GateResources:
    """Classify a controlled primitive gate.

    Args:
        gate_name (str): Lowercase base gate name.
        num_controls (ResourceExpr): Number of active controls.

    Returns:
        GateResources: Resource contribution of the controlled primitive.
    """
    if gate_name in _SINGLE_QUBIT_GATES:
        base_qubits = 1
    elif gate_name in _TWO_QUBIT_GATES:
        base_qubits = 2
    else:
        base_qubits = _GATE_BASE_QUBITS.get(gate_name, 1)
    total_qubits = num_controls + base_qubits
    two = cast(
        ResourceExpr, sp.Piecewise((_ONE, sp.Eq(total_qubits, 2)), (_ZERO, True))
    )
    multi = cast(ResourceExpr, sp.Piecewise((_ONE, total_qubits > 2), (_ZERO, True)))
    if gate_name in _CONTROLLED_CLIFFORD_GATES:
        clifford = cast(
            ResourceExpr,
            sp.Piecewise((_ONE, sp.Eq(num_controls, 1)), (_ZERO, True)),
        )
    else:
        clifford = _ZERO
    rotation = _ONE if gate_name in _ROTATION_GATES else _ZERO
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    effective_x_controls = num_controls + inherent_controls.get(gate_name, 0)
    is_x_family = gate_name in inherent_controls
    toffoli = (
        cast(
            ResourceExpr,
            sp.Piecewise(
                (_ONE, sp.Eq(effective_x_controls, 2)),
                (_ZERO, True),
            ),
        )
        if is_x_family
        else _ZERO
    )
    return GateResources(
        total=_ONE,
        single_qubit=_ZERO,
        two_qubit=two,
        multi_qubit=multi,
        clifford=clifford,
        rotation=rotation,
        t=_ZERO,
        toffoli=toffoli,
        non_clifford=_ONE - clifford,
    )


def _pauli_terms_share_local_basis(
    terms: Sequence[Sequence[Any]],
) -> bool:
    """Return whether every qubit uses at most one Pauli basis across terms.

    Terms drawn from one local tensor-product basis commute pairwise. This
    linear fast path covers diagonal Ising/QUBO Hamiltonians without scanning
    every term pair; mixed-basis commuting sets fall back to the exact
    pairwise anticommutation test.

    Args:
        terms (Sequence[Sequence[Any]]): Active non-identity Pauli strings.

    Returns:
        bool: Whether one consistent Pauli basis exists at every qubit index.
    """
    basis_by_qubit: dict[int, Any] = {}
    for term in terms:
        for operator in term:
            previous = basis_by_qubit.setdefault(operator.index, operator.pauli)
            if previous != operator.pauli:
                return False
    return True


def _classify_pauli_evolve_depth(
    operators: Sequence[Any],
    *,
    basis_h_layer: DepthResources,
    basis_s_layer: DepthResources,
    parity_layer: DepthResources,
    rotation: DepthResources,
) -> DepthResources:
    """Estimate the exact logical depth of one Pauli-gadget term.

    Basis changes acting on distinct qubits share a layer. The forward and
    inverse parity ladders remain sequential because neighboring CNOTs overlap,
    and the axial rotation follows the forward ladder.

    Args:
        operators (Sequence[Any]): Non-identity Pauli operators in the term.
        basis_h_layer (DepthResources): Depth of one Hadamard layer.
        basis_s_layer (DepthResources): Depth of one phase-gate layer.
        parity_layer (DepthResources): Depth of one parity-ladder CNOT.
        rotation (DepthResources): Depth of the axial rotation, including any
            surrounding controls.

    Returns:
        DepthResources: Exact logical depth of the Pauli-gadget term.
    """
    import qamomile.observable as qm_o

    has_h_basis_change = any(
        operator.pauli in (qm_o.Pauli.X, qm_o.Pauli.Y) for operator in operators
    )
    has_s_basis_change = any(operator.pauli == qm_o.Pauli.Y for operator in operators)
    depth = rotation
    if has_h_basis_change:
        depth = _add_depth(depth, _scale_depth(basis_h_layer, sp.Integer(2)))
    if has_s_basis_change:
        depth = _add_depth(depth, _scale_depth(basis_s_layer, sp.Integer(2)))
    parity_layers = sp.Integer(2 * max(0, len(operators) - 1))
    return _add_depth(depth, _scale_depth(parity_layer, parity_layers))


def _quantum_operand_width_constraints(
    attrs: Mapping[str, Any],
    operands: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    source: str,
) -> tuple[_ResourceConstraint, ...]:
    """Decode exact quantum widths from callable resource metadata.

    Operand indices refer only to quantum operands, so controls added by a
    callable transform and interleaved classical parameters do not change the
    source callable's register contract. Legacy fixed-width opaque callables
    expose only ``num_target_qubits``; when no per-operand contract exists,
    that value constrains the sum of all quantum target operands.

    Args:
        attrs (Mapping[str, Any]): Callable definition or operation attrs.
        operands (Sequence[ValueBase]): Source-call quantum operands, possibly
            mixed with classical values.
        resolver (ExprResolver): Resolver for symbolic operand dimensions.
        source (str): Callable name used in diagnostics.

    Returns:
        tuple[_ResourceConstraint, ...]: Per-operand or legacy total-width
            requirements carried by the callable.

    Raises:
        ValueError: If present resource metadata is malformed or references a
            missing quantum operand.
    """
    quantum_operands = [
        operand
        for operand in operands
        if isinstance(operand, Value) and operand.type.is_quantum()
    ]
    constraints: list[_ResourceConstraint] = []
    operand_widths = quantum_operand_widths(attrs, source=source)
    for entry in operand_widths:
        if entry.index >= len(quantum_operands):
            raise ValueError(
                f"{source} resource contract references quantum operand "
                f"{entry.index}, "
                f"but the call has only {len(quantum_operands)} quantum operands."
            )
        constraints.append(
            _ResourceConstraint(
                expression=_qubit_value_size(
                    quantum_operands[entry.index],
                    resolver,
                ),
                minimum=None,
                expected=sp.Integer(entry.width),
                label=f"{source} {entry.name} register width",
                unit="qubit",
            )
        )
    legacy_total_width = attrs.get("num_target_qubits")
    if (
        not operand_widths
        and type(legacy_total_width) is int
        and legacy_total_width > 0
    ):
        constraints.append(
            _ResourceConstraint(
                expression=sum(
                    (
                        _qubit_value_size(operand, resolver)
                        for operand in quantum_operands
                    ),
                    _ZERO,
                ),
                minimum=None,
                expected=sp.Integer(legacy_total_width),
                label=f"{source} target register width",
                unit="qubit",
            )
        )
    return tuple(constraints)


def _block_input_constraints(
    block: Block,
    resolver: ExprResolver,
) -> tuple[_ResourceConstraint, ...]:
    """Return structural constraints for typed block inputs.

    Args:
        block (Block): Block whose formal quantum dimensions are constrained.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements for quantum-array shapes
        and scalar UInt/Bit parameter domains.
    """
    constraints = [
        _ResourceConstraint(
            expression=resolver.resolve(dimension),
            minimum=0,
            label=f"Quantum input '{value.name}' dimension {position}",
            unit="element",
        )
        for value in block.input_values
        if isinstance(value, ArrayValue) and value.type.is_quantum()
        for position, dimension in enumerate(value.shape)
    ]
    values_by_name = {
        value.name: value
        for value in (*block.input_values, *block.parameters.values())
        if isinstance(value, Value)
    }
    for slot in block.param_slots:
        if slot.ndim != 0 or not isinstance(slot.type, (UIntType, BitType)):
            continue
        value = values_by_name.get(slot.name)
        if value is None:
            continue
        expression = (
            (
                sp.Integer(int(slot.bound_value))
                if isinstance(slot.bound_value, bool)
                else cast(sp.Expr, sp.sympify(slot.bound_value))
            )
            if slot.bound_value is not None
            else resolver.resolve(value)
        )
        constraints.append(
            _ResourceConstraint(
                expression=expression,
                minimum=0,
                label=f"Parameter '{slot.name}'",
            )
        )
        if isinstance(slot.type, BitType):
            constraints.append(
                _ResourceConstraint(
                    expression=_ONE - expression,
                    minimum=0,
                    label=f"Bit parameter '{slot.name}' upper bound",
                )
            )
    return tuple(constraints)


def _block_output_constraints(
    block: Block,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic = sp.true,
    proven_cache: dict[_ResourceConstraint, bool] | None = None,
) -> tuple[_ResourceConstraint, ...]:
    """Return array-access requirements carried only by block outputs.

    A returned array element need not appear in any operation operand. The
    output interface must therefore be inspected explicitly so a root return
    such as ``register[index]`` cannot bypass the same bounds validation used
    for ordinary operations and nested callable bodies.

    Args:
        block (Block): Block whose output values are inspected.
        resolver (ExprResolver): Resolver for output indices and array shapes.
        active_when (sp.Basic): Predicate under which the block output is
            reached. Defaults to true.
        proven_cache (dict[_ResourceConstraint, bool] | None): Optional cache
            of previously proved structural requirements. Defaults to
            ``None``.

    Returns:
        tuple[_ResourceConstraint, ...]: Validated requirements not already
            implied by symbolic type assumptions.

    Raises:
        ValueError: If an output access is malformed or concretely out of
            bounds.
    """
    return _validated_unproven_array_constraints(
        tuple(
            constraint.when(active_when)
            for constraint in _collect_array_value_constraints(
                block.output_values,
                resolver,
            )
        ),
        proven_cache=proven_cache,
    )


def _free_symbols(estimate: ResourceEstimate) -> set[sp.Symbol]:
    """Collect all free symbols from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        set[sp.Symbol]: Free symbols used by metrics, constraints, metadata,
            or scheduler state.
    """
    symbols: set[sp.Symbol] = set()
    for expr in _all_exprs(estimate):
        symbols.update(cast(set[sp.Symbol], sp.sympify(expr).free_symbols))
    for constraint in estimate._constraints:
        constraint_symbols = set(sp.sympify(constraint.expression).free_symbols)
        constraint_symbols.update(sp.sympify(constraint.active_when).free_symbols)
        if constraint.expected is not None:
            constraint_symbols.update(sp.sympify(constraint.expected).free_symbols)
        bound_symbols = {loop_range.symbol for loop_range in constraint.ranges}
        for loop_range in constraint.ranges:
            constraint_symbols.update(sp.sympify(loop_range.start).free_symbols)
            constraint_symbols.update(sp.sympify(loop_range.step).free_symbols)
            constraint_symbols.update(sp.sympify(loop_range.iterations).free_symbols)
        symbols.update(cast(set[sp.Symbol], constraint_symbols - bound_symbols))
    for fact in (
        *(estimate._guarded_assumptions or ()),
        *(estimate._guarded_derivations or ()),
        *(estimate._guarded_qualities or ()),
        *(estimate._guarded_approximations or ()),
    ):
        symbols.update(cast(set[sp.Symbol], sp.sympify(fact.active_when).free_symbols))
    symbols.update(
        cast(
            set[sp.Symbol],
            sp.sympify(estimate._global_barrier_condition).free_symbols,
        )
    )
    return symbols


def _trace_guard_free_symbols(
    trace: ResourceTraceNode | None,
) -> set[sp.Symbol]:
    """Collect symbols used only by explanation-node activity guards.

    Trace-only symbols do not become public substitution parameters, but an
    internal runtime outcome or unresolved carry must still be rejected if it
    would escape through :meth:`ResourceEstimate.explain` or serialization.

    Args:
        trace (ResourceTraceNode | None): Optional explanation tree root.

    Returns:
        set[sp.Symbol]: Free symbols used by trace activity guards.
    """
    symbols: set[sp.Symbol] = set()
    pending = [trace] if trace is not None else []
    while pending:
        node = pending.pop()
        symbols.update(
            cast(
                set[sp.Symbol],
                sp.sympify(node.active_when).free_symbols,
            )
        )
        pending.extend(node.children)
    return symbols


def _serialization_expressions(
    estimate: ResourceEstimate,
) -> list[sp.Basic | int | float]:
    """Return expressions sharing the estimate's public symbol namespace.

    Metric expressions, structural requirements, quantified range variables,
    guarded metadata, and explanation guards must use one identity registry.
    Otherwise two symbols that collide only across separate fields could
    receive the same public name and make the combined payload ambiguous.

    Args:
        estimate (ResourceEstimate): Estimate to serialize.

    Returns:
        list[sp.Basic | int | float]: Expressions in deterministic payload
            encounter order.
    """
    expressions: list[sp.Basic | int | float] = list(_all_exprs(estimate))
    for constraint in estimate._constraints:
        expressions.append(constraint.expression)
        expressions.append(constraint.active_when)
        if constraint.expected is not None:
            expressions.append(constraint.expected)
        for loop_range in constraint.ranges:
            expressions.extend(
                (
                    loop_range.symbol,
                    loop_range.start,
                    loop_range.step,
                    loop_range.iterations,
                )
            )
    expressions.extend(
        fact.active_when
        for fact in (
            *(estimate._guarded_assumptions or ()),
            *(estimate._guarded_derivations or ()),
            *(estimate._guarded_qualities or ()),
            *(estimate._guarded_approximations or ()),
        )
    )
    expressions.append(estimate._global_barrier_condition)
    pending_trace_nodes = [estimate.trace] if estimate.trace is not None else []
    while pending_trace_nodes:
        trace_node = pending_trace_nodes.pop()
        expressions.append(trace_node.active_when)
        pending_trace_nodes.extend(reversed(trace_node.children))
    return expressions


def _serialization_registry(estimate: ResourceEstimate) -> SymbolRegistry:
    """Build the shared public symbol registry for an estimate.

    Args:
        estimate (ResourceEstimate): Estimate whose symbols need names.

    Returns:
        SymbolRegistry: Deterministic identity-to-public-name mapping.
    """
    return SymbolRegistry.from_expressions(
        _serialization_expressions(estimate),
        estimate._symbol_aliases,
    )


def _all_exprs(estimate: ResourceEstimate) -> list[ResourceExpr]:
    """Return every symbolic expression in an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        list[ResourceExpr]: Metric expressions.
    """
    return [
        estimate.width.input_qubits,
        estimate.width.allocated_qubits,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
        estimate.width.peak_qubits,
        estimate.gates.total,
        estimate.gates.single_qubit,
        estimate.gates.two_qubit,
        estimate.gates.multi_qubit,
        estimate.gates.clifford,
        estimate.gates.rotation,
        estimate.gates.t,
        estimate.gates.toffoli,
        estimate.gates.non_clifford,
        estimate.measurements.total,
        estimate.resets.total,
        estimate.depth.depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.depth.measurement_depth,
        estimate.depth.gate_depth,
        estimate.depth.reset_depth,
        *estimate.calls.calls_by_name.values(),
        *estimate.calls.queries_by_name.values(),
        *estimate._output_sizes.values(),
        *estimate._input_sizes.values(),
    ]


def _collect_parameters(
    estimate: ResourceEstimate,
    registry: SymbolRegistry | None = None,
) -> dict[str, sp.Symbol]:
    """Collect symbolic parameters from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.
        registry (SymbolRegistry | None): Shared estimate registry. Defaults to
            a newly derived registry.

    Returns:
        dict[str, sp.Symbol]: Symbol map keyed by unique public name.
    """
    active_registry = registry or _serialization_registry(estimate)
    return {
        active_registry.name(symbol): symbol
        for symbol in sorted(
            _free_symbols(estimate),
            key=active_registry.name,
        )
    }


def _substitute_bindings(
    estimate: ResourceEstimate,
    bindings: Mapping[str, Any],
) -> ResourceEstimate:
    """Apply scalar and dictionary-cardinality bindings.

    Args:
        estimate (ResourceEstimate): Estimate to rewrite.
        bindings (Mapping[str, Any]): User bindings.

    Returns:
        ResourceEstimate: Rewritten estimate.
    """
    values: dict[str, int | float] = {}
    for key, value in bindings.items():
        if isinstance(value, dict):
            values[f"|{key}|"] = len(value)
        elif isinstance(value, (int, float)):
            values[key] = value
    if not values:
        return estimate
    applicable = {
        name: value for name, value in values.items() if name in estimate.parameters
    }
    if not applicable:
        return estimate
    return estimate.substitute(**applicable)


def _partition_estimation_inputs(
    kernel: Any,
    inputs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition one user input mapping by its role during estimation.

    Scalar and numeric-array qkernel inputs remain symbolic until after
    interpretation. Structural values such as observables and dictionaries are
    supplied while tracing. A raw ``Block`` or operation list is already
    traced: recognized structural values are supplied directly during abstract
    interpretation, while UInt/Float scalars and arrays stay symbolic until
    the compact estimate is built. Unknown names remain post-interpretation
    inputs so strict validation can report them as typos.

    Args:
        kernel (Any): QKernel, block, or operation sequence being estimated.
        inputs (Mapping[str, Any] | None): User-provided input values.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Build-time structural inputs and
        post-interpretation estimation inputs.
    """
    values = dict(inputs or {})
    input_types = getattr(kernel, "input_types", None)
    if not isinstance(input_types, dict):
        declared_inputs = _declared_ir_inputs(kernel)
        build_inputs = {
            name: value
            for name, value in values.items()
            if name in declared_inputs
            and _raw_input_requires_interpretation(declared_inputs[name])
        }
        estimation_inputs = {
            name: value for name, value in values.items() if name not in build_inputs
        }
        return build_inputs, estimation_inputs

    from qamomile.circuit.frontend.func_to_block import is_array_type
    from qamomile.circuit.frontend.handle.primitives import Qubit
    from qamomile.circuit.frontend.qkernel_inputs import is_parameterizable_type
    from qamomile.circuit.frontend.qkernel_utils import get_array_element_type

    def is_quantum_port(input_type: Any) -> bool:
        """Return whether an annotation denotes a quantum input port.

        Args:
            input_type (Any): Resolved qkernel input annotation.

        Returns:
            bool: Whether the annotation is a qubit or qubit vector.
        """
        if input_type is Qubit:
            return True
        return is_array_type(input_type) and get_array_element_type(input_type) is Qubit

    build_inputs = {
        name: value
        for name, value in values.items()
        if name in input_types
        and not is_parameterizable_type(input_types[name])
        and not is_quantum_port(input_types[name])
    }
    estimation_inputs = {
        name: value for name, value in values.items() if name not in build_inputs
    }
    return build_inputs, estimation_inputs


def _declared_ir_inputs(
    kernel: Any,
) -> dict[str, ValueBase]:
    """Collect public input values recoverable from an IR estimator target.

    A raw ``Block`` is already traced, so supplied classical/object values and
    quantum-array widths can be used directly by the abstract interpreter
    without constructing a larger circuit. Scalar quantum ports remain actual
    wires rather than estimator inputs. A raw operation sequence has no
    explicit argument manifest and therefore falls back to parameter metadata
    carried by operation operands. Unrecognized names are deliberately omitted
    so downstream strict input validation still diagnoses typos.

    Args:
        kernel (Any): Block or raw operation sequence being estimated.

    Returns:
        dict[str, ValueBase]: Public names mapped to their IR values.
    """
    inputs: dict[str, ValueBase] = {}
    if isinstance(kernel, Block):
        inputs.update(
            {
                name: value
                for name, value in zip(kernel.label_args, kernel.input_values)
                if not value.type.is_quantum() or isinstance(value, ArrayValue)
            }
        )
        inputs.update(kernel.parameters)
        operations = kernel.operations
    elif isinstance(kernel, Sequence):
        operations = kernel
    else:
        return {}

    for operation in walk_operations(operations):
        for operand in operation.all_input_values():
            parameter_name = operand.parameter_name()
            if parameter_name is not None and (
                not operand.type.is_quantum() or isinstance(operand, ArrayValue)
            ):
                inputs.setdefault(parameter_name, operand)
    return inputs


def _raw_input_requires_interpretation(value: ValueBase) -> bool:
    """Return whether a raw-IR input must be bound before interpretation.

    Numeric UInt/Float scalars and arrays stay symbolic until the compact
    estimate has been built, matching qkernel estimation and preventing a
    concrete problem size from unrolling a large region loop. Object/container
    values and quantum-array widths instead affect operation meaning or shape
    and must be available to the interpreter.

    Args:
        value (ValueBase): Declared raw-IR input value.

    Returns:
        bool: Whether the value belongs in interpreter bindings.
    """
    if isinstance(value, ArrayValue) and value.type.is_quantum():
        return True
    return not isinstance(value.type, (UIntType, FloatType))


def _root_input_binding_context(
    block_or_ops: Block | Sequence[Operation],
    bindings: Mapping[str, Any],
) -> dict[str, sp.Expr]:
    """Map concrete raw-IR inputs to the resolver's root value identities.

    Scalar values bind the matching parameter UUID directly. Array payloads
    bind only their dimension UUIDs; resource estimation does not interpret
    arbitrary classical array contents as scalar expressions. The public
    ``Block`` labels are authoritative when available, while a raw operation
    list relies on each operand's parameter metadata.

    Args:
        block_or_ops (Block | Sequence[Operation]): Raw IR estimator target.
        bindings (Mapping[str, Any]): Concrete inputs supplied for abstract
            interpretation.

    Returns:
        dict[str, sp.Expr]: IR value UUIDs mapped to concrete or symbolic SymPy
            expressions.

    Raises:
        ValueError: If a concrete array input has a rank different from its IR
            declaration.
    """
    scalar_bindings = _scalar_values(bindings)
    scalar_bindings.update(
        {
            name: cast(sp.Expr, value)
            for name, value in bindings.items()
            if isinstance(value, sp.Basic) and not value.is_number
        }
    )
    context: dict[str, sp.Expr] = {}

    def bind_value(name: str, value: ValueBase) -> None:
        """Bind one named public value into the root resolver context.

        Args:
            name (str): Public input name used in ``bindings``.
            value (ValueBase): Matching IR input or parameter operand.

        Raises:
            ValueError: If an array payload rank differs from ``value.shape``.
        """
        if name not in bindings:
            return
        if isinstance(value, ArrayValue):
            supplied = bindings[name]
            if (
                value.type.is_quantum()
                and len(value.shape) == 1
                and not isinstance(supplied, bool)
                and isinstance(supplied, numbers.Integral)
            ):
                shape = (int(supplied),)
            else:
                if (
                    value.type.is_quantum()
                    and len(value.shape) == 1
                    and isinstance(supplied, (bool, numbers.Real))
                ):
                    raise ValueError(
                        f"quantum array input '{name}' requires an integer "
                        "width or an array-like value."
                    )
                shape = _concrete_input_shape(supplied)
            if shape and len(shape) != len(value.shape):
                raise ValueError(
                    f"array input '{name}' has rank {len(shape)}, but the "
                    f"qkernel declares rank {len(value.shape)}."
                )
            for dimension, size in zip(value.shape, shape):
                if dimension.is_constant():
                    expected = dimension.get_const()
                    if not _input_values_equal(size, expected):
                        raise ValueError(
                            f"array input '{name}' dimension '{dimension.name}' "
                            f"is fixed at {expected}, but the supplied shape "
                            f"has size {size}."
                        )
                    continue
                context[dimension.uuid] = sp.Integer(size)
            return
        if name in scalar_bindings:
            context[value.uuid] = scalar_bindings[name]

    if isinstance(block_or_ops, Block):
        for name, value in zip(block_or_ops.label_args, block_or_ops.input_values):
            bind_value(name, value)
        for name, value in block_or_ops.parameters.items():
            bind_value(name, value)
        return context

    for operation in walk_operations(block_or_ops):
        for operand in operation.all_input_values():
            parameter_name = operand.parameter_name()
            if parameter_name is not None:
                bind_value(parameter_name, operand)
    return context


def _estimator_parameters(
    kernel: Any,
    kwargs: Mapping[str, Any],
) -> list[str] | None:
    """Choose symbolic classical parameters for resource estimation.

    Resource estimation is symbolic-first: unless callers provide explicit
    Every unbound parameterizable classical argument remains a symbol, including
    arguments with Python defaults. Estimation inputs are substituted only after
    interpretation.

    Args:
        kernel (Any): QKernel being built.
        kwargs (Mapping[str, Any]): Compile-time build bindings.
    Returns:
        list[str] | None: Parameter list used to build the estimator IR.
    """
    input_types = getattr(kernel, "input_types", None)
    if not isinstance(input_types, dict):
        return None
    from qamomile.circuit.frontend.qkernel_inputs import (
        is_parameterizable_type,
    )

    return [
        name
        for name, input_type in input_types.items()
        if name not in kwargs and is_parameterizable_type(input_type)
    ]


def _contract_names(
    block_or_ops: "Block | Sequence[Operation]",
) -> frozenset[str] | None:
    """Return the declared classical argument names of a built block, if any.

    Only classical parameters (``param_slots``) remain after quantum-array
    inputs have been translated to their dimension symbols. A scalar quantum
    port has fixed width one and is therefore not a valid specialization name.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.

    Returns:
        frozenset[str] | None: Classical parameter names when the input is a
        ``Block``; ``None`` for a raw operation sequence (which carries no
        user-facing input contract, so substitution stays strict there).
    """
    if not isinstance(block_or_ops, Block):
        return None
    return frozenset(slot.name for slot in block_or_ops.param_slots)


def _scalar_input_types(
    block_or_ops: "Block | Sequence[Operation]",
) -> dict[str, Any]:
    """Return recoverable scalar qkernel input types by public name.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.

    Returns:
        dict[str, Any]: Scalar classical input types, or an empty mapping when
            the input has no root interface manifest.
    """
    if not isinstance(block_or_ops, Block):
        return {}
    input_types = {
        slot.name: slot.type for slot in block_or_ops.param_slots if slot.ndim == 0
    }
    for name, value in zip(block_or_ops.label_args, block_or_ops.input_values):
        if not isinstance(value, ArrayValue) and not value.type.is_quantum():
            input_types.setdefault(name, value.type)
    for name, value in block_or_ops.parameters.items():
        input_types.setdefault(name, value.type)
    return input_types


def _expand_array_shape_inputs(
    block_or_ops: "Block | Sequence[Operation]",
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Expand supplied array shapes into their IR dimension symbols.

    A one-dimensional quantum register also accepts an integer width, avoiding
    the need to construct a dummy Python sequence solely for resource
    estimation. Once expanded, a quantum port name is removed because its
    element values are not estimator inputs; classical arrays retain their
    original value for branch and parameter specialization.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.
        inputs (Mapping[str, Any]): User-provided estimation inputs.

    Returns:
        tuple[dict[str, Any], frozenset[str]]: Inputs plus one concrete value
            for each matching array dimension symbol, and input names whose
            shape was consumed by the estimate.

    Raises:
        ValueError: If a supplied concrete array rank differs from the qkernel
            input rank.
    """
    expanded = dict(inputs)
    consumed: set[str] = set()
    if not isinstance(block_or_ops, Block):
        return expanded, frozenset()
    shape_aliases = input_shape_dimension_aliases(block_or_ops)
    dimensions_by_uuid = {
        dimension.uuid: dimension
        for value in block_or_ops.input_values
        if isinstance(value, ArrayValue)
        for dimension in value.shape
    }
    for dimension_uuid, dimension_name in shape_aliases.items():
        if dimension_name in inputs and isinstance(inputs[dimension_name], bool):
            raise ValueError(
                f"array dimension input '{dimension_name}' requires a numeric "
                "integer or symbolic expression; bool is not a dimension."
            )
        dimension = dimensions_by_uuid[dimension_uuid]
        if dimension_name in inputs and dimension.is_constant():
            expected = dimension.get_const()
            supplied = inputs[dimension_name]
            if not _input_values_equal(supplied, expected):
                raise ValueError(
                    f"array dimension '{dimension_name}' is fixed at {expected}, "
                    f"but inputs specify {supplied!r}."
                )
            expanded.pop(dimension_name, None)
            consumed.add(dimension_name)
    for name, ir_value in zip(block_or_ops.label_args, block_or_ops.input_values):
        if name not in inputs or not isinstance(ir_value, ArrayValue):
            continue
        supplied = inputs[name]
        is_quantum_array = ir_value.type.is_quantum()
        if is_quantum_array and len(ir_value.shape) == 1 and isinstance(supplied, bool):
            raise ValueError(
                f"quantum array input '{name}' requires an integer width or "
                "an array-like value; bool is not a width."
            )
        if (
            is_quantum_array
            and len(ir_value.shape) == 1
            and isinstance(supplied, numbers.Integral)
        ):
            shape = (int(supplied),)
        else:
            shape = _concrete_input_shape(supplied)
            if is_quantum_array and not shape and not _is_array_like_input(supplied):
                raise ValueError(
                    f"quantum array input '{name}' requires an integer width "
                    "or an array-like value; got "
                    f"{type(supplied).__name__} ({supplied!r})."
                )
        if len(shape) != len(ir_value.shape):
            raise ValueError(
                f"array input '{name}' has rank {len(shape)}, but the qkernel "
                f"declares rank {len(ir_value.shape)}."
            )
        consumed.add(name)
        if ir_value.type.is_quantum():
            expanded.pop(name, None)
        for dimension, size in zip(ir_value.shape, shape):
            dimension_name = shape_aliases.get(dimension.uuid)
            if dimension_name:
                if dimension.is_constant():
                    expected = dimension.get_const()
                    if not _input_values_equal(size, expected):
                        raise ValueError(
                            f"array input '{name}' dimension '{dimension_name}' "
                            f"is fixed at {expected}, but the supplied shape "
                            f"has size {size}."
                        )
                    continue
                has_existing = dimension_name in inputs
                existing = inputs.get(dimension_name)
                if has_existing and not _input_values_equal(existing, size):
                    raise ValueError(
                        f"array input '{name}' implies {dimension_name}={size}, "
                        f"but inputs also specify {dimension_name}={existing!r}."
                    )
                expanded[dimension_name] = size
                consumed.add(dimension_name)
    return expanded, frozenset(consumed)


def _input_values_equal(left: Any, right: Any) -> bool:
    """Return whether two scalar estimation inputs are provably equal.

    Args:
        left (Any): Explicit input value.
        right (Any): Shape-derived input value.

    Returns:
        bool: Whether SymPy proves the scalar values equal.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        difference = sp.sympify(left) - sp.sympify(right)
    except (TypeError, ValueError, sp.SympifyError):
        return False
    return _safe_simplify(cast(ResourceExpr, difference)) == _ZERO


def _is_array_like_input(value: Any) -> bool:
    """Return whether an estimation input exposes at least one array axis.

    Args:
        value (Any): Candidate array payload.

    Returns:
        bool: Whether ``value`` is a non-scalar array or sequence.
    """
    try:
        shape = getattr(value, "shape", None)
    except _ARRAY_PROTOCOL_ERRORS:
        return False
    if shape is not None:
        try:
            return len(shape) > 0
        except _ARRAY_PROTOCOL_ERRORS:
            return False
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _concrete_input_shape(value: Any) -> tuple[int, ...]:
    """Return the concrete shape of an array-like estimation input.

    Args:
        value (Any): Array-like user input.

    Returns:
        tuple[int, ...]: Concrete dimensions, or an empty tuple when the value
        has no discoverable array shape.

    Raises:
        ValueError: If shape dimensions are not nonnegative integers or nested
            sequences have inconsistent shapes.
    """
    return _rectangular_array_shape(value)


def _scalar_values(values: Mapping[str, Any]) -> dict[str, sp.Expr]:
    """Keep numeric scalars for branches and physical dependency resolution.

    Accepts Python and NumPy numeric scalars (anything registered as
    ``numbers.Real``, normalized via ``.item()`` when present so a ``np.int64``
    from a notebook works) and SymPy numbers. Dicts, Hamiltonians, and
    symbolic-expression substitution values are dropped: only concrete numbers
    can decide a branch or select one physical array/control index during the
    initial scheduling pass. Integer-valued numbers are represented as exact
    SymPy integers so accepted float bounds follow the same scheduling path as
    Python integers.

    Args:
        values (Mapping[str, Any]): Concrete input values.

    Returns:
        dict[str, sp.Expr]: Name -> numeric SymPy value.
    """
    out: dict[str, sp.Expr] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            out[name] = sp.Integer(int(value))
        elif isinstance(value, numbers.Real):
            # Normalize NumPy scalars (np.int64, np.float64, ...) to a Python
            # scalar before sympifying.
            scalar = value.item() if hasattr(value, "item") else value
            normalized = cast(sp.Expr, sp.sympify(scalar))
            out[name] = _canonicalize_concrete_integer(normalized)
        elif isinstance(value, sp.Basic) and value.is_number:
            out[name] = _canonicalize_concrete_integer(cast(sp.Expr, value))
    return out


def _apply_inputs(
    estimate: ResourceEstimate,
    inputs: Mapping[str, Any],
    *,
    contract_names: frozenset[str] | None = None,
    input_types: Mapping[str, Any] | None = None,
    branch_condition_names: set[str] | None = None,
    consumed_input_names: frozenset[str] | None = None,
) -> ResourceEstimate:
    """Specialize a symbolic estimate with qkernel input values.

    Unlike :func:`_substitute_bindings`, substitution values may be SymPy
    expressions (e.g. an optimal-iteration formula). Each name is classified:

    - a **free symbol** of the estimate is substituted;
    - a name used for a **branch or physical dependency decision** is silently
      accepted; the initial interpretation already accounts for it;
    - an array input whose **shape specialized dimension symbols** is accepted
      without claiming that the whole input was ignored;
    - any other **declared kernel argument** not appearing in the estimate (e.g.
      a rotation angle) is a no-op, recorded as an assumption so it stays
      auditable;
    - a name that is **none of these** is a typo and raises.

    Args:
        estimate (ResourceEstimate): Symbolic estimate to rewrite.
        inputs (Mapping[str, Any]): Input values keyed by parameter name. Values
            may be numbers or SymPy expressions.
        contract_names (frozenset[str] | None): Declared kernel argument names.
            ``None`` (raw op sequence, no contract) keeps every name strict.
        input_types (Mapping[str, Any] | None): Recoverable scalar input types
            used to distinguish Bit booleans from invalid UInt/Float booleans.
            Defaults to ``None``.
        branch_condition_names (set[str] | None): Names that participated in a
            compile-time branch or dependency-scheduling decision during
            interpretation. Defaults to ``None``.
        consumed_input_names (frozenset[str] | None): Declared inputs whose
            array shape already specialized dimension symbols. Defaults to
            ``None``.

    Returns:
        ResourceEstimate: Estimate with the inputs applied.

    Raises:
        ValueError: If an input name is neither a free symbol of the
            estimate nor a declared kernel argument, or a negative input is
            supplied for a nonnegative resource symbol.
        TypeError: If a boolean is supplied for a non-Bit scalar input, or if
            a string is supplied where a scalar value or explicit SymPy
            expression is required.
    """
    # SymPy treats same-named symbols with different assumptions as distinct, so
    # a name can map to more than one symbol object; substitute every match.
    symbols_by_name: dict[str, list[sp.Symbol]] = {}
    for symbol in _free_symbols(estimate):
        # Identity-fresh internal fallbacks are not qkernel inputs even when a
        # frontend-generated display name collides with a declared argument.
        # Users may still specialize them explicitly through
        # ``ResourceEstimate.substitute`` after estimation.
        if isinstance(symbol, sp.Dummy):
            continue
        symbols_by_name.setdefault(_symbol_display_name(symbol), []).append(symbol)
    known = contract_names or frozenset()
    referenced = set(branch_condition_names or ()) | set(consumed_input_names or ())
    declared_types = input_types or {}
    unknown = [
        name
        for name in inputs
        if name not in symbols_by_name and name not in known and name not in referenced
    ]
    if unknown:
        available = ", ".join(sorted(set(symbols_by_name) | set(known))) or "(none)"
        raise ValueError(
            f"input names {sorted(unknown)} are neither free symbols of "
            f"the estimate nor kernel arguments; available: {available}. Use "
            "the qkernel's declared input names."
        )
    subs: dict[sp.Symbol, sp.Expr] = {}
    ignored: list[str] = []
    for name, value in inputs.items():
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"resource input '{name}' requires a numeric scalar or explicit "
                f"SymPy expression, got {type(value).__name__} ({value!r})."
            )
        if isinstance(value, bool) and (
            name in declared_types and not isinstance(declared_types[name], BitType)
        ):
            raise TypeError(
                f"resource input '{name}' expects "
                f"{type(declared_types[name]).__name__}, got bool ({value!r})."
            )
        if name not in symbols_by_name:
            # Not a free symbol: either initial branch/dependency
            # interpretation already consumed it (silent), or it genuinely
            # affects nothing (recorded as an ignored no-op).
            if name not in referenced:
                ignored.append(name)
            continue
        _validate_finite_source_domain_input(
            estimate,
            symbols_by_name[name],
            value,
        )
        sympified = _normalize_resource_scalar(
            value,
            label=f"resource input '{name}'",
            allow_symbolic=True,
            allow_bool=(
                name not in declared_types or isinstance(declared_types[name], BitType)
            ),
        )
        if (
            sympified.is_number
            and any(symbol.is_integer is True for symbol in symbols_by_name[name])
            and not _is_concrete_integer(cast(sp.Expr, sympified))
        ):
            raise ValueError(
                f"Cannot apply non-integer value {value!r} to integer "
                f"resource parameter '{name}'."
            )
        if any(symbol.is_integer is True for symbol in symbols_by_name[name]):
            sympified = _canonicalize_concrete_integer(sympified)
        if sympified.is_negative is True and any(
            symbol.is_nonnegative is True for symbol in symbols_by_name[name]
        ):
            raise ValueError(
                f"Cannot apply negative value {value!r} to nonnegative "
                f"resource parameter '{name}'."
            )
        for symbol in symbols_by_name[name]:
            subs[symbol] = sympified
    # Simultaneous substitution: the only way a requested name survives is that
    # the caller's own value reintroduced it (a legitimate shift like n -> n+1),
    # never a partially-applied replacement.
    substituted = estimate._map_expr(
        lambda expr: _substitute_resource_expr(expr, subs),
        constraint_fn=lambda expr: _safe_constraint_substitute(expr, subs),
        dependency_fn=lambda expr: _substitute_resource_expr(expr, subs),
    )
    if ignored:
        note = ResourceAssumption(
            "input(s) "
            + ", ".join(repr(name) for name in sorted(ignored))
            + " do not affect any resource metric; ignored",
        )
        substituted = substituted._with_metadata(assumptions=(note,))
    return substituted


def _validate_finite_source_domain_input(
    estimate: ResourceEstimate,
    symbols: Sequence[sp.Symbol],
    value: object,
) -> None:
    """Preserve source-operation diagnostics for finite-domain inputs.

    Unary mathematical operations retain their own finite-domain constraints.
    Validate those constraints before the generic public-scalar guard so an
    invalid concrete input reports the operation whose domain was violated.
    Other structural inputs still use the generic finite-real diagnostic.

    Args:
        estimate (ResourceEstimate): Estimate carrying source-domain
            constraints.
        symbols (Sequence[sp.Symbol]): Internal symbols represented by the
            public input name.
        value (object): Candidate input value.

    Raises:
        ValueError: If a concrete value violates a retained finite-domain
            constraint.
    """
    scalar = value.item() if hasattr(value, "item") else value
    try:
        normalized = sp.sympify(scalar)
    except (TypeError, ValueError, sp.SympifyError):
        return
    if not isinstance(normalized, sp.Expr) or normalized.is_number is not True:
        return
    substitutions = {symbol: normalized for symbol in symbols}
    symbol_set = set(symbols)
    for constraint in estimate._constraints:
        if not constraint.finite or constraint.expression not in symbol_set:
            continue
        constraint.mapped(lambda expr: _safe_constraint_substitute(expr, substitutions))


def _opaque_cost_target_shapes(
    operation: InvokeOperation,
    resolver: ExprResolver,
    *,
    added_controls: int,
    declared_controls: int,
) -> dict[str, tuple[ResourceExpr, ...]]:
    """Resolve definition target shapes for an opaque-cost callback.

    Args:
        operation (InvokeOperation): Bodyless invocation being modeled.
        resolver (ExprResolver): Resolver for symbolic shapes.
        added_controls (int): Call-site control prefix excluded from the
            definition.
        declared_controls (int): Definition control prefix excluded from
            target shapes.

    Returns:
        dict[str, tuple[ResourceExpr, ...]]: Target shapes keyed by definition
        formal name. Scalar qubits use an empty tuple.
    """
    base_operands = operation.operands[added_controls:]
    signature = operation.definition.signature if operation.definition else None
    hints = signature.operands if signature is not None else []
    target_operands = base_operands[declared_controls:]
    target_hints = hints[declared_controls:]
    shapes: dict[str, tuple[ResourceExpr, ...]] = {}
    for index, operand in enumerate(target_operands):
        if not operand.type.is_quantum():
            continue
        hint = target_hints[index] if index < len(target_hints) else None
        name = hint.name if hint is not None else operand.name or f"target_{index}"
        shape = (
            tuple(resolver.resolve(dimension) for dimension in operand.shape)
            if isinstance(operand, ArrayValue)
            else ()
        )
        shapes[name] = shape
    return shapes


def _controlled_u_child_resolver(
    operation: ControlledUOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build a resolver for a controlled-U body.

    Args:
        operation (ControlledUOperation): Controlled-U operation.
        resolver (ExprResolver): Call-site resolver.

    Returns:
        ExprResolver: Resolver scoped to the controlled body.
    """
    block = operation.block
    if not isinstance(block, Block):
        return resolver.child_scope(block)

    extra: dict[str, ResourceExpr] = {}
    classical_inputs: list[tuple[Value, Value]] = []
    array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
    actual_operands = operation.body_operands
    for formal, actual in pair_block_operands(block, actual_operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and not formal.type.is_quantum()
            and not isinstance(formal, ArrayValue)
            and not isinstance(actual, ArrayValue)
        ):
            classical_inputs.append((formal, actual))
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            if not formal.type.is_quantum():
                array_inputs.append((formal, actual))
            for formal_dim, actual_dim in zip(
                formal.shape,
                actual.shape,
                strict=True,
            ):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    child = resolver.isolated_scope(
        block,
        extra,
        structural_scope=resolver.call_structural_scope(operation, block),
    )
    for formal, actual in classical_inputs:
        child.bind_classical_fact(
            formal,
            resolver.resolve_classical_fact(actual),
        )
    for formal, actual in array_inputs:
        child.bind_call_array_input(
            block,
            formal,
            resolver.snapshot_array_state(actual),
        )
    return child


def _select_case_child_resolver(
    operation: SelectOperation,
    case_block: Block,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build an independent resolver for one SELECT case body.

    Args:
        operation (SelectOperation): SELECT operation owning the case.
        case_block (Block): Case block whose formal inputs are mapped.
        resolver (ExprResolver): Resolver for the SELECT call site.

    Returns:
        ExprResolver: Resolver scoped exclusively to ``case_block`` with
        target, parameter, and array-shape formals bound to actual operands.
    """
    actual_operands = [*operation.target_operands, *operation.param_operands]
    extra: dict[str, ResourceExpr] = {}
    classical_inputs: list[tuple[Value, Value]] = []
    array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
    for formal, actual in pair_block_operands(case_block, actual_operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and not formal.type.is_quantum()
            and not isinstance(formal, ArrayValue)
            and not isinstance(actual, ArrayValue)
        ):
            classical_inputs.append((formal, actual))
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            if not formal.type.is_quantum():
                array_inputs.append((formal, actual))
            for formal_dim, actual_dim in zip(
                formal.shape,
                actual.shape,
                strict=True,
            ):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    child = resolver.isolated_scope(
        case_block,
        extra,
        structural_scope=resolver.call_structural_scope(operation, case_block),
    )
    for formal, actual in classical_inputs:
        child.bind_classical_fact(
            formal,
            resolver.resolve_classical_fact(actual),
        )
    for formal, actual in array_inputs:
        child.bind_call_array_input(
            case_block,
            formal,
            resolver.snapshot_array_state(actual),
        )
    return child


def _scalar_target_broadcast_factor(
    body: Block,
    target_operands: Sequence[Value],
    resolver: ExprResolver,
) -> ResourceExpr:
    """Return the broadcast count for a scalar body applied to one vector.

    Args:
        body (Block): Callable body with formal quantum inputs.
        target_operands (Sequence[Value]): Actual quantum targets supplied at
            the call site.
        resolver (ExprResolver): Resolver for symbolic target dimensions.

    Returns:
        ResourceExpr: Vector width when one scalar formal target is applied to
        one vector actual target, otherwise one.
    """
    if len(target_operands) != 1 or not isinstance(target_operands[0], ArrayValue):
        return _ONE
    quantum_inputs = [value for value in body.input_values if value.type.is_quantum()]
    if len(quantum_inputs) != 1 or isinstance(quantum_inputs[0], ArrayValue):
        return _ONE
    return _qubit_value_size(target_operands[0], resolver)


def _inverse_block_child_resolver(
    operation: InverseBlockOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build a resolver for an inverse implementation block.

    The inverse operation stores operands in call-site layout: quantum targets
    first, then classical parameters. The implementation block may declare
    classical formals before quantum formals, so this helper maps by formal
    type rather than by raw position.

    Args:
        operation (InverseBlockOperation): Inverse operation to resolve.
        resolver (ExprResolver): Resolver for the call site.

    Returns:
        ExprResolver: Resolver scoped to the inverse implementation.
    """
    impl = operation.implementation_block
    if not isinstance(impl, Block):
        return resolver.child_scope(impl)

    extra: dict[str, ResourceExpr] = {}
    classical_inputs: list[tuple[Value, Value]] = []
    array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
    operands = [*operation.target_qubits, *operation.parameters]
    for formal, actual in pair_block_operands(impl, operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and not formal.type.is_quantum()
            and not isinstance(formal, ArrayValue)
            and not isinstance(actual, ArrayValue)
        ):
            classical_inputs.append((formal, actual))
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            if not formal.type.is_quantum():
                array_inputs.append((formal, actual))
            for formal_dim, actual_dim in zip(
                formal.shape,
                actual.shape,
                strict=True,
            ):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    child = resolver.isolated_scope(
        impl,
        extra,
        structural_scope=resolver.call_structural_scope(operation, impl),
    )
    for formal, actual in classical_inputs:
        child.bind_classical_fact(
            formal,
            resolver.resolve_classical_fact(actual),
        )
    for formal, actual in array_inputs:
        child.bind_call_array_input(
            impl,
            formal,
            resolver.snapshot_array_state(actual),
        )
    return child
