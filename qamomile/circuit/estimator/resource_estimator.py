"""Estimate logical resources by abstractly interpreting qkernel IR."""

from __future__ import annotations

import dataclasses
import enum
import itertools
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._metrics import (
    CallResources,
    DepthResources,
    EstimateQuality,
    GateBasis,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceExpr,
    ResourceTraceNode,
    WidthResources,
    _active_assumptions,
    _active_quality,
    _add_calls,
    _add_depth,
    _add_gates,
    _add_measurements,
    _add_resets,
    _boolean_condition,
    _combine_quality,
    _conditional_calls,
    _conditional_depth,
    _conditional_gates,
    _conditional_measurements,
    _conditional_resets,
    _conditional_trace,
    _conditional_width,
    _depth_from_gate_resources,
    _expr,
    _GuardedAssumption,
    _GuardedQuality,
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
    _sum_gates,
    _sum_measurements,
    _sum_resets,
    _symbol_display_name,
    _wrap_trace,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
    UnresolvedValueError,
    input_shape_dimension_aliases,
)
from qamomile.circuit.estimator._scheduling import (
    WireKey,
    _activate_allocation_sites,
    _anonymous_allocation_width,
    _block_input_allocations,
    _branch_owner_sizes,
    _branch_width_with_static_allocations,
    _captured_quantum_allocations,
    _controlled_u_control_wire_keys,
    _count_qinit,
    _definitely_consumed_captured_allocations,
    _dependency_depth,
    _disjoint_concrete_loop_depth,
    _estimate_has_nonzero_depth,
    _invoke_quantum_output_sizes,
    _liveness_width,
    _LocalBlock,
    _loop_body_has_symbolic_quantum_index,
    _map_body_dependency_keys,
    _maximum_width_over_range,
    _merge_allocation_sites,
    _merge_dependency_keys,
    _namespace_allocation_sites,
    _operation_has_unresolved_quantum_index,
    _quantum_allocation_owner,
    _quantum_wire_keys,
    _qubit_value_size,
    _root_callable_resource_attrs,
    _root_callable_shape_inputs,
    _symbolic_disjoint_loop_depth,
    _width_with_identity_aware_allocations,
    _wire_keys_for_values,
    _with_aggregate_boundary_depth_metadata,
    _with_body_boundary_depth_metadata,
    _with_operation_output_summary,
    _without_input_allocation_sites,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
    walk_operations,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.callable import (
    CallTransform,
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
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
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
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation
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
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel
    from qamomile.circuit.ir.operation.callable import CallableRef

_ZERO = sp.Integer(0)
_ONE = sp.Integer(1)
_CONTROL_BATCH_MIN_WEIGHT = 2
_CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS = frozenset(
    {"x", "z", "cx", "cz", "toffoli", "rzz"}
)
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


class UnknownResourcePolicy(enum.Enum):
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
    """Carry the full logical resource estimate for a qkernel or block.

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
        quality (EstimateQuality): Confidence classification. Defaults to
            ``EXACT`` for body-derived logical estimates.
        basis (GateBasis): Gate basis used for the estimate. Defaults to the
            backend-neutral ``PORTABLE`` basis.
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
        _guarded_assumptions (tuple[_GuardedAssumption, ...] | None): Internal
            condition-aware assumption provenance. ``None`` initializes facts
            from the public ``assumptions`` tuple.
        _guarded_qualities (tuple[_GuardedQuality, ...] | None): Internal
            condition-aware non-exact quality provenance. ``None`` initializes
            a fact from the public ``quality`` value.
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
    quality: EstimateQuality = EstimateQuality.EXACT
    basis: GateBasis = GateBasis.PORTABLE
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
    _guarded_assumptions: tuple[_GuardedAssumption, ...] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _guarded_qualities: tuple[_GuardedQuality, ...] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _symbol_aliases: dict[sp.Symbol, str] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalize guarded metadata and derive the public parameter map."""
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
        self.assumptions = _active_assumptions(self._guarded_assumptions)
        self.quality = _active_quality(self._guarded_qualities)
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
        quality: EstimateQuality = EstimateQuality.EXACT,
        active_when: sp.Basic = sp.true,
    ) -> ResourceEstimate:
        """Append guarded assumptions and quality provenance.

        Args:
            assumptions (Sequence[ResourceAssumption]): Assumptions to append.
                Defaults to none.
            quality (EstimateQuality): Quality fact to append. ``EXACT`` adds
                no fact. Defaults to ``EXACT``.
            active_when (sp.Basic): Activation condition shared by the new
                facts. Defaults to true.

        Returns:
            ResourceEstimate: Copy with condition-aware metadata appended.
        """
        condition = _boolean_condition(active_when)
        guarded_assumptions = self._guarded_assumptions or ()
        guarded_qualities = self._guarded_qualities or ()
        return dataclasses.replace(
            self,
            _guarded_assumptions=(
                *guarded_assumptions,
                *(
                    _GuardedAssumption(condition, assumption)
                    for assumption in assumptions
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
        basis, precision = _merge_estimate_provenance(self, other)
        return ResourceEstimate(
            width=_seq_width(self.width, other.width),
            gates=_add_gates(self.gates, other.gates),
            depth=_add_depth(self.depth, other.depth),
            calls=_add_calls(self.calls, other.calls),
            measurements=_add_measurements(self.measurements, other.measurements),
            resets=_add_resets(self.resets, other.resets),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("seq", self.trace, other.trace),
            quality=_combine_quality(self.quality, other.quality),
            basis=basis,
            precision=precision,
            _allocation_sites=_merge_allocation_sites(
                self._allocation_sites,
                other._allocation_sites,
            ),
            _constraints=(*self._constraints, *other._constraints),
            _dependency_keys=_merge_dependency_keys(self, other),
            _guarded_assumptions=(
                *(self._guarded_assumptions or ()),
                *(other._guarded_assumptions or ()),
            ),
            _guarded_qualities=(
                *(self._guarded_qualities or ()),
                *(other._guarded_qualities or ()),
            ),
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
        basis, precision = _merge_estimate_provenance(self, other)
        return ResourceEstimate(
            width=_parallel_width(self.width, other.width),
            gates=_add_gates(self.gates, other.gates),
            depth=_max_depth(self.depth, other.depth),
            calls=_add_calls(self.calls, other.calls),
            measurements=_add_measurements(self.measurements, other.measurements),
            resets=_add_resets(self.resets, other.resets),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("parallel", self.trace, other.trace),
            quality=_combine_quality(self.quality, other.quality),
            basis=basis,
            precision=precision,
            _allocation_sites=_merge_allocation_sites(
                self._allocation_sites,
                other._allocation_sites,
            ),
            _constraints=(*self._constraints, *other._constraints),
            _dependency_keys=_merge_dependency_keys(self, other),
            _guarded_assumptions=(
                *(self._guarded_assumptions or ()),
                *(other._guarded_assumptions or ()),
            ),
            _guarded_qualities=(
                *(self._guarded_qualities or ()),
                *(other._guarded_qualities or ()),
            ),
        )

    def choice(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose a conservative branch choice.

        Args:
            other (ResourceEstimate): Alternative branch estimate.

        Returns:
            ResourceEstimate: Element-wise maximum of both branches.
        """
        basis, precision = _merge_estimate_provenance(self, other)
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
            quality=_combine_quality(
                EstimateQuality.UPPER_BOUND,
                _combine_quality(self.quality, other.quality),
            ),
            basis=basis,
            precision=precision,
            _allocation_sites=allocation_sites,
            _constraints=(*self._constraints, *other._constraints),
            _dependency_keys=_merge_dependency_keys(self, other),
            _guarded_assumptions=(
                *(self._guarded_assumptions or ()),
                *(other._guarded_assumptions or ()),
            ),
            _guarded_qualities=(
                *(self._guarded_qualities or ()),
                *(other._guarded_qualities or ()),
                _GuardedQuality(sp.true, EstimateQuality.UPPER_BOUND),
            ),
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
        basis, precision = _merge_estimate_provenance(self, other)
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
            quality=_combine_quality(self.quality, other.quality),
            basis=basis,
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
            _guarded_assumptions=(
                *(fact.when(condition) for fact in (self._guarded_assumptions or ())),
                *(
                    fact.when(sp.Not(condition))
                    for fact in (other._guarded_assumptions or ())
                ),
            ),
            _guarded_qualities=(
                *(fact.when(condition) for fact in (self._guarded_qualities or ())),
                *(
                    fact.when(sp.Not(condition))
                    for fact in (other._guarded_qualities or ())
                ),
            ),
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
                quality=EstimateQuality.UPPER_BOUND,
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
            quality=EstimateQuality.EXACT,
            basis=self.basis,
            precision=self.precision,
            _allocation_sites=active_sites,
            _constraints=tuple(
                constraint.when(sp.Gt(f, _ZERO)) for constraint in self._constraints
            )
            + ((factor_constraint,) if not f.is_number else ()),
            _dependency_keys=(frozenset() if f == _ZERO else self._dependency_keys),
            _guarded_assumptions=tuple(
                fact.when(active_when) for fact in (self._guarded_assumptions or ())
            ),
            _guarded_qualities=tuple(
                fact.when(active_when) for fact in (self._guarded_qualities or ())
            ),
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
                quality=EstimateQuality.UPPER_BOUND,
                active_when=active_when,
            )
        return estimate

    def controlled(self, num_controls: ResourceExpr | int) -> ResourceEstimate:
        """Estimate controls on an aggregate cost from its known arity profile.

        A portable estimate with declared one- or two-qubit gates projects
        those known primitives through an aggregate-level batching model. At
        least two modeled operations under at least two controls share one
        body-wide control ladder; smaller cases retain the per-primitive
        fallback. Gates outside the supported arity buckets remain unit-cost
        opaque placeholders in the total and serial depth. Gate names and
        scheduling are unavailable, so each projected field is an independent
        upper bound over the supported portable primitive families. The
        selected gate-name-independent two-control model may differ from a
        concrete body's gate-name-specific path. Other aggregate costs remain
        unchanged and carry an explicit assumption. Aggregate measurement or
        reset costs fail closed because no coherent transform can be inferred
        from counts alone. Body-backed qkernels are controlled by the estimator
        interpreter instead.

        Args:
            num_controls (ResourceExpr | int): Number of active controls.

        Returns:
            ResourceEstimate: Estimate with a recorded controlled assumption.

        Raises:
            ValueError: If a concrete control count or projected gate count
                is negative or non-integral, or if the estimate contains
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
        activity = _estimate_activity(self)
        _require_unitary_resource_estimate(
            self,
            transform="coherently control",
        )
        projected, reason = _project_portable_aggregate_controlled_cost(
            self,
            controls,
        )
        if projected is not None:
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
            quality=self.quality,
            basis=self.basis,
            precision=self.precision,
            _allocation_sites=self._allocation_sites,
            _constraints=self._constraints
            + ((control_constraint,) if not controls.is_number else ()),
            _output_sizes=self._output_sizes,
            _input_sizes=self._input_sizes,
            _has_output_summary=self._has_output_summary,
            _dependency_keys=self._dependency_keys,
            _guarded_assumptions=self._guarded_assumptions,
            _guarded_qualities=self._guarded_qualities,
            _symbol_aliases=self._symbol_aliases,
        )
        # A raw zero aggregate may still omit a global phase. In contrast, a
        # symbolic nonzero body that specializes to zero has proven inactivity,
        # so its transform metadata must disappear with that specialization.
        activity_guard = (
            sp.true if activity == _ZERO else _resource_activity_condition(activity)
        )
        return controlled_estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.MODELED,
            active_when=sp.And(
                sp.Gt(controls, _ZERO),
                activity_guard,
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
            quality=self.quality,
            basis=self.basis,
            precision=self.precision,
            _allocation_sites=self._allocation_sites,
            _constraints=self._constraints,
            _output_sizes=self._output_sizes,
            _input_sizes=self._input_sizes,
            _has_output_summary=self._has_output_summary,
            _dependency_keys=self._dependency_keys,
            _guarded_assumptions=self._guarded_assumptions,
            _guarded_qualities=self._guarded_qualities,
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
        step_constraint = _ResourceConstraint(
            expression=sp.Abs(step),
            minimum=1,
            label="Loop step magnitude",
        )
        step_constraint.validate()
        iterations = symbolic_iterations(start, stop, step)
        if loop_symbol not in _free_symbols(self):
            return _with_constraints(
                self.repeat(iterations),
                step_constraint,
            )
        width, allocation_sites, width_is_exact = _maximum_width_over_range(
            self.width,
            self._allocation_sites,
            loop_symbol,
            start,
            step,
            iterations,
        )
        assumptions: tuple[ResourceAssumption, ...] = ()
        quality = EstimateQuality.EXACT
        if not width_is_exact:
            assumptions = (
                *assumptions,
                ResourceAssumption(
                    "symbolic loop width uses a conservative sum bound "
                    "because its maximum could not be proven",
                    source=str(loop_symbol),
                ),
            )
            quality = _combine_quality(quality, EstimateQuality.UPPER_BOUND)
        return ResourceEstimate(
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
                    )
                    for constraint in self._constraints
                ),
            ),
            _dependency_keys=self._dependency_keys,
            _guarded_assumptions=tuple(
                (
                    fact.when(sp.Gt(iterations, _ZERO))
                    if loop_symbol not in fact.active_when.free_symbols
                    else dataclasses.replace(
                        fact,
                        active_when=sp.Gt(iterations, _ZERO),
                    )
                )
                for fact in (self._guarded_assumptions or ())
            ),
            _guarded_qualities=tuple(
                (
                    fact.when(sp.Gt(iterations, _ZERO))
                    if loop_symbol not in fact.active_when.free_symbols
                    else dataclasses.replace(
                        fact,
                        active_when=sp.Gt(iterations, _ZERO),
                    )
                )
                for fact in (self._guarded_qualities or ())
            ),
        )._with_metadata(
            assumptions=assumptions,
            quality=quality,
            active_when=sp.Gt(iterations, _ZERO),
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
            if isinstance(value, bool) or not (
                isinstance(value, numbers.Real)
                or (isinstance(value, sp.Expr) and value.is_number)
            ):
                raise TypeError(
                    f"Resource parameter '{name}' requires a concrete numeric "
                    f"scalar, got {type(value).__name__} ({value!r})."
                )
            scalar = value.item() if hasattr(value, "item") else value
            replacement = sp.sympify(scalar)
            if not isinstance(replacement, sp.Expr):
                raise TypeError(
                    f"Resource parameter '{name}' requires a concrete numeric "
                    f"SymPy expression, got {type(value).__name__} ({value!r})."
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
            if replacement.is_negative is True and parameter.is_nonnegative is True:
                raise ValueError(
                    f"Cannot substitute negative value {value!r} for "
                    f"nonnegative resource parameter '{name}'."
                )
            subs[parameter] = replacement
        return self._map_expr(
            lambda expr: _substitute_resource_expr(expr, subs),
            constraint_fn=lambda expr: _safe_constraint_substitute(expr, subs),
        )

    def simplify(self) -> ResourceEstimate:
        """Simplify all symbolic expressions.

        Returns:
            ResourceEstimate: Simplified estimate.
        """
        return self._map_expr(_safe_simplify)

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
        return f"{heading}\n{self.trace.render(2)}"

    def to_dict(self) -> dict[str, Any]:
        """Convert this estimate to a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: Stringified resource expressions.
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
            "quality": self.quality.value,
            "basis": self.basis.value,
            "precision": self.precision,
            "requirements": [
                {
                    "expression": serialize(constraint.expression),
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
    ) -> ResourceEstimate:
        """Apply a function to every symbolic expression.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.
            constraint_fn (Any | None): Optional non-clamping rewrite for
                structural constraints. Defaults to ``fn``.

        Returns:
            ResourceEstimate: Rewritten estimate.
        """
        rewrite_constraint = fn if constraint_fn is None else constraint_fn
        mapped_constraints = tuple(
            constraint.mapped(rewrite_constraint) for constraint in self._constraints
        )
        mapped_assumptions = tuple(
            mapped
            for fact in (self._guarded_assumptions or ())
            if (mapped := fact.mapped(rewrite_constraint)) is not None
        )
        mapped_qualities = tuple(
            mapped
            for fact in (self._guarded_qualities or ())
            if (mapped := fact.mapped(rewrite_constraint)) is not None
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
            _dependency_keys=self._dependency_keys,
            _guarded_assumptions=mapped_assumptions,
            _guarded_qualities=mapped_qualities,
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
            f"resources ({fields}). Use a context-aware opaque cost model "
            "for an implementation-specific transformed cost."
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
) -> tuple[GateBasis, float | None]:
    """Merge compatible basis provenance for resource algebra.

    Basis-neutral width and call estimates may compose with any gate basis.
    Two estimates carrying basis-sensitive metrics must agree on both the
    basis and, for Clifford+T, synthesis precision.

    Args:
        left (ResourceEstimate): Left operand.
        right (ResourceEstimate): Right operand.

    Returns:
        tuple[GateBasis, float | None]: Basis and precision for the result.

    Raises:
        ValueError: If basis-sensitive estimates use incompatible provenance.
    """
    left_sensitive = _estimate_has_basis_sensitive_resources(left)
    right_sensitive = _estimate_has_basis_sensitive_resources(right)
    if left_sensitive and right_sensitive:
        if left.basis is not right.basis:
            raise ValueError(
                "Cannot compose resource estimates from different gate "
                f"bases: {left.basis.value!r} and {right.basis.value!r}."
            )
        if left.basis is GateBasis.CLIFFORD_T and not _precisions_match(
            left.precision,
            right.precision,
        ):
            raise ValueError(
                "Cannot compose Clifford+T estimates with different "
                f"precisions: {left.precision!r} and {right.precision!r}."
            )
        return left.basis, left.precision
    if left_sensitive:
        return left.basis, left.precision
    if right_sensitive:
        return right.basis, right.precision
    return left.basis, left.precision


def _validate_opaque_cost_provenance(
    estimate: ResourceEstimate,
    *,
    name: str,
    basis: GateBasis,
    precision: float,
) -> None:
    """Validate a fixed or callback-produced opaque cost model.

    Args:
        estimate (ResourceEstimate): Opaque cost result to validate.
        name (str): User-facing callable name for diagnostics.
        basis (GateBasis): Basis requested by the active estimator.
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
    if basis is GateBasis.CLIFFORD_T and not _precisions_match(
        estimate.precision,
        precision,
    ):
        raise ValueError(
            f"Opaque cost for '{name}' uses precision "
            f"{estimate.precision!r}, but the estimator uses {precision!r}."
        )


@dataclasses.dataclass(frozen=True)
class OpaqueCallContext:
    """Describe one bodyless callable invocation for cost evaluation.

    A context-aware cost callback is authoritative for the complete invocation,
    including repetition, the invocation's explicit controls and activation
    value, and controls inherited from surrounding controlled regions. The
    callback should use :attr:`total_controls` when pricing coherent-control
    overhead; the estimator validates and records its result without applying
    those controls a second time. Fixed ``ResourceEstimate`` costs remain
    transform-agnostic and are projected automatically by the estimator.

    Args:
        callable_ref (CallableRef | None): Stable callable identity.
        argument_values (tuple[Value, ...]): IR operand values from the call
            site.
        operand_shapes (Mapping[str, ResourceExpr]): Resolved operand shape
            expressions keyed by operand name.
        attrs (Mapping[str, Any]): Callable attrs copied from the invocation,
            including ``control_value`` when explicit controls use a
            non-default activation state.
        loop_symbols (Mapping[str, sp.Symbol]): Loop symbols in scope.
        controls (ResourceExpr): Number of controls inherited from surrounding
            controlled regions, excluding the invocation's own explicit
            controls.
        power (ResourceExpr): Repetition power that a context-aware callback
            must include in its returned cost. Defaults to one.
        transform (CallTransform): Requested call transform.
        strategy (str | None): Selected resource strategy.
        bindings (Mapping[str, Any]): Concrete bindings supplied by the user.
        basis (GateBasis): Gate basis requested from the estimator. Defaults
            to ``PORTABLE``.
        precision (float | None): Requested Clifford+T synthesis precision, or
            ``None`` for other bases. Defaults to ``None``.
    """

    callable_ref: "CallableRef | None"
    argument_values: tuple[Value, ...]
    operand_shapes: Mapping[str, ResourceExpr]
    attrs: Mapping[str, Any]
    loop_symbols: Mapping[str, sp.Expr]
    controls: ResourceExpr = _ZERO
    power: ResourceExpr = _ONE
    transform: CallTransform = CallTransform.DIRECT
    strategy: str | None = None
    bindings: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    basis: GateBasis = GateBasis.PORTABLE
    precision: float | None = None

    @property
    def own_controls(self) -> int:
        """Return the invocation's own control-qubit count.

        Reads ``attrs["num_control_qubits"]`` when the call is controlled. This
        is the control the callable is applied *with* (e.g. Shor conditioning a
        modular multiplication on an exponent qubit), distinct from inherited
        controls in :attr:`controls`. A context-aware cost callback must include
        both sources exactly once in its returned estimate.

        Returns:
            int: Own control-qubit count, or ``0`` for a direct call.
        """
        if self.transform is not CallTransform.CONTROLLED:
            return 0
        return int(self.attrs.get("num_control_qubits", 0) or 0)

    @property
    def total_controls(self) -> ResourceExpr:
        """Return every coherent control the cost callback must price.

        Returns:
            ResourceExpr: Sum of inherited and invocation-owned controls.
        """
        return self.controls + self.own_controls


@dataclasses.dataclass
class ResourceEstimatorConfig:
    """Configure ``ResourceEstimator`` behavior.

    Args:
        strategies (dict[str, str]): Strategy overrides by callable name.
        trace (bool): Whether estimates should carry trace nodes.
        simplify (bool): Whether to simplify the final estimate.
        unknown_policy (UnknownResourcePolicy): Handling for unknown opaque
            callables.
        basis (GateBasis): Output gate basis. Defaults to ``PORTABLE``.
        precision (float): Approximation precision for rotation synthesis in
            ``CLIFFORD_T`` basis. Defaults to ``1e-10``.
    """

    strategies: dict[str, str] = dataclasses.field(default_factory=dict)
    trace: bool = False
    simplify: bool = True
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR
    basis: GateBasis = GateBasis.PORTABLE
    precision: float = 1e-10


class ResourceEstimator:
    """Estimate logical resources for qkernels and IR blocks."""

    def __init__(
        self,
        *,
        strategies: dict[str, str] | None = None,
        trace: bool = False,
        simplify: bool = True,
        unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
        basis: str | GateBasis = GateBasis.PORTABLE,
        precision: float = 1e-10,
    ) -> None:
        """Initialize a resource estimator.

        Args:
            strategies (dict[str, str] | None): Strategy overrides by callable
                name. Defaults to ``None``.
            trace (bool): Whether to keep explanation traces. Defaults to
                ``False``.
            simplify (bool): Whether to simplify the final estimate. Defaults
                to ``True``.
            unknown_policy (UnknownResourcePolicy): Handling for unknown
                bodyless callables. Defaults to ``ERROR``.
            basis (str | GateBasis): Output gate basis. Defaults to
                ``PORTABLE``.
            precision (float): Rotation-synthesis precision for
                ``CLIFFORD_T`` basis. Defaults to ``1e-10``.

        Raises:
            ValueError: If ``basis`` is unknown or ``precision`` is outside
                ``(0, 1)``.
        """
        if not 0 < precision < 1:
            raise ValueError("precision must satisfy 0 < precision < 1.")
        try:
            normalized_basis = GateBasis(basis)
        except ValueError as error:
            valid = ", ".join(member.value for member in GateBasis)
            raise ValueError(
                f"unknown gate basis {basis!r}; expected one of: {valid}"
            ) from error
        self.config = ResourceEstimatorConfig(
            strategies=dict(strategies or {}),
            trace=trace,
            simplify=simplify,
            unknown_policy=unknown_policy,
            basis=normalized_basis,
            precision=precision,
        )

    def estimate(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        *,
        inputs: dict[str, Any] | None = None,
        strategies: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        """Estimate logical resources for a qkernel, block, or operation list.

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
            ResourceEstimate: Logical resource estimate.

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
            precision=(
                config.precision if config.basis is GateBasis.CLIFFORD_T else None
            ),
        )
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
    the estimator avoids changing compiler effect validation.

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


class ResourceInterpreter:
    """Abstractly interpret IR operations into resource algebra values."""

    def __init__(
        self,
        *,
        config: ResourceEstimatorConfig,
        bindings: Mapping[str, Any],
        condition_values: Mapping[str, sp.Expr] | None = None,
    ) -> None:
        """Initialize an interpreter.

        Args:
            config (ResourceEstimatorConfig): Estimator configuration.
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
        self._measurement_derived: set[str] = set()
        # Array legality is a pure function of the fully resolved structural
        # constraint. Repeated body invocations commonly rediscover the same
        # element bounds, so retain whether each one was already proven.
        self._array_constraint_proven: dict[_ResourceConstraint, bool] = {}
        # Dependency analysis depends only on operation-list identity, not the
        # resolver used for a particular concrete loop iteration. Keep the
        # sequence strongly referenced so an ``id`` cannot be reused for an
        # unrelated transient list during this interpretation.
        self._operation_taint_cache: dict[
            int,
            tuple[list[Operation], frozenset[str]],
        ] = {}
        # Synthetic tuple carriers retain physical parent UUIDs rather than
        # Value ancestry. Keep the corresponding allocation-owner identity
        # across nested control-flow and callable evaluation scopes.
        self._allocation_owners_by_uuid: dict[str, str] = {}
        # Recursive kernels are valid when concrete inputs reach a base case.
        # Track resolved call states so only cycles or symbolically changing
        # recurrences fail early; a terminating concrete recursion has no
        # estimator-specific depth ceiling.
        self._active_call_states: dict[int, list[tuple[sp.Expr, ...]]] = {}
        # While loops have no source-level induction variable, so expose a
        # deterministic traversal-order name for each independent trip count.
        # Retaining ``|while|`` for the first loop preserves the original API.
        self._while_count = 0

    def estimate(self, block_or_ops: Block | Sequence[Operation]) -> ResourceEstimate:
        """Estimate resources for a block or operation sequence.

        Args:
            block_or_ops (Block | Sequence[Operation]): IR block or operations.

        Returns:
            ResourceEstimate: Estimated logical resources.
        """
        if isinstance(block_or_ops, Block):
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
        resolver = ExprResolver(
            context=_root_input_binding_context(block_or_ops, self.bindings)
        )
        return self.eval_operations(list(block_or_ops), resolver)

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

    def _controlled_operation_batch_weight(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> int:
        """Return the shared-control batching weight of one operation.

        The weight mirrors the portable emitter: zero means no emitted
        controlled work, one means one leaf, and two means the operation can
        justify a body-wide shared AND ladder by itself.

        Args:
            operation (Operation): Operation inside a controlled body.
            resolver (ExprResolver): Resolver for compile-time structure.

        Returns:
            int: Batching weight clamped to zero, one, or two.
        """
        if isinstance(operation, (BinOp, CompOp, CondOp, NotOp, ReturnOperation)):
            return 0
        if isinstance(operation, QInitOperation):
            return 0
        if isinstance(operation, GateOperation):
            return 1
        if isinstance(operation, ControlledUOperation):
            power = self._apply_condition_values(
                resolver.resolve(operation.power),
                record_usage=False,
            )
            return 0 if power.is_zero is True else 1
        if isinstance(operation, PauliEvolveOp):
            return _CONTROL_BATCH_MIN_WEIGHT
        if isinstance(operation, ForOperation):
            if len(operation.operands) < 2:
                return 0
            child, start, stop, step, _loop_symbol = build_for_loop_scope(
                operation,
                resolver,
            )
            iterations = symbolic_iterations(start, stop, step)
            count = self._concrete_scalar(iterations)
            if count is None:
                return 0
            if count <= 0:
                return 0
            body_weight = self._controlled_body_batch_weight(
                operation.operations,
                child,
            )
            if count == 1:
                return body_weight
            return _CONTROL_BATCH_MIN_WEIGHT if body_weight >= 1 else 0
        if isinstance(operation, IfOperation):
            taken = self._peek_branch_decision(resolver.resolve(operation.condition))
            if taken is None:
                return 1
            true_child, false_child = build_if_scopes(operation, resolver)
            return self._controlled_body_batch_weight(
                (operation.true_operations if taken else operation.false_operations),
                true_child if taken else false_child,
            )
        if isinstance(operation, InvokeOperation):
            strategy = self._strategy_for(operation)
            body = operation.effective_body(strategy=strategy)
            if not isinstance(body, Block):
                return 0
            selected_impl = operation.implementation_for(strategy=strategy)
            body_implements_transform = (
                selected_impl is not None and selected_impl.body is body
            )
            child = resolver.call_child_scope(
                operation,
                called_block=body,
                body_implements_transform=body_implements_transform,
            )
            return self._controlled_body_batch_weight(body.operations, child)
        if isinstance(operation, InverseBlockOperation):
            if not isinstance(operation.implementation_block, Block):
                return 0
            child = _inverse_block_child_resolver(operation, resolver)
            return self._controlled_body_batch_weight(
                operation.implementation_block.operations,
                child,
            )
        if isinstance(operation, SelectOperation):
            return 1
        return 1

    def _controlled_body_batch_weight(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
    ) -> int:
        """Return a controlled body's batching weight up to the threshold.

        Args:
            operations (Sequence[Operation]): Controlled body operations.
            resolver (ExprResolver): Resolver for compile-time structure.

        Returns:
            int: Sum of operation weights, capped at two.
        """
        total = 0
        for operation in operations:
            total += self._controlled_operation_batch_weight(operation, resolver)
            if total >= _CONTROL_BATCH_MIN_WEIGHT:
                return _CONTROL_BATCH_MIN_WEIGHT
        return total

    def _controlled_body_benefits_from_two_control_batch(
        self,
        operations: Sequence[Operation],
    ) -> bool:
        """Return whether the emitter batches this body at two controls.

        Args:
            operations (Sequence[Operation]): Controlled body operations.

        Returns:
            bool: Whether at least one leaf is not in the emitter's
            two-control native set.
        """
        for operation in operations:
            if isinstance(operation, GateOperation):
                name = (
                    operation.gate_type.name.lower()
                    if operation.gate_type is not None
                    else "unknown"
                )
                if name not in _CONTROL_BATCH_NATIVE_AT_TWO_CONTROLS:
                    return True
            elif isinstance(
                operation,
                (
                    ControlledUOperation,
                    ForOperation,
                    InvokeOperation,
                    InverseBlockOperation,
                    PauliEvolveOp,
                    SelectOperation,
                ),
            ):
                return True
        return False

    def _should_batch_controlled_body(
        self,
        operations: Sequence[Operation],
        resolver: ExprResolver,
        controls: ResourceExpr,
    ) -> bool:
        """Return whether portable estimation should use one shared ladder.

        Args:
            operations (Sequence[Operation]): Controlled body operations.
            resolver (ExprResolver): Resolver for compile-time structure.
            controls (ResourceExpr): Number of surrounding controls.

        Returns:
            bool: Whether the concrete body matches the emitter's batching
            policy.
        """
        if self.config.basis is not GateBasis.PORTABLE:
            return False
        if not controls.is_number or controls.is_integer is not True:
            return False
        count = int(controls)
        if count < 2:
            return False
        if (
            self._controlled_body_batch_weight(operations, resolver)
            < _CONTROL_BATCH_MIN_WEIGHT
        ):
            return False
        return count != 2 or self._controlled_body_benefits_from_two_control_batch(
            operations
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
            ResourceEstimate: Portable shared-fallback cost with concurrently
            held clean ancillas.
        """
        outer_clean_ancillas = controls - _ONE
        toffoli = _estimate_named_gate_in_basis(
            "toffoli",
            _ZERO,
            basis=self.config.basis,
            precision=self.config.precision,
        )
        ladder = toffoli.repeat(2 * outer_clean_ancillas)
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
                    source_kind="portable_fallback",
                    summary=(
                        f"toffoli_steps={2 * outer_clean_ancillas}, "
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
        )._with_metadata(quality=EstimateQuality.UPPER_BOUND)

    def eval_operations(
        self,
        operations: list[Operation],
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
        initial_allocations: Mapping[str, ResourceExpr] | None = None,
    ) -> ResourceEstimate:
        """Evaluate a list of operations sequentially.

        Args:
            operations (list[Operation]): Operations to evaluate.
            resolver (ExprResolver): Value resolver for this scope.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.
            initial_allocations (Mapping[str, ResourceExpr] | None): Live
                quantum inputs keyed by logical wire ID. Defaults to ``None``.

        Returns:
            ResourceEstimate: Sequential composition of operation resources.
        """
        control_count = _expr(controls)
        if self._should_batch_controlled_body(
            operations,
            resolver,
            control_count,
        ):
            body = self.eval_operations(
                operations,
                resolver,
                controls=_ONE,
                initial_allocations=initial_allocations,
            )
            return self._with_shared_control_ladder(body, control_count)

        previous_taint = self._measurement_derived
        cache_entry = self._operation_taint_cache.get(id(operations))
        if cache_entry is not None and cache_entry[0] is operations:
            local_taint = cache_entry[1]
        else:
            graph = build_dependency_graph(operations)
            local_taint = frozenset(
                find_measurement_derived_values(
                    graph,
                    _find_runtime_observation_results(operations),
                )
            )
            self._operation_taint_cache[id(operations)] = (operations, local_taint)
        self._measurement_derived = previous_taint | local_taint
        try:
            scheduled: list[tuple[Operation, ResourceEstimate]] = []
            seen_array_constraints: set[_ResourceConstraint] = set()
            for operation in operations:
                operation_estimate = self.eval_operation(
                    operation,
                    resolver,
                    controls=controls,
                )
                if _estimate_has_nonzero_depth(
                    operation_estimate
                ) and _operation_has_unresolved_quantum_index(
                    operation,
                    resolver,
                    scalar_values=self.condition_values,
                    used_names=self.branch_condition_names,
                ):
                    assumption = ResourceAssumption(
                        "symbolic quantum index uses an owner-wide dependency "
                        "footprint",
                        source=type(operation).__name__,
                    )
                    operation_estimate = operation_estimate._with_metadata(
                        assumptions=(assumption,),
                        quality=EstimateQuality.UPPER_BOUND,
                    )
                if not self.config.trace:
                    # Every operation estimate is freshly produced for this
                    # traversal. Dropping its optional explanation before the
                    # sequential fold avoids constructing a deep tree that the
                    # public estimator would discard at the end anyway.
                    operation_estimate.trace = None
                array_constraints = tuple(
                    constraint
                    for constraint in _operation_array_constraints(
                        operation,
                        resolver,
                        proven_cache=self._array_constraint_proven,
                    )
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
                    precision=(
                        self.config.precision
                        if self.config.basis is GateBasis.CLIFFORD_T
                        else None
                    ),
                )
                scheduled.append((operation, operation_estimate))
            estimate = ResourceEstimate.seq_all(
                operation_estimate for _, operation_estimate in scheduled
            )
            dependency_keys: set[WireKey] = set()
            for operation, operation_estimate in scheduled:
                if not _estimate_has_nonzero_depth(operation_estimate):
                    continue
                if operation_estimate._dependency_keys is not None:
                    dependency_keys.update(operation_estimate._dependency_keys)
                    continue
                reads, writes = _quantum_wire_keys(
                    operation,
                    resolver,
                    scalar_values=self.condition_values,
                    used_names=self.branch_condition_names,
                )
                dependency_keys.update(reads | writes)
            return dataclasses.replace(
                estimate,
                depth=(
                    _dependency_depth(
                        scheduled,
                        resolver,
                        measurement_derived=self._measurement_derived,
                        scalar_values=self.condition_values,
                        used_names=self.branch_condition_names,
                    )
                    if _expr(controls) == _ZERO
                    else estimate.depth
                ),
                width=_liveness_width(
                    scheduled,
                    initial_allocations or {},
                    resolver,
                    allocation_owners_by_uuid=self._allocation_owners_by_uuid,
                ),
                quality=estimate.quality,
                _allocation_sites=_without_input_allocation_sites(
                    estimate._allocation_sites,
                    operations,
                    initial_allocations or {},
                ),
                _dependency_keys=frozenset(dependency_keys),
            )
        finally:
            self._measurement_derived = previous_taint

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
            NotImplementedError: If the operation kind is not supported by
                resource estimation.
        """
        match operation:
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
                return self.eval_unary_math(operation, resolver)
            case CastOperation() | ReturnQuantumArrayElementOperation():
                return ResourceEstimate.zero()
            case HasNestedOps():
                raise NotImplementedError(
                    "Resource estimation does not support nested operation "
                    f"{type(operation).__name__}."
                )
            case _:
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
        """
        block_identity = id(block)
        call_state = tuple(
            [
                self._apply_condition_values(
                    child.resolve(formal),
                    record_usage=False,
                )
                for formal in block.input_values
                if not formal.type.is_quantum()
            ]
            + [_expr(controls)]
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
            formal.uuid
            for formal, actual in pair_block_operands(block, actual_operands)
            if actual.uuid in self._measurement_derived
        }
        previous_taint = self._measurement_derived
        self._measurement_derived = previous_taint | tainted_formals
        try:
            return self.eval_operations(
                block.operations,
                child,
                controls=controls,
                initial_allocations=_block_input_allocations(block, child),
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
            self._measurement_derived = previous_taint
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
        """
        if self.config.basis is GateBasis.PORTABLE:
            return _estimate_portable_gate(operation, _expr(controls))

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
            upper_bound_when = _clifford_t_upper_bound_condition(
                name,
                _expr(controls),
            )
            if upper_bound_when is not sp.false:
                return estimate._with_metadata(
                    quality=EstimateQuality.UPPER_BOUND,
                    active_when=upper_bound_when,
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
        phase = resolver.resolve(operation.phase)
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
                    # The shared portable emitter preserves every nontrivial
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
        return bracketed.conditional(estimate, sp.Gt(active, _ZERO))

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
            quality=EstimateQuality.MODELED,
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
        """
        if len(operation.operands) < 2:
            return ResourceEstimate.zero("empty_for")
        child, start, stop, step, loop_symbol = build_for_loop_scope(
            operation,
            resolver,
        )
        iterations = symbolic_iterations(start, stop, step)
        if operation.region_args:
            estimate = self._eval_region_for(
                operation,
                resolver,
                start=start,
                stop=stop,
                step=step,
                loop_symbol=loop_symbol,
                controls=controls,
            )
        else:
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
            estimate = inner.sum_over(loop_symbol, start, stop, step)
            parallel_depth: DepthResources | None = None
            if _expr(controls) == _ZERO:
                parallel_depth = _disjoint_concrete_loop_depth(
                    operation,
                    resolver,
                    inner.depth,
                    start=start,
                    stop=stop,
                    step=step,
                    loop_symbol=loop_symbol,
                    clean_ancillas=inner.width.clean_ancilla_qubits,
                    scalar_values=self.condition_values,
                    used_names=self.branch_condition_names,
                )
                if parallel_depth is None:
                    parallel_depth = _symbolic_disjoint_loop_depth(
                        operation,
                        child,
                        inner.depth,
                        loop_symbol=loop_symbol,
                        iterations=iterations,
                        clean_ancillas=inner.width.clean_ancilla_qubits,
                        scalar_values=self.condition_values,
                        used_names=self.branch_condition_names,
                    )
            if parallel_depth is not None:
                estimate = dataclasses.replace(estimate, depth=parallel_depth)
            elif _expr(controls) == _ZERO and _loop_body_has_symbolic_quantum_index(
                operation,
                child,
                loop_symbol,
                scalar_values=self.condition_values,
                used_names=self.branch_condition_names,
            ):
                assumption = ResourceAssumption(
                    "symbolic loop depth is sequential because disjoint "
                    "iteration footprints could not be proven",
                    source="for",
                )
                estimate = estimate._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.UPPER_BOUND,
                    active_when=sp.Gt(iterations, _ONE),
                )
        estimate = _with_operation_output_summary(
            estimate,
            operation,
            resolver,
            active_when=sp.Gt(iterations, _ZERO),
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
        )
        return _with_aggregate_boundary_depth_metadata(
            estimate,
            estimate,
            source="for",
            boundary="control-flow",
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
        coupled or nonlinear recurrences remain explicit symbols with a visible
        modeling assumption instead of silently resolving to a stale body value.

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
        concrete_bounds = tuple(
            self._concrete_scalar(bound) for bound in (start, stop, step)
        )
        if all(bound is not None for bound in concrete_bounds):
            concrete_start, concrete_stop, concrete_step = cast(
                tuple[int, int, int], concrete_bounds
            )
            if concrete_step == 0:
                raise ValueError(
                    "Resource estimation cannot evaluate a zero-step loop."
                )
            return self._eval_concrete_region_for(
                operation,
                resolver,
                range(concrete_start, concrete_stop, concrete_step),
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
        """Interpret a concrete region-argument loop exactly.

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
        composer = _SequentialEstimateComposer()
        iteration_width = WidthResources.zero()
        anonymous_allocated = _ZERO
        body = _LocalBlock(operation.operations)
        for loop_value in iterations:
            loop_expr = sp.Integer(loop_value)
            context = dict(carried)
            if operation.loop_var_value is not None:
                context[operation.loop_var_value.uuid] = loop_expr
            child = resolver.child_scope(
                inner_block=body,
                extra_context=context,
                extra_loop_vars={operation.loop_var: loop_expr},
            )
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
            composer.append(iteration_estimate)
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
        for arg in operation.region_args:
            resolver.bind(arg.result, carried[arg.block_arg.uuid])
        estimate = composer.finish()
        return dataclasses.replace(
            estimate,
            width=_width_with_identity_aware_allocations(
                iteration_width,
                estimate._allocation_sites,
                anonymous_allocated=anonymous_allocated,
            ),
        )

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
        """Evaluate independent affine region recurrences in closed form.

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
        """
        carry_symbols = {
            arg.block_arg.uuid: _typed_value_symbol(
                arg.block_arg,
                f"{arg.var_name}_carry",
                fresh=True,
            )
            for arg in operation.region_args
        }
        context: dict[str, sp.Expr] = dict(carry_symbols)
        if operation.loop_var_value is not None:
            context[operation.loop_var_value.uuid] = loop_symbol
        probe = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=context,
            extra_loop_vars={operation.loop_var: loop_symbol},
        )
        # Evaluate once so branch phi results become available to the resolver
        # before recurrence expressions are inspected. The estimate itself is
        # discarded and recomputed with the closed-form carry-at-iteration values.
        self.eval_operations(
            operation.operations,
            probe,
            controls=controls,
            initial_allocations=_captured_quantum_allocations(
                operation.operations,
                probe,
                self._allocation_owners_by_uuid,
            ),
        )

        iterations = symbolic_iterations(start, stop, step)
        at_iteration: dict[str, sp.Expr] = {}
        final_values: dict[str, sp.Expr] = {}
        assumptions: list[ResourceAssumption] = []
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
                at_value = sp.Function(f"{arg.var_name}_carry")(loop_symbol)
                final_value = _typed_value_symbol(
                    arg.result,
                    f"{arg.var_name}_after_loop",
                    fresh=True,
                )
                assumptions.append(
                    ResourceAssumption(
                        "loop-carried recurrence could not be reduced to an "
                        "independent affine closed form; its final value remains "
                        "symbolic",
                        source=arg.var_name,
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
        estimate = inner.sum_over(loop_symbol, start, stop, step)
        for arg in operation.region_args:
            resolver.bind(arg.result, final_values[arg.result.uuid])
        if assumptions:
            estimate = estimate._with_metadata(assumptions=assumptions)
        return estimate

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
        """Resolve an integer expression already concrete in the traced IR.

        Estimation ``inputs`` deliberately do not participate here. They are
        applied after symbolic loop summarization, preventing a large concrete
        problem size from turning a compact parametric loop into thousands of
        interpreter iterations. Structural inputs still reach this path as
        constants because they are baked into the built block.

        Args:
            expression (sp.Expr): Symbolic scalar expression.

        Returns:
            int | None: Concrete integer, or ``None`` when unresolved.
        """
        if expression.is_number and expression.is_integer:
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
        self._while_count += 1
        trip_count_name = (
            "|while|" if self._while_count == 1 else f"|while[{self._while_count}]|"
        )
        child, trip_count = build_while_scope(
            operation,
            resolver,
            trip_count_name=trip_count_name,
        )
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
        estimate = _with_operation_output_summary(
            inner.repeat(trip_count),
            operation,
            resolver,
            active_when=sp.Gt(trip_count, _ZERO),
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
        )
        return _with_aggregate_boundary_depth_metadata(
            estimate,
            estimate,
            source="while",
            boundary="control-flow",
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
        taken, note = self._decide_branch(resolver.resolve(operation.condition))
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
            )
            output_sizes = self._if_output_sizes(
                operation,
                resolver,
                true_child,
                false_child,
                taken=taken,
                runtime_condition=False,
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
            )
            return _with_aggregate_boundary_depth_metadata(
                estimate,
                estimate,
                source="if",
                boundary="control-flow",
            )
        is_runtime_condition = operation.condition.uuid in self._measurement_derived
        if is_runtime_condition:
            _require_uncontrolled_operation(operation, controls)
        true_estimate = self.eval_operations(
            operation.true_operations,
            true_child,
            controls=controls,
            initial_allocations=true_inputs,
        )
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
        )
        condition = resolver.resolve(operation.condition)
        output_sizes = self._if_output_sizes(
            operation,
            resolver,
            true_child,
            false_child,
            taken=None,
            runtime_condition=is_runtime_condition,
            true_consumed=true_consumed,
            false_consumed=false_consumed,
        )
        input_sizes = _branch_owner_sizes(
            true_inputs,
            false_inputs,
            condition=_boolean_condition(condition),
            runtime_condition=is_runtime_condition,
        )
        if not is_runtime_condition:
            combined = true_estimate.conditional(
                false_estimate,
                _boolean_condition(condition),
            )
            combined = dataclasses.replace(
                combined,
                _output_sizes=output_sizes,
                _input_sizes=input_sizes,
                _has_output_summary=True,
            )
            return _with_aggregate_boundary_depth_metadata(
                combined,
                combined,
                source="if",
                boundary="control-flow",
            )
        combined = true_estimate.choice(false_estimate)
        if note is None:
            combined = dataclasses.replace(
                combined,
                _output_sizes=output_sizes,
                _input_sizes=input_sizes,
                _has_output_summary=True,
            )
            return _with_aggregate_boundary_depth_metadata(
                combined,
                combined,
                source="if",
                boundary="control-flow",
            )
        trace = combined.trace
        if trace is not None:
            trace = dataclasses.replace(trace, assumptions=(*trace.assumptions, note))
        combined = dataclasses.replace(
            combined,
            trace=trace,
            _output_sizes=output_sizes,
            _input_sizes=input_sizes,
            _has_output_summary=True,
        )
        combined = combined._with_metadata(assumptions=(note,))
        return _with_aggregate_boundary_depth_metadata(
            combined,
            combined,
            source="if",
            boundary="control-flow",
        )

    def _publish_if_results(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        true_resolver: ExprResolver,
        false_resolver: ExprResolver,
        *,
        taken: bool | None,
    ) -> None:
        """Publish branch-merge results into the enclosing symbolic environment.

        Args:
            operation (IfOperation): Conditional carrying the merge records.
            resolver (ExprResolver): Enclosing resolver to update.
            true_resolver (ExprResolver): Resolver for the true branch.
            false_resolver (ExprResolver): Resolver for the false branch.
            taken (bool | None): Decided branch, or ``None`` when the condition
                remains symbolic or runtime-dependent.
        """
        condition = resolver.resolve(operation.condition)
        runtime_condition = operation.condition.uuid in self._measurement_derived
        predicate = _boolean_condition(condition)
        for merge in operation.iter_merges():
            true_value = true_resolver.resolve(merge.true_value)
            false_value = false_resolver.resolve(merge.false_value)
            if taken is True:
                merged = true_value
            elif taken is False:
                merged = false_value
            elif runtime_condition:
                # A runtime measurement chooses the merge value shot by shot.
                # Keeping every such merge as a nested Piecewise expression
                # makes feed-forward-heavy FTQC circuits grow exponentially
                # during final SymPy simplification, even though resource
                # counting already conservatively combines the branches with
                # ``choice`` above. A fresh typed symbol preserves the unknown
                # runtime value without coupling unrelated later estimates to
                # the complete measurement history.
                merged = _typed_value_symbol(
                    merge.result,
                    merge.result.name,
                    fresh=True,
                )
            elif bool(getattr(condition, "is_Boolean", False)):
                merged = sp.Piecewise(
                    (true_value, cast(Any, condition)),
                    (false_value, True),
                )
            else:
                merged = _typed_value_symbol(
                    merge.result,
                    merge.result.name,
                    fresh=True,
                )
            resolver.bind(merge.result, cast(sp.Expr, merged))
            if not all(
                isinstance(value, ArrayValue)
                for value in (
                    merge.true_value,
                    merge.false_value,
                    merge.result,
                )
            ):
                continue
            true_array = cast(ArrayValue, merge.true_value)
            false_array = cast(ArrayValue, merge.false_value)
            result_array = cast(ArrayValue, merge.result)
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
                elif runtime_condition:
                    merged_size = sp.Max(true_size, false_size)
                else:
                    merged_size = _piecewise(
                        true_size,
                        false_size,
                        predicate,
                    )
                resolver.bind(result_dim, cast(sp.Expr, merged_size))

    def _if_output_sizes(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        true_resolver: ExprResolver,
        false_resolver: ExprResolver,
        *,
        taken: bool | None,
        runtime_condition: bool,
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
            taken (bool | None): Statically selected branch, if any.
            runtime_condition (bool): Whether a measurement selects the branch
                at runtime.
            true_consumed (Mapping[str, ResourceExpr]): Captured owner widths
                destroyed by the true branch.
            false_consumed (Mapping[str, ResourceExpr]): Captured owner widths
                destroyed by the false branch.

        Returns:
            dict[str, ResourceExpr]: Live merged output width by root owner.
        """
        condition = _boolean_condition(resolver.resolve(operation.condition))
        true_sizes: dict[str, ResourceExpr] = {}
        false_sizes: dict[str, ResourceExpr] = {}
        for merge in operation.iter_merges():
            if not merge.result.type.is_quantum():
                continue
            owner = _quantum_allocation_owner(merge.result)
            true_sizes[owner] = true_sizes.get(owner, _ZERO) + _qubit_value_size(
                merge.true_value,
                true_resolver,
            )
            false_sizes[owner] = false_sizes.get(owner, _ZERO) + _qubit_value_size(
                merge.false_value,
                false_resolver,
            )
        output_sizes: dict[str, ResourceExpr] = {}
        for owner in true_sizes.keys() | false_sizes.keys():
            true_size = sp.Max(
                _ZERO,
                true_sizes.get(owner, _ZERO) - true_consumed.get(owner, _ZERO),
            )
            false_size = sp.Max(
                _ZERO,
                false_sizes.get(owner, _ZERO) - false_consumed.get(owner, _ZERO),
            )
            if taken is True:
                size = true_size
            elif taken is False:
                size = false_size
            elif runtime_condition:
                size = sp.Max(true_size, false_size)
            else:
                size = _piecewise(true_size, false_size, condition)
            if size != _ZERO:
                output_sizes[owner] = size
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
                whose resource use depends on the current key or value.
        """
        _require_uncontrolled_operation(operation, controls)
        cardinality = resolve_for_items_cardinality(operation)
        if operation.region_args:
            estimate = self._eval_region_for_items(
                operation,
                resolver,
                cardinality=cardinality,
                controls=controls,
            )
        else:
            entries = self._for_items_entries(operation)
            if entries is not None:
                estimate = self._eval_concrete_for_items(
                    operation,
                    resolver,
                    entries,
                    controls=controls,
                )
            else:
                context, item_symbols = self._symbolic_for_items_context(operation)
                child = resolver.child_scope(
                    inner_block=_LocalBlock(operation.operations),
                    extra_context=context,
                )
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
                self._ensure_for_items_resource_independent(inner, item_symbols)
                estimate = inner.repeat(cardinality)
        estimate = _with_operation_output_summary(
            estimate,
            operation,
            resolver,
            active_when=sp.Gt(cardinality, _ZERO),
            allocation_owners_by_uuid=self._allocation_owners_by_uuid,
        )
        return _with_aggregate_boundary_depth_metadata(
            estimate,
            estimate,
            source="for_items",
            boundary="control-flow",
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
            ResourceEstimate: Exact estimate for bound dictionaries, otherwise a
            cardinality-based symbolic estimate.

        Raises:
            NotImplementedError: If an unbound loop-carried value has a
                recurrence that depends on the current item key or value.
        """
        entries = self._for_items_entries(operation)
        if entries is not None:
            return self._eval_concrete_region_for_items(
                operation,
                resolver,
                entries,
                controls=controls,
            )

        item_symbol = sp.Symbol("item_index", integer=True, nonnegative=True)
        context, item_symbols = self._symbolic_for_items_context(operation)
        carry_symbols = {
            arg.block_arg.uuid: _typed_value_symbol(
                arg.block_arg,
                f"{arg.var_name}_carry",
                fresh=True,
            )
            for arg in operation.region_args
        }
        context.update(carry_symbols)
        probe = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=context,
        )
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
        self._ensure_for_items_resource_independent(probe_estimate, item_symbols)

        at_iteration: dict[str, sp.Expr] = {}
        final_values: dict[str, sp.Expr] = {}
        assumptions: list[ResourceAssumption] = []
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
                at_value = sp.Function(f"{arg.var_name}_carry")(item_symbol)
                final_value = _typed_value_symbol(
                    arg.result,
                    f"{arg.var_name}_after_items",
                    fresh=True,
                )
                assumptions.append(
                    ResourceAssumption(
                        "items-loop carry could not be reduced to an independent "
                        "affine closed form; its final value remains symbolic",
                        source=arg.var_name,
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
        estimate = self.eval_operations(
            operation.operations,
            child,
            controls=controls,
            initial_allocations=_captured_quantum_allocations(
                operation.operations,
                child,
                self._allocation_owners_by_uuid,
            ),
        ).sum_over(item_symbol, _ZERO, cardinality, _ONE)
        for arg in operation.region_args:
            resolver.bind(arg.result, final_values[arg.result.uuid])
        if assumptions:
            estimate = estimate._with_metadata(assumptions=assumptions)
        return estimate

    def _eval_concrete_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        entries: tuple[tuple[Any, Any], ...],
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Interpret a bound items loop without carried values exactly.

        Args:
            operation (ForItemsOperation): Bound dictionary loop to evaluate.
            resolver (ExprResolver): Enclosing symbolic environment.
            entries (tuple[tuple[Any, Any], ...]): Concrete key-value entries
                in insertion order.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Per-entry composition — gates, depth, and
                calls accumulate sequentially; peak and ancilla width use the
                per-entry maximum, while allocated width is the union of
                distinct QInit identities.
        """
        composer = _SequentialEstimateComposer()
        iteration_width = WidthResources.zero()
        anonymous_allocated = _ZERO
        for key, value in entries:
            child = resolver.child_scope(
                inner_block=_LocalBlock(operation.operations),
                extra_context=self._concrete_for_items_context(
                    operation,
                    key,
                    value,
                ),
            )
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
            composer.append(entry_estimate)
            iteration_width = _max_width(iteration_width, entry_estimate.width)
            anonymous_allocated = sp.Max(
                anonymous_allocated,
                _anonymous_allocation_width(
                    entry_estimate.width,
                    entry_estimate._allocation_sites,
                ),
            )
        # ``seq`` counts every concrete visit to a QInit. Keep sequential
        # gate/depth/call totals, take reusable width fields per-entry, and
        # replace only allocated_qubits with the distinct static-site union.
        estimate = composer.finish()
        return dataclasses.replace(
            estimate,
            width=_width_with_identity_aware_allocations(
                iteration_width,
                estimate._allocation_sites,
                anonymous_allocated=anonymous_allocated,
            ),
        )

    def _eval_concrete_region_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        entries: tuple[tuple[Any, Any], ...],
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Interpret a bound items loop exactly.

        Args:
            operation (ForItemsOperation): Items loop carrying region arguments.
            resolver (ExprResolver): Enclosing symbolic environment.
            entries (tuple[tuple[Any, Any], ...]): Bound key-value entries.
            controls (ResourceExpr | int): Surrounding controls.

        Returns:
            ResourceEstimate: Sequential work with per-entry peak width and
                identity-deduplicated static allocations.
        """
        carried = {
            arg.block_arg.uuid: self._apply_condition_values(resolver.resolve(arg.init))
            for arg in operation.region_args
        }
        composer = _SequentialEstimateComposer()
        iteration_width = WidthResources.zero()
        anonymous_allocated = _ZERO
        for key, value in entries:
            context = {
                **carried,
                **self._concrete_for_items_context(operation, key, value),
            }
            child = resolver.child_scope(
                inner_block=_LocalBlock(operation.operations),
                extra_context=context,
            )
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
            composer.append(iteration_estimate)
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
        for arg in operation.region_args:
            resolver.bind(arg.result, carried[arg.block_arg.uuid])
        estimate = composer.finish()
        return dataclasses.replace(
            estimate,
            width=_width_with_identity_aware_allocations(
                iteration_width,
                estimate._allocation_sites,
                anonymous_allocated=anonymous_allocated,
            ),
        )

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
        metadata = getattr(operand, "metadata", None)
        dict_runtime = getattr(metadata, "dict_runtime", None)
        if dict_runtime is not None:
            return tuple(dict_runtime.bound_data)
        return None

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
        body = operation.effective_body(strategy=strategy)
        if isinstance(body, Block):
            return _with_constraints(
                self._estimate_invoke_body(operation, body, resolver, controls),
                *width_constraints,
            )
        ctx = self._opaque_call_context(
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
                self._estimate_opaque_cost(operation, opaque_cost, ctx),
                *width_constraints,
            )
        return _with_constraints(
            self._handle_unknown_invoke(operation, ctx),
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
        total_controls = _expr(controls) + _expr(local_controls)
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
        unresolved_control_selection = control_indices is not None and any(
            not index.is_number or not _is_concrete_integer(index)
            for index in resolved_indices
        )
        callable_name = (
            operation.callable_ref.name
            if operation.callable_ref is not None
            else "controlled_u"
        )
        structural_constraints = [
            *_quantum_operand_width_constraints(
                operation.callable_attrs,
                _controlled_u_body_operands(operation),
                resolver,
                source=callable_name,
            ),
            _ResourceConstraint(
                expression=_expr(local_controls),
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
                expected=_expr(local_controls),
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
            child = _controlled_u_child_resolver(operation, resolver)
            actual_operands = _controlled_u_body_operands(operation)
            body = self._eval_call_body(
                operation.block,
                child,
                actual_operands,
                controls=total_controls,
            )
            body_dependency_estimate = body
            broadcast = _scalar_target_broadcast_factor(
                operation.block,
                [
                    operand
                    for operand in _controlled_u_body_operands(operation)
                    if operand.type.is_quantum()
                ],
                resolver,
            )
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
                    surrounding_controls=controls,
                    active_when=repetitions,
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
            if dependency_keys is not None:
                mapped_keys = set(dependency_keys)
                if _estimate_has_nonzero_depth(estimate):
                    mapped_keys.update(
                        _controlled_u_control_wire_keys(
                            operation,
                            resolved_indices,
                            resolver,
                            scalar_values=self.condition_values,
                            used_names=self.branch_condition_names,
                        )
                    )
                estimate = dataclasses.replace(
                    estimate,
                    _dependency_keys=frozenset(mapped_keys),
                )
            estimate = _with_constraints(
                _namespace_allocation_sites(estimate, operation),
                *structural_constraints,
            )
            if unresolved_control_selection:
                assumption = ResourceAssumption(
                    "symbolic control selection uses the whole control-pool "
                    "dependency footprint",
                    source=callable_name,
                )
                estimate = estimate._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.UPPER_BOUND,
                )
            return _with_body_boundary_depth_metadata(
                estimate,
                body_dependency_estimate,
                source=callable_name,
                zero_controls=zero_controls,
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
                quality=EstimateQuality.MODELED,
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
                    surrounding_controls=controls,
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
            quality=EstimateQuality.MODELED,
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
                surrounding_controls=controls,
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
        outer controls, matching the portable SELECT fallback.

        Args:
            operation (SelectOperation): SELECT operation to evaluate.
            resolver (ExprResolver): Resolver for the SELECT call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Sequential estimate of all controlled case
            bodies, including scalar-case broadcast over vector targets.
        """
        index_controls = (
            resolver.resolve(operation.num_index_qubits)
            if isinstance(operation.num_index_qubits, Value)
            else _expr(operation.num_index_qubits)
        )
        total_controls = _expr(controls) + index_controls
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
        index_constraint.validate()
        index_operand_constraint.validate()
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
            broadcast = _scalar_target_broadcast_factor(
                case_block,
                operation.target_operands,
                resolver,
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
                surrounding_controls=controls,
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
                )
            case_estimate = _with_body_boundary_depth_metadata(
                case_estimate,
                case_body,
                source=f"select[{case_index}]",
                zero_controls=zero_controls,
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
                    quality=EstimateQuality.MODELED,
                    trace=ResourceTraceNode(
                        name,
                        "opaque",
                        assumptions=(assumption,),
                    ),
                )
            else:
                estimate = ResourceEstimate(
                    assumptions=(assumption,),
                    quality=EstimateQuality.MODELED,
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
                surrounding_controls=controls,
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
            surrounding_controls=controls,
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
            )
        estimate = _with_constraints(
            _namespace_allocation_sites(estimate, operation),
            *width_constraints,
        )
        return _with_body_boundary_depth_metadata(
            estimate,
            body_estimate,
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
            ResourceEstimate: Portable Pauli-gadget resources, or a modeled
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
                quality=EstimateQuality.MODELED,
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
        if gamma.is_zero is True:
            return _with_constraints(
                ResourceEstimate.zero("pauli_evolve"),
                register_constraint,
            )

        constant = complex(hamiltonian.constant)
        if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
            raise ValueError(
                "PauliEvolveOp requires a Hermitian Hamiltonian (real "
                "constant), but the constant has a nonzero imaginary part."
            )

        term_estimates: list[ResourceEstimate] = []
        active_pauli_terms: list[tuple[qm_o.PauliOperator, ...]] = []
        for operators, coefficient in hamiltonian:
            resolved_coefficient = complex(coefficient)
            if abs(resolved_coefficient.imag) > HERMITIAN_IMAG_ATOL:
                raise ValueError(
                    "PauliEvolveOp requires a Hermitian Hamiltonian (real "
                    "Pauli coefficients), but a term has a nonzero imaginary "
                    "part."
                )
            if abs(resolved_coefficient) < PAULI_TERM_ZERO_ATOL or not operators:
                continue
            active_pauli_terms.append(operators)
            x_count = sum(operator.pauli == qm_o.Pauli.X for operator in operators)
            y_count = sum(operator.pauli == qm_o.Pauli.Y for operator in operators)
            basis_h_gate = _estimate_named_gate_in_basis(
                "h",
                _ZERO,
                basis=self.config.basis,
                precision=self.config.precision,
            )
            basis_h = basis_h_gate.repeat(2 * (x_count + y_count))
            basis_s_gate = _estimate_named_gate_in_basis(
                "s",
                _ZERO,
                basis=self.config.basis,
                precision=self.config.precision,
            )
            basis_s = basis_s_gate.repeat(2 * y_count)
            parity_gate = _estimate_named_gate_in_basis(
                "cx",
                _ZERO,
                basis=self.config.basis,
                precision=self.config.precision,
            )
            parity = parity_gate.repeat(2 * max(0, len(operators) - 1))
            rotation = _estimate_named_gate_in_basis(
                "rz",
                _expr(controls),
                basis=self.config.basis,
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
                "Lie-Trotter product-formula step in Hamiltonian term order. "
                "EXACT quality, when present, describes the resource count of "
                "that selected circuit, not exact full-Hamiltonian evolution.",
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
                assumptions=(trotter_assumption,)
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

    def _opaque_call_context(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr,
        strategy: str | None,
    ) -> OpaqueCallContext:
        """Build an opaque-cost context for a bodyless invocation.

        Args:
            operation (InvokeOperation): Invocation operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr): Surrounding control count.
            strategy (str | None): Selected strategy.

        Returns:
            OpaqueCallContext: Opaque call context.
        """
        return OpaqueCallContext(
            callable_ref=operation.target,
            argument_values=tuple(operation.operands),
            operand_shapes=_operand_shapes(operation.operands, resolver),
            attrs=operation.attrs,
            loop_symbols=resolver.loop_var_names,
            controls=controls,
            transform=operation.transform,
            strategy=strategy,
            bindings=self.bindings,
            basis=self.config.basis,
            precision=(
                self.config.precision
                if self.config.basis is GateBasis.CLIFFORD_T
                else None
            ),
        )

    def _estimate_opaque_cost(
        self,
        operation: InvokeOperation,
        cost: Any,
        ctx: OpaqueCallContext,
    ) -> ResourceEstimate:
        """Evaluate an explicit cost attached to a bodyless callable.

        Fixed ``ResourceEstimate`` costs describe a transform-agnostic base
        implementation, so the estimator applies repetition and all coherent
        controls automatically. A callable cost is authoritative for the full
        :class:`OpaqueCallContext`, including ``power`` and
        ``total_controls``; its result is therefore validated and wrapped
        without additional transform pricing.

        Args:
            operation (InvokeOperation): Invocation operation.
            cost (Any): ``ResourceEstimate`` or callable accepting ``ctx``.
            ctx (OpaqueCallContext): Opaque call context.

        Returns:
            ResourceEstimate: Explicit opaque estimate.

        Raises:
            TypeError: If ``cost`` is neither a ``ResourceEstimate`` nor a
                callable returning one.
            ValueError: If the opaque cost reports basis-sensitive resources
                for a different basis or synthesis precision, or if a fixed
                non-unitary cost is controlled or inverted without a
                transform-aware callback.
        """
        if ctx.power == _ZERO:
            return ResourceEstimate.zero(f"{operation.custom_name}^0")
        if isinstance(cost, ResourceEstimate):
            _validate_opaque_cost_provenance(
                cost,
                name=operation.custom_name,
                basis=self.config.basis,
                precision=self.config.precision,
            )
            estimate = cost.repeat(ctx.power)
            if ctx.transform is CallTransform.INVERSE:
                estimate = estimate.inverse()
            if ctx.total_controls != _ZERO:
                estimate = estimate.controlled(ctx.total_controls)
            if ctx.own_controls:
                estimate = self._with_zero_control_bracket(
                    estimate,
                    zero_controls=_zero_control_count(
                        ctx.own_controls,
                        cast(int | None, ctx.attrs.get("control_value")),
                    ),
                    surrounding_controls=ctx.controls,
                )
        elif callable(cost):
            defer_token = _DEFER_RESOURCE_SYMBOL_METADATA.set(False)
            try:
                estimate = cost(ctx)
            finally:
                _DEFER_RESOURCE_SYMBOL_METADATA.reset(defer_token)
            if not isinstance(estimate, ResourceEstimate):
                raise TypeError(
                    f"Opaque cost for '{operation.custom_name}' must return "
                    "ResourceEstimate."
                )
            _validate_opaque_cost_provenance(
                estimate,
                name=operation.custom_name,
                basis=self.config.basis,
                precision=self.config.precision,
            )
        else:
            raise TypeError(
                f"Opaque cost for '{operation.custom_name}' must be a "
                "ResourceEstimate or callable."
            )
        estimate = dataclasses.replace(
            estimate,
            trace=_wrap_trace(
                operation.custom_name,
                estimate.trace,
                source_kind="opaque_cost",
                strategy=ctx.strategy,
            ),
        )
        return estimate._with_metadata(
            quality=EstimateQuality.MODELED,
            active_when=ctx.power,
        )

    def _estimate_invoke_body(
        self,
        operation: InvokeOperation,
        body: Block,
        resolver: ExprResolver,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Estimate an invocation by traversing its body.

        When the invocation is itself a controlled call
        (``transform is CallTransform.CONTROLLED``), its own control qubits are
        added to the surrounding controls so every primitive gate inside the
        body is classified as controlled, matching how ``eval_controlled_u``
        treats a block body.

        Args:
            operation (InvokeOperation): Invocation operation.
            body (Block): Selected callable body.
            resolver (ExprResolver): Call-site resolver.
            controls (ResourceExpr | int): Surrounding control count.

        Returns:
            ResourceEstimate: Body-derived estimate.
        """
        selected_impl = operation.implementation_for(
            strategy=self._strategy_for(operation)
        )
        body_implements_transform = (
            selected_impl is not None and selected_impl.body is body
        )
        child = resolver.call_child_scope(
            operation,
            called_block=body,
            body_implements_transform=body_implements_transform,
        )
        own_controls = 0
        if (
            operation.transform is CallTransform.CONTROLLED
            and not body_implements_transform
        ):
            own_controls = int(operation.attrs.get("num_control_qubits", 0) or 0)
        total_controls = _expr(controls) + own_controls
        actual_operands = operation.operands
        if (
            operation.transform is CallTransform.CONTROLLED
            and not body_implements_transform
        ):
            actual_operands = actual_operands[operation.num_control_qubits :]
        body_estimate = self._eval_call_body(
            body,
            child,
            actual_operands,
            controls=total_controls,
        )
        body_dependency_estimate = body_estimate
        if (
            operation.transform is CallTransform.INVERSE
            and not body_implements_transform
        ):
            body_estimate = body_estimate.inverse()
        zero_controls = _ZERO
        if own_controls:
            zero_controls = _zero_control_count(
                own_controls,
                operation.control_value,
            )
            body_estimate = self._with_zero_control_bracket(
                body_estimate,
                zero_controls=zero_controls,
                surrounding_controls=controls,
            )
        caller_results: Sequence[ValueBase] = operation.results
        if (
            operation.transform is CallTransform.CONTROLLED
            and not body_implements_transform
        ):
            caller_results = operation.results[operation.num_control_qubits :]
        dependency_keys = _map_body_dependency_keys(
            body,
            body_dependency_estimate,
            actual_operands,
            caller_results,
            resolver,
            scalar_values=self.condition_values,
            used_names=self.branch_condition_names,
        )
        if dependency_keys is not None:
            mapped_keys = set(dependency_keys)
            if own_controls and _estimate_has_nonzero_depth(body_estimate):
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
            )
        output_sizes, has_output_summary = _invoke_quantum_output_sizes(
            operation,
            body,
            child,
            resolver,
            body_implements_transform=body_implements_transform,
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
                _has_output_summary=has_output_summary,
            ),
            operation,
        )
        return _with_body_boundary_depth_metadata(
            estimate,
            body_dependency_estimate,
            source=operation.custom_name,
            zero_controls=zero_controls,
        )

    def _handle_unknown_invoke(
        self,
        operation: InvokeOperation,
        ctx: OpaqueCallContext,
    ) -> ResourceEstimate:
        """Handle an invocation without a body or opaque cost.

        Args:
            operation (InvokeOperation): Invocation operation.
            ctx (OpaqueCallContext): Call-site context.

        Returns:
            ResourceEstimate: Opaque or zero estimate when policy permits.

        Raises:
            ValueError: If the configured unknown policy is ``ERROR``.
        """
        name = operation.custom_name
        if self.config.unknown_policy is UnknownResourcePolicy.OPAQUE_CALL:
            estimate = ResourceEstimate(
                calls=CallResources(
                    calls_by_name={name: ctx.power},
                    queries_by_name={name: ctx.power},
                ),
                trace=ResourceTraceNode(name, "opaque", summary=f"power={ctx.power}"),
                quality=EstimateQuality.MODELED,
            )
        elif self.config.unknown_policy is UnknownResourcePolicy.ZERO_WITH_WARNING:
            assumption = ResourceAssumption(
                "unknown callable counted as zero resources",
                source=name,
            )
            estimate = ResourceEstimate(
                assumptions=(assumption,),
                trace=ResourceTraceNode(name, "opaque", assumptions=(assumption,)),
                quality=EstimateQuality.MODELED,
            )
        else:
            raise ValueError(
                f"Cannot estimate resources for callable '{name}': no body or "
                "opaque cost is available."
            )
        if ctx.own_controls:
            estimate = self._with_zero_control_bracket(
                estimate,
                zero_controls=_zero_control_count(
                    ctx.own_controls,
                    cast(int | None, ctx.attrs.get("control_value")),
                ),
                surrounding_controls=ctx.controls,
                active_when=ctx.power,
            )
        return estimate


def estimate_resources(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
    *,
    inputs: dict[str, Any] | None = None,
    strategies: dict[str, str] | None = None,
    trace: bool = False,
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
    basis: str | GateBasis = GateBasis.PORTABLE,
    precision: float = 1e-10,
) -> ResourceEstimate:
    """Estimate logical resources using the default estimator facade.

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
        unknown_policy (UnknownResourcePolicy): Unknown callable handling.
            Defaults to ``ERROR``.
        basis (str | GateBasis): Output gate basis. Defaults to ``PORTABLE``.
        precision (float): Rotation-synthesis precision for ``CLIFFORD_T``.
            Defaults to ``1e-10``.

    Returns:
        ResourceEstimate: Logical resource estimate.

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
            "portable emitter does not support this operation inside a "
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

    for value in (*operation.all_input_values(), *operation.results):
        if isinstance(value, ValueBase):
            visit(value)

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
        constraints,
        proven_cache=proven_cache,
    )


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
        label = f"Array '{display_name}' {access_kind} index {axis}"
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
                    label=f"{label} upper bound",
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
    coefficient = sp.simplify(sp.diff(yielded, carry_symbol))
    remainder = sp.simplify(yielded - coefficient * carry_symbol)
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

    completed_at_loop_value = sp.simplify((loop_symbol - start) / step)
    return value_after(completed_at_loop_value), value_after(iterations)


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


_CLIFFORD_GATES = {"h", "x", "y", "z", "s", "sdg", "cx", "cz", "swap"}
_T_GATES = {"t", "tdg"}
_SINGLE_QUBIT_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
}
_TWO_QUBIT_GATES = {"cx", "cz", "swap", "cp", "rzz"}
_ROTATION_GATES = {"rx", "ry", "rz", "p", "cp", "rzz"}
_MULTI_QUBIT_GATES = {"toffoli", "ccx"}
_GATE_BASE_QUBITS: dict[str, int] = {"toffoli": 3, "ccx": 3}
_CONTROLLED_CLIFFORD_GATES = {"x", "y", "z"}


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


def _portable_sequence_estimate(
    name: str,
    gates: GateResources,
    *,
    clean_ancillas: ResourceExpr = _ZERO,
    quality: EstimateQuality = EstimateQuality.EXACT,
) -> ResourceEstimate:
    """Build one portable logical-decomposition estimate.

    Args:
        name (str): Human-readable gate or decomposition name.
        gates (GateResources): Aggregate logical gate resources.
        clean_ancillas (ResourceExpr): Reusable clean-ancilla demand.
            Defaults to zero.
        quality (EstimateQuality): Confidence classification. Defaults to
            ``EXACT``.

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
            source_kind="portable_fallback",
            summary=f"gates={gates.total}, clean_ancillas={clean_ancillas}",
        ),
    )


def _portable_primitive_estimate(name: str) -> ResourceEstimate:
    """Return one uncontrolled logical primitive estimate.

    Args:
        name (str): Lowercase primitive gate name.

    Returns:
        ResourceEstimate: One-gate portable estimate.
    """
    return _portable_sequence_estimate(name, _classify_uncontrolled_gate(name))


def _portable_single_control_estimate(name: str) -> ResourceEstimate:
    """Return one singly controlled logical primitive estimate.

    Fixed S/T-family controls are represented by a controlled phase gate,
    matching the shared emitter's single-control dispatch.

    Args:
        name (str): Lowercase base gate name.

    Returns:
        ResourceEstimate: Singly controlled logical estimate.
    """
    controlled_name = "p" if name in {"s", "sdg", "t", "tdg"} else name
    return _portable_sequence_estimate(
        f"c-{name}",
        _classify_controlled_gate(controlled_name, _ONE),
    )


def _portable_generic_multi_control_estimate(
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Lower a generic multi-controlled single-target gate portably.

    The model mirrors the shared clean-ancilla fallback per primitive: compute
    the AND of ``n`` controls with ``n - 1`` clean ancillas, apply one singly
    controlled primitive, then uncompute the ladder. Backend-native gates and
    whole-body ladder sharing may be cheaper, so the result is an upper bound.

    Args:
        name (str): Lowercase single-target base gate name.
        controls (ResourceExpr): Number of controls, interpreted on the branch
            where it is at least two.

    Returns:
        ResourceEstimate: Portable fallback estimate.
    """
    ladder_steps = sp.Integer(2) * (controls - _ONE)
    ladder = _scale_gates(
        _classify_uncontrolled_gate("toffoli"),
        ladder_steps,
    )
    central = _portable_single_control_estimate(name).gates
    return _portable_sequence_estimate(
        f"mc-{name}",
        _add_gates(ladder, central),
        clean_ancillas=controls - _ONE,
        quality=EstimateQuality.UPPER_BOUND,
    )


def _select_control_count_estimate(
    controls: ResourceExpr,
    *,
    zero: ResourceEstimate,
    one: ResourceEstimate,
    two: ResourceEstimate,
    many: ResourceEstimate,
) -> ResourceEstimate:
    """Select a portable decomposition by a concrete or symbolic control count.

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


def _portable_single_target_estimate(
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a controlled single-target primitive in the portable basis.

    Args:
        name (str): Lowercase base gate name.
        controls (ResourceExpr): Surrounding control count.

    Returns:
        ResourceEstimate: Portable logical estimate.
    """
    zero = _portable_primitive_estimate(name)
    one = _portable_single_control_estimate(name)
    generic = _portable_generic_multi_control_estimate(name, controls)
    return _select_control_count_estimate(
        controls,
        zero=zero,
        one=one,
        two=generic,
        many=generic,
    )


def _portable_x_estimate(controls: ResourceExpr) -> ResourceEstimate:
    """Estimate an X gate with the shared multi-control fast paths.

    Args:
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceEstimate: Portable X-family estimate.
    """
    return _select_control_count_estimate(
        controls,
        zero=_portable_primitive_estimate("x"),
        one=_portable_single_control_estimate("x"),
        two=_portable_primitive_estimate("toffoli"),
        many=_portable_generic_multi_control_estimate("x", controls),
    )


def _portable_z_estimate(controls: ResourceExpr) -> ResourceEstimate:
    """Estimate a Z gate with the shared two-control fast path.

    Args:
        controls (ResourceExpr): Number of controls on the Z target.

    Returns:
        ResourceEstimate: Portable Z-family estimate.
    """
    two = (
        _portable_primitive_estimate("h")
        .seq(_portable_primitive_estimate("toffoli"))
        .seq(_portable_primitive_estimate("h"))
    )
    two = two._with_metadata(quality=EstimateQuality.UPPER_BOUND)
    return _select_control_count_estimate(
        controls,
        zero=_portable_primitive_estimate("z"),
        one=_portable_single_control_estimate("z"),
        two=two,
        many=_portable_generic_multi_control_estimate("z", controls),
    )


def _portable_controlled_branch(
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


def _estimate_portable_named_gate(
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate one named gate after portable controlled lowering.

    Args:
        name (str): Lowercase Qamomile gate name.
        controls (ResourceExpr): Additional surrounding controls.

    Returns:
        ResourceEstimate: Portable logical decomposition estimate.
    """
    if name == "ccx":
        name = "toffoli"
    if name == "x":
        return _portable_x_estimate(controls)
    if name == "z":
        return _portable_z_estimate(controls)
    if name == "cx":
        return _portable_controlled_branch(
            controls,
            _portable_primitive_estimate("cx"),
            _portable_x_estimate(controls + _ONE),
        )
    if name == "cz":
        return _portable_controlled_branch(
            controls,
            _portable_primitive_estimate("cz"),
            _portable_z_estimate(controls + _ONE),
        )
    if name == "cp":
        return _portable_controlled_branch(
            controls,
            _portable_primitive_estimate("cp"),
            _portable_single_target_estimate("p", controls + _ONE),
        )
    if name == "toffoli":
        return _portable_controlled_branch(
            controls,
            _portable_primitive_estimate("toffoli"),
            _portable_x_estimate(controls + 2),
        )
    if name == "swap":
        middle = _portable_x_estimate(controls + _ONE)
        controlled = (
            _portable_primitive_estimate("cx")
            .seq(middle)
            .seq(_portable_primitive_estimate("cx"))
        )
        controlled = controlled._with_metadata(
            quality=EstimateQuality.UPPER_BOUND,
        )
        return _portable_controlled_branch(
            controls,
            _portable_primitive_estimate("swap"),
            controlled,
        )
    if name == "rzz":
        controlled = (
            _portable_primitive_estimate("cx")
            .seq(_portable_single_target_estimate("rz", controls))
            .seq(_portable_primitive_estimate("cx"))
        )
        controlled = controlled._with_metadata(
            quality=EstimateQuality.UPPER_BOUND,
        )
        return _portable_controlled_branch(
            controls,
            _portable_primitive_estimate("rzz"),
            controlled,
        )
    return _portable_single_target_estimate(name, controls)


def _portable_controlled_arity_envelope(
    num_qubits: int,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Bound one unknown primitive from its operand arity.

    Every resource field is maximized independently over the portable gate
    kinds of the requested arity. The resulting fields need not describe one
    common gate decomposition, but each is a valid upper bound when the gate
    name is unavailable.

    Args:
        num_qubits (int): Primitive operand arity. Supported values are one
            and two.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        ResourceEstimate: Field-wise portable upper bound for one primitive.

    Raises:
        ValueError: If ``num_qubits`` is not one or two.
    """
    if num_qubits == 1:
        gate_names = sorted(_SINGLE_QUBIT_GATES)
    elif num_qubits == 2:
        gate_names = sorted(_TWO_QUBIT_GATES)
    else:
        raise ValueError(
            "Portable aggregate control projection supports only one- and "
            f"two-qubit primitives; got arity {num_qubits}."
        )

    envelope = _estimate_portable_named_gate(gate_names[0], controls)
    for gate_name in gate_names[1:]:
        envelope = envelope.choice(_estimate_portable_named_gate(gate_name, controls))
    return dataclasses.replace(
        envelope,
        trace=ResourceTraceNode(
            name=f"unknown_{num_qubits}q",
            source_kind="portable_arity_upper_bound",
            summary=f"controls={controls}",
            children=(envelope.trace,) if envelope.trace is not None else (),
        ),
    )


def _aggregate_arity_projection_reason(
    estimate: ResourceEstimate,
) -> str | None:
    """Return why an aggregate estimate cannot use portable arity projection.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        str | None: Ineligibility reason, or ``None`` when at least one
            one- or two-qubit gate can be projected portably.
    """
    if estimate.basis is not GateBasis.PORTABLE:
        return (
            "automatic arity projection is available only in the portable "
            f"basis, not {estimate.basis.value}"
        )
    gates = estimate.gates
    has_opaque_calls = any(
        value != _ZERO
        for value in (
            *estimate.calls.calls_by_name.values(),
            *estimate.calls.queries_by_name.values(),
        )
    )
    if _safe_simplify(gates.total) == _ZERO and has_opaque_calls:
        return "the opaque call or query has no declared one- or two-qubit gate profile"
    if _safe_simplify(gates.total) == _ZERO:
        return (
            "a zero aggregate gate profile cannot distinguish an exact "
            "identity from an undeclared global phase"
        )
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
    if _safe_simplify(gates.single_qubit + gates.two_qubit) == _ZERO:
        return "the aggregate has no declared one- or two-qubit gate profile"
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
    control_scope = (
        f"its {controls} controls" if controls.is_number else "its active controls"
    )
    return ResourceAssumption(
        message=(
            "aggregate controlled cost is unchanged because "
            f"{reason}; no primitive body is available for {control_scope}"
        )
    )


def _aggregate_arity_projection_constraints(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[_ResourceConstraint, ...]:
    """Retain unresolved validity requirements for an arity projection.

    Args:
        estimate (ResourceEstimate): Eligible aggregate estimate.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements that become active only
            when one or more controls use the projection.
    """
    gates = estimate.gates
    requirements: list[_ResourceConstraint] = []
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
        if _safe_simplify(count).is_nonnegative is not True:
            requirements.append(
                _ResourceConstraint(
                    expression=count,
                    minimum=0,
                    label=f"Portable aggregate {label} gate count",
                    unit="gate",
                )
            )
    arity_remainder = _safe_simplify(
        gates.total - gates.single_qubit - gates.two_qubit - gates.multi_qubit
    )
    if arity_remainder.is_nonnegative is not True:
        requirements.append(
            _ResourceConstraint(
                expression=arity_remainder,
                minimum=0,
                label="Portable aggregate unclassified arity remainder",
                unit="gate",
            )
        )
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
        if _safe_simplify(count - upper).is_nonpositive is not True:
            requirements.append(
                _ResourceConstraint(
                    expression=upper - count,
                    minimum=0,
                    label=(
                        f"Portable aggregate {label} count within "
                        f"{upper_label} gate count"
                    ),
                    unit="gate",
                )
            )
    active_when = sp.And(
        sp.Gt(controls, _ZERO),
        _resource_activity_condition(_estimate_activity(estimate)),
    )
    guarded = tuple(requirement.when(active_when) for requirement in requirements)
    for requirement in guarded:
        requirement.validate()
    return guarded


def _portable_shared_aggregate_control_ladder(
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
        ResourceEstimate: Body-wide portable control projection with one
        compute/uncompute ladder and concurrently held clean ancillas.
    """
    outer_clean_ancillas = controls - _ONE
    toffoli = _portable_primitive_estimate("toffoli")
    compute = toffoli.repeat(outer_clean_ancillas)
    ladder = toffoli.repeat(2 * outer_clean_ancillas)
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
            source_kind="portable_fallback",
            summary=(
                f"toffoli_steps={2 * outer_clean_ancillas}, "
                f"clean_ancillas={outer_clean_ancillas}"
            ),
            children=trace_children,
        ),
        _output_sizes=body._output_sizes,
        _input_sizes=body._input_sizes,
        _has_output_summary=body._has_output_summary,
        _dependency_keys=body._dependency_keys,
        _symbol_aliases=body._symbol_aliases,
    )._with_metadata(quality=EstimateQuality.UPPER_BOUND)


def _project_portable_aggregate_controlled_cost(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[ResourceEstimate | None, str]:
    """Project the known part of an aggregate arity profile through controls.

    The projection uses one shared conjunction when both the control count and
    modeled operation count reach the aggregate batching threshold.
    Otherwise, it keeps the per-primitive projection used for a one-operation
    controlled body. Gate-name uncertainty is represented by field-wise one-
    and two-qubit envelopes. Any remaining gate count stays as a unit-cost
    opaque serial placeholder without being assigned a false arity.
    Decomposition ancillas for the known portion are added to any scratch width
    already declared by the opaque cost.

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
    profile_constraints = _aggregate_arity_projection_constraints(
        estimate,
        controls,
    )

    single = _portable_controlled_arity_envelope(1, _ONE).repeat(
        estimate.gates.single_qubit
    )
    two = _portable_controlled_arity_envelope(2, _ONE).repeat(estimate.gates.two_qubit)
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
        quality=EstimateQuality.MODELED,
        basis=estimate.basis,
        precision=estimate.precision,
    )
    projected_body = single.seq(two).seq(unresolved)
    shared_projection = _portable_shared_aggregate_control_ladder(
        projected_body,
        controls,
    )
    per_primitive_projection = (
        _portable_controlled_arity_envelope(1, controls)
        .repeat(estimate.gates.single_qubit)
        .seq(
            _portable_controlled_arity_envelope(2, controls).repeat(
                estimate.gates.two_qubit
            )
        )
        .seq(unresolved)
    )
    shares_control_ladder = sp.And(
        sp.Ge(controls, _CONTROL_BATCH_MIN_WEIGHT),
        sp.Ge(estimate.gates.total, _CONTROL_BATCH_MIN_WEIGHT),
    )
    projected = shared_projection.conditional(
        per_primitive_projection,
        shares_control_ladder,
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
            "controlled aggregate cost uses an aggregate-level portable "
            "batching model: at least two modeled operations under at least two "
            "active controls share one computed AND ladder and use "
            "conservative single-control arity upper bounds; smaller cases use "
            "conservative per-primitive arity upper bounds. "
            "Arity fields are independent field-wise upper bounds and may not "
            "sum to total. "
            "The bounds range over Qamomile's supported portable primitive "
            "families. Gate kinds and original scheduling are unavailable, so "
            "the selected gate-name-independent two-control model may differ "
            "from a concrete body's gate-name-specific path. Fallback clean "
            "ancillas are counted in addition to declared opaque workspace, "
            "and any undeclared controlled global-phase overhead is outside "
            "this model"
        )
    )
    unclassified_remainder = _safe_simplify(
        unresolved_count - estimate.gates.multi_qubit
    )
    partial_assumption = ResourceAssumption(
        message=(
            "controlled aggregate cost uses an aggregate-level portable "
            "batching model: at least two modeled operations under at least two "
            "active controls share one computed AND ladder and use "
            "conservative single-control arity upper bounds; smaller cases use "
            "conservative per-primitive arity upper bounds; "
            f"{unresolved_count} gate(s) outside those buckets remain one "
            "modeled operation each in total and serial depth, including "
            f"{unclassified_remainder} gate(s) with unclassified arity; "
            "their controlled decomposition and additional clean ancillas "
            "are unavailable, unclassified gates are not reported as "
            "multi_qubit. Arity fields are independent field-wise upper bounds "
            "and may not sum to total. The supported arity bounds range over "
            "Qamomile's portable primitive families, but the overall result is "
            "a model rather than a full upper bound because the remainder "
            "decomposition is unavailable. The selected gate-name-independent "
            "two-control model may differ from a concrete body's "
            "gate-name-specific path. Declared gate-family counts are retained "
            "only as field-wise floors, and any undeclared controlled "
            "global-phase overhead is outside this model"
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
        quality=estimate.quality,
        basis=estimate.basis,
        precision=estimate.precision,
        _allocation_sites=estimate._allocation_sites,
        _constraints=(
            *estimate._constraints,
            *projected._constraints,
            *profile_constraints,
        ),
        _output_sizes=estimate._output_sizes,
        _input_sizes=estimate._input_sizes,
        _has_output_summary=estimate._has_output_summary,
        _dependency_keys=estimate._dependency_keys,
        _guarded_assumptions=estimate._guarded_assumptions,
        _guarded_qualities=estimate._guarded_qualities,
        _symbol_aliases=estimate._symbol_aliases,
    )
    active_controls = sp.And(
        sp.Gt(controls, _ZERO),
        _resource_activity_condition(_estimate_activity(estimate)),
    )
    if unresolved_count == _ZERO:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            quality=EstimateQuality.MODELED,
            active_when=active_controls,
        )
    elif unresolved_count.is_positive is True:
        controlled = controlled._with_metadata(
            assumptions=(partial_assumption,),
            quality=EstimateQuality.MODELED,
            active_when=active_controls,
        )
    else:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            quality=EstimateQuality.MODELED,
            active_when=sp.And(
                active_controls,
                sp.Eq(unresolved_count, _ZERO),
            ),
        )._with_metadata(
            assumptions=(partial_assumption,),
            quality=EstimateQuality.MODELED,
            active_when=sp.And(
                active_controls,
                sp.Gt(unresolved_count, _ZERO),
            ),
        )
    known_arity_count = _safe_simplify(
        estimate.gates.single_qubit + estimate.gates.two_qubit
    )
    if known_arity_count.is_positive is not True:
        no_profile_reason = (
            "the aggregate has no declared one- or two-qubit gate profile"
        )
        retained = dataclasses.replace(
            estimate,
            trace=_wrap_trace(f"controlled({controls})", estimate.trace),
        )._with_metadata(
            assumptions=(
                _unprojected_aggregate_control_assumption(
                    no_profile_reason,
                    controls,
                ),
            ),
            quality=EstimateQuality.MODELED,
            active_when=active_controls,
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
            _symbol_aliases=estimate._symbol_aliases,
        )
    return controlled, ""


def _estimate_portable_gate(
    operation: GateOperation,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a primitive through the backend-neutral control fallback.

    Args:
        operation (GateOperation): Primitive gate operation.
        controls (ResourceExpr): Surrounding control count.

    Returns:
        ResourceEstimate: Portable logical decomposition estimate.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    return _estimate_portable_named_gate(name, controls)


def _estimate_named_gate_in_basis(
    name: str,
    controls: ResourceExpr,
    *,
    basis: GateBasis,
    precision: float,
) -> ResourceEstimate:
    """Estimate a named primitive in one public resource basis.

    This helper is used for decomposition gates introduced by the estimator
    itself, such as control-value brackets and Pauli-gadget basis changes.

    Args:
        name (str): Lowercase gate name.
        controls (ResourceExpr): Number of additional coherent controls.
        basis (GateBasis): Requested output basis.
        precision (float): Rotation-synthesis precision for ``CLIFFORD_T``.

    Returns:
        ResourceEstimate: Gate, depth, and decomposition-ancilla resources.
    """
    if basis is GateBasis.PORTABLE:
        return _estimate_portable_named_gate(name, controls)

    normalized_name = "toffoli" if name == "ccx" else name
    if basis is GateBasis.CLIFFORD_T:
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
            precision=precision,
        )
        upper_bound_when = _clifford_t_upper_bound_condition(
            normalized_name,
            controls,
        )
        if upper_bound_when is not sp.false:
            estimate = estimate._with_metadata(
                quality=EstimateQuality.UPPER_BOUND,
                active_when=upper_bound_when,
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
    """Return clean ancillas for one canonical Clifford+T fallback.

    Args:
        name (str): Lowercase logical gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.

    Returns:
        ResourceExpr: Reusable clean-ancilla demand.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        controls = surrounding_controls + inherent_controls[name]
        return sp.Max(_ZERO, controls - 2)
    if name in {"z", "y"}:
        return sp.Max(_ZERO, surrounding_controls - 2)
    if name == "cz":
        return sp.Max(_ZERO, surrounding_controls - _ONE)
    if name == "swap":
        return sp.Max(_ZERO, surrounding_controls - _ONE)
    if name == "p":
        return sp.Max(_ZERO, surrounding_controls - _ONE)
    if name == "cp":
        return _piecewise(
            _ZERO,
            surrounding_controls,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"s", "sdg", "t", "tdg"}:
        return surrounding_controls
    return sp.Max(_ZERO, surrounding_controls - _ONE)


def _named_clifford_t_depth(
    name: str,
    surrounding_controls: ResourceExpr,
    gates: GateResources,
) -> DepthResources:
    """Return canonical depth for an estimator-introduced Clifford+T gate.

    Args:
        name (str): Lowercase gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.
        gates (GateResources): Already-classified aggregate gate counts.

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
    return _serial_depth_from_gate_resources(gates)


def _classify_gate(
    operation: GateOperation,
    *,
    num_controls: ResourceExpr | int = 0,
    basis: GateBasis = GateBasis.LOGICAL,
    precision: float = 1e-10,
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


def _clifford_t_upper_bound_condition(
    gate_name: str,
    num_controls: ResourceExpr,
) -> sp.Basic:
    """Return when a Clifford+T gate estimate uses an upper-bound synthesis.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Number of surrounding coherent controls.

    Returns:
        sp.Basic: Boolean activation condition for upper-bound provenance.
    """
    if gate_name in _ROTATION_GATES:
        return sp.true
    exact_controlled = {
        "x",
        "cx",
        "toffoli",
        "swap",
        "y",
        "z",
        "cz",
        "s",
        "sdg",
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
    fixed phase fallbacks. Arbitrary uncontrolled axial rotations use the
    Ross-Selinger asymptotic upper bound
    ``ceil(3 log2(1 / precision))`` T gates. A controlled primitive is
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

    Multi-control lowering mirrors the portable fallback: clean ancillas
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
        ladder_steps = 2 * sp.Max(_ZERO, effective_controls - _ONE)
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            ladder_steps,
        )
        central = _classify_uncontrolled_clifford_t_gate("cp", precision)
        return _add_gates(ladder, central)
    if gate_name in {"s", "sdg", "t", "tdg"}:
        # Compute the conjunction of every control and the target, phase one
        # clean carrier exactly, then uncompute it. This avoids assuming a
        # controlled-T primitive is itself a Clifford+T gate.
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            2 * num_controls,
        )
        return _add_gates(ladder, _classify_uncontrolled_gate(gate_name))
    raise ValueError(
        "Clifford+T lowering is not defined for controlled gate "
        f"'{gate_name}'. Use the logical or portable basis, or provide an "
        "explicit cost model."
    )


def _multi_controlled_x_clifford_t(
    num_controls: ResourceExpr,
) -> GateResources:
    """Lower a multi-controlled X using a clean-ancilla Toffoli ladder.

    Args:
        num_controls (ResourceExpr): Number of controls on the X target.

    Returns:
        GateResources: Exact aggregate Clifford+T counts for the selected
            ladder decomposition.
    """
    toffolis = sp.Max(_ZERO, 2 * num_controls - 3)
    return GateResources(
        total=_resource_expr(
            sp.Piecewise(
                (_ONE, num_controls <= 1),
                (15 * toffolis, True),
            )
        ),
        single_qubit=_resource_expr(
            sp.Piecewise(
                (_ONE, sp.Eq(num_controls, 0)),
                (_ZERO, sp.Eq(num_controls, 1)),
                (9 * toffolis, True),
            )
        ),
        two_qubit=_resource_expr(
            sp.Piecewise(
                (_ZERO, sp.Eq(num_controls, 0)),
                (_ONE, sp.Eq(num_controls, 1)),
                (6 * toffolis, True),
            )
        ),
        clifford=_resource_expr(
            sp.Piecewise(
                (_ONE, num_controls <= 1),
                (8 * toffolis, True),
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
    toffolis = sp.Max(_ZERO, 2 * controls - 3)
    return DepthResources(
        depth=_resource_expr(
            sp.Piecewise((_ONE, controls <= 1), (15 * toffolis, True))
        ),
        clifford_depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (8 * toffolis, True),
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
            sp.Piecewise((_ONE, controls <= 1), (15 * toffolis, True))
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


def _free_symbols(estimate: ResourceEstimate) -> set[sp.Symbol]:
    """Collect all free symbols from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        set[sp.Symbol]: Free symbols used by any metric.
    """
    symbols: set[sp.Symbol] = set()
    for expr in _all_exprs(estimate):
        symbols.update(cast(set[sp.Symbol], sp.sympify(expr).free_symbols))
    for constraint in estimate._constraints:
        constraint_symbols = set(sp.sympify(constraint.expression).free_symbols)
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
        *(estimate._guarded_qualities or ()),
    ):
        symbols.update(cast(set[sp.Symbol], sp.sympify(fact.active_when).free_symbols))
    return symbols


def _serialization_expressions(
    estimate: ResourceEstimate,
) -> list[sp.Basic | int | float]:
    """Return expressions sharing the estimate's public symbol namespace.

    Metric expressions, structural requirements, quantified range variables,
    and guarded metadata must use one identity registry. Otherwise two symbols
    that collide only across separate fields could receive the same public
    name and make the combined payload ambiguous.

    Args:
        estimate (ResourceEstimate): Estimate to serialize.

    Returns:
        list[sp.Basic | int | float]: Expressions in deterministic payload
            encounter order.
    """
    expressions: list[sp.Basic | int | float] = list(_all_exprs(estimate))
    for constraint in estimate._constraints:
        expressions.append(constraint.expression)
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
            *(estimate._guarded_qualities or ()),
        )
    )
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
        if shape and len(shape) != len(ir_value.shape):
            raise ValueError(
                f"array input '{name}' has rank {len(shape)}, but the qkernel "
                f"declares rank {len(ir_value.shape)}."
            )
        if shape:
            consumed.add(name)
        if ir_value.type.is_quantum() and shape:
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
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return len(shape) > 0
        except TypeError:
            return False
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _concrete_input_shape(value: Any) -> tuple[int, ...]:
    """Return the concrete shape of an array-like estimation input.

    Args:
        value (Any): Array-like user input.

    Returns:
        tuple[int, ...]: Concrete dimensions, or an empty tuple when the value
        has no discoverable array shape.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dimension) for dimension in shape)
    dimensions: list[int] = []
    current = value
    while isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        dimensions.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(dimensions)


def _scalar_values(values: Mapping[str, Any]) -> dict[str, sp.Expr]:
    """Keep numeric scalars for branches and physical dependency resolution.

    Accepts Python and NumPy numeric scalars (anything registered as
    ``numbers.Real``, normalized via ``.item()`` when present so a ``np.int64``
    from a notebook works) and SymPy numbers. Dicts, Hamiltonians, and
    symbolic-expression substitution values are dropped: only concrete numbers
    can decide a branch or select one physical array/control index during the
    initial scheduling pass.

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
            out[name] = cast(sp.Expr, sp.sympify(scalar))
        elif isinstance(value, sp.Basic) and value.is_number:
            out[name] = cast(sp.Expr, value)
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
        name for name in inputs if name not in symbols_by_name and name not in known
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
        if not (
            isinstance(value, (bool, numbers.Complex)) or isinstance(value, sp.Expr)
        ):
            raise TypeError(
                f"resource input '{name}' requires a numeric scalar or explicit "
                f"SymPy expression, got {type(value).__name__} ({value!r})."
            )
        if isinstance(value, bool):
            sympified: sp.Basic = sp.Integer(int(value))
        else:
            try:
                sympified = sp.sympify(value)
            except (TypeError, ValueError, sp.SympifyError) as error:
                raise ValueError(
                    f"Cannot apply non-scalar value {value!r} to resource "
                    f"parameter '{name}'."
                ) from error
        if not isinstance(sympified, sp.Expr):
            raise ValueError(
                f"Cannot apply non-numeric value {value!r} to resource "
                f"parameter '{name}'."
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
    )
    if ignored:
        note = ResourceAssumption(
            "input(s) "
            + ", ".join(repr(name) for name in sorted(ignored))
            + " do not affect any resource metric; ignored",
        )
        substituted = substituted._with_metadata(assumptions=(note,))
    return substituted


def _operand_shapes(
    operands: Sequence[Value],
    resolver: ExprResolver,
) -> dict[str, ResourceExpr]:
    """Resolve array operand shapes for a resource context.

    Args:
        operands (Sequence[Value]): Invocation operands.
        resolver (ExprResolver): Resolver for symbolic shapes.

    Returns:
        dict[str, ResourceExpr]: Mapping from operand name to scalar width for
        one-dimensional arrays.
    """
    shapes: dict[str, ResourceExpr] = {}
    for operand in operands:
        if isinstance(operand, ArrayValue) and operand.shape:
            count: ResourceExpr = _ONE
            for dim in operand.shape:
                count *= resolver.resolve(dim)
            shapes[operand.name] = count
    return shapes


def _controlled_u_body_operands(
    operation: ControlledUOperation,
) -> list[Value]:
    """Return wrapped-body actuals after the external control prefix.

    The operation stores controls followed by the wrapped qkernel's complete
    argument list. Slicing that layout directly is the canonical arity
    contract; reconstructing it from ``target_operands`` and
    ``param_operands`` can duplicate classical values because historical
    target accessors include every post-control operand.

    Args:
        operation (ControlledUOperation): Controlled call to inspect.

    Returns:
        list[Value]: Quantum and classical/object body operands in call-site
            storage order.
    """
    return list(operation.operands[len(operation.control_operands) :])


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
    actual_operands = _controlled_u_body_operands(operation)
    for formal, actual in pair_block_operands(block, actual_operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    return resolver.isolated_scope(block, extra)


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
    for formal, actual in pair_block_operands(case_block, actual_operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    return resolver.isolated_scope(case_block, extra)


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
    operands = [*operation.target_qubits, *operation.parameters]
    for formal, actual in pair_block_operands(impl, operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    return resolver.isolated_scope(impl, extra)
