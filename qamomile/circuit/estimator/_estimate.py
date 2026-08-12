"""Represent and compose algorithmic resource estimates."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._config import _DEFAULT_CONTROL_DECOMPOSITION
from qamomile.circuit.estimator._constants import (
    _ONE,
)
from qamomile.circuit.estimator._dependency_indices import WireKey
from qamomile.circuit.estimator._dependency_synchronization import (
    _SynchronizedEntryCertificate,
)
from qamomile.circuit.estimator._estimate_composition import (
    _compose_choice,
    _compose_conditional,
    _compose_parallel,
    _compose_sequential,
    _SequentialEstimateComposer as _SequentialEstimateComposer,
)
from qamomile.circuit.estimator._estimate_loops import _sum_estimate_over_range
from qamomile.circuit.estimator._estimate_provenance import (
    _initialize_estimate_provenance,
    _refresh_symbol_metadata as _refresh_estimate_symbol_metadata,
    _with_estimate_metadata,
)
from qamomile.circuit.estimator._estimate_reporting import (
    _estimate_to_dict,
    _explain_estimate,
)
from qamomile.circuit.estimator._estimate_rewrite import (
    _map_estimate_expressions,
    _simplify_estimate,
    _substitute_estimate,
)
from qamomile.circuit.estimator._estimate_transforms import (
    _control_estimate,
    _invert_estimate,
    _repeat_estimate,
)
from qamomile.circuit.estimator._resource_algebra import _depth_from_gate_resources
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
    _GuardedApproximation,
    _GuardedAssumption,
    _GuardedDerivation,
    _GuardedQuality,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry


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
        control_decomposition (ControlDecomposition): Coherent-control
            decomposition used for the estimate. Defaults to the clean-ancilla
            Toffoli model.
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
        _dependency_synchronized_entry_conditions (dict[WireKey, Boolean]):
            Conditions under which an aggregate depth formula assumes that
            the listed caller-visible wires enter the operation at the same
            dependency layer. The enclosing scheduler marks a result
            conservative when prior work may violate that requirement.
        _dependency_synchronized_entry_certificates (
            tuple[_SynchronizedEntryCertificate, ...]
        ): Grouped synchronized-entry premises. Each certificate keeps its
            complete reset coverage, exact safe first-gate frontier, and
            activation guard together so unrelated frontiers cannot be
            combined. Defaults to an empty tuple.
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
    control_decomposition: ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION
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
    _dependency_synchronized_entry_conditions: dict[WireKey, Boolean] = (
        dataclasses.field(
            default_factory=dict,
            repr=False,
            compare=False,
        )
    )
    _dependency_synchronized_entry_certificates: tuple[
        _SynchronizedEntryCertificate, ...
    ] = dataclasses.field(
        default_factory=tuple,
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
        _initialize_estimate_provenance(self)

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
        return _refresh_estimate_symbol_metadata(
            self,
            registry,
        )

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
        return _with_estimate_metadata(
            self,
            assumptions=assumptions,
            derivation=derivation,
            quality=quality,
            approximation=approximation,
            active_when=active_when,
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
        return _compose_sequential(self, other)

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
            ValueError: If the estimates use incompatible control-decomposition
                provenance.
        """
        composer = _SequentialEstimateComposer(ResourceEstimate.zero())
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
        return _compose_parallel(self, other)

    def choice(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose a conservative branch choice.

        Args:
            other (ResourceEstimate): Alternative branch estimate.

        Returns:
            ResourceEstimate: Element-wise maximum of both branches.
        """
        return _compose_choice(self, other)

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
        return _compose_conditional(self, other, condition)

    def repeat(self, factor: ResourceExpr | int) -> ResourceEstimate:
        """Repeat this estimate with reusable width.

        Args:
            factor (ResourceExpr | int): Iteration or power factor.

        Returns:
            ResourceEstimate: Repeated estimate.

        Raises:
            ValueError: If a concrete factor is negative or non-integral.
        """
        return _repeat_estimate(
            self,
            factor,
            conservative_nonuniform=True,
        )

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
        Aggregate measurement or reset costs fail closed. Body-backed qkernels
        are controlled by the estimator interpreter instead.

        Args:
            num_controls (ResourceExpr | int): Number of active controls.

        Returns:
            ResourceEstimate: Estimate with a recorded controlled assumption.

        Raises:
            ValueError: If a concrete control count or projected gate count is
                negative or non-integral, or if the estimate contains
                measurement or reset resources.
        """
        return _control_estimate(self, num_controls)

    def inverse(self) -> ResourceEstimate:
        """Apply an inverse transform.

        Returns:
            ResourceEstimate: Estimate with identical logical resources.

        Raises:
            ValueError: If the estimate contains measurement or reset
                resources and is therefore not unitary.
        """
        return _invert_estimate(self)

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
            conservative_nonuniform=True,
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
        conservative_nonuniform: bool = False,
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
            conservative_nonuniform (bool): Whether to disclose scalar depth
                summation over a nonuniform per-wire completion profile.

        Returns:
            ResourceEstimate: Estimate with additive metrics summed over the
                loop and width kept reusable.
        """
        return _sum_estimate_over_range(
            self,
            loop_symbol,
            start,
            stop,
            step,
            dependency_start=dependency_start,
            dependency_stop=dependency_stop,
            dependency_step=dependency_step,
            conservative_nonuniform=conservative_nonuniform,
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
        return _substitute_estimate(self, values)

    def simplify(self) -> ResourceEstimate:
        """Simplify all symbolic expressions.

        Returns:
            ResourceEstimate: Simplified estimate.
        """
        return _simplify_estimate(self)

    def explain(self, metric: str | None = None) -> str:
        """Render the resource-estimation trace.

        Args:
            metric (str | None): Optional metric name to mention in the
                heading. Filtering is reserved for a later pass. Defaults to
                ``None``.

        Returns:
            str: Human-readable explanation tree.
        """
        return _explain_estimate(self, metric)

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
        return _estimate_to_dict(self)

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
        return _map_estimate_expressions(
            self,
            fn,
            constraint_fn=constraint_fn,
            guard_fn=guard_fn,
            dependency_fn=dependency_fn,
        )
