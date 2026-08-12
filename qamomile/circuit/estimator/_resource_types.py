"""Define public resource records and guarded estimate metadata."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
    _combine_approximation,
    _combine_derivation,
    _combine_quality,
    _validate_event_count,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
    _rewrite_condition,
    _safe_simplify,
)
from qamomile.circuit.estimator._serialization import (
    SymbolRegistry,
    stringify_expression,
)


@dataclasses.dataclass(frozen=True)
class ResourceAssumption:
    """Record a premise needed to interpret a resource estimate.

    Assumptions disclose modeling choices, recognized approximations, and any
    still-symbolic valid-input condition consumed while simplifying a resource
    formula. A valid-input premise limits where the formula applies; by itself
    it does not make an otherwise exact count conservative.

    Args:
        message (str): Human-readable premise or qualification.
        source (str | None): Optional callable or operation that caused the
            premise. Domain-derived entries use ``"qkernel input domain"``.
            Defaults to ``None``.
    """

    message: str
    source: str | None = None


@dataclasses.dataclass(frozen=True)
class _GuardedAssumption:
    """Associate one modeling assumption with a symbolic activation guard.

    Args:
        active_when (sp.Basic): Boolean condition under which the assumption
            contributes to the selected estimate.
        assumption (ResourceAssumption): Modeling assumption being guarded.
    """

    active_when: sp.Basic
    assumption: ResourceAssumption

    def when(self, condition: sp.Basic) -> _GuardedAssumption:
        """Conjoin another activation condition.

        Args:
            condition (sp.Basic): Additional branch or repetition guard.

        Returns:
            _GuardedAssumption: Assumption guarded by both conditions.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def mapped(self, fn: Any) -> _GuardedAssumption | None:
        """Rewrite the activation guard and prune a false assumption.

        Args:
            fn (Any): Symbolic-expression rewrite function.

        Returns:
            _GuardedAssumption | None: Rewritten fact, or ``None`` when its
                guard resolves false.
        """
        active_when = _rewrite_condition(self.active_when, fn)
        if active_when is sp.false:
            return None
        return dataclasses.replace(self, active_when=active_when)


@dataclasses.dataclass(frozen=True)
class _GuardedDerivation:
    """Associate one modeled-derivation fact with a guard.

    Args:
        active_when (sp.Basic): Boolean condition under which the derivation
            fact contributes.
        derivation (EstimateDerivation): Non-structural derivation being
            guarded.
    """

    active_when: sp.Basic
    derivation: EstimateDerivation

    def when(self, condition: sp.Basic) -> _GuardedDerivation:
        """Conjoin another activation condition.

        Args:
            condition (sp.Basic): Additional branch or repetition guard.

        Returns:
            _GuardedDerivation: Derivation fact guarded by both conditions.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def mapped(self, fn: Any) -> _GuardedDerivation | None:
        """Rewrite the activation guard and prune a false derivation fact.

        Args:
            fn (Any): Symbolic-expression rewrite function.

        Returns:
            _GuardedDerivation | None: Rewritten fact, or ``None`` when its
                guard resolves false.
        """
        active_when = _rewrite_condition(self.active_when, fn)
        if active_when is sp.false:
            return None
        return dataclasses.replace(self, active_when=active_when)


@dataclasses.dataclass(frozen=True)
class _GuardedQuality:
    """Associate one non-exact count quality with a guard.

    Args:
        active_when (sp.Basic): Boolean condition under which the quality
            fact contributes.
        quality (EstimateQuality): Non-exact quality being guarded.
    """

    active_when: sp.Basic
    quality: EstimateQuality

    def when(self, condition: sp.Basic) -> _GuardedQuality:
        """Conjoin another activation condition.

        Args:
            condition (sp.Basic): Additional branch or repetition guard.

        Returns:
            _GuardedQuality: Quality fact guarded by both conditions.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def mapped(self, fn: Any) -> _GuardedQuality | None:
        """Rewrite the activation guard and prune a false quality fact.

        Args:
            fn (Any): Symbolic-expression rewrite function.

        Returns:
            _GuardedQuality | None: Rewritten fact, or ``None`` when its
                guard resolves false.
        """
        active_when = _rewrite_condition(self.active_when, fn)
        if active_when is sp.false:
            return None
        return dataclasses.replace(self, active_when=active_when)


@dataclasses.dataclass(frozen=True)
class _GuardedApproximation:
    """Associate an approximation status with a symbolic activation guard.

    Args:
        active_when (sp.Basic): Condition under which the approximation is
            present.
        approximation (ApproximationStatus): Non-exact approximation status.
    """

    active_when: sp.Basic
    approximation: ApproximationStatus

    def when(self, condition: sp.Basic) -> _GuardedApproximation:
        """Conjoin another activation condition.

        Args:
            condition (sp.Basic): Additional branch or repetition guard.

        Returns:
            _GuardedApproximation: Status guarded by both conditions.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def mapped(self, fn: Any) -> _GuardedApproximation | None:
        """Rewrite the guard and prune an inactive approximation.

        Args:
            fn (Any): Symbolic-expression rewrite function.

        Returns:
            _GuardedApproximation | None: Rewritten status, or ``None`` when
                its guard resolves false.
        """
        active_when = _rewrite_condition(self.active_when, fn)
        if active_when is sp.false:
            return None
        return dataclasses.replace(self, active_when=active_when)


def _active_assumptions(
    facts: Sequence[_GuardedAssumption],
) -> tuple[ResourceAssumption, ...]:
    """Return assumptions whose guards have not resolved false.

    Args:
        facts (Sequence[_GuardedAssumption]): Guarded assumption provenance.

    Returns:
        tuple[ResourceAssumption, ...]: Active or potentially active
            assumptions, deduplicated in encounter order.
    """
    active: list[ResourceAssumption] = []
    for fact in facts:
        if fact.active_when is not sp.false and fact.assumption not in active:
            active.append(fact.assumption)
    return tuple(active)


def _active_derivation(
    facts: Sequence[_GuardedDerivation],
) -> EstimateDerivation:
    """Return whether any active fact uses modeled derivation.

    Args:
        facts (Sequence[_GuardedDerivation]): Guarded derivation facts.

    Returns:
        EstimateDerivation: Active or potentially active derivation.
    """
    derivation = EstimateDerivation.STRUCTURAL
    for fact in facts:
        if fact.active_when is not sp.false:
            derivation = _combine_derivation(derivation, fact.derivation)
    return derivation


def _active_quality(
    facts: Sequence[_GuardedQuality],
) -> EstimateQuality:
    """Return the weakest active or potentially active count quality.

    Args:
        facts (Sequence[_GuardedQuality]): Guarded quality facts.

    Returns:
        EstimateQuality: Active or potentially active quality.
    """
    quality = EstimateQuality.EXACT
    for fact in facts:
        if fact.active_when is not sp.false:
            quality = _combine_quality(quality, fact.quality)
    return quality


def _active_approximation(
    facts: Sequence[_GuardedApproximation],
) -> ApproximationStatus:
    """Return the active or potentially active approximation status.

    Args:
        facts (Sequence[_GuardedApproximation]): Guarded approximation facts.

    Returns:
        ApproximationStatus: Combined active approximation status.
    """
    approximation = ApproximationStatus.EXACT
    for fact in facts:
        if fact.active_when is not sp.false:
            approximation = _combine_approximation(
                approximation,
                fact.approximation,
            )
    return approximation


@dataclasses.dataclass
class WidthResources:
    """Track logical width and ancilla resources.

    Args:
        input_qubits (ResourceExpr): Qubits supplied by the caller.
        allocated_qubits (ResourceExpr): Qubits allocated by the body.
        clean_ancilla_qubits (ResourceExpr): Clean ancilla qubits required at
            peak. Defaults to zero.
        dirty_ancilla_qubits (ResourceExpr): Dirty ancilla qubits required at
            peak. Defaults to zero.
        peak_qubits (ResourceExpr): Conservative peak logical width.
    """

    input_qubits: ResourceExpr = _ZERO
    allocated_qubits: ResourceExpr = _ZERO
    clean_ancilla_qubits: ResourceExpr = _ZERO
    dirty_ancilla_qubits: ResourceExpr = _ZERO
    peak_qubits: ResourceExpr = _ZERO

    @property
    def circuit_qubits(self) -> ResourceExpr:
        """Return the conservative static circuit width.

        Unlike ``peak_qubits``, which follows logical wire liveness, a static
        circuit must reserve every distinct allocation site plus reusable
        decomposition ancillas up front.

        Returns:
            ResourceExpr: Input, allocated, clean-ancilla, and dirty-ancilla
            widths summed into one static-width expression.
        """
        return (
            self.input_qubits
            + self.allocated_qubits
            + self.clean_ancilla_qubits
            + self.dirty_ancilla_qubits
        )

    @staticmethod
    def zero() -> WidthResources:
        """Return a zero width estimate.

        Returns:
            WidthResources: Empty width resources.
        """
        return WidthResources()

    def simplify(self) -> WidthResources:
        """Simplify all width expressions.

        Returns:
            WidthResources: Simplified copy.
        """
        return WidthResources(
            input_qubits=_safe_simplify(self.input_qubits),
            allocated_qubits=_safe_simplify(self.allocated_qubits),
            clean_ancilla_qubits=_safe_simplify(self.clean_ancilla_qubits),
            dirty_ancilla_qubits=_safe_simplify(self.dirty_ancilla_qubits),
            peak_qubits=_safe_simplify(self.peak_qubits),
        )


@dataclasses.dataclass
class GateResources:
    """Track logical gate resources.

    Args:
        total (ResourceExpr): Total logical gate count.
        single_qubit (ResourceExpr): Single-qubit gate count.
        two_qubit (ResourceExpr): Two-qubit gate count.
        multi_qubit (ResourceExpr): Three-or-more-qubit gate count.
        clifford (ResourceExpr): Clifford gate count.
        rotation (ResourceExpr): Parametric rotation gate count.
        t (ResourceExpr): T/T-dagger gate count.
        toffoli (ResourceExpr): Toffoli gate count.
        non_clifford (ResourceExpr): Non-Clifford gate count.
    """

    total: ResourceExpr = _ZERO
    single_qubit: ResourceExpr = _ZERO
    two_qubit: ResourceExpr = _ZERO
    multi_qubit: ResourceExpr = _ZERO
    clifford: ResourceExpr = _ZERO
    rotation: ResourceExpr = _ZERO
    t: ResourceExpr = _ZERO
    toffoli: ResourceExpr = _ZERO
    non_clifford: ResourceExpr = _ZERO

    @property
    def t_gates(self) -> ResourceExpr:
        """Return the T-gate count alias.

        Returns:
            ResourceExpr: The same value as ``t``.
        """
        return self.t

    @property
    def clifford_gates(self) -> ResourceExpr:
        """Return the Clifford-gate count alias.

        Returns:
            ResourceExpr: The same value as ``clifford``.
        """
        return self.clifford

    @property
    def rotation_gates(self) -> ResourceExpr:
        """Return the rotation-gate count alias.

        Returns:
            ResourceExpr: The same value as ``rotation``.
        """
        return self.rotation

    @staticmethod
    def zero() -> GateResources:
        """Return a zero gate estimate.

        Returns:
            GateResources: Empty gate resources.
        """
        return GateResources()

    def simplify(self) -> GateResources:
        """Simplify all gate expressions.

        Returns:
            GateResources: Simplified copy.
        """
        return dataclasses.replace(
            self,
            total=_safe_simplify(self.total),
            single_qubit=_safe_simplify(self.single_qubit),
            two_qubit=_safe_simplify(self.two_qubit),
            multi_qubit=_safe_simplify(self.multi_qubit),
            clifford=_safe_simplify(self.clifford),
            rotation=_safe_simplify(self.rotation),
            t=_safe_simplify(self.t),
            toffoli=_safe_simplify(self.toffoli),
            non_clifford=_safe_simplify(self.non_clifford),
        )


@dataclasses.dataclass
class MeasurementResources:
    """Track logical measurement resources.

    Args:
        total (ResourceExpr): Number of per-qubit measurement events. Measuring
            an ``N``-qubit vector contributes ``N``, independently of how many
            source-level or IR operations express the measurement.

    Raises:
        ValueError: If ``total`` is a concrete value that is not a
            nonnegative integer.
    """

    total: ResourceExpr = _ZERO

    def __post_init__(self) -> None:
        """Validate the concrete measurement count.

        Raises:
            ValueError: If ``total`` is a concrete value that is not a
                nonnegative integer.
        """
        _validate_event_count(self.total, label="Measurement count")

    @staticmethod
    def zero() -> MeasurementResources:
        """Return a zero measurement estimate.

        Returns:
            MeasurementResources: Empty measurement resources.
        """
        return MeasurementResources()

    def simplify(self) -> MeasurementResources:
        """Simplify all measurement expressions.

        Returns:
            MeasurementResources: Simplified copy.
        """
        return dataclasses.replace(
            self,
            total=_safe_simplify(self.total),
        )


@dataclasses.dataclass
class ResetResources:
    """Track logical reset resources.

    Args:
        total (ResourceExpr): Number of per-qubit reset events.

    Raises:
        ValueError: If ``total`` is a concrete value that is not a
            nonnegative integer.
    """

    total: ResourceExpr = _ZERO

    def __post_init__(self) -> None:
        """Validate the concrete reset count.

        Raises:
            ValueError: If ``total`` is a concrete value that is not a
                nonnegative integer.
        """
        _validate_event_count(self.total, label="Reset count")

    @staticmethod
    def zero() -> ResetResources:
        """Return a zero reset estimate.

        Returns:
            ResetResources: Empty reset resources.
        """
        return ResetResources()

    def simplify(self) -> ResetResources:
        """Simplify all reset expressions.

        Returns:
            ResetResources: Simplified copy.
        """
        return dataclasses.replace(
            self,
            total=_safe_simplify(self.total),
        )


@dataclasses.dataclass
class DepthResources:
    """Track logical depth resources.

    Args:
        depth (ResourceExpr): Total logical depth.
        clifford_depth (ResourceExpr): Clifford-layer depth.
        rotation_depth (ResourceExpr): Rotation-layer depth.
        t_depth (ResourceExpr): T-layer depth.
        toffoli_depth (ResourceExpr): Toffoli-layer depth.
        non_clifford_depth (ResourceExpr): Non-Clifford-layer depth.
        measurement_depth (ResourceExpr): Measurement-layer depth.
        gate_depth (ResourceExpr): Gate-only logical depth.
        reset_depth (ResourceExpr): Reset-layer depth.
    """

    depth: ResourceExpr = _ZERO
    clifford_depth: ResourceExpr = _ZERO
    rotation_depth: ResourceExpr = _ZERO
    t_depth: ResourceExpr = _ZERO
    toffoli_depth: ResourceExpr = _ZERO
    non_clifford_depth: ResourceExpr = _ZERO
    measurement_depth: ResourceExpr = _ZERO
    gate_depth: ResourceExpr = _ZERO
    reset_depth: ResourceExpr = _ZERO

    @staticmethod
    def zero() -> DepthResources:
        """Return a zero depth estimate.

        Returns:
            DepthResources: Empty depth resources.
        """
        return DepthResources()

    def simplify(self) -> DepthResources:
        """Simplify all depth expressions.

        Returns:
            DepthResources: Simplified copy.
        """
        return dataclasses.replace(
            self,
            depth=_safe_simplify(self.depth),
            clifford_depth=_safe_simplify(self.clifford_depth),
            rotation_depth=_safe_simplify(self.rotation_depth),
            t_depth=_safe_simplify(self.t_depth),
            toffoli_depth=_safe_simplify(self.toffoli_depth),
            non_clifford_depth=_safe_simplify(self.non_clifford_depth),
            measurement_depth=_safe_simplify(self.measurement_depth),
            gate_depth=_safe_simplify(self.gate_depth),
            reset_depth=_safe_simplify(self.reset_depth),
        )


@dataclasses.dataclass
class CallResources:
    """Track opaque callable and oracle query resources.

    Body-backed qkernel calls are recursively expanded into their primitive
    resources and therefore do not appear here. These maps retain only calls
    whose body is intentionally opaque under the selected policy or model.

    Args:
        calls_by_name (dict[str, ResourceExpr]): Opaque invocation count by
            callable name.
        queries_by_name (dict[str, ResourceExpr]): Opaque query complexity by
            callable name.
    """

    calls_by_name: dict[str, ResourceExpr] = dataclasses.field(default_factory=dict)
    queries_by_name: dict[str, ResourceExpr] = dataclasses.field(default_factory=dict)

    @property
    def oracle_calls(self) -> dict[str, ResourceExpr]:
        """Return oracle-call compatible aliases.

        Returns:
            dict[str, ResourceExpr]: Same mapping as ``calls_by_name``.
        """
        return self.calls_by_name

    @property
    def oracle_queries(self) -> dict[str, ResourceExpr]:
        """Return oracle-query compatible aliases.

        Returns:
            dict[str, ResourceExpr]: Same mapping as ``queries_by_name``.
        """
        return self.queries_by_name

    @staticmethod
    def zero() -> CallResources:
        """Return a zero call estimate.

        Returns:
            CallResources: Empty call resources.
        """
        return CallResources()

    def simplify(self) -> CallResources:
        """Simplify all call expressions.

        Returns:
            CallResources: Simplified copy.
        """
        return CallResources(
            calls_by_name={
                name: simplified
                for name, count in self.calls_by_name.items()
                if (simplified := _safe_simplify(count)) != _ZERO
            },
            queries_by_name={
                name: simplified
                for name, count in self.queries_by_name.items()
                if (simplified := _safe_simplify(count)) != _ZERO
            },
        )


@dataclasses.dataclass
class ResourceTraceNode:
    """Represent one node in the resource-estimation explanation tree.

    Args:
        name (str): Operation or callable name.
        source_kind (str): Source type such as ``"primitive"``, ``"body"``,
            ``"opaque_cost"``, or ``"opaque"``.
        strategy (str | None): Selected resource strategy. Defaults to
            ``None``.
        summary (str): Short expression summary. Defaults to an empty string.
        assumptions (tuple[ResourceAssumption, ...]): Assumptions local to the
            node. Defaults to an empty tuple.
        children (tuple[ResourceTraceNode, ...]): Nested trace nodes.
            Defaults to an empty tuple.
        active_when (sp.Basic): Symbolic activation condition. Defaults to
            true.
    """

    name: str
    source_kind: str
    strategy: str | None = None
    summary: str = ""
    assumptions: tuple[ResourceAssumption, ...] = ()
    children: tuple[ResourceTraceNode, ...] = ()
    active_when: sp.Basic = sp.true

    def when(self, condition: sp.Basic) -> ResourceTraceNode:
        """Return this trace guarded by an additional condition.

        Args:
            condition (sp.Basic): Branch or repetition activation guard.

        Returns:
            ResourceTraceNode: Trace guarded by both conditions.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def mapped(self, fn: Any) -> ResourceTraceNode | None:
        """Rewrite activation guards and remove inactive trace branches.

        Args:
            fn (Any): Symbolic-expression rewrite function.

        Returns:
            ResourceTraceNode | None: Rewritten trace, or ``None`` when this
                node resolves inactive.
        """
        active_when = _rewrite_condition(self.active_when, fn)
        if active_when is sp.false:
            return None
        children = tuple(
            mapped
            for child in self.children
            if (mapped := child.mapped(fn)) is not None
        )
        return dataclasses.replace(
            self,
            children=children,
            active_when=active_when,
        )

    def render(
        self,
        indent: int = 0,
        registry: SymbolRegistry | None = None,
    ) -> str:
        """Render this trace node as plain text.

        Args:
            indent (int): Number of leading spaces. Defaults to ``0``.
            registry (SymbolRegistry | None): Shared estimate symbol registry.
                Defaults to a registry local to each activation condition.

        Returns:
            str: Multi-line explanation text.
        """
        prefix = " " * indent
        strategy = f" strategy={self.strategy}" if self.strategy else ""
        summary = f" {self.summary}" if self.summary else ""
        guard = (
            ""
            if self.active_when is sp.true
            else f" when={stringify_expression(self.active_when, registry)}"
        )
        lines = [f"{prefix}{self.name} [{self.source_kind}{strategy}]{summary}{guard}"]
        for assumption in self.assumptions:
            source = f" ({assumption.source})" if assumption.source else ""
            lines.append(f"{prefix}  assumption: {assumption.message}{source}")
        for child in self.children:
            lines.append(child.render(indent + 2, registry))
        return "\n".join(lines)
