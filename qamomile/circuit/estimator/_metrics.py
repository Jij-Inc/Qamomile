"""Resource metric values and symbolic composition primitives.

This module contains the IR-independent part of resource estimation: public
metric records, structural constraints, and field-wise SymPy algebra. Keeping
these definitions independent from interpretation and scheduling makes their
semantics reusable without importing the qkernel IR.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, cast

import sympy as sp
from sympy.calculus.util import minimum as calculus_minimum
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._serialization import (
    SymbolRegistry,
    stringify_expression,
)

ResourceExpr = sp.Expr
_ZERO = sp.Integer(0)
_ONE = sp.Integer(1)


def _symbol_display_name(symbol: sp.Basic) -> str:
    """Return a stable user-facing name for a SymPy symbol.

    ``Dummy`` symbols print with a leading underscore even though their
    declared ``name`` is unchanged. Resource input/substitution APIs are keyed
    by the declared Qamomile name, so identity-distinct dummies must retain the
    same external spelling as ordinary symbols.

    Args:
        symbol (sp.Basic): Symbol or identity-distinct ``Dummy`` to name.

    Returns:
        str: The symbol's declared name without SymPy's dummy-printing prefix.
    """
    name = getattr(symbol, "name", None)
    return name if isinstance(name, str) else str(symbol)


def _is_concrete_integer(value: sp.Expr) -> bool:
    """Return whether a concrete SymPy number is integer-valued.

    SymPy leaves ``Float(1.0).is_integer`` undecided even though accepting an
    integer-valued NumPy or Python float for a UInt input is useful and
    unambiguous.

    Args:
        value (sp.Expr): Concrete numeric expression.

    Returns:
        bool: Whether the value is mathematically integral.
    """
    if value.is_integer is True:
        return True
    if value.is_number and value.is_real is True:
        return sp.Eq(value, sp.floor(value)) is sp.true
    return False


def _combine_derivation(
    left: EstimateDerivation,
    right: EstimateDerivation,
) -> EstimateDerivation:
    """Return whether either estimate depends on a resource model.

    Args:
        left (EstimateDerivation): Left derivation classification.
        right (EstimateDerivation): Right derivation classification.

    Returns:
        EstimateDerivation: ``MODELED`` if either input is modeled.
    """
    if EstimateDerivation.MODELED in {left, right}:
        return EstimateDerivation.MODELED
    return EstimateDerivation.STRUCTURAL


def _combine_guarantee(
    left: EstimateGuarantee,
    right: EstimateGuarantee,
) -> EstimateGuarantee:
    """Return the weakest resource-count guarantee of two estimates.

    Args:
        left (EstimateGuarantee): Left count guarantee.
        right (EstimateGuarantee): Right count guarantee.

    Returns:
        EstimateGuarantee: Combined count guarantee.
    """
    rank = {
        EstimateGuarantee.EXACT: 0,
        EstimateGuarantee.UPPER_BOUND: 1,
        EstimateGuarantee.UNKNOWN: 2,
    }
    return left if rank[left] >= rank[right] else right


def _combine_approximation(
    left: ApproximationStatus,
    right: ApproximationStatus,
) -> ApproximationStatus:
    """Return whether either composed estimate uses an approximation.

    Args:
        left (ApproximationStatus): Left approximation status.
        right (ApproximationStatus): Right approximation status.

    Returns:
        ApproximationStatus: ``APPROXIMATE`` if either input is approximate.
    """
    if ApproximationStatus.APPROXIMATE in {left, right}:
        return ApproximationStatus.APPROXIMATE
    return ApproximationStatus.EXACT


def _validate_event_count(value: ResourceExpr, *, label: str) -> None:
    """Validate one concrete per-qubit event count.

    Args:
        value (ResourceExpr): Concrete or symbolic event-count expression.
        label (str): User-facing resource label for diagnostics.

    Raises:
        ValueError: If a concrete value is Boolean, negative, non-finite, or
            non-integral.
    """
    sympified = sp.sympify(value)
    if sympified is sp.true or sympified is sp.false:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}.")
    expression = cast(sp.Expr, sympified)
    if expression.is_number and (
        expression.is_finite is not True
        or expression.is_negative is True
        or not _is_concrete_integer(expression)
    ):
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}.")


class GateBasis(enum.StrEnum):
    """Select the gate basis reported by resource estimation.

    Values:
        LOGICAL: Report gates in Qamomile's logical algorithmic vocabulary.
        CLIFFORD_T: Lower the supported logical operations to aggregate
            Clifford+T resources at the requested synthesis precision.
    """

    LOGICAL = "logical"
    CLIFFORD_T = "clifford_t"


class ControlDecomposition(enum.StrEnum):
    """Select how coherent controls are represented in resource estimates.

    Values:
        ABSTRACT: Keep each controlled primitive as one abstract logical
            operation, independently of its control arity. A nonzero abstract
            control cannot be expressed in the ``CLIFFORD_T`` basis, so that
            combination fails instead of inventing a decomposition.
        CLEAN_ANCILLA_TOFFOLI: Use the fixed clean-ancilla Toffoli-ladder
            resource model, including its body-wide sharing rule. This
            algorithmic model is independent of any engine's native or
            fallback emission policy.
    """

    ABSTRACT = "abstract"
    CLEAN_ANCILLA_TOFFOLI = "clean_ancilla_toffoli"


class EstimateDerivation(enum.StrEnum):
    """Describe how the estimator obtained the reported resource counts.

    Values:
        STRUCTURAL: Derive counts recursively from visible IR and the selected
            estimator decomposition rules.
        MODELED: Use a declared, callback-provided, or fallback resource model
            for at least one part of the estimate.
    """

    STRUCTURAL = "structural"
    MODELED = "modeled"


class EstimateGuarantee(enum.StrEnum):
    """Describe how reported counts relate to the selected circuit cost.

    This axis is independent of both count derivation and mathematical
    approximation. A modeled estimate can therefore be exact, an upper bound,
    or unknown with respect to the selected resource model.

    Values:
        EXACT: Reported counts exactly follow the selected circuit model.
        UPPER_BOUND: Reported counts are conservative upper bounds.
        UNKNOWN: No exact or upper-bound guarantee is available.
    """

    EXACT = "exact"
    UPPER_BOUND = "upper_bound"
    UNKNOWN = "unknown"


class ApproximationStatus(enum.StrEnum):
    """Describe whether the selected circuit approximates an ideal operation.

    Values:
        EXACT: No mathematical approximation is known to the estimator.
        APPROXIMATE: At least one selected circuit construction approximates
            its ideal mathematical operation.
    """

    EXACT = "exact"
    APPROXIMATE = "approximate"


@dataclasses.dataclass(frozen=True)
class ResourceAssumption:
    """Record a modeling assumption made during resource estimation.

    Args:
        message (str): Human-readable assumption text.
        source (str | None): Optional callable or operation that caused the
            assumption. Defaults to ``None``.
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
class _GuardedGuarantee:
    """Associate one non-exact count guarantee with a guard.

    Args:
        active_when (sp.Basic): Boolean condition under which the guarantee
            fact contributes.
        guarantee (EstimateGuarantee): Non-exact guarantee being guarded.
    """

    active_when: sp.Basic
    guarantee: EstimateGuarantee

    def when(self, condition: sp.Basic) -> _GuardedGuarantee:
        """Conjoin another activation condition.

        Args:
            condition (sp.Basic): Additional branch or repetition guard.

        Returns:
            _GuardedGuarantee: Guarantee fact guarded by both conditions.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def mapped(self, fn: Any) -> _GuardedGuarantee | None:
        """Rewrite the activation guard and prune a false guarantee fact.

        Args:
            fn (Any): Symbolic-expression rewrite function.

        Returns:
            _GuardedGuarantee | None: Rewritten fact, or ``None`` when its
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


def _active_guarantee(
    facts: Sequence[_GuardedGuarantee],
) -> EstimateGuarantee:
    """Return the weakest active or potentially active count guarantee.

    Args:
        facts (Sequence[_GuardedGuarantee]): Guarded guarantee facts.

    Returns:
        EstimateGuarantee: Active or potentially active guarantee.
    """
    guarantee = EstimateGuarantee.EXACT
    for fact in facts:
        if fact.active_when is not sp.false:
            guarantee = _combine_guarantee(guarantee, fact.guarantee)
    return guarantee


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


@dataclasses.dataclass(frozen=True)
class _ConstraintRange:
    """Describe one quantified loop range for a structural requirement.

    Args:
        symbol (sp.Symbol): Internal loop variable.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.
    """

    symbol: sp.Symbol
    start: ResourceExpr
    step: ResourceExpr
    iterations: ResourceExpr

    def mapped(self, fn: Any) -> _ConstraintRange:
        """Rewrite external expressions in this loop range.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.

        Returns:
            _ConstraintRange: Rewritten quantified range.
        """
        return dataclasses.replace(
            self,
            start=fn(self.start),
            step=fn(self.step),
            iterations=fn(self.iterations),
        )


@dataclasses.dataclass(frozen=True)
class _ResourceConstraint:
    """Retain a structural requirement across symbolic estimation.

    Resource metrics alone cannot preserve every legality condition. For
    example, a zero control count or SELECT address width may simplify gate
    counts to zero even though the underlying operation is invalid. These
    constraints travel with an estimate and are rechecked whenever symbolic
    expressions are rewritten.

    Args:
        expression (ResourceExpr): Symbolic value constrained by the IR
            operation.
        minimum (int | None): Lower bound, or ``None`` when only an equality
            is required.
        label (str): User-facing name of the constrained value.
        unit (str): Optional singular unit appended to diagnostics. Defaults
            to an empty string.
        integer (bool): Whether concrete values must be integers. Defaults to
            ``True``.
        minimum_inclusive (bool): Whether ``minimum`` itself is accepted.
            Defaults to ``True``.
        finite (bool): Whether concrete values must be finite. Defaults to
            ``False``.
        expected (ResourceExpr | None): Required exact value. Defaults to
            ``None``.
        ranges (tuple[_ConstraintRange, ...]): Outer-to-inner loop ranges that
            quantify internal symbols in ``expression``. Defaults to an empty
            tuple.
    """

    expression: ResourceExpr
    minimum: int | None
    label: str
    unit: str = ""
    integer: bool = True
    minimum_inclusive: bool = True
    finite: bool = False
    expected: ResourceExpr | None = None
    ranges: tuple[_ConstraintRange, ...] = ()

    def mapped(self, fn: Any) -> _ResourceConstraint:
        """Rewrite and validate the constrained expression.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.

        Returns:
            _ResourceConstraint: Rewritten structural constraint.

        Raises:
            ValueError: If rewriting resolves the expression to an invalid
                concrete value.
        """
        mapped = dataclasses.replace(
            self,
            expression=fn(self.expression),
            expected=(fn(self.expected) if self.expected is not None else None),
            ranges=tuple(loop_range.mapped(fn) for loop_range in self.ranges),
        )
        mapped.validate()
        return mapped

    def when(self, condition: sp.Basic) -> _ResourceConstraint:
        """Make this requirement vacuous outside one symbolic branch.

        Args:
            condition (sp.Basic): Boolean condition selecting the branch that
                owns this requirement.

        Returns:
            _ResourceConstraint: Conditionally active requirement.
        """
        predicate = _boolean_condition(condition)
        fallback = self._valid_fallback()
        expected = self.expected
        if expected is not None:
            expected = _piecewise(expected, _ZERO, predicate)
            fallback = _ZERO
        return dataclasses.replace(
            self,
            expression=_piecewise(self.expression, fallback, predicate),
            expected=expected,
        )

    def validate(self) -> None:
        """Reject an invalid concrete structural value.

        Raises:
            ValueError: If the expression is concrete and violates its
                integer or lower-bound requirement.
        """
        self._validate_ranges(0, {}, [4096])

    def _valid_fallback(self) -> sp.Integer:
        """Return one concrete value satisfying this lower-bound requirement.

        Conditional constraints use this value outside their active branch.
        Equality requirements replace it separately with zero.

        Returns:
            sp.Integer: Finite integer that satisfies the lower bound.
        """
        if self.minimum is None:
            return _ZERO
        offset = 0 if self.minimum_inclusive else 1
        return sp.Integer(self.minimum + offset)

    def _validate_ranges(
        self,
        range_index: int,
        substitutions: Mapping[sp.Symbol, sp.Expr],
        budget: list[int],
    ) -> None:
        """Validate all concrete points in quantified loop ranges.

        Args:
            range_index (int): Current range position.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Values already bound
                by enclosing ranges.
            budget (list[int]): Remaining exhaustive validation points, stored
                in a mutable single-item list across recursive calls.

        Raises:
            ValueError: If a concrete range or constrained value is invalid.
        """
        if range_index == len(self.ranges):
            resolved = _safe_constraint_substitute(self.expression, substitutions)
            if resolved.is_number:
                self._validate_value(resolved, substitutions)
                budget[0] -= 1
            return

        loop_range = self.ranges[range_index]
        start = _safe_constraint_substitute(loop_range.start, substitutions)
        step = _safe_constraint_substitute(loop_range.step, substitutions)
        iterations = _safe_constraint_substitute(
            loop_range.iterations,
            substitutions,
        )
        if not all(value.is_number for value in (start, step, iterations)):
            return
        if not all(value.is_integer is True for value in (start, step, iterations)):
            raise ValueError(
                f"Cannot validate {self.label}: its quantified loop range "
                "must resolve to integer values."
            )
        count = int(iterations)
        if count < 0:
            raise ValueError(
                f"Cannot validate {self.label}: loop iterations must be "
                f"nonnegative; got {count}."
            )
        if count > budget[0]:
            if range_index == len(self.ranges) - 1 and self._validate_large_range(
                loop_range,
                start,
                step,
                count,
                substitutions,
            ):
                return
            raise ValueError(
                f"Cannot validate {self.label} exhaustively across {count} "
                "loop iterations; simplify the constrained expression or "
                "estimate with a smaller bound."
            )
        for offset in range(count):
            value = start + step * offset
            self._validate_ranges(
                range_index + 1,
                {**substitutions, loop_range.symbol: value},
                budget,
            )

    def _validate_large_range(
        self,
        loop_range: _ConstraintRange,
        start: sp.Expr,
        step: sp.Expr,
        count: int,
        substitutions: Mapping[sp.Symbol, sp.Expr],
    ) -> bool:
        """Prove a large one-dimensional quantified requirement analytically.

        Args:
            loop_range (_ConstraintRange): Final quantified range.
            start (sp.Expr): Concrete first loop value.
            step (sp.Expr): Concrete loop step.
            count (int): Concrete positive iteration count.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Outer loop values.

        Returns:
            bool: Whether integrality and the lower bound were proven without
            exhaustive enumeration.

        Raises:
            ValueError: If an analytically selected integer point violates the
                requirement.
        """
        expression = _safe_constraint_substitute(self.expression, substitutions)
        expected = (
            _safe_constraint_substitute(self.expected, substitutions)
            if self.expected is not None
            else None
        )
        external_symbols = expression.free_symbols - {loop_range.symbol}
        if expected is not None:
            external_symbols.update(expected.free_symbols - {loop_range.symbol})
        if external_symbols:
            return True

        index = sp.Dummy("constraint_index", integer=True, nonnegative=True)
        indexed = cast(
            ResourceExpr,
            expression.subs(loop_range.symbol, start + step * index),
        )
        indexed_expected = (
            cast(
                ResourceExpr,
                expected.subs(loop_range.symbol, start + step * index),
            )
            if expected is not None
            else None
        )
        integer_proven = not self.integer or indexed.is_integer is True
        if self.integer and not integer_proven:
            try:
                polynomial = sp.Poly(indexed, index)
            except sp.PolynomialError:
                polynomial = None
            integer_proven = polynomial is not None and all(
                coefficient.is_integer is True
                for coefficient in polynomial.all_coeffs()
            )
        finite_proven = not self.finite or indexed.is_finite is True
        real_proven = self.minimum is None or indexed.is_real is True

        if indexed_expected is not None:
            difference = _safe_simplify(cast(ResourceExpr, indexed - indexed_expected))
            if difference == _ZERO and integer_proven and finite_proven and real_proven:
                return True
            for offset in {0, count - 1}:
                resolved = _safe_constraint_substitute(
                    indexed,
                    {index: sp.Integer(offset)},
                )
                self._validate_value(
                    resolved,
                    {
                        **substitutions,
                        loop_range.symbol: start + step * offset,
                    },
                )
            return False

        domain = sp.Interval(_ZERO, sp.Integer(count - 1))
        try:
            lower_bound = calculus_minimum(indexed, index, domain)
        except (NotImplementedError, RecursionError, TypeError, ValueError):
            lower_bound = None
        if (
            integer_proven
            and finite_proven
            and real_proven
            and self.minimum is not None
            and isinstance(lower_bound, sp.Expr)
            and lower_bound.is_number
            and self._minimum_relation(lower_bound) is sp.true
        ):
            return True

        try:
            polynomial = sp.Poly(indexed, index)
        except sp.PolynomialError:
            return False
        stationary = sp.solveset(
            sp.diff(indexed, index),
            index,
            domain=domain,
        )
        if not isinstance(stationary, sp.FiniteSet):
            return False
        candidates = {0, count - 1}
        for point in stationary:
            if point.is_real is not True:
                continue
            try:
                neighbors = (int(sp.floor(point)), int(sp.ceiling(point)))
            except TypeError:
                return False
            candidates.update(
                neighbor for neighbor in neighbors if 0 <= neighbor < count
            )
        for offset in candidates:
            resolved = _safe_constraint_substitute(
                indexed,
                {index: sp.Integer(offset)},
            )
            if not resolved.is_number:
                return False
            self._validate_value(
                resolved,
                {
                    **substitutions,
                    loop_range.symbol: start + step * offset,
                },
            )
        return integer_proven and finite_proven and real_proven

    def _validate_value(
        self,
        resolved: sp.Expr,
        substitutions: Mapping[sp.Symbol, sp.Expr],
    ) -> None:
        """Validate one concrete constrained value.

        Args:
            resolved (sp.Expr): Concrete value to validate.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Quantified loop values
                used to resolve it.

        Raises:
            ValueError: If the value is non-finite, non-integral, or outside
                the lower bound.
        """
        location = ""
        if substitutions:
            assignments = ", ".join(
                f"{_symbol_display_name(symbol)}={value}"
                for symbol, value in substitutions.items()
            )
            location = f" at {assignments}"
        if self.finite and resolved.is_finite is not True:
            raise ValueError(f"{self.label} must be finite; got {resolved}{location}.")
        if self.minimum is not None and resolved.is_real is not True:
            raise ValueError(f"{self.label} must be real; got {resolved}{location}.")
        if self.integer and not _is_concrete_integer(resolved):
            raise ValueError(
                f"{self.label} must be an integer; got {resolved}{location}."
            )
        if self.expected is not None:
            expected = _safe_constraint_substitute(
                self.expected,
                substitutions,
            )
            if expected.is_number and sp.Ne(resolved, expected) is sp.true:
                raise ValueError(
                    f"{self.label} must equal {expected}; got {resolved}{location}."
                )
        if self.minimum is not None and self._minimum_violation(resolved) is sp.true:
            unit = ""
            if self.unit:
                plural = "" if self.minimum == 1 else "s"
                unit = f" {self.unit}{plural}"
            comparison = (
                f"greater than {self.minimum}"
                if not self.minimum_inclusive
                else f"at least {self.minimum}"
            )
            raise ValueError(
                f"{self.label} must be {comparison}{unit}; got {resolved}{location}."
            )

    def _minimum_relation(self, value: sp.Expr) -> Boolean:
        """Return the predicate that proves ``value`` satisfies the bound.

        Args:
            value (sp.Expr): Value or analytic lower bound to compare.

        Returns:
            Boolean: Inclusive or exclusive lower-bound predicate.
        """
        assert self.minimum is not None
        relation = sp.Ge if self.minimum_inclusive else sp.Gt
        return cast(Boolean, relation(value, self.minimum))

    def _minimum_violation(self, value: sp.Expr) -> Boolean:
        """Return the predicate that proves ``value`` violates the bound.

        Args:
            value (sp.Expr): Concrete constrained value.

        Returns:
            Boolean: Inclusive or exclusive lower-bound violation predicate.
        """
        assert self.minimum is not None
        relation = sp.Lt if self.minimum_inclusive else sp.Le
        return cast(Boolean, relation(value, self.minimum))

    def bound_over(
        self,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
    ) -> _ResourceConstraint:
        """Quantify a requirement over a repeated loop body.

        Affine constrained expressions attain their minimum at one endpoint,
        so they can be reduced to an external expression and remain
        checkable after later input substitution. Non-affine expressions keep
        the loop symbol as an internal bound variable instead of exposing it
        as a user parameter.

        Args:
            loop_symbol (sp.Symbol): Internal loop variable to bind.
            start (ResourceExpr): First loop value.
            step (ResourceExpr): Loop step.
            iterations (ResourceExpr): Number of executed iterations.

        Returns:
            _ResourceConstraint: Constraint with the loop variable reduced or
            marked as internally bound.

        Raises:
            ValueError: If a concrete nonempty range violates the constraint.
        """
        range_expressions = (
            expression
            for loop_range in self.ranges
            for expression in (
                loop_range.start,
                loop_range.step,
                loop_range.iterations,
            )
        )
        appears_in_nested_range = any(
            loop_symbol in expression.free_symbols for expression in range_expressions
        )
        appears_in_expected = (
            self.expected is not None and loop_symbol in self.expected.free_symbols
        )
        if (
            loop_symbol not in self.expression.free_symbols
            and not appears_in_expected
            and not appears_in_nested_range
        ):
            return self
        quantified_range = _ConstraintRange(
            symbol=loop_symbol,
            start=_expr(start),
            step=_expr(step),
            iterations=_expr(iterations),
        )
        if (
            self.expected is not None
            or self.ranges
            or loop_symbol not in self.expression.free_symbols
        ):
            bound = dataclasses.replace(
                self,
                ranges=(quantified_range, *self.ranges),
            )
            bound.validate()
            return bound
        try:
            polynomial = sp.Poly(self.expression, loop_symbol)
        except sp.PolynomialError:
            polynomial = None
        if polynomial is None or polynomial.degree() > 1:
            bound = dataclasses.replace(
                self,
                ranges=(quantified_range,),
            )
            bound.validate()
            return bound

        first = cast(sp.Expr, self.expression.subs(loop_symbol, start))
        last_value = start + step * (iterations - _ONE)
        last = cast(sp.Expr, self.expression.subs(loop_symbol, last_value))
        direction = _safe_simplify(
            cast(
                ResourceExpr,
                sp.diff(self.expression, loop_symbol) * step,
            )
        )
        if direction.is_nonnegative:
            range_minimum = first
        elif direction.is_nonpositive:
            range_minimum = last
        else:
            range_minimum = sp.Min(first, last)
        expression = _resource_expr(
            sp.Piecewise(
                (range_minimum, sp.Gt(iterations, _ZERO)),
                (self._valid_fallback(), True),
            )
        )
        bound = dataclasses.replace(
            self,
            expression=expression,
        )
        bound.validate()
        return bound


def _expr(value: ResourceExpr | int | float) -> ResourceExpr:
    """Convert a Python scalar to a SymPy expression.

    Args:
        value (ResourceExpr | int | float): Value to convert.

    Returns:
        ResourceExpr: SymPy expression.
    """
    if isinstance(value, sp.Basic):
        return cast(ResourceExpr, value)
    if isinstance(value, int):
        return sp.Integer(value)
    return sp.Float(value)


def _substitute_resource_expr(
    expression: sp.Expr,
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> sp.Expr:
    """Substitute a resource expression and enforce its concrete lower bound.

    Symbolic simplification and later substitution are separate phases, so a
    boundary substitution can expose a negative concrete expression even when
    the symbolic estimate was retained in an unspecialized form. Resource
    metrics cannot be negative, so only fully concrete negative results are
    clamped; partially symbolic expressions stay unchanged.

    Args:
        expression (sp.Expr): Resource expression to rewrite.
        substitutions (Mapping[sp.Symbol, sp.Expr]): Simultaneous symbol
            replacements.

    Returns:
        sp.Expr: Substituted expression, or zero for a concrete negative
            resource count.
    """
    substituted = cast(
        sp.Expr,
        _substitute_basic_lazily(_expr(expression), substitutions),
    )
    resolved = cast(sp.Expr, substituted.doit())
    if not resolved.free_symbols <= substituted.free_symbols:
        resolved = substituted
    if resolved.is_number and resolved.is_negative is True:
        return _ZERO
    return resolved


def _substitute_basic_lazily(
    expression: sp.Basic,
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> sp.Basic:
    """Apply simultaneous substitutions without visiting inactive branches.

    SymPy's ordinary substitution eagerly rebuilds every ``Piecewise`` value
    before deciding its conditions. A concrete resource input can therefore
    evaluate a large ``Sum`` in a branch that is immediately discarded. This
    walker resolves conditions first and stops at the first definitely active
    branch while preserving exact symbol-identity replacement everywhere
    else.

    Args:
        expression (sp.Basic): Symbolic expression or predicate to rewrite.
        substitutions (Mapping[sp.Symbol, sp.Expr]): Simultaneous free-symbol
            replacements.

    Returns:
        sp.Basic: Rewritten expression with unreachable branches untouched.
    """
    if not substitutions or not expression.free_symbols.intersection(substitutions):
        return expression
    replacement = substitutions.get(cast(sp.Symbol, expression))
    if replacement is not None and isinstance(expression, sp.Symbol):
        return replacement
    if isinstance(expression, sp.Piecewise):
        branches: list[tuple[sp.Expr, Boolean | bool]] = []
        for branch in expression.args:
            value, condition = branch.args
            rewritten_condition = _boolean_condition(
                _substitute_basic_lazily(
                    cast(sp.Basic, condition),
                    substitutions,
                )
            )
            if rewritten_condition is sp.false:
                continue
            rewritten_value = cast(
                sp.Expr,
                _substitute_basic_lazily(
                    cast(sp.Basic, value),
                    substitutions,
                ),
            )
            if rewritten_condition is sp.true:
                if not branches:
                    return rewritten_value
                branches.append((rewritten_value, True))
                break
            branches.append((rewritten_value, rewritten_condition))
        if not branches:
            return sp.nan
        return sp.Piecewise(*branches)
    rewritten_args = tuple(
        _substitute_basic_lazily(cast(sp.Basic, argument), substitutions)
        for argument in expression.args
    )
    if rewritten_args == expression.args:
        return expression
    return cast(sp.Basic, expression.func(*rewritten_args))


@lru_cache(maxsize=4096)
def _safe_simplify(expression: ResourceExpr) -> ResourceExpr:
    """Simplify an expression without releasing internal bound symbols.

    SymPy can incorrectly move a bound ``Sum`` index into a Piecewise
    condition while simplifying some nested loop expressions. Any newly free
    symbol would be exposed as a fake qkernel parameter, so such a rewrite is
    rejected in favor of the original equivalent expression.

    Args:
        expression (ResourceExpr): Resource expression to simplify.

    Returns:
        ResourceExpr: Simplified expression when symbol provenance is
        preserved, otherwise the original expression when SymPy cannot
        simplify it safely.
    """
    normalized = _expr(expression)
    try:
        simplified = cast(ResourceExpr, sp.simplify(normalized))
    except (
        ArithmeticError,
        AttributeError,
        NotImplementedError,
        RecursionError,
        TypeError,
    ):
        return normalized
    if simplified.free_symbols <= normalized.free_symbols:
        return simplified
    return normalized


def _safe_constraint_substitute(
    expression: ResourceExpr,
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> ResourceExpr:
    """Substitute constraint variables without clamping invalid values.

    Args:
        expression (ResourceExpr): Constraint or range expression.
        substitutions (Mapping[sp.Symbol, sp.Expr]): Bound loop values.

    Returns:
        ResourceExpr: Safely evaluated substituted expression.
    """
    substituted = cast(
        ResourceExpr,
        _substitute_basic_lazily(_expr(expression), substitutions),
    )
    evaluated = cast(ResourceExpr, substituted.doit())
    if not evaluated.free_symbols <= substituted.free_symbols:
        return substituted
    return _safe_simplify(evaluated)


def _resource_expr(value: sp.Basic) -> ResourceExpr:
    """Narrow a SymPy scalar expression to the resource expression type.

    SymPy annotates relational ``Piecewise`` results as ``Basic`` even though
    they participate in the same scalar arithmetic as ``Expr`` throughout the
    estimator.

    Args:
        value (sp.Basic): SymPy scalar expression to narrow.

    Returns:
        ResourceExpr: Expression accepted by resource result records.

    Raises:
        TypeError: If ``value`` is not a scalar SymPy expression.
    """
    if not isinstance(value, sp.Expr):
        raise TypeError(
            f"Expected a scalar SymPy expression, got {type(value).__name__}"
        )
    return value


def _add_maps(
    left: Mapping[str, ResourceExpr],
    right: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Add two expression dictionaries key-wise.

    Args:
        left (Mapping[str, ResourceExpr]): Left mapping.
        right (Mapping[str, ResourceExpr]): Right mapping.

    Returns:
        dict[str, ResourceExpr]: Merged mapping.
    """
    merged = dict(left)
    for name, value in right.items():
        merged[name] = merged.get(name, _ZERO) + value
    return merged


def _boolean_condition(condition: sp.Basic) -> Boolean:
    """Normalize a symbolic branch predicate to a SymPy Boolean.

    Args:
        condition (sp.Basic): Boolean or numeric branch expression.

    Returns:
        Boolean: Boolean predicate with numeric truth represented as nonzero.
    """
    if (
        condition in (sp.true, sp.false)
        or isinstance(condition, sp.logic.boolalg.BooleanFunction)
        or getattr(condition, "is_Relational", False)
    ):
        return cast(Boolean, condition)
    return cast(Boolean, sp.Ne(condition, 0))


def _unresolved_condition_guard(condition: sp.Basic) -> Boolean:
    """Keep metadata active only while a branch predicate remains symbolic.

    A symbolic condition indicator is intentionally compared with both of its
    concrete values. SymPy cannot resolve either comparison until parameter
    substitution chooses zero or one, at which point the conjunction becomes
    false and guarded branch-union metadata is pruned.

    Args:
        condition (sp.Basic): Branch predicate whose resolution state guards
            conservative metadata.

    Returns:
        Boolean: Potentially active guard while ``condition`` is symbolic, or
            false after it resolves to either Boolean value.
    """
    indicator = _ConditionIndicator(_boolean_condition(condition))
    return cast(
        Boolean,
        sp.And(
            sp.Ne(indicator, _ZERO),
            sp.Ne(indicator, _ONE),
        ),
    )


class _ConditionIndicator(sp.Function):
    """Encode a Boolean predicate as a binder-safe zero-or-one expression.

    This node stays opaque to SymPy's ``piecewise_fold`` while symbolic.
    A nested ``Piecewise`` under ``Sum`` can otherwise release the sum's bound
    variable into a public resource expression.
    """

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, condition: sp.Basic) -> sp.Integer | None:
        """Reduce a concrete Boolean predicate to zero or one.

        Args:
            condition (sp.Basic): Boolean predicate to encode.

        Returns:
            sp.Integer | None: One for true, zero for false, or ``None`` while
            the predicate remains symbolic.
        """
        if condition is sp.true:
            return _ONE
        if condition is sp.false:
            return _ZERO
        return None


_RANGE_ANY_REPLAY_LIMIT = 1024


class _RangeAny(sp.Function):
    """Represent whether any value in a finite integer range satisfies a guard.

    The predicate is stored in a one-argument ``Lambda`` so its induction
    variable remains bound across substitution and ``Piecewise`` rewriting.
    Small concrete ranges are replayed directly. Affine relational predicates
    over larger concrete ranges are decided from their finitely many truth
    boundaries. Nonlinear or symbolic large ranges retain this expression
    instead of asking SymPy to perform expensive general set construction.
    """

    nargs = 4
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(
        cls,
        predicate: sp.Basic,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
    ) -> sp.Integer | None:
        """Resolve a finite existential guard over a concrete range.

        Args:
            predicate (sp.Basic): One-argument Boolean Lambda.
            start (sp.Expr): First Python-range value.
            step (sp.Expr): Python-range step.
            iterations (sp.Expr): Number of executed iterations.

        Returns:
            sp.Integer | None: One when any position satisfies the predicate,
            zero when none do, or ``None`` when symbolic or unresolved.
        """
        if not isinstance(predicate, sp.Lambda) or len(predicate.variables) != 1:
            return None
        if iterations.is_zero is True:
            return _ZERO
        loop_symbol = predicate.variables[0]
        condition = _boolean_condition(cast(sp.Basic, predicate.expr))
        if loop_symbol not in condition.free_symbols:
            if condition is sp.true:
                if iterations.is_positive is True:
                    return _ONE
                return None
            if condition is sp.false:
                return _ZERO
            return None
        if not (
            start.is_integer is True
            and start.is_number
            and step.is_integer is True
            and step.is_number
            and iterations.is_integer is True
            and iterations.is_number
        ):
            return None
        count = int(iterations)
        if count <= 0:
            return _ZERO
        if condition.free_symbols - {loop_symbol}:
            return None
        if count > _RANGE_ANY_REPLAY_LIMIT:
            return _resolve_large_affine_range_any(
                condition,
                loop_symbol,
                start,
                step,
                count,
            )
        unresolved = False
        for offset in range(count):
            transformed = _boolean_condition(
                cast(
                    sp.Basic,
                    condition.subs(loop_symbol, start + step * offset),
                )
            )
            if transformed is sp.true:
                return _ONE
            if transformed is not sp.false:
                unresolved = True
        return None if unresolved else _ZERO


def _resolve_large_affine_range_any(
    condition: Boolean,
    loop_symbol: sp.Symbol,
    start: sp.Expr,
    step: sp.Expr,
    count: int,
) -> sp.Integer | None:
    """Decide an affine Boolean guard from its finite truth boundaries.

    After mapping the Python-range value to a zero-based position, a Boolean
    combination of affine relational atoms can change truth only at an atom's
    root. Testing both endpoints, every integral root, and the adjacent integer
    positions therefore decides existence independently of the range length.

    Args:
        condition (Boolean): Guard whose only free symbol is ``loop_symbol``.
        loop_symbol (sp.Symbol): Symbol representing the Python-range value.
        start (sp.Expr): Concrete first range value.
        step (sp.Expr): Concrete nonzero range step.
        count (int): Positive number of range positions.

    Returns:
        sp.Integer | None: One when a position satisfies the guard, zero when
        none do, or ``None`` when the predicate is not an affine relational
        Boolean formula or a candidate cannot be resolved exactly.
    """
    position = sp.Dummy("range_position", integer=True, nonnegative=True)
    transformed = _boolean_condition(
        cast(
            sp.Basic,
            condition.subs(loop_symbol, start + step * position),
        )
    )
    boundaries, supported = _linear_condition_boundaries(
        transformed,
        position,
    )
    if not supported:
        return None

    candidates = {0, count - 1}
    for boundary in boundaries:
        if boundary.is_number is not True or boundary.is_finite is not True:
            return None
        floor = cast(sp.Expr, sp.floor(boundary))
        ceiling = cast(sp.Expr, sp.ceiling(boundary))
        for candidate in (
            floor - _ONE,
            floor,
            ceiling,
            ceiling + _ONE,
        ):
            if candidate.is_number is not True or not _is_concrete_integer(candidate):
                return None
            candidates.add(int(candidate))

    unresolved = False
    for candidate in candidates:
        if not 0 <= candidate < count:
            continue
        resolved = _boolean_condition(
            cast(
                sp.Basic,
                transformed.subs(position, sp.Integer(candidate)),
            )
        )
        if resolved is sp.true:
            return _ONE
        if resolved is not sp.false:
            unresolved = True
    return None if unresolved else _ZERO


def _resource_activity_condition(expression: ResourceExpr) -> Boolean:
    """Return a conservative nonzero-resource guard with bound hygiene.

    SymPy may simplify ``Sum(..., (k, ...)) > 0`` into a predicate that leaks
    the bound ``k`` outside the sum. Each finite sum is replaced by a
    zero-or-one nonempty-range proxy before testing the enclosing nonnegative
    expression. This retains surrounding multiplication and branch structure
    without exposing bound symbols. A nonempty all-zero integrand can overstate
    activity, but never creates an optimistic depth.

    Args:
        expression (ResourceExpr): Nonnegative resource expression.

    Returns:
        Boolean: True, false, or a bound-safe positivity predicate.
    """
    if expression == _ZERO or expression.is_zero is True:
        return sp.false
    if expression.is_positive is True:
        return sp.true
    if isinstance(expression, sp.Piecewise):
        remaining: Boolean = sp.true
        active_branches: list[Boolean] = []
        for branch in expression.args:
            value, condition = branch.args
            branch_condition = _boolean_condition(cast(sp.Basic, condition))
            effective_condition = _and_conditions(remaining, branch_condition)
            branch_activity = _resource_activity_condition(cast(ResourceExpr, value))
            active_branches.append(
                _and_conditions(effective_condition, branch_activity)
            )
            remaining = _and_conditions(
                remaining,
                cast(Boolean, sp.Not(branch_condition)),
            )
            if remaining is sp.false:
                break
        return cast(Boolean, sp.Or(*active_branches))
    replacements: dict[sp.Sum, ResourceExpr] = {}
    for summation in expression.atoms(sp.Sum):
        range_conditions: list[Boolean] = []
        for limit in summation.limits:
            if len(limit) != 3:
                range_conditions = [sp.true]
                break
            _symbol, lower, upper = limit
            range_conditions.append(cast(Boolean, sp.Ge(upper, lower)))
        nonempty = sp.And(*range_conditions) if range_conditions else sp.true
        replacements[summation] = _piecewise(_ONE, _ZERO, nonempty)
    proxy = cast(ResourceExpr, expression.xreplace(replacements))
    return _boolean_condition(sp.Gt(proxy, _ZERO))


def _and_conditions(left: sp.Basic, right: sp.Basic) -> Boolean:
    """Conjoin two symbolic activation predicates with trivial folding.

    Args:
        left (sp.Basic): Existing activation predicate.
        right (sp.Basic): Additional activation predicate.

    Returns:
        Boolean: Simplified conjunction.
    """
    left_condition = _boolean_condition(left)
    right_condition = _boolean_condition(right)
    if left_condition is sp.false or right_condition is sp.false:
        return sp.false
    if left_condition is sp.true:
        return right_condition
    if right_condition is sp.true:
        return left_condition
    return sp.And(left_condition, right_condition)


def _rewrite_condition(condition: sp.Basic, fn: Any) -> Boolean:
    """Rewrite and normalize one symbolic activation predicate.

    Args:
        condition (sp.Basic): Predicate to rewrite.
        fn (Any): Symbolic-expression rewrite function.

    Returns:
        Boolean: Rewritten Boolean predicate.
    """
    rewritten = cast(sp.Basic, fn(condition))
    return _boolean_condition(rewritten)


def _piecewise(
    true_value: ResourceExpr,
    false_value: ResourceExpr,
    condition: sp.Basic,
) -> ResourceExpr:
    """Select one resource expression with a symbolic Boolean.

    Args:
        true_value (ResourceExpr): Value when ``condition`` is true.
        false_value (ResourceExpr): Value when ``condition`` is false.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        ResourceExpr: Simplified piecewise expression.
    """
    return cast(
        ResourceExpr,
        sp.Piecewise((true_value, cast(Any, condition)), (false_value, True)),
    )


def _conditional_width(
    true_value: WidthResources,
    false_value: WidthResources,
    condition: sp.Basic,
) -> WidthResources:
    """Select width resources with a symbolic condition.

    Args:
        true_value (WidthResources): True-branch width.
        false_value (WidthResources): False-branch width.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        WidthResources: Field-wise piecewise width.
    """
    return WidthResources(
        **{
            field.name: _piecewise(
                getattr(true_value, field.name),
                getattr(false_value, field.name),
                condition,
            )
            for field in dataclasses.fields(WidthResources)
        }
    )


def _conditional_gates(
    true_value: GateResources,
    false_value: GateResources,
    condition: sp.Basic,
) -> GateResources:
    """Select gate resources with a symbolic condition.

    Args:
        true_value (GateResources): True-branch gates.
        false_value (GateResources): False-branch gates.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        GateResources: Field-wise piecewise gate resources.
    """
    return GateResources(
        **{
            field.name: _piecewise(
                getattr(true_value, field.name),
                getattr(false_value, field.name),
                condition,
            )
            for field in dataclasses.fields(GateResources)
        }
    )


def _conditional_measurements(
    true_value: MeasurementResources,
    false_value: MeasurementResources,
    condition: sp.Basic,
) -> MeasurementResources:
    """Select measurement resources with a symbolic condition.

    Args:
        true_value (MeasurementResources): True-branch measurements.
        false_value (MeasurementResources): False-branch measurements.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        MeasurementResources: Field-wise piecewise measurement resources.
    """
    return MeasurementResources(
        total=_piecewise(
            true_value.total,
            false_value.total,
            condition,
        )
    )


def _conditional_resets(
    true_value: ResetResources,
    false_value: ResetResources,
    condition: sp.Basic,
) -> ResetResources:
    """Select reset resources with a symbolic condition.

    Args:
        true_value (ResetResources): True-branch resets.
        false_value (ResetResources): False-branch resets.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        ResetResources: Field-wise piecewise reset resources.
    """
    return ResetResources(
        total=_piecewise(
            true_value.total,
            false_value.total,
            condition,
        )
    )


def _conditional_depth(
    true_value: DepthResources,
    false_value: DepthResources,
    condition: sp.Basic,
) -> DepthResources:
    """Select depth resources with a symbolic condition.

    Args:
        true_value (DepthResources): True-branch depth.
        false_value (DepthResources): False-branch depth.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        DepthResources: Field-wise piecewise depth resources.
    """
    return DepthResources(
        **{
            field.name: _piecewise(
                getattr(true_value, field.name),
                getattr(false_value, field.name),
                condition,
            )
            for field in dataclasses.fields(DepthResources)
        }
    )


def _conditional_calls(
    true_value: CallResources,
    false_value: CallResources,
    condition: sp.Basic,
) -> CallResources:
    """Select callable counts with a symbolic condition.

    Args:
        true_value (CallResources): True-branch calls.
        false_value (CallResources): False-branch calls.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        CallResources: Key-wise piecewise callable counts.
    """

    def select_maps(
        true_map: Mapping[str, ResourceExpr],
        false_map: Mapping[str, ResourceExpr],
    ) -> dict[str, ResourceExpr]:
        """Select two call-count mappings key by key.

        Args:
            true_map (Mapping[str, ResourceExpr]): True-branch mapping.
            false_map (Mapping[str, ResourceExpr]): False-branch mapping.

        Returns:
            dict[str, ResourceExpr]: Piecewise mapping over the union of keys.
        """
        return {
            name: _piecewise(
                true_map.get(name, _ZERO),
                false_map.get(name, _ZERO),
                condition,
            )
            for name in sorted(set(true_map) | set(false_map))
        }

    return CallResources(
        calls_by_name=select_maps(
            true_value.calls_by_name,
            false_value.calls_by_name,
        ),
        queries_by_name=select_maps(
            true_value.queries_by_name,
            false_value.queries_by_name,
        ),
    )


def _max_maps(
    left: Mapping[str, ResourceExpr],
    right: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Take key-wise maxima of two expression dictionaries.

    Args:
        left (Mapping[str, ResourceExpr]): Left mapping.
        right (Mapping[str, ResourceExpr]): Right mapping.

    Returns:
        dict[str, ResourceExpr]: Merged mapping.
    """
    merged: dict[str, ResourceExpr] = {}
    for key in sorted(set(left) | set(right)):
        merged[key] = sp.Max(left.get(key, _ZERO), right.get(key, _ZERO))
    return merged


def _add_gates(left: GateResources, right: GateResources) -> GateResources:
    """Add gate resources.

    Args:
        left (GateResources): Left resources.
        right (GateResources): Right resources.

    Returns:
        GateResources: Sum.
    """
    return GateResources(
        total=left.total + right.total,
        single_qubit=left.single_qubit + right.single_qubit,
        two_qubit=left.two_qubit + right.two_qubit,
        multi_qubit=left.multi_qubit + right.multi_qubit,
        clifford=left.clifford + right.clifford,
        rotation=left.rotation + right.rotation,
        t=left.t + right.t,
        toffoli=left.toffoli + right.toffoli,
        non_clifford=left.non_clifford + right.non_clifford,
    )


def _max_gates(left: GateResources, right: GateResources) -> GateResources:
    """Take element-wise maxima of gate resources.

    Args:
        left (GateResources): Left resources.
        right (GateResources): Right resources.

    Returns:
        GateResources: Element-wise maximum.
    """
    return GateResources(
        total=sp.Max(left.total, right.total),
        single_qubit=sp.Max(left.single_qubit, right.single_qubit),
        two_qubit=sp.Max(left.two_qubit, right.two_qubit),
        multi_qubit=sp.Max(left.multi_qubit, right.multi_qubit),
        clifford=sp.Max(left.clifford, right.clifford),
        rotation=sp.Max(left.rotation, right.rotation),
        t=sp.Max(left.t, right.t),
        toffoli=sp.Max(left.toffoli, right.toffoli),
        non_clifford=sp.Max(left.non_clifford, right.non_clifford),
    )


def _scale_gates(gates: GateResources, factor: ResourceExpr) -> GateResources:
    """Scale gate resources.

    Args:
        gates (GateResources): Gate resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        GateResources: Scaled resources.
    """
    return GateResources(
        total=gates.total * factor,
        single_qubit=gates.single_qubit * factor,
        two_qubit=gates.two_qubit * factor,
        multi_qubit=gates.multi_qubit * factor,
        clifford=gates.clifford * factor,
        rotation=gates.rotation * factor,
        t=gates.t * factor,
        toffoli=gates.toffoli * factor,
        non_clifford=gates.non_clifford * factor,
    )


def _add_measurements(
    left: MeasurementResources,
    right: MeasurementResources,
) -> MeasurementResources:
    """Add measurement resources.

    Args:
        left (MeasurementResources): Left resources.
        right (MeasurementResources): Right resources.

    Returns:
        MeasurementResources: Sum.
    """
    return MeasurementResources(total=left.total + right.total)


def _max_measurements(
    left: MeasurementResources,
    right: MeasurementResources,
) -> MeasurementResources:
    """Take element-wise maxima of measurement resources.

    Args:
        left (MeasurementResources): Left resources.
        right (MeasurementResources): Right resources.

    Returns:
        MeasurementResources: Element-wise maximum.
    """
    return MeasurementResources(total=sp.Max(left.total, right.total))


def _scale_measurements(
    measurements: MeasurementResources,
    factor: ResourceExpr,
) -> MeasurementResources:
    """Scale measurement resources.

    Args:
        measurements (MeasurementResources): Measurement resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        MeasurementResources: Scaled resources.
    """
    return MeasurementResources(total=measurements.total * factor)


def _add_resets(
    left: ResetResources,
    right: ResetResources,
) -> ResetResources:
    """Add reset resources.

    Args:
        left (ResetResources): Left resources.
        right (ResetResources): Right resources.

    Returns:
        ResetResources: Sum.
    """
    return ResetResources(total=left.total + right.total)


def _max_resets(
    left: ResetResources,
    right: ResetResources,
) -> ResetResources:
    """Take element-wise maxima of reset resources.

    Args:
        left (ResetResources): Left resources.
        right (ResetResources): Right resources.

    Returns:
        ResetResources: Element-wise maximum.
    """
    return ResetResources(total=sp.Max(left.total, right.total))


def _scale_resets(
    resets: ResetResources,
    factor: ResourceExpr,
) -> ResetResources:
    """Scale reset resources.

    Args:
        resets (ResetResources): Reset resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        ResetResources: Scaled resources.
    """
    return ResetResources(total=resets.total * factor)


@lru_cache(maxsize=4096)
def _is_structurally_nonnegative(expression: ResourceExpr) -> bool:
    """Prove nonnegativity from resource-expression constructors alone.

    This deliberately avoids general symbolic simplification. Resource width
    expressions frequently contain nested ``Min``, ``Max``, and ``Piecewise``
    nodes; recognizing their explicit nonnegative branches is both cheaper and
    more reliable than asking SymPy to derive the same invariant globally. An
    additive expression with exactly one such node also distributes its affine
    remainder into that node's values, which proves common-offset bounds
    without expanding combinations of independent extrema.

    Args:
        expression (ResourceExpr): Expression whose sign should be inspected.

    Returns:
        bool: Whether the expression structure proves a nonnegative value.
    """
    if expression == _ZERO:
        return True
    if isinstance(expression, sp.Number):
        return expression.is_nonnegative is True
    if isinstance(expression, sp.Symbol):
        return expression.is_nonnegative is True
    if getattr(type(expression), "is_nonnegative", None) is True:
        return True
    if isinstance(expression, sp.Max):
        return any(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        )
    if isinstance(expression, sp.Min):
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        )
    if isinstance(expression, sp.Piecewise):
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, pair.args[0]))
            for pair in expression.args
        )
    if isinstance(expression, sp.Add):
        if all(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        ):
            return True
        extrema = tuple(
            argument
            for argument in expression.args
            if isinstance(argument, (sp.Max, sp.Min, sp.Piecewise))
        )
        if len(extrema) != 1:
            return False
        extremum = extrema[0]
        remainder = cast(
            ResourceExpr,
            sp.Add(
                *(argument for argument in expression.args if argument is not extremum)
            ),
        )
        if isinstance(extremum, sp.Max):
            return any(
                _is_structurally_nonnegative(cast(ResourceExpr, argument + remainder))
                for argument in extremum.args
            )
        values = (
            tuple(pair.args[0] for pair in extremum.args)
            if isinstance(extremum, sp.Piecewise)
            else extremum.args
        )
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, value + remainder))
            for value in values
        )
    if isinstance(expression, sp.Mul):
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        )
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        return _is_structurally_nonnegative(cast(ResourceExpr, base)) or (
            exponent.is_integer is True and exponent.is_even is True
        )
    return False


@lru_cache(maxsize=4096)
def _is_structurally_less_equal(
    left: ResourceExpr,
    right: ResourceExpr,
) -> bool:
    """Prove one resource expression does not exceed another structurally.

    Args:
        left (ResourceExpr): Candidate lower expression.
        right (ResourceExpr): Candidate upper expression.

    Returns:
        bool: Whether every structural branch proves ``left <= right``.
    """
    if left == right:
        return True
    if isinstance(left, _ConditionIndicator):
        return _is_structurally_less_equal(_ONE, right)
    if isinstance(left, sp.Mul) and any(
        isinstance(argument, _ConditionIndicator) for argument in left.args
    ):
        unguarded = cast(
            ResourceExpr,
            sp.Mul(
                *(
                    argument
                    for argument in left.args
                    if not isinstance(argument, _ConditionIndicator)
                )
            ),
        )
        if _is_structurally_nonnegative(unguarded) and _is_structurally_less_equal(
            unguarded, right
        ):
            return True
    if isinstance(left, sp.Piecewise):
        return all(
            _is_structurally_less_equal(
                cast(ResourceExpr, pair.args[0]),
                right,
            )
            for pair in left.args
        )
    if isinstance(right, sp.Piecewise):
        return all(
            _is_structurally_less_equal(
                left,
                cast(ResourceExpr, pair.args[0]),
            )
            for pair in right.args
        )
    if isinstance(left, sp.Max):
        return all(
            _is_structurally_less_equal(cast(ResourceExpr, argument), right)
            for argument in left.args
        )
    if isinstance(left, sp.Min):
        return any(
            _is_structurally_less_equal(cast(ResourceExpr, argument), right)
            for argument in left.args
        )
    if isinstance(right, sp.Max):
        return any(
            _is_structurally_less_equal(left, cast(ResourceExpr, argument))
            for argument in right.args
        )
    if isinstance(right, sp.Min):
        return all(
            _is_structurally_less_equal(left, cast(ResourceExpr, argument))
            for argument in right.args
        )
    return _is_structurally_nonnegative(right - left)


def _guarded_resource_extension(
    base: ResourceExpr,
    candidate: ResourceExpr,
) -> ResourceExpr | None:
    """Recover an exact maximum from one activity-gated extension.

    Scheduling can represent an optional completion as
    ``(base + extension) * indicator(extension > 0)``. Its maximum with the
    nonnegative ``base`` is exactly ``base + extension``: while inactive the
    extension is zero, and while active the guarded candidate dominates.

    Args:
        base (ResourceExpr): Completion retained when the extension is absent.
        candidate (ResourceExpr): Potential activity-gated completion.

    Returns:
        ResourceExpr | None: Unguarded dominating completion when the pattern
            is proven structurally, otherwise ``None``.
    """
    if not isinstance(candidate, sp.Mul) or not _is_structurally_nonnegative(base):
        return None
    indicators = tuple(
        argument
        for argument in candidate.args
        if isinstance(argument, _ConditionIndicator)
    )
    if len(indicators) != 1:
        return None
    unguarded = cast(
        ResourceExpr,
        sp.Mul(
            *(
                argument
                for argument in candidate.args
                if not isinstance(argument, _ConditionIndicator)
            )
        ),
    )
    if not _is_structurally_less_equal(base, unguarded):
        return None
    extension = cast(ResourceExpr, unguarded - base)
    if not _is_structurally_nonnegative(extension):
        return None
    if indicators[0].args[0] != _resource_activity_condition(extension):
        return None
    return unguarded


def _resource_max(left: ResourceExpr, right: ResourceExpr) -> ResourceExpr:
    """Return a compact maximum using structural resource bounds.

    Args:
        left (ResourceExpr): First candidate.
        right (ResourceExpr): Second candidate.

    Returns:
        ResourceExpr: Dominating expression when proven, otherwise ``Max``.
    """
    if (extension := _guarded_resource_extension(left, right)) is not None:
        return extension
    if (extension := _guarded_resource_extension(right, left)) is not None:
        return extension
    if _is_structurally_less_equal(left, right):
        return right
    if _is_structurally_less_equal(right, left):
        return left
    candidates = (
        *(left.args if isinstance(left, sp.Max) else (left,)),
        *(right.args if isinstance(right, sp.Max) else (right,)),
    )
    unique_candidates = tuple(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return cast(ResourceExpr, unique_candidates[0])
    return cast(ResourceExpr, sp.Max(*unique_candidates, evaluate=False))


def _resource_max_many(expressions: Sequence[ResourceExpr]) -> ResourceExpr:
    """Fold resource maxima without triggering eager general simplification.

    The dependency scheduler repeatedly compares expressions already known to
    be nonnegative and monotone. Folding through :func:`_resource_max` proves
    those common dominance relations structurally before building an
    unevaluated ``Max``. This avoids SymPy's general pairwise relation proofs,
    whose cost is quadratic in the number and size of candidates.

    Args:
        expressions (Sequence[ResourceExpr]): Candidate resource expressions.

    Returns:
        ResourceExpr: Structural maximum, or zero for an empty sequence.
    """
    maximum = _ZERO
    for expression in expressions:
        maximum = _resource_max(maximum, expression)
    return maximum


def _add_depth(left: DepthResources, right: DepthResources) -> DepthResources:
    """Add depth resources.

    Args:
        left (DepthResources): Left resources.
        right (DepthResources): Right resources.

    Returns:
        DepthResources: Sum.
    """
    return DepthResources(
        depth=left.depth + right.depth,
        clifford_depth=left.clifford_depth + right.clifford_depth,
        rotation_depth=left.rotation_depth + right.rotation_depth,
        t_depth=left.t_depth + right.t_depth,
        toffoli_depth=left.toffoli_depth + right.toffoli_depth,
        non_clifford_depth=left.non_clifford_depth + right.non_clifford_depth,
        measurement_depth=left.measurement_depth + right.measurement_depth,
        gate_depth=left.gate_depth + right.gate_depth,
        reset_depth=left.reset_depth + right.reset_depth,
    )


def _max_depth(left: DepthResources, right: DepthResources) -> DepthResources:
    """Take element-wise maxima of depth resources.

    Args:
        left (DepthResources): Left resources.
        right (DepthResources): Right resources.

    Returns:
        DepthResources: Element-wise maximum.
    """
    return DepthResources(
        depth=sp.Max(left.depth, right.depth),
        clifford_depth=sp.Max(left.clifford_depth, right.clifford_depth),
        rotation_depth=sp.Max(left.rotation_depth, right.rotation_depth),
        t_depth=sp.Max(left.t_depth, right.t_depth),
        toffoli_depth=sp.Max(left.toffoli_depth, right.toffoli_depth),
        non_clifford_depth=sp.Max(left.non_clifford_depth, right.non_clifford_depth),
        measurement_depth=sp.Max(left.measurement_depth, right.measurement_depth),
        gate_depth=sp.Max(left.gate_depth, right.gate_depth),
        reset_depth=sp.Max(left.reset_depth, right.reset_depth),
    )


def _scale_depth(depth: DepthResources, factor: ResourceExpr) -> DepthResources:
    """Scale depth resources.

    Args:
        depth (DepthResources): Depth resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        DepthResources: Scaled resources.
    """
    return DepthResources(
        depth=depth.depth * factor,
        clifford_depth=depth.clifford_depth * factor,
        rotation_depth=depth.rotation_depth * factor,
        t_depth=depth.t_depth * factor,
        toffoli_depth=depth.toffoli_depth * factor,
        non_clifford_depth=depth.non_clifford_depth * factor,
        measurement_depth=depth.measurement_depth * factor,
        gate_depth=depth.gate_depth * factor,
        reset_depth=depth.reset_depth * factor,
    )


def _add_calls(left: CallResources, right: CallResources) -> CallResources:
    """Add call resources.

    Args:
        left (CallResources): Left resources.
        right (CallResources): Right resources.

    Returns:
        CallResources: Sum.
    """
    return CallResources(
        calls_by_name=_add_maps(left.calls_by_name, right.calls_by_name),
        queries_by_name=_add_maps(left.queries_by_name, right.queries_by_name),
    )


def _max_calls(left: CallResources, right: CallResources) -> CallResources:
    """Take key-wise maxima of call resources.

    Args:
        left (CallResources): Left resources.
        right (CallResources): Right resources.

    Returns:
        CallResources: Element-wise maximum.
    """
    return CallResources(
        calls_by_name=_max_maps(left.calls_by_name, right.calls_by_name),
        queries_by_name=_max_maps(left.queries_by_name, right.queries_by_name),
    )


def _scale_calls(calls: CallResources, factor: ResourceExpr) -> CallResources:
    """Scale call resources.

    Args:
        calls (CallResources): Call resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        CallResources: Scaled resources.
    """
    return CallResources(
        calls_by_name={
            name: value * factor for name, value in calls.calls_by_name.items()
        },
        queries_by_name={
            name: value * factor for name, value in calls.queries_by_name.items()
        },
    )


def _seq_width(left: WidthResources, right: WidthResources) -> WidthResources:
    """Compose width resources sequentially.

    Args:
        left (WidthResources): Left resources.
        right (WidthResources): Right resources.

    Returns:
        WidthResources: Sequential width estimate.
    """
    allocated = left.allocated_qubits + right.allocated_qubits
    return WidthResources(
        input_qubits=sp.Max(left.input_qubits, right.input_qubits),
        allocated_qubits=allocated,
        clean_ancilla_qubits=_resource_max(
            left.clean_ancilla_qubits,
            right.clean_ancilla_qubits,
        ),
        dirty_ancilla_qubits=_resource_max(
            left.dirty_ancilla_qubits,
            right.dirty_ancilla_qubits,
        ),
        peak_qubits=sp.Max(left.peak_qubits, left.allocated_qubits + right.peak_qubits),
    )


def _parallel_width(left: WidthResources, right: WidthResources) -> WidthResources:
    """Compose width resources in parallel.

    Args:
        left (WidthResources): Left resources.
        right (WidthResources): Right resources.

    Returns:
        WidthResources: Parallel width estimate.
    """
    return WidthResources(
        input_qubits=left.input_qubits + right.input_qubits,
        allocated_qubits=left.allocated_qubits + right.allocated_qubits,
        clean_ancilla_qubits=left.clean_ancilla_qubits + right.clean_ancilla_qubits,
        dirty_ancilla_qubits=left.dirty_ancilla_qubits + right.dirty_ancilla_qubits,
        peak_qubits=left.peak_qubits + right.peak_qubits,
    )


def _max_width(left: WidthResources, right: WidthResources) -> WidthResources:
    """Take element-wise maxima of width resources.

    Args:
        left (WidthResources): Left resources.
        right (WidthResources): Right resources.

    Returns:
        WidthResources: Element-wise maximum.
    """
    return WidthResources(
        input_qubits=sp.Max(left.input_qubits, right.input_qubits),
        allocated_qubits=sp.Max(left.allocated_qubits, right.allocated_qubits),
        clean_ancilla_qubits=_resource_max(
            left.clean_ancilla_qubits,
            right.clean_ancilla_qubits,
        ),
        dirty_ancilla_qubits=_resource_max(
            left.dirty_ancilla_qubits,
            right.dirty_ancilla_qubits,
        ),
        peak_qubits=sp.Max(left.peak_qubits, right.peak_qubits),
    )


def _maximum_expr_over_range(
    expr: ResourceExpr,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[ResourceExpr, bool]:
    """Maximize one nonnegative resource expression over a loop range.

    Constant, affine, finite, and provably monotone expressions are reduced
    exactly. When SymPy cannot prove a maximum, summing the nonnegative body
    expression gives a conservative loop-symbol-free upper bound.

    Args:
        expr (ResourceExpr): Per-iteration resource expression.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[ResourceExpr, bool]: Maximum expression and whether it is exact.
    """
    condition = sp.Gt(iterations, _ZERO)
    if loop_symbol not in expr.free_symbols:
        return _piecewise(expr, _ZERO, condition), True

    if not isinstance(expr, sp.Piecewise) and expr.has(sp.Piecewise):
        folded = cast(ResourceExpr, sp.piecewise_fold(expr))
        if folded != expr:
            return _maximum_expr_over_range(
                folded,
                loop_symbol,
                start,
                step,
                iterations,
            )

    if isinstance(iterations, sp.Integer) and 0 <= int(iterations) <= 64:
        count = int(iterations)
        if count == 0:
            return _ZERO, True
        values = [
            cast(ResourceExpr, expr.subs(loop_symbol, start + step * index))
            for index in range(count)
        ]
        return cast(ResourceExpr, sp.Max(*values)), True

    if isinstance(expr, sp.Max):
        argument_maxima = [
            _maximum_expr_over_range(
                cast(ResourceExpr, argument),
                loop_symbol,
                start,
                step,
                iterations,
            )
            for argument in expr.args
        ]
        return (
            cast(ResourceExpr, sp.Max(*(maximum for maximum, _ in argument_maxima))),
            all(exact for _, exact in argument_maxima),
        )

    if isinstance(expr, sp.Piecewise):
        return _maximum_piecewise_over_range(
            expr,
            loop_symbol,
            start,
            step,
            iterations,
        )

    index = sp.Dummy("k", integer=True, nonnegative=True)
    transformed = cast(
        sp.Expr,
        expr.subs(loop_symbol, start + step * index),
    )
    first = cast(ResourceExpr, transformed.subs(index, _ZERO))
    last = cast(ResourceExpr, transformed.subs(index, iterations - _ONE))
    try:
        polynomial = sp.Poly(transformed, index)
    except sp.PolynomialError:
        polynomial = None
    if polynomial is not None and polynomial.degree() <= 1:
        return _piecewise(sp.Max(first, last), _ZERO, condition), True

    upper_bound = _sum_expr(
        cast(ResourceExpr, sp.Max(_ZERO, expr)),
        loop_symbol,
        start,
        step,
        iterations,
    )
    return _piecewise(upper_bound, _ZERO, condition), False


def _maximum_piecewise_over_range(
    expression: sp.Piecewise,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[ResourceExpr, bool]:
    """Maximize an endpoint-bounded Piecewise expression over a loop.

    Piecewise resource formulas commonly encode special control arities, such
    as zero ancillas for one or two controls and an affine fallback above that.
    Maximizing every branch over the full loop ignores those predicates and can
    select an unreachable value. This routine instead evaluates the complete
    Piecewise expression at loop endpoints and every linear predicate boundary.

    Args:
        expression (sp.Piecewise): Per-iteration resource expression.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[ResourceExpr, bool]: Predicate-aware maximum and whether the
        endpoint proof was exact.
    """
    index = sp.Dummy("piecewise_index", integer=True, nonnegative=True)
    folded = cast(sp.Piecewise, sp.piecewise_fold(expression))
    transformed = cast(
        sp.Piecewise,
        folded.subs(loop_symbol, start + step * index),
    )
    boundaries: set[ResourceExpr] = set()
    conditions_supported = True
    branches_endpoint_bounded = True
    for branch_expression, condition in cast(
        tuple[tuple[sp.Expr, sp.Basic], ...],
        transformed.args,
    ):
        branch_boundaries, supported = _linear_condition_boundaries(
            condition,
            index,
        )
        boundaries.update(branch_boundaries)
        conditions_supported = conditions_supported and supported
        branches_endpoint_bounded = branches_endpoint_bounded and (
            _endpoint_bounded_expression(branch_expression, index)
            or _condition_restricts_to_finite_points(condition, index)
        )

    if conditions_supported and branches_endpoint_bounded:
        candidates: set[ResourceExpr] = {_ZERO, iterations - _ONE}
        for boundary in boundaries:
            floor = cast(ResourceExpr, sp.floor(boundary))
            ceiling = cast(ResourceExpr, sp.ceiling(boundary))
            candidates.update(
                {
                    boundary,
                    floor - _ONE,
                    floor,
                    ceiling,
                    ceiling + _ONE,
                }
            )
        values = [
            _guarded_range_candidate(transformed, index, candidate, iterations)
            for candidate in candidates
        ]
        return cast(ResourceExpr, sp.Max(*values)), True

    upper_bound = _sum_expr(
        cast(ResourceExpr, sp.Max(_ZERO, folded)),
        loop_symbol,
        start,
        step,
        iterations,
    )
    return (
        _piecewise(upper_bound, _ZERO, sp.Gt(iterations, _ZERO)),
        False,
    )


def _linear_condition_boundaries(
    condition: sp.Basic,
    index: sp.Symbol,
) -> tuple[set[ResourceExpr], bool]:
    """Extract roots of linear relational atoms in a branch predicate.

    Args:
        condition (sp.Basic): Piecewise branch condition.
        index (sp.Symbol): Integer loop-position symbol.

    Returns:
        tuple[set[ResourceExpr], bool]: Predicate boundaries and whether every
        index-dependent atom was supported.
    """
    if condition in (sp.true, sp.false):
        return set(), True
    if isinstance(
        condition,
        (sp.logic.boolalg.And, sp.logic.boolalg.Or, sp.logic.boolalg.Not),
    ):
        boundaries: set[ResourceExpr] = set()
        supported = True
        for argument in condition.args:
            nested, nested_supported = _linear_condition_boundaries(
                cast(sp.Basic, argument),
                index,
            )
            boundaries.update(nested)
            supported = supported and nested_supported
        return boundaries, supported
    if not isinstance(condition, Relational):
        return set(), index not in condition.free_symbols
    difference = cast(sp.Expr, condition.lhs) - cast(sp.Expr, condition.rhs)
    if index not in difference.free_symbols:
        return set(), True
    try:
        polynomial = sp.Poly(difference, index)
    except sp.PolynomialError:
        return set(), False
    if polynomial.degree() != 1:
        return set(), False
    slope, intercept = polynomial.all_coeffs()
    return {cast(ResourceExpr, -intercept / slope)}, True


def _endpoint_bounded_expression(expression: sp.Expr, index: sp.Symbol) -> bool:
    """Return whether interval endpoints determine an expression's maximum.

    Args:
        expression (sp.Expr): One Piecewise branch expression.
        index (sp.Symbol): Integer loop-position symbol.

    Returns:
        bool: Whether the expression is affine, constant, or a maximum of such
        expressions.
    """
    if index not in expression.free_symbols:
        return True
    if isinstance(expression, sp.Max):
        return all(
            _endpoint_bounded_expression(cast(sp.Expr, argument), index)
            for argument in expression.args
        )
    try:
        return sp.Poly(expression, index).degree() <= 1
    except sp.PolynomialError:
        return False


def _condition_restricts_to_finite_points(
    condition: sp.Basic,
    index: sp.Symbol,
) -> bool:
    """Return whether a predicate admits only finitely many loop positions.

    Args:
        condition (sp.Basic): Piecewise branch predicate.
        index (sp.Symbol): Integer loop-position symbol.

    Returns:
        bool: Whether linear equalities restrict the branch to finite points.
    """
    if isinstance(condition, sp.Equality):
        boundaries, supported = _linear_condition_boundaries(condition, index)
        return supported and bool(boundaries)
    if isinstance(condition, sp.logic.boolalg.Or):
        return all(
            _condition_restricts_to_finite_points(
                cast(sp.Basic, argument),
                index,
            )
            for argument in condition.args
        )
    if isinstance(condition, sp.logic.boolalg.And):
        return any(
            _condition_restricts_to_finite_points(
                cast(sp.Basic, argument),
                index,
            )
            for argument in condition.args
        )
    return False


def _guarded_range_candidate(
    expression: sp.Expr,
    index: sp.Symbol,
    candidate: ResourceExpr,
    iterations: ResourceExpr,
) -> ResourceExpr:
    """Evaluate one candidate only when it is a valid loop position.

    Args:
        expression (sp.Expr): Loop-indexed Piecewise expression.
        index (sp.Symbol): Integer loop-position symbol.
        candidate (ResourceExpr): Candidate index to evaluate.
        iterations (ResourceExpr): Number of loop iterations.

    Returns:
        ResourceExpr: Candidate resource value, or zero when out of range.
    """
    condition: Boolean = sp.And(
        sp.Gt(iterations, _ZERO),
        sp.Ge(candidate, _ZERO),
        sp.Lt(candidate, iterations),
    )
    if candidate.is_integer is not True:
        condition = sp.And(
            condition,
            cast(Boolean, sp.Eq(candidate, sp.floor(candidate))),
        )
    value = cast(ResourceExpr, expression.subs(index, candidate))
    return _piecewise(value, _ZERO, condition)


def _sum_expr(
    expr: ResourceExpr,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> ResourceExpr:
    """Sum an expression over Python ``range`` semantics.

    Args:
        expr (ResourceExpr): Expression to sum.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        ResourceExpr: Summed expression.
    """
    if expr == _ZERO or iterations == _ZERO:
        return _ZERO
    if loop_symbol not in expr.free_symbols:
        return cast(ResourceExpr, expr * iterations)
    k = sp.Dummy("k", integer=True, nonnegative=True)
    transformed = expr.subs(loop_symbol, start + step * k)
    transformed = _simplify_sum_range_guards(
        transformed,
        symbol=k,
        lower=_ZERO,
        upper=iterations - 1,
    )
    summation = sp.Sum(transformed, (k, 0, iterations - 1))
    evaluated = cast(ResourceExpr, summation.doit())
    if not evaluated.free_symbols <= summation.free_symbols:
        return cast(ResourceExpr, summation)
    return cast(ResourceExpr, evaluated)


def _activation_over_range(
    condition: sp.Basic,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> sp.Basic:
    """Return whether a guarded fact is active in any loop iteration.

    Args:
        condition (sp.Basic): Per-iteration activation condition.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        sp.Basic: Condition that is true exactly when at least one reachable
            iteration activates the fact.
    """
    active = _boolean_condition(condition)
    nonempty = sp.Gt(iterations, _ZERO)
    if loop_symbol not in active.free_symbols:
        return _boolean_condition(sp.And(nonempty, active))
    active_iterations = _RangeAny(
        sp.Lambda(loop_symbol, active),
        start,
        step,
        iterations,
    )
    return _boolean_condition(sp.Ne(active_iterations, _ZERO))


def _simplify_sum_range_guards(
    expression: sp.Expr,
    *,
    symbol: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
) -> sp.Expr:
    """Remove nonnegative affine guards proven by a finite sum range.

    Nested ``qmc.range`` loops produce exact trip counts such as
    ``Max(0, n - 1 - k)``. SymPy does not use the enclosing summation bound
    ``0 <= k <= n - 1`` when simplifying that guard, so triangular loops stay
    as unevaluated sums. This helper removes only guards whose affine argument
    is provably nonnegative at the endpoint where it reaches its minimum.

    Args:
        expression (sp.Expr): Summand to simplify.
        symbol (sp.Symbol): Summation index.
        lower (sp.Expr): Inclusive lower index bound.
        upper (sp.Expr): Inclusive upper index bound.

    Returns:
        sp.Expr: Expression with range-proven ``Max(0, affine)`` guards removed.
    """
    replacements: dict[sp.Basic, sp.Basic] = {}
    for node in sp.preorder_traversal(expression):
        if (
            not isinstance(node, sp.Max)
            or len(node.args) != 2
            or _ZERO not in node.args
        ):
            continue
        guarded = node.args[0] if node.args[1] == _ZERO else node.args[1]
        try:
            polynomial = sp.Poly(guarded, symbol)
        except sp.PolynomialError:
            continue
        if polynomial.degree() > 1:
            continue
        slope = sp.diff(guarded, symbol)
        if slope.is_nonnegative:
            minimum = guarded.subs(symbol, lower)
        elif slope.is_nonpositive:
            minimum = guarded.subs(symbol, upper)
        else:
            continue
        if sp.simplify(minimum).is_nonnegative:
            replacements[node] = guarded
    return cast(sp.Expr, expression.xreplace(replacements))


def _sum_gates(
    gates: GateResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> GateResources:
    """Sum gate resources over a loop.

    Args:
        gates (GateResources): Gate resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        GateResources: Summed gate resources.
    """
    return GateResources(
        total=_sum_expr(gates.total, loop_symbol, start, step, iterations),
        single_qubit=_sum_expr(
            gates.single_qubit, loop_symbol, start, step, iterations
        ),
        two_qubit=_sum_expr(gates.two_qubit, loop_symbol, start, step, iterations),
        multi_qubit=_sum_expr(gates.multi_qubit, loop_symbol, start, step, iterations),
        clifford=_sum_expr(gates.clifford, loop_symbol, start, step, iterations),
        rotation=_sum_expr(gates.rotation, loop_symbol, start, step, iterations),
        t=_sum_expr(gates.t, loop_symbol, start, step, iterations),
        toffoli=_sum_expr(gates.toffoli, loop_symbol, start, step, iterations),
        non_clifford=_sum_expr(
            gates.non_clifford,
            loop_symbol,
            start,
            step,
            iterations,
        ),
    )


def _sum_depth(
    depth: DepthResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> DepthResources:
    """Sum depth resources over a loop.

    Args:
        depth (DepthResources): Depth resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        DepthResources: Summed depth resources.
    """
    return DepthResources(
        depth=_sum_expr(depth.depth, loop_symbol, start, step, iterations),
        clifford_depth=_sum_expr(
            depth.clifford_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        rotation_depth=_sum_expr(
            depth.rotation_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        t_depth=_sum_expr(depth.t_depth, loop_symbol, start, step, iterations),
        toffoli_depth=_sum_expr(
            depth.toffoli_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        non_clifford_depth=_sum_expr(
            depth.non_clifford_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        measurement_depth=_sum_expr(
            depth.measurement_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        gate_depth=_sum_expr(
            depth.gate_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        reset_depth=_sum_expr(
            depth.reset_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
    )


def _sum_measurements(
    measurements: MeasurementResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> MeasurementResources:
    """Sum measurement resources over a loop.

    Args:
        measurements (MeasurementResources): Measurement resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        MeasurementResources: Summed measurement resources.
    """
    return MeasurementResources(
        total=_sum_expr(
            measurements.total,
            loop_symbol,
            start,
            step,
            iterations,
        )
    )


def _sum_resets(
    resets: ResetResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> ResetResources:
    """Sum reset resources over a loop.

    Args:
        resets (ResetResources): Reset resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        ResetResources: Summed reset resources.
    """
    return ResetResources(
        total=_sum_expr(
            resets.total,
            loop_symbol,
            start,
            step,
            iterations,
        )
    )


def _sum_calls(
    calls: CallResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> CallResources:
    """Sum call resources over a loop.

    Args:
        calls (CallResources): Call resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        CallResources: Summed call resources.
    """
    return CallResources(
        calls_by_name={
            name: _sum_expr(value, loop_symbol, start, step, iterations)
            for name, value in calls.calls_by_name.items()
        },
        queries_by_name={
            name: _sum_expr(value, loop_symbol, start, step, iterations)
            for name, value in calls.queries_by_name.items()
        },
    )


def _depth_from_gate_resources(gates: GateResources) -> DepthResources:
    """Create a primitive depth estimate from gate resources.

    Args:
        gates (GateResources): Primitive gate resources.

    Returns:
        DepthResources: One-layer depth categorized by gate type.
    """
    active = cast(
        ResourceExpr,
        sp.Piecewise((_ONE, sp.Gt(gates.total, 0)), (_ZERO, True)),
    )
    return DepthResources(
        depth=active,
        clifford_depth=active if gates.clifford != 0 else _ZERO,
        rotation_depth=active if gates.rotation != 0 else _ZERO,
        t_depth=active if gates.t != 0 else _ZERO,
        toffoli_depth=active if gates.toffoli != 0 else _ZERO,
        non_clifford_depth=active if gates.non_clifford != 0 else _ZERO,
        gate_depth=active,
    )


def _merge_trace(
    name: str,
    left: ResourceTraceNode | None,
    right: ResourceTraceNode | None,
) -> ResourceTraceNode | None:
    """Merge two trace nodes under a parent.

    Args:
        name (str): Parent node name.
        left (ResourceTraceNode | None): Left child.
        right (ResourceTraceNode | None): Right child.

    Returns:
        ResourceTraceNode | None: Parent node or ``None`` when both children
        are absent.
    """
    children = tuple(child for child in (left, right) if child is not None)
    if not children:
        return None
    if len(children) == 1 and name == "seq":
        return children[0]
    return ResourceTraceNode(name=name, source_kind="algebra", children=children)


def _conditional_trace(
    condition: sp.Basic,
    true_trace: ResourceTraceNode | None,
    false_trace: ResourceTraceNode | None,
) -> ResourceTraceNode | None:
    """Merge two trace branches with structured activation guards.

    Args:
        condition (sp.Basic): Predicate selecting ``true_trace``.
        true_trace (ResourceTraceNode | None): True-branch trace.
        false_trace (ResourceTraceNode | None): False-branch trace.

    Returns:
        ResourceTraceNode | None: Conditional trace, or ``None`` when neither
            branch carries a trace.
    """
    guarded_true = true_trace.when(condition) if true_trace is not None else None
    guarded_false = (
        false_trace.when(sp.Not(condition)) if false_trace is not None else None
    )
    return _merge_trace(
        f"if[{condition}]",
        guarded_true,
        guarded_false,
    )


def _wrap_trace(
    name: str,
    child: ResourceTraceNode | None,
    *,
    source_kind: str = "algebra",
    strategy: str | None = None,
) -> ResourceTraceNode:
    """Wrap an optional child trace with a parent node.

    Args:
        name (str): Parent node name.
        child (ResourceTraceNode | None): Optional child.
        source_kind (str): Parent source kind. Defaults to ``"algebra"``.
        strategy (str | None): Selected strategy. Defaults to ``None``.

    Returns:
        ResourceTraceNode: Parent trace node.
    """
    children = (child,) if child is not None else ()
    return ResourceTraceNode(
        name=name,
        source_kind=source_kind,
        strategy=strategy,
        children=children,
    )
