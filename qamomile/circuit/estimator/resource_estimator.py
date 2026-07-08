"""Estimate logical resources by abstractly interpreting qkernel IR."""

from __future__ import annotations

import dataclasses
import enum
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import sympy as sp

from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallTransform,
    CompositeGateType,
    InvokeOperation,
    ResourceModelBinding,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue, Value

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel
    from qamomile.circuit.ir.operation.callable import CallableRef

ResourceExpr = sp.Expr
_ZERO = sp.Integer(0)
_ONE = sp.Integer(1)


class ResourcePolicy(enum.Enum):
    """Select how callable resources are resolved.

    Values:
        EXACT_BODY: Prefer abstract interpretation of callable bodies.
        MODEL_IF_AVAILABLE: Prefer a matching resource model, then body.
        MODEL_ONLY: Require a matching resource model.
        ASYMPTOTIC: Prefer asymptotic or strategy models when present.
        LITERATURE: Prefer literature-backed models when present.
    """

    EXACT_BODY = "exact_body"
    MODEL_IF_AVAILABLE = "model_if_available"
    MODEL_ONLY = "model_only"
    ASYMPTOTIC = "asymptotic"
    LITERATURE = "literature"


class CostBasis(enum.Enum):
    """Describe the logical resource basis requested by the user.

    Values:
        LOGICAL_GATES: Count logical gates without decomposition to a
            fault-tolerant basis.
        CLIFFORD_T: Track Clifford, rotation, and T resources.
        TOFFOLI: Track Toffoli-oriented arithmetic costs.
        QUERY: Track oracle and subroutine query complexity.
        LITERATURE: Use literature-level formulas when available.
    """

    LOGICAL_GATES = "logical_gates"
    CLIFFORD_T = "clifford_t"
    TOFFOLI = "toffoli"
    QUERY = "query"
    LITERATURE = "literature"


class UnknownResourcePolicy(enum.Enum):
    """Control how the estimator handles bodyless unknown callables.

    Values:
        ERROR: Raise when a callable has no body or resource model.
        OPAQUE_CALL: Count one opaque call/query and continue.
        ZERO_WITH_WARNING: Record an assumption and continue with zero cost.
    """

    ERROR = "error"
    OPAQUE_CALL = "opaque_call"
    ZERO_WITH_WARNING = "zero_with_warning"


class EstimateKind(enum.Enum):
    """Classify the source of a resource estimate.

    Values:
        EXACT_DECOMPOSED: Estimate obtained from IR body traversal.
        STRATEGY_MODEL: Estimate obtained from a strategy-specific model.
        ASYMPTOTIC: Estimate obtained from an asymptotic model.
        LITERATURE: Estimate obtained from a literature-backed model.
    """

    EXACT_DECOMPOSED = "exact_decomposed"
    STRATEGY_MODEL = "strategy_model"
    ASYMPTOTIC = "asymptotic"
    LITERATURE = "literature"

    @staticmethod
    def from_name(name: str | None) -> "EstimateKind":
        """Map a ``ResourceModelBinding.estimate_kind`` string to an enum value.

        Args:
            name (str | None): Estimate-kind tag recorded on a resource model
                binding. ``None`` defaults to ``STRATEGY_MODEL``.

        Returns:
            EstimateKind: Matching enum value.

        Raises:
            ValueError: If ``name`` is a non-``None`` string that does not match
                any ``EstimateKind`` value. This surfaces typos (e.g.
                ``"literture"``) that would otherwise silently defeat
                policy-based model selection.
        """
        if name is None:
            return EstimateKind.STRATEGY_MODEL
        for kind in EstimateKind:
            if kind.value == name:
                return kind
        valid = ", ".join(repr(kind.value) for kind in EstimateKind)
        raise ValueError(
            f"Unknown resource model estimate_kind {name!r}; expected one of {valid}."
        )


# Map a resource policy to the estimate-kind tag it prefers when several
# resource models are attached to the same callable. Policies not listed here
# do not express a preference and select the first compatible binding.
_POLICY_PREFERRED_KIND: dict[ResourcePolicy, str] = {
    ResourcePolicy.LITERATURE: EstimateKind.LITERATURE.value,
    ResourcePolicy.ASYMPTOTIC: EstimateKind.ASYMPTOTIC.value,
}


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
            input_qubits=sp.simplify(self.input_qubits),
            allocated_qubits=sp.simplify(self.allocated_qubits),
            clean_ancilla_qubits=sp.simplify(self.clean_ancilla_qubits),
            dirty_ancilla_qubits=sp.simplify(self.dirty_ancilla_qubits),
            peak_qubits=sp.simplify(self.peak_qubits),
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
            total=sp.simplify(self.total),
            single_qubit=sp.simplify(self.single_qubit),
            two_qubit=sp.simplify(self.two_qubit),
            multi_qubit=sp.simplify(self.multi_qubit),
            clifford=sp.simplify(self.clifford),
            rotation=sp.simplify(self.rotation),
            t=sp.simplify(self.t),
            toffoli=sp.simplify(self.toffoli),
            non_clifford=sp.simplify(self.non_clifford),
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
    """

    depth: ResourceExpr = _ZERO
    clifford_depth: ResourceExpr = _ZERO
    rotation_depth: ResourceExpr = _ZERO
    t_depth: ResourceExpr = _ZERO
    toffoli_depth: ResourceExpr = _ZERO
    non_clifford_depth: ResourceExpr = _ZERO
    measurement_depth: ResourceExpr = _ZERO

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
            depth=sp.simplify(self.depth),
            clifford_depth=sp.simplify(self.clifford_depth),
            rotation_depth=sp.simplify(self.rotation_depth),
            t_depth=sp.simplify(self.t_depth),
            toffoli_depth=sp.simplify(self.toffoli_depth),
            non_clifford_depth=sp.simplify(self.non_clifford_depth),
            measurement_depth=sp.simplify(self.measurement_depth),
        )


@dataclasses.dataclass
class CallResources:
    """Track callable and oracle query resources.

    Args:
        calls_by_name (dict[str, ResourceExpr]): Invocation count by callable
            name.
        queries_by_name (dict[str, ResourceExpr]): Query complexity by
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
                name: sp.simplify(count) for name, count in self.calls_by_name.items()
            },
            queries_by_name={
                name: sp.simplify(count) for name, count in self.queries_by_name.items()
            },
        )


@dataclasses.dataclass
class ResourceTraceNode:
    """Represent one node in the resource-estimation explanation tree.

    Args:
        name (str): Operation or callable name.
        source_kind (str): Source type such as ``"primitive"``, ``"body"``,
            ``"model"``, or ``"opaque"``.
        strategy (str | None): Selected resource strategy. Defaults to
            ``None``.
        estimate_kind (EstimateKind | None): Estimate-kind tag. Defaults to
            ``None``.
        summary (str): Short expression summary. Defaults to an empty string.
        assumptions (tuple[ResourceAssumption, ...]): Assumptions local to the
            node. Defaults to an empty tuple.
        children (tuple[ResourceTraceNode, ...]): Nested trace nodes.
            Defaults to an empty tuple.
    """

    name: str
    source_kind: str
    strategy: str | None = None
    estimate_kind: EstimateKind | None = None
    summary: str = ""
    assumptions: tuple[ResourceAssumption, ...] = ()
    children: tuple[ResourceTraceNode, ...] = ()

    def render(self, indent: int = 0) -> str:
        """Render this trace node as plain text.

        Args:
            indent (int): Number of leading spaces. Defaults to ``0``.

        Returns:
            str: Multi-line explanation text.
        """
        prefix = " " * indent
        strategy = f" strategy={self.strategy}" if self.strategy else ""
        kind = f" kind={self.estimate_kind.value}" if self.estimate_kind else ""
        summary = f" {self.summary}" if self.summary else ""
        lines = [f"{prefix}{self.name} [{self.source_kind}{strategy}{kind}]{summary}"]
        for assumption in self.assumptions:
            source = f" ({assumption.source})" if assumption.source else ""
            lines.append(f"{prefix}  assumption: {assumption.message}{source}")
        for child in self.children:
            lines.append(child.render(indent + 2))
        return "\n".join(lines)


@dataclasses.dataclass
class ResourceEstimate:
    """Carry the full logical resource estimate for a qkernel or block.

    Args:
        width (WidthResources): Logical width and ancilla estimate.
        gates (GateResources): Logical gate-resource estimate.
        depth (DepthResources): Logical depth-resource estimate.
        calls (CallResources): Callable/query-resource estimate.
        assumptions (tuple[ResourceAssumption, ...]): Modeling assumptions.
        trace (ResourceTraceNode | None): Explanation tree root. Defaults to
            ``None``.
        parameters (dict[str, sp.Symbol]): Symbols present in the estimate.
            Defaults to an empty dict.
    """

    width: WidthResources = dataclasses.field(default_factory=WidthResources.zero)
    gates: GateResources = dataclasses.field(default_factory=GateResources.zero)
    depth: DepthResources = dataclasses.field(default_factory=DepthResources.zero)
    calls: CallResources = dataclasses.field(default_factory=CallResources.zero)
    assumptions: tuple[ResourceAssumption, ...] = ()
    trace: ResourceTraceNode | None = None
    parameters: dict[str, sp.Symbol] = dataclasses.field(default_factory=dict)

    @property
    def qubits(self) -> ResourceExpr:
        """Return the peak logical qubit alias.

        Returns:
            ResourceExpr: Same value as ``width.peak_qubits``.
        """
        return self.width.peak_qubits

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
        return ResourceEstimate(
            width=_seq_width(self.width, other.width),
            gates=_add_gates(self.gates, other.gates),
            depth=_add_depth(self.depth, other.depth),
            calls=_add_calls(self.calls, other.calls),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("seq", self.trace, other.trace),
        )

    def parallel(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose this estimate in parallel with another estimate.

        Args:
            other (ResourceEstimate): Estimate that runs concurrently.

        Returns:
            ResourceEstimate: Parallel composition.
        """
        return ResourceEstimate(
            width=_parallel_width(self.width, other.width),
            gates=_add_gates(self.gates, other.gates),
            depth=_max_depth(self.depth, other.depth),
            calls=_add_calls(self.calls, other.calls),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("parallel", self.trace, other.trace),
        )

    def choice(self, other: ResourceEstimate) -> ResourceEstimate:
        """Compose a conservative branch choice.

        Args:
            other (ResourceEstimate): Alternative branch estimate.

        Returns:
            ResourceEstimate: Element-wise maximum of both branches.
        """
        return ResourceEstimate(
            width=_max_width(self.width, other.width),
            gates=_max_gates(self.gates, other.gates),
            depth=_max_depth(self.depth, other.depth),
            calls=_max_calls(self.calls, other.calls),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace("choice", self.trace, other.trace),
        )

    def repeat(self, factor: ResourceExpr | int) -> ResourceEstimate:
        """Repeat this estimate with reusable width.

        Args:
            factor (ResourceExpr | int): Iteration or power factor.

        Returns:
            ResourceEstimate: Repeated estimate.
        """
        f = _expr(factor)
        return ResourceEstimate(
            width=self.width,
            gates=_scale_gates(self.gates, f),
            depth=_scale_depth(self.depth, f),
            calls=_scale_calls(self.calls, f),
            assumptions=self.assumptions,
            trace=_wrap_trace(f"repeat({f})", self.trace),
        )

    def controlled(self, num_controls: ResourceExpr | int) -> ResourceEstimate:
        """Record a controlled-call context without scaling cost.

        This deliberately does **not** change gate/width/depth counts. Controlled
        cost is owned by whatever knows the decomposition: a body gets exact
        per-primitive reclassification via ``eval_controlled_u``, and a resource
        model declares its own controlled cost (optionally a distinct model bound
        with ``transform=CallTransform.CONTROLLED``). This method only attaches an
        assumption so that surrounding controls a model did not account for are
        surfaced rather than silently dropped — scaling here would double-count
        calibrated models whose cost already includes their control.

        Args:
            num_controls (ResourceExpr | int): Number of active controls.

        Returns:
            ResourceEstimate: Estimate with a recorded controlled assumption.
        """
        controls = _expr(num_controls)
        assumption = ResourceAssumption(
            message=(
                "controlled transform reuses the selected body/model cost; "
                f"primitive gates inside available bodies see {controls} controls"
            )
        )
        return ResourceEstimate(
            width=self.width,
            gates=self.gates,
            depth=self.depth,
            calls=self.calls,
            assumptions=(*self.assumptions, assumption),
            trace=_wrap_trace(f"controlled({controls})", self.trace),
        )

    def inverse(self) -> ResourceEstimate:
        """Apply an inverse transform.

        Returns:
            ResourceEstimate: Estimate with identical logical resources.
        """
        return ResourceEstimate(
            width=self.width,
            gates=self.gates,
            depth=self.depth,
            calls=self.calls,
            assumptions=self.assumptions,
            trace=_wrap_trace("inverse", self.trace),
            parameters=self.parameters,
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
        iterations = symbolic_iterations(start, stop, step)
        if loop_symbol not in _free_symbols(self):
            return self.repeat(iterations)
        return ResourceEstimate(
            width=self.width,
            gates=_sum_gates(self.gates, loop_symbol, start, step, iterations),
            depth=_sum_depth(self.depth, loop_symbol, start, step, iterations),
            calls=_sum_calls(self.calls, loop_symbol, start, step, iterations),
            assumptions=self.assumptions,
            trace=_wrap_trace(f"sum({loop_symbol}={start}..{stop})", self.trace),
        )

    def substitute(self, **values: int | float) -> ResourceEstimate:
        """Substitute concrete values for symbolic parameters.

        Args:
            **values (int | float): Mapping from parameter name to value.

        Returns:
            ResourceEstimate: Estimate with substituted expressions.
        """
        subs: dict[Any, Any] = {}
        for name, value in values.items():
            symbol = self.parameters.get(name)
            if symbol is None:
                symbol = sp.Symbol(name, integer=True, positive=True)
            subs[symbol] = value
        return self._map_expr(lambda expr: cast(sp.Expr, expr.subs(subs).doit()))

    def simplify(self) -> ResourceEstimate:
        """Simplify all symbolic expressions.

        Returns:
            ResourceEstimate: Simplified estimate.
        """
        simplified = ResourceEstimate(
            width=self.width.simplify(),
            gates=self.gates.simplify(),
            depth=self.depth.simplify(),
            calls=self.calls.simplify(),
            assumptions=self.assumptions,
            trace=self.trace,
        )
        simplified.parameters = _collect_parameters(simplified)
        return simplified

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
        return {
            "width": {
                "input_qubits": str(self.width.input_qubits),
                "allocated_qubits": str(self.width.allocated_qubits),
                "clean_ancilla_qubits": str(self.width.clean_ancilla_qubits),
                "dirty_ancilla_qubits": str(self.width.dirty_ancilla_qubits),
                "peak_qubits": str(self.width.peak_qubits),
            },
            "gates": {
                "total": str(self.gates.total),
                "single_qubit": str(self.gates.single_qubit),
                "two_qubit": str(self.gates.two_qubit),
                "multi_qubit": str(self.gates.multi_qubit),
                "clifford": str(self.gates.clifford),
                "rotation": str(self.gates.rotation),
                "t": str(self.gates.t),
                "toffoli": str(self.gates.toffoli),
                "non_clifford": str(self.gates.non_clifford),
            },
            "depth": {
                "depth": str(self.depth.depth),
                "clifford_depth": str(self.depth.clifford_depth),
                "rotation_depth": str(self.depth.rotation_depth),
                "t_depth": str(self.depth.t_depth),
                "toffoli_depth": str(self.depth.toffoli_depth),
                "non_clifford_depth": str(self.depth.non_clifford_depth),
                "measurement_depth": str(self.depth.measurement_depth),
            },
            "calls": {
                "calls_by_name": {
                    name: str(value) for name, value in self.calls.calls_by_name.items()
                },
                "queries_by_name": {
                    name: str(value)
                    for name, value in self.calls.queries_by_name.items()
                },
            },
            "assumptions": [
                {"message": assumption.message, "source": assumption.source}
                for assumption in self.assumptions
            ],
            "parameters": {
                name: str(symbol) for name, symbol in self.parameters.items()
            },
        }

    def _map_expr(self, fn: Any) -> ResourceEstimate:
        """Apply a function to every symbolic expression.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.

        Returns:
            ResourceEstimate: Rewritten estimate.
        """
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
            depth=DepthResources(
                depth=fn(self.depth.depth),
                clifford_depth=fn(self.depth.clifford_depth),
                rotation_depth=fn(self.depth.rotation_depth),
                t_depth=fn(self.depth.t_depth),
                toffoli_depth=fn(self.depth.toffoli_depth),
                non_clifford_depth=fn(self.depth.non_clifford_depth),
                measurement_depth=fn(self.depth.measurement_depth),
            ),
            calls=CallResources(
                calls_by_name={
                    name: fn(value) for name, value in self.calls.calls_by_name.items()
                },
                queries_by_name={
                    name: fn(value)
                    for name, value in self.calls.queries_by_name.items()
                },
            ),
            assumptions=self.assumptions,
            trace=self.trace,
        )
        mapped.parameters = _collect_parameters(mapped)
        return mapped


@dataclasses.dataclass(frozen=True)
class ResourceContext:
    """Provide call-site information to resource models.

    Controlled-cost contract for model authors: a controlled invocation is its
    own operation with its own cost, so a model owns the cost of the control it
    was designed around. ``controls`` counts only *surrounding* controls (from
    enclosing controlled scopes) that the model did not build in; the
    invocation's *own* control is signalled by ``transform`` being
    ``CallTransform.CONTROLLED`` and counted by :attr:`own_controls`. A model
    that already prices in its own control (like the constant modular
    multiplication model, calibrated per *controlled* multiplication) should not
    add anything for ``own_controls``; the estimator records leftover
    ``controls`` as an assumption but never scales a model's counts.

    Args:
        callable_ref (CallableRef | None): Stable callable identity.
        argument_values (tuple[Value, ...]): IR operand values from the call
            site.
        operand_shapes (Mapping[str, ResourceExpr]): Resolved operand shape
            expressions keyed by operand name.
        attrs (Mapping[str, Any]): Callable attrs copied from the invocation.
        loop_symbols (Mapping[str, sp.Symbol]): Loop symbols in scope.
        controls (ResourceExpr): Number of surrounding controls (not the
            invocation's own control).
        power (ResourceExpr): Power for repeated calls. Reserved: the
            invoke-model path does not populate it today (powered invokes are
            handled structurally by ``eval_controlled_u`` above the model), so it
            stays at the default ``1``.
        transform (CallTransform): Requested call transform.
        strategy (str | None): Selected resource strategy.
        policy (ResourcePolicy): Estimation policy.
        cost_basis (CostBasis): Requested cost basis.
        bindings (Mapping[str, Any]): Concrete bindings supplied by the user.
    """

    callable_ref: "CallableRef | None"
    argument_values: tuple[Value, ...]
    operand_shapes: Mapping[str, ResourceExpr]
    attrs: Mapping[str, Any]
    loop_symbols: Mapping[str, sp.Symbol]
    controls: ResourceExpr = _ZERO
    power: ResourceExpr = _ONE
    transform: CallTransform = CallTransform.DIRECT
    strategy: str | None = None
    policy: ResourcePolicy = ResourcePolicy.MODEL_IF_AVAILABLE
    cost_basis: CostBasis = CostBasis.LOGICAL_GATES
    bindings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def own_controls(self) -> int:
        """Return the invocation's own control-qubit count.

        Reads ``attrs["num_control_qubits"]`` when the call is controlled. This
        is the control the callable is applied *with* (e.g. Shor conditioning a
        modular multiplication on an exponent qubit), distinct from surrounding
        controls in :attr:`controls`.

        Returns:
            int: Own control-qubit count, or ``0`` for a direct call.
        """
        if self.transform is not CallTransform.CONTROLLED:
            return 0
        return int(self.attrs.get("num_control_qubits", 0) or 0)


class ResourceModel(Protocol):
    """Protocol for context-aware callable resource models."""

    def estimate(self, ctx: ResourceContext) -> ResourceEstimate:
        """Estimate resources for a callable invocation.

        Args:
            ctx (ResourceContext): Call-site resource context.

        Returns:
            ResourceEstimate: Estimated logical resources.
        """
        ...


@dataclasses.dataclass(frozen=True, init=False)
class FixedResourceModel:
    """Return a fixed resource estimate for bodyless callables.

    Args:
        estimate_value (ResourceEstimate | None): Estimate returned for every
            call. When omitted, an estimate is built from the resource parts.
        width (WidthResources | None): Width resources for the fixed estimate.
            Defaults to ``None``.
        gates (GateResources | None): Gate resources for the fixed estimate.
            Defaults to ``None``.
        depth (DepthResources | None): Depth resources for the fixed estimate.
            Defaults to ``None``.
        calls (CallResources | None): Call/query resources for the fixed
            estimate. Defaults to ``None``.
        assumptions (Sequence[ResourceAssumption]): Assumptions attached to the
            fixed estimate. Defaults to an empty tuple.

    Raises:
        ValueError: If ``estimate_value`` is supplied together with explicit
            resource parts.
    """

    estimate_value: ResourceEstimate

    def __init__(
        self,
        estimate_value: ResourceEstimate | None = None,
        *,
        width: WidthResources | None = None,
        gates: GateResources | None = None,
        depth: DepthResources | None = None,
        calls: CallResources | None = None,
        assumptions: Sequence[ResourceAssumption] = (),
    ) -> None:
        """Initialize a fixed resource model.

        Args:
            estimate_value (ResourceEstimate | None): Complete estimate to
                return. Defaults to ``None``.
            width (WidthResources | None): Width resources to include when
                building an estimate from parts. Defaults to ``None``.
            gates (GateResources | None): Gate resources to include when
                building an estimate from parts. Defaults to ``None``.
            depth (DepthResources | None): Depth resources to include when
                building an estimate from parts. Defaults to ``None``.
            calls (CallResources | None): Call/query resources to include when
                building an estimate from parts. Defaults to ``None``.
            assumptions (Sequence[ResourceAssumption]): Assumptions to attach
                when building an estimate from parts. Defaults to ``()``.

        Raises:
            ValueError: If a complete estimate and explicit parts are both
                supplied.
        """
        has_parts = any(
            part is not None for part in (width, gates, depth, calls)
        ) or bool(assumptions)
        if estimate_value is not None and has_parts:
            raise ValueError(
                "FixedResourceModel accepts either estimate_value or resource "
                "parts, not both."
            )
        if estimate_value is None:
            estimate_value = ResourceEstimate(
                width=width or WidthResources(),
                gates=gates or GateResources(),
                depth=depth or DepthResources(),
                calls=calls or CallResources(),
                assumptions=tuple(assumptions),
            )
        object.__setattr__(self, "estimate_value", estimate_value)

    def estimate(self, ctx: ResourceContext) -> ResourceEstimate:
        """Return the fixed estimate scaled by call power.

        Args:
            ctx (ResourceContext): Call-site context carrying ``power``.

        Returns:
            ResourceEstimate: Fixed estimate repeated by ``ctx.power``.
        """
        return self.estimate_value.repeat(ctx.power)


@dataclasses.dataclass
class ResourceEstimatorConfig:
    """Configure ``ResourceEstimator`` behavior.

    Args:
        policy (ResourcePolicy): Callable resolution policy.
        cost_basis (CostBasis): Logical cost basis.
        strategies (dict[str, str]): Strategy overrides by callable name.
        trace (bool): Whether estimates should carry trace nodes.
        simplify (bool): Whether to simplify the final estimate.
        unknown_policy (UnknownResourcePolicy): Handling for unknown opaque
            callables.
    """

    policy: ResourcePolicy = ResourcePolicy.MODEL_IF_AVAILABLE
    cost_basis: CostBasis = CostBasis.LOGICAL_GATES
    strategies: dict[str, str] = dataclasses.field(default_factory=dict)
    trace: bool = True
    simplify: bool = True
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR


class ResourceEstimator:
    """Estimate logical resources for qkernels and IR blocks."""

    def __init__(
        self,
        *,
        policy: ResourcePolicy = ResourcePolicy.MODEL_IF_AVAILABLE,
        cost_basis: CostBasis = CostBasis.LOGICAL_GATES,
        strategies: dict[str, str] | None = None,
        trace: bool = True,
        simplify: bool = True,
        unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
    ) -> None:
        """Initialize a resource estimator.

        Args:
            policy (ResourcePolicy): Callable resolution policy.
            cost_basis (CostBasis): Logical cost basis.
            strategies (dict[str, str] | None): Strategy overrides by callable
                name. Defaults to ``None``.
            trace (bool): Whether to keep explanation traces. Defaults to
                ``True``.
            simplify (bool): Whether to simplify the final estimate. Defaults
                to ``True``.
            unknown_policy (UnknownResourcePolicy): Handling for unknown
                bodyless callables. Defaults to ``ERROR``.
        """
        self.config = ResourceEstimatorConfig(
            policy=policy,
            cost_basis=cost_basis,
            strategies=dict(strategies or {}),
            trace=trace,
            simplify=simplify,
            unknown_policy=unknown_policy,
        )

    def estimate(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        *,
        bindings: dict[str, Any] | None = None,
        substitutions: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        strategies: dict[str, str] | None = None,
    ) -> ResourceEstimate:
        """Estimate logical resources for a qkernel, block, or operation list.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Object to
                estimate. QKernel-like objects are built before traversal.
            bindings (dict[str, Any] | None): Compile-time values baked into the
                circuit at ``build`` time (they change the constructed IR). Use
                for genuinely structural values. Defaults to ``None``.
            substitutions (dict[str, Any] | None): Estimation-only substitutions
                applied to the *symbolic* estimate after building. The kernel is
                built with these names left symbolic, so
                ``substitutions={"n": 2048}`` yields the concrete estimate
                without constructing a 2048-scale circuit. Values may be numbers
                or SymPy expressions. Defaults to ``None``.
            parameters (list[str] | None): Runtime parameter names to preserve
                when a qkernel must be built. Defaults to ``None``.
            strategies (dict[str, str] | None): Per-call override merged over
                estimator-level strategies. Defaults to ``None``.

        Returns:
            ResourceEstimate: Logical resource estimate.

        Raises:
            ValueError: If a name appears in both ``bindings`` and
                ``substitutions``, or a ``substitutions`` name is neither a free
                symbol of the estimate nor a declared kernel argument (a kernel
                argument that affects no resource metric is a recorded no-op).
        """
        if bindings and substitutions:
            overlap = set(bindings) & set(substitutions)
            if overlap:
                raise ValueError(
                    f"names {sorted(overlap)} appear in both bindings and "
                    "substitutions; a name is either baked in at build time "
                    "(bindings) or substituted into the estimate "
                    "(substitutions), not both."
                )
        block_or_ops = self._coerce_input(kernel, bindings, parameters, substitutions)
        config = dataclasses.replace(
            self.config,
            strategies={**self.config.strategies, **dict(strategies or {})},
        )
        interpreter = ResourceInterpreter(
            config=config,
            bindings=bindings or {},
            condition_values=_scalar_values(
                {**(bindings or {}), **(substitutions or {})}
            ),
        )
        estimate = interpreter.estimate(block_or_ops)
        if bindings:
            estimate = _substitute_bindings(estimate, bindings)
        if substitutions:
            estimate = _apply_substitutions(
                estimate,
                substitutions,
                contract_names=_contract_names(block_or_ops),
                branch_condition_names=interpreter.branch_condition_names,
            )
        if config.simplify:
            estimate = estimate.simplify()
        estimate.parameters = _collect_parameters(estimate)
        return estimate

    def _coerce_input(
        self,
        kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
        bindings: dict[str, Any] | None,
        parameters: list[str] | None,
        substitutions: dict[str, Any] | None = None,
    ) -> Block | Sequence[Operation]:
        """Coerce a supported input into an IR block or operation list.

        When a qkernel is built and ``substitutions`` names classical arguments,
        those arguments are forced to stay symbolic (added to the build
        ``parameters``) so a Python-signature default cannot silently bake a
        value the caller believes they passed via ``substitutions``.

        Args:
            kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Input
                object.
            bindings (dict[str, Any] | None): Compile-time values for qkernel
                build. Defaults to ``None``.
            parameters (list[str] | None): Runtime parameter names for qkernel
                build. Defaults to ``None``.
            substitutions (dict[str, Any] | None): Estimation-only substitution
                names, used to force matching classical arguments symbolic.
                Defaults to ``None``.

        Returns:
            Block | Sequence[Operation]: IR object ready for interpretation.
        """
        if isinstance(kernel, Block):
            return kernel
        if isinstance(kernel, Sequence):
            return kernel
        build = getattr(kernel, "build", None)
        if callable(build):
            kwargs = dict(bindings or {})
            parameters = _force_symbolic_substitution_params(
                kernel, kwargs, parameters, substitutions
            )
            return build(parameters=parameters, **kwargs)
        block = getattr(kernel, "block", None)
        if isinstance(block, Block):
            return block
        raise TypeError(
            "ResourceEstimator.estimate() expects a QKernel, Block, or "
            "sequence of Operation objects."
        )


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
                values (from bindings and substitutions) used to decide
                compile-time ``if`` branches. Defaults to ``None``.
        """
        self.config = config
        self.bindings = bindings
        self.condition_values: Mapping[str, sp.Expr] = condition_values or {}
        # Names of classical parameters whose value participated in a
        # compile-time branch predicate (whether or not it decided the branch),
        # so downstream substitution reporting does not misfile them as "does
        # not affect any resource metric".
        self.branch_condition_names: set[str] = set()
        # Undecidable-branch messages already reported, so an IfOperation
        # duplicated at trace time (e.g. a Python-level loop) yields one
        # assumption, not one per copy.
        self._reported_undecidable: set[str] = set()

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
            resolver = ExprResolver(block=block_or_ops)
            body = self.eval_block(block_or_ops, resolver)
            input_qubits = _count_input_qubits(block_or_ops.input_values, resolver)
            width = dataclasses.replace(
                body.width,
                input_qubits=input_qubits,
                peak_qubits=sp.simplify(input_qubits + body.width.allocated_qubits),
            )
            return dataclasses.replace(
                body,
                width=width,
                trace=_wrap_trace(block_or_ops.name or "qkernel", body.trace),
            )
        resolver = ExprResolver()
        return self.eval_operations(list(block_or_ops), resolver)

    def eval_block(self, block: Block, resolver: ExprResolver) -> ResourceEstimate:
        """Evaluate a block body.

        Args:
            block (Block): Block to evaluate.
            resolver (ExprResolver): Resolver scoped to ``block``.

        Returns:
            ResourceEstimate: Estimated resources for the body.
        """
        return self.eval_operations(block.operations, resolver)

    def eval_operations(
        self,
        operations: list[Operation],
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a list of operations sequentially.

        Args:
            operations (list[Operation]): Operations to evaluate.
            resolver (ExprResolver): Value resolver for this scope.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Sequential composition of operation resources.
        """
        estimate = ResourceEstimate.zero()
        for operation in operations:
            estimate = estimate.seq(
                self.eval_operation(operation, resolver, controls=controls)
            )
        return estimate

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
        """
        match operation:
            case GateOperation():
                return self.eval_gate(operation, controls=controls)
            case QInitOperation():
                return self.eval_qinit(operation, resolver)
            case (
                MeasureOperation() | MeasureVectorOperation() | MeasureQFixedOperation()
            ):
                return self.eval_measure(operation)
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
            case ControlledUOperation():
                return self.eval_controlled_u(operation, resolver, controls=controls)
            case InverseBlockOperation():
                return self.eval_inverse_block(operation, resolver, controls=controls)
            case PauliEvolveOp():
                return self.eval_pauli_evolve(operation)
            case HasNestedOps():
                assumption = ResourceAssumption(
                    f"unhandled nested operation {type(operation).__name__}",
                    source=type(operation).__name__,
                )
                return ResourceEstimate(assumptions=(assumption,))
            case _:
                return ResourceEstimate.zero()

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
        gates = _classify_gate(operation, num_controls=controls)
        name = operation.gate_type.name.lower() if operation.gate_type else "gate"
        return ResourceEstimate.primitive(name, gates)

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
        width = WidthResources(
            allocated_qubits=count,
            peak_qubits=count,
        )
        return ResourceEstimate(
            width=width,
            trace=ResourceTraceNode("qinit", "primitive", summary=f"qubits={count}"),
        )

    def eval_measure(self, operation: Operation) -> ResourceEstimate:
        """Evaluate a measurement operation.

        Args:
            operation (Operation): Measurement-like operation.

        Returns:
            ResourceEstimate: Measurement depth estimate.
        """
        return ResourceEstimate(
            depth=DepthResources(depth=_ONE, measurement_depth=_ONE),
            trace=ResourceTraceNode(type(operation).__name__, "primitive"),
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
        inner = self.eval_operations(
            operation.operations,
            child,
            controls=controls,
        )
        return inner.sum_over(loop_symbol, start, stop, step)

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
        """
        child, trip_count = build_while_scope(operation, resolver)
        inner = self.eval_operations(operation.operations, child, controls=controls)
        return inner.repeat(trip_count)

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
        ``substitutions``), only the taken branch is counted. A measurement-backed
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
        if taken is not None:
            branch_ops = (
                operation.true_operations if taken else operation.false_operations
            )
            branch_child = true_child if taken else false_child
            estimate = self.eval_operations(branch_ops, branch_child, controls=controls)
            return dataclasses.replace(
                estimate,
                trace=_wrap_trace(
                    f"if[{'true' if taken else 'false'} branch]",
                    estimate.trace,
                ),
            )
        true_estimate = self.eval_operations(
            operation.true_operations,
            true_child,
            controls=controls,
        )
        false_estimate = self.eval_operations(
            operation.false_operations,
            false_child,
            controls=controls,
        )
        combined = true_estimate.choice(false_estimate)
        if note is None:
            return combined
        trace = combined.trace
        if trace is not None:
            trace = dataclasses.replace(trace, assumptions=(*trace.assumptions, note))
        return dataclasses.replace(
            combined, assumptions=(*combined.assumptions, note), trace=trace
        )

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
                name = str(symbol)
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
        unresolved = ", ".join(sorted(str(s) for s in condition.free_symbols))
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
        """
        child = build_for_items_scope(operation, resolver)
        cardinality = resolve_for_items_cardinality(operation)
        inner = self.eval_operations(operation.operations, child, controls=controls)
        return inner.repeat(cardinality)

    def eval_invoke(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a callable invocation using policy-driven resolution.

        Args:
            operation (InvokeOperation): Callable invocation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Invocation resource estimate.

        Raises:
            ValueError: If the invocation cannot be estimated under the
                configured policy.
        """
        strategy = self._strategy_for(operation)
        ctx = self._resource_context(
            operation,
            resolver,
            controls=_expr(controls),
            strategy=strategy,
        )
        body = operation.effective_body(strategy=strategy)
        binding = self._resource_model_for(operation, strategy=strategy)

        if self.config.policy is ResourcePolicy.EXACT_BODY:
            if isinstance(body, Block):
                return self._estimate_invoke_body(operation, body, resolver, controls)
            if binding is not None:
                return self._estimate_model(operation, binding, ctx)
            return self._handle_unknown_invoke(operation, ctx)

        if self.config.policy is ResourcePolicy.MODEL_ONLY:
            if binding is None:
                raise ValueError(
                    f"Callable '{operation.custom_name}' has no resource model."
                )
            return self._estimate_model(operation, binding, ctx)

        if binding is not None:
            return self._estimate_model(operation, binding, ctx)
        if isinstance(body, Block):
            return self._estimate_invoke_body(operation, body, resolver, controls)
        if operation.gate_type in (CompositeGateType.QFT, CompositeGateType.IQFT):
            return self._estimate_qft_iqft(operation, resolver)
        return self._handle_unknown_invoke(operation, ctx)

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
        """
        local_controls, num_targets = _resolve_controlled_u(operation, resolver)
        total_controls = _expr(controls) + _expr(local_controls)
        power = resolver.resolve(operation.power)
        if isinstance(operation.block, Block):
            child = _controlled_u_child_resolver(operation, resolver)
            body = self.eval_operations(
                operation.block.operations,
                child,
                controls=total_controls,
            )
            return body.repeat(power)

        name = (
            operation.callable_ref.name
            if operation.callable_ref is not None
            else "controlled_u"
        )
        gates = GateResources(
            total=power,
            multi_qubit=sp.Piecewise(
                (_ONE, sp.Gt(total_controls + num_targets, 2)),
                (_ZERO, True),
            )
            * power,
            non_clifford=power,
        )
        calls = CallResources(
            calls_by_name={name: power},
            queries_by_name={name: power},
        )
        return ResourceEstimate(
            gates=gates,
            calls=calls,
            depth=DepthResources(depth=power, non_clifford_depth=power),
            trace=ResourceTraceNode(
                name,
                "opaque",
                summary=f"controlled power={power}",
            ),
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
        """
        if not isinstance(operation.implementation_block, Block):
            return ResourceEstimate.zero(operation.name).inverse()
        child = _inverse_block_child_resolver(operation, resolver)
        return self.eval_operations(
            operation.implementation_block.operations,
            child,
            controls=_expr(controls) + operation.num_control_qubits,
        ).inverse()

    def eval_pauli_evolve(self, operation: PauliEvolveOp) -> ResourceEstimate:
        """Evaluate a Pauli evolution operation when its Hamiltonian is bound.

        Args:
            operation (PauliEvolveOp): Pauli evolution operation.

        Returns:
            ResourceEstimate: Gate resources from the Hamiltonian structure,
            or a zero estimate with an assumption when unbound.
        """
        import qamomile.observable as qm_o

        hamiltonian = None
        observable = operation.observable
        if hasattr(observable, "name") and observable.name in self.bindings:
            hamiltonian = self.bindings[observable.name]
        if hamiltonian is None and hasattr(observable, "uuid"):
            hamiltonian = self.bindings.get(observable.uuid)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            assumption = ResourceAssumption(
                "PauliEvolveOp requires a bound Hamiltonian for gate resources.",
                source="PauliEvolveOp",
            )
            return ResourceEstimate(assumptions=(assumption,))
        return ResourceEstimate.primitive(
            "pauli_evolve",
            _classify_pauli_evolve(hamiltonian),
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

    def _resource_model_for(
        self,
        operation: InvokeOperation,
        *,
        strategy: str | None,
    ) -> "ResourceModelBinding | None":
        """Select a resource model binding for an invocation.

        Bindings are filtered by strategy and transform compatibility. When the
        active policy prefers a specific estimate kind (``LITERATURE`` prefers
        literature-tagged models, ``ASYMPTOTIC`` prefers asymptotic ones), a
        compatible binding carrying that kind is returned first. Otherwise, under
        the default policy, a callable may pin its default explicitly via the
        ``default_estimate_kind`` attr (so the choice is not silently
        order-dependent); if neither applies, the first compatible binding wins.
        This keeps callables that ship both a literature-backed formula and a
        structural/asymptotic model usable under either policy without changing
        the callable definition.

        Args:
            operation (InvokeOperation): Invocation to resolve.
            strategy (str | None): Requested strategy.

        Returns:
            ResourceModelBinding | None: Matching binding, if available.
        """
        definition = operation.definition
        bindings = getattr(definition, "resource_models", ()) if definition else ()
        compatible: list[ResourceModelBinding] = []
        for binding in bindings:
            binding_strategy = getattr(binding, "strategy", None)
            binding_transform = getattr(binding, "transform", None)
            if binding_strategy in (None, strategy) and binding_transform in (
                None,
                operation.transform,
            ):
                model = getattr(binding, "model", None)
                if _is_resource_model(model):
                    compatible.append(cast("ResourceModelBinding", binding))
        if not compatible:
            return None
        # A policy preference (LITERATURE/ASYMPTOTIC) is a soft request: prefer a
        # matching binding, but fall back gracefully so a callable with only one
        # model is still usable under either policy.
        policy_kind = _POLICY_PREFERRED_KIND.get(self.config.policy)
        if policy_kind is not None:
            for binding in compatible:
                if getattr(binding, "estimate_kind", None) == policy_kind:
                    return binding
            return compatible[0]
        # No policy preference: an explicit author pin
        # (``attrs["default_estimate_kind"]``) makes the default order-independent
        # and is STRICT — a pin that matches no compatible model is an authoring
        # error, not a silent fallback.
        def_attrs: Mapping[str, Any] = getattr(definition, "attrs", None) or {}
        pinned_kind = def_attrs.get("default_estimate_kind")
        if pinned_kind is not None:
            for binding in compatible:
                if getattr(binding, "estimate_kind", None) == pinned_kind:
                    return binding
            name = operation.custom_name or operation.target.name
            raise ValueError(
                f"Callable '{name}' pins default_estimate_kind={pinned_kind!r} "
                "but no compatible resource model has that estimate kind. Attach "
                "a model tagged with that kind or remove the pin."
            )
        return compatible[0]

    def _resource_context(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr,
        strategy: str | None,
    ) -> ResourceContext:
        """Build a resource model context for an invocation.

        Args:
            operation (InvokeOperation): Invocation operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr): Surrounding control count.
            strategy (str | None): Selected strategy.

        Returns:
            ResourceContext: Model context.
        """
        return ResourceContext(
            callable_ref=operation.target,
            argument_values=tuple(operation.operands),
            operand_shapes=_operand_shapes(operation.operands, resolver),
            attrs=operation.attrs,
            loop_symbols=resolver.loop_var_names,
            controls=controls,
            transform=operation.transform,
            strategy=strategy,
            policy=self.config.policy,
            cost_basis=self.config.cost_basis,
            bindings=self.bindings,
        )

    def _estimate_model(
        self,
        operation: InvokeOperation,
        binding: "ResourceModelBinding",
        ctx: ResourceContext,
    ) -> ResourceEstimate:
        """Estimate an invocation through a resource model binding.

        Args:
            operation (InvokeOperation): Invocation operation.
            binding (ResourceModelBinding): Selected resource model binding.
                Its ``estimate_kind`` tag classifies the resulting trace node.
            ctx (ResourceContext): Model context.

        Returns:
            ResourceEstimate: Modeled estimate.
        """
        model = cast(ResourceModel, binding.model)
        estimate = model.estimate(ctx)
        # Controlled cost is owned by the model (a controlled composite is its
        # own bloq with its own cost formula), so the estimator never scales a
        # model's counts. But any *surrounding* controls the model did not
        # account for are recorded as an assumption via the no-op
        # ``.controlled`` so a ``qmc.control(...)`` around a modeled box cannot
        # silently under-report. ``ctx.controls`` is surrounding controls only;
        # the invoke's own control is visible via ``ctx.transform`` /
        # ``ctx.own_controls`` and is the model author's responsibility.
        if ctx.controls != _ZERO:
            estimate = estimate.controlled(ctx.controls)
        return dataclasses.replace(
            estimate,
            trace=_wrap_trace(
                operation.custom_name,
                estimate.trace,
                source_kind="model",
                strategy=ctx.strategy,
                estimate_kind=EstimateKind.from_name(
                    getattr(binding, "estimate_kind", None)
                ),
            ),
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
        body is classified as controlled — matching how ``eval_controlled_u``
        treats a block body and keeping the body path symmetric with the model
        path's own-control contract.

        Args:
            operation (InvokeOperation): Invocation operation.
            body (Block): Selected callable body.
            resolver (ExprResolver): Call-site resolver.
            controls (ResourceExpr | int): Surrounding control count.

        Returns:
            ResourceEstimate: Body-derived estimate.
        """
        child = resolver.call_child_scope(operation, called_block=body)
        own_controls = 0
        if operation.transform is CallTransform.CONTROLLED:
            own_controls = int(operation.attrs.get("num_control_qubits", 0) or 0)
        total_controls = _expr(controls) + own_controls
        body_estimate = self.eval_operations(
            body.operations, child, controls=total_controls
        )
        if operation.transform is CallTransform.INVERSE:
            body_estimate = body_estimate.inverse()
        return dataclasses.replace(
            body_estimate,
            trace=_wrap_trace(
                operation.custom_name,
                body_estimate.trace,
                source_kind="body",
                estimate_kind=EstimateKind.EXACT_DECOMPOSED,
            ),
        )

    def _estimate_qft_iqft(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Estimate standard QFT/IQFT when only a symbolic box is present.

        Args:
            operation (InvokeOperation): QFT or IQFT invocation.
            resolver (ExprResolver): Call-site resolver.

        Returns:
            ResourceEstimate: Standard logical gate estimate.
        """
        n = _resolve_qft_n(operation, resolver)
        gates = _qft_iqft_resources(n)
        return ResourceEstimate(
            gates=gates,
            depth=DepthResources(depth=gates.total, rotation_depth=gates.rotation),
            trace=ResourceTraceNode(operation.custom_name, "model", summary=f"n={n}"),
        )

    def _handle_unknown_invoke(
        self,
        operation: InvokeOperation,
        ctx: ResourceContext,
    ) -> ResourceEstimate:
        """Handle an invocation without body or resource model.

        Args:
            operation (InvokeOperation): Invocation operation.
            ctx (ResourceContext): Call-site context.

        Returns:
            ResourceEstimate: Opaque or zero estimate when policy permits.

        Raises:
            ValueError: If the configured unknown policy is ``ERROR``.
        """
        name = operation.custom_name
        if self.config.unknown_policy is UnknownResourcePolicy.OPAQUE_CALL:
            return ResourceEstimate(
                calls=CallResources(
                    calls_by_name={name: ctx.power},
                    queries_by_name={name: ctx.power},
                ),
                trace=ResourceTraceNode(name, "opaque", summary=f"power={ctx.power}"),
            )
        if self.config.unknown_policy is UnknownResourcePolicy.ZERO_WITH_WARNING:
            assumption = ResourceAssumption(
                "unknown callable counted as zero resources",
                source=name,
            )
            return ResourceEstimate(
                assumptions=(assumption,),
                trace=ResourceTraceNode(name, "opaque", assumptions=(assumption,)),
            )
        raise ValueError(
            f"Cannot estimate resources for callable '{name}': no resource "
            f"model or body is available."
        )


def estimate_resources(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
    *,
    bindings: dict[str, Any] | None = None,
    substitutions: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
    policy: ResourcePolicy = ResourcePolicy.MODEL_IF_AVAILABLE,
    cost_basis: CostBasis = CostBasis.LOGICAL_GATES,
    strategies: dict[str, str] | None = None,
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
) -> ResourceEstimate:
    """Estimate logical resources using the default estimator facade.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): QKernel,
            block, or operation sequence to estimate.
        bindings (dict[str, Any] | None): Compile-time bindings baked into the
            circuit at build time. Defaults to ``None``.
        substitutions (dict[str, Any] | None): Estimation-only substitutions
            applied to the symbolic estimate after building, so
            ``substitutions={"n": 2048}`` yields the concrete estimate without
            constructing a 2048-scale circuit. Values may be numbers or SymPy
            expressions. Defaults to ``None``.
        parameters (list[str] | None): Runtime parameter names preserved during
            qkernel build. Defaults to ``None``.
        policy (ResourcePolicy): Callable resolution policy. Defaults to
            ``MODEL_IF_AVAILABLE``.
        cost_basis (CostBasis): Logical cost basis. Defaults to
            ``LOGICAL_GATES``.
        strategies (dict[str, str] | None): Strategy overrides by callable
            name. Defaults to ``None``.
        unknown_policy (UnknownResourcePolicy): Unknown callable handling.
            Defaults to ``ERROR``.

    Returns:
        ResourceEstimate: Logical resource estimate.

    Example:
        >>> from qamomile.circuit.stdlib import shor_order_finding
        >>> est = estimate_resources(
        ...     shor_order_finding, substitutions={"n": 2048}
        ... )
        >>> int(est.qubits)
        6144
    """
    estimator = ResourceEstimator(
        policy=policy,
        cost_basis=cost_basis,
        strategies=strategies,
        unknown_policy=unknown_policy,
    )
    return estimator.estimate(
        kernel,
        bindings=bindings,
        substitutions=substitutions,
        parameters=parameters,
    )


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


class _LocalBlock:
    """Provide a minimal operation container for nested resource scopes.

    Args:
        operations (list[Operation]): Operations visible in the nested scope.
    """

    __slots__ = ("operations",)

    def __init__(self, operations: list[Operation]) -> None:
        """Initialize a local operation container.

        Args:
            operations (list[Operation]): Operations visible in the nested
                scope.
        """
        self.operations = operations


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
    """
    loop_symbol = sp.Symbol(operation.loop_var, integer=True, positive=True)
    loop_var_names = _collect_loop_var_names(
        operation.operations,
        operation.loop_var,
        loop_symbol,
    )
    child = resolver.child_scope(
        inner_block=_LocalBlock(operation.operations),
        extra_loop_vars=loop_var_names,
    )
    start = child.resolve(operation.operands[0])
    stop = child.resolve(operation.operands[1])
    step = (
        child.resolve(operation.operands[2]) if len(operation.operands) >= 3 else _ONE
    )
    return child, start, stop, step, loop_symbol


def _collect_loop_var_names(
    operations: list[Operation],
    loop_var_name: str,
    loop_symbol: sp.Symbol,
) -> dict[str, sp.Symbol]:
    """Collect value-name aliases for a loop variable.

    Args:
        operations (list[Operation]): Loop body operations.
        loop_var_name (str): Display name of the loop variable.
        loop_symbol (sp.Symbol): Symbol representing the loop variable.

    Returns:
        dict[str, sp.Symbol]: Name-to-symbol aliases for resolver fallback.
    """
    result: dict[str, sp.Symbol] = {loop_var_name: loop_symbol}
    seen: set[str] = set()

    def check(operation: Operation) -> None:
        """Inspect one operation's operands.

        Args:
            operation (Operation): Operation to inspect.
        """
        for operand in getattr(operation, "operands", []):
            if (
                hasattr(operand, "name")
                and hasattr(operand, "uuid")
                and operand.name == loop_var_name
                and operand.uuid not in seen
            ):
                seen.add(operand.uuid)
                result[operand.name] = loop_symbol

    for operation in operations:
        check(operation)
        for nested in getattr(operation, "operations", []):
            check(nested)
    return result


def build_while_scope(
    operation: WhileOperation,
    resolver: ExprResolver,
) -> tuple[ExprResolver, sp.Symbol]:
    """Build resolver and symbolic trip count for a while loop.

    Args:
        operation (WhileOperation): While-loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, sp.Symbol]: Child resolver and ``|while|`` symbol.
    """
    child = resolver.child_scope(inner_block=_LocalBlock(operation.operations))
    return child, sp.Symbol("|while|", integer=True, positive=True)


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
    return sp.Symbol(f"|{dict_name}|", integer=True, positive=True)


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


def _classify_gate(
    operation: GateOperation,
    *,
    num_controls: ResourceExpr | int = 0,
) -> GateResources:
    """Classify one primitive gate into logical gate resources.

    Args:
        operation (GateOperation): Primitive gate operation.
        num_controls (ResourceExpr | int): Surrounding controls. Defaults to
            zero.

    Returns:
        GateResources: Resource contribution of the primitive gate.
    """
    gate_name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if gate_name == "ccx":
        gate_name = "toffoli"
    if _expr(num_controls) == 0:
        return _classify_uncontrolled_gate(gate_name)
    return _classify_controlled_gate(gate_name, _expr(num_controls))


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
    return GateResources(
        total=_ONE,
        single_qubit=_ZERO,
        two_qubit=two,
        multi_qubit=multi,
        clifford=clifford,
        rotation=rotation,
        t=_ZERO,
        toffoli=multi,
        non_clifford=_ONE - clifford,
    )


def _qft_iqft_resources(n: ResourceExpr) -> GateResources:
    """Return standard QFT/IQFT logical resources.

    Args:
        n (ResourceExpr): Number of target qubits.

    Returns:
        GateResources: Standard decomposition resources.
    """
    h_count = n
    cp_count = n * (n - 1) / 2
    swap_count = n // 2
    total = h_count + cp_count + swap_count
    return GateResources(
        total=total,
        single_qubit=h_count,
        two_qubit=cp_count + swap_count,
        multi_qubit=_ZERO,
        clifford=h_count + swap_count,
        rotation=cp_count,
        t=_ZERO,
        toffoli=_ZERO,
        non_clifford=cp_count,
    )


def _classify_pauli_evolve(hamiltonian: Any) -> GateResources:
    """Estimate Pauli evolution resources from a concrete Hamiltonian.

    Args:
        hamiltonian (Any): Concrete Qamomile Hamiltonian.

    Returns:
        GateResources: Logical gate resources for Pauli gadget decomposition.
    """
    import qamomile.observable as qm_o

    total_single = _ZERO
    total_two = _ZERO
    total_clifford = _ZERO
    total_rotation = _ZERO
    total = _ZERO
    for operators, coeff in hamiltonian:
        if abs(complex(coeff)) < 1e-15 or not operators:
            continue
        size = len(operators)
        x_count = sum(1 for operator in operators if operator.pauli == qm_o.Pauli.X)
        y_count = sum(1 for operator in operators if operator.pauli == qm_o.Pauli.Y)
        basis_single = sp.Integer(2 * x_count + 4 * y_count)
        rz_count = _ONE
        cx_count = sp.Integer(2 * max(0, size - 1))
        total_single += basis_single + rz_count
        total_two += cx_count
        total_clifford += basis_single + cx_count
        total_rotation += rz_count
        total += basis_single + rz_count + cx_count
    return GateResources(
        total=total,
        single_qubit=total_single,
        two_qubit=total_two,
        multi_qubit=_ZERO,
        clifford=total_clifford,
        rotation=total_rotation,
        t=_ZERO,
        toffoli=_ZERO,
        non_clifford=total - total_clifford,
    )


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
    for key in set(left) | set(right):
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
        clean_ancilla_qubits=sp.Max(
            left.clean_ancilla_qubits,
            right.clean_ancilla_qubits,
        ),
        dirty_ancilla_qubits=sp.Max(
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
        clean_ancilla_qubits=sp.Max(
            left.clean_ancilla_qubits,
            right.clean_ancilla_qubits,
        ),
        dirty_ancilla_qubits=sp.Max(
            left.dirty_ancilla_qubits,
            right.dirty_ancilla_qubits,
        ),
        peak_qubits=sp.Max(left.peak_qubits, right.peak_qubits),
    )


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
    k = sp.Dummy("k", integer=True, nonnegative=True)
    transformed = expr.subs(loop_symbol, start + step * k)
    return cast(ResourceExpr, sp.Sum(transformed, (k, 0, iterations - 1)).doit())


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


def _wrap_trace(
    name: str,
    child: ResourceTraceNode | None,
    *,
    source_kind: str = "algebra",
    strategy: str | None = None,
    estimate_kind: EstimateKind | None = None,
) -> ResourceTraceNode:
    """Wrap an optional child trace with a parent node.

    Args:
        name (str): Parent node name.
        child (ResourceTraceNode | None): Optional child.
        source_kind (str): Parent source kind. Defaults to ``"algebra"``.
        strategy (str | None): Selected strategy. Defaults to ``None``.
        estimate_kind (EstimateKind | None): Estimate kind. Defaults to
            ``None``.

    Returns:
        ResourceTraceNode: Parent trace node.
    """
    children = (child,) if child is not None else ()
    return ResourceTraceNode(
        name=name,
        source_kind=source_kind,
        strategy=strategy,
        estimate_kind=estimate_kind,
        children=children,
    )


def _free_symbols(estimate: ResourceEstimate) -> set[sp.Symbol]:
    """Collect all free symbols from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        set[sp.Symbol]: Free symbols used by any metric.
    """
    symbols: set[sp.Symbol] = set()
    for expr in _all_exprs(estimate):
        symbols.update(cast(set[sp.Symbol], expr.free_symbols))
    return symbols


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
        estimate.depth.depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.depth.measurement_depth,
        *estimate.calls.calls_by_name.values(),
        *estimate.calls.queries_by_name.values(),
    ]


def _collect_parameters(estimate: ResourceEstimate) -> dict[str, sp.Symbol]:
    """Collect symbolic parameters from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        dict[str, sp.Symbol]: Symbol map keyed by printed name.
    """
    return {str(symbol): symbol for symbol in sorted(_free_symbols(estimate), key=str)}


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
    return estimate.substitute(**values)


def _force_symbolic_substitution_params(
    kernel: Any,
    kwargs: Mapping[str, Any],
    parameters: list[str] | None,
    substitutions: Mapping[str, Any] | None,
) -> list[str] | None:
    """Force classical substitution names to stay symbolic when building.

    Names passed via ``substitutions`` that are classical kernel arguments are
    added to the build ``parameters`` list so they are not baked in by a Python
    signature default — otherwise ``substitutions={"n": 2048}`` on a kernel
    where ``n`` has a default would estimate the default and silently ignore the
    request.

    Args:
        kernel (Any): QKernel being built.
        kwargs (Mapping[str, Any]): Compile-time build bindings.
        parameters (list[str] | None): Explicit runtime parameters, or ``None``
            to auto-detect.
        substitutions (Mapping[str, Any] | None): Estimation-only substitution
            names.

    Returns:
        list[str] | None: Parameter list to build with. ``None`` when there are
        no substitutions to force and no explicit parameters (preserving
        auto-detection).
    """
    if not substitutions:
        return parameters
    input_types = getattr(kernel, "input_types", None)
    if not isinstance(input_types, dict):
        return parameters
    from qamomile.circuit.frontend.qkernel_inputs import (
        auto_detect_parameters,
        is_parameterizable_type,
    )

    base = (
        list(parameters)
        if parameters is not None
        else auto_detect_parameters(kernel.signature, input_types, dict(kwargs))
    )
    forced = [
        name
        for name in substitutions
        if name in input_types
        and name not in kwargs
        and is_parameterizable_type(input_types[name])
    ]
    if not forced:
        return parameters
    return sorted(set(base) | set(forced))


def _contract_names(
    block_or_ops: "Block | Sequence[Operation]",
) -> frozenset[str] | None:
    """Return the declared classical argument names of a built block, if any.

    Only classical parameters (``param_slots``) count: ``substitutions`` supplies
    numeric/expression values, so a quantum port name is never a valid
    substitution target and must be rejected as a typo rather than accepted as a
    no-op.

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


def _scalar_values(values: Mapping[str, Any]) -> dict[str, sp.Expr]:
    """Keep only numeric scalar values, for deciding compile-time branches.

    Accepts Python and NumPy numeric scalars (anything registered as
    ``numbers.Real``, normalized via ``.item()`` when present so a ``np.int64``
    from a notebook works) and SymPy numbers. Dicts, Hamiltonians, and
    symbolic-expression substitution values are dropped: a branch condition can
    only be decided by a concrete number, and a non-numeric value simply leaves
    the branch undecidable (conservative ``choice``).

    Args:
        values (Mapping[str, Any]): Merged bindings and substitutions.

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


def _apply_substitutions(
    estimate: ResourceEstimate,
    substitutions: Mapping[str, Any],
    *,
    contract_names: frozenset[str] | None = None,
    branch_condition_names: set[str] | None = None,
) -> ResourceEstimate:
    """Substitute estimation-only values into a symbolic estimate.

    Unlike :func:`_substitute_bindings`, substitution values may be SymPy
    expressions (e.g. an optimal-iteration formula). Each name is classified:

    - a **free symbol** of the estimate is substituted;
    - a name that **participated in a compile-time branch predicate** — whether
      it decided the branch or left it undecidable — is silently accepted; the
      branch specialization or the undecidable-branch assumption already
      accounts for it;
    - any other **declared kernel argument** not appearing in the estimate (e.g.
      a rotation angle) is a no-op, recorded as an assumption so it stays
      auditable;
    - a name that is **none of these** is a typo and raises.

    Args:
        estimate (ResourceEstimate): Symbolic estimate to rewrite.
        substitutions (Mapping[str, Any]): Estimation-only substitutions keyed
            by parameter name. Values may be numbers or SymPy expressions.
        contract_names (frozenset[str] | None): Declared kernel argument names.
            ``None`` (raw op sequence, no contract) keeps every name strict.
        branch_condition_names (set[str] | None): Names that participated in a
            compile-time branch predicate during interpretation. Defaults to
            ``None``.

    Returns:
        ResourceEstimate: Estimate with the substitutions applied.

    Raises:
        ValueError: If a substitution name is neither a free symbol of the
            estimate nor a declared kernel argument.
    """
    # SymPy treats same-named symbols with different assumptions as distinct, so
    # a name can map to more than one symbol object; substitute every match.
    symbols_by_name: dict[str, list[sp.Symbol]] = {}
    for symbol in _free_symbols(estimate):
        symbols_by_name.setdefault(str(symbol), []).append(symbol)
    known = contract_names or frozenset()
    referenced = branch_condition_names or set()
    unknown = [
        name
        for name in substitutions
        if name not in symbols_by_name and name not in known
    ]
    if unknown:
        available = ", ".join(sorted(set(symbols_by_name) | set(known))) or "(none)"
        raise ValueError(
            f"substitutions names {sorted(unknown)} are neither free symbols of "
            f"the estimate nor kernel arguments; available: {available}. Use "
            "bindings for structural build-time values."
        )
    subs: dict[sp.Symbol, sp.Expr] = {}
    ignored: list[str] = []
    for name, value in substitutions.items():
        if name not in symbols_by_name:
            # Not a free symbol: either it participated in a branch predicate
            # (silent — specialization or undecidable-note covers it) or it
            # genuinely affects nothing (recorded as an ignored no-op).
            if name not in referenced:
                ignored.append(name)
            continue
        sympified = sp.sympify(value)
        for symbol in symbols_by_name[name]:
            subs[symbol] = sympified
    # Simultaneous substitution: the only way a requested name survives is that
    # the caller's own value reintroduced it (a legitimate shift like n -> n+1),
    # never a partially-applied replacement.
    substituted = estimate._map_expr(
        lambda expr: cast(sp.Expr, expr.subs(subs, simultaneous=True).doit())
    )
    if ignored:
        note = ResourceAssumption(
            "substitution(s) "
            + ", ".join(repr(name) for name in sorted(ignored))
            + " do not affect any resource metric; ignored",
        )
        substituted = dataclasses.replace(
            substituted, assumptions=(*substituted.assumptions, note)
        )
    return substituted


def _count_input_qubits(
    values: Sequence[Value], resolver: ExprResolver
) -> ResourceExpr:
    """Count quantum input values in a block signature.

    Args:
        values (Sequence[Value]): Block input values.
        resolver (ExprResolver): Resolver for symbolic shapes.

    Returns:
        ResourceExpr: Logical input-qubit count.
    """
    count: ResourceExpr = _ZERO
    for value in values:
        if isinstance(value, ArrayValue) and isinstance(value.type, QubitType):
            element_count: ResourceExpr = _ONE
            for dim in value.shape:
                element_count *= resolver.resolve(dim)
            count += element_count
        elif isinstance(value.type, QubitType):
            count += _ONE
    return count


def _count_qinit(operation: QInitOperation, resolver: ExprResolver) -> ResourceExpr:
    """Count qubits allocated by a qinit operation.

    Args:
        operation (QInitOperation): Qubit initialization operation.
        resolver (ExprResolver): Resolver for symbolic array shapes.

    Returns:
        ResourceExpr: Allocated-qubit count.
    """
    result = operation.results[0]
    if isinstance(result, ArrayValue) and isinstance(result.type, QubitType):
        count: ResourceExpr = _ONE
        for dim in result.shape:
            count *= resolver.resolve(dim)
        return count
    if isinstance(result.type, QubitType):
        return _ONE
    return _ZERO


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


def _is_resource_model(value: Any) -> bool:
    """Return whether an object implements the resource-model protocol.

    Args:
        value (Any): Object to inspect.

    Returns:
        bool: ``True`` when ``value.estimate`` is callable.
    """
    return callable(getattr(value, "estimate", None))


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
    quantum_formals = [value for value in block.input_values if value.type.is_quantum()]
    actuals = [value for value in operation.target_operands if value.type.is_quantum()]
    for formal, actual in zip(quantum_formals, actuals):
        extra[formal.uuid] = resolver.resolve(actual)
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    context = resolver.context
    context.update(extra)
    return ExprResolver(
        block=block,
        context=context,
        loop_var_names=resolver.loop_var_names,
        parent_blocks=[],
    )


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
    quantum_actuals = iter(operation.target_qubits)
    parameter_actuals = iter(operation.parameters)
    for formal in impl.input_values:
        if formal.type.is_quantum():
            actual = next(quantum_actuals, None)
            if actual is None:
                continue
            extra[formal.uuid] = resolver.resolve(actual)
            if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
                for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                    extra[formal_dim.uuid] = resolver.resolve(actual_dim)
            continue
        actual = next(parameter_actuals, None)
        if actual is not None:
            extra[formal.uuid] = resolver.resolve(actual)

    context = resolver.context
    context.update(extra)
    return ExprResolver(
        block=impl,
        context=context,
        loop_var_names=resolver.loop_var_names,
        parent_blocks=[],
    )


def _resolve_qft_n(operation: InvokeOperation, resolver: ExprResolver) -> ResourceExpr:
    """Resolve QFT/IQFT width from invocation operands or attrs.

    Args:
        operation (InvokeOperation): QFT or IQFT invocation.
        resolver (ExprResolver): Call-site resolver.

    Returns:
        ResourceExpr: Number of target qubits.
    """
    targets = operation.target_qubits
    if targets:
        first = targets[0]
        if isinstance(first, ArrayValue) and first.shape:
            return resolver.resolve(first.shape[0])
        parent = getattr(first, "parent_array", None)
        if isinstance(parent, ArrayValue) and parent.shape:
            return resolver.resolve(parent.shape[0])
        return sp.Integer(len(targets))
    raw = operation.attrs.get("num_target_qubits")
    if isinstance(raw, int) and raw > 0:
        return sp.Integer(raw)
    return sp.Symbol("n", integer=True, positive=True)
