"""Estimate logical resources by abstractly interpreting qkernel IR."""

from __future__ import annotations

import dataclasses
import enum
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallTransform,
    InvokeOperation,
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
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.passes.analyze import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
)

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel
    from qamomile.circuit.ir.operation.callable import CallableRef

ResourceExpr = sp.Expr
_ZERO = sp.Integer(0)
_ONE = sp.Integer(1)


def _combine_quality(
    left: EstimateQuality,
    right: EstimateQuality,
) -> EstimateQuality:
    """Return the least exact of two estimate-quality values.

    Args:
        left (EstimateQuality): Left quality.
        right (EstimateQuality): Right quality.

    Returns:
        EstimateQuality: Combined quality classification.
    """
    rank = {
        EstimateQuality.EXACT: 0,
        EstimateQuality.UPPER_BOUND: 1,
        EstimateQuality.MODELED: 2,
    }
    return left if rank[left] >= rank[right] else right


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


class GateBasis(enum.StrEnum):
    """Select the gate basis reported by resource estimation."""

    LOGICAL = "logical"
    CLIFFORD_T = "clifford_t"


class EstimateQuality(enum.StrEnum):
    """Describe how directly an estimate follows executable semantics."""

    EXACT = "exact"
    UPPER_BOUND = "upper_bound"
    MODELED = "modeled"


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
            ``"opaque_cost"``, or ``"opaque"``.
        strategy (str | None): Selected resource strategy. Defaults to
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
        summary = f" {self.summary}" if self.summary else ""
        lines = [f"{prefix}{self.name} [{self.source_kind}{strategy}]{summary}"]
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
        quality (EstimateQuality): Confidence classification. Defaults to
            ``EXACT`` for body-derived logical estimates.
    """

    width: WidthResources = dataclasses.field(default_factory=WidthResources.zero)
    gates: GateResources = dataclasses.field(default_factory=GateResources.zero)
    depth: DepthResources = dataclasses.field(default_factory=DepthResources.zero)
    calls: CallResources = dataclasses.field(default_factory=CallResources.zero)
    assumptions: tuple[ResourceAssumption, ...] = ()
    trace: ResourceTraceNode | None = None
    parameters: dict[str, sp.Symbol] = dataclasses.field(default_factory=dict)
    quality: EstimateQuality = EstimateQuality.EXACT

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
            quality=_combine_quality(self.quality, other.quality),
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
            quality=_combine_quality(self.quality, other.quality),
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
            quality=EstimateQuality.UPPER_BOUND,
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
        return ResourceEstimate(
            width=_conditional_width(self.width, other.width, condition),
            gates=_conditional_gates(self.gates, other.gates, condition),
            depth=_conditional_depth(self.depth, other.depth, condition),
            calls=_conditional_calls(self.calls, other.calls, condition),
            assumptions=(*self.assumptions, *other.assumptions),
            trace=_merge_trace(f"if[{condition}]", self.trace, other.trace),
            quality=_combine_quality(self.quality, other.quality),
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
            quality=self.quality,
        )

    def controlled(self, num_controls: ResourceExpr | int) -> ResourceEstimate:
        """Record a controlled-call context without scaling cost.

        This deliberately does **not** change gate/width/depth counts. Controlled
        cost is owned by whatever knows the decomposition: a body gets exact
        per-primitive reclassification via ``eval_controlled_u``, while an opaque
        cost callable receives its control context directly. This method only
        attaches an assumption for a fixed opaque cost so controls that cost did
        not account for are surfaced rather than silently dropped.

        Args:
            num_controls (ResourceExpr | int): Number of active controls.

        Returns:
            ResourceEstimate: Estimate with a recorded controlled assumption.
        """
        controls = _expr(num_controls)
        assumption = ResourceAssumption(
            message=(
                "controlled transform reuses the selected body/opaque cost; "
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
            quality=self.quality,
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
            "quality": self.quality.value,
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
            quality=self.quality,
        )
        mapped.parameters = _collect_parameters(mapped)
        return mapped


@dataclasses.dataclass(frozen=True)
class OpaqueCallContext:
    """Describe one bodyless callable invocation for cost evaluation.

    A controlled opaque invocation owns the cost of its explicit control.
    ``controls`` counts only surrounding controls that the cost callable did not
    build in; the
    invocation's *own* control is signalled by ``transform`` being
    ``CallTransform.CONTROLLED`` and counted by :attr:`own_controls`.

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
        power (ResourceExpr): Repetition power. Defaults to one.
        transform (CallTransform): Requested call transform.
        strategy (str | None): Selected resource strategy.
        bindings (Mapping[str, Any]): Concrete bindings supplied by the user.
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


@dataclasses.dataclass
class ResourceEstimatorConfig:
    """Configure ``ResourceEstimator`` behavior.

    Args:
        strategies (dict[str, str]): Strategy overrides by callable name.
        trace (bool): Whether estimates should carry trace nodes.
        simplify (bool): Whether to simplify the final estimate.
        unknown_policy (UnknownResourcePolicy): Handling for unknown opaque
            callables.
        basis (GateBasis): Output gate basis. Defaults to ``LOGICAL``.
        precision (float): Approximation precision for rotation synthesis in
            ``CLIFFORD_T`` basis. Defaults to ``1e-10``.
    """

    strategies: dict[str, str] = dataclasses.field(default_factory=dict)
    trace: bool = False
    simplify: bool = True
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR
    basis: GateBasis = GateBasis.LOGICAL
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
        basis: str | GateBasis = GateBasis.LOGICAL,
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
            basis (str | GateBasis): Output gate basis. Defaults to ``LOGICAL``.
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
                problem-sized circuit. Defaults to ``None``.
            strategies (dict[str, str] | None): Per-call override merged over
                estimator-level strategies. Defaults to ``None``.

        Returns:
            ResourceEstimate: Logical resource estimate.

        Raises:
            ValueError: If an input name is neither a free symbol nor a declared
                kernel argument.
        """
        build_inputs, estimation_inputs = _partition_estimation_inputs(kernel, inputs)
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
        if build_inputs:
            estimate = _substitute_bindings(estimate, build_inputs)
        if estimation_inputs:
            estimate = _apply_inputs(
                estimate,
                _expand_array_shape_inputs(block_or_ops, estimation_inputs),
                contract_names=_contract_names(block_or_ops),
                branch_condition_names=interpreter.branch_condition_names,
            )
        if config.simplify:
            estimate = estimate.simplify()
        if not config.trace:
            estimate = dataclasses.replace(estimate, trace=None)
        estimate.parameters = _collect_parameters(estimate)
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
                concrete input values used to decide
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
        self._measurement_derived: set[str] = set()

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
            input_allocations = {
                value.logical_id: _qubit_value_size(value, resolver)
                for value in block_or_ops.input_values
                if value.type.is_quantum()
            }
            body = self.eval_operations(
                block_or_ops.operations,
                resolver,
                initial_allocations=input_allocations,
            )
            input_qubits = sum(input_allocations.values(), _ZERO)
            width = dataclasses.replace(
                body.width,
                input_qubits=input_qubits,
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
        previous_taint = self._measurement_derived
        graph = build_dependency_graph(operations)
        local_taint = find_measurement_derived_values(
            graph,
            find_measurement_results(operations),
        )
        self._measurement_derived = previous_taint | local_taint
        try:
            estimate = ResourceEstimate.zero()
            scheduled: list[tuple[Operation, ResourceEstimate]] = []
            for operation in operations:
                operation_estimate = self.eval_operation(
                    operation,
                    resolver,
                    controls=controls,
                )
                scheduled.append((operation, operation_estimate))
                estimate = estimate.seq(operation_estimate)
            primitive_only = all(
                isinstance(
                    operation,
                    (
                        GateOperation,
                        QInitOperation,
                        MeasureOperation,
                        MeasureVectorOperation,
                        MeasureQFixedOperation,
                        ProjectOperation,
                        ResetOperation,
                    ),
                )
                for operation, _ in scheduled
            )
            return dataclasses.replace(
                estimate,
                depth=(
                    _dependency_depth(scheduled) if primitive_only else estimate.depth
                ),
                width=_liveness_width(scheduled, initial_allocations or {}),
                quality=(
                    estimate.quality
                    if primitive_only
                    else _combine_quality(
                        estimate.quality,
                        EstimateQuality.UPPER_BOUND,
                    )
                ),
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
            case ProjectOperation():
                return self.eval_project(operation)
            case ResetOperation():
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
        if self.config.basis is GateBasis.CLIFFORD_T and gates.rotation == 0:
            return dataclasses.replace(
                estimate,
                quality=(
                    EstimateQuality.UPPER_BOUND
                    if _gate_has_rotation(operation)
                    else EstimateQuality.EXACT
                ),
            )
        return estimate

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

    def eval_project(self, operation: ProjectOperation) -> ResourceEstimate:
        """Evaluate a projective measurement operation.

        Args:
            operation (ProjectOperation): Projection operation.

        Returns:
            ResourceEstimate: Measurement-like resource estimate.
        """
        return ResourceEstimate(
            depth=DepthResources(depth=_ONE, measurement_depth=_ONE),
            trace=ResourceTraceNode(f"project_{operation.axis}", "primitive"),
        )

    def eval_reset(self, operation: ResetOperation) -> ResourceEstimate:
        """Evaluate a reset operation.

        Args:
            operation (ResetOperation): Reset operation.

        Returns:
            ResourceEstimate: Reset primitive resource estimate.
        """
        return ResourceEstimate.primitive(
            "reset",
            GateResources(total=_ONE, single_qubit=_ONE),
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
        if operation.region_args:
            return self._eval_region_for(
                operation,
                resolver,
                start=start,
                stop=stop,
                step=step,
                loop_symbol=loop_symbol,
                controls=controls,
            )
        inner = self.eval_operations(
            operation.operations,
            child,
            controls=controls,
        )
        return inner.sum_over(loop_symbol, start, stop, step)

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
            ResourceEstimate: Sequential composition of every iteration.
        """
        carried = {
            arg.block_arg.uuid: self._apply_condition_values(resolver.resolve(arg.init))
            for arg in operation.region_args
        }
        estimate = ResourceEstimate.zero()
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
            estimate = estimate.seq(
                self.eval_operations(operation.operations, child, controls=controls)
            )
            carried = {
                arg.block_arg.uuid: self._apply_condition_values(
                    child.resolve(arg.yielded)
                )
                for arg in operation.region_args
            }
        for arg in operation.region_args:
            resolver.bind(arg.result, carried[arg.block_arg.uuid])
        return estimate

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
            arg.block_arg.uuid: sp.Symbol(f"{arg.var_name}_carry", integer=True)
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
        self.eval_operations(operation.operations, probe, controls=controls)

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
                final_value = sp.Symbol(f"{arg.var_name}_after_loop", integer=True)
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
        inner = self.eval_operations(operation.operations, child, controls=controls)
        estimate = inner.sum_over(loop_symbol, start, stop, step)
        for arg in operation.region_args:
            resolver.bind(arg.result, final_values[arg.result.uuid])
        if assumptions:
            estimate = dataclasses.replace(
                estimate,
                assumptions=(*estimate.assumptions, *assumptions),
            )
        return estimate

    def _apply_condition_values(self, expression: sp.Expr) -> sp.Expr:
        """Substitute supplied scalar values into an expression.

        Args:
            expression (sp.Expr): Expression to specialize.

        Returns:
            sp.Expr: Specialized expression.
        """
        condition_inputs = {
            symbol: self.condition_values[str(symbol)]
            for symbol in expression.free_symbols
            if str(symbol) in self.condition_values
        }
        if not condition_inputs:
            return expression
        self.branch_condition_names.update(str(symbol) for symbol in condition_inputs)
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
        if taken is not None:
            branch_ops = (
                operation.true_operations if taken else operation.false_operations
            )
            branch_child = true_child if taken else false_child
            estimate = self.eval_operations(branch_ops, branch_child, controls=controls)
            self._publish_if_results(
                operation,
                resolver,
                true_child,
                false_child,
                taken=taken,
            )
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
        self._publish_if_results(
            operation,
            resolver,
            true_child,
            false_child,
            taken=None,
        )
        condition = resolver.resolve(operation.condition)
        if operation.condition.uuid not in self._measurement_derived:
            combined = true_estimate.conditional(
                false_estimate,
                _boolean_condition(condition),
            )
            return combined
        combined = true_estimate.choice(false_estimate)
        if note is None:
            return combined
        trace = combined.trace
        if trace is not None:
            trace = dataclasses.replace(trace, assumptions=(*trace.assumptions, note))
        return dataclasses.replace(
            combined, assumptions=(*combined.assumptions, note), trace=trace
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
        for merge in operation.iter_merges():
            true_value = true_resolver.resolve(merge.true_value)
            false_value = false_resolver.resolve(merge.false_value)
            if taken is True:
                merged = true_value
            elif taken is False:
                merged = false_value
            elif bool(getattr(condition, "is_Boolean", False)):
                merged = sp.Piecewise(
                    (true_value, cast(Any, condition)),
                    (false_value, True),
                )
            else:
                merged = sp.Symbol(merge.result.name, integer=True)
            resolver.bind(merge.result, cast(sp.Expr, merged))

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
        cardinality = resolve_for_items_cardinality(operation)
        if operation.region_args:
            return self._eval_region_for_items(
                operation,
                resolver,
                cardinality=cardinality,
                controls=controls,
            )
        child = build_for_items_scope(operation, resolver)
        inner = self.eval_operations(operation.operations, child, controls=controls)
        return inner.repeat(cardinality)

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
        context = self._symbolic_for_items_context(operation)
        carry_symbols = {
            arg.block_arg.uuid: sp.Symbol(f"{arg.var_name}_carry", integer=True)
            for arg in operation.region_args
        }
        context.update(carry_symbols)
        probe = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context=context,
        )
        self.eval_operations(operation.operations, probe, controls=controls)

        at_iteration: dict[str, sp.Expr] = {}
        final_values: dict[str, sp.Expr] = {}
        assumptions: list[ResourceAssumption] = []
        all_carry_symbols = set(carry_symbols.values())
        for arg in operation.region_args:
            carry_symbol = carry_symbols[arg.block_arg.uuid]
            recurrence = _solve_affine_recurrence(
                yielded=probe.resolve(arg.yielded),
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
                final_value = sp.Symbol(f"{arg.var_name}_after_items", integer=True)
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
        ).repeat(cardinality)
        for arg in operation.region_args:
            resolver.bind(arg.result, final_values[arg.result.uuid])
        if assumptions:
            estimate = dataclasses.replace(
                estimate,
                assumptions=(*estimate.assumptions, *assumptions),
            )
        return estimate

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
            ResourceEstimate: Sequential composition of all item iterations.
        """
        carried = {
            arg.block_arg.uuid: self._apply_condition_values(resolver.resolve(arg.init))
            for arg in operation.region_args
        }
        estimate = ResourceEstimate.zero()
        for key, value in entries:
            context = {
                **carried,
                **self._concrete_for_items_context(operation, key, value),
            }
            child = resolver.child_scope(
                inner_block=_LocalBlock(operation.operations),
                extra_context=context,
            )
            estimate = estimate.seq(
                self.eval_operations(operation.operations, child, controls=controls)
            )
            carried = {
                arg.block_arg.uuid: self._apply_condition_values(
                    child.resolve(arg.yielded)
                )
                for arg in operation.region_args
            }
        for arg in operation.region_args:
            resolver.bind(arg.result, carried[arg.block_arg.uuid])
        return estimate

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
        get_items = getattr(operand, "get_bound_data_items", None)
        if callable(get_items):
            items = tuple(get_items())
            if items:
                return items
        parameter_name = getattr(operand, "parameter_name", lambda: None)()
        bound = self.bindings.get(parameter_name) if parameter_name else None
        if isinstance(bound, Mapping):
            return tuple(bound.items())
        return None

    def _symbolic_for_items_context(
        self,
        operation: ForItemsOperation,
    ) -> dict[str, sp.Expr]:
        """Build symbolic key and value bindings for an items loop.

        Args:
            operation (ForItemsOperation): Items loop to bind.

        Returns:
            dict[str, sp.Expr]: UUID-keyed symbolic loop-variable context.
        """
        context: dict[str, sp.Expr] = {}
        for index, key_value in enumerate(operation.key_var_values or ()):
            name = (
                operation.key_vars[index] if index < len(operation.key_vars) else "key"
            )
            context[key_value.uuid] = sp.Symbol(name, integer=True)
        if operation.value_var_value is not None:
            context[operation.value_var_value.uuid] = sp.Symbol(operation.value_var)
        return context

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
        """
        context: dict[str, sp.Expr] = {}
        key_values = list(operation.key_var_values or ())
        if len(key_values) > 1 and isinstance(key, Sequence):
            for ir_value, concrete in zip(key_values, key, strict=False):
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
        strategy = self._strategy_for(operation)
        body = operation.effective_body(strategy=strategy)
        if isinstance(body, Block):
            return self._estimate_invoke_body(operation, body, resolver, controls)
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
            return self._estimate_opaque_cost(operation, opaque_cost, ctx)
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
        )

    def _estimate_opaque_cost(
        self,
        operation: InvokeOperation,
        cost: Any,
        ctx: OpaqueCallContext,
    ) -> ResourceEstimate:
        """Evaluate an explicit cost attached to a bodyless callable.

        Args:
            operation (InvokeOperation): Invocation operation.
            cost (Any): ``ResourceEstimate`` or callable accepting ``ctx``.
            ctx (OpaqueCallContext): Opaque call context.

        Returns:
            ResourceEstimate: Explicit opaque estimate.

        Raises:
            TypeError: If ``cost`` is neither a ``ResourceEstimate`` nor a
                callable returning one.
        """
        if isinstance(cost, ResourceEstimate):
            estimate = cost.repeat(ctx.power)
        elif callable(cost):
            estimate = cost(ctx)
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
        if ctx.controls != _ZERO:
            estimate = estimate.controlled(ctx.controls)
        return dataclasses.replace(
            estimate,
            quality=EstimateQuality.MODELED,
            trace=_wrap_trace(
                operation.custom_name,
                estimate.trace,
                source_kind="opaque_cost",
                strategy=ctx.strategy,
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
        body_estimate = self.eval_operations(
            body.operations, child, controls=total_controls
        )
        if (
            operation.transform is CallTransform.INVERSE
            and not body_implements_transform
        ):
            body_estimate = body_estimate.inverse()
        return dataclasses.replace(
            body_estimate,
            trace=_wrap_trace(
                operation.custom_name,
                body_estimate.trace,
                source_kind="body",
            ),
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
            return ResourceEstimate(
                calls=CallResources(
                    calls_by_name={name: ctx.power},
                    queries_by_name={name: ctx.power},
                ),
                trace=ResourceTraceNode(name, "opaque", summary=f"power={ctx.power}"),
                quality=EstimateQuality.MODELED,
            )
        if self.config.unknown_policy is UnknownResourcePolicy.ZERO_WITH_WARNING:
            assumption = ResourceAssumption(
                "unknown callable counted as zero resources",
                source=name,
            )
            return ResourceEstimate(
                assumptions=(assumption,),
                trace=ResourceTraceNode(name, "opaque", assumptions=(assumption,)),
                quality=EstimateQuality.MODELED,
            )
        raise ValueError(
            f"Cannot estimate resources for callable '{name}': no body or "
            "opaque cost is available."
        )


def estimate_resources(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
    *,
    inputs: dict[str, Any] | None = None,
    strategies: dict[str, str] | None = None,
    trace: bool = False,
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR,
    basis: str | GateBasis = GateBasis.LOGICAL,
    precision: float = 1e-10,
) -> ResourceEstimate:
    """Estimate logical resources using the default estimator facade.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): QKernel,
            block, or operation sequence to estimate.
        inputs (dict[str, Any] | None): QKernel input values used to specialize
            the symbolic estimate without building a problem-sized circuit.
            Defaults to ``None``.
        strategies (dict[str, str] | None): Strategy overrides by callable
            name. Defaults to ``None``.
        trace (bool): Whether to retain the explanation tree. Defaults to
            ``False``.
        unknown_policy (UnknownResourcePolicy): Unknown callable handling.
            Defaults to ``ERROR``.
        basis (str | GateBasis): Output gate basis. Defaults to ``LOGICAL``.
        precision (float): Rotation-synthesis precision for ``CLIFFORD_T``.
            Defaults to ``1e-10``.

    Returns:
        ResourceEstimate: Logical resource estimate.

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
    loop_var_names: dict[str, sp.Expr] = {
        name: symbol
        for name, symbol in _collect_loop_var_names(
            operation.operations,
            operation.loop_var,
            loop_symbol,
        ).items()
    }
    context: dict[str, sp.Expr] | None = (
        {operation.loop_var_value.uuid: loop_symbol}
        if operation.loop_var_value is not None
        else None
    )
    child = resolver.child_scope(
        inner_block=_LocalBlock(operation.operations),
        extra_context=context,
        extra_loop_vars=loop_var_names,
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


def _classify_clifford_t_gate(
    gate_name: str,
    num_controls: ResourceExpr,
    precision: float,
) -> GateResources:
    """Lower one logical primitive to aggregate Clifford+T resources.

    Exact canonical decompositions are used for SWAP and Toffoli. Arbitrary
    axial rotations use the Ross-Selinger asymptotic upper bound
    ``ceil(3 log2(1 / precision))`` T gates. A controlled phase is first
    decomposed into three axial rotations and two CNOTs.

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
    if gate_name == "swap":
        if num_controls != 0:
            middle = _multi_controlled_x_clifford_t(num_controls + 1)
            return _add_gates(
                middle,
                GateResources(
                    total=sp.Integer(2),
                    two_qubit=sp.Integer(2),
                    clifford=sp.Integer(2),
                ),
            )
        return GateResources(
            total=sp.Integer(3),
            two_qubit=sp.Integer(3),
            clifford=sp.Integer(3),
        )
    rotation_t = sp.Integer(math.ceil(3 * math.log2(1 / precision)))
    rotation_multiplicity = 0
    extra_clifford = 0
    if gate_name in {"rz", "p"}:
        rotation_multiplicity = 1
    elif gate_name == "rx":
        rotation_multiplicity = 1
        extra_clifford = 2
    elif gate_name == "ry":
        rotation_multiplicity = 1
        extra_clifford = 4
    elif gate_name == "cp":
        rotation_multiplicity = 3
        extra_clifford = 2
    elif gate_name == "rzz":
        rotation_multiplicity = 1
        extra_clifford = 2
    if rotation_multiplicity:
        t_count = rotation_multiplicity * rotation_t
        return GateResources(
            total=t_count + extra_clifford,
            single_qubit=t_count,
            two_qubit=sp.Integer(extra_clifford),
            clifford=sp.Integer(extra_clifford),
            t=t_count,
            non_clifford=t_count,
        )
    if num_controls != 0:
        raise ValueError(
            "No canonical Clifford+T lowering is defined for controlled "
            f"gate '{gate_name}' with {num_controls} surrounding control(s)."
        )
    return _classify_uncontrolled_gate(gate_name)


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
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        controls = surrounding_controls + inherent_controls[name]
        return sp.Max(_ZERO, controls - 2)
    if name == "swap" and surrounding_controls != 0:
        return sp.Max(_ZERO, surrounding_controls - 1)
    return _ZERO


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
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        controls = surrounding_controls + inherent_controls[name]
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
        )
    if name == "swap":
        if surrounding_controls == 0:
            return DepthResources(depth=sp.Integer(3), clifford_depth=sp.Integer(3))
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls + 1)
        return dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
        )
    rotation_t = sp.Integer(math.ceil(3 * math.log2(1 / precision)))
    extra_clifford = {
        "rz": 0,
        "p": 0,
        "rx": 2,
        "ry": 4,
        "cp": 2,
        "rzz": 2,
    }.get(name)
    multiplicity = 3 if name == "cp" else 1
    if extra_clifford is not None:
        t_depth = multiplicity * rotation_t
        return DepthResources(
            depth=t_depth + extra_clifford,
            clifford_depth=sp.Integer(extra_clifford),
            t_depth=t_depth,
            non_clifford_depth=t_depth,
        )
    return DepthResources(depth=_ONE, clifford_depth=_ONE)


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


def _boolean_condition(condition: sp.Basic) -> sp.Basic:
    """Normalize a symbolic branch predicate to a SymPy Boolean.

    Args:
        condition (sp.Basic): Boolean or numeric branch expression.

    Returns:
        sp.Basic: Boolean predicate with numeric truth represented as nonzero.
    """
    if isinstance(condition, sp.logic.boolalg.Boolean):
        return condition
    return sp.Ne(condition, 0)


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
            for name in set(true_map) | set(false_map)
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


def _quantum_wire_keys(operation: Operation) -> tuple[set[str], set[str]]:
    """Collect quantum logical wires read and written by one operation.

    Args:
        operation (Operation): Operation whose dependency footprint is needed.

    Returns:
        tuple[set[str], set[str]]: Logical IDs read and written. Nested control
            flow conservatively treats every touched wire as both read and
            written at the enclosing boundary.
    """
    reads = {
        value.logical_id
        for value in operation.all_input_values()
        if isinstance(value, Value) and value.type.is_quantum()
    }
    writes = {
        value.logical_id for value in operation.results if value.type.is_quantum()
    }
    if isinstance(operation, HasNestedOps):
        nested_keys: set[str] = set()
        for body in operation.nested_op_lists():
            for child in body:
                child_reads, child_writes = _quantum_wire_keys(child)
                nested_keys |= child_reads | child_writes
        reads |= nested_keys
        writes |= nested_keys
    return reads, writes


def _dependency_depth(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
) -> DepthResources:
    """Schedule operation summaries by their quantum wire dependencies.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations in
            program order paired with their internally computed summaries.

    Returns:
        DepthResources: Critical-path depth for every tracked gate family.
    """
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    availability: dict[str, dict[str, ResourceExpr]] = {field: {} for field in fields}
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    for operation, estimate in scheduled:
        reads, writes = _quantum_wire_keys(operation)
        touched = reads | writes
        for field in fields:
            wire_depth = availability[field]
            start = (
                sp.Max(*(wire_depth.get(key, _ZERO) for key in reads))
                if reads
                else _ZERO
            )
            duration = cast(ResourceExpr, getattr(estimate.depth, field))
            finish = start + duration
            peaks[field] = sp.Max(peaks[field], finish)
            for key in touched:
                wire_depth[key] = finish
    return DepthResources(**peaks)


def _liveness_width(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    initial_allocations: Mapping[str, ResourceExpr],
) -> WidthResources:
    """Compute peak width from affine allocation and consumption lifetimes.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations and
            their nested width summaries in program order.
        initial_allocations (Mapping[str, ResourceExpr]): Live input wire sizes
            keyed by logical ID.

    Returns:
        WidthResources: Total body allocations and liveness-aware peak width.
    """
    live = dict(initial_allocations)
    current = sum(live.values(), _ZERO)
    peak = current
    allocated = _ZERO
    clean = _ZERO
    dirty = _ZERO
    for operation, estimate in scheduled:
        allocated += estimate.width.allocated_qubits
        clean = sp.Max(clean, estimate.width.clean_ancilla_qubits)
        dirty = sp.Max(dirty, estimate.width.dirty_ancilla_qubits)
        if isinstance(operation, QInitOperation):
            result = operation.results[0]
            amount = estimate.width.allocated_qubits
            live[result.logical_id] = amount
            current += amount
            peak = sp.Max(peak, current)
        else:
            peak = sp.Max(peak, current + estimate.width.peak_qubits)
        if isinstance(
            operation,
            (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation),
        ):
            for value in operation.all_input_values():
                released = live.pop(value.logical_id, _ZERO)
                current -= released
    return WidthResources(
        allocated_qubits=allocated,
        clean_ancilla_qubits=clean,
        dirty_ancilla_qubits=dirty,
        peak_qubits=peak,
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
    transformed = _simplify_sum_range_guards(
        transformed,
        symbol=k,
        lower=_ZERO,
        upper=iterations - 1,
    )
    return cast(ResourceExpr, sp.Sum(transformed, (k, 0, iterations - 1)).doit())


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


def _partition_estimation_inputs(
    kernel: Any,
    inputs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition one user input mapping by its role during estimation.

    Scalar and numeric-array qkernel inputs remain symbolic until after
    interpretation. Structural values such as observables and dictionaries are
    supplied while tracing. Blocks and raw operation lists have no Python input
    annotations, so all supplied values specialize their symbolic estimate.

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
        return {}, values

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

    Only classical parameters (``param_slots``) count: ``inputs`` supplies
    numeric or expression values, so a quantum port name is never a valid input
    target and must be rejected as a typo rather than accepted as a no-op.

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


def _expand_array_shape_inputs(
    block_or_ops: "Block | Sequence[Operation]",
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand supplied array shapes into their IR dimension symbols.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.
        inputs (Mapping[str, Any]): User-provided estimation inputs.

    Returns:
        dict[str, Any]: Inputs plus one concrete value for each matching array
        dimension symbol.
    """
    expanded = dict(inputs)
    if not isinstance(block_or_ops, Block):
        return expanded
    for name, ir_value in zip(block_or_ops.label_args, block_or_ops.input_values):
        if name not in inputs or not isinstance(ir_value, ArrayValue):
            continue
        shape = _concrete_input_shape(inputs[name])
        for dimension, size in zip(ir_value.shape, shape):
            if dimension.name:
                expanded[dimension.name] = size
    return expanded


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
    """Keep only numeric scalar values, for deciding compile-time branches.

    Accepts Python and NumPy numeric scalars (anything registered as
    ``numbers.Real``, normalized via ``.item()`` when present so a ``np.int64``
    from a notebook works) and SymPy numbers. Dicts, Hamiltonians, and
    symbolic-expression substitution values are dropped: a branch condition can
    only be decided by a concrete number, and a non-numeric value simply leaves
    the branch undecidable (conservative ``choice``).

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
    branch_condition_names: set[str] | None = None,
) -> ResourceEstimate:
    """Specialize a symbolic estimate with qkernel input values.

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
        inputs (Mapping[str, Any]): Input values keyed by parameter name. Values
            may be numbers or SymPy expressions.
        contract_names (frozenset[str] | None): Declared kernel argument names.
            ``None`` (raw op sequence, no contract) keeps every name strict.
        branch_condition_names (set[str] | None): Names that participated in a
            compile-time branch predicate during interpretation. Defaults to
            ``None``.

    Returns:
        ResourceEstimate: Estimate with the inputs applied.

    Raises:
        ValueError: If an input name is neither a free symbol of the
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
            "input(s) "
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


def _qubit_value_size(value: Value, resolver: ExprResolver) -> ResourceExpr:
    """Return the logical width represented by one quantum value.

    Args:
        value (Value): Scalar or array quantum value.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        ResourceExpr: Number of represented qubits.
    """
    if isinstance(value, ArrayValue) and isinstance(value.type, QubitType):
        count: ResourceExpr = _ONE
        for dim in value.shape:
            count *= resolver.resolve(dim)
        return count
    if isinstance(value.type, QubitType):
        return _ONE
    return _ZERO


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
    for formal, actual in pair_block_operands(block, operation.target_operands):
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
    operands = [*operation.target_qubits, *operation.parameters]
    for formal, actual in pair_block_operands(impl, operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    context = resolver.context
    context.update(extra)
    return ExprResolver(
        block=impl,
        context=context,
        loop_var_names=resolver.loop_var_names,
        parent_blocks=[],
    )
