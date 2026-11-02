"""Explicit symbolic physical-resource models built on logical studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .expr import (
    PHYSICAL_QUBIT_SECONDS,
    PHYSICAL_QUBITS,
    SECONDS,
    Domain,
    Expr,
    Number,
    Symbol,
    Unit,
    as_expr,
    ceiling,
)
from .metrics import (
    LOGICAL_NON_CLIFFORD_GATES,
    LOGICAL_PEAK_QUBITS,
    PHYSICAL_CODE_DISTANCE,
    PHYSICAL_DURATION,
    PHYSICAL_QUBIT_COUNT,
    PHYSICAL_QUBIT_TIME,
    MetricDefinition,
)
from .projection import project
from .report import ResourceStudy
from .values import Evidence, EvidenceAuthority, Relation, ResourceValue


@dataclass(frozen=True, slots=True)
class _SurfaceCodeModel:
    """Internal implementation of the public surface-code projection function."""

    physical_error_rate: float
    threshold: float
    alpha: float
    syndrome_cycle_seconds: float
    logical_qubits: Expr | Number | None
    non_clifford_gates: Expr | Number | None

    @property
    def identity(self) -> str:
        """Return the stable model identity recorded in the projected study."""

        return "qamomile.models.surface-code-toy.v1"

    def project(
        self,
        study: ResourceStudy,
    ) -> Mapping[str, ResourceValue]:
        """Project logical width and non-Clifford work into physical coordinates."""

        logical = _input_value(
            study,
            LOGICAL_PEAK_QUBITS,
            self.logical_qubits,
        )
        non_clifford = _input_value(
            study,
            LOGICAL_NON_CLIFFORD_GATES,
            self.non_clifford_gates,
        )
        logical_lower, logical_upper = logical.finite_enclosure()
        work_lower, work_upper = non_clifford.finite_enclosure()
        assumptions = (
            "toy surface-code model; no routing, factories, or decoder latency",
            "each logical non-Clifford-family operation consumes one magic state",
        )
        evidence = (
            Evidence(
                EvidenceAuthority.HEURISTIC_MODEL,
                self.identity,
                details=(
                    ("physical_error_rate", repr(self.physical_error_rate)),
                    ("threshold", repr(self.threshold)),
                    ("alpha", repr(self.alpha)),
                    (
                        "syndrome_cycle_seconds",
                        repr(self.syndrome_cycle_seconds),
                    ),
                ),
            ),
        )
        if (
            logical_lower is None
            or logical_upper is None
            or work_lower is None
            or work_upper is None
        ):
            reason = "surface-code inputs have no finite logical enclosure"
            return {
                metric.identifier: ResourceValue.unknown(
                    metric,
                    reason,
                    assumptions=assumptions,
                )
                for metric in (
                    PHYSICAL_CODE_DISTANCE,
                    PHYSICAL_QUBIT_COUNT,
                    PHYSICAL_DURATION,
                    PHYSICAL_QUBIT_TIME,
                )
            }
        lower = _surface_code_point(
            logical_lower,
            work_lower,
            alpha=self.alpha,
            threshold=self.threshold,
            physical_error_rate=self.physical_error_rate,
            syndrome_cycle_seconds=self.syndrome_cycle_seconds,
        )
        upper = (
            lower
            if logical_upper == logical_lower and work_upper == work_lower
            else _surface_code_point(
                logical_upper,
                work_upper,
                alpha=self.alpha,
                threshold=self.threshold,
                physical_error_rate=self.physical_error_rate,
                syndrome_cycle_seconds=self.syndrome_cycle_seconds,
            )
        )
        exact = (
            logical.relation is Relation.EQUALITY
            and non_clifford.relation is Relation.EQUALITY
        )
        return {
            metric.identifier: (
                ResourceValue.equality(
                    metric,
                    lower[index],
                    evidence=evidence,
                    assumptions=assumptions,
                )
                if exact
                else ResourceValue.enclosure(
                    metric,
                    lower[index],
                    upper[index],
                    evidence=evidence,
                    assumptions=assumptions,
                )
            )
            for index, metric in enumerate(
                (
                    PHYSICAL_CODE_DISTANCE,
                    PHYSICAL_QUBIT_COUNT,
                    PHYSICAL_DURATION,
                    PHYSICAL_QUBIT_TIME,
                )
            )
        }


def surface_code(
    study: ResourceStudy,
    *,
    logical_qubits: Expr | Number | None = None,
    non_clifford_gates: Expr | Number | None = None,
    physical_error_rate: float = 1e-3,
    threshold: float = 1e-2,
    alpha: float = 0.05,
    syndrome_cycle_seconds: float = 1e-6,
) -> ResourceStudy:
    """Project a logical study through an explicit toy surface-code model."""

    normalized = {
        name: _positive_finite(value, name=name)
        for name, value in {
            "physical_error_rate": physical_error_rate,
            "threshold": threshold,
            "alpha": alpha,
            "syndrome_cycle_seconds": syndrome_cycle_seconds,
        }.items()
    }
    if normalized["threshold"] <= normalized["physical_error_rate"]:
        raise ValueError("surface_code requires threshold > physical_error_rate")
    model = _SurfaceCodeModel(
        normalized["physical_error_rate"],
        normalized["threshold"],
        normalized["alpha"],
        normalized["syndrome_cycle_seconds"],
        logical_qubits,
        non_clifford_gates,
    )
    return project(study, model)


def _input_value(
    study: ResourceStudy,
    metric: MetricDefinition,
    override: Expr | Number | None,
) -> ResourceValue:
    """Read one logical input or form an explicit model override."""

    if override is None:
        try:
            return study.metric(metric)
        except KeyError:
            return ResourceValue.unknown(metric, f"missing {metric.identifier}")
    expression = as_expr(override, unit=metric.unit)
    if expression.unit != metric.unit:
        if not expression.unit.dimensionless:
            raise ValueError(f"{metric.identifier} override has the wrong unit")
        expression = _convert(expression, metric.unit, metric.domain)
    numeric = expression.numeric()
    if numeric is not None and numeric < 0:
        raise ValueError(f"{metric.identifier} override must be nonnegative")
    if numeric is not None and metric.domain is Domain.NATURAL:
        # Continuous asymptotic estimates such as ``0.3 * n**3`` describe a
        # provisioning count. Round upward at the model boundary so the
        # resulting logical input is a sound integer upper realization.
        expression = _convert(ceiling(expression), metric.unit, Domain.NATURAL)
    return ResourceValue.equality(metric, expression)


def _surface_code_point(
    logical_qubits: Expr,
    non_clifford_gates: Expr,
    *,
    alpha: float,
    threshold: float,
    physical_error_rate: float,
    syndrome_cycle_seconds: float,
) -> tuple[Expr, Expr, Expr, Expr]:
    """Return code distance, qubits, duration, and spacetime at one endpoint."""

    from qamomile._native import compiler

    from ._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    project_surface_code = getattr(compiler, "project_surface_code_resources", None)
    if not callable(project_surface_code):
        raise RuntimeError(
            "Q4239: the required native surface-code resource projection is unavailable"
        )
    projection = project_surface_code(
        encode_resource_expression_graph(logical_qubits),
        encode_resource_expression_graph(non_clifford_gates),
        alpha,
        threshold,
        physical_error_rate,
        syndrome_cycle_seconds,
    )
    if (
        not isinstance(projection, tuple)
        or len(projection) != 4
        or any(not isinstance(payload, bytes) for payload in projection)
    ):
        raise TypeError("Q4239: native surface-code projection violated the ABI")
    decoded = tuple(decode_resource_expression_graph(payload) for payload in projection)
    if any(not isinstance(expression, Expr) for expression in decoded):
        raise TypeError("Q4239: native surface-code projection returned a predicate")
    expected = (
        (Domain.NATURAL, Unit("1")),
        (Domain.NATURAL, PHYSICAL_QUBITS),
        (Domain.REAL, SECONDS),
        (Domain.REAL, PHYSICAL_QUBIT_SECONDS),
    )
    for expression, (domain, unit) in zip(decoded, expected, strict=True):
        if not isinstance(expression, Expr) or expression.domain is not domain:
            raise TypeError("Q4239: native surface-code projection changed a domain")
        if expression.unit != unit:
            raise TypeError("Q4239: native surface-code projection changed a unit")
    first, second, third, fourth = decoded
    assert isinstance(first, Expr)
    assert isinstance(second, Expr)
    assert isinstance(third, Expr)
    assert isinstance(fourth, Expr)
    return first, second, third, fourth


def _convert(expression: Expr, unit: Unit, domain: Domain) -> Expr:
    """Apply one explicit model-coordinate conversion to an expression root."""

    declaration = expression.symbol
    return Expr(
        expression.kind,
        domain,
        unit,
        expression.arguments,
        expression.value,
        (
            Symbol(
                declaration.scope,
                declaration.name,
                domain,
                unit,
                declaration.origin,
                declaration.constraints,
            )
            if declaration is not None
            else None
        ),
        expression.attributes,
        expression.predicates,
    )


def _positive_finite(value: object, *, name: str) -> float:
    """Normalize one positive finite model coefficient."""

    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"surface_code requires {name} to be a real scalar")
    try:
        normalized = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"surface_code requires {name} to be a real scalar") from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"surface_code requires {name} to be positive and finite")
    return normalized


__all__ = ["surface_code"]
