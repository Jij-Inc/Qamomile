"""Bound relations, evidence, assumptions, and sound resource arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

from ._validation import validate_resource_value
from .expr import Expr, Number, literal, resolve_substitutions
from .metrics import MetricDefinition


class Relation(str, Enum):
    """A mathematical relation between a metric and symbolic expressions."""

    EQUALITY = "equality"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    ENCLOSURE = "enclosure"
    PARTIAL_ENCLOSURE = "partial_enclosure"
    UNKNOWN = "unknown"


class EvidenceAuthority(str, Enum):
    """The authority supporting a relation; testing alone is never proof."""

    STRUCTURALLY_DERIVED = "structurally_derived"
    TRUSTED_AXIOM = "trusted_axiom"
    HEURISTIC_MODEL = "heuristic_model"


_EVIDENCE_MINT = object()


@dataclass(frozen=True, slots=True, init=False)
class Evidence:
    """One structured justification claim for a resource relation.

    Ordinary construction records a claim but does not grant compiler-proof
    authority.  Only a checker in this package can mint the process-local
    verification bit, and wire decoding intentionally drops that bit because
    an integrity digest is not authentication or a proof certificate.
    """

    authority: EvidenceAuthority
    reference: str
    details: tuple[tuple[str, str], ...] = ()
    _verified: bool = field(default=False, repr=False, compare=False)

    def __init__(
        self,
        authority: EvidenceAuthority,
        reference: str,
        details: tuple[tuple[str, str], ...] = (),
        *,
        _mint: object | None = None,
    ) -> None:
        """Create an unverified claim, or an internally minted checked record."""

        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "_verified", _mint is _EVIDENCE_MINT)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Require a stable evidence reference."""

        if not self.reference:
            raise ValueError("resource evidence requires a stable reference")

    @property
    def verified(self) -> bool:
        """Return whether a local checker minted this evidence record."""

        return self._verified


def _verified_evidence(
    authority: EvidenceAuthority,
    reference: str,
    details: tuple[tuple[str, str], ...] = (),
) -> Evidence:
    """Mint evidence after an in-process structural or certificate check."""

    if authority is not EvidenceAuthority.STRUCTURALLY_DERIVED:
        raise ValueError("only structural evidence may be checker-minted")
    return Evidence(
        authority,
        reference,
        details,
        _mint=_EVIDENCE_MINT,
    )


@dataclass(frozen=True, slots=True)
class ResourceValue:
    """One metric value with relation, evidence, and explicit unknowns."""

    metric: MetricDefinition
    relation: Relation
    expression: Expr | None = None
    lower: Expr | None = None
    upper: Expr | None = None
    unknown_contributors: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate relation shape, units, and nonempty unknown explanations."""

        if not isinstance(self.metric, MetricDefinition):
            raise TypeError("resource value metric must be a MetricDefinition")
        if not isinstance(self.relation, Relation):
            raise TypeError("resource value relation must be a Relation")
        if any(
            not isinstance(reason, str) or not reason
            for reason in self.unknown_contributors
        ):
            raise ValueError("resource unknown contributors must be nonempty strings")

        validate_resource_value(
            self.metric,
            self.relation,
            self.expression,
            self.lower,
            self.upper,
            self.unknown_contributors,
        )

    @classmethod
    def equality(
        cls,
        metric: MetricDefinition,
        expression: Expr | Number,
        *,
        evidence: tuple[Evidence, ...] = (),
        assumptions: tuple[str, ...] = (),
    ) -> ResourceValue:
        """Construct an exact or explicitly model-conditional equality."""

        return cls(
            metric,
            Relation.EQUALITY,
            expression=_coerce(metric, expression),
            evidence=evidence,
            assumptions=assumptions,
        )

    @classmethod
    def enclosure(
        cls,
        metric: MetricDefinition,
        lower: Expr | Number,
        upper: Expr | Number,
        *,
        evidence: tuple[Evidence, ...] = (),
        assumptions: tuple[str, ...] = (),
    ) -> ResourceValue:
        """Construct a two-sided pointwise enclosure."""

        return cls(
            metric,
            Relation.ENCLOSURE,
            lower=_coerce(metric, lower),
            upper=_coerce(metric, upper),
            evidence=evidence,
            assumptions=assumptions,
        )

    @classmethod
    def lower_bound(
        cls,
        metric: MetricDefinition,
        expression: Expr | Number,
        *,
        evidence: tuple[Evidence, ...] = (),
        assumptions: tuple[str, ...] = (),
    ) -> ResourceValue:
        """Construct a safe pointwise lower bound."""

        return cls(
            metric,
            Relation.LOWER_BOUND,
            expression=_coerce(metric, expression),
            evidence=evidence,
            assumptions=assumptions,
        )

    @classmethod
    def upper_bound(
        cls,
        metric: MetricDefinition,
        expression: Expr | Number,
        *,
        evidence: tuple[Evidence, ...] = (),
        assumptions: tuple[str, ...] = (),
    ) -> ResourceValue:
        """Construct a safe pointwise upper bound."""

        return cls(
            metric,
            Relation.UPPER_BOUND,
            expression=_coerce(metric, expression),
            evidence=evidence,
            assumptions=assumptions,
        )

    @classmethod
    def partial(
        cls,
        metric: MetricDefinition,
        *,
        lower: Expr | Number | None = None,
        upper: Expr | Number | None = None,
        unknown: tuple[str, ...],
        evidence: tuple[Evidence, ...] = (),
        assumptions: tuple[str, ...] = (),
    ) -> ResourceValue:
        """Preserve safe endpoints while naming unresolved contributors."""

        return cls(
            metric,
            Relation.PARTIAL_ENCLOSURE,
            lower=None if lower is None else _coerce(metric, lower),
            upper=None if upper is None else _coerce(metric, upper),
            unknown_contributors=unknown,
            evidence=evidence,
            assumptions=assumptions,
        )

    @classmethod
    def unknown(
        cls,
        metric: MetricDefinition,
        reason: str,
        *,
        assumptions: tuple[str, ...] = (),
    ) -> ResourceValue:
        """Construct a value with no safe useful finite enclosure."""

        return cls(
            metric,
            Relation.UNKNOWN,
            unknown_contributors=(reason,),
            assumptions=assumptions,
        )

    def evaluate(self, values: Mapping[str, Number]) -> ResourceValue:
        """Substitute symbols without claiming to have re-run program analysis."""

        expressions = tuple(
            expression
            for expression in (self.expression, self.lower, self.upper)
            if expression is not None
        )
        return self._substitute(resolve_substitutions(expressions, values))

    def _substitute(
        self,
        values: Mapping[str, Expr | Number],
    ) -> ResourceValue:
        """Apply already-resolved qualified substitutions to every endpoint."""

        return replace(
            self,
            expression=(
                self.expression.substitute(values) if self.expression else None
            ),
            lower=self.lower.substitute(values) if self.lower else None,
            upper=self.upper.substitute(values) if self.upper else None,
        )

    def finite_enclosure(self) -> tuple[Expr | None, Expr | None]:
        """Return safe finite endpoints represented by this relation."""

        if self.relation is Relation.EQUALITY:
            return self.expression, self.expression
        if self.relation is Relation.LOWER_BOUND:
            return self.expression, None
        if self.relation is Relation.UPPER_BOUND:
            return None, self.expression
        if self.relation in {Relation.ENCLOSURE, Relation.PARTIAL_ENCLOSURE}:
            return self.lower, self.upper
        return None, None

    def numeric_enclosure(self) -> tuple[Number | None, Number | None]:
        """Return numeric endpoints when evaluation has made them literal."""

        lower, upper = self.finite_enclosure()
        return (
            lower.numeric() if lower is not None else None,
            upper.numeric() if upper is not None else None,
        )

    def __str__(self) -> str:
        """Render the mathematical relation without the dataclass internals.

        Returns:
            str: A compact relation, nominal unit, and any unresolved
                contributors.
        """

        unit = self.metric.unit.name
        if self.relation is Relation.EQUALITY:
            rendered = f"= {self.expression} {unit}"
        elif self.relation is Relation.LOWER_BOUND:
            rendered = f">= {self.expression} {unit}"
        elif self.relation is Relation.UPPER_BOUND:
            rendered = f"<= {self.expression} {unit}"
        elif self.relation in {Relation.ENCLOSURE, Relation.PARTIAL_ENCLOSURE}:
            lower = "?" if self.lower is None else str(self.lower)
            upper = "?" if self.upper is None else str(self.upper)
            rendered = f"in [{lower}, {upper}] {unit}"
        else:
            rendered = f"unknown {unit}"
        if self.unknown_contributors:
            rendered += "; unknown: " + ", ".join(self.unknown_contributors)
        return rendered


def _coerce(metric: MetricDefinition, value: Expr | Number) -> Expr:
    """Coerce one expression to a metric's nominal unit and domain."""

    if isinstance(value, Expr):
        return value
    return literal(value, metric.unit, metric.domain)


__all__ = [
    "Evidence",
    "EvidenceAuthority",
    "Relation",
    "ResourceValue",
]
