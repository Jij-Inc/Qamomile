"""Portable declarative resource rules for opaque semantic callables."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._validation import validate_resource_rule
from .expr import ONE, Domain, Expr, Symbol, scoped_symbol
from .metrics import MetricDefinition
from .values import EvidenceAuthority, Relation


class TransformMode(str, Enum):
    """Callable transformations for which a resource rule is valid."""

    DIRECT = "direct"
    CONTROLLED = "controlled"
    ADJOINT = "adjoint"
    CONTROLLED_ADJOINT = "controlled_adjoint"


@dataclass(frozen=True, slots=True)
class SchedulingFootprint:
    """Whole quantum parameters occupied by one opaque callable invocation."""

    parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require a nonempty ordered set of source parameter names."""

        if (
            not self.parameters
            or any(not isinstance(name, str) or not name for name in self.parameters)
            or len(self.parameters) != len(set(self.parameters))
        ):
            raise ValueError(
                "scheduling footprints require distinct nonempty parameter names"
            )


@dataclass(frozen=True, slots=True)
class ResourceRule:
    """A declarative parametric metric rule without executable callbacks."""

    identity: str
    callable: object
    metric: MetricDefinition
    relation: Relation
    expression: Expr
    modes: frozenset[TransformMode] = frozenset({TransformMode.DIRECT})
    assumptions: tuple[str, ...] = ()
    validity_domain: tuple[str, ...] = ()
    authority: EvidenceAuthority = EvidenceAuthority.TRUSTED_AXIOM
    provenance: str = ""
    scheduling_footprint: SchedulingFootprint | None = None

    def __post_init__(self) -> None:
        """Require persistent identities and a dimensionless cost formula."""

        callable_name = _callable_name(self.callable)
        object.__setattr__(self, "callable", callable_name)
        if not self.identity or not callable_name or not self.provenance:
            raise ValueError(
                "resource rules require identity, callable, and provenance"
            )
        validate_resource_rule(
            self.metric,
            self.relation,
            self.expression,
        )
        if self.relation not in {
            Relation.EQUALITY,
            Relation.LOWER_BOUND,
            Relation.UPPER_BOUND,
        }:
            raise ValueError(
                "resource rules support equality, lower-bound, or upper-bound relations"
            )
        if not self.modes:
            raise ValueError("resource rules require at least one transform mode")


def rule_parameter(
    callable_value: object,
    name: str,
    *,
    domain: Domain = Domain.INTEGER,
    constraints: tuple[str, ...] = (),
) -> Expr:
    """Return a symbol scoped exactly like a callable's source parameter."""

    return scoped_symbol(
        Symbol(
            _callable_name(callable_value),
            name,
            domain,
            ONE,
            "callable_parameter",
            constraints,
        )
    )


def _callable_name(value: object) -> str:
    """Normalize a QKernel, QCore Function, or qualified string identity."""

    if isinstance(value, str):
        name = value
    else:
        name = getattr(value, "symbol", None)
    if not isinstance(name, str) or not name or "." not in name:
        raise TypeError(
            "resource-rule callable must be a QKernel, QCore Function, "
            "or qualified symbol"
        )
    return name


__all__ = [
    "ResourceRule",
    "SchedulingFootprint",
    "TransformMode",
    "rule_parameter",
]
