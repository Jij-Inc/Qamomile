"""Native mathematical validation for resource facts and rules."""

from __future__ import annotations

from .expr import ONE, Expr, Unit
from .metrics import MetricDefinition


def validate_resource_rule(
    metric: MetricDefinition,
    relation: object,
    expression: Expr,
) -> None:
    """Validate one declarative resource rule through the native relation.

    Args:
        metric (MetricDefinition): Metric constrained by the rule.
        relation (object): String-valued public resource relation enum.
        expression (Expr): Dimensionless rule expression.

    Raises:
        RuntimeError: If the required native validation API is unavailable.
        TypeError: If the expression or native result violates the bridge ABI.
        ValueError: If the rule expression violates its metric contract.
    """

    _validate_native_resource_value(
        metric,
        relation,
        expression,
        None,
        None,
        (),
        unit=ONE,
    )


def validate_resource_value(
    metric: MetricDefinition,
    relation: object,
    expression: Expr | None,
    lower: Expr | None,
    upper: Expr | None,
    unknown_contributors: tuple[str, ...],
) -> None:
    """Validate a complete resource relation through the native compiler.

    Args:
        metric (MetricDefinition): Metric constrained by the relation.
        relation (object): String-valued public resource relation enum.
        expression (Expr | None): Optional point-relation expression.
        lower (Expr | None): Optional enclosure lower endpoint.
        upper (Expr | None): Optional enclosure upper endpoint.
        unknown_contributors (tuple[str, ...]): Explicit unresolved contributors.

    Raises:
        RuntimeError: If the required native validation API is unavailable.
        TypeError: If an endpoint or native result violates the bridge ABI.
        ValueError: If the relation shape or mathematical value is invalid.
    """

    _validate_native_resource_value(
        metric,
        relation,
        expression,
        lower,
        upper,
        unknown_contributors,
        unit=metric.unit,
    )


def _validate_native_resource_value(
    metric: MetricDefinition,
    relation: object,
    expression: Expr | None,
    lower: Expr | None,
    upper: Expr | None,
    unknown_contributors: tuple[str, ...],
    *,
    unit: Unit,
) -> None:
    """Encode and invoke the single native resource-relation validator."""

    roots = (
        ("resource expression", expression),
        ("resource lower endpoint", lower),
        ("resource upper endpoint", upper),
    )
    for context, root in roots:
        if root is not None and not isinstance(root, Expr):
            raise TypeError(f"{context} must be a resource expression")
    from qamomile._native import compiler

    from ._rule_wire import encode_resource_expression_graph

    validate = getattr(compiler, "validate_resource_value", None)
    if not callable(validate):
        raise RuntimeError("Q4242: native resource-value validation is unavailable")
    relation_name = getattr(relation, "value", None)
    if not isinstance(relation_name, str):
        raise TypeError("resource value relation must be a Relation")
    payloads = tuple(
        None if root is None else encode_resource_expression_graph(root)
        for _, root in roots
    )
    result = validate(
        metric.domain.value,
        unit.name,
        relation_name,
        payloads[0],
        payloads[1],
        payloads[2],
        len(unknown_contributors),
    )
    if result is not None:
        raise TypeError(
            "Q4242: native resource-value validation returned malformed data"
        )


__all__: list[str] = []
