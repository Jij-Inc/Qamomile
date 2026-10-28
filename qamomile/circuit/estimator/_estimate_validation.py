"""Validate aggregate resource-estimate contracts."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_constraints import _ResourceConstraint
from qamomile.circuit.estimator._resource_expressions import _safe_simplify

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _require_unitary_resource_estimate(
    estimate: ResourceEstimate,
    *,
    transform: str,
) -> None:
    """Reject an aggregate cost that declares non-unitary resources.

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
            f"resources ({fields}). Move measurement and reset outside the "
            "coherent transform, or describe a unitary base implementation."
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
