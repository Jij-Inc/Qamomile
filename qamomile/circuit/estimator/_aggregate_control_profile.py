"""Validate and profile aggregate costs before coherent control."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._control_decomposition import (
    CLEAN_ANCILLA_BATCH_MIN_WORK,
)
from qamomile.circuit.estimator._resource_base import (
    ControlDecomposition,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _expr,
    _resource_activity_condition,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
    ResourceAssumption,
    WidthResources,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate

from qamomile.circuit.estimator._constants import (
    _ZERO,
)
from qamomile.circuit.estimator._control_model import (
    _EstimatorControlBatchProfile,
)


def _clean_ancilla_aggregate_control_profile(
    estimate: ResourceEstimate,
) -> _EstimatorControlBatchProfile:
    """Return controlled work declared by one aggregate opaque cost.

    The profile is shared by the enclosing-body preflight and the aggregate
    projection itself. This ensures both stages choose the same per-primitive
    or shared-ladder recipe after symbolic specialization.

    Args:
        estimate (ResourceEstimate): Definition-level aggregate cost.

    Returns:
        _EstimatorControlBatchProfile: Total modeled work, capped at the
            sharing threshold, when known one- or two-qubit work makes the
            aggregate projection eligible.
    """
    if _aggregate_arity_projection_reason(estimate) is not None:
        return _EstimatorControlBatchProfile()
    known_arity_work = estimate.gates.single_qubit + estimate.gates.two_qubit
    return _EstimatorControlBatchProfile(
        work=sp.Min(
            CLEAN_ANCILLA_BATCH_MIN_WORK,
            estimate.gates.total,
        )
    ).when(sp.Gt(known_arity_work, _ZERO))


def _aggregate_zero_gate_residual_condition(
    estimate: ResourceEstimate,
) -> Boolean:
    """Return whether non-total transform-sensitive resources are active.

    A symbolic complete gate profile may later specialize to zero. Calls,
    queries, gate-depth declarations, or decomposition workspace can still
    make that zero-gate specialization different from an explicit empty cost.
    Every non-total gate field is included so a symbolic ``gates.total`` that
    later becomes zero cannot erase a separately declared arity or family
    profile before its guarded constraints are checked.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        Boolean: Condition under which a zero total still has declared
            transform-sensitive resources.
    """
    expressions = (
        *estimate.calls.calls_by_name.values(),
        *estimate.calls.queries_by_name.values(),
        *(
            getattr(estimate.gates, field.name)
            for field in dataclasses.fields(GateResources)
            if field.name != "total"
        ),
        estimate.depth.depth,
        estimate.depth.gate_depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
    )
    return sp.Or(*(_resource_activity_condition(_expr(value)) for value in expressions))


def _aggregate_resource_profile_constraints(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[_ResourceConstraint, ...]:
    """Retain the basic domain requirements of an aggregate resource profile.

    Every resource metric is a nonnegative integer. Gate-family fields are
    independent bounds and are not cross-classified here. Arity fields,
    however, partition ``total`` for controlled aggregate projection and
    therefore cannot exceed it.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to validate.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements guarded by a positive
            control count.

    Raises:
        ValueError: If active concrete values violate a requirement.
    """
    requirements: list[_ResourceConstraint] = []
    resources: list[tuple[str, ResourceExpr]] = [
        *(
            (f"gates.{field.name}", getattr(estimate.gates, field.name))
            for field in dataclasses.fields(GateResources)
        ),
        ("measurements.total", estimate.measurements.total),
        ("resets.total", estimate.resets.total),
        *(
            (f"depth.{field.name}", getattr(estimate.depth, field.name))
            for field in dataclasses.fields(DepthResources)
        ),
        *(
            (f"width.{field.name}", getattr(estimate.width, field.name))
            for field in dataclasses.fields(WidthResources)
        ),
        *(
            (f"calls.calls_by_name[{name!r}]", count)
            for name, count in estimate.calls.calls_by_name.items()
        ),
        *(
            (f"calls.queries_by_name[{name!r}]", count)
            for name, count in estimate.calls.queries_by_name.items()
        ),
    ]
    for label, count in resources:
        simplified = _safe_simplify(count)
        if simplified.is_nonnegative is not True or simplified.is_integer is not True:
            requirements.append(
                _ResourceConstraint(
                    expression=count,
                    minimum=0,
                    label=f"Aggregate controlled {label}",
                )
            )
    gates = estimate.gates
    arity_remainder = _safe_simplify(
        gates.total - gates.single_qubit - gates.two_qubit - gates.multi_qubit
    )
    if arity_remainder.is_nonnegative is not True:
        requirements.append(
            _ResourceConstraint(
                expression=arity_remainder,
                minimum=0,
                label="Aggregate controlled unclassified arity remainder",
                unit="gate",
            )
        )
    active_when = sp.Gt(controls, _ZERO)
    guarded = tuple(requirement.when(active_when) for requirement in requirements)
    for requirement in guarded:
        requirement.validate()
    return guarded


def _simplify_aggregate_metadata_guard(condition: sp.Basic) -> Boolean:
    """Simplify a small aggregate-profile metadata condition.

    Args:
        condition (sp.Basic): Boolean condition built from aggregate counts.

    Returns:
        Boolean: Simplified guard with contradictory zero/nonzero predicates
            removed.
    """
    simplified = _safe_simplify(cast(ResourceExpr, condition))
    return _boolean_condition(cast(sp.Basic, simplified))


def _aggregate_arity_profile_reason(
    estimate: ResourceEstimate,
) -> str | None:
    """Return why an aggregate gate profile cannot be transformed safely.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        str | None: Invalid or unavailable profile reason, or ``None`` when
            the declared aggregate fields satisfy the common transform
            contract.
    """
    gates = estimate.gates
    if _safe_simplify(gates.total) == _ZERO:
        return "the zero gate profile has no declared primitive arity to project"
    for label, count in (
        ("total", gates.total),
        ("one-qubit", gates.single_qubit),
        ("two-qubit", gates.two_qubit),
        ("three-or-more-qubit", gates.multi_qubit),
        ("Clifford", gates.clifford),
        ("rotation", gates.rotation),
        ("T", gates.t),
        ("Toffoli", gates.toffoli),
        ("non-Clifford", gates.non_clifford),
    ):
        if _safe_simplify(count).is_negative is True:
            return f"the {label} count is negative"
    arity_remainder = _safe_simplify(
        gates.total - gates.single_qubit - gates.two_qubit - gates.multi_qubit
    )
    if arity_remainder.is_negative is True:
        return "the declared arity gate counts exceed the total gate count"
    for label, count, upper_label, upper in (
        ("Clifford", gates.clifford, "total", gates.total),
        ("rotation", gates.rotation, "total", gates.total),
        (
            "T",
            gates.t,
            "possible one-qubit",
            gates.total - gates.two_qubit - gates.multi_qubit,
        ),
        (
            "Toffoli",
            gates.toffoli,
            "possible three-or-more-qubit",
            gates.total - gates.single_qubit - gates.two_qubit,
        ),
        ("non-Clifford", gates.non_clifford, "total", gates.total),
    ):
        excess = _safe_simplify(count - upper)
        if excess.is_positive is True:
            return f"the {label} count exceeds the declared {upper_label} gate count"
    if any(
        _safe_simplify(expression) != _ZERO
        for expression in (
            estimate.measurements.total,
            estimate.resets.total,
            estimate.depth.measurement_depth,
            estimate.depth.reset_depth,
        )
    ):
        return "controlled opaque costs cannot contain measurement or reset resources"
    return None


def _aggregate_arity_projection_reason(
    estimate: ResourceEstimate,
) -> str | None:
    """Return why clean-ancilla aggregate projection is unavailable.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to inspect.

    Returns:
        str | None: Ineligibility reason, or ``None`` when at least one
            one- or two-qubit gate can use the selected control recipe.
    """
    if estimate.control_decomposition is not ControlDecomposition.CLEAN_ANCILLA_TOFFOLI:
        return (
            "automatic arity projection is available only with the "
            "clean-ancilla Toffoli control decomposition, not "
            f"{estimate.control_decomposition.value}"
        )
    reason = _aggregate_arity_profile_reason(estimate)
    if reason is not None:
        return reason
    gates = estimate.gates
    if _safe_simplify(gates.single_qubit + gates.two_qubit) == _ZERO:
        return "the aggregate has no declared one- or two-qubit gate profile"
    return None


def _unprojected_aggregate_control_assumption(
    reason: str,
    controls: ResourceExpr,
) -> ResourceAssumption:
    """Describe why an aggregate controlled cost remains unchanged.

    Args:
        reason (str): Missing information or invalid profile description.
        controls (ResourceExpr): Number of requested coherent controls.

    Returns:
        ResourceAssumption: User-facing modeled-cost assumption.
    """
    return ResourceAssumption(
        message=(
            "aggregate controlled cost is unchanged because "
            f"{reason}; no primitive body is available under the active "
            "coherent controls"
        )
    )


def _aggregate_arity_projection_constraints(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
    *,
    model_label: str,
) -> tuple[_ResourceConstraint, ...]:
    """Retain unresolved gate-family compatibility requirements.

    Args:
        estimate (ResourceEstimate): Eligible aggregate estimate.
        controls (ResourceExpr): Number of added coherent controls.
        model_label (str): Human-readable projection model used in
            requirement diagnostics.

    Returns:
        tuple[_ResourceConstraint, ...]: Family requirements that become
            active only when one or more controls use the projection.
    """
    gates = estimate.gates
    requirements: list[_ResourceConstraint] = []
    for label, count, upper_label, upper in (
        ("Clifford", gates.clifford, "total", gates.total),
        ("rotation", gates.rotation, "total", gates.total),
        (
            "T",
            gates.t,
            "possible one-qubit",
            gates.total - gates.two_qubit - gates.multi_qubit,
        ),
        (
            "Toffoli",
            gates.toffoli,
            "possible three-or-more-qubit",
            gates.total - gates.single_qubit - gates.two_qubit,
        ),
        ("non-Clifford", gates.non_clifford, "total", gates.total),
    ):
        if _safe_simplify(count) == _ZERO:
            continue
        if _safe_simplify(count - upper).is_nonpositive is not True:
            requirements.append(
                _ResourceConstraint(
                    expression=upper - count,
                    minimum=0,
                    label=(
                        f"{model_label} aggregate {label} count within "
                        f"{upper_label} gate count"
                    ),
                    unit="gate",
                )
            )
    active_when = sp.Gt(controls, _ZERO)
    guarded = tuple(requirement.when(active_when) for requirement in requirements)
    for requirement in guarded:
        requirement.validate()
    return guarded
