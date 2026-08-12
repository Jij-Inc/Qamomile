"""Project aggregate costs through coherent-control models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy as sp

from qamomile.circuit.estimator._resource_algebra import (
    _max_depth,
    _merge_trace,
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _piecewise,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate

from qamomile.circuit.estimator._aggregate_control_profile import (
    _aggregate_arity_profile_reason,
    _aggregate_arity_projection_reason,
    _aggregate_zero_gate_residual_condition,
    _clean_ancilla_aggregate_control_profile,
    _simplify_aggregate_metadata_guard,
    _unprojected_aggregate_control_assumption,
)
from qamomile.circuit.estimator._clean_ancilla_projection import (
    _clean_ancilla_primitive_estimate,
    _estimate_clean_ancilla_named_gate,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._control_model import (
    _clean_ancilla_shared_ladder_condition,
)
from qamomile.circuit.estimator._gate_catalog import (
    _SINGLE_QUBIT_GATES,
    _TWO_QUBIT_GATES,
)


def _clean_ancilla_controlled_arity_envelope(
    empty: ResourceEstimate,
    num_qubits: int,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Bound one unknown primitive from its operand arity.

    Every resource field is maximized independently over the supported gate
    kinds of the requested arity. The resulting fields need not describe one
    common gate decomposition, but each is a valid upper bound when the gate
    name is unavailable.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        num_qubits (int): Primitive operand arity. Supported values are one
            and two.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        ResourceEstimate: Field-wise upper bound for one primitive.

    Raises:
        ValueError: If ``num_qubits`` is not one or two.
    """
    if num_qubits == 1:
        gate_names = sorted(_SINGLE_QUBIT_GATES)
    elif num_qubits == 2:
        gate_names = sorted(_TWO_QUBIT_GATES)
    else:
        raise ValueError(
            "Clean-ancilla aggregate control projection supports only one- and "
            f"two-qubit primitives; got arity {num_qubits}."
        )

    envelope = _estimate_clean_ancilla_named_gate(empty, gate_names[0], controls)
    for gate_name in gate_names[1:]:
        envelope = envelope.choice(
            _estimate_clean_ancilla_named_gate(empty, gate_name, controls)
        )
    return dataclasses.replace(
        envelope,
        trace=ResourceTraceNode(
            name=f"unknown_{num_qubits}q",
            source_kind="clean_ancilla_arity_upper_bound",
            summary=f"controls={controls}",
            children=(envelope.trace,) if envelope.trace is not None else (),
        ),
    )


def _project_abstract_aggregate_controlled_cost(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[ResourceEstimate | None, str]:
    """Project an aggregate profile to abstract controlled primitives.

    Each source primitive remains one logical operation. Declared arity
    buckets shift by the added control count, while an undeclared arity
    remainder stays unclassified. Because aggregate costs omit gate names,
    gate-family fields use independent upper bounds over Qamomile's supported
    logical primitive families. Every projected primitive shares the coherent
    controls, so its aggregate gate depth is serial.

    Args:
        estimate (ResourceEstimate): Aggregate logical cost before adding
            coherent controls.
        controls (ResourceExpr): Positive or symbolically nonnegative number
            of added coherent controls.

    Returns:
        tuple[ResourceEstimate | None, str]: Projected abstract estimate and an
            empty reason, or ``None`` with the reason the profile cannot be
            transformed safely.
    """
    if estimate.control_decomposition is not ControlDecomposition.ABSTRACT:
        return (
            None,
            "abstract aggregate projection requires the abstract control "
            f"decomposition, not {estimate.control_decomposition.value}",
        )
    reason = _aggregate_arity_profile_reason(estimate)
    if reason is not None:
        return None, reason

    gates = estimate.gates
    known_arity_count = _safe_simplify(
        gates.single_qubit + gates.two_qubit + gates.multi_qubit
    )
    unclassified_count = _safe_simplify(gates.total - known_arity_count)
    one_control = sp.Eq(controls, _ONE)
    two_controls = sp.Eq(controls, sp.Integer(2))

    projected_two_qubit = _piecewise(
        gates.single_qubit,
        _ZERO,
        one_control,
    )
    projected_multi_qubit = _safe_simplify(known_arity_count - projected_two_qubit)

    # Aggregate profiles do not identify X/Y/Z, CX, or parametric gate names.
    # Bound every family independently from the largest compatible arity
    # bucket instead of pretending the fields describe one concrete gate mix.
    possible_single_qubit = _safe_simplify(
        gates.total - gates.two_qubit - gates.multi_qubit
    )
    possible_two_qubit = _safe_simplify(
        gates.total - gates.single_qubit - gates.multi_qubit
    )
    possible_one_or_two_qubit = _safe_simplify(gates.total - gates.multi_qubit)
    projected_clifford = _piecewise(
        possible_single_qubit,
        _ZERO,
        one_control,
    )
    projected_toffoli = _piecewise(
        possible_two_qubit,
        _piecewise(
            possible_single_qubit,
            _ZERO,
            two_controls,
        ),
        one_control,
    )
    projected_gates = GateResources(
        total=gates.total,
        single_qubit=_ZERO,
        two_qubit=projected_two_qubit,
        multi_qubit=projected_multi_qubit,
        clifford=projected_clifford,
        rotation=possible_one_or_two_qubit,
        t=_ZERO,
        toffoli=projected_toffoli,
        non_clifford=gates.total,
    )
    serial_depth = DepthResources(
        depth=sp.Max(estimate.depth.depth, gates.total),
        clifford_depth=projected_clifford,
        rotation_depth=possible_one_or_two_qubit,
        t_depth=_ZERO,
        toffoli_depth=projected_toffoli,
        non_clifford_depth=gates.total,
        measurement_depth=_ZERO,
        gate_depth=sp.Max(estimate.depth.gate_depth, gates.total),
        reset_depth=_ZERO,
    )
    controlled = dataclasses.replace(
        estimate,
        gates=projected_gates,
        depth=serial_depth,
        trace=_merge_trace(
            f"controlled_abstract_projection({controls})",
            estimate.trace,
            ResourceTraceNode(
                name="abstract_controlled_aggregate",
                source_kind="abstract_control",
                summary=f"gates={gates.total}, controls={controls}",
            ),
        ),
        parameters={},
        _dependency_completion=None,
        _dependency_completion_uniform=None,
    )
    common_message = (
        "controlled aggregate cost uses one abstract operation per source "
        "primitive and serializes gate depth because every operation shares "
        "the coherent controls. Gate names and original scheduling are "
        "unavailable. Gate-family fields are independent field-wise upper "
        "bounds and may not sum to total; controlled primitives are not "
        "reported as T gates"
    )
    complete_assumption = ResourceAssumption(
        message=(
            f"{common_message}. Every declared arity bucket is shifted by the "
            "added control count. The explicit aggregate cost is treated as a "
            "complete contract, including any phase-relevant work required "
            "under later coherent controls"
        )
    )
    partial_assumption = ResourceAssumption(
        message=(
            f"{common_message}; one or more source gates have unclassified "
            "arity and remain outside single_qubit, two_qubit, and multi_qubit. "
            "The transformed arity fields therefore may not sum to total"
        )
    )
    active_controls = sp.Gt(controls, _ZERO)
    active_projection = sp.And(active_controls, sp.Gt(gates.total, _ZERO))
    if unclassified_count == _ZERO:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=active_projection,
        )
    elif unclassified_count.is_positive is True:
        controlled = controlled._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=active_controls,
        )
    else:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=sp.And(
                active_projection,
                sp.Eq(unclassified_count, _ZERO),
            ),
        )._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=sp.And(
                active_controls,
                sp.Gt(unclassified_count, _ZERO),
            ),
        )
    missing_gate_profile_active = _simplify_aggregate_metadata_guard(
        sp.And(
            active_controls,
            sp.Eq(gates.total, _ZERO),
            _aggregate_zero_gate_residual_condition(estimate),
        )
    )
    controlled = controlled._with_metadata(
        assumptions=(
            _unprojected_aggregate_control_assumption(
                "the zero gate profile has no declared primitive arity to project",
                controls,
            ),
        ),
        derivation=EstimateDerivation.MODELED,
        quality=EstimateQuality.UNKNOWN,
        active_when=missing_gate_profile_active,
    )
    return controlled, ""


def _clean_ancilla_shared_aggregate_control_ladder(
    body: ResourceEstimate,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Wrap a singly controlled aggregate body in one shared AND ladder.

    The caller supplies the body cost after every modeled primitive has been
    reduced to one effective control. The outer ladder remains live while that
    body runs, so its clean workspace is added rather than reused.

    Args:
        body (ResourceEstimate): Aggregate body projected beneath one effective
            coherent control.
        controls (ResourceExpr): Original positive control count. A symbolic
            zero branch is restored by :meth:`ResourceEstimate.controlled`
            after this helper returns.

    Returns:
        ResourceEstimate: Body-wide clean-ancilla control projection with one
        compute/uncompute ladder and concurrently held clean ancillas.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    outer_clean_ancillas = recipe.clean_ancillas
    toffoli = _clean_ancilla_primitive_estimate(body.zero(), "toffoli")
    compute = toffoli.repeat(recipe.compute_toffolis)
    ladder = toffoli.repeat(recipe.total_toffolis)
    estimate = ladder.seq(body)
    trace_children = tuple(
        node for node in (compute.trace, body.trace, compute.trace) if node is not None
    )
    return dataclasses.replace(
        estimate,
        width=dataclasses.replace(
            body.width,
            clean_ancilla_qubits=(
                body.width.clean_ancilla_qubits + outer_clean_ancillas
            ),
            peak_qubits=body.width.peak_qubits + outer_clean_ancillas,
        ),
        trace=ResourceTraceNode(
            name=f"shared_control_ladder({controls})",
            source_kind="clean_ancilla_toffoli",
            summary=(
                f"toffoli_steps={recipe.total_toffolis}, "
                f"clean_ancillas={outer_clean_ancillas}"
            ),
            children=trace_children,
        ),
        _output_sizes=body._output_sizes,
        _input_sizes=body._input_sizes,
        _has_output_summary=body._has_output_summary,
        _dependency_keys=body._dependency_keys,
        _dependency_reads=body._dependency_reads,
        _dependency_writes=body._dependency_writes,
        _symbol_aliases=body._symbol_aliases,
    )


def _project_clean_ancilla_aggregate_controlled_cost(
    estimate: ResourceEstimate,
    controls: ResourceExpr,
) -> tuple[ResourceEstimate | None, str]:
    """Project the known part of an aggregate arity profile through controls.

    The projection uses one shared conjunction when at least two controls
    surround at least two modeled operations. Smaller cases retain the
    per-primitive projection. Gate-name uncertainty is represented by
    field-wise one- and two-qubit envelopes. Any remaining gate count stays as
    a unit-cost opaque serial placeholder without being assigned a false
    arity. Decomposition ancillas for the known portion are added to any
    scratch width already declared by the opaque cost.

    Args:
        estimate (ResourceEstimate): Aggregate cost before adding controls.
        controls (ResourceExpr): Number of added coherent controls.

    Returns:
        tuple[ResourceEstimate | None, str]: Partially projected estimate and
            an empty reason, or ``None`` with the reason projection was
            unavailable.
    """
    reason = _aggregate_arity_projection_reason(estimate)
    if reason is not None:
        return None, reason
    single = _clean_ancilla_controlled_arity_envelope(estimate.zero(), 1, _ONE).repeat(
        estimate.gates.single_qubit
    )
    two = _clean_ancilla_controlled_arity_envelope(estimate.zero(), 2, _ONE).repeat(
        estimate.gates.two_qubit
    )
    unresolved_count = _safe_simplify(
        estimate.gates.total - estimate.gates.single_qubit - estimate.gates.two_qubit
    )
    unresolved = dataclasses.replace(
        estimate.zero(),
        gates=GateResources(
            total=unresolved_count,
            multi_qubit=estimate.gates.multi_qubit,
        ),
        depth=DepthResources(
            depth=unresolved_count,
            gate_depth=unresolved_count,
        ),
        trace=ResourceTraceNode(
            name="unresolved_controlled_aggregate",
            source_kind="opaque_arity_remainder",
            summary=f"gates={unresolved_count}",
        ),
        derivation=EstimateDerivation.MODELED,
        control_decomposition=estimate.control_decomposition,
    )
    projected_body = single.seq(two).seq(unresolved)
    shared_projection = _clean_ancilla_shared_aggregate_control_ladder(
        projected_body,
        controls,
    )
    per_primitive_projection = (
        _clean_ancilla_controlled_arity_envelope(estimate.zero(), 1, controls)
        .repeat(estimate.gates.single_qubit)
        .seq(
            _clean_ancilla_controlled_arity_envelope(
                estimate.zero(), 2, controls
            ).repeat(estimate.gates.two_qubit)
        )
        .seq(unresolved)
    )
    aggregate_profile = _clean_ancilla_aggregate_control_profile(estimate)
    use_shared_ladder = _clean_ancilla_shared_ladder_condition(
        controls,
        aggregate_profile,
    )
    projected = shared_projection.conditional(
        per_primitive_projection,
        use_shared_ladder,
    )
    has_unresolved_gates = sp.Gt(unresolved_count, _ZERO)
    projected_gates = dataclasses.replace(
        projected.gates,
        clifford=sp.Max(
            projected.gates.clifford,
            _piecewise(
                estimate.gates.clifford,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        rotation=sp.Max(
            projected.gates.rotation,
            _piecewise(
                estimate.gates.rotation,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        t=sp.Max(
            projected.gates.t,
            _piecewise(
                estimate.gates.t,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        toffoli=sp.Max(
            projected.gates.toffoli,
            _piecewise(
                estimate.gates.toffoli,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
        non_clifford=sp.Max(
            projected.gates.non_clifford,
            _piecewise(
                estimate.gates.non_clifford,
                _ZERO,
                has_unresolved_gates,
            ),
        ),
    )
    added_clean_ancillas = projected.width.clean_ancilla_qubits
    width = dataclasses.replace(
        estimate.width,
        clean_ancilla_qubits=(
            estimate.width.clean_ancilla_qubits + added_clean_ancillas
        ),
        peak_qubits=estimate.width.peak_qubits + added_clean_ancillas,
    )
    complete_assumption = ResourceAssumption(
        message=(
            "controlled aggregate cost uses the aggregate clean-ancilla "
            "Toffoli "
            "batching model: at least two modeled operations under at least "
            "two active controls share one computed AND ladder and use "
            "conservative single-control arity upper bounds. Smaller cases "
            "use conservative per-primitive arity upper bounds. "
            "Arity fields are independent field-wise upper bounds and may not "
            "sum to total. "
            "The bounds range over Qamomile's supported logical primitive "
            "families. Gate kinds and original scheduling are unavailable, so "
            "this fixed aggregate recipe may differ from a concrete engine's "
            "emission choices. Decomposition clean ancillas are counted in "
            "addition to declared opaque workspace. The explicit aggregate "
            "cost is treated as a complete contract, including any "
            "phase-relevant work required under later coherent controls"
        )
    )
    partial_assumption = ResourceAssumption(
        message=(
            "controlled aggregate cost uses the aggregate clean-ancilla "
            "Toffoli "
            "batching model: at least two modeled operations under at least "
            "two active controls share one computed AND ladder and use "
            "conservative single-control arity upper bounds. Smaller cases "
            "use conservative per-primitive arity upper bounds; "
            "one or more gates outside the supported one- and two-qubit "
            "buckets remain one modeled operation each in total and serial "
            "depth. This unresolved portion may include gates with "
            "unclassified arity; declared multi_qubit gates remain classified "
            "but are not decomposed. "
            "Their controlled decomposition and additional clean ancillas "
            "are unavailable, unclassified gates are not reported as "
            "multi_qubit. Arity fields are independent field-wise upper bounds "
            "and may not sum to total. The supported arity bounds range over "
            "Qamomile's logical primitive families, but the overall result is "
            "a model rather than a full upper bound because the remainder "
            "decomposition is unavailable. This fixed aggregate recipe may "
            "differ from a concrete engine's emission choices. Declared "
            "gate-family counts are retained only as field-wise floors"
        )
    )
    controlled = dataclasses.replace(
        estimate,
        width=width,
        gates=projected_gates,
        depth=_max_depth(estimate.depth, projected.depth),
        trace=_merge_trace(
            f"controlled_arity_projection({controls})",
            estimate.trace,
            projected.trace,
        ),
        parameters={},
        _constraints=(
            *estimate._constraints,
            *projected._constraints,
        ),
        _dependency_completion=None,
        _dependency_completion_uniform=None,
    )
    active_controls = sp.Gt(controls, _ZERO)
    active_projection = sp.And(
        active_controls,
        sp.Gt(estimate.gates.total, _ZERO),
    )
    if unresolved_count == _ZERO:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=active_projection,
        )
    elif unresolved_count.is_positive is True:
        controlled = controlled._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=active_controls,
        )
    else:
        controlled = controlled._with_metadata(
            assumptions=(complete_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.CONSERVATIVE,
            active_when=sp.And(
                active_projection,
                sp.Eq(unresolved_count, _ZERO),
            ),
        )._with_metadata(
            assumptions=(partial_assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            active_when=sp.And(
                active_controls,
                sp.Gt(unresolved_count, _ZERO),
            ),
        )
    known_arity_count = _safe_simplify(
        estimate.gates.single_qubit + estimate.gates.two_qubit
    )
    if known_arity_count.is_positive is not True:
        positive_total_active = _simplify_aggregate_metadata_guard(
            sp.And(
                active_controls,
                sp.Eq(known_arity_count, _ZERO),
                sp.Gt(estimate.gates.total, _ZERO),
            )
        )
        zero_gate_residual_active = _simplify_aggregate_metadata_guard(
            sp.And(
                active_controls,
                sp.Eq(known_arity_count, _ZERO),
                sp.Eq(estimate.gates.total, _ZERO),
                _aggregate_zero_gate_residual_condition(estimate),
            )
        )
        retained = (
            dataclasses.replace(
                estimate,
                trace=_wrap_trace(f"controlled({controls})", estimate.trace),
            )
            ._with_metadata(
                assumptions=(
                    _unprojected_aggregate_control_assumption(
                        "the aggregate has no declared one- or two-qubit gate profile",
                        controls,
                    ),
                ),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
                active_when=positive_total_active,
            )
            ._with_metadata(
                assumptions=(
                    _unprojected_aggregate_control_assumption(
                        "the zero gate profile has no declared primitive arity to project",
                        controls,
                    ),
                ),
                derivation=EstimateDerivation.MODELED,
                quality=EstimateQuality.UNKNOWN,
                active_when=zero_gate_residual_active,
            )
        )
        controlled = retained.conditional(
            controlled,
            sp.Eq(known_arity_count, _ZERO),
        )
        controlled = dataclasses.replace(
            controlled,
            _allocation_sites=estimate._allocation_sites,
            _output_sizes=estimate._output_sizes,
            _input_sizes=estimate._input_sizes,
            _has_output_summary=estimate._has_output_summary,
            _dependency_keys=estimate._dependency_keys,
            _dependency_reads=estimate._dependency_reads,
            _dependency_writes=estimate._dependency_writes,
            _symbol_aliases=estimate._symbol_aliases,
        )
    return controlled, ""
