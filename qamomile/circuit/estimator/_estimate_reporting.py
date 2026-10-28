"""Render resource estimates for people and report consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.estimator._estimate_domain import (
    _validate_domain_rewrite_state,
)
from qamomile.circuit.estimator._symbol_discovery import _serialization_registry

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _explain_estimate(
    estimate: ResourceEstimate,
    metric: str | None = None,
) -> str:
    """Render a resource-estimation explanation tree.

    Args:
        estimate (ResourceEstimate): Estimate whose trace is rendered.
        metric (str | None): Optional metric name to mention in the heading.
            Filtering is reserved for a later pass. Defaults to ``None``.
    Returns:
        str: Human-readable explanation tree.

    Raises:
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    _validate_domain_rewrite_state(estimate)
    heading = "Resource estimate"
    if metric is not None:
        heading = f"{heading} for {metric}"
    if estimate.trace is None:
        return heading
    registry = _serialization_registry(estimate)
    return f"{heading}\n{estimate.trace.render(2, registry)}"


def _estimate_to_dict(
    estimate: ResourceEstimate,
) -> dict[str, Any]:
    """Convert an estimate to a JSON-friendly report snapshot.

    Args:
        estimate (ResourceEstimate): Estimate to serialize for reporting.
    Returns:
        dict[str, Any]: Report fields with stringified resource expressions.

    Raises:
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    _validate_domain_rewrite_state(estimate)
    registry = _serialization_registry(estimate)
    serialize = registry.stringify
    return {
        "width": {
            "input_qubits": serialize(estimate.width.input_qubits),
            "allocated_qubits": serialize(estimate.width.allocated_qubits),
            "clean_ancilla_qubits": serialize(estimate.width.clean_ancilla_qubits),
            "dirty_ancilla_qubits": serialize(estimate.width.dirty_ancilla_qubits),
            "peak_qubits": serialize(estimate.width.peak_qubits),
            "circuit_qubits": serialize(estimate.width.circuit_qubits),
        },
        "gates": {
            "total": serialize(estimate.gates.total),
            "single_qubit": serialize(estimate.gates.single_qubit),
            "two_qubit": serialize(estimate.gates.two_qubit),
            "multi_qubit": serialize(estimate.gates.multi_qubit),
            "clifford": serialize(estimate.gates.clifford),
            "rotation": serialize(estimate.gates.rotation),
            "t": serialize(estimate.gates.t),
            "toffoli": serialize(estimate.gates.toffoli),
            "non_clifford": serialize(estimate.gates.non_clifford),
        },
        "measurements": {
            "total": serialize(estimate.measurements.total),
        },
        "resets": {
            "total": serialize(estimate.resets.total),
        },
        "depth": {
            "depth": serialize(estimate.depth.depth),
            "clifford_depth": serialize(estimate.depth.clifford_depth),
            "rotation_depth": serialize(estimate.depth.rotation_depth),
            "t_depth": serialize(estimate.depth.t_depth),
            "toffoli_depth": serialize(estimate.depth.toffoli_depth),
            "non_clifford_depth": serialize(estimate.depth.non_clifford_depth),
            "measurement_depth": serialize(estimate.depth.measurement_depth),
            "gate_depth": serialize(estimate.depth.gate_depth),
            "reset_depth": serialize(estimate.depth.reset_depth),
        },
        "calls": {
            "calls_by_name": {
                name: serialize(value)
                for name, value in estimate.calls.calls_by_name.items()
            },
            "queries_by_name": {
                name: serialize(value)
                for name, value in estimate.calls.queries_by_name.items()
            },
        },
        "assumptions": [
            {"message": assumption.message, "source": assumption.source}
            for assumption in estimate.assumptions
        ],
        "parameters": {
            name: registry.name(symbol) for name, symbol in estimate.parameters.items()
        },
        "derivation": estimate.derivation.value,
        "quality": estimate.quality.value,
        "approximation": estimate.approximation.value,
        "control_decomposition": estimate.control_decomposition.value,
        "requirements": [
            {
                "expression": serialize(constraint.expression),
                "active_when": serialize(constraint.active_when),
                "minimum": constraint.minimum,
                "minimum_inclusive": constraint.minimum_inclusive,
                "finite": constraint.finite,
                "expected": (
                    serialize(constraint.expected)
                    if constraint.expected is not None
                    else None
                ),
                "integer": constraint.integer,
                "label": constraint.label,
                "unit": constraint.unit,
                "ranges": [
                    {
                        "symbol": registry.name(loop_range.symbol),
                        "start": serialize(loop_range.start),
                        "step": serialize(loop_range.step),
                        "iterations": serialize(loop_range.iterations),
                    }
                    for loop_range in constraint.ranges
                ],
            }
            for constraint in estimate._constraints
        ],
    }


__all__: list[str] = []
