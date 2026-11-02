"""Versioned resource coordinates and metric definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .expr import (
    BYTES,
    CALLS,
    DEPENDENCY_STEPS,
    INSTRUCTIONS,
    LOGICAL_QUBITS,
    MEASUREMENTS,
    ONE,
    PHYSICAL_QUBIT_SECONDS,
    PHYSICAL_QUBITS,
    QUERIES,
    RESETS,
    SECONDS,
    SEMANTIC_OPS,
    SHOTS,
    Domain,
    Unit,
)


class Coordinate(str, Enum):
    """The abstraction coordinate to which a resource fact belongs."""

    ALGORITHMIC = "algorithmic"
    LOGICAL = "logical"
    CIRCUIT = "circuit"
    FAULT_TOLERANT = "fault_tolerant"
    PHYSICAL = "physical"
    EMITTED = "emitted"
    OBSERVED = "observed_execution"


class Composition(str, Enum):
    """A registered structured-composition algebra."""

    ADDITIVE_WORK = "additive_work"
    DEPENDENCY_SPAN = "dependency_span"
    PEAK_LIVENESS = "peak_liveness"
    REPLACEMENT = "replacement"
    ERROR_BOUND = "error_bound"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """The complete semantics required to interpret one metric identity."""

    identifier: str
    coordinate: Coordinate
    domain: Domain
    unit: Unit
    composition: Composition = Composition.ADDITIVE_WORK
    version: str = "1"

    def __post_init__(self) -> None:
        """Reject ambiguous or unversioned metric identities."""

        if "." not in self.identifier or not self.version:
            raise ValueError("metric identifiers must be namespaced and versioned")


def logical_operation_count(opcode: str) -> MetricDefinition:
    """Return the metric definition for one QCore operation count.

    Args:
        opcode (str): Canonical QCore operation name such as ``quantum.h``.

    Returns:
        MetricDefinition: A deterministic logical-coordinate count metric.

    Raises:
        ValueError: If ``opcode`` is not a stable namespaced identifier.
    """

    if (
        "." not in opcode
        or not opcode
        or any(
            not (character.isalnum() or character in {".", "_"}) for character in opcode
        )
    ):
        raise ValueError(f"invalid QCore opcode for resource counting: {opcode!r}")
    return MetricDefinition(
        f"qamomile.logical.operation_count.{opcode}",
        Coordinate.LOGICAL,
        Domain.NATURAL,
        SEMANTIC_OPS,
    )


def algorithmic_call_count(callable_name: str) -> MetricDefinition:
    """Return an invocation-count metric for a source or external callable.

    Args:
        callable_name (str): Qualified callable identity in the selected package.

    Returns:
        MetricDefinition: Additive algorithmic calls, not an elementary gate cost.

    Raises:
        ValueError: If the callable identity cannot form a metric name.
    """

    _validate_callable_metric_name(callable_name)
    return MetricDefinition(
        f"qamomile.algorithmic.call_count.{callable_name}",
        Coordinate.ALGORITHMIC,
        Domain.NATURAL,
        CALLS,
    )


def algorithmic_query_count(callable_name: str) -> MetricDefinition:
    """Return the oracle-query metric for one external quantum callable."""

    _validate_callable_metric_name(callable_name)
    return MetricDefinition(
        f"qamomile.algorithmic.query_count.{callable_name}",
        Coordinate.ALGORITHMIC,
        Domain.NATURAL,
        QUERIES,
    )


def _validate_callable_metric_name(callable_name: str) -> None:
    """Reject callable identities that cannot be embedded in metric names."""

    if (
        not callable_name
        or "." not in callable_name
        or any(
            not (character.isalnum() or character in {".", "_"})
            for character in callable_name
        )
    ):
        raise ValueError(f"invalid callable metric identity: {callable_name!r}")


def portable_metric(name: str) -> MetricDefinition:
    """Select a metric in the explicit portable circuit model.

    Args:
        name (str): Gate-family name, ``gate_depth``, or ``clean_ancilla_qubits``.

    Returns:
        MetricDefinition: Basis-relative circuit metric, not a source metric.

    Raises:
        ValueError: If the metric name is unsupported.
    """

    families = {
        "gates",
        "single_qubit_gates",
        "two_qubit_gates",
        "multi_qubit_gates",
        "clifford_gates",
        "rotation_gates",
        "t_gates",
        "toffoli_gates",
        "non_clifford_gates",
    }
    if name == "gate_depth":
        unit, composition = DEPENDENCY_STEPS, Composition.DEPENDENCY_SPAN
    elif name == "clean_ancilla_qubits":
        unit, composition = LOGICAL_QUBITS, Composition.PEAK_LIVENESS
    elif name in families:
        unit, composition = INSTRUCTIONS, Composition.ADDITIVE_WORK
    else:
        raise ValueError(f"unknown portable resource metric: {name!r}")
    return MetricDefinition(
        f"qamomile.circuit.portable.{name}",
        Coordinate.CIRCUIT,
        Domain.NATURAL,
        unit,
        composition,
    )


CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND = MetricDefinition(
    "qamomile.circuit.output_trace_distance_bound",
    Coordinate.CIRCUIT,
    Domain.REAL,
    ONE,
    composition=Composition.ERROR_BOUND,
)
"""Certified upper bound on distance from the ideal semantic output."""


LOGICAL_PEAK_QUBITS = MetricDefinition(
    "qamomile.logical.peak_live_qubits",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    LOGICAL_QUBITS,
    composition=Composition.PEAK_LIVENESS,
)
LOGICAL_INPUT_QUBITS = MetricDefinition(
    "qamomile.logical.input_qubits",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    LOGICAL_QUBITS,
    composition=Composition.ADDITIVE_WORK,
)
LOGICAL_ALLOCATED_QUBITS = MetricDefinition(
    "qamomile.logical.allocated_qubits",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    LOGICAL_QUBITS,
    composition=Composition.ADDITIVE_WORK,
)
LOGICAL_CIRCUIT_QUBITS = MetricDefinition(
    "qamomile.logical.static_circuit_qubits",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    LOGICAL_QUBITS,
    composition=Composition.ADDITIVE_WORK,
)
LOGICAL_WORK = MetricDefinition(
    "qamomile.logical.semantic_work",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_GATES = MetricDefinition(
    "qamomile.logical.gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_SINGLE_QUBIT_GATES = MetricDefinition(
    "qamomile.logical.single_qubit_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_TWO_QUBIT_GATES = MetricDefinition(
    "qamomile.logical.two_qubit_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_MULTI_QUBIT_GATES = MetricDefinition(
    "qamomile.logical.multi_qubit_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_CLIFFORD_GATES = MetricDefinition(
    "qamomile.logical.clifford_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_ROTATION_GATES = MetricDefinition(
    "qamomile.logical.rotation_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_T_GATES = MetricDefinition(
    "qamomile.logical.t_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_TOFFOLI_GATES = MetricDefinition(
    "qamomile.logical.toffoli_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_NON_CLIFFORD_GATES = MetricDefinition(
    "qamomile.logical.non_clifford_gates",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
LOGICAL_SPAN = MetricDefinition(
    "qamomile.logical.dependency_span",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_GATE_DEPTH = MetricDefinition(
    "qamomile.logical.gate_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_CLIFFORD_DEPTH = MetricDefinition(
    "qamomile.logical.clifford_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_ROTATION_DEPTH = MetricDefinition(
    "qamomile.logical.rotation_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_T_DEPTH = MetricDefinition(
    "qamomile.logical.t_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_TOFFOLI_DEPTH = MetricDefinition(
    "qamomile.logical.toffoli_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_NON_CLIFFORD_DEPTH = MetricDefinition(
    "qamomile.logical.non_clifford_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_MEASUREMENT_DEPTH = MetricDefinition(
    "qamomile.logical.measurement_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_RESET_DEPTH = MetricDefinition(
    "qamomile.logical.reset_depth",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
LOGICAL_MEASUREMENTS = MetricDefinition(
    "qamomile.logical.measurements",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    MEASUREMENTS,
)
LOGICAL_RESETS = MetricDefinition(
    "qamomile.logical.resets",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    RESETS,
)
LOGICAL_CLEANUP_WORK = MetricDefinition(
    "qamomile.logical.cleanup_work",
    Coordinate.LOGICAL,
    Domain.NATURAL,
    SEMANTIC_OPS,
)
CIRCUIT_VIRTUAL_QUBITS = MetricDefinition(
    "qamomile.circuit.virtual_qubits",
    Coordinate.CIRCUIT,
    Domain.NATURAL,
    LOGICAL_QUBITS,
    composition=Composition.PEAK_LIVENESS,
)
CIRCUIT_VIRTUAL_GATES = MetricDefinition(
    "qamomile.circuit.virtual_gates",
    Coordinate.CIRCUIT,
    Domain.NATURAL,
    INSTRUCTIONS,
)
CIRCUIT_CLEAN_ANCILLA_QUBITS = MetricDefinition(
    "qamomile.circuit.clean_ancilla_qubits",
    Coordinate.CIRCUIT,
    Domain.NATURAL,
    LOGICAL_QUBITS,
    composition=Composition.PEAK_LIVENESS,
)
CIRCUIT_TOFFOLI_GATES = MetricDefinition(
    "qamomile.circuit.toffoli_gates",
    Coordinate.CIRCUIT,
    Domain.NATURAL,
    INSTRUCTIONS,
)
CIRCUIT_GATE_DEPTH = MetricDefinition(
    "qamomile.circuit.gate_depth",
    Coordinate.CIRCUIT,
    Domain.NATURAL,
    DEPENDENCY_STEPS,
    composition=Composition.DEPENDENCY_SPAN,
)
EMITTED_INSTRUCTIONS = MetricDefinition(
    "qamomile.emitted.instructions",
    Coordinate.EMITTED,
    Domain.NATURAL,
    INSTRUCTIONS,
)
EMITTED_ARTIFACT_BYTES = MetricDefinition(
    "qamomile.emitted.artifact_bytes",
    Coordinate.EMITTED,
    Domain.NATURAL,
    BYTES,
)
PHYSICAL_QUBIT_COUNT = MetricDefinition(
    "qamomile.physical.qubits",
    Coordinate.PHYSICAL,
    Domain.NATURAL,
    PHYSICAL_QUBITS,
    composition=Composition.PEAK_LIVENESS,
)
PHYSICAL_CODE_DISTANCE = MetricDefinition(
    "qamomile.fault_tolerant.code_distance",
    Coordinate.FAULT_TOLERANT,
    Domain.NATURAL,
    ONE,
    composition=Composition.REPLACEMENT,
)
PHYSICAL_DURATION = MetricDefinition(
    "qamomile.physical.duration",
    Coordinate.PHYSICAL,
    Domain.REAL,
    SECONDS,
    composition=Composition.DEPENDENCY_SPAN,
)
PHYSICAL_QUBIT_TIME = MetricDefinition(
    "qamomile.physical.qubit_seconds",
    Coordinate.PHYSICAL,
    Domain.REAL,
    PHYSICAL_QUBIT_SECONDS,
    composition=Composition.ADDITIVE_WORK,
)
OBSERVED_SHOTS = MetricDefinition(
    "qamomile.observed_execution.shots",
    Coordinate.OBSERVED,
    Domain.NATURAL,
    SHOTS,
)
OBSERVED_SEGMENT_SHOTS = MetricDefinition(
    "qamomile.observed_execution.physical_segment_shots",
    Coordinate.OBSERVED,
    Domain.NATURAL,
    SHOTS,
)
OBSERVED_SUBMISSIONS = MetricDefinition(
    "qamomile.observed_execution.submissions",
    Coordinate.OBSERVED,
    Domain.NATURAL,
    ONE,
)

__all__ = [
    "CIRCUIT_OUTPUT_TRACE_DISTANCE_BOUND",
    "CIRCUIT_VIRTUAL_GATES",
    "CIRCUIT_VIRTUAL_QUBITS",
    "algorithmic_call_count",
    "algorithmic_query_count",
    "Composition",
    "Coordinate",
    "EMITTED_ARTIFACT_BYTES",
    "EMITTED_INSTRUCTIONS",
    "OBSERVED_SHOTS",
    "OBSERVED_SEGMENT_SHOTS",
    "OBSERVED_SUBMISSIONS",
    "LOGICAL_ALLOCATED_QUBITS",
    "LOGICAL_CIRCUIT_QUBITS",
    "LOGICAL_CLIFFORD_DEPTH",
    "LOGICAL_CLIFFORD_GATES",
    "LOGICAL_CLEANUP_WORK",
    "LOGICAL_GATE_DEPTH",
    "LOGICAL_GATES",
    "LOGICAL_INPUT_QUBITS",
    "LOGICAL_MEASUREMENT_DEPTH",
    "LOGICAL_MEASUREMENTS",
    "LOGICAL_MULTI_QUBIT_GATES",
    "LOGICAL_NON_CLIFFORD_DEPTH",
    "LOGICAL_NON_CLIFFORD_GATES",
    "LOGICAL_PEAK_QUBITS",
    "LOGICAL_RESETS",
    "LOGICAL_RESET_DEPTH",
    "LOGICAL_ROTATION_DEPTH",
    "LOGICAL_ROTATION_GATES",
    "LOGICAL_SINGLE_QUBIT_GATES",
    "LOGICAL_SPAN",
    "LOGICAL_T_DEPTH",
    "LOGICAL_T_GATES",
    "LOGICAL_TOFFOLI_DEPTH",
    "LOGICAL_TOFFOLI_GATES",
    "LOGICAL_TWO_QUBIT_GATES",
    "LOGICAL_WORK",
    "logical_operation_count",
    "portable_metric",
    "MetricDefinition",
    "PHYSICAL_DURATION",
    "PHYSICAL_CODE_DISTANCE",
    "PHYSICAL_QUBIT_COUNT",
    "PHYSICAL_QUBIT_TIME",
]
