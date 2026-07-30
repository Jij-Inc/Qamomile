"""Algorithmic symbolic resource estimation for Qamomile circuits.

The default logical basis uses the clean-ancilla Toffoli control-decomposition
model, including reusable clean ancillas and body-wide shared control ladders
when concrete structure permits them. The abstract control model represents
each controlled source primitive as one logical operation, while the
``clifford_t`` basis reports the supported synthesis model. The physical
(surface-code) conversion in :mod:`qamomile.circuit.estimator.physical`
remains experimental and intentionally is not re-exported here, keeping
algorithmic estimation and physical assumptions clearly separated.
Measurements and resets are reported independently from gates, while depth
retains both the complete critical path and per-operation-class layers.
"""

from qamomile.circuit.estimator._metrics import (
    ApproximationStatus,
    CallResources,
    ControlDecomposition,
    DepthResources,
    EstimateQuality,
    GateBasis,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
)
from qamomile.circuit.estimator.resource_estimator import (
    OpaqueCostContext,
    ResourceEstimate,
    ResourceEstimator,
    UnknownResourcePolicy,
    estimate_resources,
)

__all__ = [
    "ApproximationStatus",
    "CallResources",
    "ControlDecomposition",
    "DepthResources",
    "EstimateQuality",
    "GateBasis",
    "GateResources",
    "MeasurementResources",
    "OpaqueCostContext",
    "ResetResources",
    "ResourceAssumption",
    "ResourceEstimate",
    "ResourceEstimator",
    "ResourceTraceNode",
    "UnknownResourcePolicy",
    "WidthResources",
    "estimate_resources",
]
