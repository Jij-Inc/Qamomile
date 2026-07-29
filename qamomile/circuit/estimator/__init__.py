"""Algorithmic symbolic resource estimation for Qamomile circuits.

The default ``portable`` basis recursively expands coherent controls through
Qamomile's backend-neutral fallback, including reusable clean ancillas and
body-wide shared control ladders when concrete structure permits them.
``logical`` retains the former abstract one-source-operation view, and
``clifford_t`` reports the supported synthesis model. The physical
(surface-code) conversion in :mod:`qamomile.circuit.estimator.physical`
remains experimental and intentionally is not re-exported here, keeping
algorithmic estimation and physical assumptions clearly separated.
Measurements and resets are reported independently from gates, while depth
retains both the complete critical path and per-operation-class layers.
"""

from qamomile.circuit.estimator.resource_estimator import (
    CallResources,
    DepthResources,
    EstimateQuality,
    GateBasis,
    GateResources,
    MeasurementResources,
    OpaqueCallContext,
    ResetResources,
    ResourceAssumption,
    ResourceEstimate,
    ResourceEstimator,
    ResourceEstimatorConfig,
    ResourceTraceNode,
    UnknownResourcePolicy,
    WidthResources,
    estimate_resources,
)

__all__ = [
    "CallResources",
    "DepthResources",
    "EstimateQuality",
    "GateBasis",
    "GateResources",
    "MeasurementResources",
    "OpaqueCallContext",
    "ResetResources",
    "ResourceAssumption",
    "ResourceEstimate",
    "ResourceEstimator",
    "ResourceEstimatorConfig",
    "ResourceTraceNode",
    "UnknownResourcePolicy",
    "WidthResources",
    "estimate_resources",
]
