"""Logical symbolic resource estimation for Qamomile circuits.

This package's public surface is the *logical* resource estimator. The physical
(surface-code) conversion in :mod:`qamomile.circuit.estimator.physical` is
experimental and intentionally not re-exported here — import it explicitly from
that module if you need it, keeping logical estimation and physical assumptions
clearly separated.
"""

from qamomile.circuit.estimator.resource_estimator import (
    CallResources,
    CostBasis,
    DepthResources,
    EstimateKind,
    FixedResourceModel,
    GateResources,
    ResourceAssumption,
    ResourceContext,
    ResourceEstimate,
    ResourceEstimator,
    ResourceEstimatorConfig,
    ResourceModel,
    ResourcePolicy,
    ResourceTraceNode,
    UnknownResourcePolicy,
    WidthResources,
    estimate_resources,
)

__all__ = [
    "CallResources",
    "CostBasis",
    "DepthResources",
    "EstimateKind",
    "FixedResourceModel",
    "GateResources",
    "ResourceAssumption",
    "ResourceContext",
    "ResourceEstimate",
    "ResourceEstimator",
    "ResourceEstimatorConfig",
    "ResourceModel",
    "ResourcePolicy",
    "ResourceTraceNode",
    "UnknownResourcePolicy",
    "WidthResources",
    "estimate_resources",
]
