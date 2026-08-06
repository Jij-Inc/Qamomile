"""Preserve legacy pickle lookups for relocated resource metric types.

The estimator implementation now lives in responsibility-specific modules.
These aliases do not form a second implementation surface; they only let
pickles created before the split resolve their original module globals.
"""

from __future__ import annotations

from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus as ApproximationStatus,
    ControlDecomposition as ControlDecomposition,
    EstimateDerivation as EstimateDerivation,
    EstimateQuality as EstimateQuality,
    GateBasis as GateBasis,
)
from qamomile.circuit.estimator._resource_conditions import (
    _ConditionIndicator as _ConditionIndicator,
    _RangeAny as _RangeAny,
    _RangeAtLeastTwo as _RangeAtLeastTwo,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ConstraintRange as _ConstraintRange,
    _ResourceConstraint as _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources as CallResources,
    DepthResources as DepthResources,
    GateResources as GateResources,
    MeasurementResources as MeasurementResources,
    ResetResources as ResetResources,
    ResourceAssumption as ResourceAssumption,
    ResourceTraceNode as ResourceTraceNode,
    WidthResources as WidthResources,
    _GuardedApproximation as _GuardedApproximation,
    _GuardedAssumption as _GuardedAssumption,
    _GuardedDerivation as _GuardedDerivation,
    _GuardedQuality as _GuardedQuality,
)
