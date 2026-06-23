"""Public resource-estimation APIs for Qamomile programs."""

from qamomile.circuit.estimator import (
    GateCount,
    ResourceEstimate,
    count_gates,
    estimate_resources,
    qubits_counter,
)
from qamomile.resource_estimation.ftqc import (
    FTQCCostModel,
    FTQCPhysicalResourceEstimate,
    estimate_physical_resources,
)

__all__ = [
    "FTQCCostModel",
    "FTQCPhysicalResourceEstimate",
    "GateCount",
    "ResourceEstimate",
    "count_gates",
    "estimate_physical_resources",
    "estimate_resources",
    "qubits_counter",
]
