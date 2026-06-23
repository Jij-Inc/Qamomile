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
from qamomile.resource_estimation.hamiltonian import (
    PauliHamiltonianResource,
    summarize_pauli_hamiltonian,
)

__all__ = [
    "FTQCCostModel",
    "FTQCPhysicalResourceEstimate",
    "GateCount",
    "PauliHamiltonianResource",
    "ResourceEstimate",
    "count_gates",
    "estimate_physical_resources",
    "estimate_resources",
    "qubits_counter",
    "summarize_pauli_hamiltonian",
]
