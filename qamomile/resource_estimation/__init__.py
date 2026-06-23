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
    SurfaceCodeCostModel,
    estimate_physical_resources,
)
from qamomile.resource_estimation.hamiltonian import (
    PauliHamiltonianResource,
    hamiltonian_from_openfermion_qubit_operator,
    summarize_openfermion_qubit_operator,
    summarize_pauli_hamiltonian,
)
from qamomile.resource_estimation.hamiltonian_algorithms import (
    HamiltonianQPEWorkload,
    HamiltonianRepresentation,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    estimate_trotter_qpe_resources,
    estimate_trotter_qpe_resources_from_hamiltonian,
)
from qamomile.resource_estimation.quantities import (
    ResourceCategory,
    ResourceComparisonRow,
    ResourceQuantity,
    ResourceQuantitySpec,
    compare_resource_values,
    describe_resource_quantity,
    iter_resource_quantity_specs,
)

__all__ = [
    "FTQCCostModel",
    "FTQCPhysicalResourceEstimate",
    "GateCount",
    "HamiltonianQPEWorkload",
    "HamiltonianRepresentation",
    "PauliHamiltonianResource",
    "ResourceCategory",
    "ResourceComparisonRow",
    "ResourceQuantity",
    "ResourceQuantitySpec",
    "ResourceEstimate",
    "SurfaceCodeCostModel",
    "compare_resource_values",
    "count_gates",
    "describe_resource_quantity",
    "estimate_physical_resources",
    "estimate_qubitized_qpe_resources",
    "estimate_qubitized_qpe_resources_from_workload",
    "estimate_resources",
    "estimate_trotter_qpe_resources",
    "estimate_trotter_qpe_resources_from_hamiltonian",
    "hamiltonian_from_openfermion_qubit_operator",
    "iter_resource_quantity_specs",
    "qubits_counter",
    "summarize_openfermion_qubit_operator",
    "summarize_pauli_hamiltonian",
]
