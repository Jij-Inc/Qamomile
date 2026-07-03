"""Public resource-estimation APIs for Qamomile programs."""

from qamomile.circuit.estimator import (
    GateCount,
    ResourceEstimate,
    count_gates,
    estimate_resources,
    qubits_counter,
)
from qamomile.resource_estimation.block_encoding import (
    BlockEncodingResource,
    estimate_qubitized_qpe_resources_from_block_encoding,
)
from qamomile.resource_estimation.ftqc import (
    ActiveVolumeCostModel,
    FTQCActiveVolumeResourceEstimate,
    FTQCCostModel,
    FTQCPhysicalResourceEstimate,
    SurfaceCodeCostModel,
    estimate_active_volume_resources,
    estimate_physical_resources,
    resource_estimate_expressions,
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
    TrotterQPEWorkload,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    estimate_trotter_qpe_resources,
    estimate_trotter_qpe_resources_from_hamiltonian,
    estimate_trotter_qpe_resources_from_workload,
    qubitized_qpe_workload_from_openfermion,
    trotter_qpe_workload_from_openfermion,
)
from qamomile.resource_estimation.hamiltonian_simulation import (
    estimate_qdrift,
    estimate_qsvt,
    estimate_trotter,
)
from qamomile.resource_estimation.qaoa import estimate_qaoa, estimate_qaoa_ising
from qamomile.resource_estimation.qpe import (
    estimate_eigenvalue_filtering,
    estimate_qpe,
)
from qamomile.resource_estimation.quantities import (
    ResourceCategory,
    ResourceComparisonRow,
    ResourceQuantity,
    ResourceQuantitySpec,
    SupportsResourceValues,
    compare_resource_values,
    describe_resource_quantity,
    iter_resource_quantity_specs,
    register_resource_quantity,
    resource_values_from_estimate,
)
from qamomile.resource_estimation.workload import HamiltonianWorkloadMixin

__all__ = [
    "FTQCCostModel",
    "ActiveVolumeCostModel",
    "FTQCActiveVolumeResourceEstimate",
    "FTQCPhysicalResourceEstimate",
    "GateCount",
    "BlockEncodingResource",
    "HamiltonianQPEWorkload",
    "HamiltonianRepresentation",
    "HamiltonianWorkloadMixin",
    "PauliHamiltonianResource",
    "ResourceCategory",
    "ResourceComparisonRow",
    "ResourceQuantity",
    "ResourceQuantitySpec",
    "ResourceEstimate",
    "SurfaceCodeCostModel",
    "SupportsResourceValues",
    "TrotterQPEWorkload",
    "compare_resource_values",
    "count_gates",
    "describe_resource_quantity",
    "estimate_active_volume_resources",
    "estimate_eigenvalue_filtering",
    "estimate_physical_resources",
    "estimate_qaoa",
    "estimate_qaoa_ising",
    "estimate_qdrift",
    "estimate_qpe",
    "estimate_qsvt",
    "estimate_qubitized_qpe_resources",
    "estimate_qubitized_qpe_resources_from_block_encoding",
    "estimate_qubitized_qpe_resources_from_workload",
    "estimate_resources",
    "estimate_trotter",
    "estimate_trotter_qpe_resources",
    "estimate_trotter_qpe_resources_from_hamiltonian",
    "estimate_trotter_qpe_resources_from_workload",
    "hamiltonian_from_openfermion_qubit_operator",
    "iter_resource_quantity_specs",
    "qubitized_qpe_workload_from_openfermion",
    "qubits_counter",
    "register_resource_quantity",
    "resource_values_from_estimate",
    "resource_estimate_expressions",
    "summarize_openfermion_qubit_operator",
    "summarize_pauli_hamiltonian",
    "trotter_qpe_workload_from_openfermion",
]
