"""Public resource-estimation APIs for Qamomile programs.

Every estimate flows through five layers, and every value at every layer is
a SymPy expression so estimates stay symbolic until the caller substitutes
concrete parameters:

1. Problem summary — ``PauliHamiltonianResource`` (from Qamomile or
   OpenFermion-style inputs).
2. Algorithm workload — ``HamiltonianQPEWorkload``, ``TrotterQPEWorkload``,
   ``BlockEncodingResource``.
3. Logical estimate — the architecture-independent
   ``qamomile.circuit.estimator.ResourceEstimate``.
4. Architecture lift — ``FTQCCostModel``, ``SurfaceCodeCostModel``,
   ``ActiveVolumeCostModel``.
5. Comparison — canonical quantity keys and ``compare_resource_values``.

The package is extension-first; adding an algorithm does not require
editing it:

- Resource values are exchanged as ``dict[str, sp.Expr]`` keyed by plain
  strings. ``ResourceQuantity`` catalogs the built-ins and is not a
  whitelist; attach display metadata for new keys with
  ``register_resource_quantity()`` (registries are last-wins so notebook
  re-execution keeps working).
- New workloads subclass ``HamiltonianWorkloadMixin`` and declare fields,
  validation tuples, and ``_own_resource_values()``; validation,
  precision-budget accounting, and serialization are inherited.
- Custom Hamiltonian representations register a logical-qubit scaling model
  via ``register_hamiltonian_representation()``.
- ``GateCount.oracle_calls`` counters pass through ``resource_values()``
  generically under their own names.

Formula-based algorithm estimators live here (``estimate_trotter``,
``estimate_qpe``, ``estimate_qaoa``, and the workload estimators);
``qamomile.circuit.estimator`` owns circuit-derived counting only, and the
dependency direction is ``resource_estimation -> qamomile.circuit``, never
the reverse.
"""

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
    register_hamiltonian_representation,
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
    "register_hamiltonian_representation",
    "register_resource_quantity",
    "resource_values_from_estimate",
    "resource_estimate_expressions",
    "summarize_openfermion_qubit_operator",
    "summarize_pauli_hamiltonian",
    "trotter_qpe_workload_from_openfermion",
]
