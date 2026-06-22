"""Algorithmic resource estimators based on theoretical complexity formulas.

These estimators provide resource bounds based on published complexity
results from quantum algorithms literature, particularly from:

arXiv:2310.03011v2 - "Quantum algorithms: A survey of applications
and end-to-end complexities"

Unlike the circuit-based estimators (gate_counter, qubits_counter),
these provide theoretical estimates based on algorithm parameters
without needing the actual circuit implementation.
"""

from qamomile.circuit.estimator.algorithmic.ftqc_block_encoding import (
    BlockEncodingResource,
    block_encoding_from_chemistry_model,
    estimate_qubitized_qpe_from_block_encoding,
)
from qamomile.circuit.estimator.algorithmic.ftqc_chemistry import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCAccuracyBudget,
    FTQCCostModel,
    FTQCReference,
    FTQCResourceEstimate,
    PauliHamiltonianResource,
    QPEStatePreparationBudget,
    SurfaceCodeCostModel,
    SurfaceCodeDistanceBudget,
    estimate_qubitized_chemistry_qpe,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    hamiltonian_from_openfermion_qubit_operator,
    references_for_chemistry_qpe_method,
    summarize_openfermion_qubit_operator,
    summarize_pauli_hamiltonian,
)
from qamomile.circuit.estimator.algorithmic.ftqc_resources import (
    FTQCResourceCategory,
    FTQCResourceChangeDirection,
    FTQCResourceComparisonRow,
    FTQCResourceComparisonSummary,
    FTQCResourceFormula,
    FTQCResourceQuantity,
    FTQCResourceQuantitySpec,
    SupportsFTQCResourceValues,
    compare_ftqc_resource_estimates,
    describe_ftqc_resource_quantity,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
)
from qamomile.circuit.estimator.algorithmic.hamiltonian_simulation import (
    estimate_qdrift,
    estimate_qsvt,
    estimate_trotter,
)
from qamomile.circuit.estimator.algorithmic.qaoa import estimate_qaoa
from qamomile.circuit.estimator.algorithmic.qpe import estimate_qpe

__all__ = [
    "ChemistryQPEMethod",
    "ChemistryQPEModel",
    "BlockEncodingResource",
    "FTQCResourceChangeDirection",
    "FTQCResourceCategory",
    "FTQCResourceComparisonRow",
    "FTQCResourceComparisonSummary",
    "FTQCCostModel",
    "FTQCAccuracyBudget",
    "FTQCResourceFormula",
    "FTQCReference",
    "FTQCResourceEstimate",
    "FTQCResourceQuantity",
    "FTQCResourceQuantitySpec",
    "PauliHamiltonianResource",
    "QPEStatePreparationBudget",
    "SurfaceCodeCostModel",
    "SurfaceCodeDistanceBudget",
    "SupportsFTQCResourceValues",
    "block_encoding_from_chemistry_model",
    "compare_ftqc_resource_estimates",
    "describe_ftqc_resource_quantity",
    "estimate_qaoa",
    "estimate_qpe",
    "estimate_qubitized_chemistry_qpe",
    "estimate_qubitized_qpe_from_block_encoding",
    "estimate_qubitized_chemistry_qpe_from_model",
    "estimate_single_ancilla_trotter_qpe",
    "estimate_single_ancilla_trotter_qpe_from_hamiltonian",
    "hamiltonian_from_openfermion_qubit_operator",
    "references_for_chemistry_qpe_method",
    "iter_ftqc_resource_quantity_specs",
    "summarize_openfermion_qubit_operator",
    "summarize_pauli_hamiltonian",
    "summarize_ftqc_resource_comparison",
    "estimate_trotter",
    "estimate_qsvt",
    "estimate_qdrift",
]
