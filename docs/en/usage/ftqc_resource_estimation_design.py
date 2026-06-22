# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [usage, chemistry, resource-estimation]
# ---
#
# # FTQC Resource Estimation Design
#
# This notebook explains the resource-estimation quantities Qamomile tracks for
# fault-tolerant quantum chemistry. It focuses on how to separate problem
# metadata, algorithmic work, logical resources, and architecture assumptions
# before lowering anything into backend-specific circuits.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCResourceCategory,
    FTQCResourceQuantity,
    SurfaceCodeCostModel,
    compare_ftqc_resource_estimates,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    iter_ftqc_resource_quantity_specs,
    summarize_pauli_hamiltonian,
)

# %% [markdown]
# ## Design Boundary
#
# FTQC chemistry papers often reduce cost by changing the Hamiltonian
# representation, not by changing the compiler IR. Qamomile therefore keeps
# this layer as algorithmic metadata:
#
# - Hamiltonian summaries capture representation-level quantities such as
#   `lambda_norm` and Pauli term count.
# - Algorithm estimators turn those quantities into QPE iterations, Toffoli or
#   T counts, logical qubits, logical depth, physical qubits, and runtime
#   proxies.
# - Backend emitters remain responsible for concrete circuit lowering.
#
# This matches the compiler rule used elsewhere in Qamomile: keep the IR
# abstract and push concretization as late as possible.

# %% [markdown]
# ## Research Signals
#
# The current quantity catalog follows the cost drivers used in recent FTQC
# chemistry work:
#
# | Research direction | Cost signal for Qamomile |
# | --- | --- |
# | Symmetry-compressed double factorization reduces the Hamiltonian 1-norm and Toffoli count for qubitized chemistry QPE ([arXiv:2403.03502](https://arxiv.org/abs/2403.03502)). | Track `lambda_norm`, QPE iterations, per-walk Toffoli cost, and total Toffoli count separately. |
# | Simultaneous symmetry shifts and tensor factorizations reduce the block-encoding scaling constant for electronic Hamiltonians ([arXiv:2412.01338](https://arxiv.org/abs/2412.01338)). | Treat Hamiltonian normalization as representation metadata, not as an emitted-circuit property. |
# | Early-FTQC single-ancilla QPE with unitary weight concentration targets smaller physical-qubit budgets and limited depth ([arXiv:2603.22778](https://arxiv.org/abs/2603.22778)). | Track T gates, logical depth, physical qubits, runtime, and architecture knobs in addition to Toffoli-native qubitization costs. |
#
# These are modeling quantities. They should be validated against each paper's
# assumptions before being used as a molecule-specific resource claim.

# %% [markdown]
# ## Quantity Catalog
#
# Qamomile exposes canonical quantity keys so reports and tutorials do not have
# to invent ad hoc column names.

# %%
catalog = [
    spec.to_dict()
    for spec in iter_ftqc_resource_quantity_specs()
    if spec.category
    in {
        FTQCResourceCategory.PROBLEM,
        FTQCResourceCategory.ALGORITHM,
        FTQCResourceCategory.LOGICAL,
        FTQCResourceCategory.PHYSICAL,
        FTQCResourceCategory.ARCHITECTURE,
    }
]

for row in catalog:
    print(row["quantity"], row["unit"], row["category"])

assert FTQCResourceQuantity.LAMBDA_NORM.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.TOFFOLI_GATES.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.RUNTIME_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.CODE_DISTANCE.value in {
    row["quantity"] for row in catalog
}

# %% [markdown]
# ## Minimal Example
#
# Start from a Qamomile observable, summarize it once, then construct two
# representation-level models. The example below uses synthetic scaling values
# so it demonstrates the workflow without claiming a specific molecule.

# %%
toy_hamiltonian = 0.5 * qm_o.Z(0) + 0.25 * qm_o.X(1) * qm_o.X(2)
summary = summarize_pauli_hamiltonian(
    toy_hamiltonian,
    n_spin_orbitals=40,
    source="toy_pauli_lcu",
)
scaled_summary = summary.with_lambda_scale(
    sp.Float("2.0e5") / summary.lambda_norm,
    source="scaled_toy_pauli_lcu",
)

architecture = SurfaceCodeCostModel(
    code_distance=20,
    physical_cycle_time_seconds=sp.Float("5e-8"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=1,
    factory_count=4,
    physical_qubits_per_factory=5000,
    factory_cycles_per_toffoli=2,
)
cost_model = architecture.to_cost_model()

assert architecture.physical_qubits_per_logical == 800
assert architecture.factory_qubits == 20000
assert (
    sp.Abs(architecture.toffoli_throughput_per_second - sp.Float("2e6"))
    < sp.Float("1e-9")
)

baseline_model = ChemistryQPEModel(
    hamiltonian=scaled_summary,
    method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
    walk_cost_toffoli=sp.Integer(4_000),
)
compressed_model = ChemistryQPEModel(
    hamiltonian=scaled_summary.with_lambda_scale(
        sp.Float("0.5"),
        source="compressed_scaled_toy_pauli_lcu",
    ),
    method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
    walk_cost_toffoli=sp.Integer(4_400),
    second_factor_rank=9,
)

baseline = estimate_qubitized_chemistry_qpe_from_model(
    baseline_model,
    precision=sp.Float("0.0016"),
    cost_model=cost_model,
)
compressed = estimate_qubitized_chemistry_qpe_from_model(
    compressed_model,
    precision=sp.Float("0.0016"),
    cost_model=cost_model,
)

comparison = compare_ftqc_resource_estimates(
    baseline,
    compressed,
    quantities=("qpe_iterations", "toffoli_gates", "physical_qubits"),
)

for row in comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert comparison[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison[0].ratio == sp.Float("0.5")
assert comparison[1].ratio == sp.Float("0.55")

# %% [markdown]
# ## Early-FTQC Pattern
#
# Early-FTQC estimates may not be Toffoli-native. The same comparison API works
# for Trotter-style models that primarily report T gates and logical depth.

# %%
plain_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=sp.Float("0.0016"),
    trotter_steps_per_sample=8,
    samples=128,
    cost_model=cost_model,
)
uwc_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=sp.Float("0.0016"),
    trotter_steps_per_sample=8,
    samples=128,
    unitary_weight_factor=sp.Float("0.1"),
    randomized_compilation_factor=sp.Float("0.5"),
    rotation_synthesis_t_gates=3,
    cost_model=cost_model,
)

trotter_comparison = compare_ftqc_resource_estimates(
    plain_trotter,
    uwc_trotter,
    quantities=("qpe_iterations", "logical_depth", "t_gates"),
)

for row in trotter_comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert trotter_comparison[0].ratio == sp.Float("0.1")
assert trotter_comparison[1].ratio == sp.Float("0.05")

# %% [markdown]
# ## Notes
#
# :::{note}
# The numbers above are synthetic. They demonstrate how Qamomile separates cost
# drivers and compares estimates. A publication-quality molecule study still
# needs molecule-specific integrals, factorization ranks, truncation errors,
# synthesis assumptions, and architecture calibration.
# :::
#
# Keep these boundaries in mind when adding new FTQC estimators:
#
# - Add new problem metadata as structured summaries, not as IR operations.
# - Add new measured quantities to the canonical catalog before exposing new
#   report columns.
# - Use structured architecture models such as `SurfaceCodeCostModel` before
#   falling back to hand-written physical-resource knobs.
# - Keep architecture assumptions explicit so physical-qubit and runtime
#   estimates can be swapped without changing algorithm metadata.

# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - Recent FTQC chemistry work motivates tracking Hamiltonian normalization,
#   QPE iterations, non-Clifford counts, logical depth, physical qubits, and
#   runtime separately.
# - Qamomile keeps those quantities in algorithmic metadata so the circuit IR
#   remains backend-neutral.
# - Surface-code assumptions can be modeled separately and converted into the
#   cost model consumed by chemistry estimators.
# - `compare_ftqc_resource_estimates` turns symbolic estimates into reviewable
#   savings tables without hard-coding a particular chemistry factorization.
