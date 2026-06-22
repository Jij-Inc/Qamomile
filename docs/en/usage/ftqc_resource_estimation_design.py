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
    FTQCAccuracyBudget,
    FTQCResourceCategory,
    FTQCResourceQuantity,
    SurfaceCodeCostModel,
    SurfaceCodeDistanceBudget,
    compare_ftqc_resource_estimates,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
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
# - Accuracy budgets such as target precision and truncation error are tracked
#   as first-class quantities, because they explain why two cost estimates are
#   comparable.
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

assert FTQCResourceQuantity.LAMBDA_NORM.value in {row["quantity"] for row in catalog}
assert FTQCResourceQuantity.TARGET_PRECISION.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.TRUNCATION_ERROR.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.TOFFOLI_GATES.value in {row["quantity"] for row in catalog}
assert FTQCResourceQuantity.RUNTIME_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.CODE_DISTANCE.value in {row["quantity"] for row in catalog}

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

distance_budget = SurfaceCodeDistanceBudget(
    physical_error_rate=sp.Float("1e-3"),
    threshold_error_rate=sp.Float("1e-2"),
    target_logical_failure_probability=sp.Float("1e-9"),
    logical_operation_budget=1000,
)

print(distance_budget.to_dict())
assert distance_budget.code_distance == 21
assert (
    sp.Abs(
        distance_budget.resource_values()[FTQCResourceQuantity.LOGICAL_ERROR_RATE]
        - distance_budget.logical_failure_probability_per_operation
    )
    < sp.Float("1e-24")
)

architecture = distance_budget.to_surface_code_cost_model(
    physical_cycle_time_seconds=sp.Float("5e-8"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=1,
    factory_count=4,
    physical_qubits_per_factory=5000,
    factory_cycles_per_toffoli=2,
)
cost_model = architecture

assert architecture.code_distance == 21
assert architecture.physical_qubits_per_logical == 882
assert architecture.factory_qubits == 20000
assert sp.Abs(
    architecture.toffoli_throughput_per_second - sp.Float("4e7") / 21
) < sp.Float(
    "1e-9",
)

accuracy_budget = FTQCAccuracyBudget(
    target_precision=sp.Float("0.0016"),
    truncation_error=sp.Float("1e-4"),
)
print(accuracy_budget.to_dict())
assert sp.Abs(accuracy_budget.qpe_precision - sp.Float("0.0015")) < sp.Float("1e-12")

baseline_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary,
        method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
        walk_cost_toffoli=sp.Integer(4_000),
    )
)
compressed_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary.with_lambda_scale(
            sp.Float("0.5"),
            source="compressed_scaled_toy_pauli_lcu",
        ),
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        walk_cost_toffoli=sp.Integer(4_400),
        second_factor_rank=9,
    )
)

baseline = estimate_qubitized_chemistry_qpe_from_model(
    baseline_model,
    precision=accuracy_budget.qpe_precision,
    cost_model=cost_model,
)
compressed = estimate_qubitized_chemistry_qpe_from_model(
    compressed_model,
    precision=accuracy_budget.qpe_precision,
    cost_model=cost_model,
)

assert compressed.resource_values()[FTQCResourceQuantity.CODE_DISTANCE] == 21
assert compressed.to_dict()["architecture_values"]["code_distance"] == "21"
assert compressed_model.resource_values()[FTQCResourceQuantity.TRUNCATION_ERROR] == (
    sp.Float("1e-4")
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
assert sp.Abs(comparison[1].ratio - sp.Float("0.55")) < sp.Float("1e-12")

# %% [markdown]
# For design reviews, the summary helper groups the same rows by whether the
# candidate is smaller, larger, unchanged, or still symbolic under the current
# assumptions. The first `smaller` rows are the largest numeric reductions.

# %%
comparison_summary = summarize_ftqc_resource_comparison(
    baseline,
    compressed,
    quantities=("qpe_iterations", "toffoli_gates", "physical_qubits"),
)

for row in comparison_summary.smaller:
    print("smaller:", row.label, "by", sp.N(row.reduction, 4))
for row in comparison_summary.larger:
    print("larger:", row.label, "by", sp.N(-row.reduction, 4))

assert comparison_summary.smaller[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison_summary.larger[0].quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
assert comparison_summary.symbolic == ()

# %% [markdown]
# ## Reference Provenance
#
# Estimates also carry method-level research references. This is intentionally
# separate from the numeric formulas: a report can show which papers motivated
# the model without treating the synthetic example above as a molecule-specific
# reproduction of those papers.

# %%
for reference in compressed.references:
    print(reference.key, "-", reference.title)

compressed_reference_keys = {reference.key for reference in compressed.references}
assert "arXiv:2403.03502" in compressed_reference_keys
assert "arXiv:2412.01338" in compressed_reference_keys
assert compressed.to_dict()["references"][0]["url"].startswith("https://arxiv.org/")

# %% [markdown]
# ## Formula Provenance
#
# Values are not enough for FTQC design review: reviewers also need to know
# which symbolic formula produced each important resource. Estimates expose a
# formula table whose rows use the same canonical quantity keys as
# `resource_values()`.

# %%
formula_rows = compressed.to_formula_table()
for row in formula_rows:
    if row["quantity"] in {"qpe_iterations", "toffoli_gates", "runtime_seconds"}:
        print(row["label"], "=", row["expression"])

formula_by_quantity = {row["quantity"]: row for row in formula_rows}
assert formula_by_quantity["qpe_iterations"]["expression"] == (
    "lambda_norm/target_precision"
)
assert formula_by_quantity["toffoli_gates"]["depends_on"] == [
    "qpe_iterations",
    "walk_cost_toffoli",
]
assert "formulas" in compressed.to_dict()

# %% [markdown]
# ## Common Logical Resource Shape
#
# FTQC estimates can also expose their logical work through the same
# `ResourceEstimate` shape used by circuit-level `estimate_resources()`. This
# view deliberately omits physical qubits and runtime, which remain on the
# FTQC estimate because they depend on architecture assumptions.

# %%
logical_view = compressed.to_logical_resource_estimate()

print(logical_view)
assert logical_view.qubits == compressed.logical_qubits
assert logical_view.gates.total == compressed.toffoli_gates
assert logical_view.gates.multi_qubit == compressed.toffoli_gates
assert logical_view.gates.oracle_calls["qpe_iterations"] == compressed.qpe_iterations
assert "physical_qubits_per_logical" not in logical_view.parameters

# %% [markdown]
# ## Architecture Sensitivity
#
# Once an estimate exists, relift it with a different architecture model to
# study hardware assumptions without rebuilding the algorithm model.

# %%
faster_architecture = SurfaceCodeCostModel(
    code_distance=10,
    physical_cycle_time_seconds=sp.Float("5e-8"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=1,
    factory_count=8,
    physical_qubits_per_factory=2500,
    factory_cycles_per_toffoli=2,
)
relifted_baseline = baseline.with_cost_model(faster_architecture)

architecture_comparison = compare_ftqc_resource_estimates(
    baseline,
    relifted_baseline,
    quantities=("physical_qubits", "runtime_seconds"),
)

for row in architecture_comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4))

assert relifted_baseline.logical_qubits == baseline.logical_qubits
assert relifted_baseline.toffoli_gates == baseline.toffoli_gates
assert relifted_baseline.resource_values()[FTQCResourceQuantity.CODE_DISTANCE] == 10
assert sp.Abs(architecture_comparison[0].ratio - sp.Rational(350, 691)) < sp.Float(
    "1e-12"
)
assert sp.Abs(architecture_comparison[1].ratio - sp.Rational(10, 21)) < sp.Float(
    "1e-12"
)

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
#   target precision, truncation error, QPE iterations, non-Clifford counts,
#   logical depth, physical qubits, and runtime separately.
# - Qamomile keeps those quantities in algorithmic metadata so the circuit IR
#   remains backend-neutral.
# - Accuracy budgets split a total target precision into representation
#   truncation error and QPE precision before estimates are compared.
# - Formula provenance exposes the symbolic derivation behind important
#   resource quantities.
# - Surface-code assumptions can be modeled separately and converted into the
#   cost model consumed by chemistry estimators.
# - Surface-code distance can be selected from a logical failure budget before
#   lifting logical resources to physical qubits and runtime.
# - Architecture quantities such as code distance remain on each estimate so
#   reports can audit the physical-resource assumptions.
# - Estimates carry research references so reports can audit which paper
#   motivated a symbolic model.
# - FTQC estimates can be viewed as common logical `ResourceEstimate` objects
#   when reports need the same shape as circuit-level estimates.
# - Existing logical estimates can be relifted under new architecture
#   assumptions without rebuilding the algorithm estimate.
# - `compare_ftqc_resource_estimates` turns symbolic estimates into reviewable
#   savings tables without hard-coding a particular chemistry factorization.
# - `summarize_ftqc_resource_comparison` groups those rows into smaller,
#   larger, unchanged, and symbolic changes for design review.
