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
    BlockEncodingResource,
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCAccuracyBudget,
    FTQCResourceAggregationRule,
    FTQCResourceCategory,
    FTQCResourceConstraint,
    FTQCResourcePlan,
    FTQCResourcePlanStep,
    FTQCResourceProfile,
    FTQCResourceQuantity,
    QPEStatePreparationBudget,
    SurfaceCodeCostModel,
    SurfaceCodeDistanceBudget,
    build_ftqc_resource_comparison_report,
    compare_ftqc_resource_estimates,
    default_ftqc_resource_aggregation_rule,
    evaluate_ftqc_resource_constraints,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    ftqc_resource_profile_quantities,
    iter_ftqc_research_signals,
    iter_ftqc_resource_profile_specs,
    iter_ftqc_resource_quantity_specs,
    plan_qubitized_qpe_from_block_encoding,
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
# chemistry work. Qamomile exposes that mapping as structured research signals
# so reports can show why a quantity is being measured:
#
# | Research direction | Cost signal for Qamomile |
# | --- | --- |
# | Symmetry-compressed double factorization reduces the Hamiltonian 1-norm and Toffoli count for qubitized chemistry QPE ([arXiv:2403.03502](https://arxiv.org/abs/2403.03502)). | Track `lambda_norm`, QPE iterations, per-walk Toffoli cost, and total Toffoli count separately. |
# | Simultaneous symmetry shifts and tensor factorizations reduce the block-encoding scaling constant for electronic Hamiltonians ([arXiv:2412.01338](https://arxiv.org/abs/2412.01338)). | Treat Hamiltonian normalization as representation metadata, not as an emitted-circuit property. |
# | Symmetry-adapted filtering can increase state-preparation success probability before QPE ([arXiv:2601.08533](https://arxiv.org/abs/2601.08533)). | Track success probability, expected QPE repetitions, filtering overhead, T gates, and runtime. |
# | Early-FTQC single-ancilla QPE with unitary weight concentration targets smaller physical-qubit budgets and limited depth ([arXiv:2603.22778](https://arxiv.org/abs/2603.22778)). | Track T gates, logical depth, logical space-time volume, physical qubits, runtime, physical qubit-seconds, and architecture knobs in addition to Toffoli-native qubitization costs. |
#
# These are modeling quantities. They should be validated against each paper's
# assumptions before being used as a molecule-specific resource claim.

# %%
research_signals = [signal.to_dict() for signal in iter_ftqc_research_signals()]
for signal in research_signals:
    print(signal["reference_key"], "->", ", ".join(signal["quantities"][:4]))

signal_by_key = {signal["reference_key"]: signal for signal in research_signals}
assert "lambda_norm" in signal_by_key["arXiv:2403.03502"]["quantities"]
assert "state_preparation_success_probability" in signal_by_key["arXiv:2601.08533"][
    "quantities"
]
assert "t_gates" in signal_by_key["arXiv:2603.22778"]["quantities"]

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
assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.RUNTIME_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.CODE_DISTANCE.value in {row["quantity"] for row in catalog}

# %% [markdown]
# Standard review profiles provide reusable quantity bundles for recurring
# questions such as "what is the space-time footprint?" They keep reports from
# hand-copying ad hoc column lists, and comparison helpers accept them directly.

# %%
profile_catalog = {
    spec.profile: spec.to_dict() for spec in iter_ftqc_resource_profile_specs()
}
space_time_quantities = ftqc_resource_profile_quantities(
    FTQCResourceProfile.SPACETIME
)

print(profile_catalog[FTQCResourceProfile.SPACETIME]["description"])
print(profile_catalog[FTQCResourceProfile.SPACETIME]["quantities"])

assert FTQCResourceProfile.CHEMISTRY_QPE in profile_catalog
assert space_time_quantities[-1] == FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS

# %% [markdown]
# ## Compositional Algorithm Plans
#
# Before a new FTQC algorithm has a concrete Qamomile circuit, model it as a
# sequence of abstract subroutines. A resource plan composes canonical
# quantities with explicit aggregation rules: counts, depth, runtime, and
# space-time costs add across sequential steps; qubit footprints use the peak;
# problem metadata and architecture knobs must stay consistent.

# %%
prepare_step = FTQCResourcePlanStep(
    "prepare_filter",
    {
        FTQCResourceQuantity.LOGICAL_QUBITS: 12,
        FTQCResourceQuantity.TOFFOLI_GATES: 7,
        FTQCResourceQuantity.LOGICAL_DEPTH: 11,
        FTQCResourceQuantity.TARGET_PRECISION: sp.Float("0.001"),
    },
    repetitions=3,
    label="State preparation and filtering",
)
qpe_step = FTQCResourcePlanStep(
    "phase_estimation",
    {
        "logical_qubits": 18,
        "toffoli_gates": 100,
        "logical_depth": 120,
        "runtime_seconds": 20,
        "physical_qubits": 4000,
        "target_precision": sp.Float("0.001"),
    },
    repetitions=2,
)
resource_plan = FTQCResourcePlan(
    (prepare_step, qpe_step),
    title="Filtered QPE plan",
)

for row in resource_plan.to_quantity_table():
    print(row["quantity"], row["aggregation"], row["value"])

plan_values = resource_plan.resource_values()
assert plan_values[FTQCResourceQuantity.TOFFOLI_GATES] == 221
assert plan_values[FTQCResourceQuantity.LOGICAL_DEPTH] == 273
assert plan_values[FTQCResourceQuantity.LOGICAL_QUBITS] == 18
assert default_ftqc_resource_aggregation_rule("logical_qubits") == (
    FTQCResourceAggregationRule.PEAK
)

# %% [markdown]
# The same plan object exposes `resource_values()`, so comparison and budget
# helpers can consume it exactly like a chemistry estimate. If a resource
# should be combined as a peak across steps, override the plan-level rule for
# that quantity; each step's own repetition scaling still applies first.

# %%
parallel_runtime_plan = FTQCResourcePlan(
    (prepare_step, qpe_step),
    aggregation={"runtime_seconds": FTQCResourceAggregationRule.PEAK},
)
plan_budget = evaluate_ftqc_resource_constraints(
    resource_plan,
    (
        FTQCResourceConstraint("logical_qubits", 20),
        FTQCResourceConstraint("toffoli_gates", 200),
    ),
    title="Toy plan budget",
)

assert parallel_runtime_plan.resource_values()[FTQCResourceQuantity.RUNTIME_SECONDS] == 40
assert plan_budget.satisfied[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
assert plan_budget.violated[0].quantity == FTQCResourceQuantity.TOFFOLI_GATES

# %% [markdown]
# A block-encoding model can also produce a plan directly. The plan separates
# the reusable block-encoding contract from the repeated qubitized-walk step,
# so reviewers can inspect PREPARE, SELECT, reflection, walk cost, and QPE
# iterations before a loader circuit exists.

# %%
block_encoding = BlockEncodingResource(
    system_qubits=40,
    normalization=sp.Float("2.0e5"),
    select_cost_toffoli=4_000,
    prepare_cost_toffoli=500,
    reflection_cost_toffoli=100,
    ancilla_qubits=80,
    name="toy_lcu_loader",
)
block_plan = plan_qubitized_qpe_from_block_encoding(
    block_encoding,
    precision=sp.Float("0.0015"),
    qpe_register_qubits=12,
)
block_plan_values = block_plan.resource_values()

for step in block_plan.to_dict()["steps"]:
    print(step["name"], step["repetitions"])

for formula in block_plan.to_dict()["formulas"]:
    print(formula["quantity"], "<-", ", ".join(formula["depends_on"]))

assert block_plan.steps[0].name == "block_encoding_contract"
assert block_plan.steps[1].name == "qubitized_walk_qpe"
assert block_plan_values[FTQCResourceQuantity.WALK_COST_TOFFOLI] == 5100
assert block_plan_values[FTQCResourceQuantity.LOGICAL_QUBITS] == 132
assert block_plan_values[FTQCResourceQuantity.TOFFOLI_GATES] == (
    block_plan_values[FTQCResourceQuantity.QPE_ITERATIONS] * 5100
)
assert block_plan.reference_keys() == ("arXiv:1610.06546",)
assert block_plan.to_dict()["formulas"][0]["quantity"] == "walk_cost_toffoli"

# %% [markdown]
# Plan provenance is intentionally separate from circuit lowering. A review can
# inspect derivation formulas and citation keys before deciding whether a
# PREPARE/SELECT implementation should become a concrete Qamomile kernel.

# %%
block_plan_dict = block_plan.to_dict()

print(block_plan_dict["reference_keys"])
print(block_plan_dict["steps"][0]["formulas"][0]["description"])

assert block_plan_dict["reference_keys"] == ["arXiv:1610.06546"]
assert block_plan_dict["steps"][0]["formulas"][0]["reference_keys"] == [
    "arXiv:1610.06546"
]

# %% [markdown]
# State-preparation success probability can be layered onto the same plan. This
# keeps trial-state overlap or filtering assumptions visible without requiring
# a concrete state-preparation circuit.

# %%
preparation_budget = QPEStatePreparationBudget(
    success_probability=sp.Rational(1, 5),
    state_preparation_toffoli=100,
    state_preparation_t_gates=20,
    state_preparation_logical_depth=50,
    description="symmetry-filtered trial state",
)
filtered_block_plan = preparation_budget.apply_to_plan(block_plan)
filtered_block_values = filtered_block_plan.resource_values()

for step in filtered_block_plan.to_dict()["steps"]:
    print(step["name"], step["repetitions"])

assert filtered_block_values[FTQCResourceQuantity.QPE_REPETITIONS] == 5
assert filtered_block_values[FTQCResourceQuantity.QPE_ITERATIONS] == (
    block_plan_values[FTQCResourceQuantity.QPE_ITERATIONS] * 5
)
assert filtered_block_values[FTQCResourceQuantity.TOFFOLI_GATES] == (
    (block_plan_values[FTQCResourceQuantity.TOFFOLI_GATES] + 100) * 5
)
assert filtered_block_values[FTQCResourceQuantity.LOGICAL_DEPTH] == (
    (block_plan_values[FTQCResourceQuantity.LOGICAL_DEPTH] + 50) * 5
)

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
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

for row in comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert comparison[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison[0].ratio == sp.Float("0.5")
assert sp.Abs(comparison[1].ratio - sp.Float("0.55")) < sp.Float("1e-12")
assert comparison[4].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
assert compressed.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS] == (
    compressed.physical_qubits * compressed.runtime_seconds
)

# %% [markdown]
# For design reviews, the summary helper groups the same rows by whether the
# candidate is smaller, larger, unchanged, or still symbolic under the current
# assumptions. The first `smaller` rows are the largest numeric reductions.

# %%
comparison_summary = summarize_ftqc_resource_comparison(
    baseline,
    compressed,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

for row in comparison_summary.smaller:
    print("smaller:", row.label, "by", sp.N(row.reduction, 4))
for row in comparison_summary.larger:
    print("larger:", row.label, "by", sp.N(-row.reduction, 4))

assert comparison_summary.smaller[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison_summary.larger[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
assert any(
    row.quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
    for row in comparison_summary.larger
)
assert comparison_summary.symbolic == ()

# %% [markdown]
# When you need a self-contained artifact for a design review, build a report.
# It preserves the labels, selected profile, ordered rows, and grouped summary
# counts together. The report can also produce prioritized findings: largest
# savings first, then the largest resource tradeoffs.

# %%
comparison_report = build_ftqc_resource_comparison_report(
    baseline,
    compressed,
    title="Toy factorization comparison",
    baseline_label="THC-style",
    candidate_label="Compressed",
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)
report_rows = comparison_report.to_row_table()
review_findings = comparison_report.to_review_findings(
    max_improvements=2,
    max_tradeoffs=2,
)

print(comparison_report.to_dict()["title"])
print(comparison_report.to_dict()["counts"])
for finding in review_findings:
    print(finding.direction, "-", finding.headline)

assert comparison_report.profile == FTQCResourceProfile.SPACETIME
assert report_rows[0]["baseline_label"] == "THC-style"
assert report_rows[0]["candidate_label"] == "Compressed"
assert report_rows[0]["quantity"] == "qpe_iterations"
assert review_findings[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison_report.to_dict()["findings"][0]["direction"] == "smaller"

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
# ## Budget Constraints
#
# Early-FTQC studies often ask a different review question: does an estimate fit
# inside a physical-qubit or runtime budget? A budget report evaluates the same
# canonical resource values against explicit constraints and keeps symbolic
# margins undecided until architecture assumptions are bound.

# %%
budget_report = evaluate_ftqc_resource_constraints(
    uwc_trotter,
    (
        FTQCResourceConstraint(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            100_000,
            label="Early-FTQC physical-qubit budget",
        ),
        FTQCResourceConstraint(
            FTQCResourceQuantity.RUNTIME_SECONDS,
            60 * 60,
            label="One-hour runtime budget",
        ),
        FTQCResourceConstraint(
            FTQCResourceQuantity.LOGICAL_DEPTH,
            plain_trotter.logical_depth,
            label="No worse than plain Trotter depth",
        ),
    ),
    title="Synthetic early-FTQC budget",
)

for result in budget_report.results:
    print(result.status, result.label, "margin:", sp.N(result.margin, 4))

assert budget_report.satisfied[0].quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
assert budget_report.violated[0].quantity == FTQCResourceQuantity.RUNTIME_SECONDS
assert budget_report.to_dict()["counts"] == {
    "satisfied": 2,
    "violated": 1,
    "symbolic": 0,
}

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
#   logical depth, logical space-time volume, physical qubits, runtime, and
#   physical qubit-seconds separately.
# - `iter_ftqc_research_signals` maps research directions to the canonical
#   quantities that Qamomile reports.
# - `FTQCResourceProfile` gives reusable quantity bundles such as the
#   space-time profile and can be passed directly to comparison helpers.
# - `FTQCResourcePlan` composes abstract FTQC subroutines before a concrete
#   circuit implementation exists.
# - `plan_qubitized_qpe_from_block_encoding` turns a block-encoding contract
#   into a PREPARE/SELECT/reflection/QPE resource plan.
# - `QPEStatePreparationBudget.apply_to_plan` layers success probability and
#   per-attempt preparation overhead onto an abstract QPE plan.
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
# - `build_ftqc_resource_comparison_report` packages labels, profile, rows,
#   prioritized findings, and grouped counts for review artifacts.
# - `evaluate_ftqc_resource_constraints` checks estimates against explicit
#   physical-qubit, runtime, depth, or other resource budgets.
