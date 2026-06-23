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
# tags: [algorithm, chemistry, resource-estimation, simulation]
# ---
#
# # FTQC Chemistry Resource Estimation
#
# This notebook shows how to compare fault-tolerant quantum chemistry resource
# models with Qamomile. It focuses on algorithm-level quantities that matter
# before a full logical circuit is available: Hamiltonian normalization,
# phase-estimation iterations, Toffoli or T counts, logical qubits, physical
# qubits, logical space-time volume, physical qubit-seconds, and runtime
# proxies.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp
from openfermion import QubitOperator

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCAccuracyBudget,
    FTQCResourceProfile,
    FTQCResourceQuantity,
    HamiltonianResourceReduction,
    QPEStatePreparationBudget,
    SurfaceCodeDistanceBudget,
    block_encoding_from_chemistry_model,
    compare_ftqc_resource_estimates,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_qubitized_qpe_from_block_encoding,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    ftqc_resource_profile_quantities,
    iter_ftqc_research_signals,
    iter_ftqc_resource_profile_specs,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
    summarize_openfermion_qubit_operator,
    summarize_pauli_hamiltonian,
)

# %% [markdown]
# ## Background
#
# Fault-tolerant chemistry algorithms are usually compared through different
# resource quantities than NISQ variational examples. For qubitized QPE, the
# Hamiltonian block-encoding normalization controls the number of walk calls,
# while the per-walk implementation controls the Toffoli count. Recent
# chemistry proposals try to reduce these costs by changing the Hamiltonian
# representation rather than changing only backend-level gate decompositions.
#
# Examples include symmetry-compressed double factorization
# ([arXiv:2403.03502](https://arxiv.org/abs/2403.03502)), simultaneous symmetry
# shifts and tensor factorizations
# ([arXiv:2412.01338](https://arxiv.org/abs/2412.01338)), and early-FTQC
# single-ancilla QPE with unitary weight concentration
# ([arXiv:2603.22778](https://arxiv.org/abs/2603.22778)). State-preparation
# improvements such as symmetry-adapted filtering
# ([arXiv:2601.08533](https://arxiv.org/abs/2601.08533)) can also change the
# expected number of QPE attempts. Because these proposals trade algorithmic
# work, logical depth, logical qubits, and hardware runtime differently,
# space-time quantities such as logical qubit-layers and physical qubit-seconds
# are useful review targets. The estimators below are deliberately symbolic.
# They let you test how a proposed representation changes the cost-driving
# quantities before committing to a concrete Hamiltonian-loading circuit.

# %% [markdown]
# ## Problem Settings
#
# We will compare three scenarios for the same active-space scale:
#
# 1. a tensor-hypercontraction-like qubitized QPE baseline,
# 2. a symmetry-compressed double-factorization-style qubitized QPE estimate,
# 3. an early-FTQC single-ancilla Trotter QPE estimate with a unitary-weight
#    concentration factor.
#
# The numbers are intentionally small and synthetic so the notebook is fast and
# reviewable. They should be read as a workflow demonstration, not as a claim
# about a particular molecule.

# %%
n_spin_orbitals = 40
target_precision = sp.Float("0.0016")  # about chemical accuracy in Hartree
accuracy_budget = FTQCAccuracyBudget(
    target_precision=target_precision,
    truncation_error=sp.Float("1e-4"),
)
qpe_precision = accuracy_budget.qpe_precision

distance_budget = SurfaceCodeDistanceBudget(
    physical_error_rate=sp.Float("1e-3"),
    threshold_error_rate=sp.Float("1e-2"),
    target_logical_failure_probability=sp.Float("1e-9"),
    logical_operation_budget=1000,
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

assert distance_budget.code_distance == 21
assert architecture.physical_qubits_per_logical == 882
assert architecture.factory_qubits == 20000
assert sp.Abs(qpe_precision - sp.Float("0.0015")) < sp.Float("1e-12")

# %% [markdown]
# ## Resource Quantities
#
# Symbolic FTQC estimates should keep problem quantities, logical work, and
# physical assumptions separate. Qamomile exposes the canonical quantity
# catalog so downstream reports can use stable keys while still showing
# reader-facing labels, units, and modeling layers.

# %%
quantity_catalog = [
    spec.to_dict()
    for spec in iter_ftqc_resource_quantity_specs()
    if spec.quantity.value
    in {
        "lambda_norm",
        "target_precision",
        "state_preparation_success_probability",
        "qpe_repetitions",
        "qpe_iterations",
        "toffoli_gates",
        "t_gates",
        "logical_qubits",
        "logical_spacetime_volume",
        "physical_qubits",
        "physical_qubit_seconds",
        "runtime_seconds",
        "logical_error_rate",
        "code_distance",
    }
]

for row in quantity_catalog:
    print(row["quantity"], row["unit"], row["category"])

assert {row["quantity"] for row in quantity_catalog} == {
    "lambda_norm",
    "target_precision",
    "state_preparation_success_probability",
    "qpe_repetitions",
    "qpe_iterations",
    "toffoli_gates",
    "t_gates",
    "logical_qubits",
    "logical_spacetime_volume",
    "physical_qubits",
    "physical_qubit_seconds",
    "runtime_seconds",
    "logical_error_rate",
    "code_distance",
}

# %% [markdown]
# Qamomile also provides standard review profiles: small reusable bundles of
# quantities for common audit questions. The profile does not compute new
# resources; it names the columns to compare and can be passed directly to
# comparison helpers.

# %%
profile_catalog = {
    spec.profile: spec.to_dict() for spec in iter_ftqc_resource_profile_specs()
}
space_time_quantities = ftqc_resource_profile_quantities(
    FTQCResourceProfile.SPACETIME
)

print(profile_catalog[FTQCResourceProfile.SPACETIME]["quantities"])

assert space_time_quantities == (
    FTQCResourceQuantity.LOGICAL_QUBITS,
    FTQCResourceQuantity.LOGICAL_DEPTH,
    FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
    FTQCResourceQuantity.PHYSICAL_QUBITS,
    FTQCResourceQuantity.RUNTIME_SECONDS,
    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
)

# %% [markdown]
# The research-signal catalog maps recent papers to the quantities they ask a
# Qamomile model to expose, plus the review profiles that should be inspected
# first. This keeps the tutorial grounded in a small, inspectable contract
# instead of prose-only claims.

# %%
signal_by_key = {
    signal.reference_key: signal for signal in iter_ftqc_research_signals()
}
scdf_signal = signal_by_key["arXiv:2403.03502"]
early_ftqc_signal = signal_by_key["arXiv:2603.22778"]

print(scdf_signal.title)
print([quantity.value for quantity in scdf_signal.quantities])
print([profile.value for profile in scdf_signal.profiles])
print(early_ftqc_signal.title)
print([quantity.value for quantity in early_ftqc_signal.quantities])
print([profile.value for profile in early_ftqc_signal.profiles])

assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in scdf_signal.quantities
assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME in early_ftqc_signal.quantities
assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in early_ftqc_signal.quantities
assert FTQCResourceProfile.SPACETIME in early_ftqc_signal.profiles

# %% [markdown]
# We start from a small Qamomile observable so the workflow has the same shape
# as a real chemistry pipeline: build or import a Hamiltonian, summarize its LCU
# quantities, then pass that summary into an FTQC estimator. The rescaling below
# makes the toy Hamiltonian stand in for a larger active-space model without
# claiming that these coefficients describe a particular molecule.

# %%
toy_hamiltonian = 0.5 * qm_o.Z(0) + 0.25 * qm_o.X(1) * qm_o.X(2) + 0.125
toy_summary = summarize_pauli_hamiltonian(
    toy_hamiltonian,
    n_spin_orbitals=n_spin_orbitals,
    source="toy_pauli_lcu",
)
scaled_summary = toy_summary.with_lambda_scale(
    sp.Float("2.0e5") / toy_summary.lambda_norm,
    source="scaled_toy_pauli_lcu",
)

assert toy_summary.n_pauli_terms == 2
assert toy_summary.constant == sp.Float("0.125")
assert sp.simplify(scaled_summary.lambda_norm - sp.Float("2.0e5")) == 0

# %% [markdown]
# If your chemistry preprocessing already produces an OpenFermion
# `QubitOperator`, summarize it through the same boundary. This keeps the
# electronic-structure toolchain outside Qamomile's compiler IR while preserving
# the cost-driving Hamiltonian metadata.

# %%
openfermion_hamiltonian = (
    QubitOperator("Z0", 0.5)
    + QubitOperator("X1 X2", 0.25)
    + QubitOperator((), 0.125)
)
openfermion_summary = summarize_openfermion_qubit_operator(
    openfermion_hamiltonian,
    n_spin_orbitals=n_spin_orbitals,
    source="openfermion_toy_pauli_lcu",
)

assert openfermion_summary.n_pauli_terms == toy_summary.n_pauli_terms
assert openfermion_summary.lambda_norm == toy_summary.lambda_norm
assert openfermion_summary.constant == toy_summary.constant

# %% [markdown]
# ## Qubitized QPE Comparison
#
# Qamomile keeps the chemistry factorization cost external to the IR. The
# model-driven estimator accepts a Hamiltonian summary plus the
# representation-dependent one-walk Toffoli cost:
#
# ```text
# qpe_iterations = lambda_norm / qpe_precision
# toffoli_gates = qpe_iterations * walk_cost_toffoli
# ```
#
# This keeps the model honest: changing Hamiltonian normalization, changing the
# walk circuit, and changing physical architecture remain separate design
# choices.

# %%
thc_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary,
        method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
        walk_cost_toffoli=sp.Integer(4_000),
        description="THC-style scaled toy model",
    )
)
scdf_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary.with_lambda_scale(
            sp.Float("0.5"),
            source="SCDF-style scaled toy model",
        ),
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        walk_cost_toffoli=sp.Integer(4_400),
        second_factor_rank=9,
        description="SCDF-style scaled toy model",
    )
)

thc = estimate_qubitized_chemistry_qpe_from_model(
    thc_model,
    precision=qpe_precision,
    cost_model=cost_model,
)
scdf = estimate_qubitized_chemistry_qpe_from_model(
    scdf_model,
    precision=qpe_precision,
    cost_model=cost_model,
)

assert scdf.qpe_iterations < thc.qpe_iterations
assert scdf.toffoli_gates < thc.toffoli_gates
assert scdf.resource_values()[FTQCResourceQuantity.CODE_DISTANCE] == 21
assert scdf.to_dict()["architecture_values"]["code_distance"] == "21"

print("THC Toffoli gates:", sp.N(thc.toffoli_gates, 4))
print("SCDF-style Toffoli gates:", sp.N(scdf.toffoli_gates, 4))
print("Toy Pauli terms:", toy_summary.n_pauli_terms)
print("SCDF-style logical qubits:", scdf.logical_qubits)

qubitized_quantities = (
    FTQCResourceQuantity.QPE_ITERATIONS,
    FTQCResourceQuantity.TOFFOLI_GATES,
    *space_time_quantities,
)

for row in scdf.to_quantity_table():
    if row["quantity"] in {quantity.value for quantity in qubitized_quantities}:
        print(row["label"], row["value"], row["unit"])

qubitized_savings = compare_ftqc_resource_estimates(
    thc,
    scdf,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)
for row in qubitized_savings:
    print(
        row.label,
        "ratio:",
        sp.N(row.ratio, 4),
        "reduction:",
        sp.N(row.reduction, 4),
    )

assert qubitized_savings[0].quantity.value == "qpe_iterations"
assert qubitized_savings[0].ratio == sp.Float("0.5")
assert sp.Abs(qubitized_savings[1].ratio - sp.Float("0.55")) < sp.Float("1e-12")
assert qubitized_savings[4].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME

qubitized_summary = summarize_ftqc_resource_comparison(
    thc,
    scdf,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

assert qubitized_summary.smaller[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert qubitized_summary.larger[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
assert any(
    row.quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
    for row in qubitized_summary.larger
)

# %% [markdown]
# ## State-Preparation Success Budget
#
# QPE cost also depends on whether the prepared state has enough overlap with
# the target eigenstate. A symmetry filter or better trial-state preparation
# can raise the success probability, while adding a small per-attempt overhead.
# `QPEStatePreparationBudget` keeps that assumption explicit and scales the
# expected repeated QPE work.

# %%
weak_overlap_budget = QPEStatePreparationBudget(
    success_probability=sp.Rational(1, 8),
    description="unfiltered trial state",
)
symmetry_filtered_budget = QPEStatePreparationBudget(
    success_probability=sp.Rational(1, 2),
    state_preparation_t_gates=sp.Integer(1_000_000),
    state_preparation_logical_depth=sp.Integer(1_000_000),
    description="symmetry-filtered trial state",
)

weak_overlap_scdf = weak_overlap_budget.apply(scdf)
symmetry_filtered_scdf = symmetry_filtered_budget.apply(scdf)

preparation_savings = compare_ftqc_resource_estimates(
    weak_overlap_scdf,
    symmetry_filtered_scdf,
    quantities=(
        FTQCResourceQuantity.QPE_REPETITIONS,
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

for row in preparation_savings:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert weak_overlap_scdf.resource_values()[FTQCResourceQuantity.QPE_REPETITIONS] == 8
assert (
    symmetry_filtered_scdf.resource_values()[FTQCResourceQuantity.QPE_REPETITIONS] == 2
)
assert symmetry_filtered_scdf.resource_values()[
    FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
] == (symmetry_filtered_scdf.logical_qubits * symmetry_filtered_scdf.logical_depth)
assert symmetry_filtered_scdf.qpe_iterations == scdf.qpe_iterations * 2
assert symmetry_filtered_scdf.toffoli_gates < weak_overlap_scdf.toffoli_gates
assert "state_preparation_success_probability" in symmetry_filtered_scdf.to_dict()[
    "algorithm_values"
]

# %% [markdown]
# The same chemistry model can also be converted into a block-encoding
# contract. This is the point where a future loader implementation can split
# cost into PREPARE, SELECT, reflection, and workspace pieces without changing
# the chemistry summary or the compiler IR.

# %%
scdf_block = block_encoding_from_chemistry_model(
    scdf_model,
    prepare_cost_toffoli=sp.Integer(200),
    reflection_cost_toffoli=sp.Integer(50),
    name="scdf_block_contract",
)
scdf_block_estimate = estimate_qubitized_qpe_from_block_encoding(
    scdf_block,
    precision=qpe_precision,
    qpe_register_qubits=12,
    cost_model=cost_model,
)

print(scdf_block.to_dict())
assert scdf_block.walk_cost_toffoli == scdf_model.walk_cost_toffoli + 450
assert scdf_block_estimate.target_precision == qpe_precision
assert scdf_block_estimate.logical_qubits == scdf_block.logical_qubits + 12

# %% [markdown]
# ## Early-FTQC Trotter QPE
#
# Early-FTQC proposals may favor shallower single-ancilla QPE and Pauli
# rotations over qubitized walks when qubits and depth are tightly constrained.
# The resource reduction below represents a spectrally invariant Hamiltonian
# transformation, such as unitary weight concentration, that reduces the
# cost-driving effective weight while staying separate from circuit lowering.

# %%
uwc_reduction = HamiltonianResourceReduction(
    lambda_norm_factor=sp.Float("0.1"),
    description="unitary weight concentration",
)
plain_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=qpe_precision,
    trotter_steps_per_sample=8,
    samples=128,
    cost_model=cost_model,
)

uwc_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=qpe_precision,
    trotter_steps_per_sample=8,
    samples=128,
    randomized_compilation_factor=sp.Float("0.5"),
    rotation_synthesis_t_gates=3,
    resource_reduction=uwc_reduction,
    cost_model=cost_model,
)

assert uwc_trotter.qpe_iterations < plain_trotter.qpe_iterations
assert uwc_trotter.logical_depth < plain_trotter.logical_depth

print("Plain Trotter QPE depth proxy:", sp.N(plain_trotter.logical_depth, 4))
print("UWC-style Trotter QPE depth proxy:", sp.N(uwc_trotter.logical_depth, 4))
print("UWC-style T gates:", sp.N(uwc_trotter.t_gates, 4))

trotter_savings = compare_ftqc_resource_estimates(
    plain_trotter,
    uwc_trotter,
    quantities=(FTQCResourceQuantity.QPE_ITERATIONS,),
    profile=FTQCResourceProfile.SPACETIME,
)
for row in trotter_savings:
    print(
        row.label,
        "ratio:",
        sp.N(row.ratio, 4),
        "reduction:",
        sp.N(row.reduction, 4),
    )

assert trotter_savings[0].ratio == sp.Float("0.1")
assert trotter_savings[2].ratio == sp.Float("0.05")
assert trotter_savings[3].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
assert "lambda_norm=0.100000000000000" in uwc_trotter.assumptions[
    "resource_reduction_factors"
]

# %% [markdown]
# ## Result
#
# We can put the estimates into a compact table. The important point is that
# each column has a distinct design meaning: changing the Hamiltonian
# representation should affect `qpe_iterations` and per-step cost, while
# changing the hardware model should affect `physical_qubits`, runtime, and
# physical qubit-seconds.

# %%
rows = [
    ("THC qubitized QPE", thc),
    ("SCDF-style qubitized QPE", scdf),
    ("Plain Trotter QPE", plain_trotter),
    ("UWC-style Trotter QPE", uwc_trotter),
]

for name, estimate in rows:
    print(
        name,
        {
            "logical_qubits": sp.N(estimate.logical_qubits, 4),
            "physical_qubits": sp.N(estimate.physical_qubits, 4),
            "qpe_iterations": sp.N(estimate.qpe_iterations, 4),
            "toffoli_gates": sp.N(estimate.toffoli_gates, 4),
            "t_gates": sp.N(estimate.t_gates, 4),
            "logical_spacetime_volume": sp.N(
                estimate.resource_values()[FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME],
                4,
            ),
            "runtime_seconds": sp.N(estimate.runtime_seconds, 4),
            "physical_qubit_seconds": sp.N(
                estimate.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS],
                4,
            ),
        },
    )

assert scdf.physical_qubits > thc.physical_qubits
assert uwc_trotter.physical_qubits == plain_trotter.physical_qubits
assert uwc_trotter.resource_values()[
    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS
] < plain_trotter.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS]

# %% [markdown]
# ## Summary
#
# In this notebook, we:
#
# - Separated FTQC chemistry resource quantities into Hamiltonian
#   normalization, QPE iterations, non-Clifford counts, logical qubits,
#   logical space-time volume, physical qubits, runtime proxies, and physical
#   qubit-seconds.
# - Allocated a total target precision into truncation error and QPE precision
#   before building comparable estimates.
# - Modeled state-preparation success probability as an explicit expected
#   repetition factor for QPE resources.
# - Compared qubitized QPE representations without lowering the Qamomile IR
#   into backend-specific chemistry loading circuits.
# - Selected surface-code distance from an explicit logical failure budget
#   before lifting logical resources to physical resources.
# - Kept architecture quantities such as code distance attached to each
#   resource estimate for later report auditing.
# - Converted a chemistry QPE model into a block-encoding contract so PREPARE,
#   SELECT, reflection, and workspace costs can be reviewed separately.
# - Demonstrated how a unitary-weight concentration factor can be modeled as a
#   cost-driver reduction for early-FTQC single-ancilla Trotter QPE.
# - Connected recent FTQC chemistry research signals to the canonical
#   quantities that the tutorial compares.
# - Used a standard `FTQCResourceProfile` so space-time comparisons have the
#   same quantity set across examples.
