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
# tags: [usage, resource-estimation]
# ---
#
# # Designing FTQC resource reviews
#
# This page explains how to turn fault-tolerant algorithm claims into Qamomile resource quantities.
# It is a design map for reviewers: start from a paper-level workload claim, keep the logical algorithm model separate from hardware assumptions, and then compare candidates through canonical quantities.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.resource_estimation as qre

# %% [markdown]
# ## Research Signals
#
# Recent FTQC chemistry papers often improve resource estimates by changing the Hamiltonian representation, the QPE implementation, or the architecture assumptions.
# Qamomile does not bake any one paper's table into its API.
# Instead, it exposes the quantities that make those tables auditable.
#
# | Research signal | Example paper | Qamomile review surface |
# | --- | --- | --- |
# | Reduce Hamiltonian normalization before qubitized QPE. | [Symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502) reduces the 1-norm that drives QPE iterations and Toffoli cost. | `lambda_norm`, `representation_error`, `walk_cost_toffoli`, `qpe_iterations`, `t_gates` |
# | Replace the block-encoding workload with an early-FTQC Trotter workload. | [Unitary weight concentration](https://arxiv.org/abs/2603.22778) reports reduced effective Hamiltonian weight for a single-ancilla, Trotter-based QPE setting. | `effective_lambda_norm`, `unitary_weight_factor`, `trotter_steps_per_sample`, `pauli_rotations`, `rotation_synthesis_t_gates` |
# | Move from logical resources to architecture bottlenecks. | Surface-code and magic-state-factory estimates, such as [Gidney and Ekerå's factoring estimate](https://arxiv.org/abs/1905.09749), make runtime depend on both logical depth and non-Clifford throughput. | `physical_qubits`, `depth_limited_runtime_seconds`, `non_clifford_limited_runtime_seconds`, `physical_qubit_seconds` |
#
# This split keeps three contracts explicit:
#
# - **Problem contract**: Hamiltonian size, locality, and normalization.
# - **Algorithm contract**: QPE iterations, oracle or product-formula work, precision budget, and non-Clifford work.
# - **Architecture contract**: code distance, cycle time, factory footprint, and non-Clifford throughput.

# %% [markdown]
# ## Quantity Profiles
#
# `ResourceReviewProfile` names the quantity sets that Qamomile expects reviewers to inspect.
# The profile is not a score; it is a stable checklist.

# %%
profiles = {
    "qubitized workload": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.HAMILTONIAN_QPE_WORKLOAD
    ),
    "Trotter workload": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.TROTTER_QPE_WORKLOAD
    ),
    "logical outcomes": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.FTQC_LOGICAL_OUTCOMES
    ),
    "physical outcomes": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.FTQC_PHYSICAL_OUTCOMES
    ),
}

for name, profile in profiles.items():
    print(name, [quantity.value for quantity in profile.quantities])

assert qre.ResourceQuantity.LAMBDA_NORM in profiles["qubitized workload"].quantities
assert (
    qre.ResourceQuantity.EFFECTIVE_LAMBDA_NORM
    in profiles["Trotter workload"].quantities
)
assert (
    qre.ResourceQuantity.NON_CLIFFORD_COUNT in profiles["logical outcomes"].quantities
)
assert (
    qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS
    in profiles["physical outcomes"].quantities
)

# %% [markdown]
# ## Logical Workload Comparison
#
# A useful review first compares logical algorithm contracts.
# The example below compares a sparse Pauli-LCU qubitized-QPE workload with a toy unitary-weight-concentration-style Trotter-QPE workload.
# The numbers are deliberately small so that the page remains executable; the inspected quantities are the same ones used for paper-scale estimates.

# %%
summary = qre.PauliHamiltonianResource(
    n_qubits=4,
    n_pauli_terms=12,
    lambda_norm=12,
    max_locality=2,
)

qubitized_workload = qre.HamiltonianQPEWorkload(
    summary,
    walk_cost_toffoli=120,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    qpe_register_qubits=3,
    representation_error=sp.Rational(1, 10),
    description="sparse Pauli LCU baseline",
)
uwc_workload = qre.TrotterQPEWorkload.from_effective_lambda_norm(
    summary,
    effective_lambda_norm=2,
    trotter_steps_per_sample=2,
    samples=12,
    randomized_compilation_factor=sp.Rational(1, 2),
    rotation_synthesis_t_gates=2,
    description="unitary weight concentration toy model",
)

qubitized_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    qubitized_workload,
    precision=1,
)
uwc_logical = qre.estimate_trotter_qpe_resources_from_workload(
    uwc_workload,
    precision=1,
)

logical_rows = qre.compare_resource_values(
    qubitized_logical,
    uwc_logical,
    quantities=profiles["logical outcomes"].quantities,
)
for row in logical_rows:
    print(row.to_dict())

assert qubitized_workload.algorithmic_precision(1) == sp.Rational(9, 10)
assert uwc_workload.unitary_weight_factor == sp.Rational(1, 6)
assert uwc_logical.qubits < qubitized_logical.qubits
non_clifford_row = next(
    row
    for row in logical_rows
    if row.quantity == qre.ResourceQuantity.NON_CLIFFORD_COUNT
)
assert non_clifford_row.candidate < non_clifford_row.baseline

# %% [markdown]
# ## Architecture Bottlenecks
#
# After the logical comparison is clear, use an architecture model to expose bottlenecks.
# `SurfaceCodeCostModel` is intentionally compact: it records explicit assumptions and derives the generic FTQC knobs used by `FTQCCostModel`.
# It does not choose a code distance or factory layout from an error budget.

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=7,
    physical_cycle_time_seconds=sp.Rational(1, 1_000_000),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=2,
    physical_qubits_per_factory=500,
    factory_cycles_per_non_clifford=4,
)

qubitized_physical = qre.estimate_physical_resources(qubitized_logical, surface_code)
uwc_physical = qre.estimate_physical_resources(uwc_logical, surface_code)

physical_rows = qre.compare_resource_values(
    qubitized_physical,
    uwc_physical,
    quantities=profiles["physical outcomes"].quantities,
)
for row in physical_rows:
    print(row.to_dict())

uwc_values = uwc_physical.resource_values()
assert uwc_values["runtime_seconds"] == sp.Max(
    uwc_values["depth_limited_runtime_seconds"],
    uwc_values["non_clifford_limited_runtime_seconds"],
)
assert (
    uwc_values["physical_qubit_seconds"]
    < qubitized_physical.resource_values()["physical_qubit_seconds"]
)

# %% [markdown]
# ## Symbolic Scenarios
#
# Early FTQC reviews often leave architecture knobs symbolic until a hardware assumption is chosen.
# Scenario rows keep those assumptions visible without changing the logical estimate.

# %%
distance = sp.symbols("distance", positive=True)
cycle_time = sp.symbols("cycle_time", positive=True)
symbolic_surface_code = qre.SurfaceCodeCostModel(
    code_distance=distance,
    physical_cycle_time_seconds=cycle_time,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=2,
    physical_qubits_per_factory=500,
    factory_cycles_per_non_clifford=4,
)
symbolic_physical = qre.estimate_physical_resources(
    uwc_logical,
    symbolic_surface_code,
)

driver_rows = qre.audit_resource_value_drivers(
    symbolic_physical,
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in driver_rows:
    print(row.to_dict())

scenario_rows = qre.evaluate_resource_value_scenarios(
    symbolic_physical,
    {
        "fast small distance": {"distance": 5, "cycle_time": sp.Rational(1, 1_000_000)},
        "slow large distance": {"distance": 9, "cycle_time": sp.Rational(2, 1_000_000)},
    },
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in scenario_rows:
    print(row.to_dict())

assert {row.symbol for row in driver_rows} == {"cycle_time", "distance"}
assert len(scenario_rows) == 6
assert all(row.is_resolved for row in scenario_rows)

# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - FTQC resource reviews should keep problem, algorithm, and architecture contracts separate.
# - Recent chemistry-resource papers can be audited through canonical quantities such as Hamiltonian normalization, effective lambda, non-Clifford work, logical qubits, runtime bottlenecks, and physical qubit-seconds.
# - Qamomile's resource-estimation API is designed to compare those quantities before adding any report, snapshot, or manifest layer.
