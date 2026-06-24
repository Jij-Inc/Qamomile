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
# tags: [tutorial, resource-estimation]
# ---
#
# # Resource Estimation
#
# Before running a quantum kernel on real hardware, you may want to know its required resources, such as qubit count and gate count. Or, you may want to know the resource requirements of a quantum kernel you defined in the first place. Qamomile's `estimate_resources()` fills this need **without executing the qkernel**. It works with both concrete and symbolic (parameterized) qkernels.
#
# This chapter covers:
#
# - Basic resource estimation for fixed qkernels
# - Symbolic resource estimation for parameterized qkernels
# - The full `ResourceEstimate` field reference
# - Scaling analysis with `.substitute()`
# - FTQC-oriented comparison of logical and physical resource proxies

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o
import qamomile.resource_estimation as qre

# %% [markdown]
# ## Estimating Resources of a Fixed QKernel
#
# For a qkernel with no parameters, `estimate_resources()` returns concrete numbers.


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
assert est.qubits == 3
print("total gates:", est.gates.total)
assert est.gates.total == 3
print("single-qubit gates:", est.gates.single_qubit)
assert est.gates.single_qubit == 1
print("two-qubit gates:", est.gates.two_qubit)
assert est.gates.two_qubit == 2

# %% [markdown]
# ## Symbolic Resource Estimation
#
# When a qkernel has unbound parameters (like `n: qmc.UInt`), `estimate_resources()` returns **SymPy expressions** that show how costs scale with the parameter. This lets you analyze scaling without picking a specific value.


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q = qmc.h(q)
    q = qmc.ry(q, theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4, fold_loops=False)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
assert str(est.qubits) == "n"
print("total gates:", est.gates.total)
assert str(est.gates.total) == "3*n - 1"
print("single-qubit gates:", est.gates.single_qubit)
assert str(est.gates.single_qubit) == "2*n"
print("two-qubit gates:", est.gates.two_qubit)
assert str(est.gates.two_qubit) == "n - 1"
print("rotation gates:", est.gates.rotation_gates)
assert str(est.gates.rotation_gates) == "n"
print("parameters:", est.parameters)
assert set(est.parameters.keys()) == {"n"}

# %% [markdown]
# The output contains SymPy expressions like `n` for qubits and `3*n - 1` for total gates. These are exact — not approximations. When an estimate is still symbolic, `audit_resource_value_symbols()` shows which canonical resource quantities still depend on unresolved symbols.

# %%
symbol_rows = qre.audit_resource_value_symbols(est, include_resolved=False)
for row in symbol_rows:
    print(row.to_dict())

symbolic_quantities = {row.quantity for row in symbol_rows}
assert qre.ResourceQuantity.LOGICAL_QUBITS in symbolic_quantities
assert qre.ResourceQuantity.LOGICAL_DEPTH in symbolic_quantities
assert qre.ResourceQuantity.LOGICAL_SPACETIME_VOLUME in symbolic_quantities
assert all(row.is_symbolic for row in symbol_rows)

# %% [markdown]
# The inverse view is often more useful when reviewing an FTQC estimate: `audit_resource_value_drivers()` groups affected quantities by symbol. In the example below, the single size symbol `n` drives all three selected logical quantities.

# %%
driver_rows = qre.audit_resource_value_drivers(
    est,
    quantities=(
        qre.ResourceQuantity.LOGICAL_QUBITS,
        qre.ResourceQuantity.LOGICAL_DEPTH,
        qre.ResourceQuantity.LOGICAL_SPACETIME_VOLUME,
    ),
)
for row in driver_rows:
    print(row.to_dict())

assert len(driver_rows) == 1
assert driver_rows[0].symbol == "n"
assert driver_rows[0].quantity_count == 3

# %% [markdown]
# ## `ResourceEstimate` Fields Reference
#
# | Field | Description |
# |-------|------------|
# | `est.qubits` | Logical qubit count |
# | `est.gates.total` | Total gate count |
# | `est.gates.single_qubit` | Single-qubit gates |
# | `est.gates.two_qubit` | Two-qubit gates |
# | `est.gates.multi_qubit` | Multi-qubit gates (3+ qubits) |
# | `est.gates.t_gates` | T-gate count |
# | `est.gates.clifford_gates` | Clifford gate count |
# | `est.gates.rotation_gates` | Rotation gate count |
# | `est.gates.oracle_calls` | Oracle call counts (dict by name) |
# | `est.parameters` | Dict of symbol names → SymPy symbols |
#
# All fields are SymPy expressions. For fixed qkernels they evaluate to plain integers.

# %% [markdown]
# ## Scaling Analysis with `.substitute()`
#
# The symbolic expressions tell you the *formula*, but often you want concrete numbers at specific sizes. Use `.substitute()` to evaluate:

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )
    assert int(c.gates.total) == 3 * n_val - 1
    assert int(c.gates.two_qubit) == n_val - 1

# %% [markdown]
# ## Comparing FTQC Cost Drivers
#
# Fault-tolerant algorithms are usually compared before they are lowered to a backend circuit. Qamomile keeps that layer separate: use `qamomile.resource_estimation` to describe the Hamiltonian, estimate algorithm-level logical work, compare canonical quantities, and only then lift the result through an architecture model.
#
# In this toy example, each candidate starts as a block-encoding contract. The contract records the Hamiltonian normalization, PREPARE/SELECT/reflection costs, ancilla footprint, QPE readout register size, and optional representation error. Those quantities are enough to build a Hamiltonian QPE workload without committing to a backend circuit. When a target precision is supplied, the workload can also expose the remaining QPE precision budget as a canonical quantity.
#
# :::{note}
# Recent chemistry resource-estimation work, such as [symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502) and [unitary weight concentration](https://arxiv.org/abs/2603.22778), often compares algorithms through the Hamiltonian normalization, representation error, walk-operator cost, Toffoli count, logical qubits, runtime, and space-time volume. This tutorial does not reproduce those papers; it shows the Qamomile resource quantities needed to build that kind of comparison.
# :::

# %% [markdown]
# `ResourceReviewProfile` groups canonical quantities by review task. A workload profile records the symbols that drive Hamiltonian QPE, while logical and physical outcome profiles are suitable for `compare_resource_values()` and `pareto_resource_values()`.

# %%
workload_profile = qre.describe_resource_review_profile(
    qre.ResourceReviewProfile.HAMILTONIAN_QPE_WORKLOAD
)
logical_profile = qre.describe_resource_review_profile(
    qre.ResourceReviewProfile.FTQC_LOGICAL_OUTCOMES
)
physical_profile = qre.describe_resource_review_profile(
    qre.ResourceReviewProfile.FTQC_PHYSICAL_OUTCOMES
)
trotter_profile = qre.describe_resource_review_profile(
    qre.ResourceReviewProfile.TROTTER_QPE_WORKLOAD
)
for profile in (workload_profile, trotter_profile, logical_profile, physical_profile):
    print(profile.to_dict())

assert qre.ResourceQuantity.LAMBDA_NORM in workload_profile.quantities
assert qre.ResourceQuantity.EFFECTIVE_LAMBDA_NORM in trotter_profile.quantities
assert qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS in physical_profile.quantities

# %% [markdown]
# Chemistry estimates often start from an OpenFermion `QubitOperator`. Qamomile does not need to own that object: it only needs the OpenFermion-style `terms` mapping so the Hamiltonian can be summarized into resource quantities.


# %%
class OpenFermionQubitOperatorStub:
    terms = {
        ((0, "Z"),): 4,
        ((1, "Z"),): 3,
        ((0, "X"), (1, "X")): 2,
    }


openfermion_workload = qre.qubitized_qpe_workload_from_openfermion(
    OpenFermionQubitOperatorStub(),
    walk_cost_toffoli=100,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    qpe_register_qubits=2,
    description="OpenFermion sparse Pauli LCU",
)
openfermion_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    openfermion_workload,
    precision=1,
)

print(openfermion_workload.hamiltonian.to_dict())
print(openfermion_logical)
assert openfermion_workload.hamiltonian.n_pauli_terms == 3
assert sp.simplify(openfermion_workload.hamiltonian.lambda_norm - 9) == 0
assert sp.simplify(openfermion_logical.gates.oracle_calls["qpe_iterations"] - 9) == 0

# %%
hamiltonian = 4 * qm_o.Z(0) + 3 * qm_o.Z(1) + 2 * qm_o.X(0) * qm_o.X(1)
summary = qre.summarize_pauli_hamiltonian(hamiltonian)

baseline_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=summary.lambda_norm,
    prepare_cost_toffoli=20,
    select_cost_toffoli=70,
    reflection_cost_toffoli=10,
    ancilla_qubits=1,
    name="sparse Pauli LCU",
)
candidate_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=sp.Rational(2, 5) * summary.lambda_norm,
    prepare_cost_toffoli=15,
    select_cost_toffoli=45,
    reflection_cost_toffoli=5,
    ancilla_qubits=2,
    name="compressed factorization",
)

baseline_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    baseline_block,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    qpe_register_qubits=2,
    description="sparse Pauli LCU",
)
candidate_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    candidate_block,
    representation=qre.HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
    second_factor_rank=4,
    qpe_register_qubits=2,
    representation_error=sp.Rational(1, 10),
    description="compressed factorization",
)

baseline_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    baseline_workload,
    precision=1,
)
candidate_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    candidate_workload,
    precision=1,
)

logical_rows = qre.compare_resource_values(
    baseline_logical,
    candidate_logical,
    quantities=logical_profile.quantities,
)
for row in logical_rows:
    print(row.to_dict())

precision_rows = qre.compare_resource_values(
    baseline_workload.resource_values_for_precision(1),
    candidate_workload.resource_values_for_precision(1),
    quantities=(
        qre.ResourceQuantity.TARGET_PRECISION,
        qre.ResourceQuantity.ALGORITHMIC_PRECISION,
    ),
)
for row in precision_rows:
    print(row.to_dict())

assert (
    candidate_logical.gates.oracle_calls["qpe_iterations"]
    < baseline_logical.gates.oracle_calls["qpe_iterations"]
)
assert candidate_logical.gates.multi_qubit < baseline_logical.gates.multi_qubit
assert candidate_workload.qpe_register_qubits == 2
assert candidate_workload.algorithmic_precision(1) == sp.Rational(9, 10)
assert precision_rows[0].ratio == 1
assert precision_rows[1].candidate == sp.Rational(9, 10)

# %% [markdown]
# A different algorithm family may trade block-encoding oracles for product-formula time evolution and fewer logical qubits. `TrotterQPEWorkload` records those assumptions explicitly. Paper tables often report the effective Hamiltonian weight after concentration, so `from_effective_lambda_norm()` derives the multiplicative factor while keeping the original Hamiltonian summary visible. The tutorial numbers are intentionally tiny, but the visible quantities are the same ones you would audit in a paper-scale model.

# %%
uwc_workload = qre.TrotterQPEWorkload.from_effective_lambda_norm(
    summary,
    effective_lambda_norm=1,
    trotter_steps_per_sample=2,
    samples=10,
    randomized_compilation_factor=sp.Rational(1, 2),
    rotation_synthesis_t_gates=2,
    description="unitary weight concentration toy model",
)
uwc_rows = qre.evaluate_resource_values(
    uwc_workload.resource_values_for_precision(1),
    quantities=trotter_profile.quantities,
)
for row in uwc_rows:
    print(row.to_dict())

uwc_logical = qre.estimate_trotter_qpe_resources_from_workload(
    uwc_workload,
    precision=1,
)
uwc_vs_baseline_rows = qre.compare_resource_values(
    baseline_logical,
    uwc_logical,
    quantities=logical_profile.quantities,
)
for row in uwc_vs_baseline_rows:
    print(row.to_dict())

assert sp.Abs(uwc_workload.effective_lambda_norm - 1) < sp.Float("1e-12")
assert uwc_logical.qubits == summary.n_qubits + 1
assert sp.Abs(uwc_logical.gates.oracle_calls["qpe_iterations"] - 1) < sp.Float("1e-12")
assert uwc_logical.gates.t_gates < baseline_logical.gates.multi_qubit

# %% [markdown]
# You can also keep the reported effective lambda symbolic until review time. This makes the break between "no concentration" and a reported concentration table explicit without changing the workload structure.

# %%
lambda_eff = sp.symbols("lambda_eff", positive=True)
symbolic_uwc_workload = qre.TrotterQPEWorkload.from_effective_lambda_norm(
    summary,
    effective_lambda_norm=lambda_eff,
    trotter_steps_per_sample=2,
    samples=10,
    randomized_compilation_factor=sp.Rational(1, 2),
    rotation_synthesis_t_gates=2,
    description="symbolic unitary weight concentration",
)
symbolic_uwc_logical = qre.estimate_trotter_qpe_resources_from_workload(
    symbolic_uwc_workload,
    precision=1,
)
lambda_scenario_rows = qre.evaluate_resource_value_scenarios(
    symbolic_uwc_logical,
    {
        "no concentration": {"lambda_eff": summary.lambda_norm},
        "reported concentration": {"lambda_eff": 1},
    },
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.T_GATES,
        qre.ResourceQuantity.LOGICAL_QUBITS,
    ),
)
for row in lambda_scenario_rows:
    print(row.to_dict())

lambda_scenario_values = {
    (row.scenario, row.quantity): row.value for row in lambda_scenario_rows
}
assert len(lambda_scenario_rows) == 6
assert all(row.is_resolved for row in lambda_scenario_rows)
assert (
    sp.simplify(
        lambda_scenario_values[
            ("reported concentration", qre.ResourceQuantity.QPE_ITERATIONS)
        ]
        - 1
    )
    == 0
)
assert (
    lambda_scenario_values[("reported concentration", qre.ResourceQuantity.T_GATES)]
    < lambda_scenario_values[("no concentration", qre.ResourceQuantity.T_GATES)]
)

# %% [markdown]
# `compare_resource_values()` accepts logical `ResourceEstimate` objects directly. For a physical proxy, provide a compact architecture model. The estimate below is not a hardware design; it is a consistent way to compare candidates under the same surface-code-style assumptions.

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=1e-6,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=1,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)

baseline_physical = qre.estimate_physical_resources(baseline_logical, surface_code)
candidate_physical = qre.estimate_physical_resources(candidate_logical, surface_code)

physical_rows = qre.compare_resource_values(
    baseline_physical,
    candidate_physical,
    quantities=physical_profile.quantities,
)
for row in physical_rows:
    print(row.to_dict())

assert candidate_physical.runtime_seconds < baseline_physical.runtime_seconds
assert (
    candidate_physical.resource_values()["physical_qubit_seconds"]
    < baseline_physical.resource_values()["physical_qubit_seconds"]
)

# %% [markdown]
# When more than two candidates are involved, `pareto_resource_values()` keeps the non-dominated tradeoffs visible. The slower candidate below uses the same logical algorithm and qubit overhead as the candidate above, but assumes a slower cycle time. It is dominated because it uses the same physical qubits and takes longer.

# %%
slow_surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=2e-6,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=1,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)
slow_candidate_physical = qre.estimate_physical_resources(
    candidate_logical,
    slow_surface_code,
)
pareto_rows = qre.pareto_resource_values(
    {
        "baseline": baseline_physical,
        "candidate": candidate_physical,
        "slow candidate": slow_candidate_physical,
    },
    quantities=physical_profile.quantities,
)
for row in pareto_rows:
    print(row.to_dict())

frontier_labels = {row.label for row in pareto_rows if row.is_frontier}
assert frontier_labels == {"baseline", "candidate"}
assert any(
    row.label == "slow candidate" and row.dominated_by == ("candidate",)
    for row in pareto_rows
)

# %% [markdown]
# If some architecture assumptions are still symbolic, `evaluate_resource_value_scenarios()` turns them into a compact scenario table. This is useful when the algorithm estimate is fixed, but code distance, cycle time, or factory assumptions are still design variables.

# %%
d, cycle_time = sp.symbols("d cycle_time", positive=True)
symbolic_surface_code = qre.SurfaceCodeCostModel(
    code_distance=d,
    physical_cycle_time_seconds=cycle_time,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=1,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)
symbolic_physical = qre.estimate_physical_resources(
    candidate_logical,
    symbolic_surface_code,
)
scenario_rows = qre.evaluate_resource_value_scenarios(
    symbolic_physical,
    {
        "distance-5 fast cycle": {"d": 5, "cycle_time": sp.Float("5e-7")},
        "distance-7 nominal": {"d": 7, "cycle_time": sp.Float("1e-6")},
    },
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in scenario_rows:
    print(row.to_dict())

assert len(scenario_rows) == 6
assert all(row.is_resolved for row in scenario_rows)

# %% [markdown]
# ## Summary
#
# - `estimate_resources()` reports qubit and gate costs without executing.
# - For parameterized qkernels, results are SymPy expressions showing exact scaling.
# - Use `.substitute(n=...)` to evaluate at specific sizes and check feasibility.
# - Use `qamomile.resource_estimation` to compare FTQC algorithm candidates by canonical logical and physical quantities, keep Pareto tradeoffs visible, audit which symbols drive those quantities, then evaluate remaining architecture symbols across scenarios.
#
# **Next**: [Execution Models](06_execution_models.ipynb) — `sample()` vs `run()`, observables, and bit ordering.
