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
import qamomile.circuit as qmc
import qamomile.observable as qm_o
import qamomile.resource_estimation as qre
import sympy as sp

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
# The output contains SymPy expressions like `n` for qubits and `3*n - 1` for total gates. These are exact — not approximations.

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
# In this toy example, the candidate workload has a smaller Hamiltonian normalization and a cheaper walk operator. The numbers are placeholders for an algorithm study, but the quantities are the same ones you would track when comparing block-encoding or Hamiltonian-representation choices.
#
# :::{note}
# Recent chemistry resource-estimation work, such as [symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502), often compares algorithms through the Hamiltonian normalization, walk-operator cost, Toffoli count, logical qubits, runtime, and space-time volume. This tutorial does not reproduce that paper; it shows the Qamomile resource quantities needed to build that kind of comparison.
# :::

# %%
hamiltonian = 4 * qm_o.Z(0) + 3 * qm_o.Z(1) + 2 * qm_o.X(0) * qm_o.X(1)
summary = qre.summarize_pauli_hamiltonian(hamiltonian)

baseline_workload = qre.HamiltonianQPEWorkload(
    hamiltonian=summary,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    walk_cost_toffoli=120,
    description="sparse Pauli LCU",
)
candidate_workload = qre.HamiltonianQPEWorkload(
    hamiltonian=summary.with_lambda_scale(
        sp.Rational(2, 5),
        source="compressed representation",
    ),
    representation=qre.HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
    second_factor_rank=4,
    walk_cost_toffoli=80,
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
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.NON_CLIFFORD_COUNT,
        qre.ResourceQuantity.LOGICAL_QUBITS,
    ),
)
for row in logical_rows:
    print(row.to_dict())

assert (
    candidate_logical.gates.oracle_calls["qpe_iterations"]
    < baseline_logical.gates.oracle_calls["qpe_iterations"]
)
assert candidate_logical.gates.multi_qubit < baseline_logical.gates.multi_qubit

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
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in physical_rows:
    print(row.to_dict())

assert candidate_physical.runtime_seconds < baseline_physical.runtime_seconds
assert (
    candidate_physical.resource_values()["physical_qubit_seconds"]
    < baseline_physical.resource_values()["physical_qubit_seconds"]
)

# %% [markdown]
# ## Summary
#
# - `estimate_resources()` reports qubit and gate costs without executing.
# - For parameterized qkernels, results are SymPy expressions showing exact scaling.
# - Use `.substitute(n=...)` to evaluate at specific sizes and check feasibility.
# - Use `qamomile.resource_estimation` to compare FTQC algorithm candidates by canonical logical and physical quantities.
#
# **Next**: [Execution Models](06_execution_models.ipynb) — `sample()` vs `run()`, observables, and bit ordering.
