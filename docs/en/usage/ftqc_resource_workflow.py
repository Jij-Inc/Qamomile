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
# # FTQC Resource Workflow
#
# This page shows how to compare fault-tolerant quantum chemistry resource
# estimates with Qamomile's resource-estimation primitives.
# The examples are intentionally small.
# They demonstrate the quantities and API boundaries needed to review recent
# paper-scale estimates without turning the usage page into a paper
# reproduction.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.observable as qm_o
import qamomile.resource_estimation as qre

# %% [markdown]
# ## Workflow Boundary
#
# FTQC resource estimates are usually compared before an algorithm is lowered
# to a backend circuit.
# Qamomile keeps that layer as a symbolic resource model:
#
# | Paper-level claim | Qamomile quantity to inspect |
# | --- | --- |
# | Hamiltonian compression reduces the QPE work signal | `lambda_norm`, `effective_lambda_norm`, `representation_error` |
# | The walk or time-evolution routine is cheaper | `walk_cost_toffoli`, `pauli_rotations`, `t_gates`, `non_clifford_count` |
# | The method spends memory to reduce work | `logical_qubits`, `system_qubits`, `block_encoding_ancilla_qubits`, `qpe_register_qubits` |
# | The architecture lift changes the bottleneck | `physical_qubits`, `runtime_seconds`, `physical_qubit_seconds`, `active_volume` |
#
# :::{note}
# Recent examples include
# [symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502),
# [unitary weight concentration](https://arxiv.org/abs/2603.22778), and
# [active-volume-style resource estimates](https://arxiv.org/abs/2501.06165).
# The code below uses toy numbers to exercise the same resource quantities.
# :::

# %% [markdown]
# ## Hamiltonian Summary
#
# Start by reducing the Hamiltonian to resource quantities.
# The summary records the encoded width, the number of non-identity Pauli
# terms, the Hamiltonian normalization, and the maximum locality.

# %%
hamiltonian = 4 * qm_o.Z(0) + 3 * qm_o.Z(1) + 2 * qm_o.X(0) * qm_o.X(1)
summary = qre.summarize_pauli_hamiltonian(hamiltonian)

print(summary.to_dict())

assert summary.n_qubits == 2
assert summary.n_pauli_terms == 3
assert sp.Abs(summary.lambda_norm - 9) < sp.Float("1e-12")
assert summary.max_locality == 2

# %% [markdown]
# ## Qubitized QPE Candidates
#
# A block-encoding contract is enough to compare two qubitized-QPE candidates.
# The baseline below keeps the original Hamiltonian normalization.
# The compressed candidate models a smaller normalization and cheaper
# PREPARE/SELECT/reflection costs, while spending one extra ancilla qubit and a
# small representation-error budget.

# %%
target_precision = sp.Integer(1)

baseline_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=summary.lambda_norm,
    prepare_cost_toffoli=20,
    select_cost_toffoli=70,
    reflection_cost_toffoli=10,
    ancilla_qubits=1,
    name="sparse Pauli LCU",
)
compressed_block = qre.BlockEncodingResource(
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
)
compressed_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    compressed_block,
    representation=qre.HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
    second_factor_rank=4,
    qpe_register_qubits=2,
    representation_error=sp.Rational(1, 10),
)

baseline_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    baseline_workload,
    precision=target_precision,
)
compressed_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    compressed_workload,
    precision=target_precision,
)

logical_rows = qre.compare_resource_values(
    baseline_logical,
    compressed_logical,
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.LOGICAL_QUBITS,
        qre.ResourceQuantity.NON_CLIFFORD_COUNT,
    ),
)

for row in logical_rows:
    print(row.to_dict())

assert compressed_logical.qubits == baseline_logical.qubits + 1
assert (
    compressed_logical.gates.oracle_calls["qpe_iterations"]
    < (baseline_logical.gates.oracle_calls["qpe_iterations"])
)
assert compressed_logical.gates.multi_qubit < baseline_logical.gates.multi_qubit

# %% [markdown]
# ## Precision Budget
#
# A representation error should be visible beside the target precision.
# `resource_values_for_precision()` exposes both the requested budget and the
# precision left for phase estimation.

# %%
precision_rows = qre.compare_resource_values(
    baseline_workload.resource_values_for_precision(target_precision),
    compressed_workload.resource_values_for_precision(target_precision),
    quantities=(
        qre.ResourceQuantity.TARGET_PRECISION,
        qre.ResourceQuantity.ALGORITHMIC_PRECISION,
    ),
)
compressed_precision_values = compressed_workload.resource_values_for_precision(
    target_precision
)

for row in precision_rows:
    print(row.to_dict())

assert precision_rows[0].ratio == 1
assert sp.Abs(precision_rows[1].candidate - sp.Rational(9, 10)) < sp.Float("1e-12")
assert compressed_precision_values["representation_error"] == sp.Rational(1, 10)

# %% [markdown]
# ## Unitary-Weight-Style Trotter QPE
#
# Some estimates report an effective Hamiltonian weight after concentration
# rather than a block-encoding normalization.
# `TrotterQPEWorkload.from_effective_lambda_norm()` keeps the original
# Hamiltonian summary and derives the multiplicative weight factor.

# %%
uwc_workload = qre.TrotterQPEWorkload.from_effective_lambda_norm(
    summary,
    effective_lambda_norm=1,
    trotter_steps_per_sample=2,
    samples=10,
    randomized_compilation_factor=sp.Rational(1, 2),
    rotation_synthesis_t_gates=2,
    description="unitary-weight-style toy estimate",
)
uwc_logical = qre.estimate_trotter_qpe_resources_from_workload(
    uwc_workload,
    precision=target_precision,
)

uwc_rows = qre.compare_resource_values(
    baseline_logical,
    uwc_logical,
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.LOGICAL_QUBITS,
        qre.ResourceQuantity.NON_CLIFFORD_COUNT,
    ),
)

for row in uwc_rows:
    print(row.to_dict())

assert sp.Abs(uwc_workload.unitary_weight_factor - sp.Rational(1, 9)) < sp.Float(
    "1e-12"
)
assert uwc_logical.qubits == summary.n_qubits + 1
assert sp.Abs(uwc_logical.gates.oracle_calls["qpe_iterations"] - 1) < sp.Float("1e-12")
assert uwc_logical.gates.t_gates < baseline_logical.gates.multi_qubit

# %% [markdown]
# ## Architecture Lift
#
# Logical estimates can be lifted through an explicit architecture model.
# The surface-code-style model reports physical qubits, runtime components,
# and physical qubit-seconds.
# The active-volume model separately demonstrates operation-volume accounting
# for algorithms whose cost is better described by active resources.

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=sp.Float("1e-6"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=1,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)
compressed_physical = qre.estimate_physical_resources(
    compressed_logical,
    surface_code,
)

active_volume_model = qre.ActiveVolumeCostModel(
    active_volume_per_logical_gate=2,
    active_volume_per_non_clifford=1,
    active_volume_throughput_per_second=100,
)
uwc_active_volume = qre.estimate_active_volume_resources(
    uwc_logical,
    active_volume_model,
)

physical_values = compressed_physical.resource_values()
active_values = uwc_active_volume.resource_values()

print(
    {
        "physical_qubits": physical_values["physical_qubits"],
        "runtime_seconds": physical_values["runtime_seconds"],
        "physical_qubit_seconds": physical_values["physical_qubit_seconds"],
        "active_volume": active_values["active_volume"],
        "active_volume_runtime_seconds": active_values["active_volume_runtime_seconds"],
    }
)

assert physical_values["runtime_seconds"] == sp.Max(
    physical_values["depth_limited_runtime_seconds"],
    physical_values["non_clifford_limited_runtime_seconds"],
)
assert sp.Abs(active_values["active_volume"] - 180) < sp.Float("1e-12")
assert sp.Abs(
    active_values["active_volume_runtime_seconds"] - sp.Rational(9, 5)
) < sp.Float("1e-12")

# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - Use `PauliHamiltonianResource` and workload objects to keep FTQC estimates
#   symbolic and architecture-independent.
# - Compare candidates through canonical quantities such as `lambda_norm`,
#   `qpe_iterations`, `logical_qubits`, `non_clifford_count`, and
#   `representation_error`.
# - Lift logical estimates only after the algorithm-level comparison, using a
#   shared architecture model for physical or active-volume proxies.
