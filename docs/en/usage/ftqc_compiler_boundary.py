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
# tags: [usage, resource-estimation, circuit-compilation]
# ---
#
# # FTQC Compiler Boundary
#
# This page explains where fault-tolerant algorithm work belongs in Qamomile's
# compiler stack.
# It is a design checklist for contributors who need to decide whether a new
# FTQC idea should become circuit IR, a resource-estimation workload, a
# backend emitter feature, or documentation.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc
import qamomile.observable as qm_o
import qamomile.resource_estimation as qre
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## The Boundary
#
# Qamomile has two different, connected FTQC surfaces.
#
# | Layer | What it should express | What should stay out |
# | --- | --- | --- |
# | Compiler IR | The quantum program: registers, control flow, measurements, and high-level composite gates such as QPE/IQFT | Paper-specific chemistry tables, factory schedules, or report schemas |
# | Resource workloads | Algorithm contracts: Hamiltonian normalization, QPE precision budgets, Trotter samples, block-encoding costs | Backend-specific qubit placement or emitted SDK instructions |
# | Physical lifts | Explicit architecture assumptions such as code distance, cycle time, factory throughput, and active volume | A hidden choice of hardware model or a fixed report format |
#
# This split keeps the IR abstract.
# The compiler should preserve high-level meaning until a backend has enough
# information to lower it, while resource-estimation objects can compare
# paper-scale assumptions before a circuit is worth emitting.

# %% [markdown]
# ## QPE as IR
#
# QPE is an FTQC building block, but it should still enter the compiler as an
# abstract quantum program.
# The counting register, controlled unitary, IQFT, and fixed-point measurement
# are compiler concepts.
# The chemistry-specific workload that decided how many QPE iterations are
# needed is not part of this IR.

# %%


@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply the unitary whose phase QPE estimates."""
    return qmc.p(q, theta)


@qmc.qkernel
def phase_probe(theta: qmc.Float) -> qmc.Float:
    counting = qmc.qubit_array(3, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    phase = qmc.qpe(target, counting, phase_gate, theta=theta)
    return qmc.measure(phase)


def flatten_ops(ops):
    """Return operations including nested control-flow bodies."""
    flat = []
    for op in ops:
        flat.append(op)
        for nested in getattr(op, "nested_op_lists", lambda: [])():
            flat.extend(flatten_ops(nested))
    return flat


transpiler = QiskitTranspiler()
block = transpiler.to_block(phase_probe, parameters=["theta"])
block = transpiler.inline(block)
ops = flatten_ops(block.operations)
composites = [op for op in ops if isinstance(op, CompositeGateOperation)]

print(
    {
        "block_kind": block.kind.name,
        "composite_types": [op.gate_type for op in composites],
    }
)

assert block.kind.name == "AFFINE"
assert any(op.gate_type == CompositeGateType.IQFT for op in composites)

# %% [markdown]
# The test above checks the compiler boundary directly.
# Inlining removed ordinary qkernel calls, but the IQFT inside QPE remains a
# composite operation.
# A backend with native support can emit it directly; another backend can use
# a decomposition at emit time.
#
# That is the desired FTQC compiler shape: the IR says "this is IQFT", not
# "here is one particular backend's sequence of primitive gates."

# %% [markdown]
# ## Algorithm Workloads as Resource Contracts
#
# Resource estimation starts from a different question.
# A recent paper may claim a better Hamiltonian representation, a smaller
# effective normalization, or an active-volume scheduling improvement.
# Those claims should first become symbolic resource quantities.

# %%
hamiltonian = qre.summarize_pauli_hamiltonian(4 * qm_o.Z(0) + 2 * qm_o.X(0) * qm_o.X(1))

block_encoding = qre.BlockEncodingResource(
    system_qubits=hamiltonian.n_qubits,
    normalization=hamiltonian.lambda_norm,
    prepare_cost_toffoli=8,
    select_cost_toffoli=20,
    reflection_cost_toffoli=4,
    ancilla_qubits=1,
    name="toy block encoding",
)
workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    hamiltonian,
    block_encoding,
    qpe_register_qubits=3,
)
logical = qre.estimate_qubitized_qpe_resources_from_workload(workload, precision=1)
values = qre.resource_values_from_estimate(logical)

for quantity in (
    qre.ResourceQuantity.LAMBDA_NORM,
    qre.ResourceQuantity.WALK_COST_TOFFOLI,
    qre.ResourceQuantity.QPE_ITERATIONS,
    qre.ResourceQuantity.NON_CLIFFORD_COUNT,
):
    spec = qre.describe_resource_quantity(quantity)
    print({"quantity": spec.quantity.value, "category": spec.category.value})

assert values["logical_qubits"] == 6
assert math.isclose(float(values["qpe_iterations"]), 6.0, rel_tol=0.0, abs_tol=1e-12)
assert math.isclose(
    float(values["non_clifford_count"]), 240.0, rel_tol=0.0, abs_tol=1e-12
)

# %% [markdown]
# These values are not backend instructions.
# They are reviewable assumptions that can be compared before choosing a
# backend circuit representation.
# This is why Qamomile keeps `HamiltonianQPEWorkload`,
# `TrotterQPEWorkload`, and `BlockEncodingResource` outside
# `qamomile.circuit.ir`.

# %% [markdown]
# ## Physical Lifts as Explicit Assumptions
#
# A physical estimate should say which architecture assumptions were used.
# The compact surface-code model below is intentionally symbolic enough to
# keep those assumptions visible.

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=1e-6,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=2,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)
physical = qre.estimate_physical_resources(logical, surface_code)
physical_values = physical.resource_values()

for name in (
    "physical_qubits",
    "runtime_seconds",
    "depth_limited_runtime_seconds",
    "non_clifford_limited_runtime_seconds",
):
    print({name: physical_values[name]})

assert physical_values["code_distance"] == 5
assert physical_values["physical_qubits"] == 2300
assert float(physical_values["non_clifford_limited_runtime_seconds"]) > 0

# %% [markdown]
# The physical lift does not change the compiler IR.
# It prices the logical estimate under a named model.
# If a future PR adds a richer surface-code or factory model, it should expose
# its assumptions as canonical quantities before it adds a report format.

# %% [markdown]
# ## Contributor Checklist
#
# Use this checklist when adding an FTQC algorithm or resource-estimation
# feature:
#
# - If the feature changes the meaning of a quantum program, add or reuse an
#   abstract IR operation, composite gate, or frontend helper.
# - If the feature changes a paper-level algorithmic cost, add or reuse a
#   resource workload and canonical quantities.
# - If the feature changes hardware pricing, add or reuse an explicit physical
#   lift model.
# - If the feature only changes how results are reviewed, keep it in docs or a
#   later report layer until the underlying quantities are stable.
#
# ## Summary
#
# In this notebook, we learned:
#
# - QPE belongs in the compiler as abstract quantum structure.
# - FTQC chemistry claims should first be represented as resource quantities
#   and workload contracts.
# - Physical estimates should keep architecture assumptions visible instead of
#   hiding them behind a report schema.
