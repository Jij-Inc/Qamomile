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
# tags: [usage, resource-estimation, primitive]
# ---
#
# # Block-Encoding Resource Estimates
#
# Block-encoding is the interface between a Hamiltonian representation and
# qubitized FTQC algorithms. This notebook shows how to model PREPARE, SELECT,
# reflection, and QPE readout costs separately before committing to any backend
# circuit decomposition.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

from qamomile.circuit.estimator.algorithmic import (
    BlockEncodingResource,
    FTQCCostModel,
    FTQCResourceQuantity,
    compare_ftqc_resource_estimates,
    estimate_qubitized_qpe_from_block_encoding,
)

# %% [markdown]
# ## Workflow
#
# A qubitized walk is usually built from reusable subroutines:
#
# - PREPARE loads amplitudes or coefficients into an index register.
# - SELECT applies the indexed unitary or oracle.
# - A reflection completes the walk operator.
#
# Qamomile keeps these as resource-model fields. The circuit IR does not need a
# special block-encoding operation just to compare algorithmic costs.

# %% [markdown]
# ## Minimal Example
#
# The numbers below are synthetic. They show the accounting pattern:
# one qubitized walk costs two PREPARE calls, one SELECT call, and one
# reflection.

# %%
block = BlockEncodingResource(
    system_qubits=12,
    normalization=sp.Integer(240),
    prepare_cost_toffoli=30,
    select_cost_toffoli=120,
    reflection_cost_toffoli=8,
    ancilla_qubits=5,
    name="toy_lcu",
)

print(block.to_dict())

assert block.logical_qubits == 17
assert block.walk_cost_toffoli == 188
assert block.resource_values()[FTQCResourceQuantity.WALK_COST_TOFFOLI] == 188

# %% [markdown]
# ## Qubitized QPE
#
# QPE repeatedly calls the qubitized walk. For energy precision
# $\epsilon$, the symbolic call proxy is $\alpha / \epsilon$, where
# $\alpha$ is the block-encoding normalization.

# %%
architecture = FTQCCostModel(
    physical_qubits_per_logical=100,
    logical_cycle_time_seconds=sp.Float("1e-6"),
    factory_qubits=2000,
    toffoli_throughput_per_second=sp.Float("5e5"),
)

estimate = estimate_qubitized_qpe_from_block_encoding(
    block,
    precision=sp.Integer(3),
    qpe_register_qubits=6,
    cost_model=architecture,
)

print("iterations:", estimate.qpe_iterations)
print("Toffoli gates:", estimate.toffoli_gates)
print("logical qubits:", estimate.logical_qubits)

assert estimate.qpe_iterations == 80
assert estimate.toffoli_gates == 15040
assert estimate.logical_qubits == 23
assert estimate.physical_qubits == 4300
assert estimate.assumptions["block_encoding"] == "toy_lcu"
assert any(reference.key == "arXiv:1610.06546" for reference in estimate.references)

# %% [markdown]
# ## Compare Representations
#
# A new factorization or symmetry shift can reduce the normalization while
# increasing SELECT/PREPARE cost. Keeping the fields separate makes that
# trade-off visible.

# %%
compressed_block = BlockEncodingResource(
    system_qubits=12,
    normalization=sp.Integer(120),
    prepare_cost_toffoli=36,
    select_cost_toffoli=144,
    reflection_cost_toffoli=8,
    ancilla_qubits=7,
    name="compressed_toy_lcu",
)

compressed_estimate = estimate_qubitized_qpe_from_block_encoding(
    compressed_block,
    precision=sp.Integer(3),
    qpe_register_qubits=6,
    cost_model=architecture,
)

comparison = compare_ftqc_resource_estimates(
    estimate,
    compressed_estimate,
    quantities=("qpe_iterations", "toffoli_gates", "logical_qubits"),
)

for row in comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4))

assert comparison[0].ratio == sp.Rational(1, 2)
assert sp.simplify(comparison[1].ratio - sp.Rational(28, 47)) == 0
assert sp.simplify(comparison[2].ratio - sp.Rational(25, 23)) == 0

# %% [markdown]
# ## Notes
#
# :::{note}
# Treat `BlockEncodingResource` as a symbolic contract for algorithm design.
# It records what the estimator consumes; it does not claim that a particular
# PREPARE or SELECT circuit has already been synthesized for a backend.
# :::

# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - Block-encoding estimates separate normalization, PREPARE, SELECT,
#   reflection, ancilla, and QPE readout costs.
# - Qubitized QPE composes the block-encoding walk cost with
#   normalization-over-precision iterations.
# - Representation trade-offs can reduce total Toffoli count even when one
#   subroutine becomes more expensive.
