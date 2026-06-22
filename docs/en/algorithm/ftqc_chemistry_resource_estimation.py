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
# qubits, and runtime proxies.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp
from openfermion import QubitOperator

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEModel,
    ChemistryQPEMethod,
    FTQCCostModel,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
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
# ([arXiv:2603.22778](https://arxiv.org/abs/2603.22778)). The estimators below
# are deliberately symbolic. They let you test how a proposed representation
# changes the cost-driving quantities before committing to a concrete
# Hamiltonian-loading circuit.

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
precision = sp.Float("0.0016")  # about chemical accuracy in Hartree

cost_model = FTQCCostModel(
    physical_qubits_per_logical=800,
    logical_cycle_time_seconds=sp.Float("1e-6"),
    factory_qubits=20000,
    toffoli_throughput_per_second=sp.Float("2e6"),
)

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
# qpe_iterations = lambda_norm / precision
# toffoli_gates = qpe_iterations * walk_cost_toffoli
# ```
#
# This keeps the model honest: changing Hamiltonian normalization, changing the
# walk circuit, and changing physical architecture remain separate design
# choices.

# %%
thc_model = ChemistryQPEModel(
    hamiltonian=scaled_summary,
    method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
    walk_cost_toffoli=sp.Integer(4_000),
    description="THC-style scaled toy model",
)
scdf_model = ChemistryQPEModel(
    hamiltonian=scaled_summary.with_lambda_scale(
        sp.Float("0.5"),
        source="SCDF-style scaled toy model",
    ),
    method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
    walk_cost_toffoli=sp.Integer(4_400),
    second_factor_rank=9,
    description="SCDF-style scaled toy model",
)

thc = estimate_qubitized_chemistry_qpe_from_model(
    thc_model,
    precision=precision,
    cost_model=cost_model,
)
scdf = estimate_qubitized_chemistry_qpe_from_model(
    scdf_model,
    precision=precision,
    cost_model=cost_model,
)

assert scdf.qpe_iterations < thc.qpe_iterations
assert scdf.toffoli_gates < thc.toffoli_gates

print("THC Toffoli gates:", sp.N(thc.toffoli_gates, 4))
print("SCDF-style Toffoli gates:", sp.N(scdf.toffoli_gates, 4))
print("Toy Pauli terms:", toy_summary.n_pauli_terms)
print("SCDF-style logical qubits:", scdf.logical_qubits)

# %% [markdown]
# ## Early-FTQC Trotter QPE
#
# Early-FTQC proposals may favor shallower single-ancilla QPE and Pauli
# rotations over qubitized walks when qubits and depth are tightly constrained.
# The unitary-weight concentration factor below represents a spectrally
# invariant Hamiltonian transformation that reduces the cost-driving effective
# weight.

# %%
plain_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=precision,
    trotter_steps_per_sample=8,
    samples=128,
    cost_model=cost_model,
)

uwc_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=precision,
    trotter_steps_per_sample=8,
    samples=128,
    unitary_weight_factor=sp.Float("0.1"),
    randomized_compilation_factor=sp.Float("0.5"),
    rotation_synthesis_t_gates=3,
    cost_model=cost_model,
)

assert uwc_trotter.qpe_iterations < plain_trotter.qpe_iterations
assert uwc_trotter.logical_depth < plain_trotter.logical_depth

print("Plain Trotter QPE depth proxy:", sp.N(plain_trotter.logical_depth, 4))
print("UWC-style Trotter QPE depth proxy:", sp.N(uwc_trotter.logical_depth, 4))
print("UWC-style T gates:", sp.N(uwc_trotter.t_gates, 4))

# %% [markdown]
# ## Result
#
# We can put the estimates into a compact table. The important point is that
# each column has a distinct design meaning: changing the Hamiltonian
# representation should affect `qpe_iterations` and per-step cost, while
# changing the hardware model should affect `physical_qubits` and runtime.

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
            "runtime_seconds": sp.N(estimate.runtime_seconds, 4),
        },
    )

assert scdf.physical_qubits > thc.physical_qubits
assert uwc_trotter.physical_qubits == plain_trotter.physical_qubits

# %% [markdown]
# ## Summary
#
# In this notebook, we:
#
# - Separated FTQC chemistry resource quantities into Hamiltonian
#   normalization, QPE iterations, non-Clifford counts, logical qubits,
#   physical qubits, and runtime proxies.
# - Compared qubitized QPE representations without lowering the Qamomile IR
#   into backend-specific chemistry loading circuits.
# - Demonstrated how a unitary-weight concentration factor can be modeled as a
#   cost-driver reduction for early-FTQC single-ancilla Trotter QPE.
