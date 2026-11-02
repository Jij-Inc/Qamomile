# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [usage, optimization]
# ---
#
# # How to use `BinaryModel`
#
# `BinaryModel` represents an unconstrained polynomial objective over BINARY
# (0/1) or SPIN (-1/1) variables. This page starts with a coefficient mapping,
# then shows how to evaluate assignments, change variable domains, and use the
# model with Hamiltonians, samples, and OMMX.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools
import math

import ommx.v1

import qamomile as qm
from qamomile.optimization import BinaryModel, OMMXProblem, VarType

# %% [markdown]
# ## Create a model from coefficients
#
# Use `BinaryModel.from_hubo` for a BINARY polynomial. Each key is a tuple of
# variable IDs in one term, and its value is that term's coefficient. Pass the
# constant separately. Variable IDs need not start at zero or be consecutive.
#
# For example, construct $x_1 + 2x_3 + 3x_1x_3 + 5$:

# %%
COEFFICIENTS = {
    (1,): 1.0,
    (3,): 2.0,
    (1, 3): 3.0,
}

model = BinaryModel.from_hubo(
    COEFFICIENTS,
    constant=5.0,
    variables=(1, 3),
)

print("vartype:", model.vartype)
print("constant:", model.constant)
print("variables:", model.variables)
print("order:", model.order)
print("coefficients:", dict(model.coefficients))

assert model.vartype is VarType.BINARY
assert model.constant == 5.0
assert model.variables == (1, 3)
assert model.num_variables == 2
assert model.order == 2
assert model.coefficients == {
    (1,): 1.0,
    (1, 3): 3.0,
    (3,): 2.0,
}

# %% [markdown]
# The one-element keys `(1,)` and `(3,)` are linear terms; `(1, 3)` is a
# quadratic term. `model.variables == (1, 3)` is the order used by sequence
# assignments, Hamiltonian qubits, and decoded samples. The model keeps these
# original variable IDs.

# %% [markdown]
# Convert a BINARY model to the equivalent SPIN model with `to_ising()`.
# `to_binary()` performs the reverse conversion.

# %%
spin_equivalent = model.to_ising()

print("vartype:", spin_equivalent.vartype)
print("constant:", spin_equivalent.constant)
print("coefficients:", dict(spin_equivalent.coefficients))

assert spin_equivalent.vartype is VarType.SPIN
assert spin_equivalent.constant == 29.0 / 4.0
assert spin_equivalent.coefficients == {
    (1,): -5.0 / 4.0,
    (1, 3): 3.0 / 4.0,
    (3,): -7.0 / 4.0,
}

# %% [markdown]
# Binary and spin variables are related by $s=1-2x$. Substituting that relation
# into the running example gives
#
# $$
# \begin{align*}
# x_1 + 2x_3 + 3x_1x_3 + 5
# &= \left(\frac{1-s_1}{2}\right)
# + 2\left(\frac{1-s_3}{2}\right)
# + 3\left(\frac{1-s_1}{2}\right)\left(\frac{1-s_3}{2}\right)+5 \\
# &= \frac{3}{4}s_1s_3-\frac{5}{4}s_1-\frac{7}{4}s_3+\frac{29}{4}.
# \end{align*}
# $$
#
# This matches the values printed above. The SPIN model retains variable IDs
# `(1, 3)`.

# %% [markdown]
# ## Building a `BinaryModel` from QUBO, HUBO, or Ising
#
# `from_hubo` also accepts quadratic QUBO models and higher-order BINARY
# polynomials. Use `from_ising` for SPIN polynomials. Both use tuples of
# variable IDs as coefficient keys.
#
# ### Building from QUBO
#
# A QUBO is a BINARY polynomial of degree at most two, so `from_hubo` accepts
# it directly. Diagonal keys reduce by the binary identity $x_i^2=x_i$.
# Consider the QUBO matrix
#
# $$
# \begin{bmatrix}
# 1 & 0.5 & 0 \\
# 0 & 2 & 1 \\
# 0 & 0 & 3
# \end{bmatrix},
# $$
#
# which encodes
# $x_0 + 2x_1 + 3x_2 + 0.5x_0x_1 + x_1x_2$.

# %%
qubo_model = BinaryModel.from_hubo(
    {
        (0, 0): 1.0,
        (1, 1): 2.0,
        (2, 2): 3.0,
        (0, 1): 0.5,
        (1, 2): 1.0,
    },
    variables=(0, 1, 2),
)
assert qubo_model.order == 2
assert qubo_model.coefficients == {
    (0,): 1.0,
    (0, 1): 0.5,
    (1,): 2.0,
    (1, 2): 1.0,
    (2,): 3.0,
}

print("vartype:", qubo_model.vartype)
print("constant:", qubo_model.constant)
print("coefficients:", dict(qubo_model.coefficients))

# %% [markdown]
# ### Building from HUBO
#
# `from_hubo` accepts cubic and higher-order products in the same coefficient
# mapping. For example, consider
#
# $$
# x_0 + 2x_1 + 3x_2 + 0.5x_0x_1 + x_1x_2 + 0.1x_0x_1x_2.
# $$

# %%
hubo_model = BinaryModel.from_hubo(
    {
        (0,): 1.0,
        (1,): 2.0,
        (2,): 3.0,
        (0, 1): 0.5,
        (1, 2): 1.0,
        (0, 1, 2): 0.1,
    }
)
assert hubo_model.order == 3
assert hubo_model.coefficients[(0, 1, 2)] == 0.1

print("vartype:", hubo_model.vartype)
print("constant:", hubo_model.constant)
print("coefficients:", dict(hubo_model.coefficients))

# %% [markdown]
# ### Building from Ising
#
# `from_ising` accepts the same coefficient-key format for SPIN variables,
# including higher-order terms.

# %%
ising_model = BinaryModel.from_ising(
    {
        (0,): -1.0,
        (1,): 2.0,
        (2,): -3.0,
        (0, 1): 0.5,
        (1, 2): -1.0,
    }
)
assert ising_model.vartype is VarType.SPIN
assert ising_model.order == 2
assert ising_model.coefficients[(1, 2)] == -1.0

print("vartype:", ising_model.vartype)
print("constant:", ising_model.constant)
print("coefficients:", dict(ising_model.coefficients))

# %% [markdown]
# ## Normalization and energy evaluation
#
# ### Normalization
#
# `BinaryModel` ships with two normalization methods,
# `normalize_by_abs_max` and `normalize_by_rms`, and an energy evaluation
# method, `energy`. Before looking at them, note that Rust canonicalizes
# duplicate IDs using the selected algebra. For BINARY values, $x_i^n=x_i$.
# For SPIN values, $s_i^2=1$, so only the parity of a repeated ID remains.

# %%
binary_alias = BinaryModel.from_hubo(
    {
        (7, 7, 2): 3.0,
        (2, 7): -1.0,
        (): 2.0,
    },
    constant=0.5,
)
spin_alias = BinaryModel.from_ising(
    {
        (7, 7, 2): 3.0,
        (2,): -1.0,
    }
)

assert binary_alias.coefficients == {(2, 7): 2.0}
assert binary_alias.constant == 2.5
assert spin_alias.coefficients == {(2,): 2.0}
assert spin_alias.constant == 0.0

# %% [markdown]
# ### Rescale coefficients without changing the minimizer
#
# `normalize_by_abs_max` rescales every coefficient by the largest absolute
# value. For $x_1 + 2x_3 + 3x_1x_3 + 5$, the largest coefficient is 3, so the
# normalized model is
#
# $$
# \frac{1}{3}x_1 + \frac{2}{3}x_3 + x_1x_3 + \frac{5}{3}.
# $$
#
# `normalize_by_rms` uses a degree-wise RMS scale:
#
# $$
# W=\sqrt{\sum_k \frac{1}{|E_k|}\sum_{e\in E_k}w_e^2}.
# $$
#
# Here there are two linear terms and one quadratic term:
#
# - $|E_1|=2$ and $1^2+2^2=5$,
# - $|E_2|=1$ and $3^2=9$.
#
# Therefore $W=\sqrt{5/2+9}=\sqrt{11.5}\approx3.391$, and the normalized
# objective is approximately
#
# $$
# 0.295x_1+0.590x_3+0.884x_1x_3+1.475.
# $$
#
# Both methods return a new immutable model. The source model, value domain,
# variable IDs, and minimizers remain unchanged; only the energy scale changes.

# %%
normalization_model = model
abs_max_model = normalization_model.normalize_by_abs_max()
rms_model = normalization_model.normalize_by_rms()
rms_factor = math.sqrt(11.5)

assert normalization_model.coefficients == {(1,): 1.0, (1, 3): 3.0, (3,): 2.0}
assert normalization_model.constant == 5.0
assert abs_max_model.coefficients == {
    (1,): 1.0 / 3.0,
    (1, 3): 1.0,
    (3,): 2.0 / 3.0,
}
assert abs_max_model.constant == 5.0 / 3.0
assert rms_model.coefficients == {
    (1,): 1.0 / rms_factor,
    (1, 3): 3.0 / rms_factor,
    (3,): 2.0 / rms_factor,
}
assert rms_model.constant == 5.0 / rms_factor

# %% [markdown]
# ### Objective (energy) evaluation
#
# `energy` evaluates the objective function for a variable assignment. Let us
# compute the energy of $x_1 + 2x_3 + 3x_1x_3 + 5$ at $x_1=1$, $x_3=0$:
#
# $$
# 1 + 2\cdot0 + 3\cdot1\cdot0 + 5 = 6.
# $$
#
# A mapping is written in the original problem's ID space. A sequence uses the
# dense order shown by `model.variables`, here `(1, 3)`. Both forms cross the
# same checked Rust evaluation boundary.

# %%
assignment = {3: 0, 1: 1}

energy_from_ids = model.energy(assignment)
energy_from_dense_order = model.energy((1, 0))

print("energy:", energy_from_ids)
assert energy_from_ids == energy_from_dense_order == 6.0

# %% [markdown]
# The value domain is part of the type and runtime contract. A BINARY model
# rejects SPIN values instead of interpreting them as another problem.

# %%
try:
    model.energy((-1, 1))
except ValueError as error:
    print("expected domain error:", error)
else:
    raise AssertionError("a BINARY model accepted a SPIN assignment")

# %% [markdown]
# ## Convert between BINARY and SPIN exactly
#
# `to_ising` applies $s_i=1-2x_i$ while preserving the original IDs and every
# objective value. `to_binary` is its inverse. The example is small enough to
# check all four assignments deterministically rather than relying on samples.

# %%
spin_model = model.to_ising()

assert spin_model.vartype is VarType.SPIN
assert spin_model.variables == model.variables
assert spin_model.to_binary() == model

for bits in itertools.product((0, 1), repeat=model.num_variables):
    spins = tuple(1 - 2 * bit for bit in bits)
    assert spin_model.energy(spins) == model.energy(bits)

# %% [markdown]
# ## Project the same objective to a Hamiltonian
#
# `model.hamiltonian` is an immutable, backend-neutral sum of commuting Pauli-Z
# words. Qubit position zero corresponds to variable ID `1` and position one to
# variable ID `3`; Python does not rebuild the coefficients.

# %%
hamiltonian = model.hamiltonian

print("Hamiltonian terms:", len(hamiltonian.terms))
assert hamiltonian.num_qubits == model.num_variables == 2
assert hamiltonian.is_hermitian
assert hamiltonian.is_pairwise_commuting
assert max(len(word.factors) for word, _ in hamiltonian.terms) == 2

# %% [markdown]
# ## Decode logical samples in the original ID space
#
# Every backend first normalizes its native bit order to Qamomile's logical
# source order. `BinaryModel.decode` then maps each unique row back to original
# IDs, evaluates it with the same polynomial, and retains its occurrence count.
# Here we construct the already-normalized `SampleResult` directly so this
# focused usage page needs no simulator launch.

# %%
result = qm.SampleResult.from_counts(
    backend="example",
    counts={
        (1, 0): 3,
        (0, 1): 1,
    },
)
decoded = model.decode(result)

for sample in decoded.samples:
    print(sample.values, "energy =", sample.energy, "count =", sample.occurrences)

assert decoded.vartype is VarType.BINARY
assert decoded.total_occurrences == 4
assert decoded.mean_energy == 6.25
assert decoded.lowest.assignment == ((1, 1), (3, 0))
assert decoded.lowest.energy == 6.0

# %% [markdown]
# ## Persist one canonical problem identity
#
# The serialized form is canonical, bounded, integrity checked, and contains
# no Python callback or backend object. Equivalent canonical models therefore
# have the same bytes and SHA-256 semantic digest.

# %%
payload = model.to_bytes()
restored = BinaryModel.from_bytes(payload)

print("semantic digest:", model.semantic_digest)
assert restored == model
assert restored.to_bytes() == payload
assert restored.semantic_digest == model.semantic_digest
assert len(model.semantic_digest) == 64

# %% [markdown]
# ## Building a `BinaryModel` from OMMX
#
# Finally, let us build the same model from OMMX (Open Mathematical prograMming
# eXchange). OMMX is an open data format and SDK for exchanging mathematical
# optimization data between software systems and people. In a real application
# the instance will often come from a packaged problem or a mathematical
# modeling tool; here we construct it from low-level components to keep the
# example self-contained.
#
# `OMMXProblem.from_instance` copies the supplied instance, builds its BINARY
# polynomial, and retains the original objective and constraints for later
# evaluation. Use it when feasibility or original objective values matter.

# %%
x_1 = ommx.v1.DecisionVariable.binary(1, name="x_1")
x_3 = ommx.v1.DecisionVariable.binary(3, name="x_3")
instance = ommx.v1.Instance.from_components(
    decision_variables=[x_1, x_3],
    objective=x_1 + 2 * x_3 + 3 * x_1 * x_3 + 5,
    constraints=[],
    sense=ommx.v1.Instance.MINIMIZE,
)
original_payload = instance.to_bytes()

problem = OMMXProblem.from_instance(instance)
evaluation = problem.evaluate(decoded)

assert instance.to_bytes() == original_payload
assert problem.model == model
assert evaluation.total_occurrences == decoded.total_occurrences
assert evaluation.objectives == {0: 7.0, 1: 6.0}
assert all(evaluation.feasible.values())
assert evaluation.best_feasible.objective == 6.0

# %% [markdown]
# ## Summary
#
# Start with `from_hubo` or `from_ising` when you have a coefficient mapping.
# `BinaryModel` retains the original variable IDs and provides energy
# evaluation, domain conversion, Hamiltonian projection, typed sample decoding,
# and canonical persistence. Use `OMMXProblem.from_instance` when you also need
# the original constraints and objective values.

# %% [markdown]
# ## Related topics
#
# - [Solving MaxCut with QAOA](../algorithm/qaoa_maxcut) creates a graph-derived
#   polynomial and applies QAOA.
# - [QAOA for graph partitioning](../algorithm/qaoa_graph_partition) applies
#   QAOA to an OMMX instance and evaluates feasibility in the original model.
# - [QAOA for constrained graph partitioning](../algorithm/qaoa_graph_partition)
#   keeps penalty energy separate from original OMMX feasibility.
# - [Pauli Correlation Encoding for MaxCut](../algorithm/pce_maxcut) reuses the
#   same SPIN model and typed decoded distribution with fewer qubits.
