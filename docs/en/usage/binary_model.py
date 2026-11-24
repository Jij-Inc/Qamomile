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
# A `BinaryModel` represents an unconstrained polynomial objective over binary
# (0 or 1) or spin ($-1$ or $+1$) variables. This notebook shows how to use
# `BinaryModel`.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools
import math
import os

import qamomile as qm
from qamomile.optimization import BinaryModel, VarType

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Choose an input form
#
# ### HUBO input
#
# For variables that take the values 0 and 1, the problem is called a HUBO
# if any term in the simplified objective multiplies three or more distinct
# variables. If every term contains at most two variables, it is called a QUBO.
#
# In both cases, build the model with `BinaryModel.from_hubo`. Pass a dictionary
# whose keys list the variable IDs in each product and whose values are the
# corresponding coefficients.
#
# For example, take $f(x_1,x_3)=x_1+2x_3+3x_1x_3+5$. With the column vector
# $\mathbf{x}=(x_1,x_3)^{\mathsf T}$, this QUBO has the upper triangular matrix
#
# $$
# Q=\begin{bmatrix}1&3\\0&2\end{bmatrix},
# \qquad f(\mathbf{x})=\mathbf{x}^{\mathsf T}Q\mathbf{x}+5.
# $$
#
# Since $x_i^2=x_i$, the diagonal entries give the coefficients of `(1,)`
# and `(3,)`. The off-diagonal entry gives the coefficient of `(1, 3)`.

# %%
coefficients = {(1,): 1.0, (3,): 2.0, (1, 3): 3.0}
constant = 5.0
variables = (1, 3)

model = BinaryModel.from_hubo(coefficients, constant=constant, variables=variables)
print("Variable IDs:", model.variables)
print("Polynomial degree:", model.order)
assert model.vartype is VarType.BINARY
assert model.variables == tuple(sorted(variables))
assert model.order == 2

# %% [markdown]
# To make this a higher-order objective, add the key `(1, 3, 7)` for
# $0.1x_1x_3x_7$. If an ID occurs more than once, it reduces to a single
# factor. For example, `(7, 7, 2)` represents $x_7^2x_2=x_7x_2$. Its coefficient is added to
# any coefficient for `(2, 7)`. The empty key `()` adds to the constant.

# %%
hubo_model = BinaryModel.from_hubo({**coefficients, (1, 3, 7): 0.1}, constant=constant)
print("HUBO degree:", hubo_model.order)
assert hubo_model.order == 3
assert hubo_model.coefficients[(1, 3, 7)] == 0.1

# %% [markdown]
# ### Ising model input
#
# Use `BinaryModel.from_ising` when the variables take the values $-1$ and
# $+1$. The same key `(7, 7, 2)` now represents $s_7^2s_2=s_2$, since
# $s_7^2=1$. Its coefficient combines with that of `(2,)`. In the example
# below their sum is $3-1=2$, and the empty key `()` adds 2 to the constant.

# %%
terms = {(7, 7, 2): 3.0, (2,): -1.0, (): 2.0}
ising_example = BinaryModel.from_ising(terms, variables=(2, 7))

print("Ising terms:", dict(ising_example.coefficients))
assert ising_example.coefficients == {(2,): 2.0}
assert ising_example.constant == 2.0
assert ising_example.variables == (2, 7)

# %% [markdown]
# The Ising objective simplifies to $2s_2+2$. Passing `variables=(2, 7)`
# keeps ID 7 in the model even though it disappears from the expression.

# %% [markdown]
# ## Evaluate the objective
#
# Return to the QUBO stored in `model`. Substituting $x_1=1$ and $x_3=0$
# gives $1+2(0)+3(1)(0)+5=6$.
# `energy` computes this objective value. A dictionary uses the original
# variable IDs as keys. A tuple follows the ascending ID order in `model.variables`,
# here `(1, 3)`.

# %%
assignment = {1: 1, 3: 0}
energy = model.energy(assignment)
print("Objective value:", energy)
assert energy == model.energy((1, 0)) == 6.0

# %% [markdown]
# A BINARY model accepts only 0 and 1. For example, `model.energy((-1, 1))`
# raises `ValueError`. Spin models also support `energy`, with each variable
# set to −1 or +1.

# %% [markdown]
# ## Convert between binary and spin variables
#
# Our original model uses binary variables. To express the *same objective*
# in spins, call `model.to_ising()`. The substitution $s_i=1-2x_i$ maps
# bit 0 to spin $+1$ and bit 1 to spin $-1$. Substituting
# $x_i=(1-s_i)/2$ into the original polynomial gives
#
# $$
# f=\tfrac{29}{4}-\tfrac{5}{4}s_1-\tfrac{7}{4}s_3+\tfrac{3}{4}s_1s_3.
# $$
#
# `to_binary()` reverses the substitution. We can check the objective in both
# directions for all four assignments. Floating-point coefficients can pick up
# rounding error during conversion, so we allow a small difference.

# %%
spin_model = model.to_ising()
roundtrip_model = spin_model.to_binary()
print("Spin constant:", spin_model.constant)
print("Spin coefficients:", dict(spin_model.coefficients))
assert spin_model.vartype is VarType.SPIN
assert roundtrip_model.vartype is VarType.BINARY
assert spin_model.variables == roundtrip_model.variables == model.variables

assignments = itertools.product((0, 1), repeat=model.num_variables)
if docs_test_mode:
    assignments = itertools.islice(assignments, 1)
for bits in assignments:
    spins = tuple(1 - 2 * bit for bit in bits)
    expected = model.energy(bits)
    assert math.isclose(
        spin_model.energy(spins), expected, rel_tol=1e-12, abs_tol=1e-12
    )
    assert math.isclose(
        roundtrip_model.energy(bits), expected, rel_tol=1e-12, abs_tol=1e-12
    )

# %% [markdown]
# ## Normalize the objective
#
# Both normalization methods return a new model. For a constant-only objective,
# the value stays the same. Otherwise, they divide the whole objective, including
# the constant, by a positive number. In exact arithmetic the minimizing assignments
# are the same. With floating-point coefficients, rounding can create or break ties.
#
# - `normalize_by_abs_max()` divides by the largest absolute value of the
#   nonconstant coefficients. Here it is 3, giving $f/3$.
# - `normalize_by_rms()` groups the nonconstant coefficients by degree, takes
#   the mean of their squares in each group, then takes the square root of
#   the sum. Here the divisor is $W=\sqrt{(1^2+2^2)/2+3^2}=\sqrt{11.5}$.
#
# To three decimal places, $f/W$ is
# $0.295x_1+0.590x_3+0.885x_1x_3+1.474$. At the assignment we checked above,
# the new scores are $6/3$ and $6/W$.

# %%
abs_max_model = model.normalize_by_abs_max()
rms_model = model.normalize_by_rms()
print("Original:", energy)
print("Absolute maximum scaling:", abs_max_model.energy(assignment))
print("RMS scaling:", rms_model.energy(assignment))
assert math.isclose(abs_max_model.energy(assignment), energy / 3)
assert math.isclose(rms_model.energy(assignment), energy / math.sqrt(11.5))

# %% [markdown]
# ## Display the objective as a Hamiltonian
#
# `model.hamiltonian` expresses the objective as an operator on qubits.
# The Pauli operator $Z_j$ has value $+1$ on bit 0 and $-1$ on bit 1,
# just like a spin. Qubits 0 and 1 correspond to variable IDs 1 and 3
# in `model.variables` order. Replacing each spin in the converted objective
# by its $Z$ operator gives
#
# $$
# H=7.25I-1.25Z_0-1.75Z_1+0.75Z_0Z_1.
# $$
#
# The constant is stored separately from the three nonconstant terms.
# For bits `(1, 0)` we get $7.25+1.25-1.75-0.75=6$, as before.

# %%
hamiltonian = model.hamiltonian
print(f"{hamiltonian.constant.real:g} I")
for word, coefficient in hamiltonian.terms:
    operators = " ".join(
        f"{factor.pauli.value}{factor.index}" for factor in word.factors
    )
    print(f"{coefficient.real:+g} {operators}")
assert hamiltonian.num_qubits == model.num_variables
assert hamiltonian.is_hermitian and hamiltonian.is_pairwise_commuting
assert hamiltonian.constant.real == spin_model.constant

# %% [markdown]
# ## Map measurement results to variables
#
# Suppose four measurements returned `(1, 0)` three times and `(0, 1)` once.
# We can enter these counts directly as a `SampleResult`, with the bits already
# in Qamomile's logical qubit order. `decode` assigns each bit to its original
# variable ID, evaluates each distinct assignment, and keeps its count.

# %%
result = qm.SampleResult.from_counts(backend="example", counts={(1, 0): 3, (0, 1): 1})
decoded = model.decode(result)

for sample in decoded.samples:
    print(sample.values, "energy =", sample.energy, "count =", sample.occurrences)
print("Mean energy:", decoded.mean_energy)
assert decoded.total_occurrences == 4
assert decoded.mean_energy == 6.25
assert decoded.lowest.values == assignment
assert decoded.lowest.energy == energy

# %% [markdown]
# The mean is $(3\times6+1\times7)/4=6.25$. `decoded.lowest` selects the
# best *observed* assignment, with energy 6. The global minimum is 5 at
# `(0, 0)`, which did not occur in these measurements.

# %% [markdown]
# ## Save and restore the model
#
# `to_bytes()` produces the bytes you can write to a file. `from_bytes()`
# checks their integrity and restores the model, including its variable IDs.
# Models with the same domain, variable IDs, simplified coefficients and
# constant share a SHA-256 identifier, available as `semantic_digest`.

# %%
payload = model.to_bytes()
restored = BinaryModel.from_bytes(payload)
print("Model identifier:", restored.semantic_digest)
assert restored == model
assert restored.to_bytes() == payload
assert restored.semantic_digest == model.semantic_digest

# %% [markdown]
# ## Summary
#
# - Use `from_hubo` for binary coefficients and `from_ising` for spin
#   coefficients. Use `energy` to check one assignment before converting or
#   rescaling the model.
# - Use `model.variables` to read qubit and sample positions in the original
#   variable IDs. `decode` gives you the observed objective values and counts.
#
# ### Related topics
#
# - [Solving MaxCut with QAOA](../algorithm/qaoa_maxcut) starts from a graph
#   and builds its polynomial objective.
# - [QAOA for graph partitioning](../algorithm/qaoa_graph_partition) evaluates
#   measurements against the original OMMX constraints.
# - [Pauli Correlation Encoding for MaxCut](../algorithm/pce_maxcut) recovers
#   spin assignments from measured expectation values.
