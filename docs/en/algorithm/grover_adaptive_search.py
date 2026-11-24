# ---
# jupyter:
#   jupytext:
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
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # Grover Adaptive Search for Combinatorial Polynomial Binary Optimization
#
# In this tutorial, we solve a small **portfolio selection** problem with
# Grover Adaptive Search (GAS): for each of nine assets we decide whether to buy
# it, so that a cost that combines risk and expected return is as small as
# possible. The cost is a polynomial of binary variables, and there is no
# constraint.
#
# GAS finds the minimum of such a cost by repeating one step
# ([Gilliam, Woerner and Gonciulea, Quantum 5, 428 (2021)](https://doi.org/10.22331/q-2021-04-08-428)).
# It keeps a threshold $y$, the lowest cost found so far. It repeatedly uses
# Grover's search algorithm to ask which assignments have a cost below $y$.
# When a measurement returns an assignment with a cost below $y$, that cost
# becomes the new threshold. A classical loop repeats this until the
# threshold stops improving.
#
# The tutorial follows these steps:
#
# 1. Write the cost as a `BinaryModel`, and find the exact minimum by trying
#    every assignment.
# 2. Build the circuit of one round, for a fixed threshold, with Qamomile's
#    `GASExperiment`.
# 3. Run one round and compare its samples with what Grover's algorithm
#    predicts.
# 4. Write the classical loop that lowers the threshold, and compare its result
#    with the exact minimum.
#
# **Prerequisites**: Grover's algorithm and [how to use `BinaryModel`](../usage/binary_model.ipynb).

# %% [markdown]
# ## Installation

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import functools
import itertools
import math
import os

import numpy as np

import qamomile as qm
from qamomile.optimization import BinaryModel, GASExperiment

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Problem Settings
#
# ### Problem Definition
#
# We decide for each of $n$ assets whether to buy it: $x_i = 1$ if we buy asset
# $i$, and $x_i = 0$ if we do not. An *assignment*
# $\boldsymbol{x} = (x_0, \dots, x_{n-1})$ describes one portfolio. The vector
# $\boldsymbol{\mu}$ holds the expected returns of the assets, and the matrix
# $\Sigma$ is the covariance matrix of their returns. We look for the
# assignment with the smallest *cost*
#
# $$
# f(\boldsymbol{x}) = q\,\boldsymbol{x}^{\mathsf T}\Sigma\,\boldsymbol{x} - \boldsymbol{\mu}^{\mathsf T}\boldsymbol{x}
# $$
#
# The first term is the variance of the return of the portfolio, a measure of
# its risk, weighted by the *risk aversion* $q \ge 0$, which quantifies how
# strongly risk is penalized relative to expected return. The second term is
# its expected return. There is no constraint: each of the $2^n$ assignments is
# allowed.

# %% [markdown]
# ### Problem Instance
#
# The instance has $n = 9$ assets and $q = 1$. The arrays `expected_returns`
# and `covariance` in the code below hold $\boldsymbol{\mu}$ and $\Sigma$.
#
# `BinaryModel.from_hubo` takes a polynomial in the binary variables as a
# dictionary with one entry per term. The key is the tuple of the indices of
# the variables that are multiplied in the term, and the value is the
# coefficient of the term. Our cost has one term for each asset,
# $(q\,\Sigma_{ii} - \mu_i)\,x_i$, and one term for each pair $i < j$,
# $2q\,\Sigma_{ij}\,x_i x_j$, since $x_i^2 = x_i$ and $\Sigma$ is symmetric.
# For example, the term of asset 0 is $(12 - 22)\,x_0 = -10\,x_0$, with the
# entry `(0,): -10.0`, and the term of assets 0 and 1 is
# $2 \times (-3)\,x_0 x_1 = -6\,x_0 x_1$, with the entry `(0, 1): -6.0`.
# `BinaryModel` calls the cost of an assignment its *energy*.

# %%
expected_returns = np.array([22, 4, 19, 3, 23, 2, 5, 25, 3], dtype=int)
covariance = np.array(
    [
        [12, -3, 4, 0, -2, 3, 0, 2, -1],
        [-3, 15, 0, 5, 1, -4, 2, 0, 3],
        [4, 0, 10, -6, 3, 2, -1, 4, 0],
        [0, 5, -6, 18, -4, 0, 3, -2, 5],
        [-2, 1, 3, -4, 14, 2, -3, 0, 2],
        [3, -4, 2, 0, 2, 20, 4, -3, 1],
        [0, 2, -1, 3, -3, 4, 11, 2, -2],
        [2, 0, 4, -2, 0, -3, 2, 16, -4],
        [-1, 3, 0, 5, 2, 1, -2, -4, 13],
    ],
    dtype=int,
)
RISK_AVERSION = 1

NUM_ASSETS = 3 if docs_test_mode else 9


def portfolio_model(num_assets: int) -> BinaryModel:
    """Build the cost of the first `num_assets` assets as a `BinaryModel`."""
    terms = {}
    for i in range(num_assets):
        # x_i * x_i = x_i, so the variance of asset i joins its linear term.
        terms[(i,)] = float(RISK_AVERSION * covariance[i, i] - expected_returns[i])
        for j in range(i + 1, num_assets):
            # The matrix is symmetric: the pairs (i, j) and (j, i) give the factor 2.
            coefficient = float(2 * RISK_AVERSION * covariance[i, j])
            if coefficient != 0:
                terms[(i, j)] = coefficient
    return BinaryModel.from_hubo(terms, variables=tuple(range(num_assets)))


model = portfolio_model(NUM_ASSETS)

print(f"Number of assets: {model.num_variables}")
assert model.num_variables == NUM_ASSETS

# %% [markdown]
# ### Exact Solution (Brute Force)
#
# With 9 assets there are only $2^9 = 512$ assignments, so we can compute the
# cost of every one of them. This gives the exact minimum, which we use as the
# reference for the result of GAS. `model.energy(bits)` returns the cost of an
# assignment.

# %%
# Compute the cost of every assignment.
costs = {
    bits: model.energy(bits) for bits in itertools.product((0, 1), repeat=NUM_ASSETS)
}

# Find the smallest cost and every assignment that reaches it.
min_cost = min(costs.values())
optimal_assignments = [bits for bits, cost in costs.items() if cost == min_cost]

print(f"Number of assignments: {len(costs)}")
print(f"Smallest cost: {min_cost}")
for bits in optimal_assignments:
    print(f"Optimal assignment: {bits}")
print(f"Largest cost: {max(costs.values())}")

if not docs_test_mode:
    assert min_cost == -28.0
    assert optimal_assignments == [(1, 0, 0, 0, 1, 0, 0, 1, 0)]

# %% [markdown]
# Only the portfolio with assets 0, 4 and 7 reaches the smallest cost $-28$.
# Its variance is $12 + 14 + 16 + 2 \times (-2 + 2 + 0) = 42$ and the
# expected return is $22 + 23 + 25 = 70$, so the cost is $42 - 70 = -28$. The
# costs range from $-28$ to $94$. Buying nothing, the all-zero assignment, has
# the cost 0.

# %% [markdown]
# ## Algorithm
#
# GAS is Grover's algorithm with a moving target. It keeps a threshold $y$, the
# cost of the best assignment found so far, and uses Grover's algorithm to find
# an assignment with $f(\boldsymbol{x}) < y$. When it finds one, the cost of
# that assignment becomes the new threshold. This section assumes familiarity
# with Grover's algorithm and describes what GAS adds to it.

# %% [markdown]
# ### What GAS Uses from Grover's Algorithm
#
# Suppose that $M$ of the $N = 2^n$ assignments satisfy a condition. Grover's
# algorithm starts from the uniform superposition and repeats one *iteration*:
# an *oracle* that changes the sign of the amplitude of every assignment that
# satisfies the condition, and then a reflection about the start state. After
# $r$ iterations, a measurement returns such an assignment with probability
#
# $$
# P_r = \sin^2\big((2r+1)\,\theta\big), \qquad \sin^2\theta = M/N
# $$
#
# ([Boyer, Brassard, Høyer and Tapp, arXiv:quant-ph/9605034](https://doi.org/10.48550/arXiv.quant-ph/9605034)).
# The probability rises and then falls again as $r$ increases, so the best
# number of iterations depends on $M$.

# %% [markdown]
# ### The Oracle for $f(\boldsymbol{x}) < y$
#
# The circuit has two registers: $n$ *decision qubits* for the assignment, and
# a *value register* that holds a whole number in two's complement. A
# preparation $A_y$ puts the decision qubits into the uniform superposition and
# writes $f(\boldsymbol{x}) - y$ into the value register
# ([Gilliam, Woerner and Gonciulea, Quantum 5, 428 (2021)](https://doi.org/10.22331/q-2021-04-08-428)):
#
# $$
# A_y\,|0\rangle = \frac{1}{\sqrt{N}} \sum_{\boldsymbol{x}} |\boldsymbol{x}\rangle\,|f(\boldsymbol{x}) - y\rangle
# $$
#
# In two's complement the leading qubit of the value register is 1 exactly for
# the negative numbers, that is, exactly when $f(\boldsymbol{x}) < y$. So the
# oracle is a single $Z$ gate on that qubit. In place of the reflection about
# the start state, the circuit applies $A_y D_0 A_y^\dagger$. Here $D_0$
# multiplies the amplitude by $-1$ when all decision qubits are in the state
# $|0\rangle$, and does nothing otherwise. The product equals the reflection up
# to a factor $-1$, which changes no probability.
#
# The value register must be wide enough to hold every $f(\boldsymbol{x}) - y$.
# Otherwise a value wraps around and gets the wrong sign. `GASExperiment`
# chooses the width from the coefficients.

# %% [markdown]
# ### The Adaptive Loop
#
# One *round* runs the circuit with the current threshold and a number $r$ of
# iterations, and measures the decision qubits. The classical loop is:
#
# 1. Start from an assignment $\boldsymbol{x}$ and set
#    $y = f(\boldsymbol{x})$.
# 2. Draw $r$, apply $r$ Grover iterations with the threshold $y$, and then
#    measure the decision qubits.
# 3. If the resulting assignment has a cost below $y$, it becomes the new
#    $\boldsymbol{x}$, and its cost the new $y$.
# 4. Repeat from step 2. Stop after a set number of consecutive rounds
#    without a lower cost.
#
# The number $M$ of assignments below the threshold is unknown and changes with
# $y$, so no fixed $r$ fits. Following Gilliam et al., the loop draws $r$
# uniformly from $0, \dots, \lceil k \rceil - 1$, starting with $k = 1$. A
# round without improvement multiplies $k$ by a factor $\lambda > 1$, and an
# improvement sets it back to 1. At the minimum no assignment satisfies the
# condition and no round can improve, so the loop stops after a fixed number of
# rounds in a row without improvement. The result is the best assignment seen,
# without a proof that it is optimal.

# %% [markdown]
# ## Implementation with Qamomile
#
# ### Build One Round
#
# `GASExperiment` builds the program of one round from the model, a threshold
# and a number of iterations: the preparation $A_y$, then the oracle and the
# reflection `iterations` times, and a measurement. Here we set the threshold
# to $y = 0$ and use one iteration. Buying nothing costs $f(0, \dots, 0) = 0$,
# so this round searches for a portfolio that is better than buying nothing.

# %%
# The cost of buying nothing.
threshold = 0.0
experiment = GASExperiment(model, threshold=threshold, iterations=1)

print(f"Decision qubits: {model.num_variables}")
print(f"Value register qubits: {experiment.resolved_output_bits}")

# Count the qubits of the whole program without running it.
study = qm.resources.logical(experiment.program)
num_qubits, _ = study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
print(f"Qubits of one round: {num_qubits}")

# %% [markdown]
# One round uses 18 qubits: 9 decision qubits and 9 in the value register.
# `qm.resources.logical` reads this number from the program without running it
# (see [Resource Estimation](../tutorial/04_resource_estimation.ipynb)). The
# number of iterations does not change it.
#
# :::{note}
# The coefficients here are integers, so the value register holds every
# $f(\boldsymbol{x}) - y$ exactly. For real coefficients, `GASExperiment`
# quantizes the model and the threshold together and exposes the result as
# `effective_model` and `quantization_scale`.
# :::

# %% [markdown]
# ### Compile and Run
#
# `qm.Reference()`, Qamomile's built-in simulator, takes the place of a quantum
# computer here. We compile the program for this engine and run one *shot*: the
# circuit is prepared and measured once, which gives one assignment.

# %%
endpoint = qm.Reference()

compiled = experiment.compile(to=endpoint.target)
run = compiled.run(on=endpoint, shots=1, seed=7)

sample = run.samples.lowest
bits = tuple(bit for _, bit in sample.assignment)
print(f"Measured assignment: {bits}")
print(f"Cost: {sample.energy}")

# %% [markdown]
# `run.samples` holds the measured assignments. With one shot it holds one
# sample, which `run.samples.lowest` returns. Its `assignment` pairs each
# variable with its value, and its `energy` is the cost of the assignment.

# %% [markdown]
# ## Result
#
# ### One Round at a Fixed Threshold
#
# At $y = 0$, 104 of the 512 assignments have a cost below the threshold. We
# repeat the round 256 times, one shot each, for 0, 1 and 2 iterations, and
# compare the share of rounds below the threshold with $P_r$. The threshold and
# the number of iterations change the circuit, so each pair needs its own
# compiled program. `compiled_round` builds each of them once and reuses it.

# %%
NUM_ROUNDS = 4 if docs_test_mode else 256


@functools.cache
def compiled_round(threshold: float, iterations: int):
    """Compile the round for one threshold and one number of iterations."""
    round_experiment = GASExperiment(model, threshold=threshold, iterations=iterations)
    return round_experiment.compile(to=endpoint.target)


def run_round(threshold: float, iterations: int, seed: int):
    """Run one round with one shot and return the measured assignment and its cost."""
    run = compiled_round(threshold, iterations).run(on=endpoint, shots=1, seed=seed)
    sample = run.samples.lowest
    return tuple(bit for _, bit in sample.assignment), sample.energy


num_marked = sum(cost < threshold for cost in costs.values())
theta = math.asin(math.sqrt(num_marked / len(costs)))

for iterations in range(3):
    measured = [run_round(threshold, iterations, seed)[1] for seed in range(NUM_ROUNDS)]
    below = sum(cost < threshold for cost in measured)
    predicted = math.sin((2 * iterations + 1) * theta) ** 2
    print(
        f"r = {iterations}: {below} of {NUM_ROUNDS} rounds below the threshold, "
        f"share {below / NUM_ROUNDS:.3f}, predicted {predicted:.3f}"
    )
    if not docs_test_mode:
        assert abs(below / NUM_ROUNDS - predicted) < 0.1

# %% [markdown]
# With no iteration, the round picks an assignment at random, and the share of
# rounds below the threshold is 0.215. One iteration raises the share to 0.965,
# and a second iteration lowers it to 0.559, as $P_r$ predicts. The best number
# of iterations depends on how many assignments lie below the threshold, which
# the loop does not know.

# %% [markdown]
# ### Run the Adaptive Loop
#
# We now run the loop of the Algorithm section. Each round is one call of
# `run_round`. The loop starts from buying nothing, draws the number of
# iterations at random, and stops after `MAX_FAILURES` rounds in a row without
# improvement.

# %%
GROWTH = 1.2  # The factor lambda of the range after a round without improvement.
MAX_FAILURES = 2 if docs_test_mode else 20


def adaptive_search(seed: int):
    """Run the adaptive loop and return the best assignment, its cost and the history."""
    rng = np.random.default_rng(seed)
    best_bits = (0,) * NUM_ASSETS  # Start from buying nothing.
    threshold = model.energy(best_bits)
    k, failures, rounds = 1.0, 0, 0
    history = []

    while failures < MAX_FAILURES:
        rounds += 1
        # Draw the number of iterations uniformly from 0, ..., ceil(k) - 1.
        iterations = int(rng.integers(math.ceil(k)))
        bits, cost = run_round(threshold, iterations, seed=int(rng.integers(2**31)))
        if cost < threshold:
            best_bits, threshold = bits, cost
            history.append((rounds, iterations, threshold))
            k, failures = 1.0, 0
        else:
            k *= GROWTH
            failures += 1

    return best_bits, threshold, history, rounds


best_bits, best_cost, history, num_rounds = adaptive_search(seed=0)

for round_number, iterations, new_threshold in history:
    print(f"Round {round_number}: r = {iterations}, new threshold {new_threshold}")
print(f"Stopped after {num_rounds} rounds")
print(f"Best assignment: {best_bits}, cost {best_cost}")

assert best_cost == model.energy(best_bits)
if not docs_test_mode:
    assert best_cost == min_cost

# %% [markdown]
# The threshold falls from 0 to $-4$, $-11$, $-13$, $-16$, $-24$, $-25$ and
# finally $-28$ in round 50. The next 20 rounds find no improvement, which ends
# the loop at round 70. The result is the only optimal assignment. The loop
# does not use the exact minimum, and we only compare with it at the end.
# A run can also stop above the minimum: with `MAX_FAILURES = 20`, a
# calculation with $P_r$ gives the minimum in about 997 of 1,000 runs of this
# instance.

# %% [markdown]
# ## Summary
#
# In this tutorial, we solved a portfolio selection problem with Grover
# Adaptive Search:
#
# 1. We wrote the cost as a `BinaryModel` and found the exact minimum by trying
#    every assignment.
# 2. We built one round for a fixed threshold with `GASExperiment`, and saw
#    that the share of rounds below the threshold follows
#    $\sin^2((2r+1)\,\theta)$.
# 3. We ran the adaptive loop, which lowers the threshold after every
#    improvement and draws the number of iterations at random, and it reached
#    the exact minimum.
