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
# Grover Adaptive Search (GAS) minimizes a polynomial objective over binary
# variables by repeatedly asking one question of a Grover oracle — *which $x$
# satisfy $f(x) < y$?* — and lowering the threshold $y$ whenever a better
# solution is found {cite:p}`10.22331/q-2021-04-08-428`.
#
# This page solves an unconstrained **portfolio selection** problem with
# Qamomile's immutable `BinaryModel` and `GASExperiment`.
#
# The tutorial is built on the following structure:
#
# 1. Formulate the polynomial objective as a `BinaryModel`.
# 2. Create the nine-asset instance with concrete data.
# 3. Use `GASExperiment` to build the Grover circuit for the current threshold.
# 4. Sample it, keep the best candidate, and repeat until a stopping criterion
#    is reached. GAS is a hybrid loop, not a single circuit.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit]"

# %%
import itertools
import math
import os

import numpy as np

import qamomile as qm
from qamomile.optimization import BinaryModel, GASExperiment
from qamomile.qiskit import Aer
from qamomile.stdlib import grover_iteration_count

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Background
#
# Grover search takes an oracle that distinguishes target states by reversing
# their phase and uses amplitude amplification to raise their measurement
# probability. That solves a decision problem: a fixed condition says which
# states qualify, not which one is best.
#
# GAS turns it into minimization by changing this condition. It distinguishes
# the quantum states corresponding to candidates with $f(x) < y$ by reversing
# their phase and samples candidates. If a candidate has a lower objective
# value than the current threshold, the classical layer updates $y$ to that
# value. This process repeats, with the quantum circuit answering one fixed
# question at a time and the classical layer updating the threshold.


# %% [markdown]
# ## Problem Settings
#
# In this tutorial, we apply GAS to a portfolio selection problem.
#
# Given $n$ assets (stocks, bonds, etc.), we decide for each one whether to buy
# it — a binary choice. We want to maximize returns while minimizing risk. The
# tension is:
#
# - $\mu$ tells us how profitable each asset is expected to be
# - $\Sigma$ tells us how assets move together (correlated assets amplify risk)
# - $q$ controls how much we care about risk vs. return
#
# We need to pick the best subset of assets so that the portfolio earns as much
# as possible without taking on too much correlated risk.
#
# \begin{equation}
# \min_{x\in \lbrace 0 , 1 \rbrace^n} \big( q x^T \Sigma x - \mu^T x \big)
# \end{equation}


# %% [markdown]
# The instance below has 9 assets:
#
# | Name    | Expected Return | Variance   |
# |---------|-----------------|------------|
# | Asset 1 | 22              | 12         |
# | Asset 2 | 4               | 15         |
# | Asset 3 | 19              | 10         |
# | Asset 4 | 3               | 18         |
# | Asset 5 | 23              | 14         |
# | Asset 6 | 2               | 20         |
# | Asset 7 | 5               | 11         |
# | Asset 8 | 25              | 16         |
# | Asset 9 | 3               | 13         |


# %% [markdown]
# The coefficient table below produces the same immutable binary model directly.

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
risk_aversion = 1


def portfolio_model(num_assets: int) -> BinaryModel:
    terms: dict[tuple[int, ...], float] = {}
    for left in range(num_assets):
        terms[(left,)] = float(
            risk_aversion * covariance[left, left] - expected_returns[left]
        )
        for right in range(left + 1, num_assets):
            coefficient = float(2 * risk_aversion * covariance[left, right])
            if coefficient:
                terms[(left, right)] = coefficient
    return BinaryModel.from_hubo(terms, variables=tuple(range(num_assets)))


full_model = portfolio_model(9)
MODEL_ASSETS = 3 if docs_test_mode else 9
model = portfolio_model(MODEL_ASSETS)

assert full_model.num_variables == 9
assert model.num_variables == MODEL_ASSETS

energies = {
    bits: model.energy(bits)
    for bits in itertools.product((0, 1), repeat=model.num_variables)
}
optimal_bits, optimal_energy = min(energies.items(), key=lambda item: item[1])

print("exact optimum:", optimal_bits, optimal_energy)
assert optimal_bits == ((1, 0, 1) if docs_test_mode else (1, 0, 0, 0, 1, 0, 0, 1, 0))
assert optimal_energy == (-11.0 if docs_test_mode else -28.0)

full_energies = {
    bits: full_model.energy(bits)
    for bits in itertools.product((0, 1), repeat=full_model.num_variables)
}
full_optimal_bits, full_optimal_energy = min(
    full_energies.items(), key=lambda item: item[1]
)
print("nine-asset exact optimum:", full_optimal_bits, full_optimal_energy)
assert full_optimal_bits == (1, 0, 0, 0, 1, 0, 0, 1, 0)
assert full_optimal_energy == -28.0

# %% [markdown]
# ## Algorithm
#
# GAS starts by setting the threshold $y$ to the objective value of a candidate
# solution and uses Grover search to find candidates with lower objective values.
# Candidates obtained from the quantum circuit are evaluated classically, and
# the threshold is updated whenever a better candidate is found. This process
# repeats until the stopping condition is met. Here $y$ is a threshold on the
# objective value, not a candidate solution.
# GAS reverses the phase of the quantum states corresponding to candidates
# that satisfy $f(x) < y$, distinguishing them from the other candidates.
# The Grover ansatz is built from three components:
#
# - $A_y$, the preparation operator. It associates each input with $f(x) - y$
#   by building the quantum dictionary state $\sum_x \ket{x, f(x) - y}$.
#   The circuit implementation uses the QFT-based method given in
#   {cite:p}`10.22331/q-2021-04-08-428`. Because the register holds $f(x) - y$
#   in two's complement, candidates satisfying the condition can be identified
#   by a Most Significant Bit (MSB) of $1$.
# - $O_y$, the phase oracle. It reverses the phase of candidates satisfying
#   $f(x) < y$. The MSB indicates whether the encoded value $f(x) - y$ is
#   negative, so this can be implemented with a single $Z$ gate on that MSB.
# - $D$, the diffusion operator, which amplifies the amplitude of the states
#   distinguished by their reversed phase. It is a single multi-controlled-$Z$
#   sandwiched between $X$ layers.
#
# One iteration applies $O_y$, then $A_y^\dagger$, $D$, $A_y$. The phase flip on
# its own leaves every measurement probability unchanged; the reflection
# $A_y D A_y^\dagger$ is what converts it into amplitude. Measuring the input
# register returns a candidate, which the classical layer evaluates. It updates
# $y$ to the candidate's objective value only if that value is lower than the
# current threshold.
#
# The remaining question is how many times to apply the Grover operator. GAS
# answers it by sampling the iteration count from a range that grows slowly
# whenever a round brings no improvement.


# %% [markdown]
# ## Build one fixed-threshold round
#
# Suppose the current classical candidate has energy $-10$. The next oracle must
# therefore mark assignments with $f(x) < -10$. In documentation-test mode,
# only the three-asset optimum qualifies. Normal execution uses the original
# nine-asset problem and the same fixed-threshold round.
#
# `GASExperiment` is a host-side experiment boundary, not a second quantum
# decorator. It closes the model and threshold into a verified `Program`.
# Its circuit uses ordinary Qamomile composition and reversible integer
# arithmetic; there is no GAS-specific IR operation or backend implementation.

# %%
threshold = -10.0
experiment = GASExperiment(
    model,
    threshold=threshold,
    iterations=1,
)

print("output bits:", experiment.resolved_output_bits)
print("program digest:", experiment.program.semantic_digest)

inspection = experiment.program.inspect()
assert "quantum.gas" not in inspection
assert "qcore.control_call" in inspection

# %% [markdown]
# ### Reproduce amplitude-amplification scaling
#
# Section 14.1 of [](https://doi.org/10.48550/arXiv.2310.03011) gives
# $m=O(a^{-1})$ Grover iterations for initial success amplitude $a$. With one
# marked item among $N=2^n$, $a=N^{-1/2}$ and therefore $m=O(\sqrt N)$.
# Qamomile's bounded planner returns the usual
# $\lfloor(\pi/4)\sqrt{2^n}\rfloor$ count without a floating-point overflow.

# %%
grover_plans = {width: grover_iteration_count(width) for width in range(2, 11)}
for width, iterations in grover_plans.items():
    assert iterations == math.floor((math.pi / 4.0) * math.sqrt(2**width))

print("Grover plans (n: iterations):", grover_plans)

# %% [markdown]
# For a fixed GAS threshold, the full state preparation, oracle, inverse, and
# reflection body is repeated once per Grover iteration. The compiler derives
# exact semantic work from that body. Gate-family upper bounds are intentionally
# left open because controlled coherent-arithmetic transforms do not yet have a
# complete logical gate profile. `GASExperiment` deliberately exposes a closed
# program, so two endpoint studies determine the affine intercept and slope;
# the remaining displayed rows are arithmetic, not repeated compiler analyses.

# %%
gas_endpoint_studies = tuple(
    qm.resources.logical(
        GASExperiment(model, threshold=threshold, iterations=iterations).program
    )
    for iterations in (0, 1)
)
gas_endpoint_work = tuple(
    study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure()
    for study in gas_endpoint_studies
)
gas_endpoint_gate_bounds = tuple(
    study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
    for study in gas_endpoint_studies
)

assert all(lower == upper for lower, upper in gas_endpoint_work)
assert all(upper is None for _, upper in gas_endpoint_gate_bounds)
work_base = gas_endpoint_work[0][0]
work_step = gas_endpoint_work[1][0] - work_base
gate_base = gas_endpoint_gate_bounds[0][0]
gate_step = gas_endpoint_gate_bounds[1][0] - gate_base
gas_work = tuple(
    (work_base + iterations * work_step, work_base + iterations * work_step)
    for iterations in range(4)
)
gas_gate_bounds = tuple(
    (gate_base + iterations * gate_step, None) for iterations in range(4)
)

print("GAS semantic work for 0..3 iterations:", gas_work)
print("GAS logical-gate bounds for 0..3 iterations:", gas_gate_bounds)

# %% [markdown]
# The loop structure gives an affine cost in $m$ for this concrete Grover body;
# the two studies supply its finite coefficients. Composing it with
# $m=\Theta(\sqrt N)$ reproduces the survey's
# query scaling. The partial gate enclosures are also useful evidence: Qamomile
# reports the missing transform profile by name instead of silently treating it
# as zero or as an exact target gate count.

# %% [markdown]
# Integer coefficients are encoded exactly. For real coefficients,
# `GASExperiment` quantizes the complete model and threshold together and
# exposes both `effective_model` and `quantization_scale`. Set
# `approximate_real_coefficients=False` to reject noninteger arithmetic instead.
# `output_bits` is normally inferred from a safe two's-complement bound; an
# explicitly undersized register is rejected rather than silently wrapping.

# %% [markdown]
# ## Compile once and sample
#
# The experiment follows the same lifecycle as the rest of Qamomile. Compile
# its closed program for the endpoint target, then reuse the compiled artifact.
# Documentation tests use the built-in reference endpoint. Normal execution
# follows the original tutorial through local Qiskit Aer.

# %%
endpoint = qm.Reference() if docs_test_mode else Aer()
compiled = experiment.compile(to=endpoint.target)
shots = 32 if docs_test_mode else 256
run = compiled.run(on=endpoint, shots=shots, seed=7)

for sample in run.samples.samples:
    print(sample.assignment, sample.energy, sample.occurrences)

best = run.samples.lowest
assert best.assignment == tuple(enumerate(optimal_bits))
assert best.energy == optimal_energy
if docs_test_mode:
    assert best.occurrences > shots * 0.75
assert run.execution.artifact is compiled.artifact
assert run.identity == compiled.identity == experiment.identity

# %% [markdown]
# `run.samples` is a typed `DecodedSamples` value. It preserves original
# variable IDs, objective values, and multiplicities; backend adapters do not
# reinterpret the optimization model.
#
# To see the amplification rather than only its final answer, compare the same
# threshold with zero Grover iterations. That program prepares the uniform
# decision superposition but performs no amplification.

# %%
baseline = GASExperiment(model, threshold=threshold, iterations=0).run(
    on=endpoint,
    shots=shots,
    seed=7,
)

baseline_optimum_count = next(
    (
        sample.occurrences
        for sample in baseline.samples.samples
        if sample.assignment == best.assignment
    ),
    0,
)

print("uniform optimum count:", baseline_optimum_count)
print("amplified optimum count:", best.occurrences)
assert best.occurrences > baseline_optimum_count

# %% [markdown]
# ## The adaptive outer loop
#
# GAS is adaptive because the host chooses a new threshold after each sampled
# round. A typical loop performs these steps:
#
# 1. evaluate the current classical candidate and use its energy as `threshold`;
# 2. choose a bounded Grover iteration count;
# 3. create and compile the corresponding immutable `GASExperiment`;
# 4. sample it, inspect `run.samples.lowest`, and lower the threshold only when
#    the candidate improves it.
#
# Threshold and iteration count change circuit structure, so they are explicit
# experiment configuration rather than launch-time `Param` values. Within one
# round, the resulting artifact is reusable and retains an exact problem/program
# identity.


# %%
def adaptive_search(
    binary_model: BinaryModel,
    *,
    initial_threshold: float,
    max_rounds: int,
    shots_per_round: int,
) -> tuple[tuple[tuple[int, int], ...], float, tuple[float, ...]]:
    threshold = initial_threshold
    best_assignment = tuple((index, 0) for index in range(binary_model.num_variables))
    history = [threshold]

    for round_index in range(max_rounds):
        round_experiment = GASExperiment(
            binary_model,
            threshold=threshold,
            iterations=1,
        )
        round_result = round_experiment.run(
            on=endpoint,
            shots=shots_per_round,
            seed=11 + round_index,
        )
        candidate = round_result.samples.lowest
        if candidate.energy < threshold:
            threshold = candidate.energy
            best_assignment = candidate.assignment
        history.append(threshold)
        if threshold == optimal_energy:
            break

    return best_assignment, threshold, tuple(history)


adaptive_assignment, adaptive_energy, threshold_history = adaptive_search(
    model,
    initial_threshold=0.0,
    max_rounds=4,
    shots_per_round=shots,
)

print("threshold history:", threshold_history)
assert adaptive_assignment == tuple(enumerate(optimal_bits))
assert adaptive_energy == optimal_energy
assert all(
    next_threshold <= current_threshold
    for current_threshold, next_threshold in zip(
        threshold_history,
        threshold_history[1:],
    )
)

# %% [markdown]
# The executable search reaches the exact optimum and records a monotonically
# non-increasing threshold history. Normal execution uses the original
# nine-asset formulation throughout; documentation tests reduce only the model
# width and execution budget.

# %% [markdown]
# ## Summary
#
# In this notebook, we:
#
# - Solved an unconstrained portfolio problem with Grover Adaptive Search,
#   where the quantum circuit answers "which $x$ satisfy $f(x) < y$?" and a
#   classical loop updates the threshold $y$ when a better candidate is found.
# - Used `GASExperiment` for the quantum part of GAS: `compile()` builds the
#   Grover circuit for the current threshold and number of Grover iterations,
#   and `run()` decodes sampled bitstrings in the original variable order.
# - Ran GAS on a portfolio selection problem and verified that the resulting
#   objective value matched the optimum obtained by brute-force search.
