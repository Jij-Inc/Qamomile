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
# # Channel Assignment with Grover Adaptive Search
#
# This page solves the wireless **Channel Assignment Problem (CAP)** with
# Qamomile's `GASConverter`, following the higher-order binary formulation of
# Sano, Norimoto, and Ishikawa [](https://arxiv.org/abs/2208.05181). Grover
# Adaptive Search (GAS) itself {cite:p}`10.22331/q-2021-04-08-428` is
# introduced in the [Grover Adaptive Search tutorial](grover_adaptive_search);
# this page focuses on what changes for a constrained problem with real-valued
# coefficients.
#
# We will:
#
# 1. Formulate the CAP twice with
#    [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html):
#    as a QUBO with one-hot encoding and as a HUBO with binary encoding.
# 2. Build both instances from the paper's example network.
# 3. Compare how many qubits each formulation needs in the GAS circuit.
# 4. Run GAS on the HUBO and check the result against brute force.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit]"

# %%
import itertools
import os
import random
import warnings

import jijmodeling as jm
import numpy as np
import ommx.v1
from qiskit_aer import AerSimulator

from qamomile.optimization.gas import GASConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# A wireless network has $N_{AP}$ access points (APs) that must share
# $N_{CH} < N_{AP}$ channels. Each AP serves a set $U_i$ of user terminals
# (UTs). When two APs use the same channel, each one interferes with the
# other's UTs, which lowers the total Shannon capacity of the network. The CAP
# asks for the channel assignment that maximizes this capacity, which is
# equivalent to minimizing the total interference between APs that share a
# channel.
#
# ```{figure} assets/gas_channel_assignment_system_model.png
# :width: 80%
# :alt: Four access points, each serving two user terminals, with overlapping coverage areas.
#
# The example network used on this page: 4 APs, 8 UTs, and distances
# $d_{iu}$ between AP $i$ and UT $u$. Reproduced from Fig. 2 of
# [](https://arxiv.org/abs/2208.05181).
# ```

# %% [markdown]
# ## Problem Settings
#
# We use the example from Appendix A of the paper: $N_{AP} = 4$ APs,
# $N_{CH} = 3$ channels, and 8 UTs, two per AP. With a simple path-loss model
# (attenuation coefficient $\alpha = 1$), the interference cost between APs
# $i$ and $k$ is
#
# $$
# C_{ik} = -\log_2\left(1 + \frac{\sum_{u \in U_i} d_{iu}^{-\alpha}}{\sum_{v \in U_k} d_{iv}^{-\alpha}}\right)
#          -\log_2\left(1 + \frac{\sum_{u \in U_k} d_{ku}^{-\alpha}}{\sum_{v \in U_i} d_{kv}^{-\alpha}}\right),
# $$
#
# and the paper shifts it to strictly positive weights
# $D_{ik} = C_{ik} - C_{\min} + \epsilon$ with $\epsilon = 0.01$, where
# $C_{\min}$ is the smallest $C_{ik}$.

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

N_AP = 4
N_CH = 3
alpha = 1
epsilon = 0.01

# UTs served by each AP
connections = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
}

# Distance between AP i (row) and UT u (column)
d = np.array([
    [1, 1, 2, 4, 3, 5, 5, 8],
    [5, 4, 1, 1, 5, 4, 2, 4],
    [6, 5, 2, 5, 1, 1, 4, 6],
    [10, 8, 5, 2, 5, 3, 1, 1],
], dtype=float)

upper_idx = np.triu_indices(N_AP, k=1)
C = np.zeros((N_AP, N_AP))
for i, k in zip(*upper_idx):
    C[i, k] = -np.log2(
        1
        + np.sum(d[i, connections[i]] ** -alpha)
        / np.sum(d[i, connections[k]] ** -alpha)
    ) - np.log2(
        1
        + np.sum(d[k, connections[k]] ** -alpha)
        / np.sum(d[k, connections[i]] ** -alpha)
    )

D = np.zeros_like(C)
D[upper_idx] = C[upper_idx] - np.min(C[upper_idx]) + epsilon
D = np.round(D, decimals=3)
print(D)

# The paper lists these weights in Appendix A.
paper_D = {(0, 1): 1.835, (0, 2): 1.216, (0, 3): 0.010,
           (1, 2): 1.762, (1, 3): 2.333, (2, 3): 1.371}
for (i, k), value in paper_D.items():
    assert np.isclose(D[i, k], value, atol=1e-3, rtol=0.0), (i, k, D[i, k])

# %% [markdown]
# ## Algorithm
#
# GAS needs the objective as a polynomial in binary variables. The paper
# compares two encodings of "AP $i$ uses channel $c$".
#
# ### QUBO: one-hot encoding
#
# Let $x_{ic} = 1$ if AP $i$ uses channel $c$ and $0$ otherwise, so each row
# $[x_{i1}, \ldots, x_{iN_{CH}}]$ is one-hot. Two APs $i < k$ interfere when
# they pick the same channel, which gives
#
# $$
# \min_x \sum_{i < k} D_{ik} \sum_{c=1}^{N_{CH}} x_{ic} x_{kc}
# \quad\text{s.t.}\quad \sum_{c=1}^{N_{CH}} x_{ic} = 1 \;\; \forall i .
# $$
#
# This needs $N_{AP} N_{CH}$ binary variables.

# %%
@jm.Problem.define("Channel Assignment (QUBO)", sense=jm.ProblemSense.MINIMIZE)
def cap_qubo(problem: jm.DecoratedProblem):
    N_AP = problem.Length("N_AP")
    N_CH = problem.Length("N_CH")
    D = problem.Float("D", shape=(N_AP, N_AP), description="Pairwise interference weights")
    x = problem.BinaryVar(
        "x",
        shape=(N_AP, N_CH),
        description="1 if AP i uses channel c",
    )

    problem += jm.sum(
        D[i, k] * x[i, c] * x[k, c]
        for i in N_AP
        for k in N_AP
        if k > i
        for c in N_CH
    )

    problem += problem.Constraint(
        "one channel per AP",
        (jm.sum(x[i, c] for c in N_CH) == 1 for i in N_AP),
    )


cap_qubo

# %% [markdown]
# ### HUBO: binary encoding
#
# The binary encoding writes each AP's channel index as an $N_B$-bit codeword,
# with $N_B = \lceil \log_2 N_{CH} \rceil$. Let $b_c \in \{0, 1\}^{N_B}$ be the
# codeword of channel $c$ and $x_{ir}$ bit $r$ of AP $i$'s codeword. The
# polynomial
#
# $$
# \delta'_{ic}(x) = \prod_{r=1}^{N_B} \bigl(1 - b_{cr} + (2 b_{cr} - 1)\, x_{ir}\bigr)
# $$
#
# is a product of one factor per bit, where each factor is $x_{ir}$ if
# $b_{cr} = 1$ and $1 - x_{ir}$ if $b_{cr} = 0$. It therefore equals $1$
# exactly when AP $i$'s codeword is $b_c$, and $0$ otherwise. The objective
# becomes
#
# $$
# \min_x \sum_{i < k} D_{ik} \sum_{c=1}^{N_{CH}} \delta'_{ic}(x)\, \delta'_{kc}(x)
# \quad\text{s.t.}\quad \delta'_{ic}(x) = 0 \;\; \forall i,\; N_{CH} < c \le 2^{N_B} .
# $$
#
# The constraint forbids codewords that do not correspond to a real channel
# ($N_{CH} = 3$ leaves one of the four 2-bit codewords unused). The products
# make this a higher-order (HUBO) polynomial, but it needs only
# $N_{AP} N_B$ variables.

# %%
@jm.Problem.define("Channel Assignment (HUBO)", sense=jm.ProblemSense.MINIMIZE)
def cap_hubo(problem: jm.DecoratedProblem):
    N_AP = problem.Length("N_AP")
    N_CH = problem.Length("N_CH")
    N_TOT = problem.Length("N_TOT")
    N_B = problem.Length("N_B")
    b = problem.Float("b", shape=(N_TOT, N_B), description="Codeword table")
    D = problem.Float("D", shape=(N_AP, N_AP), description="Pairwise interference weights")
    x = problem.BinaryVar(
        "x",
        shape=(N_AP, N_B),
        description="Bit r of AP i's channel codeword",
    )

    def delta(i, c):
        return jm.prod((1 - b[c, r] + (2 * b[c, r] - 1) * x[i, r]) for r in N_B)

    problem += jm.sum(
        D[i, k] * delta(i, c) * delta(k, c)
        for i in N_AP
        for k in N_AP
        if k > i
        for c in N_CH
    )

    problem += problem.Constraint(
        "no nonexistent channel",
        (delta(i, c) == 0 for i in N_AP for c in N_TOT if c >= N_CH),
    )


cap_hubo

# %% [markdown]
# Evaluating both problems with the network data gives two OMMX instances.

# %%
N_B = int(np.ceil(np.log2(N_CH)))
N_TOT = 2**N_B
codewords = np.array(
    [[int(bit) for bit in np.binary_repr(c, width=N_B)] for c in range(N_TOT)]
)

instance_qubo = cap_qubo.eval({"N_AP": N_AP, "N_CH": N_CH, "D": D.tolist()})
instance_hubo = cap_hubo.eval({
    "N_AP": N_AP,
    "N_CH": N_CH,
    "N_TOT": N_TOT,
    "N_B": N_B,
    "b": codewords.tolist(),
    "D": D.tolist(),
})

assert len(instance_qubo.used_decision_variable_ids()) == N_AP * N_CH == 12
assert len(instance_hubo.used_decision_variable_ids()) == N_AP * N_B == 8

# %% [markdown]
# Both instances are small enough to enumerate. Brute force gives the
# reference optimum that GAS should reach. The two formulations describe the
# same $N_{CH}^{N_{AP}} = 81$ assignments, so they share the optimum.


# %%
def brute_force(instance: ommx.v1.Instance) -> tuple[list[int], float, int]:
    """Return the best feasible bitstring, its objective, and the feasible count."""
    ids = sorted(instance.used_decision_variable_ids())
    feasible = []
    for bits in itertools.product([0, 1], repeat=len(ids)):
        solution = instance.evaluate(dict(zip(ids, bits)))
        if solution.feasible:
            feasible.append((solution.objective, list(bits)))
    best_y, best_x = min(feasible)
    return best_x, best_y, len(feasible)


qubo_x, qubo_y, qubo_feasible = brute_force(instance_qubo)
hubo_x, hubo_y, hubo_feasible = brute_force(instance_hubo)
print(f"QUBO: optimum {qubo_y:.3f} at x={qubo_x} ({qubo_feasible} feasible)")
print(f"HUBO: optimum {hubo_y:.3f} at x={hubo_x} ({hubo_feasible} feasible)")

assert qubo_feasible == hubo_feasible == N_CH**N_AP
assert np.isclose(qubo_y, hubo_y, atol=1e-9, rtol=0.0)
# With 4 APs and 3 channels, at least one pair must share a channel. The
# cheapest pair is APs 1 and 4 (D_14 = 0.010), so the optimum shares only that one.
assert np.isclose(hubo_y, paper_D[(0, 3)], atol=1e-9, rtol=0.0)

# %% [markdown]
# ## Implementation
#
# ### Real-valued coefficients
#
# The GAS circuit stores $f(x) - y$ in a register of qubits, using QFT-based
# arithmetic. With integer coefficients, the register holds this value exactly.
# The CAP weights $D_{ik}$ are real numbers, and `GASConverter.transpile()`
# offers the two options discussed in the appendix of
# {cite:p}`10.22331/q-2021-04-08-428`:
#
# - `approximate_real_coefficients=False` encodes the real coefficients as
#   they are. The register then holds an approximation of $f(x) - y$, whose
#   amplitude concentrates on the nearest representable values. This needs
#   no preprocessing and the fewest qubits.
# - `approximate_real_coefficients=True` (the default) rescales and rounds
#   the coefficients to integers first, so the arithmetic is exact for the
#   rounded problem. `quantization_parameter` sets the number of bits of the
#   rounding. When it is `None`, Qamomile picks a value automatically, at the
#   cost of more qubits. A small manual value saves qubits but can merge
#   distinct objective values, so the true optimum may no longer be
#   distinguishable.
#
# The output register must hold $f(x) - y$ for every $x$, so its width depends
# on the model and on the threshold $y$. After `transpile()`, the public
# `required_output_bits(y)` reports that width for the model actually encoded.


# %%
def count_qubits(
    instance: ommx.v1.Instance,
    y: float,
    approximate_real_coefficients: bool,
    quantization_parameter: int | None = None,
) -> int:
    """Return the total number of qubits of the GAS circuit for threshold ``y``."""
    converter = GASConverter(instance)
    with warnings.catch_warnings():
        # transpile() warns that it rounds real coefficients; that is the
        # point of this comparison.
        warnings.simplefilter("ignore", UserWarning)
        converter.transpile(
            QiskitTranspiler(),
            y=y,
            num_iterations=0,
            approximate_real_coefficients=approximate_real_coefficients,
            quantization_parameter=quantization_parameter,
        )
    return converter.binary_model.num_bits + converter.required_output_bits(y)


settings = {
    "real coefficients": {"approximate_real_coefficients": False},
    "rounded, 8 bits": {"approximate_real_coefficients": True, "quantization_parameter": 8},
    "rounded, automatic": {"approximate_real_coefficients": True},
}
# A threshold of the size GAS starts from: the objective of a mid-range assignment.
y_example = 3.549
qubit_counts = {
    name: {
        label: count_qubits(instance, y_example, **kwargs)
        for label, kwargs in settings.items()
    }
    for name, instance in (("QUBO", instance_qubo), ("HUBO", instance_hubo))
}

print(f"{'':6}" + "".join(f"{label:>22}" for label in settings))
for name, counts in qubit_counts.items():
    print(f"{name:6}" + "".join(f"{counts[label]:>22}" for label in settings))

assert qubit_counts["QUBO"] == {
    "real coefficients": 19, "rounded, 8 bits": 24, "rounded, automatic": 36,
}
assert qubit_counts["HUBO"] == {
    "real coefficients": 16, "rounded, 8 bits": 20, "rounded, automatic": 35,
}

# %% [markdown]
# The HUBO saves $N_{AP}(N_{CH} - N_B) = 4$ input qubits over the QUBO, and its
# search space shrinks from $2^{12}$ to $2^{8}$ bitstrings. Each extra qubit
# doubles the cost of classical simulation, so on this page we run GAS on the
# HUBO with real coefficients, which needs 16 qubits.
#
# ### The classical layer
#
# The loop below is the one from the
# [Grover Adaptive Search tutorial](grover_adaptive_search), with three
# changes for this problem:
#
# - It starts from a given assignment instead of a random bitstring, because
#   a random bitstring can violate the constraint.
# - It passes the real-coefficient options through to `transpile()`.
# - It reads each candidate's bits by decision-variable ID. The HUBO variable
#   `x` is two-dimensional, so the bits have to be read in the same order as
#   the IDs the converter uses.
#
# `converter.decode()` evaluates every sampled bitstring against the OMMX
# instance, so `best_feasible` is the best sample that satisfies the
# constraints.


# %%
def bits_from_solution(solution: ommx.v1.Solution, ids: list[int]) -> list[int]:
    """Read a solution's binary values in decision-variable ID order."""
    return [int(round(solution.state.entries[i])) for i in ids]


def grover_adaptive_search(
    converter: GASConverter,
    transpiler: QiskitTranspiler,
    initial_x: list[int],
    lamb: float,
    max_no_improvement: int = 5,
    shots: int = 256,
    seed: int = 900,
    approximate_real_coefficients: bool = False,
    quantization_parameter: int | None = None,
) -> tuple[list[int], float]:
    ##########################################################
    #                   Initialization                       #
    ##########################################################

    rng = random.Random(seed)
    ids = sorted(converter.instance.used_decision_variable_ids())
    k = 1  # Controls the upper bound of the Grover iteration count sampled per step
    x = list(initial_x)
    y = converter.instance.evaluate(dict(zip(ids, x))).objective

    current_iter = 0
    no_improvement_count = 0

    print("[GAS] Initialization")
    print(f"[GAS] n={len(ids)}, lambda={lamb}")
    print(f"[GAS] Start state: x={x}, y={y:.3f}, k={k}")

    executor = transpiler.executor(
        backend=AerSimulator(seed_simulator=seed, max_parallel_threads=None)
    )

    ############################################################
    #                     Main Loop                            #
    ############################################################

    while no_improvement_count < max_no_improvement:
        # Sample t uniformly in {0, ..., ceil(k)-1}; avoid empty range when k == 1
        num_iterations = rng.randrange(max(1, int(np.ceil(k))))
        print(
            f"\n[GAS] Iteration {current_iter + 1} | current y={y:.3f}, k={k:.6f}, "
            f"Grover iters={num_iterations}"
        )

        ####################################################
        #       Call to the Quantum Grover Circuit         #
        ####################################################

        executable = converter.transpile(
            transpiler,
            y=y,
            num_iterations=num_iterations,
            approximate_real_coefficients=approximate_real_coefficients,
            quantization_parameter=quantization_parameter,
        )
        result = executable.sample(executor, shots=shots).result()
        sample_set = converter.decode(result)

        try:
            best = sample_set.best_feasible
        except RuntimeError:
            best = None

        if best is not None and best.objective < y:
            x = bits_from_solution(best, ids)
            y = best.objective
            k = 1
            no_improvement_count = 0
            print(f"[GAS] Improvement: x={x}, y={y:.3f} -> resetting k to 1")
        else:
            old_k = k
            k = lamb * k
            no_improvement_count += 1
            print(
                f"[GAS] No improvement -> keeping current solution and scaling "
                f"k: {old_k:.6f} -> {k:.6f}"
            )

        current_iter += 1

    print(f"\n[GAS] Finished after {current_iter} iterations. Best found: x={x}, y={y:.3f}")

    return x, y


# %% [markdown]
# ## Result
#
# GAS starts from a random valid assignment: each AP gets a random existing
# channel, written as its codeword.

# %%
seed = 3 if docs_test_mode else 900
start_rng = np.random.default_rng(seed)
start_channels = start_rng.integers(0, N_CH, size=N_AP)
initial_x = [int(bit) for c in start_channels for bit in codewords[c]]
print(f"Start channels: {start_channels.tolist()} -> x={initial_x}")

converter = GASConverter(instance_hubo)
transpiler = QiskitTranspiler()

x, y = grover_adaptive_search(
    converter=converter,
    transpiler=transpiler,
    initial_x=initial_x,
    lamb=1.2,
    max_no_improvement=2 if docs_test_mode else 5,
    shots=16 if docs_test_mode else 256,
    seed=seed,
    approximate_real_coefficients=False,
)

hubo_ids = sorted(instance_hubo.used_decision_variable_ids())
solution = instance_hubo.evaluate(dict(zip(hubo_ids, x)))
assert solution.feasible, "GAS must return a valid channel assignment"
assert np.isclose(y, solution.objective, atol=1e-9, rtol=0.0), (
    "the reported objective must match the returned assignment"
)

# %% [markdown]
# Decoding the bits back to channels shows which APs share a channel, and the
# brute-force reference confirms the result.

# %%
channels = [
    int(np.flatnonzero((codewords == x[i * N_B:(i + 1) * N_B]).all(axis=1))[0])
    for i in range(N_AP)
]
print(f"Channel of each AP: {channels}")
print(f"GAS         : objective={y:.3f}")
print(f"Brute force : objective={hubo_y:.3f}")

assert np.isclose(y, hubo_y, atol=1e-9, rtol=0.0), (
    f"GAS returned objective {y}, but the true optimum is {hubo_y}"
)
print("\nGAS matched the brute-force optimum.")

# %% [markdown]
# :::{note}
# This instance has only $2^8 = 256$ bitstrings, so 256 shots already cover
# much of the search space, and a round with zero Grover iterations (uniform
# sampling) can find the optimum on its own. The example checks that the
# formulation, the circuit, and the decoding work end to end. The quadratic
# speedup of GAS matters for instances too large to simulate here.
# :::
#
# ## Summary
#
# In this notebook, we:
#
# - Formulated the wireless channel assignment problem with JijModeling, both
#   as a one-hot QUBO and as a binary-encoded HUBO, and reproduced the
#   interference weights published in the paper.
# - Used `GASConverter.transpile()` and `required_output_bits()` to compare
#   qubit counts: the HUBO needs 16 qubits with real coefficients against 19
#   for the QUBO, and rounding coefficients to integers adds more.
# - Ran GAS on the HUBO with real-valued coefficients and verified that it
#   reached the brute-force optimum.
