# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: qamomile (3.11.16.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, variational]
# ---
#
# # Warm-Starting QAOA
#
# Interest in using quantum algorithms to solve integer programming and combinatorial optimization problems has been growing steadily.
# One approach to these problems is the Quantum Approximate Optimization Algorithm (QAOA), which was inspired by a Trotterized form of adiabatic quantum computing.
# However, when QAOA requires a large number of layers $p$, its circuit depth also grows, making it difficult to run on NISQ devices.
# Classical solvers for these problems, meanwhile, often use relaxations that replace binary variables with continuous variables.
# For some problems, semidefinite programming (SDP) relaxations followed by rounding are believed under the Unique Games Conjecture to provide the best approximation ratios achievable classically in polynomial time.
# This page explains the warm-start approach discussed by [Egger et al. (2021)](https://quantum-journal.org/papers/q-2021-06-17-479/), which uses an initial state corresponding to a relaxation solution to reduce the QAOA depth required in practice.
# We will implement the method with Qamomile and compare it with standard QAOA.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import ising_cost, x_mixer
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# ### Problem: Limitations of QAOA
#
# Gate-based quantum computers are expected to help solve problems in fields such as quantum chemistry, machine learning, financial engineering, and combinatorial optimization.
# QAOA was proposed as a quantum algorithm for combinatorial optimization and has been considered for a wide range of applications (see, for example, [Farhi et al. (2014)](https://arxiv.org/abs/1411.4028)).
# However, QAOA has the following limitations:
#
# 1. Performance guarantees are known for particular problem settings, but there is no general performance guarantee.
# 2. [Hastings (2019)](https://arxiv.org/abs/1905.07047) showed that certain classical local algorithms can match QAOA's performance.
# 3. In practice, only shallow-depth QAOA circuits can be implemented on NISQ devices.
#
# QAOA therefore still faces both theoretical and practical challenges that limit its usefulness for real-world problem solving.
#
# ### Prior Work
#
# An important study related to the first two limitations is [Bravyi et al. (2020)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.260505).
# It showed that constant-depth QAOA cannot outperform Goemans-Williamson randomized rounding on certain MaxCut instances.
# Motivated by this limitation, Bravyi et al. also proposed Recursive QAOA (RQAOA).
# RQAOA sequentially reduces the problem size and was shown to outperform QAOA for certain forms of Ising Hamiltonians.
# In other words, RQAOA changes the outer structure of the algorithm by progressively eliminating variables.
# Warm-Starting QAOA (WS-QAOA), proposed by [Egger et al. (2021)](https://quantum-journal.org/papers/q-2021-06-17-479/), takes a different approach by modifying the initial state and mixer.
#
# ## Proposed Method
#
# ### Preparing an Initial State with a Continuous Relaxation
#
# The standard QUBO objective can be written as
#
# $$
# \min_{\boldsymbol{x} \in \{0, 1 \}^n} \boldsymbol{x}^\top \Sigma \boldsymbol{x} + \boldsymbol{\mu}^\top \boldsymbol{x} \tag{1}
# $$
#
# where $\boldsymbol{x}$ is a vector of $n$ binary variables, $\Sigma \in \mathbb{R}^{n \times n}$ is a symmetric matrix, and $\boldsymbol{\mu} \in \mathbb{R}^n$ is a real vector.
# Because binary variables satisfy $x_i^2 = x_i$, $\boldsymbol{\mu}$ can also be incorporated into the diagonal entries of $\Sigma$.
# If $\Sigma$ is positive semidefinite, we can relax the QUBO as follows:
#
# $$
# \min_{\boldsymbol{x} \in [0, 1]^n} \boldsymbol{x}^\top \Sigma \boldsymbol{x} + \boldsymbol{\mu}^\top \boldsymbol{x} \tag{2}
# $$
#
# This is a convex quadratic program (QP), and its optimum $\boldsymbol{c}^\ast$ can be obtained efficiently with classical optimization methods.
# If $\Sigma$ is not positive semidefinite, relaxing the problem directly does not produce a convex quadratic program.
# In that case, we can instead use a semidefinite programming (SDP) relaxation:
#
# $$
# \begin{align}
# &\max_{Y \in \mathbb{S}^{n \times n}} \ \mathrm{tr} \ (\Sigma Y) \\
# &\mathrm{s.t.} \quad \mathrm{diag} (Y) = \boldsymbol{e}, Y \succeq 0
# \end{align} \tag{3}
# $$
#
# Here, $Y \in \mathbb{S}^{n \times n}$ is an $n \times n$ symmetric matrix, $\boldsymbol{e}$ is the all-ones vector of length $n$, and $Y \succeq 0$ means that $Y$ must be positive semidefinite.
# Unlike the QP, the SDP produces a matrix $Y$ as its solution.
#
# ### Continuous Warm-Start QAOA
#
# WS-QAOA classically solves the relaxation in Eq. (2) or Eq. (3) and uses its solution to prepare the QAOA initial state.
# This page focuses on embedding the optimum $\boldsymbol{c}^\ast$ of Eq. (2) into the initial state.
# To use the SDP optimum $Y^\ast$ from Eq. (3), it must first be converted into an initial bit string or initial probabilities, for example through rounding.
# We embed the optimum $\boldsymbol{c}^\ast$ of Eq. (2) as
#
# $$
# \vert \phi^\ast \rangle
# = \bigotimes_{i=0}^{n-1} \hat{R}_Y (\theta_i) \vert 0 \rangle \tag{4}
# $$
#
# By choosing $\theta_i = 2 \mathrm{arcsin} \sqrt{c_i^\ast}$, the probability of measuring $\vert 1 \rangle$ becomes $\sin^2 (\theta_i / 2) = c_i^\ast$.
# Standard QAOA commonly uses the mixer Hamiltonian $\hat{H}_M = - \sum_i X_i$.
# Accordingly, its initial state is the ground state $\vert + \rangle$ of that mixer Hamiltonian.
# When changing the initial state to Eq. (4), we also design a mixer Hamiltonian whose ground state is the new initial state.
# For each qubit, define
#
# $$
# \hat{H}_{M, i}^{(\mathrm{ws})}
# = \left( \begin{array}{cc}
# 2 c_i^\ast -1 & - 2 \sqrt{c_i^\ast (1 - c_i^\ast)} \\
# - 2 \sqrt{c_i^\ast (1 - c_i^\ast)} & 1 - 2 c_i^\ast
# \end{array} \right)
# = -\sin \theta_i \hat{X} - \cos \theta_i \hat{Z} \tag{5}
# $$
#
# This Hamiltonian has $\hat{R}_Y (\theta_i) \vert 0 \rangle$ as its ground state with eigenvalue $-1$.
# The full mixer $\hat{H}_M^{(\mathrm{ws})} = \sum_i \hat{H}_{M, i}^{(\mathrm{ws})}$ therefore has the state in Eq. (4) as its ground state with eigenvalue $-n$.
# The standard QAOA mixer rotates around the $x$ axis of the Bloch sphere, whereas the WS-QAOA mixer rotates around $(-\sin \theta_i, 0, -\cos \theta_i)$.
# The mixer time evolution required by QAOA is $e^{-i \beta_k \hat{H}_M^{(\mathrm{ws})}}$.
# It can be implemented using only single-qubit rotations:
#
# $$
# e^{-i \beta_k \hat{H}_{M_i}^{(\mathrm{ws})}}
# = \hat{R}_Y (\theta_i) \hat{R}_Z (-2\beta_k) \hat{R}_Y (- \theta_i) \tag{6}
# $$
#
# This construction nevertheless has a limitation.
# If $c_i^\ast = 0$, then $\theta_i = 0$, so $\vert \phi_i^\ast \rangle = \vert 0 \rangle$ and the mixer becomes $\hat{H}_{M, i}^{(\mathrm{ws})} = -\hat{Z}_i$.
# If the cost Hamiltonian is also composed only of $\hat{Z}_i$ and $\hat{Z}_i \hat{Z}_j$ terms, the relation $\hat{Z} \vert 0 \rangle = \vert 0 \rangle$ means that this qubit always remains in $\vert 0 \rangle$.
# Let $\boldsymbol{d}^\ast$ denote the optimum of Eq. (1).
# If $c_i^\ast = 0$ but $d_i^\ast = 1$, the true optimum cannot be reached.
# The same issue occurs when $c_i^\ast = 1$ but $d_i^\ast = 0$.
# To address this reachability issue, [Egger et al. (2021)](https://quantum-journal.org/papers/q-2021-06-17-479/) introduced a regularization parameter $\epsilon \in [0, 0.5]$ and defined
#
# $$
# \theta_i
# = \left\{ \begin{array}{ll}
# 2 \mathrm{arcsin} (\sqrt{c_i^\ast}) & \mathrm{if} \ c_i^\ast \in [\epsilon, 1 -\epsilon] \\
# 2 \mathrm{arcsin} (\sqrt{\epsilon}) & \mathrm{if} \ c_i^\ast \leq \epsilon \\
# 2 \mathrm{arcsin} (\sqrt{1-\epsilon}) & \mathrm{if} \ c_i^\ast \geq 1 - \epsilon
# \end{array} \right. \tag{7}
# $$
#
# Values of $c_i^\ast$ within $[\epsilon, 1 - \epsilon]$ are used unchanged, while values outside that interval are clipped to $\epsilon$ or $1-\epsilon$ to avoid states arbitrarily close to $c_i^\ast = 0$ or $1$.
# Even when $c_i^\ast$ is arbitrarily close to zero, the probability of measuring $x_i = 1$ is therefore kept at or above $\epsilon$.
# When $\epsilon = 0.5$, every qubit has $\theta_i = \pi / 2$ and $\hat{H}_{M, i}^{(\mathrm{ws})} = -X_i$.
# This recovers standard QAOA.

# %% [markdown]
# ## Implementation with Qamomile
#
# We now implement the WS-QAOA method described above with Qamomile.
#
# ### Defining the Convex Quadratic Program
#
# First, define $\Sigma$ and $\boldsymbol{\mu}$ from Eq. (1).
# We choose the values so that $\Sigma$ is positive definite and symmetric.

# %%
n = 6

Sigma = np.array(
    [
        [2.00, 0.40, 0.00, 0.00, 0.00, 0.00],
        [0.40, 1.80, 0.30, 0.00, 0.00, 0.00],
        [0.00, 0.30, 1.60, 0.20, 0.00, 0.00],
        [0.00, 0.00, 0.20, 1.70, 0.35, 0.00],
        [0.00, 0.00, 0.00, 0.35, 1.90, 0.25],
        [0.00, 0.00, 0.00, 0.00, 0.25, 1.50],
    ],
    dtype=float,
)

mu = np.array([-0.976, -3.226, -1.900, -2.714, -1.638, -2.790], dtype=float)

assert Sigma.shape == (n, n)
assert mu.shape == (n,)
assert np.allclose(Sigma, Sigma.T), "Sigma must be symmetric."

eigenvalues = np.linalg.eigvalsh(Sigma)
assert eigenvalues.min() >= -1e-10, "Continuous WS-QAOA in this notebook assumes Sigma is PSD."

print("eigenvalues(Sigma) =", np.round(eigenvalues, 6))

# %% [markdown]
# ### Converting to QUBO
#
# Because binary variables satisfy $x_i^2 = x_i$, we can write
#
# $$
# \boldsymbol{x}^\top \Sigma \boldsymbol{x} + \boldsymbol{\mu}^\top \boldsymbol{x}
# = \sum_i (\Sigma_{ii} + \mu_i) x_i + 2 \sum_{i<j} \Sigma_{ij} x_i x_j
# = \sum_{i, j} Q_{ij} x_i x_j \tag{8}
# $$
#
# and represent the problem with a QUBO matrix.
# For this small number of variables, we can enumerate every state to find the exact optimum.
# We calculate it here as a reference for the later comparison.

# %%
qubo = {}

for i in range(n):
    qubo[(i, i)] = float(Sigma[i, i] + mu[i])

for i in range(n):
    for j in range(i + 1, n):
        coefficient = float(2.0 * Sigma[i, j])
        if not np.isclose(coefficient, 0.0):
            qubo[(i, j)] = coefficient

print("QUBO coefficients")
for key, value in qubo.items():
    print(f"  {key}: {value:+.6f}")

all_bits = np.array(list(itertools.product([0, 1], repeat=n)), dtype=int)
exact_energies = np.einsum("bi,ij,bj->b", all_bits, Sigma, all_bits) + all_bits @ mu

qubo_energies = np.zeros(len(all_bits), dtype=float)
for (i, j), coefficient in qubo.items():
    qubo_energies += coefficient * all_bits[:, i] * all_bits[:, j]

assert np.allclose(exact_energies, qubo_energies)

exact_energy = float(exact_energies.min())
optimal_mask = np.isclose(exact_energies, exact_energy, atol=1e-10, rtol=0.0)
optimal_bits = all_bits[optimal_mask]

print()
print("Exact binary optimum (validation only)")
print("  f(x*)               =", exact_energy)
print("  number of optima    =", len(optimal_bits))
for bits in optimal_bits:
    print("  optimal bit string  =", bits.tolist())

# %% [markdown]
# ### Solving the Continuous QP Relaxation
#
# To prepare the warm-start initial state for QAOA, solve the relaxed problem. Here we use SciPy's L-BFGS-B method with the bounds $0 \leq c_i \leq 1$.

# %%
qp_result = minimize(
    fun=lambda c: float(c @ Sigma @ c + mu @ c),
    x0=np.full(n, 0.5, dtype=float),
    jac=lambda c: 2.0 * Sigma @ c + mu,
    bounds=[(0.0, 1.0)] * n,
    method="L-BFGS-B",
    options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
)

if not qp_result.success:
    raise RuntimeError(f"QP solver failed: {qp_result.message}")

c_star = np.clip(qp_result.x, 0.0, 1.0)
qp_energy = float(c_star @ Sigma @ c_star + mu @ c_star)

print("QP status :", qp_result.message)
print("c*        :", np.round(c_star, 6))
print("f(c*)     :", qp_energy)

# %% [markdown]
# ### Introducing the Regularization Parameter and Initial Distribution
#
# Introduce $\epsilon$ and restrict the $c_i^\ast$ values obtained above to the interval $\epsilon \leq c_i^\ast \leq 1 - \epsilon$.
# Then use Eq. (7) to calculate the rotation angles for QAOA.

# %%
epsilon = 0.10

if not (0.0 < epsilon <= 0.5):
    raise ValueError("epsilon must satisfy 0 < epsilon <= 0.5")

c_tilde = np.clip(c_star, epsilon, 1.0 - epsilon)
thetas = 2.0 * np.arcsin(np.sqrt(c_tilde))
theta_binding = [float(value) for value in thetas]

uniform_probabilities = np.full(len(all_bits), 1.0 / len(all_bits), dtype=float)
warm_probabilities = np.prod(
    np.where(all_bits == 1, c_tilde[None, :], 1.0 - c_tilde[None, :]),
    axis=1,
)

assert np.isclose(uniform_probabilities.sum(), 1.0)
assert np.isclose(warm_probabilities.sum(), 1.0)

standard_initial_optimum_probability = float(uniform_probabilities[optimal_mask].sum())
warm_initial_optimum_probability = float(warm_probabilities[optimal_mask].sum())
standard_initial_mean_energy = float(uniform_probabilities @ exact_energies)
warm_initial_mean_energy = float(warm_probabilities @ exact_energies)

print("epsilon   :", epsilon)
print("c_tilde   :", np.round(c_tilde, 6))
print("theta     :", np.round(thetas, 6))

initial_summary = pd.DataFrame(
    {
        "initial mean energy": [
            standard_initial_mean_energy,
            warm_initial_mean_energy,
            warm_initial_mean_energy,
        ],
        "initial mean optimality gap": [
            standard_initial_mean_energy - exact_energy,
            warm_initial_mean_energy - exact_energy,
            warm_initial_mean_energy - exact_energy,
        ],
        "initial P(optimum)": [
            standard_initial_optimum_probability,
            warm_initial_optimum_probability,
            warm_initial_optimum_probability,
        ],
    },
    index=[
        "Standard QAOA",
        "Warm initial state + X mixer",
        "Continuous WS-QAOA",
    ],
)

display(initial_summary.round(6))

# %% [markdown]
# ### Generating Binary and Ising Models with Qamomile
#
# Pass the QUBO coefficients to Qamomile's `BinaryModel` to create a binary model.
# Then pass that model to `QAOAConverter` to generate an Ising model expressed in spin variables.

# %%
binary_model = BinaryModel.from_qubo(qubo)
converter = QAOAConverter(binary_model)
spin_model = converter.spin_model

assert binary_model.num_bits == n
assert spin_model.num_bits == n
assert np.allclose(
    [binary_model.calc_energy(bits.tolist()) for bits in all_bits],
    exact_energies,
)

print("Binary model vartype :", binary_model.vartype)
print("Number of qubits     :", spin_model.num_bits)
print("Ising linear terms   :", spin_model.linear)
print("Ising quadratic terms:", spin_model.quad)
print("Ising constant       :", spin_model.constant)


# %% [markdown]
# ### Defining Qamomile Quantum Kernels
#
# Define the quantum kernels used for QAOA. We will compare three variants:
#
# 1. Standard QAOA (uniform-superposition initialization and an X mixer)
# 2. QAOA with only the initial state changed
# 3. Continuous WS-QAOA

# %%
@qmc.qkernel
def standard_qaoa_sampling(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    n: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = superposition_vector(n)

    for layer in qmc.range(p):
        q = ising_cost(quad, linear, q, gammas[layer])
        q = x_mixer(q, -betas[layer])

    return qmc.measure(q)


@qmc.qkernel
def warm_initial_only_qaoa_sampling(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    thetas: qmc.Vector[qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    n: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], thetas[i])

    for layer in qmc.range(p):
        q = ising_cost(quad, linear, q, gammas[layer])
        q = x_mixer(q, -betas[layer])

    return qmc.measure(q)


@qmc.qkernel
def continuous_ws_qaoa_sampling(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    thetas: qmc.Vector[qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    n: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], thetas[i])

    for layer in qmc.range(p):
        q = ising_cost(quad, linear, q, gammas[layer])

        for i in qmc.range(n):
            q[i] = qmc.ry(q[i], -thetas[i])
            q[i] = qmc.rz(q[i], -2.0 * betas[layer])
            q[i] = qmc.ry(q[i], thetas[i])

    return qmc.measure(q)


# %% [markdown]
# Use `QiskitTranspiler` to verify that all three quantum circuits are equivalent when $\epsilon = 0.5 \ (\theta_i = \pi / 2)$.

# %%
transpiler = QiskitTranspiler()
run_equivalence_check = True

if run_equivalence_check:
    sanity_p = 1
    sanity_gammas = [0.37]
    sanity_betas = [-0.41]
    half_thetas = [float(np.pi / 2.0)] * n

    standard_circuit = transpiler.to_circuit(
        standard_qaoa_sampling,
        bindings={
            "p": sanity_p,
            "quad": spin_model.quad,
            "linear": spin_model.linear,
            "gammas": sanity_gammas,
            "betas": sanity_betas,
            "n": n,
        },
    )

    initial_only_circuit = transpiler.to_circuit(
        warm_initial_only_qaoa_sampling,
        bindings={
            "p": sanity_p,
            "quad": spin_model.quad,
            "linear": spin_model.linear,
            "thetas": half_thetas,
            "gammas": sanity_gammas,
            "betas": sanity_betas,
            "n": n,
        },
    )

    continuous_circuit = transpiler.to_circuit(
        continuous_ws_qaoa_sampling,
        bindings={
            "p": sanity_p,
            "quad": spin_model.quad,
            "linear": spin_model.linear,
            "thetas": half_thetas,
            "gammas": sanity_gammas,
            "betas": sanity_betas,
            "n": n,
        },
    )

    standard_state = Statevector.from_instruction(
        standard_circuit.remove_final_measurements(inplace=False)
    )
    initial_only_state = Statevector.from_instruction(
        initial_only_circuit.remove_final_measurements(inplace=False)
    )
    continuous_state = Statevector.from_instruction(
        continuous_circuit.remove_final_measurements(inplace=False)
    )

    assert standard_state.equiv(initial_only_state)
    assert standard_state.equiv(continuous_state)
    print("Sanity check passed: epsilon=0.5 makes all three circuits equivalent.")

# %% [markdown]
# Transpile the three quantum circuits for Qiskit Aer.

# %%
p = 2

standard_executable = transpiler.transpile(
    standard_qaoa_sampling,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": n,
    },
    parameters=["gammas", "betas"],
)

initial_only_executable = transpiler.transpile(
    warm_initial_only_qaoa_sampling,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "thetas": theta_binding,
        "n": n,
    },
    parameters=["gammas", "betas"],
)

continuous_executable = transpiler.transpile(
    continuous_ws_qaoa_sampling,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "thetas": theta_binding,
        "n": n,
    },
    parameters=["gammas", "betas"],
)

print(f"Transpilation finished: n={n}, p={p}")

# %% [markdown]
# Configure the depth, the initial values passed to COBYLA, and the remaining optimization settings.

# %%
METHOD_STANDARD = "Standard QAOA"
METHOD_INITIAL_ONLY = "Warm initial state + X mixer"
METHOD_CONTINUOUS = "Continuous WS-QAOA"

sample_shots = 512
maxiter = 100
optimizer_seed = 900
optimization_simulator_seed = 901

rng = np.random.default_rng(optimizer_seed)
shared_initial_params = np.concatenate(
    [
        rng.uniform(0.0, np.pi, p),
        rng.uniform(-np.pi, np.pi, p),
    ]
)

standard_executor = transpiler.executor(
    backend=AerSimulator(
        seed_simulator=optimization_simulator_seed,
        max_parallel_threads=1,
    )
)
initial_only_executor = transpiler.executor(
    backend=AerSimulator(
        seed_simulator=optimization_simulator_seed,
        max_parallel_threads=1,
    )
)
continuous_executor = transpiler.executor(
    backend=AerSimulator(
        seed_simulator=optimization_simulator_seed,
        max_parallel_threads=1,
    )
)

print("shared initial gammas:", np.round(shared_initial_params[:p], 6))
print("shared initial betas :", np.round(shared_initial_params[p:], 6))
print("sample_shots         :", sample_shots)
print("maxiter              :", maxiter)


# %% [markdown]
# ## Results
#
# Optimize the QAOA parameters independently for all three methods.
# We call `executable.sample()` and then use `converter.decode_to_binary_sampleset()` to convert the samples into a form that is easier to analyze.
# The mean energy of the decoded samples is passed to the COBYLA classical optimizer.

# %%
def sampled_mean_energy(params, executable, executor, history):
    gammas = [float(value) for value in params[:p]]
    betas = [float(value) for value in params[p:]]

    result = executable.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    ).result()

    decoded = converter.decode_to_binary_sampleset(result)
    mean_energy = decoded.energy_mean()
    history.append(mean_energy)
    return mean_energy


cost_histories = {
    METHOD_STANDARD: [],
    METHOD_INITIAL_ONLY: [],
    METHOD_CONTINUOUS: [],
}

optimization_specs = [
    (METHOD_STANDARD, standard_executable, standard_executor),
    (METHOD_INITIAL_ONLY, initial_only_executable, initial_only_executor),
    (METHOD_CONTINUOUS, continuous_executable, continuous_executor),
]

optimization_results = {}

for method_name, executable, executor in optimization_specs:
    history = cost_histories[method_name]

    result = minimize(
        lambda params, exe=executable, exr=executor, hist=history: sampled_mean_energy(
            params,
            exe,
            exr,
            hist,
        ),
        shared_initial_params.copy(),
        method="COBYLA",
        options={"maxiter": maxiter},
    )

    optimization_results[method_name] = result

    print()
    print(method_name)
    print("  message :", result.message)
    print("  nfev    :", result.nfev)
    print("  estimate:", result.fun)
    print("  gammas  :", np.round(result.x[:p], 6))
    print("  betas   :", np.round(result.x[p:], 6))

# %% [markdown]
# The results show the ordering standard QAOA > QAOA with only the initial state changed > continuous WS-QAOA, with continuous WS-QAOA reaching the lowest energy.
# Next, plot the energy throughout the optimization.

# %%
plt.figure(figsize=(9, 4))
for method_name, history in cost_histories.items():
    plt.plot(np.arange(1, len(history) + 1), history, label=method_name)
plt.axhline(exact_energy, linestyle="--", label="exact binary optimum")
plt.xlabel("objective evaluation")
plt.ylabel("sampled mean energy")
plt.title("Raw optimization histories")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
for method_name, history in cost_histories.items():
    best_so_far = np.minimum.accumulate(np.asarray(history, dtype=float))
    plt.plot(np.arange(1, len(history) + 1), best_so_far, label=method_name)
plt.axhline(exact_energy, linestyle="--", label="exact binary optimum")
plt.xlabel("objective evaluation")
plt.ylabel("best sampled mean energy so far")
plt.title("Best-so-far optimization histories")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# This page showed how to implement the warm-start QAOA proposed by [Egger et al. (2021)](https://quantum-journal.org/papers/q-2021-06-17-479/) with Qamomile.
# The key takeaways are:
#
# * Warm starting successfully found a lower-energy solution in this example.
# * When changing the initial state, the mixer should also be changed accordingly.
# * Qamomile makes it straightforward to configure the QAOA X mixer and the initialization angles used for warm starting.

# %%
