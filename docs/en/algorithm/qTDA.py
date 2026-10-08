# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: qamomile (3.11.16)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, finance, variational]
# ---
#
# # Depth-Efficient Quantum Topological Data Analysis
#
# Financial market dynamics are nonlinear, and classical statistical methods cannot fully capture their complexity.
# Topological data analysis (TDA) was introduced as an efficient way to extract topological features from point-cloud data.
# However, computing its central invariants, the Betti numbers, becomes increasingly expensive and is therefore unsuitable for large-scale calculations.
# Quantum TDA (qTDA) was developed to address this computational bottleneck, but it also requires many qubits and deep circuits, making it impractical.
# This article explains the depth-efficient qTDA proposed by [Mazumder & Mazumder (2026)](https://arxiv.org/abs/2607.09906) and demonstrates an implementation in Qamomile.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]" scipy

# %%
from itertools import combinations, product
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer, rz_layer
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# ### Problem: Topological Data Analysis and Financial Engineering
#
# Forecasting financial market trends is becoming increasingly important.
# Although many statistical methods have been developed for this purpose, they have not adequately accounted for major financial crises.
# The 2008 global financial crisis is thought to have arisen from correlations and structural instabilities that volatility metrics such as variance and value at risk (VaR) failed to capture.
# [Gidea & Katz (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0378437117309202?via%3Dihub) showed that topological features based on point-cloud persistence—the lifetime or persistence of holes formed by the point cloud—began a strong upward trend 250 days before the collapse of Lehman Brothers.
# This suggests that topological indicators may provide early warning of financial crises that conventional methods fail to capture.
# TDA characterizes the $k$-dimensional holes in a simplicial complex built from the data points.
# For example, $\beta_0$, $\beta_1$, and $\beta_2$ correspond to the numbers of connected components, loops, and voids, respectively.
# In a $k$-dimensional setting, these invariants extend through $\beta_k$ and are called Betti numbers.
# Computing them is considered to have a cost of $\mathcal{O} (n_k^3)$, which limits classical methods ($n_k$ is the total number of $k$-simplices formed by the data points).
# Quantum TDA (qTDA) algorithms have been proposed to address this limitation.
# A representative approach uses quantum phase estimation (QPE) to estimate the eigenvalue spectrum of the combinatorial (Hodge) Laplacian $\Delta_k$ and reconstruct the Betti numbers from it.
# However, QPE requires many qubits and deep circuits, making it difficult to implement on NISQ or early fault-tolerant quantum computers.
#
# ### Prior Work
#
# [Lloyd et al. (2016)](https://www.nature.com/articles/ncomms10138) proposed estimating Betti numbers with QPE as described above.
# Their method applies QPE to the unitary $U = e^{i\Delta_k}$ to estimate the number of zero eigenvalues, which is the Betti number (we refer to this method as the LGZ algorithm below).
# However, because QPE requires many ancilla qubits, this approach cannot be run on current quantum computers.
# [Schmidhuber & Lloyd (2023)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.040349) investigated the computational complexity of TDA in detail.
# Since its introduction, the LGZ algorithm had been considered to provide an exponential quantum advantage over classical methods through QPE.
# Their analysis showed, however, that exact Betti-number computation is #P-hard and that even approximate computation is NP-hard.
# This result means that, in the worst case, a quantum computer can provide only a polynomial advantage for Betti-number computation.
# In fact, by analyzing the LGZ algorithm as a special case, the authors showed that it achieves only a quadratic speedup.
# Consequently, [Mazumder & Mazumder (2026)](https://arxiv.org/abs/2607.09906) explored a new approach based on PCE rather than QPE.
#
# ### Pauli Correlation Encoding (PCE)
#
# PCE is a qubit-efficient encoding method proposed by [Sciorilli et al. (2025)](https://www.nature.com/articles/s41467-024-55346-z).
# Consider $m$ binary variables $x_1, x_2, \dots, x_m \in \{-1, 1\}$ and suppose that we want to minimize an objective function $f(\boldsymbol{x})$ defined in terms of these variables.
# Variational quantum algorithms such as QAOA use one qubit to represent each variable, so representing large problems requires many qubits.
# PCE instead represents the problem efficiently with fewer qubits.
# For an $n$-qubit system, an operator in which Pauli operators $X$, $Y$, or $Z$ act nontrivially on exactly $k$ qubits is called a $k$-body Pauli correlator.
# For example, when $n=3$ and $k=2$, possible correlators include
#
# $$
# \Pi_1^{(k)}
# = Z_1 \otimes Z_2 \otimes I_3, \quad \Pi_2^{(k)}
# = X_1 \otimes I_2 \otimes Y_3 \dots \tag{1}
# $$
#
# Consider a set of $m$ Pauli strings $\Pi^{(k)} = \{\Pi_1^{(k)}, \Pi_2^{(k)}, \dots, \Pi_m^{(k)}\}$, and compute the expectation value of each $\Pi_i^{(k)}$ with respect to an $n$-qubit parameterized quantum state $\vert \Psi (\boldsymbol{\theta}) \rangle$:
#
# $$
# c_i
# = \langle \Psi (\boldsymbol{\theta}) \vert \Pi_i^{(k)} \vert \Psi (\boldsymbol{\theta}) \rangle. \tag{2}
# $$
#
# Pauli correlation encoding represents the $i$-th binary variable $x_i$ by the sign of $c_i$, namely $\mathrm{sgn} (c_i)$.
# In the preceding $k=2$ example, the number of ways to apply one of $X$, $Y$, or $Z$ to exactly two of the $n$ qubits is ${}_n C_2 \times 3^2 = \frac{3^2}{2} n (n-1)$.
# Thus, PCE can represent $m = \mathcal{O} (n^2)$ binary variables.
# Similarly, for $k=3$, the count is ${}_n C_3 \times 3^3 = \frac{3^2}{2} n (n-1) (n-2)$, allowing PCE to represent $m = \mathcal{O} (n^3)$ binary variables.
# The continuous relaxation $c_i$ of a binary variable $x_i$ can also be viewed as increasing the expressive power of the representation.
# Under the assumptions of their analysis, [Sciorilli et al. (2025)](https://www.nature.com/articles/s41467-024-55346-z) also showed a superpolynomial suppression of barren plateaus, which frequently hinder variational quantum algorithms.
#
# :::{note}
# The qTDA method discussed below falls outside the scope of that theoretical guarantee.
# :::
#
# ## Algorithm
#
# ### Betti Numbers and PCE
#
# Let the boundary operator that maps a $k$-chain to a $(k-1)$-chain be denoted by $\partial_k: C_k \rightarrow C_{k-1}$.
# The $k$-th combinatorial (Hodge) Laplacian is then defined as
#
# $$
# \Delta_k
# = \partial_{k+1} \partial_{k+1}^\top + \partial_k^\top \partial_k
# \in \mathbb{R}^{n_k \times n_k}. \tag{3}
# $$
#
# Here, $n_k = \vert \mathcal{C}_k \vert$ is the number of $k$-simplices.
# The Hodge decomposition theorem gives the Betti number $\beta_k$ as
#
# $$
# \beta_k
# = \mathrm{dim} \ \mathrm{ker} (\Delta_k). \tag{4}
# $$
#
# Here, $\mathrm{dim}$ denotes dimension and $\mathrm{ker}$ denotes the kernel.
# Equation (4) states that the Betti number equals the dimension of the space spanned by vectors $\boldsymbol{v}$ satisfying $\Delta_k \boldsymbol{v} = \mathbf{0}$.
# Equivalently, the Betti number is the number of zero eigenvalues of $\Delta_k$.
# From Equation (3), $\Delta_k$ is an $n_k \times n_k$ real symmetric matrix.
# We represent each component $v_i$ of a null-space basis vector $\boldsymbol{v} \in \mathbb{R}^{n_k}$ by a Pauli expectation value on $n$ qubits.
# Specifically, using Equation (2), let $v_i = c_i (\theta) = \langle \Psi (\boldsymbol{\theta}) \vert \Pi_i^{(k)} \vert \Psi (\boldsymbol{\theta}) \rangle$.
# In the earlier description of PCE, $c_i$ was ultimately rounded to the binary variable $\mathrm{sgn} (c_i)$.
# Because the components of a null-space vector are continuous, however, we use $c_i (\theta)$ directly as an eigenvector component rather than retaining only its sign.
#
# ### Rayleigh-Quotient Minimization and Variational Deflation
#
# The preceding discussion reduces the problem of finding a Betti number to counting the zero eigenvalues of $\Delta_k$.
# We therefore use the following Rayleigh quotient as the loss function minimized by the variational algorithm:
#
# $$
# \mathcal{L} (\boldsymbol{\theta})
# = \frac{\boldsymbol{c}(\boldsymbol{\theta})^\top \Delta_k \boldsymbol{c} (\boldsymbol{\theta})}{\boldsymbol{c} (\boldsymbol{\theta})^\top \boldsymbol{c} (\boldsymbol{\theta})}. \tag{5}
# $$
#
# Here, $\boldsymbol{c} (\boldsymbol{\theta}) = (c_1(\boldsymbol{\theta}), c_2(\boldsymbol{\theta}), \dots, c_{n_k}(\boldsymbol{\theta}))^\top$.
# In general, the minimum Rayleigh quotient over all nonzero vectors equals the smallest eigenvalue of $\Delta_k$.
# In this method, however, $\boldsymbol{c}(\boldsymbol{\theta})$ is restricted to the range reachable by the ansatz. Its minimum is therefore at least $\lambda_{\min}(\Delta_k)$ and equals it only if the corresponding eigenvector is reachable.
# Thus, if minimizing Equation (5) yields parameters $\boldsymbol{\theta}$ for which $\mathcal{L} = 0$, we have found a null-space vector.
# Computing the Betti number requires finding multiple zero eigenvalues.
# Suppose that the $j$-th null-space vector $\boldsymbol{c}^{(j)}$ has already been found. When searching for the next vector $\boldsymbol{c}^{(j+1)}$, we use the following loss function:
#
# $$
# \mathcal{L}_{j+1} (\boldsymbol{\theta})
# = \mathcal{L} (\boldsymbol{\theta}) + \mu \sum_{\ell=1}^j \left\vert \frac{\boldsymbol{c} (\boldsymbol{\theta})^\top \boldsymbol{c}^{(\ell)}}{\| \boldsymbol{c} (\boldsymbol{\theta})\| \| \boldsymbol{c}^{(\ell)} \|} \right\vert^2. \tag{6}
# $$
#
# Here, $\mu$ is the weight of the penalty term.
# The second term is zero for vectors orthogonal to every previously found null-space vector $\boldsymbol{c}^{(j)}$.
# It therefore encourages convergence to a different null-space vector.
# This approach is inspired by variational deflation ([Higgott et al. (2019)](https://quantum-journal.org/papers/q-2019-07-01-156/)), a method for computing molecular excited states with a quantum computer.

# %% [markdown]
# ## Implementation with Qamomile
#
# We now implement PCE-VQE with Qamomile and apply it to topological data analysis.
#
# ### Classical Computation of the Combinatorial Laplacian and Its Eigenvalues
#
# First, define a function that takes lists of edges and triangles in a complex and computes the combinatorial Laplacian in Equation (3).
# The boundary matrix $\partial_1$ represents the start vertex of each edge by -1 and its end vertex by +1, while $\partial_2$ represents each triangle by its three oriented edges.
# For a complex without triangles, $\partial_2$ is empty.

# %%
def build_laplacian(n_vertices, edges, triangles):
    """Construct the combinatorial Laplacian Δ₁ from edges and triangles."""

    if n_vertices < 0:
        raise ValueError("n_vertices must be non-negative")

    # --- Canonicalize edges ---
    canonical_edges = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must contain exactly 2 vertices")

        v0, v1 = edge

        if not (0 <= v0 < n_vertices and 0 <= v1 < n_vertices):
            raise ValueError(f"edge {edge} contains an invalid vertex")
        if v0 == v1:
            raise ValueError(f"self-edge {edge} is not allowed")

        canonical_edges.append(tuple(sorted((v0, v1))))

    if len(set(canonical_edges)) != len(canonical_edges):
        raise ValueError("duplicate edges are not allowed")

    # --- Canonicalize triangles ---
    canonical_triangles = []
    for triangle in triangles:
        if len(triangle) != 3:
            raise ValueError("each triangle must contain exactly 3 vertices")

        if any(v < 0 or v >= n_vertices for v in triangle):
            raise ValueError(f"triangle {triangle} contains an invalid vertex")
        if len(set(triangle)) != 3:
            raise ValueError(f"degenerate triangle {triangle} is not allowed")

        canonical_triangles.append(tuple(sorted(triangle)))

    if len(set(canonical_triangles)) != len(canonical_triangles):
        raise ValueError("duplicate triangles are not allowed")

    # --- Check that every triangle face exists in edges ---
    edge_idx = {e: k for k, e in enumerate(canonical_edges)}

    for v0, v1, v2 in canonical_triangles:
        for edge in [(v0, v1), (v0, v2), (v1, v2)]:
            if edge not in edge_idx:
                raise ValueError(
                    f"triangle {(v0, v1, v2)} requires missing edge {edge}"
                )

    # --- Boundary matrix B1 ---
    n_e = len(canonical_edges)
    B1 = np.zeros((n_vertices, n_e))

    for k, (v0, v1) in enumerate(canonical_edges):
        B1[v0, k] = -1.0
        B1[v1, k] = +1.0

    # --- Boundary matrix B2 ---
    B2 = np.zeros((n_e, len(canonical_triangles)))

    for t, (v0, v1, v2) in enumerate(canonical_triangles):
        B2[edge_idx[(v1, v2)], t] = +1.0
        B2[edge_idx[(v0, v2)], t] = -1.0
        B2[edge_idx[(v0, v1)], t] = +1.0

    # Verify ∂₁∂₂ = 0
    if not np.allclose(B1 @ B2, 0.0, atol=1e-12):
        raise ValueError("invalid simplicial complex: B1 @ B2 != 0")

    return B2 @ B2.T + B1.T @ B1


# %% [markdown]
# We also define a function that solves the combinatorial Laplacian computed above classically.
# It eigendecomposes the combinatorial Laplacian $\Delta_1$ and obtains the Betti number $\beta_1$ together with a basis for its null space.
# These results provide an initial value for warm-starting the Qamomile computation and a reference for comparing its output later.

# %%
def classical_betti1(L1, tol=1e-9):
    """Return classical β₁ = dim ker Δ₁ and a null-space basis."""
    ev, evec = np.linalg.eigh(L1)
    mask = np.abs(ev) < tol
    return int(mask.sum()), evec[:, mask]


# %% [markdown]
# ### PCE Setup
#
# Define a function that creates the Pauli strings used by Pauli correlation encoding to encode each edge of $\Delta_1$.

# %%
def build_correlator_observables(n_needed, kappa):
    """Return n_needed k-body Pauli correlators as Qamomile Hamiltonians."""

    # Validate inputs
    if n_needed < 0:
        raise ValueError("n_needed must be non-negative")
    if kappa < 1:
        raise ValueError("kappa must be at least 1")

    # Find the smallest qubit count that can represent n_needed correlators
    n_q = kappa
    while comb(n_q, kappa) * (3**kappa) < n_needed:
        n_q += 1

    # Return an empty list when no correlators are needed
    if n_needed == 0:
        return n_q, []

    pauli_fn = {
        "X": qm_o.X,
        "Y": qm_o.Y,
        "Z": qm_o.Z,
    }

    observables = []

    for pos in combinations(range(n_q), kappa):
        for ch in product("XYZ", repeat=kappa):

            # Fix the observable register width at n_q qubits
            H = qm_o.Hamiltonian.identity(num_qubits=n_q)

            for p, c in zip(pos, ch):
                H *= pauli_fn[c](p)

            observables.append(H)

            if len(observables) == n_needed:
                return n_q, observables

    return n_q, observables


# %% [markdown]
# ### Defining a Hardware-Efficient Ansatz for VQE
#
# Define the hardware-efficient ansatz (HEA) used as the PCE-VQE quantum circuit as a Qamomile qkernel.
# After placing every qubit in superposition with a Hadamard gate, the circuit repeats RY and RZ rotation layers and an entangling layer `depth` times.
# Finally, `qmc.expval(q, P)` extracts the expectation value.

# %%
@qmc.qkernel
def pce_ansatz(
    n: qmc.UInt,
    depth: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    P: qmc.Observable,
) -> qmc.Float:
    """Return the expectation value of observable P from the HEA ansatz."""
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for d in qmc.range(depth):
        offset = d * 2 * n
        q = ry_layer(q, thetas, offset)
        q = rz_layer(q, thetas, offset + n)
        q = cx_entangling_layer(q)
    return qmc.expval(q, P)

transpiler = QiskitTranspiler()
executor = transpiler.executor()
print("PCE ansatz defined")


# %% [markdown]
# Finally, combine the functions defined so far into a function that returns the correlator vector $\boldsymbol{c}(\boldsymbol{\theta})$ for a given $\theta$.

# %%
def make_correlator_evaluator(n_needed, kappa, depth):
    """Cache a transpiled program for each correlator and return theta -> c(theta)."""
    n_q, observables = build_correlator_observables(n_needed, kappa)
    executables = [
        transpiler.transpile(
            pce_ansatz,
            bindings={"n": n_q, "depth": depth, "P": P_i},
            parameters=["thetas"],
        )
        for P_i in observables
    ]
    num_thetas = 2 * n_q * depth

    def correlators(theta):
        thetas = list(np.asarray(theta, dtype=float))
        return np.array([
            exe.run(executor, bindings={"thetas": thetas}).result()
            for exe in executables
        ])
    return correlators, num_thetas, n_q


# %% [markdown]
# ### Case 1: Equally Spaced Points on a Circle
#
# Now that the required functions are ready, construct the point cloud to analyze.
# For the first case, place `n_points` equally spaced points on the unit circle.

# %%
n_points = 8
theta_pts = np.linspace(0, 2*np.pi, n_points, endpoint=False)
points = np.c_[np.cos(theta_pts), np.sin(theta_pts)]
print("Point-cloud shape:", points.shape)
points

# %% [markdown]
# Next, construct a Vietoris-Rips complex from this point cloud.
# Use `epsilon` as the distance threshold and connect each pair of points whose distance is at most this value with an edge.
# Fill every set of three points for which all three edges are present as a triangle.
# With `n_points = 8`, this produces eight edges between neighboring points but no triangles, so the loop-shaped hole remains unfilled.
# The Betti number is therefore $\beta_1 = 1$.

# %%
epsilon = 0.85
D = squareform(pdist(points))
edges = [(i, j) for i, j in combinations(range(n_points), 2) if D[i, j] <= epsilon]
edge_set = set(edges)
triangles = [(i, j, k) for i, j, k in combinations(range(n_points), 3)
             if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set]
print(f"ε={epsilon}, edges={len(edges)}, triangles={len(triangles)}")
print("Edges:", edges)

# %% [markdown]
# Compute the combinatorial Laplacian $\Delta_1$ from the point cloud, followed by its eigenvalues and the Betti number $\beta_1$.
# At the same time, compute a null-space vector.

# %%
L1 = build_laplacian(n_points, edges, triangles)
beta1_classical, null_basis = classical_betti1(L1)
print("Δ₁ shape:", L1.shape)
print("Eigenvalues of Δ₁:", np.linalg.eigvalsh(L1))
print("Classical β₁ =", beta1_classical)
print("Null-space basis (warm-start target):")
print(null_basis)

# %% [markdown]
# With the classical calculation complete, run the computation implemented in Qamomile.
# Setting `kappa = 2` selects two-body correlators, whose Pauli strings contain two non-identity operators.
# As a basic check, evaluate the correlator vector once using a random $\boldsymbol{\theta}$.
# Verify that the resulting values of $\boldsymbol{c} (\boldsymbol{\theta})$ lie in the interval [-1, 1] before proceeding to optimization.

# %%
rng = np.random.default_rng(42)

kappa = 2
n_layers = 3
n_edges = L1.shape[0]

correlators, num_thetas, n_qubits = make_correlator_evaluator(
    n_edges, kappa, n_layers)
print(f"Encode nₖ={n_edges} with κ={kappa} → n_qubits={n_qubits}")
print(f"Number of variational parameters: {num_thetas}")

# Basic check
c_test = correlators(rng.uniform(0, 2*np.pi, num_thetas))
print(f"c(θ) shape: {c_test.shape}, range: {c_test.min():.4f} to {c_test.max():.4f}")

# %% [markdown]
# Next, calculate an initial value for the warm start.
# Pretrain the quantum-circuit parameters $\boldsymbol{\theta}$ so that the resulting vector is aligned with the null-space vector obtained by the classical calculation.

# %%
target_v = null_basis[:, 0]
target_v = target_v / np.linalg.norm(target_v)

best_theta, best_fit = None, np.inf
for _ in range(5):
    def fit_loss(theta):
        c = correlators(theta)
        nc = np.linalg.norm(c)
        return 2.0 if nc < 1e-9 else 1.0 - abs((c/nc) @ target_v)
    r = minimize(fit_loss, rng.uniform(0, 2*np.pi, num_thetas),
                 method="COBYLA", options={"maxiter": 300, "rhobeg": 0.3})
    if r.fun < best_fit:
        best_theta, best_fit = r.x, r.fun

print("Warm-start fitting loss (0 means exact agreement):", round(best_fit, 4))


# %% [markdown]
# Define the Rayleigh quotient in Equation (5), then minimize it from the warm-start $\theta$ with a gradient-free optimizer.
# Compare the optimized correlator vector and the Betti number $\beta_1$ obtained with PCE-VQE against their classical counterparts.

# %%
def rayleigh(theta):
    c = correlators(theta)
    denom = c @ c
    return np.inf if denom < 1e-12 else (c @ L1 @ c) / denom

res = minimize(rayleigh, best_theta, method="COBYLA",
               options={"maxiter": 500, "rhobeg": 0.2})

c_star = correlators(res.x)
R_star = rayleigh(res.x)
print("Final Rayleigh quotient R =", R_star)
print("Candidate null-space vector c(θ*) (normalized) =")
print(c_star / np.linalg.norm(c_star))
print("Classical null-space vector (comparison) =")
print(target_v)
delta = 0.5
beta1_pce = 1 if R_star < delta else 0
print(f"R={R_star:.2e} vs δ={delta}")
print("PCE-VQE β₁ =", beta1_pce, " / classical β₁ =", beta1_classical)

# %% [markdown]
# In this run, both PCE-VQE and the classical calculation recover the expected Betti number, $\beta_1 = 1$.

# %% [markdown]
# ### Case 2: Six Toy Models
#
# We now prepare the six toy models studied by [Mazumder & Mazumder (2026)](https://arxiv.org/abs/2607.09906):
#
# * Path graph: a graph arranged in a straight line, such as 0--1--2
# * Hollow triangle: a graph whose three vertices (0, 1, 2) form a triangle with an unfilled interior
# * Filled triangle: the same triangle as the hollow triangle, but with its interior filled
# * 2 hollow triangles: two triangles, (0, 1, 2) and (3, 4, 5), neither of which has a filled interior
# * Square (4-cycle): a square formed by four vertices with an unfilled interior
# * Figure-eight: two triangles, (0, 1, 2) and (0, 3, 4), that share vertex 0
#
# Compute the Betti number $\beta_1$ classically for each model.

# %%
TOY_COMPLEXES = {
    "Path graph":         (3, [(0,1), (1,2)], []),
    "Hollow triangle":    (3, [(0,1), (0,2), (1,2)], []),
    "Filled triangle":    (3, [(0,1), (0,2), (1,2)], [(0,1,2)]),
    "2 hollow triangles": (6, [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)], []),
    "Square (4-cycle)":   (4, [(0,1),(1,2),(2,3),(0,3)], []),
    "Figure-eight":       (5, [(0,1),(0,2),(1,2),(0,3),(0,4),(3,4)], []),
}

for name, (nv, eg, tr) in TOY_COMPLEXES.items():
    Lt = build_laplacian(nv, eg, tr)
    b, _ = classical_betti1(Lt)
    print(f"{name:<22} edges={len(eg):>2} triangles={len(tr):>2} β₁={b}")


# %% [markdown]
# ### Variational Deflation
#
# Computing a Betti number $\beta_1 \geq 2$ requires finding multiple null-space vectors.
# Define the loss function in Equation (6) and use variational deflation to minimize it variationally.

# %%
def deflated_objective(c, laplacian, found, lam):
    """Compute the normalized Rayleigh quotient plus the deflation penalty."""
    denom = c @ c
    if denom < 1e-12:
        return np.inf

    # Normalize the candidate vector to make the objective scale-invariant
    c_unit = c / np.sqrt(denom)

    rayleigh_value = c_unit @ laplacian @ c_unit
    penalty = lam * sum((c_unit @ v) ** 2 for v in found)

    return rayleigh_value + penalty


def run_deflation_history(L1_, correlators, num_thetas,
                          delta=0.01, lam=10.0, maxiter=200,
                          n_restarts=3, seed=1, max_rounds=4):
    """Estimate β₁ with Qamomile correlators and return each round's loss history.

    Figure 3 uses random initialization without a warm start.
    """
    rg = np.random.default_rng(seed)
    found, histories = [], []

    for _ in range(max_rounds):
        best_hist, best_val, best_c = None, np.inf, None

        for _ in range(n_restarts):
            hist = []

            def loss(theta):
                c = correlators(theta)  # Evaluated by Qamomile
                val = deflated_objective(c, L1_, found, lam)
                hist.append(val)        # Record the loss history
                return val

            res = minimize(
                loss,
                rg.uniform(0, 2 * np.pi, num_thetas),
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.5},
            )

            c = correlators(res.x)
            final = deflated_objective(c, L1_, found, lam)

            if final < best_val:
                best_val, best_hist, best_c = final, hist, c

        histories.append(best_hist)

        if best_val < delta:
            found.append(best_c / np.linalg.norm(best_c))
        else:
            break

    return len(found), histories


# %% [markdown]
# Run PCE-VQE on the six toy models.

# %%
KAPPA = 2
DEPTH = 3
DELTA = 0.01
results = {}

for name, (nv, eg, tr) in TOY_COMPLEXES.items():
    Lt = build_laplacian(nv, eg, tr)
    n_e = Lt.shape[0]
    b_true, _ = classical_betti1(Lt)

    # Rebuild the evaluator because each complex has a different edge count
    corr, num_thetas_t, n_q = make_correlator_evaluator(n_e, KAPPA, DEPTH)

    b_est, hists = run_deflation_history(
        Lt, corr, num_thetas_t,
        delta=DELTA, maxiter=200, n_restarts=3, seed=1)

    results[name] = (b_true, b_est, hists)
    print(f"  {name:<22} edges={n_e} qubits={n_q} θ={num_thetas_t} "
          f"β₁_true={b_true} β₁_est={b_est} rounds={len(hists)} "
          f"{'OK' if b_true==b_est else 'FAIL'}")

n_ok = sum(1 for (bt, be, _) in results.values() if bt == be)
print(f"\nCorrect: {n_ok}/{len(results)}")

# %% [markdown]
# In this run, the expected Betti number $\beta_1$ is recovered for every model.
# Finally, visualize the search process for each model.

# %% Visualization in the style of Figure 3
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for ax, (name, (b_true, b_est, hists)) in zip(axes, results.items()):
    for r, h in enumerate(hists):
        if len(h) == 0:
            continue
        ax.semilogy(np.maximum(h, 1e-12), color=colors[r % len(colors)],
                    lw=1.2, label=f"Deflation {r}")
    ax.axhline(DELTA, color="red", ls="--", lw=1.0)
    mark = "✓" if b_true == b_est else "✗"
    ax.set_title(f"{name}\n" r"$\beta_1$=" f"{b_true} (est={b_est}) {mark}",
                 fontsize=10)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, ls=":")
    ax.tick_params(labelsize=8)

fig.suptitle(f"Qamomile PCE-VQE Convergence on Toy Laplacians "
             f"({n_ok}/{len(results)} correct)", fontsize=13, y=0.98)
fig.tight_layout()
plt.show()

# %% [markdown]
# Each panel shows the search for null-space vectors in one toy model.
# The vertical axis shows the value of the loss function, and the horizontal axis shows the iteration count.
# `Deflation` in the figure denotes the deflation round.
# Thus, `Deflation 0` is the round that searches for the first null-space vector, while `Deflation 1` searches for the next one.
# The red dashed line in every panel marks the threshold. A loss below this threshold is interpreted as finding a null-space vector.
# Consider the `2 hollow triangles` example in the lower-left panel.
# `Deflation 0` and `Deflation 1` fall below the threshold and successfully find null-space vectors.
# `Deflation 2`, however, terminates without falling below the threshold.
# Because two null-space vectors were found, the Betti number in this case is estimated as $\beta_1 = 2$.

# %% [markdown]
# ## Summary
#
# This article demonstrated how to implement the qTDA algorithm proposed by [Mazumder & Mazumder (2026)](https://arxiv.org/abs/2607.09906) with Qamomile.
# The main points are summarized below:
#
# * Applying PCE to Betti-number estimation makes it possible to use fewer qubits. Unlike standard PCE, this method uses continuous values directly as eigenvector components rather than rounding them to $\pm 1$.
# * Qamomile makes it straightforward to run VQE.
# * In this run for Case 1, the warm-start calculation recovered the expected Betti number of the circular point cloud.
# * In this run for Case 2, variational deflation recovered the expected number of holes in all six toy complexes with $\beta_1 = 0, 1, 2$.

# %%
