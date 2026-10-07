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
# # Large-Scale Portfolio Optimization with Pauli Correlation Encoding
#
# Portfolio optimization is central to financial decision-making.
# Classical computational methods have traditionally been used to balance risk
# and return. More recently, quantum computing has been proposed as a promising
# alternative. However, the limited qubit counts and noise of current quantum
# hardware restrict the size of problems that can be implemented.
# This tutorial introduces the study by
# [Soloviev & Krompiec (2025)](https://arxiv.org/abs/2511.21305), which applies
# variational quantum algorithms to portfolio optimization problems, and shows
# how to implement the approach with Qamomile.

# %%
# Install the latest Qamomile through pip!
# # # !pip install "qamomile[qiskit,visualization]" kagglehub networkx pandas


# %%
import glob
import os
from collections import deque

import kagglehub
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer, rz_layer
from qamomile.optimization.binary_model import BinaryModel, BinarySampleSet
from qamomile.optimization.pce import PCEConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# ### Portfolio Optimization and Quantum Computing
#
# Portfolio optimization aims to maximize expected return while minimizing risk.
# As practical constraints and the size of the asset universe grow, the problem
# can become difficult to solve classically. Quantum computing has therefore
# attracted attention as an approach to combinatorial optimization.
# Among quantum circuit methods, the Quantum Approximate Optimization Algorithm
# (QAOA) and the Variational Quantum Eigensolver (VQE) have been widely studied.
# However, their standard formulations require a number of qubits that grows
# linearly with the number of assets $m$.
# Circuit cutting, which splits a large quantum circuit into smaller circuits,
# has also been proposed, but the portfolio studies discussed here still handle
# fewer than 100 assets. Outside circuit-based methods, quantum annealers have
# been used for larger combinatorial optimization problems involving hundreds
# or thousands of assets. These approaches have not demonstrated quantum
# advantage.
#
# ### Previous Work
#
# As described above, the number of variables that QAOA and VQE can handle is a
# bottleneck. [Soloviev et al. (2025)](https://arxiv.org/abs/2506.08947) studied
# portfolio optimization with circuit cutting. Their approach partitions a
# market graph to construct a portfolio while reducing the required qubit count.
# However, post-processing the results of the cut circuits increases the runtime;
# the study reported calculations using up to 71 qubits.
# [Stopfer & Wagner (2025)](https://arxiv.org/abs/2509.17876) conducted benchmarks
# specifically for quantum portfolio optimization. In those benchmarks, mixed
# integer programming and problem-specific heuristics consistently outperformed
# QAOA and quantum annealing.
# To address these limitations,
# [Soloviev & Krompiec (2025)](https://arxiv.org/abs/2511.21305) investigated a
# variational optimization approach using Pauli Correlation Encoding (PCE),
# proposed by [Sciorilli et al. (2025)](https://www.nature.com/articles/s41467-024-55346-z).
#
# ### Formulating Portfolio Optimization
#
# Portfolio optimization was formulated by
# [Markowitz (1952)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1952.tb01525.x).
# The expected portfolio return is expressed as a weighted sum of the expected
# returns of its assets:
#
# $$
# \mu_p
# = \sum_{i=1}^N w_i \mu_i \tag{1}
# $$
#
# Here, $w_i$ and $\mu_i$ are the weight and expected return of asset $i$,
# respectively. Maximizing this expected return forms the objective of portfolio
# optimization. Risk is represented by the portfolio variance:
#
# $$
# \sigma_p^2
# = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \mathrm{Cov} (r_i, r_j) \tag{2}
# $$
#
# Here, $\mathrm{Cov} (r_i, r_j)$ is the covariance between the returns $r_i$ and
# $r_j$ of assets $i$ and $j$. Combining assets with low correlation or negative
# covariance can offset return fluctuations and reduce the total portfolio
# variance.
# In addition to this mean-variance framework, portfolio performance can be
# evaluated using the Sharpe ratio ([Sharpe (1966)](https://www.jstor.org/stable/2351741)).
# This measures risk-adjusted return by comparing the return above the risk-free
# rate with the standard deviation of portfolio returns:
#
# $$
# \mathrm{Sharpe \ Ratio}
# = \frac{\mu_p - r_f}{\sigma_p} \tag{3}
# $$
#
# Here, $r_f$ is the risk-free rate. A larger ratio indicates a greater excess
# return per unit of risk. A smaller ratio indicates less excess return for the
# risk taken.
#
# ### Pauli Correlation Encoding (PCE)
#
# PCE is a qubit-efficient encoding proposed by
# [Sciorilli et al. (2025)](https://www.nature.com/articles/s41467-024-55346-z).
# Suppose we have $m$ binary variables $x_1, x_2, \dots, x_m \in \{-1, 1\}$ and
# want to minimize an objective function $f(\boldsymbol{x})$.
# Standard formulations of variational quantum algorithms such as QAOA use one
# qubit per variable, so large problems require many qubits. PCE encodes the
# variables using fewer qubits.
# On an $n$-qubit system, a Pauli string with non-identity operators $X$, $Y$, or
# $Z$ acting on exactly $k$ qubits is called a $k$-body Pauli correlator.
# For example, when $n=3$ and $k=2$, possible operators include
#
# $$
# \Pi_1^{(2)}
# = Z_1 \otimes Z_2 \otimes I_3, \quad \Pi_2^{(2)}
# = X_1 \otimes I_2 \otimes Y_3 \dots \tag{4}
# $$
#
# Consider $m$ Pauli strings
# $\Pi^{(k)} = \{\Pi_1^{(k)}, \Pi_2^{(k)}, \dots, \Pi_m^{(k)}\}$ and calculate
# their expectation values in an $n$-qubit parameterized state
# $\vert \Psi (\boldsymbol{\theta}) \rangle$:
#
# $$
# c_i
# = \langle \Psi (\boldsymbol{\theta}) \vert \Pi_i^{(k)} \vert \Psi (\boldsymbol{\theta}) \rangle \tag{5}
# $$
#
# PCE represents the binary variable $x_i$ by the sign of $c_i$, namely
# $\mathrm{sgn}(c_i)$.
# For the $k=2$ example above, assigning any of $X$, $Y$, or $Z$ independently to
# two of the $n$ qubits gives
# ${}_n C_2 \times 3^2 = \frac{3^2}{2} n (n-1)$ possible strings. Thus, this
# encoding can represent $m = \mathcal{O}(n^2)$ binary variables.
# Similarly, for $k=3$, there are
# ${}_n C_3 \times 3^3 = \frac{3^2}{2} n (n-1)(n-2)$ strings, allowing
# $m = \mathcal{O}(n^3)$ variables.
# The expectation values $c_i$ can be viewed as a continuous relaxation of the
# binary variables $x_i$.
# [Sciorilli et al. (2025)](https://www.nature.com/articles/s41467-024-55346-z)
# also discuss superpolynomial suppression of barren plateaus, a common
# difficulty in variational quantum algorithms.
#
# :::{note}
# The description here and the implementation below use the $k$-body Pauli
# encoding provided by Qamomile's `PCEConverter`, including mixed Pauli strings.
# This is an adaptation of the portfolio workflow in
# [Soloviev & Krompiec (2025)](https://arxiv.org/abs/2511.21305).
# :::
#
# ## Proposed Method
#
# With this background, let us examine the method proposed by
# [Soloviev & Krompiec (2025)](https://arxiv.org/abs/2511.21305).
#
# ### Market Graph Representation and Graph Partitioning
#
# The previous section introduced standard mean-variance portfolio optimization.
# The approach here instead constructs a portfolio using a weighted market graph.
# Assets are represented by nodes, and edge presence and weight are defined by
#
# $$
# w_{ij}
# = \left\{ \begin{array}{ll}
# 1 - \vert \rho_{ij} \vert & \mathrm{if} \ \vert \rho_{ij} \vert > \lambda \\
# \emptyset & \mathrm{otherwise}
# \end{array} \right. \tag{6}
# $$
#
# Here,
#
# $$
# \rho_{ij}
# = \frac{\mathrm{Cov} (r_i, r_j)}{\sigma_i \sigma_j} \tag{7}
# $$
#
# is the correlation coefficient between assets $i$ and $j$, and $\sigma_i$ is
# the standard deviation of the returns of asset $i$.
# Equation (6) assigns smaller weights $w_{ij}$ to pairs with larger absolute
# correlations. A MaxCut on this graph therefore favors cutting edges with
# weaker correlations, leaving more strongly correlated assets in each cluster.
# Another possibility is to use the correlations in Equation (7) directly as
# edge weights and apply MinCut, although that gives a different formulation.
# Recursively partitioning the graph with the weights in Equation (6) groups
# strongly correlated assets together. The cut function is
#
# $$
# \mathrm{Cut} (\mathcal{G}, \boldsymbol{x})
# = \sum_{(v_i, v_j) \in E} w_{ij} \{ x_i (1-x_j) + x_j (1-x_i) \} \tag{8}
# $$
#
# Here, $x_i \in \{0, 1\}$ indicates which cluster contains asset $i$.
# In Equation (8), an edge contributes zero when its endpoints belong to the
# same cluster and $w_{ij}$ when they belong to different clusters.
# Applying this partitioning recursively produces the clusters.
# For each resulting cluster $L_i$, calculate the return vector
# $\boldsymbol{\mu}_i = [\mu_i^1, \mu_i^2, \dots, \mu_i^{\vert L_i \vert}]$ and
# select the asset with the highest return:
#
# $$
# r_i
# = \mathrm{arg} \max_j \mu_i^j \tag{9}
# $$
#
# The resulting portfolio is
# $\boldsymbol{r} = \{r_1, r_2, \dots, r_{n_\mathrm{sp} + 1}\}$, where
# $n_\mathrm{sp}$ is the number of graph splits.
#
# ### Loss Function
#
# Equation (8) gives the cut objective. Its binary variables do not directly
# support continuous optimization methods such as gradient descent.
# We therefore use the loss function proposed by
# [Sciorilli et al. (2025)](https://www.nature.com/articles/s41467-024-55346-z):
#
# $$
# \mathcal{L}
# = \sum_{(i, j) \in E} w_{ij} \tanh (\alpha \langle \Pi_i \rangle) \tanh (\alpha \langle \Pi_j \rangle) + \mathcal{L}^\mathrm{reg} \tag{10}
# $$
#
# Here, $E$ is the edge set of the graph before partitioning, and
# $\langle \Pi_i \rangle = \langle \Psi (\boldsymbol{\theta}) \vert \Pi_i^{(k)} \vert \Psi (\boldsymbol{\theta}) \rangle$.
# The parameter $\alpha$ controls the steepness of $\tanh$; the paper uses
# $\alpha = n^{\lfloor k/2 \rfloor}$ as its optimal value.
# The term $\mathcal{L}^\mathrm{reg}$ represents regularization.
# Standard VQE and QAOA use a single Hamiltonian expectation value, whereas
# $\mathcal{L}$ here is a nonlinear function of the expectation values of $m$
# Pauli strings.

# %% [markdown]
# ## Implementation with Qamomile
#
# Let us implement the approach described by
# [Soloviev & Krompiec (2025)](https://arxiv.org/abs/2511.21305) with Qamomile.
#
# ### Preparing the Data
#
# We use Kaggle's S&P 500 dataset to calculate the required returns and their
# statistics. Assets listed during the dataset period have shorter histories
# and therefore missing values. We retain only assets with complete data for
# the entire period.

# %%
path = kagglehub.dataset_download("camnugent/sandp500")
print("dataset path:", path)

# Locate the CSV with glob because the extracted directory layout can vary by version.
csv_candidates = glob.glob(os.path.join(path, "**", "all_stocks_5yr.csv"), recursive=True)
print("found:", csv_candidates)
CSV = csv_candidates[0]

raw = pd.read_csv(CSV)
raw.columns = [c.lower() for c in raw.columns]   # Normalize headers such as date/Date.
raw["date"] = pd.to_datetime(raw["date"])

print(raw.shape)
print(f"tickers: {raw['name'].nunique()}")
print(f"period : {raw['date'].min().date()} - {raw['date'].max().date()}")

# Convert long-form data to a wide table (dates x assets).
wide = raw.pivot(index="date", columns="name", values="close").sort_index()

# Handle missing data by keeping only assets with a complete history.
n_days = len(wide)
complete = wide.columns[wide.notna().sum() == n_days]
print(f"complete history : {len(complete)} / {wide.shape[1]}")
wide = wide[complete]

# %% [markdown]
# Next, select the assets to use in the optimization. Here, we take the first
# `M` tickers in alphabetical order, then split their return time series into
# training and test data.

# %%
M = 30    # Number of assets (the paper uses 10, 20, 30, 50, 100, 150, 200, 250).

# Selection rule: take the first M tickers in alphabetical order.
# Alternatives include highest volume, sector-balanced sampling, or random sampling with a fixed seed.
tickers = sorted(wide.columns)[:M]
prices = wide[tickers]

returns = prices.pct_change().dropna()

# Split chronologically at an 80:20 ratio, as in the paper.
split = int(len(returns) * 0.8)
train, test = returns.iloc[:split], returns.iloc[split:]

print(f"tickers: {tickers}")
print(f"\ntrain: {train.shape}  ({train.index[0].date()} - {train.index[-1].date()})")
print(f"test : {test.shape}  ({test.index[0].date()} - {test.index[-1].date()})")

# Inspect extreme daily returns for effects of unadjusted stock splits and dividends.
print(f"\nLargest daily changes (possible stock splits):")
print(returns.abs().max().sort_values(ascending=False).head(5).round(3))

# %% [markdown]
# ### Constructing the Market Graph
#
# Construct the market graph according to Equation (6).
# Since the paper does not specify $\lambda$, first search for a value that
# reproduces the graph densities used in the paper.

# %%
# Calibrate lambda to the density range 0.53-0.87 in Table 1 of the paper.
# Stock correlations in this dataset are mostly positive and concentrated around 0.2-0.6.
# Repeat the sweep whenever M changes.
rho = train.corr().to_numpy()
iu = np.triu_indices(M, k=1)

print(f"|rho| distribution: min={np.abs(rho[iu]).min():.3f} "
      f"median={np.median(np.abs(rho[iu])):.3f} max={np.abs(rho[iu]).max():.3f}\n")

for lam in np.arange(0.10, 0.65, 0.05):
    A = (np.abs(rho) > lam) & ~np.eye(M, dtype=bool)
    g = nx.from_numpy_array(A)
    g.remove_edges_from(nx.selfloop_edges(g))
    mark = "  <-- within the paper's range" if 0.53 <= nx.density(g) <= 0.87 else ""
    print(f"lambda={lam:.2f}  edges={g.number_of_edges():5d}  "
          f"density={nx.density(g):.3f}  clustering={nx.average_clustering(g):.3f}{mark}")

# %% [markdown]
# Choose $\lambda$ from the sweep results and construct the graph.
# The following cell displays its properties.

# %%
LAMBDA = 0.25    # Choose this value based on the sweep above.

rho = train.corr().to_numpy()   # Recalculate so this cell also works without the sweep.

G = nx.Graph()
G.add_nodes_from(range(M))
nx.set_node_attributes(G, {i: t for i, t in enumerate(tickers)}, "ticker")
for i in range(M):
    for j in range(i + 1, M):
        if abs(rho[i, j]) > LAMBDA:
            G.add_edge(i, j, weight=1.0 - abs(rho[i, j]))

print(f"nodes          : {G.number_of_nodes()}")
print(f"edges          : {G.number_of_edges()}")
print(f"density        : {nx.density(G):.3f}   (paper: 0.53-0.87)")
print(f"avg degree     : {2 * G.number_of_edges() / M:.1f}")
print(f"avg clustering : {nx.average_clustering(G):.3f}   (paper: 0.70-0.95)")

# Isolated nodes are assets with no other asset satisfying |rho| > lambda.
# They do not contribute to the cut, so exclude them from encoding (see bipartition below).
iso = [v for v in G.nodes() if G.degree(v) == 0]
print(f"isolated       : {len(iso)}  {[tickers[v] for v in iso]}")

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(6, 5))
nx.draw(G, pos, node_size=250, node_color="white", edgecolors="black",
        width=[2 * d["weight"] for _, _, d in G.edges(data=True)])
plt.title(f"Market graph (lambda={LAMBDA})")
plt.show()

# %% [markdown]
# ### Setting the Hyperparameters
#
# Set the parameters used in the optimization. `K` is the correlator order,
# `BETA` sets the regularization strength, `MAXITER` is the maximum number of
# iterations of the classical optimizer, `DEPTH` is the number of layers in the
# hardware-efficient ansatz (HEA), and `SEED` is the random seed.

# %%
K = 2                      # Correlator order.
BETA = 0.5                 # Regularization strength.
MAXITER = 200
DEPTH = None                 # None selects the paper's p = floor(N/n).
SEED = 42


# %% [markdown]
# ### Defining the HEA
#
# Define the HEA with Qamomile using Ry and Rz rotations followed by a CX
# entangling layer.

# %%
@qmc.qkernel
def hea(
    n: qmc.UInt,
    depth: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    P: qmc.Observable,
) -> qmc.Float:
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


# %% [markdown]
# ### Graph Partitioning
#
# Next, define a function for partitioning a graph. Exclude nodes without edges
# from the encoding and relabel the remaining nodes with consecutive indices
# starting at zero. Convert weighted MaxCut into an Ising model, then use
# Qamomile's `PCEConverter` to encode $N$ variables in $n$ qubits.
# Relax each variable's Pauli expectation value using
# $\tanh(\alpha \langle \Pi \rangle)$ and minimize the resulting loss with COBYLA.
# Each loss evaluation involves $N$ circuit executions.

# %%
def bipartition(sub):
    # Split a subgraph into two parts with PCE and return {node: +-1}.
    nodes = sorted(sub.nodes())

    # --- Encode only nodes with edges. ---
    act = [v for v in nodes if sub.degree(v) > 0]
    iso = [v for v in nodes if sub.degree(v) == 0]

    # There is nothing to optimize if fewer than two nodes have edges.
    if len(act) < 2:
        return {v: (1 if i < len(nodes) // 2 else -1) for i, v in enumerate(nodes)}

    ix = {v: i for i, v in enumerate(act)}     # Relabel as 0..len(act)-1.
    n_var = len(act)

    # --- Weighted MaxCut -> Ising model. ---
    q = {(ix[u], ix[v]): 0.5 * d["weight"] for u, v, d in sub.edges(data=True)}
    w_tot = sum(d["weight"] for _, _, d in sub.edges(data=True))
    conv = PCEConverter(
        BinaryModel.from_ising(linear={}, quad=q, constant=-w_tot / 2.0),
        correlator_order=K,
    )
    nq = conv.num_qubits
    obs = conv.get_encoded_pauli_list()

    # Stop here if the variable count is wrong, rather than raising an IndexError later.
    assert len(obs) == n_var, f"Variable count mismatch: encoded={len(obs)} active={n_var}"

    # --- Transpile once for each observable. ---
    # P is fixed at transpilation time, so each variable needs its own program.
    # This is a major contributor to runtime.
    d_ = DEPTH if DEPTH is not None else max(1, n_var // nq)
    exes = [
        transpiler.transpile(hea, bindings={"n": nq, "depth": d_, "P": P},
                             parameters=["thetas"])
        for P in obs
    ]

    a = float(nq ** (K // 2))
    mw = w_tot / sub.number_of_edges()
    nu_ = w_tot / 2.0 + mw * (n_var - 1) / 4.0
    sm = conv.spin_model

    def f(p):
        th = list(p)
        sg = [np.tanh(a * e.run(executor, bindings={"thetas": th}).result()) for e in exes]
        # L = sum(J * sg[i] * sg[j] for (i, j), J in sm.quad.items())
        L = sum(d["weight"] * sg[ix[u]] * sg[ix[v]] for u, v, d in sub.edges(data=True))
        L += sum(h * sg[i] for i, h in sm.linear.items())
        return L + BETA * nu_ * (sum(s * s for s in sg) / n_var) ** 2

    p0 = np.random.default_rng(SEED).uniform(-np.pi, np.pi, 2 * nq * d_)
    r = minimize(f, p0, method="COBYLA", options={"maxiter": MAXITER})

    # --- Decode PCE by taking the signs of the expectation values. ---
    fe = [e.run(executor, bindings={"thetas": list(r.x)}).result() for e in exes]
    out = {v: (1 if fe[ix[v]] > 0 else -1) for v in act}
    for i, v in enumerate(iso):          # Isolated nodes do not contribute to the cut.
        out[v] = 1 if i % 2 == 0 else -1

    ae = np.abs(fe)
    # A median |<P_i>| stuck near zero suggests that alpha is too small.
    # Expectations of k-body operators can typically be as small as n^(-k/2).
    print(f"      n={n_var:3d} qubits={nq} ({n_var / nq:.1f}x) "
          f"loss={r.fun:+.3f} |<P>|med={np.median(ae):.4f} iso={len(iso)}")

    return out


# %% [markdown]
# Run the function above to partition the graph.
# Use `n_splits` to specify the number of splits.

# %%
# Paper settings: {2, 4, 6, 9} for m < 100; m/10 - 1 for m >= 100.
n_splits = M // 10 - 1 if M >= 100 else {10: 2, 20: 4, 30: 6, 50: 9}.get(M, M // 10)
print(f"n_splits = {n_splits}  ->  {n_splits + 1} clusters\n")

queue = deque([sorted(G.nodes())])
done = []
remaining = n_splits

while remaining > 0 and queue:
    nodes = queue.popleft()
    if len(nodes) <= 1:
        done.append(nodes)          # Cannot split; use the remaining budget elsewhere.
        continue

    sp = bipartition(G.subgraph(nodes))
    s1 = [v for v in nodes if sp[v] > 0]
    s2 = [v for v in nodes if sp[v] <= 0]

    if not s1 or not s2:
        # Identical signs indicate collapse despite regularization.
        print(f"  [warn] collapse on {len(nodes)} nodes -- check alpha/beta")
        done.append(nodes)
        continue

    print(f"  split {len(nodes):3d} -> {len(s1):3d} + {len(s2):3d}")
    queue.append(s1); queue.append(s2)
    remaining -= 1

clusters = done + list(queue)
print(f"\nclusters: {[len(c) for c in clusters]}")

# %% [markdown]
# Visualize the partitioned graph.

# %%
# Visualize the final partition.
color = {}
for ci, c in enumerate(clusters):
    for v in c:
        color[v] = ci

plt.figure(figsize=(6, 5))
nx.draw(G, pos, node_size=250, edgecolors="black", cmap=plt.cm.tab20,
        node_color=[color[v] for v in G.nodes()], vmin=0, vmax=19)
plt.title(f"{len(clusters)} correlation clusters")
plt.show()

# %% [markdown]
# ## Results
#
# From each cluster, select the asset with the highest mean return during the
# training period. Average the selected assets' daily returns during the test
# period and compound them to calculate the portfolio value over time.
# Display the final portfolio value and the annualized Sharpe ratio as well.

# %%
mu = train.mean().to_numpy()
reps = [max(c, key=lambda v: mu[v]) for c in clusters if len(c) > 0]

print(f"selected {len(reps)} assets: {[tickers[v] for v in reps]}")

equity = 1000 * (1 + test[[tickers[v] for v in reps]].mean(axis=1)).cumprod()
baseline = 1000 * (1 + test.mean(axis=1)).cumprod()


def sharpe(r: pd.Series) -> float:
    """Calculate annualized Sharpe with a zero risk-free rate."""
    std = r.std()
    return float(np.sqrt(252) * r.mean() / std) if std > 0 else 0.0

pce_returns = test[[tickers[v] for v in reps]].mean(axis=1)
baseline_returns = test.mean(axis=1)



print(f"\nPCE      final = {equity.iloc[-1]:8.2f}   sharpe = {sharpe(pce_returns):+.3f}")
print(f"baseline final = {baseline.iloc[-1]:8.2f}   sharpe = {sharpe(baseline_returns):+.3f}")

plt.figure(figsize=(8, 4))
plt.plot(equity.values, label=f"PCE ({len(reps)} assets)", color="#2696EB")
plt.plot(baseline.values, label=f"baseline ({M} assets)", color="#888", ls="--")
plt.xlabel("test day"); plt.ylabel("portfolio value")
plt.legend(); plt.title("Out-of-sample performance")
plt.show()

# %% [markdown]
# The plot also shows a baseline portfolio that uses all the assets.
# In this example, the portfolio selected with PCE has a higher value than the
# baseline on most test days. Its Sharpe ratio is also higher than the baseline.
#
# ## Summary
#
# This tutorial introduced a study of portfolio optimization using PCE and
# demonstrated an implementation with Qamomile.
# The main points are:
#
# * Qamomile provides PCE functionality through `PCEConverter`.
# * Constructing a graph from asset correlations and recursively applying MaxCut
#   groups strongly correlated assets into clusters.
# * In this example, selecting assets from these clusters improves the portfolio
#   return and Sharpe ratio relative to the baseline.

# %% [markdown]
#
