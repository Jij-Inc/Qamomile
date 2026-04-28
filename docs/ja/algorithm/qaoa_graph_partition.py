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
# title: QAOA によるグラフ分割
# tags: [qaoa, optimization, variational, graph, graph-partition, jijmodeling, built-in, intermediate]
# ---
#
# # QAOA によるグラフ分割
#
# <!-- BEGIN auto-tags -->
# **タグ:** [`qaoa`](../tags/qaoa.md) · [`optimization`](../tags/optimization.md) · [`variational`](../tags/variational.md) · [`graph`](../tags/graph.md) · [`graph-partition`](../tags/graph-partition.md) · [`jijmodeling`](../tags/jijmodeling.md) · [`built-in`](../tags/built-in.md) · [`intermediate`](../tags/intermediate.md)
# <!-- END auto-tags -->
#
# 本チュートリアルでは、Quantum Approximate Optimization Algorithm（QAOA）を用いて**グラフ分割問題**を解く方法を紹介します。
#
# ワークフロー：
#
# ```
# JijModeling 問題定義 → problem.eval() → QAOAConverter → transpile → sample → decode
# ```
#
# 1. [JijModeling](https://jij-inc-jijmodeling-tutorials-ja.readthedocs-hosted.com/ja/latest/introduction.html) で問題を定式化する。
# 2. 具体的なデータでインスタンスを作成する。
# 3. `QAOAConverter` を使って QAOA 回路とハミルトニアンを構築する。
# 4. 古典オプティマイザで変分パラメータを最適化する。
# 5. 最適化された回路をサンプリングし、結果をデコードする。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %% [markdown]
# ## 問題の定式化
#
# 無向グラフ $G = (V, E)$ が与えられたとき、頂点を同じサイズの 2 つのグループに分割し、グループ間のエッジ数を最小化することが目標です。
#
# **目的関数：**
#
# $$
# \min \sum_{(u,v) \in E} \bigl[x_u(1 - x_v) + x_v(1 - x_u)\bigr]
# $$
#
# **制約条件：**
#
# $$
# \sum_{u \in V} x_u = \frac{|V|}{2}
# $$
#
# ここで $x_u \in \{0, 1\}$ は頂点 $u$ がどちらのグループに属するかを表します。

# %% [markdown]
# ## JijModeling による問題定義

# %%
import jijmodeling as jm

problem = jm.Problem("Graph Partitioning")


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Natural(ndim=2)  # エッジリスト: [[u1,v1], [u2,v2], ...]
    x = problem.BinaryVar(shape=(V,))

    # 目的関数：分割間のカットエッジ数を最小化
    problem += (
        E.rows().map(lambda e: x[e[0]] * (1 - x[e[1]]) + x[e[1]] * (1 - x[e[0]])).sum()
    )

    # 制約条件：均等な分割サイズ
    problem += problem.Constraint("equal_partition", x.sum() == V / 2)


problem

# %% [markdown]
# ## グラフインスタンス
#
# 再現性を確保するため、8 ノード 16 エッジの固定グラフを使用します。

# %%
import matplotlib.pyplot as plt
import networkx as nx

num_nodes = 8
edge_list = [
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 7],
    [2, 3],
    [2, 6],
    [3, 5],
    [4, 5],
    [4, 6],
    [5, 6],
    [5, 7],
    [6, 7],
]

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## インスタンスの作成
#
# グラフからエッジリストを取得し、JijModeling の問題を具体的なデータで評価します。

# %%
instance_data = {"V": num_nodes, "E": edge_list}
instance = problem.eval(instance_data)

# %% [markdown]
# ## QAOAConverter のセットアップ
#
# `QAOAConverter` は OMMX インスタンスを受け取り、内部で以下を行います：
# 1. 問題を QUBO（Quadratic Unconstrained Binary Optimization）形式に変換。
# 2. BINARY 変数から SPIN 変数へ変換。
# 3. Pauli-Z 演算子の和としてコストハミルトニアンを構築。
#
# 元の問題には制約があるため、QUBO 定式化では制約が**ペナルティ項**として目的関数に組み込まれます。そのため、デコードされたサンプルのエネルギー値は元の目的関数（カットエッジ数）とは**異なり**、ペナルティを含んでいます。実行可能性の確認と真の目的関数値の計算を別途行う必要があります。

# %%
from qamomile.optimization.qaoa import QAOAConverter

converter = QAOAConverter(instance)
converter.spin_model = converter.spin_model.normalize_by_abs_max()
hamiltonian = converter.get_cost_hamiltonian()
print(hamiltonian)

# %% [markdown]
# ## 実行可能な回路へのトランスパイル
#
# `converter.transpile()` は `p` 層の QAOA アンザッツ回路を構築し、`ExecutableProgram` にコンパイルします。変分パラメータ `gammas`（コスト層）と `betas`（ミキサー層）はランタイムパラメータとして残ります。

# %%
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
p = 5  # QAOA の層数
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# ## QAOA パラメータの最適化
#
# `executable.sample()` を使って各イテレーションでコストを評価します。オプティマイザはサンプリングされたビット列の平均エネルギーを最小化する `gammas` と `betas` を探索します。

# %%
import os
import numpy as np
from scipy.optimize import minimize

executor = transpiler.executor()
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 25 if docs_test_mode else 1000

rng = np.random.default_rng(900)
initial_params = rng.uniform(0, np.pi, 2 * p)

cost_history = []


def cost_fn(params):
    gammas = list(params[:p])
    betas = list(params[p:])
    job = executable.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()
    decoded = converter.decode(result)
    energy = decoded.energy_mean()
    cost_history.append(energy)
    return energy


res = minimize(
    cost_fn,
    initial_params,
    method="COBYLA",
    options={"maxiter": maxiter},
)

print(f"Optimized cost: {res.fun:.3f}")
print(f"Optimal params: {[round(v, 4) for v in res.x]}")
print(f"Function evaluations: {res.nfev}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean energy)")
plt.title("QAOA Optimization Progress")
plt.show()

# %% [markdown]
# ## 最適化されたパラメータでサンプリング
#
# 最適化されたパラメータを使い、回路をサンプリングしてビット列として候補解を取得します。

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])

sample_result = executable.sample(
    executor,
    shots=1000,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

decoded = converter.decode(sample_result)

# %% [markdown]
# ## 結果の分析
#
# ### 実行可能性のチェック
#
# QAOA のサンプルは**候補解**であり、元の制約を満たすとは限りません。制約 $\sum x_u = |V|/2$ は QUBO のペナルティとして組み込まれているため、制約を満たさないビット列も出力に含まれる可能性があります。
#
# サンプルを有効な分割として解釈する前に、実行可能性でフィルタリングする必要があります。


# %%
def is_feasible(sample: dict[int, int]) -> bool:
    """サンプルが均等分割の制約を満たすかチェック"""
    return sum(sample.values()) == num_nodes // 2


def count_cut_edges(sample: dict[int, int], graph: nx.Graph) -> int:
    """真の目的関数値（2つの分割間のエッジ数）を計算"""
    cuts = 0
    for u, v in graph.edges():
        if sample.get(u, 0) != sample.get(v, 0):
            cuts += 1
    return cuts


# %%
feasible_results = []
for sample, energy, occ in zip(
    decoded.samples, decoded.energy, decoded.num_occurrences
):
    if is_feasible(sample):
        obj = count_cut_edges(sample, G)
        feasible_results.append((sample, obj, occ))

total_feasible = sum(occ for _, _, occ in feasible_results)
total_samples = sum(decoded.num_occurrences)

print(
    f"Feasible samples: {total_feasible} / {total_samples} "
    f"({100 * total_feasible / total_samples:.1f}%)"
)

# %% [markdown]
# ### 最良の実行可能解
#
# 実行可能なサンプルの中から、カットエッジ数（真の目的関数）が最小のものを選択します。

# %%
if feasible_results:
    feasible_results.sort(key=lambda x: x[1])
    best_sample, best_obj, best_count = feasible_results[0]
    print(f"Best feasible solution: {best_sample}")
    print(f"Cut edges:             {best_obj}")
    print(f"Occurrences:           {best_count}")
else:
    print("No feasible solution found. Try increasing p or maxiter.")
    best_sample = None

# %% [markdown]
# ### 目的関数値の分布
#
# 実行可能なサンプルのみについて、真の目的関数値（カットエッジ数）の分布を表示します。

# %%
from collections import Counter

if feasible_results:
    obj_counts = Counter()
    for _, obj, occ in feasible_results:
        obj_counts[obj] += occ

    objs = sorted(obj_counts.keys())
    counts = [obj_counts[o] for o in objs]

    plt.figure(figsize=(8, 4))
    plt.bar([str(o) for o in objs], counts, color="#2696EB")
    plt.xlabel("Cut edges (objective value)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Feasible Solutions")
    plt.show()

# %% [markdown]
# ### 最良の分割の可視化
#
# QAOA が見つけた最良の実行可能な分割に基づいて、グラフのノードを色分けします。

# %%
if best_sample is not None:
    color_map = [
        "#FF6B6B" if best_sample.get(i, 0) == 1 else "#4ECDC4" for i in range(num_nodes)
    ]

    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=700,
        edgecolors="black",
    )
    plt.title(f"Best feasible partition (cut edges = {best_obj})")
    plt.show()
