# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # QAOA によるグラフ分割
#
# 本チュートリアルでは、Quantum Approximate Optimization Algorithm（QAOA）を用いて
# **グラフ分割問題**を解く方法を紹介します。
#
# ワークフロー：
#
# ```
# JijModeling 問題定義 → problem.eval() → QAOAConverter → transpile → sample → decode
# ```
#
# 1. [JijModeling](https://www.documentation.jijzept.com/docs/jijmodeling/) で問題を定式化する。
# 2. 具体的なデータでインスタンスを作成する。
# 3. `QAOAConverter` を使って QAOA 回路とハミルトニアンを構築する。
# 4. 古典オプティマイザで変分パラメータを最適化する。
# 5. 最適化された回路をサンプリングし、結果をデコードする。

# %% [markdown]
# ## 問題の定式化
#
# 無向グラフ $G = (V, E)$ が与えられたとき、頂点を同じサイズの 2 つのグループに
# 分割し、グループ間のエッジ数を最小化することが目標です。
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
    problem += E.rows().map(
        lambda e: x[e[0]] * (1 - x[e[1]]) + x[e[1]] * (1 - x[e[0]])
    ).sum()

    # 制約条件：均等な分割サイズ
    problem += problem.Constraint("equal_partition", x.sum() == V / 2)


print(problem)

# %% [markdown]
# ## グラフインスタンス
#
# 再現性を確保するため、8 ノード 16 エッジの固定グラフを使用します。

# %%
import networkx as nx
import matplotlib.pyplot as plt

num_nodes = 8
edge_list = [
    [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [1, 5], [1, 7],
    [2, 3], [2, 6], [3, 5], [4, 5], [4, 6], [5, 6], [5, 7], [6, 7],
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
# 2. BINARY ドメインから SPIN ドメインへ変換。
# 3. Pauli-Z 演算子の和としてコストハミルトニアンを構築。
#
# 元の問題には制約があるため、QUBO 定式化では制約が**ペナルティ項**として
# 目的関数に組み込まれます。そのため、デコードされたサンプルのエネルギー値は
# 元の目的関数（カットエッジ数）とは**異なり**、ペナルティを含んでいます。
# 実行可能性の確認と真の目的関数値の計算を別途行う必要があります。

# %%
from qamomile.optimization.qaoa import QAOAConverter

converter = QAOAConverter(instance)
hamiltonian = converter.get_cost_hamiltonian()
print(hamiltonian)

# %% [markdown]
# ## 実行可能な回路へのトランスパイル
#
# `converter.transpile()` は `p` 層の QAOA アンザッツ回路を構築し、
# `ExecutableProgram` にコンパイルします。変分パラメータ `gammas`（コスト層）と
# `betas`（ミキサー層）はランタイムパラメータとして残ります。

# %%
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
p = 3  # QAOA の層数
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# ## QAOA パラメータの最適化
#
# `ExecutableProgram` からパラメトリック Qiskit 回路を取り出し、
# 測定ゲートを除去した上で `StatevectorEstimator` でコストハミルトニアンを
# 評価します。これにより、ショットノイズなしの厳密な期待値
# $\langle \psi(\gamma, \beta) | H_C | \psi(\gamma, \beta) \rangle$
# を計算し、SciPy の COBYLA オプティマイザでコストを最小化する
# `gammas` と `betas` を探索します。

# %%
import numpy as np
from scipy.optimize import minimize
from qiskit.primitives import StatevectorEstimator
from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

# 厳密推定のため測定なしの回路を取得
circuit = executable.quantum_circuit
circuit_no_meas = circuit.remove_final_measurements(inplace=False)

# ハミルトニアンを Qiskit SparsePauliOp に変換
qk_hamiltonian = hamiltonian_to_sparse_pauli_op(hamiltonian)

# パラメータ名から Qiskit Parameter オブジェクトへのマッピング
param_map = {param.name: param for param in circuit_no_meas.parameters}

estimator = StatevectorEstimator()

initial_params = [
    np.pi / 4, np.pi / 2, np.pi / 2,  # gammas
    np.pi / 4, np.pi / 4, np.pi / 2,  # betas
]

cost_history = []


def cost_fn(params):
    gammas = params[:p]
    betas = params[p:]
    bindings = {}
    for i in range(p):
        bindings[param_map[f"gammas[{i}]"]] = gammas[i]
        bindings[param_map[f"betas[{i}]"]] = betas[i]
    bound_circuit = circuit_no_meas.assign_parameters(bindings)
    job = estimator.run([(bound_circuit, qk_hamiltonian)])
    result = job.result()
    cost = float(result[0].data.evs)
    cost_history.append(cost)
    return cost


res = minimize(
    cost_fn,
    initial_params,
    method="COBYLA",
    options={"maxiter": 1000},
)

print(f"最適化されたコスト: {res.fun:.3f}")
print(f"最適パラメータ:     {[round(v, 4) for v in res.x]}")
print(f"関数評価回数:       {res.nfev}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (exact expectation value)")
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
    transpiler.executor(),
    shots=1000,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

decoded = converter.decode(sample_result)

# %% [markdown]
# ## 結果の分析
#
# ### 実行可能性のチェック
#
# QAOA のサンプルは**候補解**であり、元の制約を満たすとは限りません。
# 制約 $\sum x_u = |V|/2$ は QUBO のペナルティとして組み込まれているため、
# 制約を満たさないビット列も出力に含まれる可能性があります。
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
for sample, energy, occ in zip(decoded.samples, decoded.energy, decoded.num_occurrences):
    if is_feasible(sample):
        obj = count_cut_edges(sample, G)
        feasible_results.append((sample, obj, occ))

total_feasible = sum(occ for _, _, occ in feasible_results)
total_samples = sum(decoded.num_occurrences)

print(f"実行可能なサンプル: {total_feasible} / {total_samples} "
      f"({100 * total_feasible / total_samples:.1f}%)")

# %% [markdown]
# ### 最良の実行可能解
#
# 実行可能なサンプルの中から、カットエッジ数（真の目的関数）が最小のものを選択します。

# %%
if feasible_results:
    feasible_results.sort(key=lambda x: x[1])
    best_sample, best_obj, best_count = feasible_results[0]
    print(f"最良の実行可能解:   {best_sample}")
    print(f"カットエッジ数:     {best_obj}")
    print(f"観測回数:          {best_count}")
else:
    print("実行可能な解が見つかりませんでした。p または maxiter を増やしてみてください。")
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
        "#FF6B6B" if best_sample.get(i, 0) == 1 else "#4ECDC4"
        for i in range(num_nodes)
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
