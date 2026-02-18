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
#     name: python3
# ---

# %% [markdown]
# # QAOAによるMax-Cut問題の解法
#
# このセクションでは、JijModelingとQamomileライブラリを使って、QAOAでMax-Cut問題を解きます。
#
# まず、使用する主要なライブラリをインストールしてインポートしましょう。

# %%
import jijmodeling as jm
import matplotlib.pyplot as plt

# %% [markdown]
# ## Max-Cut問題とは
#
# Max-Cut問題は、グラフの頂点を2つのグループに分割し、カットされるエッジの数（エッジに重みがある場合はカットされるエッジの総重み）が最大になるようにする問題です。ネットワーク分割や画像処理（セグメンテーション）などに応用されています。
# %%
import networkx as nx
import numpy as np

G = nx.Graph()
num_nodes = 5
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)
pos = {0: (1, 1), 1: (0, 1), 2: (-1, 0.5), 3: (0, 0), 4: (1, 0)}

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_title("Original Graph G=(V,E)")
nx.draw_networkx(G, pos, ax=ax, node_size=500, width=3, with_labels=True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 数学モデルの構築
#
# Max-Cut問題は以下の式で定式化できます:
#
# $$
#   \max \quad \frac{1}{2} \sum_{(i,j) \in E} (1 - s_i s_j)
# $$
#
# この式はイジング変数 $ s \in \{ +1, -1 \} $ で表現されていることに注意してください。今回はJijModelingのバイナリ変数 $ x \in \{ 0, 1 \} $ を使って定式化したいため、以下の式でイジング変数とバイナリ変数の変換を行います:
#
# $$
#     x_i = \frac{1 + s_i}{2} \quad \Rightarrow \quad s_i = 2x_i - 1
# $$
#


# %%
problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Graph()
    x = problem.BinaryVar(shape=(V,))

    obj = (
        E.rows()
        .map(lambda e: 1 / 2 * (1 - (2 * x[e[0]] - 1) * (2 * x[e[1]] - 1)))
        .sum()
    )
    problem += obj


problem

# %% [markdown]
# ## コンパイル済みインスタンスの作成
# 数学モデルとインスタンスデータを `problem.eval()` を使ってコンパイルします。この処理により、インスタンスデータが代入された問題の中間表現が得られます。

# %%
V = num_nodes
E = edges
data = {"V": V, "E": E}
instance = problem.eval(data)

# %% [markdown]
# ## コンパイル済みインスタンスからQAOA回路とハミルトニアンへの変換
#
# コンパイル済みInstanceからQAOA回路とハミルトニアンを生成します。このために使用するコンバータは `qamomile.optimization.qaoa` の `QAOAConverter` です。
#
# このクラスのインスタンスを作成すると、内部的にコンパイル済みInstanceからイジングハミルトニアンが生成されます。その後、以下のメソッドが利用できます:
# - `transpile()` でQAOA量子回路を生成
# - `get_cost_hamiltonian()` でコストハミルトニアンを確認
#
# QAOAの層数 $p$ はここでは $3$ に設定しています。

# %%
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create the QAOA converter and transpile
p = 3  # Number of QAOA layers
converter = QAOAConverter(instance)
executable = converter.transpile(
    transpiler=transpiler,
    p=p,
)

# %% [markdown]
# コストハミルトニアンを確認してみましょう。Max-Cutの目的関数はイジング変数 $s_i \in \{+1, -1\}$ で表現されるため、コストハミルトニアンはパウリZ演算子で構成されます。

# %%
cost_hamiltonian = converter.get_cost_hamiltonian()
cost_hamiltonian

# %% [markdown]
# グラフのエッジは $E = \{(0,1),(0,4),(1,2),(1,3),(2,3),(3,4)\}$ です。イジング形式でのMax-Cutの目的関数は $\frac{1}{2}\sum_{(i,j) \in E}(1 - Z_i Z_j)$ であるため、コストハミルトニアンには各エッジに対応する $Z_i Z_j$ 項が含まれるはずです。実際に、上記のハミルトニアンが期待されるイジング定式化と一致していることが確認できます。

# %% [markdown]
# 生成された量子回路を見てみましょう。この回路は、コスト層とミキサー層を交互に適用するQAOAアンサッツを実装しています。

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw(output="mpl")

# %% [markdown]
# ## VQE最適化
#
# 次に、変分最適化ループを設定します。scipyのCOBYLA最適化器を使用して、
# 最適なQAOAパラメータ（各層のgammaとbeta）を探索します。

# %%
from scipy.optimize import minimize

# List to save optimization history
energy_history = []


def objective_function(params, transpiler, executable, converter, shots=1024):
    """
    Objective function for VQE optimization.

    Args:
        params: Concatenated [gammas, betas] parameters
        transpiler: Quantum transpiler
        executable: Compiled QAOA circuit
        converter: QAOAConverter for decoding results
        shots: Number of measurement shots

    Returns:
        Expected energy value
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        transpiler.executor(),
        bindings={
            "gammas": gammas,
            "betas": betas,
        },
        shots=shots,
    )
    result = job.result()

    sampleset = converter.decode(result)
    energy = sampleset.energy_mean()
    energy_history.append(energy)

    return energy


# %%
# Run optimization
np.random.seed(901)

# Initial parameters: gamma in [0, 2π], beta in [0, π]
init_params = np.concatenate(
    [
        np.random.uniform(0, 2 * np.pi, size=p),  # gammas
        np.random.uniform(0, np.pi, size=p),  # betas
    ]
)

# Clear history
energy_history = []

print(f"Starting QAOA optimization with p={p} layers...")
print(f"Initial parameters: gammas={init_params[:p]}, betas={init_params[p:]}")

# Optimize with COBYLA method
result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, converter),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化結果の可視化
#
# 最適化プロセスの収束を可視化してみましょう。
#
# > **注:** Qamomileは内部的に最大化問題を最小化問題に変換するため、エネルギー値は負の値になります。エネルギーが $-5$ の場合、Max-Cutの目的関数値は $5$ に対応します。

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("QAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 最終解の分析
#
# 最適化された回路からサンプリングし、結果を分析しましょう。

# %%
# Sample with optimized parameters
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={
        "gammas": optimal_gammas,
        "betas": optimal_betas,
    },
    shots=4096,
)
result_final = job_final.result()

# Decode results using the converter
sampleset = converter.decode(result_final)

# Build frequency distribution over all sampled bitstrings
bitstrings = []
counts = []
energies = []
for i in range(len(sampleset.samples)):
    sample = sampleset.samples[i]
    bitstring_str = "".join(str(sample[j]) for j in range(num_nodes))
    bitstrings.append(bitstring_str)
    counts.append(sampleset.num_occurrences[i])
    energies.append(sampleset.energy[i])

# Sort by bitstring for consistent display
sorted_order = np.argsort(bitstrings)
bitstrings = [bitstrings[i] for i in sorted_order]
counts = [counts[i] for i in sorted_order]
energies = [energies[i] for i in sorted_order]

# Plot frequency distribution
fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(bitstrings))
bars = ax.bar(x_pos, counts)

# Highlight optimal solutions (energy = -5) with red bars
for i, e in enumerate(energies):
    if np.isclose(e, -5.0):
        bars[i].set_color("red")

ax.set_xticks(x_pos)
ax.set_xticklabels(bitstrings, rotation=90)
ax.set_xlabel("Bitstring")
ax.set_ylabel("Counts")
ax.set_title("QAOA Measurement Frequency Distribution (red = optimal, energy = -5)")
plt.tight_layout()
plt.show()

# %% [markdown]
# 赤いバーはエネルギーが $= -5$ のビット列を示しており、これはMax-Cutの最適解（6本中5本のエッジをカット）に対応しています。頻度分布から、QAOAがこれらの最適解に測定確率を集中させることに成功していることがわかります。

# %% [markdown]
# ## 解の可視化
#
# QAOAで見つけた最良解を元のグラフ上に可視化してみましょう。

# %%
# Get the best solution (lowest energy)
best_sample, best_energy, best_count = sampleset.lowest()
best_solution = {(i,): float(best_sample[i]) for i in range(num_nodes)}

print("Best solution found:")
print(f"  Bitstring: {''.join(str(best_sample[i]) for i in range(num_nodes))}")
print(f"  Energy: {best_energy:.4f}")


# Visualize the solution
def get_edge_colors(
    graph, cut_solution, in_cut_color="r", not_in_cut_color="b"
) -> tuple[list[str], list[str]]:
    cut_set_1 = [node[0] for node, value in cut_solution.items() if value == 1.0]
    cut_set_2 = [node for node in graph.nodes() if node not in cut_set_1]

    edge_colors = []
    for u, v, _ in graph.edges(data=True):
        if (u in cut_set_1 and v in cut_set_2) or (u in cut_set_2 and v in cut_set_1):
            edge_colors.append(in_cut_color)
        else:
            edge_colors.append(not_in_cut_color)
    node_colors = [
        "#2696EB" if node in cut_set_1 else "#EA9b26" for node in graph.nodes()
    ]
    return edge_colors, node_colors


edge_colors, node_colors = get_edge_colors(G, best_solution)
cut_edges = sum(1 for c in edge_colors if c == "r")

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title(f"QAOA Solution (Cut edges: {cut_edges})")
nx.draw_networkx(
    G,
    pos,
    ax=ax,
    node_size=500,
    width=3,
    with_labels=True,
    edge_color=edge_colors,
    node_color=node_colors,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 厳密解との比較
#
# Qamomileのコンバータは `ommx.v1.Instance` を受け取るため、量子計算の結果と古典ソルバーの結果を簡単に比較できます。同じインスタンスをSCIPで厳密に解き、QAOAの解と比較してみましょう。

# %%
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

solution = OMMXPySCIPOptAdapter.solve(instance)

print(f"Exact optimal value (Max-Cut): {int(solution.objective)}")
print(f"QAOA solution value:           {cut_edges}")

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、QamomileでQAOAを使ってMax-Cut問題を解く方法を実演しました:
#
# 1. **問題の定式化**: JijModelingを使ってMax-Cutをイジング問題として定式化しました
# 2. **ハミルトニアンと回路の生成**: `QAOAConverter` がコストハミルトニアンとQAOA回路を自動生成しました
# 3. **VQE最適化**: scipyのCOBYLA最適化器を使い、Qamomileで最適なQAOAパラメータを探索しました
# 4. **解の分析**: 頻度分布から、QAOAがMax-Cutの最適解に測定確率を集中させることを確認しました
