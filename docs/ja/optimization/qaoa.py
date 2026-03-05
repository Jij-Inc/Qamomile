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
# このセクションでは、JijModelingとQamomileライブラリを使ってQAOAでMaxcut問題を解きます。
#
# まず、使用するメインライブラリをインストールしてインポートしましょう。

# %%
import jijmodeling as jm
import ommx.v1
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Max-Cut問題とは
#
# Max-Cut問題は、グラフの頂点を2つのグループに分割し、カットされるエッジの数（またはエッジに重みがある場合はカットされるエッジの総重み）が最大になるようにする問題です。ネットワーク分割や画像処理（セグメンテーション）などに応用されています。

# %%
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
num_nodes = 5
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)
pos = {0: (1, 1), 1: (0, 1), 2: (-1, 0.5), 3: (0, 0), 4: (1, 0)}

cut_solution = {(1,): 1.0, (2,): 1.0, (4,): 1.0}
edge_colors = []


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
    node_colors = ["#2696EB" if node in cut_set_1 else "#EA9b26" for node in G.nodes()]
    return edge_colors, node_colors


edge_colors, node_colors = get_edge_colors(G, cut_solution)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].set_title("Original Graph G=(V,E)")
nx.draw_networkx(G, pos, ax=axes[0], node_size=500, width=3, with_labels=True)
axes[1].set_title("MaxCut Solution Visualization")
nx.draw_networkx(
    G,
    pos,
    ax=axes[1],
    node_size=500,
    width=3,
    with_labels=True,
    edge_color=edge_colors,
    node_color=node_colors,
)

plt.tight_layout()
# plt.show()


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
def Maxcut_problem() -> jm.Problem:
    V = jm.Placeholder("V")
    E = jm.Placeholder("E", ndim=2)
    x = jm.BinaryVar("x", shape=(V,))
    e = jm.Element("e", belong_to=E)
    i = jm.Element("i", belong_to=V)
    j = jm.Element("j", belong_to=V)

    problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)
    si = 2 * x[e[0]] - 1
    sj = 2 * x[e[1]] - 1
    si.set_latex("s_{e[0]}")
    sj.set_latex("s_{e[1]}")
    obj = 1 / 2 * jm.sum(e, (1 - si * sj))
    problem += obj
    return problem


problem = Maxcut_problem()
problem

# %% [markdown]
# ## インスタンスデータの準備
#
# 次に、以下のグラフに対するMax-Cut問題を解きます。解く具体的な問題のデータをインスタンスデータと呼びます。

# %%
import networkx as nx
import numpy as np
from IPython.display import display, Latex

G = nx.Graph()
num_nodes = 5
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)

weight_matrix = nx.to_numpy_array(G, nodelist=list(range(num_nodes)))

plt.title("G=(V,E)")
plt.plot(figsize=(5, 4))

nx.draw_networkx(G, pos, node_size=500)

# %%
V = num_nodes
E = edges

data = {"V": V, "E": E}

data

# %% [markdown]
# ## コンパイル済みインスタンスの作成
# `JijModeling.Interpreter` と `ommx.Instance` を使用して、先に準備した定式化とインスタンスデータからコンパイルを行います。この処理により、インスタンスデータが代入された問題の中間表現が得られます。

# %%
interpreter = jm.Interpreter(data)
instance = interpreter.eval_problem(problem)

# %% [markdown]
# ## コンパイル済みインスタンスからQAOA回路とハミルトニアンへの変換
#
# コンパイル済みInstanceからQAOA回路とハミルトニアンを生成します。これらを生成するためのコンバータが `qm.optimization.qaoa.QAOAConverter` です。
#
# このクラスのインスタンスを作成し、`ising_encode` を使用すると、内部的にコンパイル済みInstanceからイジングハミルトニアンを生成できます。QUBOへの変換時に生じるパラメータもここで設定できます。設定しない場合はデフォルト値が使用されます。
#
# イジングハミルトニアンが生成されたら、QAOA量子回路とハミルトニアンをそれぞれ生成できます。`get_qaoa_ansatz` メソッドと `get_cost_hamiltonian` メソッドで取得できます。QAOAの層数 $p$ は $7$ に固定しています。

# %%
import qamomile.circuit as qmc
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# QAOAコンバータを作成してトランスパイル
p = 3  # QAOAの層数
converter = QAOAConverter(instance)
executable = converter.transpile(
    transpiler=transpiler,
    p=p,
)

# %% [markdown]
# 生成された量子回路を見てみましょう。この回路は、コスト層とミキサー層を交互に適用するQAOAアンサッツを実装しています。

# %%
qiskit_circuit = executable.get_first_circuit()
if qiskit_circuit is not None:
    print(f"量子ビット数: {qiskit_circuit.num_qubits}")
    print(f"回路の深さ: {qiskit_circuit.depth()}")

# %% [markdown]
# ## エネルギー計算
#
# QAOAを最適化するには、測定結果から期待エネルギーを計算する必要があります。
# コンバータはイジングモデルへのアクセスを提供しており、これを使ってエネルギーを計算します。

# %%
def calculate_ising_energy(bitstring: list[int], ising_model) -> float:
    """
    ビット列からイジングモデルのエネルギーを計算する。

    ビット列 z_i ∈ {0, 1} をスピン s_i ∈ {-1, +1} に変換。
    変換規則: s_i = 1 - 2*z_i (z_i=0 → s_i=+1, z_i=1 → s_i=-1)
    """
    # ビットをスピンに変換
    spins = [1 - 2 * b for b in bitstring]
    return ising_model.calc_energy(spins)


def calculate_expectation_value(sample_result, ising_model) -> float:
    """
    測定結果から期待エネルギー値を計算する。
    """
    total_energy = 0.0
    total_counts = 0

    for bitstring, count in sample_result.results:
        energy = calculate_ising_energy(bitstring, ising_model)
        total_energy += energy * count
        total_counts += count

    return total_energy / total_counts


# %% [markdown]
# ## VQE最適化
#
# 変分最適化ループを設定します。scipyのCOBYLA最適化器を使用して、
# 最適なQAOAパラメータ（各層のgammaとbeta）を見つけます。

# %%
from scipy.optimize import minimize

# 最適化履歴を保存するリスト
energy_history = []


def objective_function(params, transpiler, executable, ising_model, shots=1024):
    """
    VQE最適化の目的関数。

    Args:
        params: 連結された [gammas, betas] パラメータ
        transpiler: 量子トランスパイラ
        executable: コンパイル済みQAOA回路
        ising_model: エネルギー計算用イジングモデル
        shots: 測定ショット数

    Returns:
        期待エネルギー値
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

    energy = calculate_expectation_value(result, ising_model)
    energy_history.append(energy)

    return energy


# %%
# 最適化の実行
np.random.seed(42)

# 初期パラメータ: gamma ∈ [0, 2π], beta ∈ [0, π]
init_params = np.concatenate([
    np.random.uniform(0, 2 * np.pi, size=p),  # gammas
    np.random.uniform(0, np.pi, size=p),       # betas
])

# 履歴をクリア
energy_history = []

print(f"QAOA最適化を開始します (p={p}層)...")
print(f"初期パラメータ: gammas={init_params[:p]}, betas={init_params[p:]}")

# COBYLAメソッドで最適化
result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, converter.ising),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print(f"\n最適化されたパラメータ:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
print(f"最終エネルギー: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化結果の可視化
#
# 最適化プロセスの収束を可視化してみましょう。

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker='o', markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("QAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
# plt.show()

# %% [markdown]
# ## 最終解の分析
#
# 最適化された回路からサンプリングし、結果を分析します。

# %%
# 最適化されたパラメータでサンプリング
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

# エネルギーでソート
results_with_energy = []
for bitstring, count in result_final.results:
    energy = calculate_ising_energy(bitstring, converter.ising)
    results_with_energy.append((bitstring, count, energy))

results_with_energy.sort(key=lambda x: x[2])

print("測定結果（エネルギー順）:")
print("-" * 60)
for bitstring, count, energy in results_with_energy[:10]:
    bitstring_str = "".join(map(str, bitstring))
    probability = count / 4096
    print(f"  {bitstring_str}: count={count:4d}, probability={probability:.3f}, energy={energy:.4f}")

# %% [markdown]
# ## 解の可視化
#
# QAOAで見つけた最良解を元のグラフ上に可視化してみましょう。

# %%
# 最良解（最低エネルギー）を取得
best_bitstring, best_count, best_energy = results_with_energy[0]
best_solution = {(i,): float(bit) for i, bit in enumerate(best_bitstring)}

print(f"\n見つかった最良解:")
print(f"  ビット列: {''.join(map(str, best_bitstring))}")
print(f"  エネルギー: {best_energy:.4f}")

# 解を可視化
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
# plt.show()

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、QamomileでQAOAを使ってMax-Cut問題を解く方法を実演しました:
#
# 1. **問題の定式化**: JijModelingを使ってMax-CutをQUBO/イジング問題として定式化
# 2. **回路の生成**: Qamomileの`QAOAConverter`がQAOA回路を自動生成
# 3. **VQE最適化**: scipyのCOBYLA最適化器で最適なQAOAパラメータを探索
# 4. **解の分析**: 測定結果を分析し、解を可視化
#
# QamomileをQAOAに使う主な利点:
# - 数学的定式化からイジングモデルを自動生成
# - バックエンド非依存の回路定義（Qiskit、Quri-Parts、PennyLane等で動作）
# - 古典最適化ライブラリとの容易な連携
