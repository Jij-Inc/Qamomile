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
# # QRAO による最大カット問題の解法
#
# このセクションでは、JijModeling と Qamomile リを使用して、QAOA で最大カット問題を解きます。
#
# まず、使用する主要なライブラリをインポートします。

# %%
import jijmodeling as jm
import ommx.v1
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## 最大カット問題とは
#
# 最大カット問題は、グラフのノードを2つのグループに分割して、カットされるエッジの数(またはエッジに重みがある場合はカットされるエッジの総重み)が最大になるようにする問題です。応用例としては、ネットワーク分割や画像処理(セグメンテーション)などがあります。

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
# ## 数理モデルの構築
#
# 最大カット問題は次の式で定式化できます:
#
# $$
#   \max \quad \frac{1}{2} \sum_{(i,j) \in E} (1 - s_i s_j)  
# $$
#
# この式はイジング変数 $ s \in \{ +1, -1 \} $ を使って表現されています。ここでは、JijModeling のバイナリ変数 $ x \in \{ 0, 1 \} $ を使って定式化したいので、次の式を使ってイジング変数とバイナリ変数の変換を行います:
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
# 次に、以下のグラフに対して最大カット問題を解きます。解く具体的な問題のデータは、インスタンスデータと呼ばれます。

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
# 先ほど準備した定式化とインスタンスデータを使って、`JijModeling.Interpreter` と `ommx.Instance` でコンパイルを実行します。このプロセスにより、インスタンスデータが代入された問題の中間表現が得られます。

# %%
interpreter = jm.Interpreter(data)
instance = interpreter.eval_problem(problem)

# %% [markdown]
# ## コンパイル済みインスタンスから QAOA 回路とハミルトニアンへの変換
#
# コンパイル済みインスタンスから QAOA 回路とハミルトニアンを生成します。これらを生成するために使用するコンバーターは `qm.optimization.qaoa.QAOAConverter` です。
#
# このクラスのインスタンスを作成して `ising_encode` を使用すると、コンパイル済みインスタンスから内部的にイジングハミルトニアンを生成できます。QUBO への変換時に発生するパラメータもここで設定できます。設定しない場合は、デフォルト値が使用されます。
#
# イジングハミルトニアンが生成されたら、QAOA 量子回路とハミルトニアンをそれぞれ生成できます。これらは `get_qaoa_ansatz` と `get_cost_hamiltonian` メソッドを使って実行できます。ここでは QAOA の層数 $p$ を 3 に固定しています。  

# %%
import qamomile.circuit as qmc
from qamomile.optimization.qrao.qrao31 import QRAC31Converter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create the QRAO31 Hamiltonian
converter = QRAC31Converter(instance)
hamiltonin = converter.get_cost_hamiltonian()


# %% [markdown]
# VQEを用いてこのハミルトニアンの基底状態を探索します。
from qamomile.circuit.algorithm.hardware_efficient_ansatz import hardware_efficient_ansatz


@qmc.qkernel
def vqe(n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    q = hardware_efficient_ansatz(q, depth, theta)
    return qmc.expval(q, h)


transpiler = QiskitTranspiler()
executable = transpiler.transpile(
    vqe,
    bindings={
        "n": hamiltonin.num_qubits,
        "h": hamiltonin,
        "depth": 2,
    },
    parameters=["theta"]
)

# %% [markdown]
# ## VQE 最適化
#
# 変分最適化ループを設定します。scipy の COBYLA オプティマイザーを使用して、
# 最適な VQE パラメータを見つけます。

# %%
from scipy.optimize import minimize
from qamomile.circuit.algorithm.hardware_efficient_ansatz import num_parameters

# パラメータ数を計算
depth = 2
n_qubits = hamiltonin.num_qubits
n_params = num_parameters(n_qubits, depth)

print(f"量子ビット数: {n_qubits}")
print(f"深さ: {depth}")
print(f"パラメータ数: {n_params}")

# 最適化履歴を保存するリスト
energy_history = []


def objective_function(params, executable):
    """
    VQE 最適化のための目的関数。
    """
    job = executable.run(
        transpiler.executor(),
        bindings={
            "theta": params,
        },
    )
    energy = job.result()
    energy_history.append(energy)
    return energy


# %%
# 最適化を実行
np.random.seed(42)

# 初期パラメータ
init_params = np.random.uniform(0, 2 * np.pi, size=n_params)

# 履歴をクリア
energy_history = []

print(f"VQE 最適化を開始...")
print(f"初期パラメータ数: {len(init_params)}")

# COBYLA 法で最適化
result_opt = minimize(
    objective_function,
    init_params,
    args=(executable,),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print(f"\n最適化完了")
print(f"最終エネルギー: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化結果の可視化
#
# 最適化プロセスの収束を可視化してみましょう。

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker='o', markersize=3)
plt.xlabel("反復回数")
plt.ylabel("エネルギー")
plt.title("VQE 最適化の収束")
plt.grid(True)
plt.tight_layout()
# plt.show()

# %% [markdown]
# ## QRAO31 デコード処理
#
# QRAO31 では、各変数が特定のパウリ演算子（X、Y、Z）にエンコードされています。
# 最適化された状態から、各パウリ演算子の期待値を測定して、元の変数の値を復元します。
#
# 期待値が正なら +1（スピン表現）、負なら -1 と推定します。

# %%
# 各変数に対応するパウリ演算子の期待値を測定するための回路を作成
pauli_observables = converter.get_encoded_pauli_list()

print(f"測定するパウリ演算子の数: {len(pauli_observables)}")
print(f"変数のインデックスとパウリ演算子のマッピング:")
for idx, pauli_op in converter.pauli_encoding.items():
    print(f"  変数 {idx} -> {pauli_op}")


# %%
# 各パウリ演算子の期待値を測定
@qmc.qkernel
def measure_pauli(n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    q = hardware_efficient_ansatz(q, depth, theta)
    return qmc.expval(q, h)


expectations = []
optimal_params = result_opt.x

for i, pauli_obs in enumerate(pauli_observables):
    executable_pauli = transpiler.transpile(
        measure_pauli,
        bindings={
            "n": n_qubits,
            "h": pauli_obs,
            "depth": depth,
        },
        parameters=["theta"]
    )

    job = executable_pauli.run(
        transpiler.executor(),
        bindings={
            "theta": optimal_params,
        },
    )
    expectation = job.result()
    expectations.append(expectation)

print("各変数のパウリ期待値:")
for i, exp in enumerate(expectations):
    print(f"  変数 {i}: {exp:.4f}")

# %% [markdown]
# ## 解の復元
#
# SignRounder を使って期待値から元の変数の値（スピン）を復元します。

# %%
from qamomile.optimization.qrao import SignRounder

rounder = SignRounder()
spins = rounder.round(expectations)

print("復元されたスピン値 (+1 または -1):")
for i, spin in enumerate(spins):
    print(f"  変数 {i}: {spin}")

# スピンをバイナリに変換 (spin=+1 -> binary=0, spin=-1 -> binary=1)
binary_solution = [(1 - s) // 2 for s in spins]

print("\nバイナリ解 (0 または 1):")
for i, bit in enumerate(binary_solution):
    print(f"  変数 {i}: {bit}")

# %% [markdown]
# ## 解の可視化
#
# QRAO31 で見つかった解を元のグラフ上で可視化してみましょう。

# %%
# 解を辞書形式に変換
solution_dict = {(i,): float(bit) for i, bit in enumerate(binary_solution)}

# エネルギーを計算
# スピン表現でのエネルギー計算
def calculate_maxcut_value(graph, binary_solution):
    """最大カット問題の目的関数値を計算"""
    cut_count = 0
    for u, v in graph.edges():
        if binary_solution[u] != binary_solution[v]:
            cut_count += 1
    return cut_count

cut_value = calculate_maxcut_value(G, binary_solution)

print(f"\n見つかった解:")
print(f"  バイナリ列: {''.join(map(str, binary_solution))}")
print(f"  カットされたエッジ数: {cut_value}")

# 解を可視化
edge_colors, node_colors = get_edge_colors(G, solution_dict)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title(f"QRAO31 の解 (カットされたエッジ数: {cut_value})")
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
# このチュートリアルでは、Qamomile を使って QRAO31 で最大カット問題を解く方法を実演しました:
#
# 1. **問題の定式化**: JijModeling を使って最大カットを QUBO/イジング問題として定式化しました
# 2. **QRAO31 エンコーディング**: `QRAC31Converter` が3つの変数を1量子ビットにエンコードし、量子ビット数を削減しました
# 3. **VQE 最適化**: Hardware-efficient ansatz を使って最適なパラメータを見つけました
# 4. **デコード**: パウリ演算子の期待値を測定し、SignRounder で元の変数の値を復元しました
# 5. **解の解析**: 測定結果を解析し、解を可視化しました
#
# QRAO31 で Qamomile を使う主な利点:
# - **量子ビット数の削減**: 最大3倍の変数を少ない量子ビット数で表現可能
# - **数理定式化からの自動変換**: JijModeling から自動的に QRAC エンコードされたハミルトニアンを生成
# - **バックエンドに依存しない**: Qiskit、Quri-Parts、PennyLane などで動作
# - **古典最適化との統合**: scipy などの古典最適化ライブラリと簡単に統合可能
#
# QRAO31 は、NISQ デバイスの限られた量子ビット数を効率的に使用する強力な手法です。

# %%
