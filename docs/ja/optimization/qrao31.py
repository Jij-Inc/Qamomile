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
# # QRAOによるMax-Cut問題の解法
#
# **Quantum Random Access Optimization (QRAO)** は、Quantum Random Access Coding (QRAC) を用いて複数の古典変数を1つの量子ビットにエンコードする手法です。標準的なQAOAでは各変数が1つの量子ビットを占有するのに対し、QRAOでは必要な量子ビット数を大幅に削減できます。
#
# 例えば、QRAC(3,1,p)エンコーディングでは最大3つの変数を1つの量子ビットに詰め込むことができます。QAOAでは12量子ビット必要な12変数問題が、QRAOではわずか4量子ビット程度で解ける可能性があります。
#
# このチュートリアルでは、QamomileのQRAO31（QRAC(3,1,p)バリアント）を使ってMax-Cut問題を解きます。

# %% [markdown]
# ## QamomileにおけるQRAOバリアント
#
# Qamomileではエンコーディング密度と近似精度のトレードオフが異なる複数のQRAOバリアントを提供しています。すべてのコンバータは `qamomile.optimization.qrao` から利用できます。
#
# | バリアント | コンバータクラス | 変数数 / 量子ビット | 説明 |
# |---------|----------------|-------------------|-------------|
# | QRAC(2,1,p) | `QRAC21Converter` | 最大2 | Z演算子とX演算子を使って1量子ビットに2変数をエンコード |
# | QRAC(3,1,p) | `QRAC31Converter` | 最大3 | Z、X、Y演算子を使って1量子ビットに3変数をエンコード |
# | QRAC(3,2,p) | `QRAC32Converter` | 最大3 | 2量子ビットのプライム演算子を使用し、より高い忠実度を実現 |
# | Space-efficient | `QRACSpaceEfficientConverter` | 2（固定） | グラフ彩色不要で、一定の2:1圧縮を実現 |
#
# このチュートリアルでは、1量子ビットあたりの圧縮率が最も高い **QRAC(3,1,p)** を使用します。

# %%
import jijmodeling as jm
import matplotlib.pyplot as plt

# %% [markdown]
# ## Max-Cut問題とは
#
# Max-Cut問題は、グラフの頂点を2つのグループに分割し、カットされるエッジの数（またはエッジに重みがある場合はカットされるエッジの総重み）が最大になるようにする問題です。ネットワーク分割や画像処理（セグメンテーション）などに応用されています。
# %%
import networkx as nx
import numpy as np

G = nx.Graph()
num_nodes = 12
# Generalized Petersen graph GP(6,2): 12 nodes, 18 edges, 3-regular
edges = [
    # Outer hexagon
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    # Spokes
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (4, 10),
    (5, 11),
    # Inner connections (step 2)
    (6, 8),
    (8, 10),
    (10, 6),
    (7, 9),
    (9, 11),
    (11, 7),
]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)
pos = {
    # Outer hexagon (radius 2)
    0: (0.00, 2.00),
    1: (1.73, 1.00),
    2: (1.73, -1.00),
    3: (0.00, -2.00),
    4: (-1.73, -1.00),
    5: (-1.73, 1.00),
    # Inner hexagon (radius 0.8)
    6: (0.00, 0.80),
    7: (0.69, 0.40),
    8: (0.69, -0.40),
    9: (0.00, -0.80),
    10: (-0.69, -0.40),
    11: (-0.69, 0.40),
}

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
# 数学モデルとインスタンスデータを `problem.eval()` でコンパイルします。この処理により、インスタンスデータが代入された問題の中間表現が得られます。

# %%
V = num_nodes
E = edges
data = {"V": V, "E": E}
instance = problem.eval(data)

# %% [markdown]
# ## コンパイル済みインスタンスからQRAO31ハミルトニアンとVQE回路への変換
#
# コンパイル済みInstanceからQRAO31エンコードされたハミルトニアンを生成します。これに使用するコンバータは `qamomile.optimization.qrao.qrao31` の `QRAC31Converter` です。
#
# QRAO31はQuantum Random Access Coding (QRAC) を用いて、異なるパウリ演算子（X, Y, Z）を使い最大3つの古典変数を1つの量子ビットにエンコードします。コンバータは内部的に以下の処理を行います:
#
# 1. **グラフ彩色**: 変数間の相互作用グラフに対してグラフ彩色を行い、隣接しない変数をグループ化
# 2. **パウリ演算子の割り当て**: 同一グループ内の各変数に、同じ量子ビット上の異なるパウリ演算子（Z, X, Y）を割り当て
# 3. **ハミルトニアンの緩和**: 元のイジングハミルトニアンをエンコードされたパウリ演算子で書き換え
#
# `get_cost_hamiltonian()` を使ってエンコードされたハミルトニアンを取得し、その基底状態を求めるVQEアンサッツを構築できます。

# %%
import qamomile.circuit as qmc
from qamomile.optimization.qrao.qrao31 import QRAC31Converter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create the QRAO31 converter
converter = QRAC31Converter(instance)

# %% [markdown]
# ### 量子ビット数の削減
#
# QRAOの最大の利点は量子ビット数の削減です。QAOA（各変数に1量子ビットが必要）と比較して、何量子ビット必要になるか確認してみましょう。

# %%
print(f"Number of classical variables: {num_nodes}")
print(f"Number of qubits (QAOA):      {num_nodes}")
print(f"Number of qubits (QRAO31):    {converter.num_qubits}")
print(
    f"Compression ratio:            {converter.num_qubits}/{num_nodes} = {converter.num_qubits / num_nodes:.0%}"
)

# %% [markdown]
# コンバータは `color_group`（各変数がどの量子ビットに割り当てられているか）と `pauli_encoding`（各変数がどのパウリ演算子で表現されているか）を通じて、変数と量子ビットの対応関係を確認できます。

# %%
print("Color groups (qubit -> variables):")
for qubit_idx, var_indices in converter.color_group.items():
    print(f"  Qubit {qubit_idx}: variables {var_indices}")

print("\nPauli encoding (variable -> (qubit, Pauli)):")
for var_idx, pauli_op in converter.pauli_encoding.items():
    print(f"  Variable {var_idx} -> {pauli_op}")

# %% [markdown]
# コストハミルトニアンを確認してみましょう。QAOAでは $n$ 量子ビット（各変数に1つ）上のパウリZ演算子のみを使用しますが、QRACエンコードされたハミルトニアンはより少数の量子ビット上で動作し、各量子ビットが複数の変数を担当するため、混合パウリ演算子（X, Y, Z）を使用します。

# %%
hamiltonian = converter.get_cost_hamiltonian()
hamiltonian

# %% [markdown]
# 次に、このハミルトニアンの基底状態を探索するためのハードウェア効率の良いVQEアンサッツを構築します。

# %%
from qamomile.circuit.algorithm.basic import cz_entangling_layer, ry_layer, rz_layer

depth = 2


@qmc.qkernel
def vqe(
    n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]
) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(depth - 1):
        q = ry_layer(q, theta, 2 * n * layer)
        q = rz_layer(q, theta, 2 * n * layer + n)
        q = cz_entangling_layer(q)

    q = ry_layer(q, theta, 2 * n * (depth - 1))
    q = rz_layer(q, theta, 2 * n * (depth - 1) + n)
    return qmc.expval(q, h)


executable = transpiler.transpile(
    vqe,
    bindings={
        "n": hamiltonian.num_qubits,
        "h": hamiltonian,
        "depth": depth,
    },
    parameters=["theta"],
)

# %% [markdown]
# 生成された量子回路を見てみましょう。

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw(output="mpl")

# %% [markdown]
# ## VQE最適化
#
# 変分最適化ループを設定します。scipyのCOBYLA最適化器を使用して、
# 最適なVQEパラメータを見つけます。

# %%
from scipy.optimize import minimize

# Calculate parameter count
n_qubits = hamiltonian.num_qubits
n_params = (
    2 * n_qubits * depth
)  # ry_layer + rz_layer each consume n_qubits params per layer

print(f"Number of qubits: {n_qubits}")
print(f"Depth: {depth}")
print(f"Number of parameters: {n_params}")

# List to store optimization history
energy_history = []


def objective_function(params, executable):
    """
    Objective function for VQE optimization.
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
# Run optimization
np.random.seed(901)

# Initial parameters
init_params = np.random.uniform(0, 2 * np.pi, size=n_params)

# Clear history
energy_history = []

print("Starting VQE optimization...")
print(f"Initial parameter count: {len(init_params)}")

# Optimize with COBYLA
result_opt = minimize(
    objective_function,
    init_params,
    args=(executable,),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimization complete")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化結果の可視化
#
# 最適化プロセスの収束を可視化してみましょう。
#
# > **注意:** Qamomileは内部的に最大化問題を最小化問題に変換するため、エネルギー値は負の値になります。

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## QRAO31のデコードプロセス
#
# 標準的なQAOAでは各変数が1つの量子ビットに対応するため、ビット列をサンプリングするだけで直接解を読み取ることができます。しかしQRAOでは複数の変数が1つの量子ビットを共有しているため、計算基底で量子ビットを測定するだけでは全変数の値を同時に復元することはできません。
#
# その代わり、QRAOでは2段階のデコードプロセスを使用します:
#
# 1. **期待値の測定**: 各変数 $x_i$ について、最適化された量子状態上でその変数に割り当てられたパウリ演算子 $P_i \in \{X, Y, Z\}$ の期待値 $\langle P_i \rangle$ を測定
# 2. **丸め処理**: 連続的な期待値を離散的なスピン値に変換する丸めスキームを適用
#
# 1つの量子ビット上のパウリ演算子は非可換（例: $[X, Y] \neq 0$）であるため、期待値はパウリ基底ごとに別々の測定回路から推定する必要があります。

# %%
# Create circuits to measure expectation values of Pauli operators for each variable
pauli_observables = converter.get_encoded_pauli_list()

print(f"Number of Pauli operators to measure: {len(pauli_observables)}")
print("Variable index to Pauli operator mapping:")
for idx, pauli_op in converter.pauli_encoding.items():
    print(f"  Variable {idx} -> {pauli_op}")


# %%
# Measure expectation values of each Pauli operator
@qmc.qkernel
def measure_pauli(
    n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]
) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(depth - 1):
        q = ry_layer(q, theta, 2 * n * layer)
        q = rz_layer(q, theta, 2 * n * layer + n)
        q = cz_entangling_layer(q)

    q = ry_layer(q, theta, 2 * n * (depth - 1))
    q = rz_layer(q, theta, 2 * n * (depth - 1) + n)
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
        parameters=["theta"],
    )

    job = executable_pauli.run(
        transpiler.executor(),
        bindings={
            "theta": optimal_params,
        },
    )
    expectation = job.result()
    expectations.append(expectation)

print("Pauli expectation values for each variable:")
for i, exp in enumerate(expectations):
    print(f"  Variable {i}: {exp:.4f}")

# %% [markdown]
# ## 解の復元: 丸め処理
#
# 期待値 $\langle P_i \rangle$ は $[-1, 1]$ の連続値ですが、離散的なスピン値 $s_i \in \{+1, -1\}$ が必要です。ここで **丸め処理** が登場します。
#
# `SignRounder` は最もシンプルな丸めルールを適用します:
#
# $$
# s_i = \begin{cases} +1 & \text{if } \langle P_i \rangle \geq 0 \\ -1 & \text{if } \langle P_i \rangle < 0 \end{cases}
# $$
#
# QRACエンコードされたハミルトニアンは元の問題の **緩和** であるため（エンコーディングは厳密ではない）、丸め処理によって緩和された最適値と復元された解の間にギャップが生じる可能性があります。これはQRAOアプローチに固有の性質であり、量子ビット数の大幅な削減と引き換えに厳密性を犠牲にしています。

# %%
from qamomile.optimization.qrao import SignRounder

rounder = SignRounder()
spins = rounder.round(expectations)

print("Recovered spin values (+1 or -1):")
for i, spin in enumerate(spins):
    print(f"  Variable {i}: ⟨P⟩ = {expectations[i]:+.4f}  →  spin = {spin:+d}")

# Convert spins to binary (spin=+1 -> binary=0, spin=-1 -> binary=1)
binary_solution = [(1 - s) // 2 for s in spins]

print("\nBinary solution (0 or 1):")
for i, bit in enumerate(binary_solution):
    print(f"  Variable {i}: {bit}")

# %% [markdown]
# ## 解の可視化
#
# QRAO31で見つけた最良解を元のグラフ上に可視化してみましょう。

# %%
# Convert solution to dictionary format
solution_dict = {(i,): float(bit) for i, bit in enumerate(binary_solution)}


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


# Calculate number of cut edges
def calculate_maxcut_value(graph, binary_solution):
    """Calculate the objective function value for the MaxCut problem"""
    cut_count = 0
    for u, v in graph.edges():
        if binary_solution[u] != binary_solution[v]:
            cut_count += 1
    return cut_count


cut_value = calculate_maxcut_value(G, binary_solution)

print("Solution found:")
print(f"  Binary string: {''.join(map(str, binary_solution))}")
print(f"  Number of cut edges: {cut_value}")

edge_colors, node_colors = get_edge_colors(G, solution_dict)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title(f"QRAO31 Solution (Cut Edges: {cut_value})")
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
# Qamomileのコンバータは `ommx.v1.Instance` を受け付けるため、古典ソルバーとの比較が容易に行えます。同じインスタンスをSCIPで厳密に解き、QRAOの解と比較してみましょう。

# %%
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

solution = OMMXPySCIPOptAdapter.solve(instance)

print(f"Exact optimal value (Max-Cut): {int(solution.objective)}")
print(f"QRAO solution value:           {cut_value}")

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、QamomileでQRAO31を使ってMax-Cut問題を解く方法を実演しました:
#
# 1. **問題の定式化**: JijModelingを使ってMax-Cutをイジング問題として定式化
# 2. **QRAO31エンコーディング**: `QRAC31Converter` がグラフ彩色とパウリ演算子の割り当てにより最大3変数を1量子ビットにエンコードし、量子ビット数を12（QAOA）からわずか数量子ビットに削減
# 3. **VQE最適化**: 削減されたハミルトニアン上でハードウェア効率の良いアンサッツを使い、最適パラメータを探索
# 4. **デコード**: 複数の変数が1つの量子ビットを共有しているため、パウリ期待値を測定し `SignRounder` で離散スピン値を復元
# 5. **解の分析**: 復元されたMax-Cut解を元のグラフ上に可視化
#
# QRAOの重要なポイントは、必要な量子ビット数を大幅に削減することで、近い将来の量子ハードウェアでより大きな組合せ最適化問題を解くことを可能にする点です。また、Qamomileは `ommx.v1.Instance` を使用しているため、古典ソルバーとのベンチマーク比較も容易に行えます。
