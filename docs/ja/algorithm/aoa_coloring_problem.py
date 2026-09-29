# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, variational]
# ---
#
# # グラフ彩色のための量子交互演算子アンザッツ
#
# このチュートリアルでは、Qamomileを用いてグラフ彩色問題を量子交互演算子アンザッツ(AOA)で解く方法を紹介します。ここで用いるAOAの定式化はHadfieldら{cite:p}`10.3390/a12020034`に従います。
#
# 以下の手順で進めていきます。
#
# 1. [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html)で問題を定式化し、具体的なデータを用いてインスタンスを作成する。
# 2. `AOAConverter`を使い、選択したミキサーと初期状態でAOA回路を構築する。
# 3. 古典オプティマイザで変分パラメータを最適化する。
# 4. 最適化された回路からサンプリングして結果をデコードし、すべてのサンプルが実行可能であることを確認する。

# %%
# 最新版のQamomileをpipでインストールしてください！
# # !pip install "qamomile[qiskit]"

# %%
import os

import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ommx.v1
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.aoa import aoa_state_dicke, xy_mixer
from qamomile.circuit.algorithm.qaoa import ising_cost
from qamomile.circuit.stdlib.state_preparation import prepare_dicke, scs_gate_2q
from qamomile.circuit.visualization import MatplotlibDrawer
from qamomile.optimization.aoa import AOAConverter
from qamomile.qiskit import QiskitTranspiler

# グラフの日本語ラベル用フォント（インストール済みの最初のフォントが使われます）。
JP_FONT = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Noto Sans CJK JP",
        "IPAexGothic",
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "DejaVu Sans",
    ],
    "axes.unicode_minus": False,
}

# %% [markdown]
# ## 背景
#
# AOAは、より一般的なミキサーハミルトニアンと初期状態を用いることで、QAOA回路を拡張したものです。これはグラフ彩色のような制約付き問題で役立ちます。
#
# 以下で用いるone-hot符号化では、グラフ彩色に$N \times K$個の二値変数が必要ですが、$2^{NK}$個のビット列のうち実行可能なものは$K^N$個しかありません（各ノードに1つの色）。以下で扱う5ノード・3色のインスタンスでは、$2^{15} = 32768$次元のヒルベルト空間の中に実行可能状態は243個しかなく、全体の$1\%$未満です。標準的なQAOAは空間全体に対する一様な重ね合わせから出発し、横磁場ミキサー($\sum X_i$)を用います。このミキサーは量子ビットを実行可能部分空間の内外へ自由に回転させるため、サンプリングされるビット列のほとんどはone-hot制約に違反して破棄されることになります。
#
# AOAはこの両端の問題に対処します。実行可能部分空間の内部から出発し、実行可能な状態の間でのみ振幅を動かすXYミキサーを用いることで、サンプルは構成上すべて実行可能になります。

# %% [markdown]
# ## 問題設定
#
# 無向グラフ$G = (V, E)$と使用できる色の数$K$が与えられたとき、各頂点に1つの色を割り当て、同じ色の頂点同士を結ぶ辺ができるだけ少なくなるようにすることが目標です。このような辺を**衝突**と呼びます。
#
# **目的関数:**
#
# $$
# \min \sum_{(u, v) \in E} \sum_{i=0}^{K-1} x_{u, i} x_{v, i}
# $$
#
# **制約:**
#
# $$
# \sum_{i=0}^{K-1} x_{u, i}=1, \forall u \in\{0, \ldots, N-1\}
# $$
#
# ここで$x_{u, i} \in\{0,1\}$は、頂点$u$に色$i$が使われている場合に1、そうでなければ0となります。目的関数は衝突の数を表します。


# %%
@jm.Problem.define("Graph Coloring", sense=jm.ProblemSense.MINIMIZE)
def graph_coloring_decorated(problem: jm.DecoratedProblem):
    N = problem.Length()
    K = problem.Natural()

    E = problem.Graph()

    x = problem.BinaryVar(
        shape=(N, K),
        description="$x_{i,k}$ is 1 if node $i$ is colored with color $k$, 0 otherwise",
    )

    problem += jm.sum(x[u, i] * x[v, i] for (u, v) in E for i in K)

    problem += problem.Constraint(
        "ColoringConstraint",
        (jm.sum(x[u, i] for i in K) == 1 for u in N),
        description="Each node must be colored with exactly one color",
    )


graph_coloring_decorated

# %% [markdown]
# ### 制約をハミング重みの条件として読み解く
#
# 制約$\sum_{i=0}^{K-1} x_{u,i} = 1$は、ノード$u$に付随する$K$個の二値変数のうち、ちょうど1つが$1$になることを意味します。これらをビット列として並べると、各ノードは$K$個の連続する量子ビットからなるブロックを占有し、実行可能性は**各ブロックのハミング重みがちょうど1であること**に対応します。
#
# 5ノード・3色のインスタンスでは、量子ビットの配置は次のようになります。
#
# ```
#   [q0 q1 q2] [q3 q4 q5] [q6 q7 q8] [q9 q10 q11] [q12 q13 q14]
#    node 0     node 1     node 2     node 3        node 4
# ```
#
# 実行可能な状態とは、括弧で囲まれた各ブロック内に`1`が1つだけある状態のことです。

# %% [markdown]
# ### グラフのインスタンス
#
# このチュートリアルでは、辺が6本の5ノード固定グラフを用います。

# %%
num_nodes = 5
num_colors = 3
edge_list = [(0, 2), (0, 3), (0, 4), (1, 3), (2, 4), (3, 4)]

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)
assert G.number_of_nodes() == num_nodes
assert G.number_of_edges() == len(edge_list)

pos = nx.spring_layout(G, seed=1)
with plt.rc_context(JP_FONT):
    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="white",
        node_size=700,
        edgecolors="black",
    )
    plt.title(f"グラフ：ノード{G.number_of_nodes()}個、辺{G.number_of_edges()}本")
    plt.show()

# %% [markdown]
# JijModelingの問題に具体的なデータを与えて評価し、$N \times K = 15$個の二値変数とノードごとに1つの制約を持つOMMXインスタンスを得ます。

# %%
instance_data = {"N": num_nodes, "E": edge_list, "K": num_colors}
instance = graph_coloring_decorated.eval(instance_data)

assert len(instance.decision_variables) == num_nodes * num_colors
assert len(instance.constraints) == num_nodes

# %% [markdown]
# ## アルゴリズム
#
# QAOAと同様に、AOAは初期状態$\lvert \psi_0 \rangle$から出発し、コスト層とミキサー層を$p$回交互に適用します。
#
# $$
# \lvert \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle
# = \prod_{l=1}^{p} U_M(\beta_l)\, e^{-i \gamma_l H_C} \lvert \psi_0 \rangle
# $$
#
# ここで$H_C$はコストハミルトニアン、$U_M$はミキサーです。AOAとQAOAの違いは、$\lvert \psi_0 \rangle$と$U_M$の選び方にあります。
#
# **XYミキサー.** ミキサーは2量子ビット間のXY相互作用から構成されます。
#
# $$
# H_{ij}^{XY}=\frac{1}{2}(X_iX_j+Y_iY_j),
# \qquad
# U_{ij}^{XY}(\beta)=e^{-i\beta H_{ij}^{XY}}.
# $$
#
# 2量子ビット上で、$H_{ij}^{XY}$は$\lvert 01 \rangle \leftrightarrow \lvert 10 \rangle$を入れ替え、$\lvert 00 \rangle$と$\lvert 11 \rangle$を0に写します。したがって$U_{ij}^{XY}(\beta)$は、1の数を変えずに$\lvert 01 \rangle$と$\lvert 10 \rangle$の振幅を混合します。この相互作用を同じノードの色ブロック内の量子ビットのペアにのみ適用すれば、one-hot制約は保たれます。各ブロックの1つの`1`は別の色へ移動できますが、消えたり増えたりすることはありません。
#
# **初期状態.** 回路は実行可能部分空間の内部から出発する必要もあります。自然な選択は各ブロックにおけるDicke状態、すなわち指定されたハミング重みを持つすべてのビット列の等しい重ね合わせです。ハミング重みが1でブロックサイズが$K$の場合、各ブロックは次の状態に準備されます。
#
# $$
# \lvert D^K_1 \rangle = \frac{1}{\sqrt{K}}
# \left(\lvert 10\ldots 0\rangle + \lvert 01\ldots 0\rangle + \cdots + \lvert 0\ldots 01\rangle\right)
# $$
#
# このため、初期状態全体はすべての実行可能な彩色に対する一様な重ね合わせになります。
#
# **コスト層.** コスト層$e^{-i \gamma H_C}$はQAOAと同じです。オプティマイザは$\boldsymbol{\gamma}$と$\boldsymbol{\beta}$を調整し、衝突の少ない彩色に振幅を集中させます。

# %% [markdown]
# ## 実装
#
# ### AOAConverterのセットアップ
#
# `AOAConverter`はOMMXのインスタンスを受け取り、内部でQUBO形式に変換した上で、QAOAのコンバーターと同じようにコストハミルトニアンを構築します（コンバーターのワークフローについては[QAOAによるグラフ分割](qaoa_graph_partition)を参照してください）。デコードされたサンプルのエネルギーにはペナルティ項が含まれるため、真の目的関数値はデコード時に別途評価されます。

# %%
converter = AOAConverter(instance)
converter.spin_model = converter.spin_model.normalize_by_abs_max()
hamiltonian = converter.get_cost_hamiltonian()
print(hamiltonian)

assert converter.spin_model.num_bits == num_nodes * num_colors

# %% [markdown]
# ### 初期状態とミキサーの選択
#
# `converter.transpile()`はQAOAの場合と同じように動作し、さらに初期状態とXYミキサーを選択するオプションを持ちます。
#
# 初期状態としては、以下の選択肢があります。
#
# - `single_basis_state`：各ブロック内で正しいハミング重みを持つ単一の計算基底状態ですコンバーターは各ブロックの末尾の量子ビットを$|1\rangle$に設定します（例えば量子ビットの順序$q_0 q_1 q_2$で$|001\rangle|001\rangle\ldots$となり、すべてのノードが最後の色から始まることを意味します）。1つの有効な彩色から出発し、準備に必要なゲートも比較的少なくて済みます。他の実行可能状態へ振幅を広げる役割はミキサーが担います。
# - `dicke`：上で説明した、各ブロックにおけるDicke状態です。準備に追加のゲート（ブロックあたり$O(K)$個）が必要ですが、すべての実行可能な彩色に対して偏りのない出発点をオプティマイザに与えます。
# - `uniform`：すべての量子ビットにアダマールゲートを適用した、標準的なQAOAの初期状態です。**これは実行不可能な状態にも振幅を割り当てる**ため、AOAの実行可能性の保証は失われます。主に、異なる初期状態が結果にどう影響するかを比較する用途に役立ちます。
#
# `hamming_weight`パラメータは、各ブロックでの目標ハミング重みを設定します。one-hot符号化を用いたグラフ彩色問題では常に`1`です。他の問題（例えば、ちょうど$k$個のアイテムを選択する必要がある基数制約付き最適化問題）では、異なる値になります。
#
# ミキサーとしては、以下から選択できます。
#
# - `ring`：各量子ビットを、ブロック内で環状に並べたときの2つの隣接量子ビットとのみ接続します。奇数番目のペア、偶数番目のペア、最後に端をつなぐペアの順に適用されるため、各層のコストはブロックあたり$O(K)$個の2量子ビットゲートです。回路は浅くなりますが、振幅がブロック全体に広がるにはより多くの層が必要です。
# - `fully-connected`：ブロック内のすべての量子ビットのペアを接続します。$\binom{K}{2}$個のペアは互いに重ならないペアからなるラウンドに分割されるため、各層のコストはブロックあたり$O(K^2)$個の2量子ビットゲートです。回路は深くなりますが、振幅は1層でブロック全体に混合されます。
#
# 目安として、$K$が大きく回路の深さがボトルネックになる場合は`ring`を、$K$が小さい（例えば$\leq 4$）場合は`fully-connected`を選ぶとよいでしょう。
#
# 最後に、`block_size`はレジスタをどのようにブロックへ分割するかをコンバーターに指定します。Dicke状態は各ブロック内で準備され、ミキサーは同じブロック内の量子ビット同士のみを結合します。グラフ彩色では、各ノードがハミング重みを$1$に保つべき$K$個の量子ビットを持つため、`block_size = num_colors`と設定します。

# %%
transpiler = QiskitTranspiler()
p = 5  # AOAの層数

executable_aoa_dicke = converter.transpile(
    transpiler,
    p=p,
    initial_state="dicke",
    hamming_weight=1,
    mixer="fully-connected",
    block_size=num_colors,
)

assert executable_aoa_dicke.quantum_circuit.num_qubits == num_nodes * num_colors

# %% [markdown]
# ### AOA回路の可視化
#
# `AOAConverter.transpile()`は内部で以下のサンプリング用qkernelを構築し、`block_size`から計算したDicke状態のスケジュールとミキサーのペアとともにトランスパイラに渡します。コンバーターはこの2つの入力を公開しているため、ここで計算できます。これらは描画には便利ですが、通常のワークフローでは必要ありません。

# %%
initial_ones, schedule_dicke = converter.compute_dicke_composition_schedule(
    hamming_weight=1, block_size=num_colors
)
resolved_pair = converter.resolve_pair_indices(
    mixer="fully-connected", pair_indices=None, block_size=num_colors
)

# ブロックごとに|1>の量子ビットが1つ、ミキサーのペアがK(K-1)/2個あります。
assert len(initial_ones) == num_nodes
assert len(resolved_pair) == num_nodes * num_colors * (num_colors - 1) // 2
# ミキサーの各ペアは同じブロック内の2つの量子ビットを結合します。
assert all(i // num_colors == j // num_colors for i, j in resolved_pair)

# %% [markdown]
# コンバーターのサンプリング用qkernelをそのまま書き直し、`Transpiler.to_block`でIRブロックに変換し、`Transpiler.inline`で内部のqkernel呼び出しを展開してから、`MatplotlibDrawer`で描画します。見やすさのために`p=1`を使用します。
#
# Dicke状態の準備は、`if True:`というラベルの付いたボックスの列として表示されます。`prepare_dicke`はスケジュールの各ステップで2量子ビットゲートと3量子ビットゲートのどちらを使うかを選びますが、この選択は回路のコンパイル時に行われます。ハミング重みが1の場合はすべてのステップで2量子ビットゲート`scs_gate_2q`が使われ、各ボックスにはその最初のゲートが表示されます。


# %%
@qmc.qkernel
def aoa_sampling_dicke(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    n: qmc.UInt,
    pair_indices_mixer: qmc.Matrix[qmc.UInt],
    initial_ones: qmc.Vector[qmc.UInt],
    schedule_dicke: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = aoa_state_dicke(
        p=p,
        quad=quad,
        linear=linear,
        n=n,
        gammas=gammas,
        betas=betas,
        pair_indices_mixer=pair_indices_mixer,
        initial_ones=initial_ones,
        schedule_dicke=schedule_dicke,
    )
    return qmc.measure(q)


block = transpiler.to_block(
    aoa_sampling_dicke,
    bindings={
        "linear": converter.spin_model.linear,
        "quad": converter.spin_model.quad,
        "n": converter.spin_model.num_bits,
        "p": 1,
        "pair_indices_mixer": resolved_pair,
        "initial_ones": initial_ones,
        "schedule_dicke": schedule_dicke,
    },
    parameters=["gammas", "betas"],
)
block = transpiler.inline(block)
assert block.operations

fig = MatplotlibDrawer(block).draw(fold_loops=False, fold_ifs=True)
assert fig.get_axes()
fig

# %% [markdown]
# ### 構成要素の確認
#
# `aoa_state_dicke`の内部では、コンバーターが複数のqkernelを呼び出しています。
#
# - `prepare_dicke(n, initial_ones, schedule)`：まず$X$ゲートで、各ブロックに指定されたハミング重みを持つ基底状態を作成します。その後、スケジュールの各ステップでsplit-and-cyclic-shift（SCS）ゲートを1つ適用し、各ブロック内にDicke状態を構築します。
# - `ising_cost(quad, linear, q, gamma)`：QAOAと同じコスト層です。$R_Z$と$R_{ZZ}$の回転ゲートを使用します。
# - `xy_mixer(q, betas[layer], pair_indices_mixer)`：ミキサー層です。`pair_indices_mixer`に列挙された量子ビットの各ペアに$U_{ij}^{XY}$を適用します。
#
# `aoa_layers(p, ...)`は`ising_cost`と`xy_mixer`を交互に並べ、`p`回繰り返したものです。

# %%
dicke_block = transpiler.inline(
    transpiler.to_block(
        prepare_dicke,
        bindings={
            "n": converter.spin_model.num_bits,
            "initial_ones": initial_ones,
            "schedule": schedule_dicke,
        },
    )
)
assert dicke_block.operations

fig = MatplotlibDrawer(dicke_block).draw(fold_loops=False, fold_ifs=True)
assert fig.get_axes()
fig

# %% [markdown]
# 各ボックスは、ブロック内の2つの量子ビットに`scs_gate_2q`を適用します。このゲートはCNOTと$R_Y$ゲートから構成されます。

# %%
fig = scs_gate_2q.draw(q=2, t=0, c=1)
assert fig.get_axes()
fig

# %%
fig = ising_cost.draw(
    q=converter.spin_model.num_bits,
    quad=converter.spin_model.quad,
    linear=converter.spin_model.linear,
    fold_loops=False,
)
assert fig.get_axes()
fig

# %%
fig = xy_mixer.draw(
    q=converter.spin_model.num_bits,
    pair_indices_mixer=resolved_pair,
    inline=True,
    fold_loops=False,
    expand_composite=True,
    inline_depth=None,
)
fig.set_size_inches(100, 8)
assert fig.get_axes()
fig

# %% [markdown]
# ## 結果
#
# ### AOAのパラメータ最適化
#
# 古典オプティマイザの各イテレーションでコストを評価するために`executable.sample()`を使用します。オプティマイザは、サンプリングされたビット列の平均エネルギーを最小化するように、異なる`gammas`と`betas`を探索します。

# %%
executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=901, max_parallel_threads=1)
)
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 25 if docs_test_mode else 1000
final_shots = 64 if docs_test_mode else 1000

rng = np.random.default_rng(900)
initial_params = rng.uniform(0, np.pi, 2 * p)
assert initial_params.shape == (2 * p,)

cost_history = []


def cost_fn(params):
    gammas = list(params[:p])
    betas = list(params[p:])
    job = executable_aoa_dicke.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()
    decoded = converter.decode_to_binary_sampleset(result)
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
assert len(cost_history) == res.nfev
assert len(res.x) == 2 * p

# %%
assert np.all(np.isfinite(cost_history))

with plt.rc_context(JP_FONT):
    plt.figure(figsize=(8, 4))
    plt.plot(cost_history, color="#2696EB")
    plt.xlabel("反復回数")
    plt.ylabel("コスト（平均エネルギー）")
    plt.title("AOAの最適化の推移")
    plt.show()

# %% [markdown]
# ### 最適化されたパラメータでのサンプリング
#
# 最適化されたパラメータを使って回路をサンプリングし、候補解をビット列として収集してOMMXの`SampleSet`にデコードします。

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])

sample_result = executable_aoa_dicke.sample(
    executor,
    shots=final_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

sample_set = converter.decode(sample_result)
assert isinstance(sample_set, ommx.v1.SampleSet)

# %% [markdown]
# ### 実行可能性のチェック
#
# 問題に合わせたXYミキサーは、探索の間ずっと量子状態を実行可能部分空間の中に保つはずです。したがって標準的なQAOAとは異なり、このAOA回路からサンプリングされる候補はすべてハミング重み1の制約を満たすはずです。

# %%
summary = sample_set.summary
total_feasible = int(summary["feasible"].sum())
total_samples = len(summary)

print(
    f"Feasible samples: {total_feasible} / {total_samples} "
    f"({100 * total_feasible / total_samples:.1f}%)"
)
assert total_samples == final_shots
assert total_feasible == total_samples, "AOA must only produce feasible colorings"

# %% [markdown]
# ### 最良の彩色パターン
#
# `SampleSet.best_feasible`は、目的関数値が最も良い（ここでは最も小さい）サンプル、すなわち衝突が最も少ないサンプルを返します。

# %%
best = sample_set.best_feasible
df = best.decision_variables_df
x_rows = df[df["name"] == "x"]

best_coloring = {}
for _, row in x_rows.iterrows():
    node, color = row["subscripts"]
    if row["value"] > 0.5:
        best_coloring[int(node)] = int(color)

num_conflicts = sum(best_coloring[u] == best_coloring[v] for u, v in edge_list)
print("最良の彩色パターン:", best_coloring)
print("衝突の数（同じ色が割り当てられた隣接頂点ペアの数）:", num_conflicts)

assert sorted(best_coloring) == list(range(num_nodes))
assert num_conflicts == round(best.objective)

# %% [markdown]
# ### 目的関数値の分布
#
# すべてのサンプルについて、目的関数値すなわち衝突の数の分布をプロットします。

# %%
obj_counts = summary["objective"].value_counts().sort_index()
assert obj_counts.sum() == total_samples

with plt.rc_context(JP_FONT):
    plt.figure(figsize=(8, 4))
    plt.bar([str(int(o)) for o in obj_counts.index], obj_counts.values, color="#2696EB")
    plt.xlabel("衝突の数（目的関数値）")
    plt.ylabel("頻度")
    plt.title("衝突の数の分布")
    plt.show()

# %% [markdown]
# ### 最良の彩色パターンの可視化
#
# AOAで見つかった最良の彩色パターンに従って、グラフのノードを色付けします。

# %%
palette = ["#FF6B6B", "#4ECDC4", "#1A535C"]
color_map = [palette[best_coloring[u]] for u in range(num_nodes)]
assert len(color_map) == num_nodes

with plt.rc_context(JP_FONT):
    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=700,
        edgecolors="black",
    )
    plt.title(f"最良の彩色パターン：衝突の数{num_conflicts}")
    plt.show()

# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを行いました。
#
# - Qamomileの`AOAConverter`を使い、量子交互演算子アンザッツでグラフ彩色問題を解きました。
# - 有効な彩色の重ね合わせから出発し、有効な彩色の間だけを移動するミキサーを用いました。
# - サンプリングされたすべての解で、各ノードにちょうど1つの色が割り当てられていることを確認しました。
