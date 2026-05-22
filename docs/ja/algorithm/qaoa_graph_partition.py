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
# tags: [algorithm, optimization, variational]
# ---
#
# # QAOA によるグラフ分割
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
# Colabで開いている場合は、下のタブで選んだTranspilerに合う行を1つ選び、行頭のコメントを外して実行してください:
# # !pip install qamomile                  # Qiskit（デフォルト）
# # !pip install "qamomile[quri_parts]"    # QURI Parts
# # !pip install "qamomile[cudaq-cu12]"    # CUDA-Q (CUDA 12.x toolchain。CUDA 13.xなら`qamomile[cudaq-cu13]`)。Linux / macOS-arm64 / WSL2のみ。

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
    problem += problem.Constraint("Equal Partition", x.sum() == V / 2)


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

# %% [markdown]
# この記事はデフォルトでQiskitを使います。Qamomileは同じ`@qkernel`を複数の量子SDKへトランスパイルできるので、下のimportを差し替えるだけで他のSDKでも同じ流れで進められます。記事本体のコードはどのSDKを選んでも同一です。Colabの場合は上のpipセルで対応する行のコメントを先に外しておいてください。
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsTranspiler
#
# transpiler = QuriPartsTranspiler()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# CUDA 12.x環境では`qamomile[cudaq-cu12]`、CUDA 13.x環境では`qamomile[cudaq-cu13]`を使ってください（インストール済みのCUDA Toolkitに合わせて選択）。CUDA-QはLinux、macOS arm64、Windows（WSL2経由）のみ対応です。
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# ```
# :::
# ::::

# %%
# Transpiler — この記事はデフォルトでQiskitを使います。
# 上のタブでQURI PartsまたはCUDA-Qを選んだ場合は、そのタブに書かれた
# 2行（importとtranspiler = ...）を以下の2行と入れ替えてください。
# あわせて、上のpipセルで対応する行のコメントも外しておくこと。
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
p = 5  # QAOA の層数
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# ## QAOA 回路の可視化
#
# `QAOAConverter._transpile_quadratic()` は内部で次のサンプリング qkernel を組み立てて transpile します。チュートリアルでは可視化のため、その qkernel を**そのまま**ここに再掲します。`Transpiler.to_block` で IR ブロックに落とし、`Transpiler.inline` でサブ qkernel を展開してから `MatplotlibDrawer.draw` に渡すことで、コンバーターが内部で扱うのと同じ Block 構造を描画できます(レイヤー構造を読みやすくするため `p=2` に縮小)。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.circuit.visualization import MatplotlibDrawer


@qmc.qkernel
def qaoa_sampling(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
    n: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qaoa_state(p=p, quad=quad, linear=linear, n=n, gammas=gammas, betas=betas)
    return qmc.measure(q)


block = transpiler.to_block(
    qaoa_sampling,
    bindings={
        "linear": converter.spin_model.linear,
        "quad": converter.spin_model.quad,
        "n": converter.spin_model.num_bits,
        "p": 2,
    },
    parameters=["gammas", "betas"],
)
block = transpiler.inline(block)
# `fold_loops=False` で全ループ(`for layer`/`for (i,j),Jij in quad`/`for i in range(n)`)を
# アンロールして、各ゲートを展開した形で描画する。`linear` は空辞書 `{}` のため、
# `for i, hi in linear` は 0 イテレーションとして消える(ボックスは現れない)。
# 結果は横長になるが、ドキュメントビルド時にライトボックス用 JS が注入されるので、
# クリックでモーダル拡大表示できる。
MatplotlibDrawer(block).draw(fold_loops=False)

# %% [markdown]
# ### 内部の構成要素を見る
#
# `qaoa_sampling` の中では `qaoa_state` を呼んでいますが、`qaoa_state` 自体は次の3つの組み合わせで構成されています:
#
# - `superposition_vector(n)` — 全ビットに `H` を作用させて初期状態 $|+\rangle^{\otimes n}$ を準備する。
# - `ising_cost(quad, linear, q, gamma)` — 二次項 `quad` の各エントリに対して `RZZ`、線形項 `linear` の各エントリに対して `RZ` を作用させるコスト層。今回の問題では線形項がない(`linear={}`)ので `RZZ` だけが現れる。
# - `x_mixer(q, beta)` — 全ビットに `RX(2\beta)` を作用させるミキサー層。
#
# `qaoa_layers(p, ...)` はこの `ising_cost` と `x_mixer` を `p` 回交互に作用させているだけです。各部品を個別に描画して中身を確認します。

# %%
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import ising_cost, x_mixer

superposition_vector.draw(n=converter.spin_model.num_bits, fold_loops=False)

# %%
ising_cost.draw(
    q=converter.spin_model.num_bits,
    quad=converter.spin_model.quad,
    linear=converter.spin_model.linear,
    fold_loops=False,
)

# %%
x_mixer.draw(q=converter.spin_model.num_bits, fold_loops=False)

# %% [markdown]
# ## QAOA パラメータの最適化
#
# `executable.sample()` を使って各イテレーションでコストを評価します。オプティマイザはサンプリングされたビット列の平均エネルギーを最小化する `gammas` と `betas` を探索します。
#
# executor配下のシミュレータをシードしておくことで、notebookを再実行してもCOBYLAの軌道と最終的なサンプリング分布が再現されます。シードがないと、ショットごとに異なる乱数が引かれるためCOBYLAはノイズのあるコスト面を辿り、実行のたびに（等価だが）異なる局所最適解に収束します。シードの方法は選んだSDKによって違うので、上のタブで別のSDKを選んだ場合は下のタブブロックから対応するスニペットをコピペしてください。
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qiskit_aer import AerSimulator
#
# executor = transpiler.executor(
#     backend=AerSimulator(seed_simulator=901, max_parallel_threads=1)
# )
# ```
#
# `seed_simulator=901`でショットごとのドローを再現可能にし、`max_parallel_threads=1`でAerの並列サンプリングがスレッド間で乱数ドローを混ぜることを防ぎます。
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# # qulacs（QURI Partsのデフォルトシミュレータ）はサンプリングのシード指定APIを公開していないため、
# # ショットごとの結果はランごとに変動します。最適化は十分なショット数なら同じ近傍に収束しますが、
# # コスト軌道はランごとに完全一致はしません。
# executor = transpiler.executor()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# ```python
# import cudaq
#
# # cudaqの乱数はプロセスグローバルです。set_random_seedはこの後の
# # cudaq.sample/observe呼び出しすべてに影響します。ノートブック内の再現性には十分ですが、
# # 同一プロセスで並行カーネルが走る場合は安全ではありません。
# cudaq.set_random_seed(901)
# executor = transpiler.executor()
# ```
# :::
# ::::

# %%
import os

import numpy as np
from scipy.optimize import minimize

# %%
# Executor — この記事はデフォルトで Qiskit の AerSimulator (シード固定) を使います。
# 上のタブで別のSDKを選んだ場合は、対応するタブのスニペットで以下を上書きしてください
# （あわせて記事冒頭のpipセルで対応する行のコメントも外しておくこと）。
from qiskit_aer import AerSimulator

executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=901, max_parallel_threads=1)
)

# %%
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
    # decode_to_binary_sampleset は QUBO ドメインの BinarySampleSet を返す。
    # その `energy` はペナルティを含む目的関数値で、COBYLA が実行不可能解の
    # コストを認識するために必要。多態的な decode() が返す ommx.v1.SampleSet
    # の `objective` はペナルティを含まない真の目的関数値であり、
    # 最適化器を実行不可能な全 0/全 1 ビット列に収束させてしまう。
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

# OMMXベースのコンバーターで`decode()`を呼ぶと、元の(ペナルティを含まない)
# インスタンスで評価された`ommx.v1.SampleSet`が返る。実行可能性、真の目的関数値、
# 制約ごとの違反量がOMMXのAPIから直接得られるため、自前の実行可能性判定や
# 目的関数値計算ヘルパーは不要。
sample_set = converter.decode(sample_result)

# %% [markdown]
# ## 結果の分析
#
# ### 実行可能性のチェック
#
# QAOAのサンプルは**候補解**であり、元の制約を満たすとは限りません。制約$\sum x_u = |V|/2$はQUBOのペナルティとして組み込まれているため、制約を満たさないビット列も出力に含まれる可能性があります。
#
# `SampleSet.summary`はショットごとに1行を持つDataFrameで、`feasible`列には*元の*制約に対する実行可能性がすでに格納されています。実行可能なショット数はそこから直接取得できます。

# %%
summary = sample_set.summary
total_feasible = int(summary["feasible"].sum())
total_samples = len(summary)

print(
    f"Feasible samples: {total_feasible} / {total_samples} "
    f"({100 * total_feasible / total_samples:.1f}%)"
)

# %% [markdown]
# ### 最良の実行可能解
#
# `SampleSet.best_feasible`は、実行可能解のうち目的関数値が最良(ここでは最小)のものを返します。コンバーターをOMMXの`Instance`から構築しているため、報告される目的関数値は*元の*ペナルティを含まないもの、すなわちカットエッジ数そのものです。決定変数の値は`decision_variables_df`に元の変数IDをキーとして格納されています。

# %%
if total_feasible > 0:
    best = sample_set.best_feasible
    best_obj = int(round(best.objective))
    best_sample = {
        i: int(round(best.decision_variables_df.loc[i, "value"]))
        for i in range(num_nodes)
    }
    print(f"Best feasible solution: {best_sample}")
    print(f"Cut edges:             {best_obj}")
else:
    print("No feasible solution found. Try increasing p or maxiter.")
    best_sample = None
    best_obj = None

# %% [markdown]
# ### 目的関数値の分布
#
# 実行可能なサンプルのみについて、真の目的関数値(カットエッジ数)の分布を表示します。`summary`にはショットごとに元の目的関数値が格納されているため、実行可能なスライスに対して`value_counts()`を呼ぶだけで分布が得られます。

# %%
if total_feasible > 0:
    feasible_objectives = (
        summary.loc[summary["feasible"], "objective"].round().astype(int)
    )
    obj_counts = feasible_objectives.value_counts().sort_index()

    plt.figure(figsize=(8, 4))
    plt.bar([str(o) for o in obj_counts.index], obj_counts.values, color="#2696EB")
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
