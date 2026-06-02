# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # 組合せ多項式二値最適化のためのGrover適応探索
#
# このチュートリアルでは、QamomileのGrover適応探索（GAS）アルゴリズムを使って**ポートフォリオ問題**を解く方法を示します。
#
# このチュートリアルでは、以下の論文に基づいた制約付き多項式二値最適化のためのGrover適応探索法の使い方を解説します。
# > Gilliam, Austin, et al. Quantum, 5, p428 (2021) https://doi.org/10.22331/q-2021-04-08-428
#
# 1. [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html) で問題を定式化する。
# 2. 具体的なデータでインスタンスを作成する。
# 3. `GASConverter` を使ってGrover適応探索回路を構築する。
# 4. 回路を古典的なレイヤーの内側で実行し、繰り返し回数を制御しながら回路をサンプリングする。停止条件を満たすまで最良解を保持し続ける。
#

# %% [markdown]
# ## ポートフォリオ問題の定式化
#
# $n$ 個の資産（株式、債券など）が与えられています。それぞれについて、購入するかしないかを決定します（二値選択）。
# リスクを最小化しながらリターンを最大化したいと考えます。ここでのトレードオフは以下の通りです：
#
# - $\mu$ は各資産の期待収益率を表します
# - $\Sigma$ は資産間の共動性を表します（相関のある資産はリスクを増幅させます）
# - $q$ はリスクとリターンのどちらをどの程度重視するかを制御します
#
# 過度な相関リスクを取らずに、できるだけ高いリターンが得られる最適な資産の部分集合を選択する必要があります。
#
# \begin{equation}
# \min_{x\in \lbrace 0 , 1 \rbrace^n} \big( q x^T \Sigma x - \mu^T x \big)
# \end{equation}
#

# %% [markdown]
# ## JijModelingによる問題定義
#

# %%
import jijmodeling as jm
import numpy as np

@jm.Problem.define("Portfolio Optimization (Unconstrained)")
def portfolio_problem(problem: jm.DecoratedProblem):
    n = problem.Length(description="Number of assets")
    q = problem.Float("q", description="Risk aversion factor")
    μ = problem.Float("μ", shape=(n,), description="Expected returns vector")
    Σ = problem.Float("Σ", shape=(n, n), description="Covariance matrix")

    x = problem.BinaryVar("x", shape=(n,), description="1 if asset i is selected")

    problem += (
        q * jm.sum(Σ[i, j] * x[i] * x[j] for i in n for j in n)
        - jm.sum(μ[i] * x[i] for i in n)
    )
portfolio_problem

# %% [markdown]
# ## 問題インスタンス
#
# 以下のインスタンスでは9つの資産を扱います：
#
# | 名前      | 期待収益率 | 分散 |
# |-----------|------------|------|
# | 資産 1    | 22         | 12   |
# | 資産 2    | 4          | 15   |
# | 資産 3    | 19         | 10   |
# | 資産 4    | 3          | 18   |
# | 資産 5    | 23         | 14   |
# | 資産 6    | 2          | 20   |
# | 資産 7    | 5          | 11   |
# | 資産 8    | 25         | 16   |
# | 資産 9    | 3          | 13   |
#

# %%
q = 1
μ = np.array([22, 4, 19, 3, 23, 2, 5, 25, 3], dtype=int)
Σ = np.array([
    [12, -3,  4,  0, -2,  3,  0,  2, -1],
    [-3, 15,  0,  5,  1, -4,  2,  0,  3],
    [ 4,  0, 10, -6,  3,  2, -1,  4,  0],
    [ 0,  5, -6, 18, -4,  0,  3, -2,  5],
    [-2,  1,  3, -4, 14,  2, -3,  0,  2],
    [ 3, -4,  2,  0,  2, 20,  4, -3,  1],
    [ 0,  2, -1,  3, -3,  4, 11,  2, -2],
    [ 2,  0,  4, -2,  0, -3,  2, 16, -4],
    [-1,  3,  0,  5,  2,  1, -2, -4, 13],
], dtype=int)

# %% [markdown]
# ## インスタンスの作成
#
# JijModelingの問題を具体的なデータで評価します。
#

# %%
instance_data = {
    "n": 9,
    "q": int(q),
    "μ": μ.tolist(),
    "Σ": Σ.tolist(),
}
instance = portfolio_problem.eval(instance_data)

# %% [markdown]
# ## GASコンバータのセットアップ
#
# `GASConverter` はOMMXインスタンスを受け取り、内部で問題をQUBO（二次制約なし二値最適化）形式に変換します。
#
# コンバータには `transpile` メソッドも備わっており、指定したトランスパイラを使って適応探索で使用するGrover回路を構築します。
#

# %%
from qamomile.optimization.gas import GASConverter
from qamomile.qiskit import QiskitTranspiler

converter = GASConverter(instance)
transpiler = QiskitTranspiler()

# %% [markdown]
# ## Grover回路の可視化
#
# `GASConverter.transpile()` は内部で以下のサンプリング用qkernelを構築し、トランスパイラに渡します。
# 可視化のためには、`Transpiler.to_block` と `MatplotlibDrawer.draw` を、transpileメソッド内で使用されるqkernelに対して呼び出すことができます。
#

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.gas import grover_algorithm
from qamomile.circuit.visualization import MatplotlibDrawer

@qmc.qkernel
def sampling_grover_algorithm(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt = 1
) -> qmc.Tuple[qmc.UInt, qmc.UInt]:
    q_output,q_input = grover_algorithm(
        n,
        m,
        y,
        linear,
        quad,
        iters
    )
    return qmc.measure(q_output), qmc.measure(q_input)


block = transpiler.to_block(
    sampling_grover_algorithm,
    bindings={
        "n" : converter.binary_model.num_bits,      # 入力量子ビット数 = 二値変数の数
        "m" : 9, # 出力量子ビット数（コンバータにより計算）
        "y" : 0,  # オラクル閾値：f(x) < y となる状態を標識（可視化用の任意値）
        "linear" : converter.binary_model.linear,   # QUBOの線形係数
        "quad" : converter.binary_model.quad,       # QUBOの二次係数
        "iters" : 1  # Groverの繰り返し回数（調整可能）
    }
)
block = transpiler.inline(block)
MatplotlibDrawer(block).draw(fold_loops=False)

# %% [markdown]
# ### 構成要素の確認
#
# 最適化に使用するGroverのアイデアは、基準解 $y$ を与えたとき、関数値が $y$ より小さいすべての要素、すなわち $f(x) < y$ となる要素を標識することです。これは準備オラクル $A_y$ とマーカー演算子 $O_y$ によって実現されます。準備演算子は関数値が $y$ より小さい入力を負の値で準備するため、2の補数表現で最上位ビット（MSB）が1になることでそれらを識別できます。したがって、マーカー演算子は単純に $Z$ ゲートとなります。最後に、拡散演算子が標識された状態の振幅を増幅します。
#
# Groverアンザッツは以下の3つの要素から構成されています：
#
# - $A_y$：状態 $\sum \ket{x, y - f(x)}$ を準備するオラクル（QFTエンコーディングで構築）
# - $O_y = Z$：MSB上のZゲート
# - $D$ 拡散演算子：$f(x) < y$ となる標識された要素の振幅を伝搬させる。単一の多重制御Zゲートで構築
#

# %%
from qamomile.circuit.algorithm.gas import apply_function_preparation_qubo

apply_function_preparation_qubo.draw(
    q_output = 9, 
    q_input = 9, 
    y = 0, 
    linear = converter.binary_model.linear, 
    quad = converter.binary_model.quad,
    inline = True,
    inline_depth = None,
    fold_loops = False
)

# %%
from qamomile.circuit.algorithm.gas import diffusion_op

diffusion_op.draw(
    q_input = 9,
    inline = True,
    inline_depth = None,
    fold_loops = False
)

# %% [markdown]
# ## 適応探索の古典レイヤー
#
# このセクションでは古典的な適応探索を定義します。適応探索はランダムな候補 $x$ と $y = f(x)$ から始まり、Grover回路からサンプリングを行い、より良い候補が得られた場合に解を更新します。停止条件を満たした時点で探索を終了します。ここでは改善が見られない繰り返しが一定回数続いた場合を停止条件として使用します。
#
# Grover探索を何回適用するかという問題が残っています。これに対処するため、適応探索は繰り返し回数をランダムにサンプリングし、改善が見られない場合にサンプリング空間を少しずつ拡大するランダム化アプローチを採用しています。
#
#
# 以下の関数はQamomileには含まれていません。`converter.transpile()` と `converter.decode()` を量子プリミティブとして使用し、GASの古典的な外側ループを実装しています。
#
# `converter.transpile(transpiler, y=y, num_iterations=num_iterations)` は現在の閾値 $y$ とGrover深度に対するGrover回路を構築します。
#
# `exec.sample(executor, shots=256)` はバックエンド上で回路を実行します。Groverであっても、NISQデバイスはノイジーであるため複数ショットが必要です。
#
# `converter.decode(result)` は生のビット列カウントを決定変数の割り当てにマッピングして返します。
#

# %%
from typing import Any
import numpy as np
import random
import ommx
from qiskit_aer import AerSimulator

def grover_adaptive_search(
    converter: Any,
    transpiler: Any,
    lamb: float,
    max_no_improvement: int = 5,
    seed: int = 900
):

    ##########################################################
    #                      初期化                         #
    ##########################################################

    random.seed(seed)
    n = converter.binary_model.num_bits
    k = 1  # 各ステップでサンプリングするGrover繰り返し回数の上限を制御
    x_int = random.randint(0, 2**n - 1)
    x = [int(b) for b in format(x_int, f"0{n}b")]
    y = converter.instance.evaluate({i: x_i for i, x_i in enumerate(x)}).objective

    current_iter = 0
    no_improvement_count = 0

    print("[GAS] 初期化")
    print(f"[GAS] n={n}, lambda={lamb}")
    print(f"[GAS] 初期状態: x={x}, y={y}, k={k}")

    executor = transpiler.executor(backend=AerSimulator(seed_simulator=seed,max_parallel_threads=None))

    ############################################################
    #                     メインループ                       #
    ############################################################

    while no_improvement_count < max_no_improvement:
        
        # t を {0, ..., ceil(k)-1} から一様にサンプリング；k == 1 のとき空の範囲を避ける
        num_iterations = random.randrange(max(1, int(np.ceil(k))))
        print(
            f"\n[GAS] イテレーション {current_iter + 1} | 現在の y={y}, k={k:.6f}, "
            f"Grover繰り返し={num_iterations}"
        )

        ####################################################
        #          量子Grover回路の呼び出し              #
        ####################################################

        exec = converter.transpile(transpiler,
                                y=y,
                                num_iterations=num_iterations)

        # NISQデバイスはノイジーなため、回路を複数回実行する
        job = exec.sample(executor, shots=256)
        result = job.result()
        sample_set = converter.decode(result)

        # 最良サンプルから x と y を抽出
        x_vals = sample_set.best_feasible.extract_decision_variables("x")
        candidate_x = [
            int(round(x_vals.get((i,), x_vals.get(i, 0.0))))
            for i in range(n)
        ]
        candidate_y = float(sample_set.best_feasible.objective)

        print(f"[GAS] 候補: x={candidate_x}, y={candidate_y}")

        if candidate_y < y:
            print("[GAS] 改善あり -> 候補を採用し k を 1 にリセット")
            x = candidate_x
            y = candidate_y
            k = 1
            no_improvement_count = 0
        else:
            old_k = k
            k = lamb * k
            no_improvement_count += 1
            print(
                f"[GAS] 改善なし -> 現在の解を維持し "
                f"k をスケール: {old_k:.6f} -> {k:.6f}"
            )

        current_iter += 1


    print(f"\n[GAS] {current_iter} 回のイテレーション後に終了。最良解: x={x}, y={y}")

    return x, y

# %% [markdown]
# ## パイプライン全体の実行
#
# Grover探索を用いたハイブリッド古典量子アルゴリズムを呼び出します。
#
# パラメータ `lamb` は繰り返し回数のサンプリング空間が拡大する速度を制御します。
# パラメータ `max_iter` と `max_no_improvement` は停止条件を設定します。
#

# %%
x,y = grover_adaptive_search(
    converter=converter,
    transpiler=transpiler,
    lamb=1.2,
    max_no_improvement=5,
)
selected = [i+1 for i, xi in enumerate(x) if xi == 1]
print(f"選択された資産: {selected}, 目的関数値: {y}")

# %% [markdown]
# $10$ 回のイテレーション後、解がそれ以上改善されないためアルゴリズムは停止します。
#
# ポートフォリオ問題のグローバル最小値が返されます。
#
# 最適解は資産 $1$、$5$、$8$ を購入することです。
#
