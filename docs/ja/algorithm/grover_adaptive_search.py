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
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # 組合せ多項式二値最適化のためのGrover適応探索
#
# Grover適応探索（GAS）は、二値変数上の多項式目的関数を最小化する手法です。Groverオラクルに「どの $x$ が $f(x) < y$ を満たすか」という1つの問いを繰り返し投げ、より良い解が見つかるたびに閾値 $y$ を下げていきます {cite:p}`10.22331/q-2021-04-08-428`。
#
# このページでは、Qamomileの`GASConverter`を使って制約なしの**ポートフォリオ選択**問題を解きます。
#
# このチュートリアルは次の構成で進めます。
#
# 1. [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html)で問題を定式化します。
# 2. 具体的なデータからインスタンスを作成します。
# 3. `GASConverter`で現在の閾値に対応するGrover回路を構築します。
# 4. サンプリングして最良の候補を保持し、停止条件を満たすまで繰り返します。GASは単一の回路ではなくハイブリッドなループです。

# %%
# 最新のQamomileをpipでインストールしましょう！
# # !pip install "qamomile[qiskit,visualization]"

# %%
import itertools
import os
import random
from typing import Any

import jijmodeling as jm
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.gas import (
    apply_function_preparation_qubo,
    diffusion_op,
    grover_algorithm,
)
from qamomile.circuit.visualization import MatplotlibDrawer
from qamomile.optimization.gas import GASConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## 背景
#
# Grover探索は、目的の状態の位相を反転させて区別するオラクルを受け取り、振幅増幅によってその測定確率を高めます。これは判定問題を解く手続きです。固定された判定条件はどの状態が条件を満たすかを示すだけで、どれが最良かは示しません。
#
# GASは判定条件を変更することで、これを最小化に応用します。$f(x) < y$ を満たす候補に対応する量子状態の位相を反転させて区別し、候補をサンプリングします。候補の目的関数値が現在の閾値を下回る場合、古典レイヤーが $y$ をその値に更新します。量子回路が一度に1つの固定された問いに答え、古典レイヤーが閾値を更新するという処理を繰り返します。

# %% [markdown]
# ## 問題設定
#
# このチュートリアルでは、GASを適用する問題例としてポートフォリオ選択問題を扱います。
#
# $n$ 個の資産（株式、債券など）があり、それぞれを購入するかどうかという二値の選択をします。リターンを最大化しつつリスクを最小化したいのですが、次のような対立があります。
#
# - $\mu$ は各資産に期待される収益性を表します
# - $\Sigma$ は資産同士の連動の仕方を表します（相関のある資産はリスクを増幅させます）
# - $q$ はリターンに対してリスクをどれだけ重視するかを制御します
#
# 相関リスクを過度に負うことなく、できるだけ大きな収益を得られる資産の部分集合を選ぶ必要があります。
#
# \begin{equation}
# \min_{x\in \lbrace 0 , 1 \rbrace^n} \big( q x^T \Sigma x - \mu^T x \big)
# \end{equation}


# %%
@jm.Problem.define("Portfolio Optimization (Unconstrained)")
def portfolio_problem(problem: jm.DecoratedProblem):
    n = problem.Length(description="資産の数")
    q = problem.Float("q", description="リスク回避係数")
    mu = problem.Float("mu", shape=(n,), description="期待収益ベクトル")
    Sigma = problem.Float("Sigma", shape=(n, n), description="共分散行列")

    x = problem.BinaryVar("x", shape=(n,), description="資産iを選ぶ場合は1")

    problem += (
        q * jm.sum(Sigma[i, j] * x[i] * x[j] for i in n for j in n)
        - jm.sum(mu[i] * x[i] for i in n)
    )


portfolio_problem

# %% [markdown]
# 以下のインスタンスは9個の資産を持ちます。
#
# | 名前     | 期待収益 | 分散       |
# |---------|-----------------|------------|
# | 資産1 | 22              | 12         |
# | 資産2 | 4               | 15         |
# | 資産3 | 19              | 10         |
# | 資産4 | 3               | 18         |
# | 資産5 | 23              | 14         |
# | 資産6 | 2               | 20         |
# | 資産7 | 5               | 11         |
# | 資産8 | 25              | 16         |
# | 資産9 | 3               | 13         |

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
num_assets = 3 if docs_test_mode else 9
q = 1
mu = np.array([22, 4, 19, 3, 23, 2, 5, 25, 3], dtype=int)
Sigma = np.array([
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
mu = mu[:num_assets]
Sigma = Sigma[:num_assets, :num_assets]

assert mu.shape == (num_assets,)
assert Sigma.shape == (num_assets, num_assets)
assert np.array_equal(Sigma, Sigma.T), "共分散行列は対称でなければなりません"

# %% [markdown]
# JijModelingの問題を上記のデータで評価すると、OMMXインスタンスが得られます。

# %%
instance = portfolio_problem.eval({
    "n": num_assets,
    "q": int(q),
    "mu": mu.tolist(),
    "Sigma": Sigma.tolist(),
})

assert len(instance.decision_variables) == num_assets
# 何も購入しないポートフォリオの目的関数値はちょうど0になります。
empty_portfolio = instance.evaluate({i: 0 for i in range(num_assets)}).objective
assert np.isclose(empty_portfolio, 0.0, atol=1e-9, rtol=0.0)

# %% [markdown]
# ## アルゴリズム
#
# GASでは、候補解の目的関数値を閾値 $y$ として設定し、それを下回る候補をGrover探索で探します。量子回路から得られた候補を古典計算で評価し、より良い候補が見つかれば閾値を更新するという処理を、停止条件を満たすまで繰り返します。ここでの $y$ は候補解ではなく、目的関数値に対する閾値です。GASは、判定条件 $f(x) < y$ を満たす候補に対応する量子状態の位相を反転させ、ほかの候補と区別します。Groverアンザッツは次の3つの要素から構成されます。
#
# - $A_y$：準備演算子です。量子辞書の状態 $\sum_x \ket{x, f(x) - y}$ を構築することで、各入力に $f(x) - y$ を対応付けます。回路実装は {cite:p}`10.22331/q-2021-04-08-428` により、QFTを用いる方法で与えられています。レジスタは $f(x) - y$ を2の補数で保持するため、判定条件を満たす候補は、最上位ビット（MSB）が $1$ になるものとして識別できます。
# - $O_y$：位相オラクルです。判定条件 $f(x) < y$ を満たす候補の位相を反転させます。符号化された値 $f(x) - y$ が負かどうかは最上位ビット（MSB）で判別できるため、そのMSBに作用する1つの $Z$ ゲートで実装できます。
# - $D$：拡散演算子で、位相を反転させて区別した状態の振幅を増幅します。$X$ 層に挟まれた1つの多重制御 $Z$ ゲートで構成されます。
#
# 1回の反復では $O_y$ を適用し、続いて $A_y^\dagger$、$D$、$A_y$ を適用します。位相反転 $O_y$ だけでは測定確率は変わりません。それを振幅へ変換するのが反射 $A_y D A_y^\dagger$ です。入力レジスタの測定で得られた候補を古典レイヤーで評価し、その目的関数値が現在の閾値を下回る場合にのみ、$y$ をその値に更新します。
#
# 残る問題は、Grover演算子を何回適用するかです。GASはこれに対して、改善が得られなかったラウンドのたびに少しずつ広がる範囲から繰り返し回数をサンプリングすることで答えます。

# %% [markdown]
# ## 実装
#
# Qamomileは、GASに用いる量子回路の実装を`GASConverter`として提供しています。このセクションでは、その使い方と回路を構成する演算を紹介し、最後に古典計算のループと組み合わせてGASを実装します。
#
# `GASConverter`はOMMXインスタンスを受け取り、QUBO/HUBO形式の問題へ変換します。`transpile()`メソッドを使うと、指定したtranspilerに対して適応探索で使うGrover回路を構築できます。

# %%
converter = GASConverter(instance)
transpiler = QiskitTranspiler()

assert converter.binary_model.num_bits == num_assets

# %% [markdown]
# ### Grover回路の実装
#
# `GASConverter.transpile()`は内部で以下のサンプリング用量子カーネルを構築し、transpilerに渡します。可視化するには、同じ量子カーネルに対して`Transpiler.to_block`と`MatplotlibDrawer.draw`を使います。
#
# 出力レジスタの幅は自由なパラメータではありません。すべての $x$ について $f(x) - y$ を保持できる必要があり、足りないと2の補数表現が巡回して、オラクルが判定している符号ビットそのものが反転してしまいます。`required_output_bits()`は、与えた閾値に対して`transpile()`が選ぶ幅を返します。

# %%
output_bits = converter.required_output_bits(y=0)
print(f"y=0に対する出力レジスタの幅: {output_bits} 量子ビット")

assert output_bits >= 2


# %%
@qmc.qkernel
def sampling_grover_algorithm(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt = 1,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    q_output, q_input = grover_algorithm(n, m, y, linear, quad, iters)
    return qmc.measure(q_output), qmc.measure(q_input)


block = transpiler.to_block(
    sampling_grover_algorithm,
    bindings={
        # 入力量子ビット数 = 二値変数の数
        "n": converter.binary_model.num_bits,
        # 出力量子ビット数（コンバータが計算した値）
        "m": output_bits,
        # オラクルの閾値：f(x) < y を満たす候補に対応する状態の位相を反転させて区別します
        "y": 0,
        "linear": converter.binary_model.linear,
        "quad": converter.binary_model.quad,
        # Groverの繰り返し回数（調整可能）
        "iters": 1,
    },
)
block = transpiler.inline(block)

assert len(block.operations) > 0, "インライン展開したGroverブロックは空であってはなりません"

MatplotlibDrawer(block).draw(fold_loops=False)

# %% [markdown]
# ### 構成要素の確認
#
# $A_y$ はQFTによる位相エンコーディングから $\sum_x \ket{x, f(x) - y}$ を準備します。

# %%
preparation_figure = apply_function_preparation_qubo.draw(
    q_output=output_bits,
    q_input=converter.binary_model.num_bits,
    y=0,
    linear=converter.binary_model.linear,
    quad=converter.binary_model.quad,
    inline=True,
    inline_depth=None,
    fold_loops=False,
)

assert preparation_figure.get_axes(), "描画した準備回路は空であってはなりません"
preparation_figure

# %% [markdown]
# $D$ は入力レジスタを一様重ね合わせのまわりで反転させます。

# %%
diffusion_figure = diffusion_op.draw(
    q_input=converter.binary_model.num_bits,
    inline=True,
    inline_depth=None,
    fold_loops=False,
)

assert diffusion_figure.get_axes(), "描画した拡散回路は空であってはなりません"
diffusion_figure

# %% [markdown]
# ### 古典レイヤー
#
# Qamomileの`GASConverter`は、指定した閾値に対するGrover回路の構築と、測定結果を候補解に変換する機能を提供します。一方、候補解の評価や閾値の更新、Grover反復回数の調整、停止条件の判定といった、探索全体を制御する古典計算部分は含まれていません。そのため、GASによる最適化を行うには、これらの処理を自分で実装する必要があります。以下の関数は、その実装例です。
#
# `converter.transpile()`と`converter.decode()`を量子プリミティブとして使い、GASの古典的な外側ループを実装しています。ランダムな候補 $x$ と $y = f(x)$ から始め、Grover回路をサンプリングし、より良いサンプルが得られるたびに暫定解を更新します。改善のないラウンドが`max_no_improvement`回続いた時点で探索を終了します。
#
# - `converter.transpile(transpiler, y=y, num_iterations=num_iterations)`は、現在の閾値 $y$ と指定したGrover反復回数に対応するGrover回路を構築します。
# - `executable.sample(executor, shots=256)`はバックエンド上で回路を実行します。Groverであっても、NISQデバイスはノイジーであるため複数ショットが必要です。
# - `converter.decode(result)`は生のビット列カウントを決定変数の割り当てにマッピングして返します。


# %%
def grover_adaptive_search(
    converter: Any,
    transpiler: Any,
    lamb: float,
    max_no_improvement: int = 5,
    shots: int = 256,
    seed: int = 900,
):
    ##########################################################
    #                        初期化                          #
    ##########################################################

    random.seed(seed)
    n = converter.binary_model.num_bits
    k = 1  # 各ステップでサンプリングするGrover繰り返し回数の上限を制御します
    x_int = random.randint(0, 2**n - 1)
    x = [int(b) for b in format(x_int, f"0{n}b")]
    y = converter.instance.evaluate({i: x_i for i, x_i in enumerate(x)}).objective

    current_iter = 0
    no_improvement_count = 0

    print("[GAS] 初期化")
    print(f"[GAS] n={n}, lambda={lamb}")
    print(f"[GAS] 初期状態: x={x}, y={y}, k={k}")

    executor = transpiler.executor(
        backend=AerSimulator(seed_simulator=seed, max_parallel_threads=None)
    )

    ############################################################
    #                      メインループ                        #
    ############################################################

    while no_improvement_count < max_no_improvement:
        # tを{0, ..., ceil(k)-1}から一様にサンプリングします。k == 1のとき空の範囲になるのを避けます
        num_iterations = random.randrange(max(1, int(np.ceil(k))))
        print(
            f"\n[GAS] イテレーション {current_iter + 1} | 現在のy={y}, k={k:.6f}, "
            f"Grover繰り返し回数={num_iterations}"
        )

        ####################################################
        #            Grover量子回路の呼び出し              #
        ####################################################

        executable = converter.transpile(
            transpiler, y=y, num_iterations=num_iterations
        )

        # NISQハードウェアはノイジーであるため、回路を複数回実行します
        job = executable.sample(executor, shots=shots)
        result = job.result()
        sample_set = converter.decode(result)

        # 最良のサンプルからxとyを取り出します
        x_vals = sample_set.best_feasible.extract_decision_variables("x")
        candidate_x = [
            int(round(x_vals.get((i,), x_vals.get(i, 0.0)))) for i in range(n)
        ]
        candidate_y = float(sample_set.best_feasible.objective)

        print(f"[GAS] 候補: x={candidate_x}, y={candidate_y}")

        if candidate_y < y:
            print("[GAS] 改善を検出 -> 候補を採用し、kを1にリセットします")
            x = candidate_x
            y = candidate_y
            k = 1
            no_improvement_count = 0
        else:
            old_k = k
            k = lamb * k
            no_improvement_count += 1
            print(
                f"[GAS] 改善なし -> 現在の解を維持し、kをスケーリングします: "
                f"{old_k:.6f} -> {k:.6f}"
            )

        current_iter += 1

    print(f"\n[GAS] {current_iter} 回のイテレーション後に終了。最良解: x={x}, y={y}")

    return x, y


# %% [markdown]
# ## 結果
#
# このセクションでは、ポートフォリオ選択問題にGASを実行し、得られた目的関数値を全探索による最適値と比較して結果を検証します。
#
# `lamb`はGrover繰り返し回数のサンプリング範囲が広がる速さを、`max_no_improvement`は停止条件を決めます。

# %%
x, y = grover_adaptive_search(
    converter=converter,
    transpiler=transpiler,
    lamb=1.2,
    max_no_improvement=2 if docs_test_mode else 5,
    shots=16 if docs_test_mode else 256,
    seed=0 if docs_test_mode else 900,
)
selected = [i + 1 for i, xi in enumerate(x) if xi == 1]
print(f"選択された資産: {selected}, 目的関数値: {y}")

assert len(x) == num_assets
assert all(xi in (0, 1) for xi in x)
assert np.isclose(
    y,
    instance.evaluate({i: xi for i, xi in enumerate(x)}).objective,
    atol=1e-9,
    rtol=0.0,
), "報告された目的関数値は返された割り当てと一致しなければなりません"

# %% [markdown]
# 探索は`max_no_improvement`回続けて解が改善しなくなった時点で停止します。これはヒューリスティックな規則であり、それ自体が返された解の最適性を保証するものではありません。今回は資産が $9$ 個であり、$2^9 = 512$ 通りの割り当てをすべて列挙できるため、厳密な全探索の結果と照合して確認しましょう。

# %%
brute_force_x, brute_force_y = min(
    (
        (list(bits), instance.evaluate({i: b for i, b in enumerate(bits)}).objective)
        for bits in itertools.product([0, 1], repeat=num_assets)
    ),
    key=lambda candidate: candidate[1],
)

print(f"GAS   : x={x}, 目的関数値={y}")
print(f"全探索 : x={brute_force_x}, 目的関数値={brute_force_y}")

assert np.isclose(y, brute_force_y, atol=1e-9, rtol=0.0), (
    f"GASは目的関数値 {y} を返しましたが、真の最適値は {brute_force_y} です"
)
print("\nGASは全探索の最適解と一致しました。")

# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを行いました。
#
# - 制約なしのポートフォリオ問題をGrover適応探索で解きました。量子回路が「どの $x$ が $f(x) < y$ を満たすか」に答え、古典ループがより良い候補を得たときに閾値 $y$ を更新します。
# - GASをQamomileで実装する例を紹介しました。量子側の処理は`GASConverter`として提供されています。`transpile()`が現在の閾値とGrover反復回数に対応するGrover回路を構築し、`decode()`がビット列カウントをOMMXを通じて決定変数に戻します。
# - ポートフォリオ選択問題を例にGASを実行し、得られた目的関数値が全探索による最適値と一致することを確認しました。
