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
# # Grover適応探索によるチャネル割当
#
# このページでは、Qamomileの`GASConverter`を使って無線ネットワークの**チャネル割当問題**（CAP）を解きます。定式化はSano、Norimoto、Ishikawaによる高次二値定式化 [](https://arxiv.org/abs/2208.05181) に従います。Grover適応探索（GAS）そのもの {cite:p}`10.22331/q-2021-04-08-428` は[Grover適応探索のチュートリアル](grover_adaptive_search)で紹介しているため、このページでは制約付きで実数係数を持つ問題に適用する際の違いに焦点を当てます。
#
# このページでは次の順に進めます。
#
# 1. [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html)でCAPを2通りに定式化します。one-hotエンコーディングによるQUBOと、二進エンコーディングによるHUBOです。
# 2. 論文の例のネットワークから両方のインスタンスを作成します。
# 3. それぞれの定式化がGAS回路で必要とする量子ビット数を比較します。
# 4. HUBOに対してGASを実行し、結果を全探索と照合します。

# %%
# 最新のQamomileをpipでインストールしましょう！
# # !pip install "qamomile[qiskit]"

# %%
import itertools
import os
import random
import warnings

import jijmodeling as jm
import numpy as np
import ommx.v1
from qiskit_aer import AerSimulator

from qamomile.optimization.gas import GASConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## 背景
#
# 無線ネットワークには $N_{AP}$ 個のアクセスポイント（AP）があり、$N_{CH} < N_{AP}$ 個のチャネルを共有しなければなりません。各APはユーザー端末（UT）の集合 $U_i$ を受け持ちます。2つのAPが同じチャネルを使うと互いのUTに干渉し、ネットワーク全体のShannon容量が下がります。CAPはこの容量を最大化するチャネル割当を求める問題で、同じチャネルを共有するAP間の干渉の総和を最小化することと等価です。
#
# ```{figure} assets/gas_channel_assignment_system_model.png
# :width: 80%
# :alt: 4つのアクセスポイントがそれぞれ2つのユーザー端末を受け持ち、カバー範囲が重なり合っている様子。
#
# このページで使う例のネットワークです。4つのAP、8つのUT、AP $i$ とUT $u$ の距離 $d_{iu}$ を示しています。[](https://arxiv.org/abs/2208.05181) のFig. 2から転載しました。
# ```

# %% [markdown]
# ## 問題設定
#
# 論文のAppendix Aの例を使います。AP数は $N_{AP} = 4$、チャネル数は $N_{CH} = 3$、UTは各APに2つずつ計8つです。単純なパスロスモデル（減衰係数 $\alpha = 1$）のもとで、AP $i$ と $k$ の間の干渉コストは次のようになります。
#
# $$
# C_{ik} = -\log_2\left(1 + \frac{\sum_{u \in U_i} d_{iu}^{-\alpha}}{\sum_{v \in U_k} d_{iv}^{-\alpha}}\right)
#          -\log_2\left(1 + \frac{\sum_{u \in U_k} d_{ku}^{-\alpha}}{\sum_{v \in U_i} d_{kv}^{-\alpha}}\right)
# $$
#
# 論文ではこれを $D_{ik} = C_{ik} - C_{\min} + \epsilon$（$\epsilon = 0.01$、$C_{\min}$ は $C_{ik}$ の最小値）とずらし、すべて正の重みにしています。

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

N_AP = 4
N_CH = 3
alpha = 1
epsilon = 0.01

# 各APが受け持つUT
connections = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
}

# AP i（行）とUT u（列）の距離
d = np.array([
    [1, 1, 2, 4, 3, 5, 5, 8],
    [5, 4, 1, 1, 5, 4, 2, 4],
    [6, 5, 2, 5, 1, 1, 4, 6],
    [10, 8, 5, 2, 5, 3, 1, 1],
], dtype=float)

upper_idx = np.triu_indices(N_AP, k=1)
C = np.zeros((N_AP, N_AP))
for i, k in zip(*upper_idx):
    C[i, k] = -np.log2(
        1
        + np.sum(d[i, connections[i]] ** -alpha)
        / np.sum(d[i, connections[k]] ** -alpha)
    ) - np.log2(
        1
        + np.sum(d[k, connections[k]] ** -alpha)
        / np.sum(d[k, connections[i]] ** -alpha)
    )

D = np.zeros_like(C)
D[upper_idx] = C[upper_idx] - np.min(C[upper_idx]) + epsilon
D = np.round(D, decimals=3)
print(D)

# 論文のAppendix Aに記載されている重みです。
paper_D = {(0, 1): 1.835, (0, 2): 1.216, (0, 3): 0.010,
           (1, 2): 1.762, (1, 3): 2.333, (2, 3): 1.371}
for (i, k), value in paper_D.items():
    assert np.isclose(D[i, k], value, atol=1e-3, rtol=0.0), (i, k, D[i, k])

# %% [markdown]
# ## アルゴリズム
#
# GASでは、目的関数を二値変数の多項式として与える必要があります。論文は「AP $i$ がチャネル $c$ を使う」ことを表す2つのエンコーディングを比較しています。
#
# ### QUBO：one-hotエンコーディング
#
# AP $i$ がチャネル $c$ を使うとき $x_{ic} = 1$、そうでないとき $0$ とします。各行 $[x_{i1}, \ldots, x_{iN_{CH}}]$ はone-hotベクトルになります。2つのAP $i < k$ が同じチャネルを選ぶと干渉するため、次の問題になります。
#
# $$
# \min_x \sum_{i < k} D_{ik} \sum_{c=1}^{N_{CH}} x_{ic} x_{kc}
# \quad\text{s.t.}\quad \sum_{c=1}^{N_{CH}} x_{ic} = 1 \;\; \forall i .
# $$
#
# この定式化には $N_{AP} N_{CH}$ 個の二値変数が必要です。

# %%
@jm.Problem.define("Channel Assignment (QUBO)", sense=jm.ProblemSense.MINIMIZE)
def cap_qubo(problem: jm.DecoratedProblem):
    N_AP = problem.Length("N_AP")
    N_CH = problem.Length("N_CH")
    D = problem.Float("D", shape=(N_AP, N_AP), description="Pairwise interference weights")
    x = problem.BinaryVar(
        "x",
        shape=(N_AP, N_CH),
        description="1 if AP i uses channel c",
    )

    problem += jm.sum(
        D[i, k] * x[i, c] * x[k, c]
        for i in N_AP
        for k in N_AP
        if k > i
        for c in N_CH
    )

    problem += problem.Constraint(
        "one channel per AP",
        (jm.sum(x[i, c] for c in N_CH) == 1 for i in N_AP),
    )


cap_qubo

# %% [markdown]
# ### HUBO：二進エンコーディング
#
# 二進エンコーディングでは、各APのチャネル番号を $N_B = \lceil \log_2 N_{CH} \rceil$ ビットの符号語で表します。チャネル $c$ の符号語を $b_c \in \{0, 1\}^{N_B}$、AP $i$ の符号語の第 $r$ ビットを $x_{ir}$ とします。多項式
#
# $$
# \delta'_{ic}(x) = \prod_{r=1}^{N_B} \bigl(1 - b_{cr} + (2 b_{cr} - 1)\, x_{ir}\bigr)
# $$
#
# はビットごとの因子の積で、各因子は $b_{cr} = 1$ なら $x_{ir}$、$b_{cr} = 0$ なら $1 - x_{ir}$ になります。そのため、AP $i$ の符号語が $b_c$ に一致するときだけ $1$、それ以外では $0$ になります。目的関数は次のようになります。
#
# $$
# \min_x \sum_{i < k} D_{ik} \sum_{c=1}^{N_{CH}} \delta'_{ic}(x)\, \delta'_{kc}(x)
# \quad\text{s.t.}\quad \delta'_{ic}(x) = 0 \;\; \forall i,\; N_{CH} < c \le 2^{N_B} .
# $$
#
# 制約は、実在するチャネルに対応しない符号語を禁止します（$N_{CH} = 3$ では、2ビットの符号語4つのうち1つが使われません）。積を含むため高次（HUBO）の多項式になりますが、必要な変数は $N_{AP} N_B$ 個だけです。

# %%
@jm.Problem.define("Channel Assignment (HUBO)", sense=jm.ProblemSense.MINIMIZE)
def cap_hubo(problem: jm.DecoratedProblem):
    N_AP = problem.Length("N_AP")
    N_CH = problem.Length("N_CH")
    N_TOT = problem.Length("N_TOT")
    N_B = problem.Length("N_B")
    b = problem.Float("b", shape=(N_TOT, N_B), description="Codeword table")
    D = problem.Float("D", shape=(N_AP, N_AP), description="Pairwise interference weights")
    x = problem.BinaryVar(
        "x",
        shape=(N_AP, N_B),
        description="Bit r of AP i's channel codeword",
    )

    def delta(i, c):
        return jm.prod((1 - b[c, r] + (2 * b[c, r] - 1) * x[i, r]) for r in N_B)

    problem += jm.sum(
        D[i, k] * delta(i, c) * delta(k, c)
        for i in N_AP
        for k in N_AP
        if k > i
        for c in N_CH
    )

    problem += problem.Constraint(
        "no nonexistent channel",
        (delta(i, c) == 0 for i in N_AP for c in N_TOT if c >= N_CH),
    )


cap_hubo

# %% [markdown]
# ネットワークのデータで両方の問題を評価すると、2つのOMMXインスタンスが得られます。

# %%
N_B = int(np.ceil(np.log2(N_CH)))
N_TOT = 2**N_B
codewords = np.array(
    [[int(bit) for bit in np.binary_repr(c, width=N_B)] for c in range(N_TOT)]
)

instance_qubo = cap_qubo.eval({"N_AP": N_AP, "N_CH": N_CH, "D": D.tolist()})
instance_hubo = cap_hubo.eval({
    "N_AP": N_AP,
    "N_CH": N_CH,
    "N_TOT": N_TOT,
    "N_B": N_B,
    "b": codewords.tolist(),
    "D": D.tolist(),
})

assert len(instance_qubo.used_decision_variable_ids()) == N_AP * N_CH == 12
assert len(instance_hubo.used_decision_variable_ids()) == N_AP * N_B == 8

# %% [markdown]
# どちらのインスタンスも全列挙できる大きさです。全探索で、GASが到達すべき参照用の最適値を求めます。2つの定式化は同じ $N_{CH}^{N_{AP}} = 81$ 通りの割当を表すため、最適値も共通です。


# %%
def brute_force(instance: ommx.v1.Instance) -> tuple[list[int], float, int]:
    """最良の実行可能ビット列、その目的関数値、実行可能解の数を返します。"""
    ids = sorted(instance.used_decision_variable_ids())
    feasible = []
    for bits in itertools.product([0, 1], repeat=len(ids)):
        solution = instance.evaluate(dict(zip(ids, bits)))
        if solution.feasible:
            feasible.append((solution.objective, list(bits)))
    best_y, best_x = min(feasible)
    return best_x, best_y, len(feasible)


qubo_x, qubo_y, qubo_feasible = brute_force(instance_qubo)
hubo_x, hubo_y, hubo_feasible = brute_force(instance_hubo)
print(f"QUBO: 最適値 {qubo_y:.3f}、x={qubo_x}（実行可能解 {qubo_feasible} 個）")
print(f"HUBO: 最適値 {hubo_y:.3f}、x={hubo_x}（実行可能解 {hubo_feasible} 個）")

assert qubo_feasible == hubo_feasible == N_CH**N_AP
assert np.isclose(qubo_y, hubo_y, atol=1e-9, rtol=0.0)
# APが4つ、チャネルが3つなので、少なくとも1組はチャネルを共有します。最もコストが
# 小さいのはAP 1と4の組（D_14 = 0.010）なので、最適解ではその組だけが共有します。
assert np.isclose(hubo_y, paper_D[(0, 3)], atol=1e-9, rtol=0.0)

# %% [markdown]
# ## 実装
#
# ### 実数係数の扱い
#
# GAS回路は、QFTに基づく算術で $f(x) - y$ を量子ビットのレジスタに格納します。係数が整数なら、レジスタはこの値を正確に保持します。CAPの重み $D_{ik}$ は実数であり、`GASConverter.transpile()`には {cite:p}`10.22331/q-2021-04-08-428` のAppendixで議論されている2つの選択肢があります。
#
# - `approximate_real_coefficients=False`は実数係数をそのまま符号化します。このときレジスタが保持するのは $f(x) - y$ の近似で、その振幅は表現可能な最も近い値の周辺に集中します。前処理が不要で、必要な量子ビット数も最小です。
# - `approximate_real_coefficients=True`（デフォルト）は、先に係数をスケーリングして整数に丸めます。丸めた問題に対しては算術が厳密になります。`quantization_parameter`は丸めのビット数を指定します。`None`のときはQamomileが自動で値を選びますが、その分多くの量子ビットが必要になります。小さな値を手動で指定すると量子ビットを節約できますが、異なる目的関数値が同じ値にまとめられ、真の最適解を区別できなくなることがあります。
#
# 出力レジスタはすべての $x$ について $f(x) - y$ を保持する必要があるため、その幅はモデルと閾値 $y$ の両方に依存します。`transpile()`の後、公開メソッド`required_output_bits(y)`で、実際に符号化されたモデルに対する幅を確認できます。


# %%
def count_qubits(
    instance: ommx.v1.Instance,
    y: float,
    approximate_real_coefficients: bool,
    quantization_parameter: int | None = None,
) -> int:
    """閾値 ``y`` に対するGAS回路の総量子ビット数を返します。"""
    converter = GASConverter(instance)
    with warnings.catch_warnings():
        # transpile()は実数係数を丸めたことを警告しますが、ここではその比較が目的です。
        warnings.simplefilter("ignore", UserWarning)
        converter.transpile(
            QiskitTranspiler(),
            y=y,
            num_iterations=0,
            approximate_real_coefficients=approximate_real_coefficients,
            quantization_parameter=quantization_parameter,
        )
    return converter.binary_model.num_bits + converter.required_output_bits(y)


settings = {
    "real coefficients": {"approximate_real_coefficients": False},
    "rounded, 8 bits": {"approximate_real_coefficients": True, "quantization_parameter": 8},
    "rounded, automatic": {"approximate_real_coefficients": True},
}
# GASが開始する程度の閾値として、中程度の割当の目的関数値を使います。
y_example = 3.549
qubit_counts = {
    name: {
        label: count_qubits(instance, y_example, **kwargs)
        for label, kwargs in settings.items()
    }
    for name, instance in (("QUBO", instance_qubo), ("HUBO", instance_hubo))
}

print(f"{'':6}" + "".join(f"{label:>22}" for label in settings))
for name, counts in qubit_counts.items():
    print(f"{name:6}" + "".join(f"{counts[label]:>22}" for label in settings))

assert qubit_counts["QUBO"] == {
    "real coefficients": 19, "rounded, 8 bits": 24, "rounded, automatic": 36,
}
assert qubit_counts["HUBO"] == {
    "real coefficients": 16, "rounded, 8 bits": 20, "rounded, automatic": 35,
}

# %% [markdown]
# HUBOはQUBOより入力量子ビットを $N_{AP}(N_{CH} - N_B) = 4$ 個節約でき、探索空間も $2^{12}$ から $2^{8}$ 通りのビット列に縮小します。量子ビットが1つ増えるごとに古典シミュレーションのコストは2倍になるため、このページでは16量子ビットで済む、実数係数のHUBOに対してGASを実行します。
#
# ### 古典レイヤー
#
# 次のループは[Grover適応探索のチュートリアル](grover_adaptive_search)のものを、この問題向けに3点変更したものです。
#
# - ランダムなビット列は制約を満たさないことがあるため、指定した割当から探索を始めます。
# - 実数係数に関するオプションを`transpile()`に渡します。
# - 各候補のビットを決定変数のID順に読み取ります。HUBOの変数`x`は2次元なので、converterが使うIDと同じ順序でビットを読む必要があります。
#
# `converter.decode()`はサンプリングしたすべてのビット列をOMMXインスタンスで評価するため、`best_feasible`は制約を満たすサンプルのうち最良のものになります。


# %%
def bits_from_solution(solution: ommx.v1.Solution, ids: list[int]) -> list[int]:
    """解の二値変数の値を決定変数のID順に読み取ります。"""
    return [int(round(solution.state.entries[i])) for i in ids]


def grover_adaptive_search(
    converter: GASConverter,
    transpiler: QiskitTranspiler,
    initial_x: list[int],
    lamb: float,
    max_no_improvement: int = 5,
    shots: int = 256,
    seed: int = 900,
    approximate_real_coefficients: bool = False,
    quantization_parameter: int | None = None,
) -> tuple[list[int], float]:
    ##########################################################
    #                        初期化                          #
    ##########################################################

    rng = random.Random(seed)
    ids = sorted(converter.instance.used_decision_variable_ids())
    k = 1  # 各ステップでサンプリングするGrover繰り返し回数の上限を制御します
    x = list(initial_x)
    y = converter.instance.evaluate(dict(zip(ids, x))).objective

    current_iter = 0
    no_improvement_count = 0

    print("[GAS] 初期化")
    print(f"[GAS] n={len(ids)}, lambda={lamb}")
    print(f"[GAS] 初期状態: x={x}, y={y:.3f}, k={k}")

    executor = transpiler.executor(
        backend=AerSimulator(seed_simulator=seed, max_parallel_threads=None)
    )

    ############################################################
    #                      メインループ                        #
    ############################################################

    while no_improvement_count < max_no_improvement:
        # tを{0, ..., ceil(k)-1}から一様にサンプリングします。k == 1のとき空の範囲になるのを避けます
        num_iterations = rng.randrange(max(1, int(np.ceil(k))))
        print(
            f"\n[GAS] イテレーション {current_iter + 1} | 現在の y={y:.3f}, k={k:.6f}, "
            f"Grover繰り返し回数={num_iterations}"
        )

        ####################################################
        #            Grover量子回路の呼び出し              #
        ####################################################

        executable = converter.transpile(
            transpiler,
            y=y,
            num_iterations=num_iterations,
            approximate_real_coefficients=approximate_real_coefficients,
            quantization_parameter=quantization_parameter,
        )
        result = executable.sample(executor, shots=shots).result()
        sample_set = converter.decode(result)

        try:
            best = sample_set.best_feasible
        except RuntimeError:
            best = None

        if best is not None and best.objective < y:
            x = bits_from_solution(best, ids)
            y = best.objective
            k = 1
            no_improvement_count = 0
            print(f"[GAS] 改善を検出: x={x}, y={y:.3f} -> kを1にリセットします")
        else:
            old_k = k
            k = lamb * k
            no_improvement_count += 1
            print(
                f"[GAS] 改善なし -> 現在の解を維持し、kをスケーリングします: "
                f"{old_k:.6f} -> {k:.6f}"
            )

        current_iter += 1

    print(f"\n[GAS] {current_iter} 回のイテレーション後に終了。最良解: x={x}, y={y:.3f}")

    return x, y


# %% [markdown]
# ## 結果
#
# GASは、ランダムに選んだ有効な割当から開始します。各APに実在するチャネルをランダムに選び、その符号語で表します。

# %%
seed = 3 if docs_test_mode else 900
start_rng = np.random.default_rng(seed)
start_channels = start_rng.integers(0, N_CH, size=N_AP)
initial_x = [int(bit) for c in start_channels for bit in codewords[c]]
print(f"初期チャネル: {start_channels.tolist()} -> x={initial_x}")

converter = GASConverter(instance_hubo)
transpiler = QiskitTranspiler()

x, y = grover_adaptive_search(
    converter=converter,
    transpiler=transpiler,
    initial_x=initial_x,
    lamb=1.2,
    max_no_improvement=2 if docs_test_mode else 5,
    shots=16 if docs_test_mode else 256,
    seed=seed,
    approximate_real_coefficients=False,
)

hubo_ids = sorted(instance_hubo.used_decision_variable_ids())
solution = instance_hubo.evaluate(dict(zip(hubo_ids, x)))
assert solution.feasible, "GASは有効なチャネル割当を返さなければなりません"
assert np.isclose(y, solution.objective, atol=1e-9, rtol=0.0), (
    "報告された目的関数値は、返された割当の値と一致しなければなりません"
)

# %% [markdown]
# ビットをチャネルに戻すと、どのAPがチャネルを共有しているかがわかります。全探索による参照値でも結果を確認します。

# %%
channels = [
    int(np.flatnonzero((codewords == x[i * N_B:(i + 1) * N_B]).all(axis=1))[0])
    for i in range(N_AP)
]
print(f"各APのチャネル: {channels}")
print(f"GAS   : 目的関数値={y:.3f}")
print(f"全探索 : 目的関数値={hubo_y:.3f}")

assert np.isclose(y, hubo_y, atol=1e-9, rtol=0.0), (
    f"GASが返した目的関数値は {y} ですが、真の最適値は {hubo_y} です"
)
print("\nGASは全探索の最適解と一致しました。")

# %% [markdown]
# :::{note}
# このインスタンスのビット列は $2^8 = 256$ 通りしかないため、256ショットで探索空間の大部分をカバーでき、Grover繰り返し回数が0のラウンド（一様サンプリング）だけで最適解が見つかることもあります。この例は、定式化、回路、デコードが一通り正しく動くことを確認するものです。GASの二次高速化が効いてくるのは、ここではシミュレーションできない大きさのインスタンスです。
# :::
#
# ## まとめ
#
# このノートブックでは、次のことを行いました。
#
# - 無線チャネル割当問題をJijModelingで、one-hotエンコーディングのQUBOと二進エンコーディングのHUBOの2通りに定式化し、論文に記載されている干渉の重みを再現しました。
# - `GASConverter.transpile()`と`required_output_bits()`を使って量子ビット数を比較しました。実数係数のまま符号化すると、HUBOは16量子ビット、QUBOは19量子ビットを必要とし、係数を整数に丸めるとさらに増えます。
# - 実数係数のHUBOに対してGASを実行し、全探索による最適値に到達することを確認しました。
