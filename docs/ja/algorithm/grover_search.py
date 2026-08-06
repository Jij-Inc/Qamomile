# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [algorithm, oracle-based]
# ---
#
# # Groverの探索アルゴリズム
#
# Grover探索は、確率振幅を増幅することで、構造を持たない探索空間から探したい状態を見つけるアルゴリズムです{cite:p}`10.1145/237814.237866`。サイズ$N$の探索空間に探したい状態が1つある場合、古典的な全探索では$O(N)$回の問い合わせが必要ですが、Grover探索では$O(\sqrt{N})$回のオラクル問い合わせで探索でき、古典計算に対して2次的な計算速度の加速を示すことが知られています。
#
# このノートブックでは、Groverの探索アルゴリズムをスクラッチ実装とQamomileの組み込み関数`grover_search`による実装の2通りで紹介します。どちらも4つの検索量子ビットから探したい状態$|0101\rangle$を見つけます。振幅増幅の前後で測定確率を比較し、オラクル問い合わせ回数のスケーリングも確認します。

# %%
# 最新のQamomileと、このノートブックで使う追加機能をインストールします。
# # !pip install "qamomile[qiskit,visualization]"

# %%
import math
import os

import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## アルゴリズム
#
# 探したい$n$ビット列を$x_\star\in\{0,1\}^n$とし、探索空間の大きさを$N=2^n$とします。$x\in\{0,1\}^n$に対して、$f(x)$を
#
# $$
# f(x)=
# \begin{cases}
# 1 & (x=x_\star),\\
# 0 & (x\ne x_\star)
# \end{cases}
# $$
#
# と定義します。この$f$に対応する位相オラクル$O_f$を、次のように定義します。
#
# $$
# O_f|x\rangle = (-1)^{f(x)}|x\rangle.
# $$
#
# オラクル$O_f$は$|x_\star\rangle$を$-|x_\star\rangle$へ変化させ、ほかの基底状態は変化させません。以下では、このように探したい状態の符号を反転することを「探したい状態をマークする」と呼びます。この符号反転だけでは測定確率は変化しません。Grover探索では、オラクル$O_f$と後述する拡散演算子を1組として繰り返し、探したい状態の確率振幅を増幅します。
#
# :::{note}
# **オラクルとは？**
#
# 量子アルゴリズムにおけるオラクルとは、アルゴリズムが内部実装に依存せず呼び出す、問題固有の操作です。Grover探索では、その役割をオラクル$O_f$として与え、探したい状態の符号を反転します。オラクルを1回適用することを1回のオラクル問い合わせと数え、ゲート数とは別の計算量指標として扱います。実際に量子回路として実行するには、探索問題ごとにオラクルの内部回路を構成する必要があります。
# :::
#
# ### ステップ1：一様重ね合わせ状態を準備する
#
# $n$量子ビットの状態$|0\rangle^{\otimes n}$から始め、アダマールゲートによって
#
# $$
# |s\rangle = H^{\otimes n}|0\rangle^{\otimes n}
# = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle,
# \qquad N=2^n
# $$
#
# を準備します。探したい状態が1つの場合、$|\mathrm{good}\rangle=|x_\star\rangle$とし、残りの$N-1$個の基底状態の正規化された重ね合わせを$|\mathrm{bad}\rangle$と定義します。
#
# $$
# |\mathrm{bad}\rangle
# = \frac{1}{\sqrt{N-1}}
# \sum_{x\ne x_\star}|x\rangle.
# $$
#
# すると、初期状態は次のように表せます。
#
# $$
# |s\rangle
# = \cos\theta|\mathrm{bad}\rangle
# + \sin\theta|\mathrm{good}\rangle,
# \qquad
# \theta=\arcsin\frac{1}{\sqrt{N}}.
# $$
#
# ### ステップ2：位相オラクルで探したい状態をマークする
#
# 一様重ね合わせ状態にオラクル$O_f$を適用すると、探したい状態の成分だけ符号が反転します。
#
# $$
# O_f|s\rangle
# = \cos\theta|\mathrm{bad}\rangle
# - \sin\theta|\mathrm{good}\rangle.
# $$
#
# ここでは、$O_f$を探したい状態の符号を反転する抽象的な演算として扱います。補助量子ビットを使う回路は$O_f$を実現する方法の1つであり、Grover探索の数学的な手順自体には含まれません。このノートブックでは、後の「位相オラクル」節で、補助量子ビットを$|-\rangle$に準備し、位相キックバックを利用して$O_f$を実装します。
#
# ### ステップ3：一様重ね合わせ状態に関して反転する
#
# オラクルの後に拡散演算子
#
# $$
# D=2|s\rangle\langle s|-I
# $$
#
# を適用します。状態$|\psi\rangle=\sum_x a_x|x\rangle$における確率振幅の平均を$\bar{a}=\frac{1}{N}\sum_x a_x$とすると、拡散演算子は各確率振幅を$a_x\mapsto2\bar{a}-a_x$と変換します。これは確率ではなく、負や複素数にもなり得る確率振幅の変換です。
#
# オラクル$O_f$を適用した後に拡散演算子$D$を適用する一連の操作を、1回のGrover反復と呼びます。すなわち、
#
# $$
# G=DO_f
# $$
#
# です。$|\mathrm{bad}\rangle$と$|\mathrm{good}\rangle$が張る2次元平面では、オラクルと拡散演算子による2つの反転を組み合わせることで、状態ベクトルを一定方向に$2\theta$だけ回転させます。
#
# ### ステップ4：Grover反復を繰り返す
#
# $r$回反復した後の検索量子ビットの状態は
#
# $$
# G^r|s\rangle
# = \cos((2r+1)\theta)|\mathrm{bad}\rangle
# + \sin((2r+1)\theta)|\mathrm{good}\rangle
# $$
#
# です。したがって、探したい状態を測定する確率は
#
# $$
# P_r=\sin^2((2r+1)\theta)
# $$
#
# です。$P_r$が最初に最大になるのは、累積した角度$(2r+1)\theta$が最適な角度$\pi/2$に最も近づくときです。それ以上反復すると、探したい状態から確率振幅が離れることがあります。
#
# ### ステップ5：検索量子ビットを測定する
#
# 適切な回数だけ反復した後に検索量子ビットを測定すると、高い確率で$x_\star$が得られます。

# %% [markdown]
# ## Qamomileでの実装
#
# ### 問題設定
#
# この例では、4つの検索量子ビットを使って、$N=2^4=16$個の状態から探したい状態$|0101\rangle$を見つけます。

# %%
# 具体的な探索問題とサンプリング設定を定義し、`grover_iteration_count`で反復回数を計算します。
SEARCH_QUBITS = 4
NUM_MARKED = 1
MARKED_STATE = "0101"
ITERATIONS = qmc.grover_iteration_count(SEARCH_QUBITS, NUM_MARKED)

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
SHOTS = 512 if docs_test_mode else 4096
SAMPLER_SEED = 42

theta = math.asin(math.sqrt(NUM_MARKED / 2**SEARCH_QUBITS))
ideal_marked_probability = math.sin((2 * ITERATIONS + 1) * theta) ** 2

print("search qubits:", SEARCH_QUBITS)
print("marked state:", MARKED_STATE)
print("Grover iterations:", ITERATIONS)
print("ideal marked-state probability:", f"{ideal_marked_probability:.6f}")

assert ITERATIONS == 3
assert ideal_marked_probability > 0.96

# %% [markdown]
# ### 位相オラクル
#
# 以下のオラクルでは、Xゲートによって$|0101\rangle$を$|1111\rangle$へ写し、専用の補助量子ビットを$|-\rangle$に準備してマルチCNOTゲートを適用します。最後に、検索量子ビットと補助量子ビットへの前処理を逆順に戻します。補助量子ビットはオラクルを適用する前後で$|0\rangle$となるため、Grover反復ごとに同じ物理量子ビットを再利用できます。

# %%
# |0101>をマークするオラクルを実装します。
@qmc.qkernel
def oracle_operator(
    search: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    oracle_aux = qmc.qubit(name="oracle_aux")

    # |q3 q2 q1 q0> = |0101>を|1111>へ写します。
    search[1] = qmc.x(search[1])
    search[3] = qmc.x(search[3])

    # 補助量子ビットを|->へ準備します。
    oracle_aux = qmc.x(oracle_aux)
    oracle_aux = qmc.h(oracle_aux)

    # すべての検索量子ビットが1の場合にのみ補助量子ビットを反転します。
    multi_controlled_x = qmc.control(
        qmc.x,
        num_controls=search.shape[0],
    )
    search, oracle_aux = multi_controlled_x(search, oracle_aux)

    # 再利用できるように補助量子ビットを|0>へ戻します。
    oracle_aux = qmc.h(oracle_aux)
    oracle_aux = qmc.x(oracle_aux)

    # もとの計算基底ラベルへ戻します。
    search[1] = qmc.x(search[1])
    search[3] = qmc.x(search[3])
    return search


oracle_operator.draw(search=SEARCH_QUBITS, fold_loops=False)

# %% [markdown]
# ### スクラッチ実装
#
# 拡散演算子は、マルチ制御ZゲートをアダマールゲートとXゲートで挟むことで実装できます。以下の`diffusion_operator`量子カーネルでは、この分解を直接記述します。`grover_search_from_scratch`では一様重ね合わせを準備し、位相オラクルと拡散演算子を順に繰り返します。

# %%
# 拡散演算子とGrover反復全体をスクラッチ実装します。
@qmc.qkernel
def diffusion_operator(
    search: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    search = qmc.h(search)
    search = qmc.x(search)

    top = search.shape[0] - 1
    search[top] = qmc.h(search[top])
    multi_controlled_x = qmc.control(qmc.x, num_controls=top)
    search[0:top], search[top] = multi_controlled_x(
        search[0:top], search[top]
    )
    search[top] = qmc.h(search[top])

    search = qmc.x(search)
    search = qmc.h(search)
    return search


@qmc.qkernel
def grover_search_from_scratch(
    iterations: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    search = qmc.qubit_array(SEARCH_QUBITS, name="search")
    search = qmc.h(search)

    for _ in qmc.range(iterations):
        search = oracle_operator(search)
        search = diffusion_operator(search)

    return qmc.measure(search)


grover_search_from_scratch.draw(iterations=ITERATIONS, fold_loops=False)

# %% [markdown]
# ### 組み込み関数：`qmc.grover_search`
#
# 組み込み関数は、同じ一様重ね合わせの準備と、位相オラクル・拡散演算子の反復を実行します。呼び出し側では、検索量子ビット、位相オラクル、反復回数を渡します。

# %%
# Qamomileの組み込み関数で同じ探索を実装します。
@qmc.qkernel
def grover_search_with_stdlib(
    iterations: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    search = qmc.qubit_array(SEARCH_QUBITS, name="search")
    search = qmc.grover_search(search, oracle_operator, iterations)
    return qmc.measure(search)


grover_search_with_stdlib.draw(iterations=ITERATIONS, fold_loops=False)

# %%
# どちらの実装も4つの検索量子ビットと1つの補助量子ビットを使うことを確認します。
scratch_resources = grover_search_from_scratch.estimate_resources(
    inputs={"iterations": ITERATIONS}
).simplify()
stdlib_resources = grover_search_with_stdlib.estimate_resources(
    inputs={"iterations": ITERATIONS}
).simplify()
print("scratch implementation qubits:", scratch_resources.qubits)
print("built-in implementation qubits:", stdlib_resources.qubits)
assert scratch_resources.qubits == SEARCH_QUBITS + 1
assert stdlib_resources.qubits == SEARCH_QUBITS + 1

# %% [markdown]
# ## 実行結果
#
# 一様重ね合わせと2つのGrover実装をQiskitへトランスパイルし、同じショット数でサンプリングします。1つ目のヒストグラムはほぼ一様になるはずです。どちらのGrover実装でも、3回反復した後は`|0101>`の確率が最も高くなるはずです。
#
# ### 量子回路の実行

# %%
# 最初のオラクル問い合わせ前の一様分布を取得します。
@qmc.qkernel
def uniform_superposition() -> qmc.Vector[qmc.Bit]:
    search = qmc.qubit_array(SEARCH_QUBITS, name="search")
    _oracle_aux = qmc.qubit(name="oracle_aux")
    search = qmc.h(search)
    return qmc.measure(search)


# 結果を再現できるようにシミュレータのseedを固定し、量子カーネルをトランスパイルしてサンプリングします。
def sample_kernel(kernel, *, bindings: dict[str, int] | None = None, seed: int):
    executable = transpiler.transpile(kernel, bindings=bindings or {})
    executor = transpiler.executor(
        backend=AerSimulator(
            seed_simulator=seed,
            max_parallel_threads=1,
        )
    )
    return executable.sample(executor, shots=SHOTS).result()


before_result = sample_kernel(uniform_superposition, seed=SAMPLER_SEED)
scratch_result = sample_kernel(
    grover_search_from_scratch,
    bindings={"iterations": ITERATIONS},
    seed=SAMPLER_SEED,
)
stdlib_result = sample_kernel(
    grover_search_with_stdlib,
    bindings={"iterations": ITERATIONS},
    seed=SAMPLER_SEED,
)

# %% [markdown]
# ### 結果のプロットと確認
#
# サンプリング結果を確率へ変換し、振幅増幅前の分布と2つの実装による振幅増幅後の分布を比較します。

# %%
# `search[0]`を右端の最下位ビット（LSB）として扱い、LSB-firstの測定タプルを通常の$|q_3q_2q_1q_0\rangle$表記の確率へ変換します。
def state_probabilities(sample_result) -> dict[str, float]:
    return {
        "".join(str(bit) for bit in reversed(state)): count / sample_result.shots
        for state, count in sample_result.results
    }


before_probabilities = state_probabilities(before_result)
scratch_probabilities = state_probabilities(scratch_result)
stdlib_probabilities = state_probabilities(stdlib_result)

# q0が右端の最下位ビットになるように基底状態を表示します。
basis_states = [
    format(index, f"0{SEARCH_QUBITS}b") for index in range(2**SEARCH_QUBITS)
]
before_values = [before_probabilities.get(state, 0.0) for state in basis_states]
scratch_values = [scratch_probabilities.get(state, 0.0) for state in basis_states]
stdlib_values = [stdlib_probabilities.get(state, 0.0) for state in basis_states]

# Grover探索前の確率と、それぞれの実装による探索後の確率を比較します。
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
axes[0].bar(basis_states, before_values, color="#8DBFE6")
axes[0].axhline(
    1 / 2**SEARCH_QUBITS,
    color="#DB4D3F",
    linestyle="--",
    label="uniform probability",
)
axes[0].set_title("Before Grover search")
axes[0].set_ylabel("probability")
axes[0].legend()

bar_colors = ["#DB4D3F" if state == MARKED_STATE else "#8DBFE6" for state in basis_states]
axes[1].bar(basis_states, scratch_values, color=bar_colors)
axes[1].set_title(f"From scratch ({ITERATIONS} iterations)")
axes[1].set_ylabel("probability")
axes[2].bar(basis_states, stdlib_values, color=bar_colors)
axes[2].set_title(f"With qmc.grover_search ({ITERATIONS} iterations)")
axes[2].set_xlabel("search state")
axes[2].set_ylabel("probability")
axes[2].tick_params(axis="x", rotation=45)

for ax in axes:
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# 一様な入力と、増幅された探したい状態を検証します。
uniform_probability = 1 / 2**SEARCH_QUBITS
assert all(abs(value - uniform_probability) < 0.06 for value in before_values)
assert max(scratch_probabilities, key=scratch_probabilities.get) == MARKED_STATE
assert max(stdlib_probabilities, key=stdlib_probabilities.get) == MARKED_STATE
assert scratch_probabilities[MARKED_STATE] > 0.85
assert stdlib_probabilities[MARKED_STATE] > 0.85

print(
    "from-scratch marked-state probability:",
    f"{scratch_probabilities[MARKED_STATE]:.6f}",
)
print(
    "built-in marked-state probability:",
    f"{stdlib_probabilities[MARKED_STATE]:.6f}",
)
print("ideal marked-state probability:", f"{ideal_marked_probability:.6f}")

# %% [markdown]
# 1つ目のヒストグラムでは、各状態の確率が一様分布の$1/16$に近くなっています。オラクルと拡散演算子を3回繰り返すと、どちらの実装でもほぼすべての確率が`|0101>`に集中し、$\sin^2((2r+1)\theta)$から予測される理想値に近づきます。サンプリングによって、厳密な確率から小さなずれが生じます。

# %% [markdown]
# ## リソース推定
#
# Grover反復では、位相オラクルを1回呼び出します。この例では`grover_iteration_count`の戻り値を反復回数として使うため、その値が位相オラクルへの問い合わせ回数にもなります。検索量子ビット数を変えてこの関数を評価し、参照用のスケーリング曲線と比較します。
# %%
# 検索量子ビット数ごとに、位相オラクルへの問い合わせ回数を計算します。
RESOURCE_SEARCH_QUBITS = [4, 6, 8, 10, 12, 14, 16]
grover_query_counts = [
    qmc.grover_iteration_count(num_search_qubits, 1)
    for num_search_qubits in RESOURCE_SEARCH_QUBITS
]

grover_scaling_reference = [
    grover_query_counts[0]
    * 2 ** ((num_search_qubits - RESOURCE_SEARCH_QUBITS[0]) // 2)
    for num_search_qubits in RESOURCE_SEARCH_QUBITS
]
exhaustive_search_reference = [
    2**num_search_qubits for num_search_qubits in RESOURCE_SEARCH_QUBITS
]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    RESOURCE_SEARCH_QUBITS,
    grover_query_counts,
    marker="o",
    color="#2696EB",
    label="Grover's oracle queries",
)
ax.plot(
    RESOURCE_SEARCH_QUBITS,
    grover_scaling_reference,
    linestyle="--",
    color="#FF6B6B",
    label=r"$O(\sqrt{N})$",
)
ax.plot(
    RESOURCE_SEARCH_QUBITS,
    exhaustive_search_reference,
    linestyle=":",
    color="#4ECDC4",
    label=r"$O(N)$",
)
ax.set_xlabel(r"search qubits $n$")
ax.set_ylabel("oracle queries")
ax.set_xticks(RESOURCE_SEARCH_QUBITS)
ax.set_yscale("log", base=10)
ax.grid(alpha=0.25, which="both")
ax.legend()
plt.tight_layout()
plt.show()

assert grover_query_counts == [3, 6, 12, 25, 50, 100, 201]
assert all(
    later > earlier
    for earlier, later in zip(grover_query_counts, grover_query_counts[1:])
)
assert all(
    grover_queries < exhaustive_queries
    for grover_queries, exhaustive_queries in zip(
        grover_query_counts, exhaustive_search_reference
    )
)

# %% [markdown]
# 探したい状態が1つの場合、`grover_iteration_count(n, 1)`によって選ばれる回数は、
#
# $$
# r(n)=\left\lfloor\frac{\pi}{4}\sqrt{2^n}\right\rfloor
# =\Theta(2^{n/2})
# $$
#
# です。探したい状態が1つの場合、$\theta=\arcsin(1/\sqrt{2^n})\approx1/\sqrt{2^n}$です。係数$\pi/4$は、最適な角度$\pi/2$をGrover反復1回あたりの回転角$2\theta$で割ることから得られます。対数プロットでは、関数が選んだ問い合わせ回数を$O(\sqrt{N})$と$O(N)$の参照曲線と比較しています。この2本の参照曲線は、Grover探索と古典的な全探索のオラクル問い合わせ回数に2次の差があることを表します。
#
# :::{note}
# この比較では、位相オラクルを1回適用することを1回の問い合わせとして数え、オラクルの内部回路は考慮していません。具体的な位相オラクルでは、1回の適用に複数のゲートや補助量子ビットが必要になることがあります。そのため、実際のリソースを評価するときは、問題に応じたオラクルのゲート数と補助量子ビット数も考慮する必要があります。
# :::

# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを学びました。
#
# - 位相オラクルと拡散演算子を使うと、探したい状態の確率振幅を増幅できます。
# - Grover反復はスクラッチ実装でき、`qmc.grover_search`を使うと同じ処理を簡潔に記述できます。
# - Grover探索のオラクル問い合わせ回数は$O(\sqrt{N})$、古典的な全探索では$O(N)$です。この比較には、問題に依存するオラクル内部のゲート数は含まれません。
