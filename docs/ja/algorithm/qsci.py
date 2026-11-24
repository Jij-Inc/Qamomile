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
# tags: [algorithm, sample-based]
# ---
#
# # Quantum Selected Configuration Interaction（QSCI）
#
# QSCI は、量子状態を測定して得たビット列に対応する計算基底を選び、その部分空間に制限したハミルトニアンを古典コンピュータで対角化する手法です。量子側では「どの基底を使うか」を選ぶための情報を集めます。古典側では、選ばれた基底で表した Hamiltonian を対角化し、その範囲での最低エネルギーと対応する状態を求めます。
#
# ここでは、分子ではなく 4 量子ビットの横磁場イジングモデルを用いることにしましょう。ここでの配置とは、4つのスピンの計算基底状態を表すビット列のことです。短い VQE で入力状態を準備し、QSCI の部分空間を大きくしたときのエネルギーを比較します。

# %%
# Install Qamomile with the QURI Parts extra and the numerical and plotting dependencies.
# # !pip install "qamomile[quri_parts]" numpy scipy matplotlib

# %%
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.quri_parts import Qulacs, QuriParts

# %% [markdown]
# ## 問題設定
#
# 両端が結合していない、4 量子ビットの横磁場イジング鎖を考えましょう。ハミルトニアンは
#
# $$
# H=-J\sum_{i=0}^{n-2} Z_iZ_{i+1}-h\sum_{i=0}^{n-1}X_i,
# \qquad n=4,\quad J=1,\quad h=0.7
# $$
#
# で与えられます。$Z_i$ と $X_i$ は量子ビット $i$ に作用するパウリ演算子です。第 1 項は隣り合うスピンの相互作用、第 2 項は横磁場を表します。
#
# 4 量子ビットの全計算基底は $2^4=16$ 種類です。比較用に、すべての基底を使ったハミルトニアンを作り、NumPy で最低固有値を求めておきましょう。`EXACT_ENERGY` は、この有限サイズのモデルの基準値です。全空間の行列を作るのは小規模な検証のためです。

# %%
N_QUBITS = 4
J = 1.0
FIELD = 0.7
HAMILTONIAN = qm.Hamiltonian.zero(num_qubits=N_QUBITS)
for index in range(N_QUBITS - 1):
    HAMILTONIAN = HAMILTONIAN - J * qm.Z(index) * qm.Z(index + 1)
for index in range(N_QUBITS):
    HAMILTONIAN = HAMILTONIAN - FIELD * qm.X(index)

ALL_CONFIGURATIONS = tuple(itertools.product((0, 1), repeat=N_QUBITS))
exact_matrix = qm.linalg.subspace_hamiltonian(ALL_CONFIGURATIONS, HAMILTONIAN)
EXACT_ENERGY = float(np.linalg.eigvalsh(exact_matrix)[0])

assert HAMILTONIAN.num_qubits == N_QUBITS
assert HAMILTONIAN.is_hermitian
assert exact_matrix.shape == (2**N_QUBITS, 2**N_QUBITS)
print(f"Exact ground-state energy: {EXACT_ENERGY:+.6f}")

# %% [markdown]
# ## アルゴリズム
#
# **量子選択配置間相互作用法（QSCI）**は、量子状態を計算基底で測定し、出現頻度の高い異なるビット列を選んで部分空間を構築し、その部分空間でハミルトニアンを古典的に対角化することで、基底状態エネルギーの推定値を得る手法です。量子計算と古典計算を組み合わせて実行します。
#
# Kannoら{cite:p}`10.48550/arXiv.2302.11320`によるアルゴリズムの手順は次の通りです。
#
# 1. 入力状態$\lvert \psi_{\mathrm{in}} \rangle$を準備する。例えば、回路のパラメータを調整してエネルギーの低い状態を探すVQEを使う。
# 2. $\lvert \psi_{\mathrm{in}} \rangle$を計算基底（Z基底）で多数回測定する。
# 3. 観測された異なるビット列から、出現頻度が高い上位$K$種類を選び、対応する基底$\{|s_i\rangle\}_{i=1}^{K}$が張る部分空間を使う。
# 4. 選んだ基底で表した有効ハミルトニアン$H^{\mathrm{sub}}_{ij}=\langle s_i|H|s_j\rangle$を構築して古典的に対角化し、その最小固有値をQSCIエネルギーとする。
#
# QSCIの特徴は、選んだ基底で元のハミルトニアンの行列要素を正しく構成する限り、結果が**変分原理（厳密な基底状態エネルギーを下回らない性質）**を満たすことです。量子状態の準備や測定にノイズがあると選ばれる基底が変わることはありますが、部分空間のエネルギーは次の関係を満たします。
#
# $$
# E_{\mathrm{QSCI}}(K)\geq E_{\mathrm{exact}}.
# $$
#
# ここで$E_{\mathrm{exact}}$は、同じハミルトニアンの全空間での基底状態エネルギーです。ただし、得られたエネルギーが厳密値に近いことまで保証されるわけではありません。
#
# すでに選んだ基底を残したまま基底を追加して$K$を増やすと、QSCIエネルギーは増加しません。全計算基底を含めた場合には、厳密な基底状態エネルギーと一致します。有限回の測定では未観測の基底が残ることがあるため、観測できたビット列の範囲で$K$を増やすだけで、必ず厳密解に到達するわけではありません。
#
# QSCIでは、入力状態のパラメータを十分に最適化してから始める必要はありません。ランダムなパラメータを使う状態や、短いVQEで調整した状態も入力として使えます。ただし、少数の基底でよい精度が得られるかは、真の基底状態を表すのに重要なビット列をどれだけ選べるかに依存します。短いVQEは、こうしたビット列を観測しやすくすることを狙いますが、その効果は入力状態と測定結果に依存します。

# %% [markdown]
# ## 実装
#
# ### 入力状態を準備する回路
#
# パラメータを持つ試行状態の回路を、アンザッツと呼びます。ここではすべての量子ビットへの RY ゲートと隣り合う量子ビットへの CX ゲートを 2 回繰り返し、最後にもう 1 層の RY ゲートを適用することにします。RY ゲートは合計 $3\times4=12$ 個、CX ゲートは $2\times3=6$ 個です。よって、RY ゲートの 12 個の回転角を `thetas` に渡します。
#
# 量子操作の結果は更新後の `state` として受け取り、次の操作へ渡します。`qsci_state(...)` は測定前の量子状態を返し、VQE の `Estimator` に渡す状態準備として使います。`qsci_samples(...)` は同じ状態準備を呼び出してから測定し、1 ショット分の 4 ビットを返します。2 つの関数で回路の本体を共有することで、VQE と QSCI のサンプリングで同じ回転角を利用することができます。
#
# この例のカーネルは 4 量子ビット・12 パラメータに固定されています。`N_QUBITS` のみ変更しても、任意の量子ビット数へ拡張できないことに注意しましょう。


# %%
@qm.qkernel
def qsci_state(
    thetas: qm.Param[qm.Vector[qm.Real, 12]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare one runtime-rebindable four-qubit QSCI trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 12]]): Runtime RY angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared trial state.
    """

    state = qm.qubits(4)
    for repetition in range(2):
        for target in range(4):
            state = qm.ry(state, thetas[repetition * 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[8 + target], target=target)
    return state


@qm.qkernel
def qsci_samples(
    thetas: qm.Param[qm.Vector[qm.Real, 12]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Sample one runtime-rebindable QSCI trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 12]]): Runtime RY angles.

    Returns:
        qm.Shot[qm.Bits[4]]: Computational-basis configuration per shot.
    """

    return qm.measure(qsci_state(thetas))


# %% [markdown]
# ### QSCI 用の測定回路を QURI Parts 向けにコンパイルする
#
# `qsci_samples()` を `qm.specialize()` でプログラムとして準備し、`QuriParts()` 向けにコンパイルしましょう。ここで得られる `artifact` は、後で `Qulacs()` を用いてサンプリングするための回路です。`thetas` は `Param` なので、具体的な回転角は実行時に渡します。
#
# このセルで明示的にコンパイルするのは、測定用の `qsci_samples(...)` です。VQE のエネルギー評価には、次の節で `qsci_state(...)` を受け取る `Estimator` を用意します。

# %%
program = qm.specialize(qsci_samples)
sample_endpoint = Qulacs()
artifact = qm.compile(program, to=QuriParts())

# %% [markdown]
# ### ステップ1：短い VQE で入力状態を選ぶ
#
# まず、入力状態の候補を得るために短い VQE を実行しましょう。エネルギーを小さくする状態を探索することで、基底状態を表すのに有用なビット列を観測しやすくなるようにします。ただし、有限回の最適化で必ず良い分布が得られるわけではなく、同じ $K$ に対する QSCI の精度向上も保証されていないことに注意が必要です。QSCI の入力状態は、VQE 以外の方法で準備することも可能です。
#
# ここでは `Estimator(qsci_state, HAMILTONIAN, grouping="qwc")` を用いることにします。同じ局所測定基底を共有できるパウリ項をまとめて測定し、`Qulacs` 上でエネルギーを推定します。この VQE の評価と、次の節で行う QSCI 用の Z 基底測定は別の処理です。
#
# 1 回のエネルギー評価の総ショット数を 128、COBYLA の最大目的関数評価回数を 500 とします。SciPy の COBYLA では、`maxiter` は目的関数の最大評価回数を意味します。

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
expectation_endpoint = Qulacs()
vqe_shots = 2 if docs_test_mode else 128
energy_estimator = Estimator(qsci_state, HAMILTONIAN, grouping="qwc")
cost_history = []


def cost_function(parameters):
    """Estimate the trial-state energy at one variational point."""

    estimate = energy_estimator.estimate(
        on=expectation_endpoint,
        parameters={"thetas": tuple(float(value) for value in parameters)},
        shots=vqe_shots,
        seed=10 + len(cost_history),
    )
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(0)
initial_parameters = rng.uniform(0.0, 2.0 * np.pi, 12)
max_evaluations = 14 if docs_test_mode else 500
optimizer_options = {"maxiter": max_evaluations, "rhobeg": 0.5}
if docs_test_mode:
    optimizer_options["f_target"] = float("inf")

optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options=optimizer_options,
)
if docs_test_mode:
    assert optimization.nfev == 1
optimized_parameters = tuple(float(value) for value in optimization.x)

print(
    f"Optimizer objective estimate = {optimization.fun:+.6f} "
    f"(gap to E_exact: {optimization.fun - EXACT_ENERGY:+.4e})"
)

print("Optimizer success:", optimization.success)
print("Optimizer message:", optimization.message)
print("Objective evaluations:", optimization.nfev)

assert np.isfinite(optimization.fun)
assert len(optimized_parameters) == 12
assert len(cost_history) == optimization.nfev

# %% [markdown]
# #### VQE の推定値と、変分原理の区別
#
# 表示した `optimization.fun` は、最適化器が報告した有限ショットの目的関数値です。返されたパラメータを独立に再測定した値や、状態の厳密な期待値ではありません。有限ショットの推定値は統計的なばらつきによって、基底状態エネルギーを下回る場合もあります。
#
# 一方、VQE が準備する規格化された状態の真の期待値は、同じハミルトニアンの基底エネルギー以上の値となります。これはノイズ後の規格化された状態でも成立します。
#
# QSCI では、測定で選んだ基底を使い、既知の $H$ から部分空間の行列要素を古典的に計算します。測定ノイズや有限ショットは、どの基底が選ばれるかに影響します。選択が変われば精度も変化しますが、正しく構成した部分空間の最低固有値が持つ変分上界は維持されます。この違いを、次の対角化の節で確認しましょう。

# %% [markdown]
# ### ステップ2：Z 基底でビット列をサンプリングする
#
# 最適化で得た `optimized_parameters` を用い、入力状態を計算基底で繰り返し測定しましょう。ここでは VQE と同じ `qsci_state(...)` を含む測定回路を、QURI Parts / Qulacs で 4,000 回測定します。
#
# 測定結果は `(b_0, ..., b_{n-1})` というタプルで表されます。要素 `b_q` は量子ビット $q$ の測定ビットであり、0 は Z の固有値 $+1$、1 は $-1$ に対応します。例えば `(0, 1, 0, 1)` は、量子ビット 0 から順に `0, 1, 0, 1` が観測された結果を表します。
#
# `counts` は、異なるビット列をキー、観測回数を値とする辞書です。同じビット列を繰り返し観測しても、部分空間の次元が増えるわけではありません。観測回数の多い順に並べ、同数の場合はタプルの辞書順で順序を確定します。

# %%
SHOTS = 1 if docs_test_mode else 4_000

record = qm.run(
    artifact,
    on=sample_endpoint,
    parameters={"thetas": optimized_parameters},
    shots=SHOTS,
    seed=2,
)
counts = record.result.counts
configurations = tuple(
    bits
    for bits, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if count > 0
)

if not configurations:
    raise ValueError("No observed configurations are available for QSCI.")


print("sample counts:", counts)
print("Number of distinct observed configurations:", len(configurations))

assert sum(counts.values()) == SHOTS
assert len(configurations) <= 2**N_QUBITS
assert all(len(bits) == N_QUBITS for bits in configurations)

# %% [markdown]
# ### ステップ3 + 4：部分空間を構築して対角化する
#
# 出現回数の多い上位 $K$ 種類のビット列を $s_1, \ldots, s_K$ とし、それらに対応する基底が張る空間を
#
# $$
# \mathcal{S}_K=\operatorname{span}\{|s_1\rangle,\ldots,|s_K\rangle\}
# $$
#
# と書くことにします。部分空間でのハミルトニアンは、$K\times K$ の行列
#
# $$
# H^{\mathrm{sub}}_{ij}=\langle s_i|H|s_j\rangle
# $$
#
# と表されます。`qm.linalg.solve_subspace(configurations[:K], HAMILTONIAN)` を用いてこの部分空間の固有値問題を解き、返り値の最初の要素から固有値を取り出し、その最小値を QSCI エネルギーとします。
#
# 観測回数を重みとしてエネルギーを平均する処理ではありません。観測回数は基底の選択に使い、選ばれた基底の係数は固有値問題
#
# $$
# H^{\mathrm{sub}}c=Ec
# $$
#
# から求めます。得られる状態は $|\psi_{\mathrm{QSCI}}\rangle=\sum_{i=1}^{K}c_i|s_i\rangle$ です。この対角化では、行列の対角成分だけでなく非対角成分も用います。
#
# #### 変分上界の保存
#
# 互いに異なる計算基底は正規直交なので、$\sum_i|c_i|^2=1$ とすれば $|\psi_{\mathrm{QSCI}}\rangle$ も規格化されています。したがって、部分空間の最低固有値は
#
# $$
# E_{\mathrm{QSCI}}(K)
# =
# \min_{\substack{|\phi\rangle\in\mathcal{S}_K\\ \langle\phi|\phi\rangle=1}}
# \langle\phi|H|\phi\rangle
# \geq E_{\mathrm{exact}}
# $$
#
# を満たします。ここで $E_{\mathrm{exact}}$ は、同じハミルトニアンの全空間での最低固有値です。
#
# この議論は、選択した基底に対して元のエルミートな $H$ の行列要素を正しく構成することが前提となっています。行列要素を別途ノイズのある測定で近似する方式や、異なる対称性条件の参照値との比較には、この説明をそのまま適用できないことに注意しましょう。また、実際の浮動小数点計算には丸め誤差が存在するため、以下の `assert` には小さな許容誤差を設けています。

# %% [markdown]
# ## 実行結果

# %%
number_observed = len(configurations)
SUBSPACE_SIZES = tuple(
    sorted(
        {
            size
            for size in (1, 2, 4, 8, min(14, number_observed))
            if 1 <= size <= number_observed
        }
    )
)
energies = tuple(
    float(qm.linalg.solve_subspace(configurations[:size], HAMILTONIAN)[0][0])
    for size in SUBSPACE_SIZES
)
for size, energy in zip(SUBSPACE_SIZES, energies):
    print(
        f"K={size}: E_QSCI={energy:+.12f}, "
        f"gap={energy - EXACT_ENERGY:+.6e}"
    )
print(f"E_exact={EXACT_ENERGY:+.12f}")

assert all(left >= right - 1e-12 for left, right in zip(energies, energies[1:]))
assert all(energy >= EXACT_ENERGY - 1e-12 for energy in energies)

# %% [markdown]
# ### 部分空間のサイズとグラフの読み方
#
# ここでは、同じ `counts` を一度だけ並べ替えたリストの先頭から、基底を追加しています。そのため、$K<K'$ なら $\mathcal{S}_K\subseteq\mathcal{S}_{K'}$ であり
#
# $$
# E_{\mathrm{exact}}\leq E_{\mathrm{QSCI}}(K')\leq E_{\mathrm{QSCI}}(K)
# $$
#
# となります。部分空間を広げると、最小化に使える状態が増えるためです。別々にサンプリングして選んだ包含関係のない部分空間同士では、この単調性は保証されません。
#
# この例では、部分空間のサイズを最大14に制限します。観測された異なるビット列が14種類未満なら、その種類数を最大の$K$とします。未観測の基底は、計算には自動で追加されません。
#
# これらの部分空間のエネルギーを、全16種類の基底から計算した厳密なエネルギーと比較します。小さい$K$で十分な精度が得られるかは入力状態と選択結果に依存し、入力状態のエネルギーより必ず低くなるわけではないことに注意しましょう。

# %%
plt.figure(figsize=(6, 4))
plt.plot(SUBSPACE_SIZES, energies, "-o", label=r"$E_{\mathrm{QSCI}}$")
plt.axhline(EXACT_ENERGY, linestyle="--", label=r"$E_{\mathrm{exact}}$")
plt.xlabel("Subspace size $K$")
plt.ylabel("Energy")
plt.title("QSCI energy in nested subspaces (4-qubit TFIM)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 補足：重複を除く理由と古典計算の負担
#
# `counts` のキーを使うことで、同じビット列を部分空間の基底として重複して登録することを避けることができます。互いに異なる計算基底は正規直交であり、重なり行列は
#
# $$
# S_{ij}=\langle s_i|s_j\rangle=\delta_{ij}
# $$
#
# のように書かれます。そのため、非直交基底で必要となる一般化固有値問題ではなく、通常のエルミート固有値問題を解くだけで良いことがわかります。これは、$H^{\mathrm{sub}}$ 自体の条件数が必ず良くなるという主張ではありません。
#
# 部分空間の行列は $K\times K$ です。$K$ を増やすことで精度の改善が期待できる一方、密行列の格納量は $K^2$ に比例し、対角化の計算量も増えます。そのため、ショット数だけでなく、選択する基底数も重要となります。
#
# 本例の`EXACT_ENERGY`は、Qamomileの`subspace_hamiltonian(...)`で全基底の行列を作ることで求めています。QSCIの計算と同じ行列構築機能を使っているため、この比較だけでは、その機能の実装を独立に検証したことにはなりません。

# %% [markdown]
# ## まとめ
#
# - 入力状態を計算基底で測定し、観測回数の多いビット列から基底を選びます。
# - 選ばれた基底に対し、既知のハミルトニアンの行列要素を古典的に計算し、部分空間で対角化します。
# - 同じハミルトニアンを正しく射影した最低固有値は、全空間の基底エネルギーに対する変分上界になります。ただし、良い精度が得られることまでは保証しません。
# - Qamomile では、状態準備カーネルを VQE とサンプリングで共有し、`solve_subspace(...)` に選んだビット列とハミルトニアンを渡すことができます。
#
# ここでは、全 16 次元の参照計算が可能な小さなモデルで QSCI の手順を確認しました。量子優位性や実機ノイズに対する精度は、この実行だけで検証できません。
