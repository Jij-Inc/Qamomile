# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# tags: [algorithm, primitive, simulation]
# ---
#
# # 鈴木・トロッター分解によるハミルトニアンシミュレーション: ラビ振動
#
# 量子系の時間発展は、ハミルトニアン $H$ を用いて $U(t) = e^{-iHt}$ のように表されます ($\hbar = 1$)。
# この時間発展を量子ゲートの組合せで実装することは、量子コンピュータの代表的な応用の一つです。
# ハミルトニアンが $H = H_z + H_x$ のように複数の項からなり、これらが可換ではない場合、一般には $e^{-i(H_z + H_x) t} \neq e^{-iH_z t} e^{-iH_x t}$ となります。
# そこで、各項による短時間の時間発展を交互に組合せる トロッター・鈴木 積公式を利用します。
# 誤差は、ステップ幅を小さくすれば減り (リー・トロッター、1次)、ステップを対称化すれば減り (ストラング、2次)、対称ステップを鈴木の構成で再帰的に入れ子にすればさらに減ります (任意の偶数次)。\
# 本記事では、1 量子ビットのラビ振動ハミルトニアンでトロッター近似を確認します。
# まず $S_1$ と $S_2$ を扱い、高次の公式には標準ライブラリの `suzuki_trotter` を用います。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import suzuki_trotter

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
FORMULA_STEPS = 1 if docs_test_mode else 4
SAMPLE_SHOTS = 1 if docs_test_mode else 2048

# %% [markdown]
# ## 問題設定
#
# ### ラビ振動ハミルトニアン
#
# 回転座標系で記述される、離調のある駆動2準位系の有効ハミルトニアンを考えましょう。
#
# $$
# H
# = \underbrace{\frac{\omega}{2} Z}_{H_z} + \underbrace{\frac{\Omega}{2} X}_{H_x}.
# $$
#
# ここで$Z$と$X$はそれぞれパウリ演算子、$\omega$は離調角周波数、$\Omega$は駆動強度を表すラビ角周波数です。
# $[Z, X] \neq 0$より、$H$ を $H_z$ と $H_x$ に分割するとトロッター誤差が生じます。


# %%
omega = 1.2
drive = 0.8
evolution_time = 1.5

drift_hamiltonian = 0.5 * omega * qm.Z(0)
drive_hamiltonian = 0.5 * drive * qm.X(0)
hamiltonians = (drift_hamiltonian, drive_hamiltonian)

commutator = qm.commutator(*hamiltonians)
expected_commutator = 0.5j * omega * drive * qm.Y(0)

assert commutator == expected_commutator
print("[H_z, H_x] =", commutator)

# %% [markdown]
# 交換子は $[A, B] = AB - BA$ と定義されます。
# パウリ演算子に対しては $[Z, X] = 2iY$ なので
#
# $$
# [H_z, H_x]
# = \frac{\omega \Omega}{4}[Z, X]
# = i \frac{\omega \Omega}{2} Y
# $$
#
# のようになります。
# この式は、上の `qm.commutator(...)` の結果に一致します。
# 交換子がゼロではないため、各項の時間発展を単純に掛け合わせるだけでは、厳密な時間発展にはなりません。
#
# ### 厳密な時間発展
#
# この場合のハミルトニアンは $2 \times 2$ の比較的小さな行列なので、NumPy を用いて時間発展 $U(t) = e^{-iHt}$ を数値的に計算することができます。
# このセルでは、エルミート行列の固有値分解 $H = V \Lambda V^\dagger$ から
#
# $$
# e^{-iHt}
# = V \mathrm{diag}(e^{-i \lambda_j t}) V^\dagger
# $$
#
# を求めます。
# ここで、$\lambda_j$ はハミルトニアン $H$ の $j$ 番目の固有値 (実数) です。
# 初期状態は $\lvert 0 \rangle$ とし、全発展時間を $T = 1.5$ として、後の比較のために厳密な状態を保存しておきましょう。

# %%
pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

matrix = 0.5 * omega * pauli_z + 0.5 * drive * pauli_x
initial_state = np.array([1.0, 0.0], dtype=np.complex128)

def matrix_evolution(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """エルミート行列に対する厳密な時間発展行列を求める。"""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * time * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T

exact_state = matrix_evolution(matrix, evolution_time) @ initial_state
print("Exact excitation probability P(1):", abs(exact_state[1]) ** 2)

# %% [markdown]
# このモデルの励起確率 (状態 $\lvert 1 \rangle$ を測定する確率) は
#
# $$
# P_1 (t)
# = \left( \frac{\Omega}{E} \right)^2 \sin^2 \left( \frac{Et}{2} \right), \qquad E
# = \sqrt{\omega^2 + \Omega^2}
# $$
#
# のように、解析的に求めることができます。
# 離調 $\omega$ がゼロでない場合、振動の最大確率は 1 より小さくなります。
#
# ## アルゴリズム
#
# 以下では、全発展時間を $T$、分割数を $N$ として、1 ステップの時間幅を $\Delta t = T / N$ となります。
#
# ### $S_1$: 1 次の鈴木・トロッター分解 (リー・トロッター)
#
# もっとも単純な分解は
#
# $$
# S_1(\Delta t)
# = e^{-i H_x \Delta t} e^{-i H_z \Delta t}, \qquad
# U(T)
# \approx [S_1(\Delta t)]^N
# $$
#
# のように表されます。
# この式では $H_z$ の時間発展を適用してから $H_x$ の時間発展を適用します。
# 1 ステップあたりの局所誤差は $O(\Delta t^2)$、$N = T/\Delta t$ ステップにわたる大域的な状態ノルム誤差は $O(\Delta t)$ となります。
#
# ### $S_2$: 2 次の鈴木・トロッター分解 (ストラング分解)
#
# 中央の項を中心にステップを対象化することで、先頭の誤差項を消すことができます。
#
# $$
# S_2(\Delta t)=e^{-iH_z\Delta t/2}
# e^{-iH_x\Delta t}e^{-iH_z\Delta t/2}
# $$
#
# 局所誤差は $O(\Delta t^3)$、大域的な状態ノルム誤差は $O(\Delta t^2)$ となります。
#
# ### 4 次以上の公式: 鈴木の再帰構成
#
# 2 次公式を組合せることで、4 次・6 次・8 次 ... の偶数次の公式を構成することができます。
# これは、$k \geq 2$ において
#
# $$
# S_{2k} (\Delta t)
# = [S_{2k-2} (p_k \Delta t)]^2 S_{2k-2} ((1-4p_k) \Delta t) [S_{2k-2} (p_k \Delta t)]^2, \qquad
# p_k
# = \frac{1}{4 - 4^{1/(2k-1)}}
# $$
#
# と定義されます。
# 例えば 4 次を作る場合は $p_2 \sim 0.4145$、6 次を作る場合には $p_3 \sim 0.3731$ などです。
# 次数を上げるたびに、係数を更新する必要があります。
# 中央の係数 $1 - 4p_k$ は負の値となりますが、これは時間発展を一部逆向きに適用することで、誤差を軽減する効果があります。
# この再帰により、$S_{2k}$ の局所誤差は、$\mathcal{O} (\Delta t^{2k+1})$ となります。
#
# ## 実装
#
# ### Qamomile で 4 次の時間発展を実装する
#
# Qamomile の `suzuki_trotter` を用いることで、積公式の各ゲートを実装することができます。
# 次の例では、ハミルトニアンの項数を 2、公式の次数を 4、時間分割数を 4 に指定しています。
# `hamiltonians` は分割した演算子を指定する静的な入力 (`Const`) であり、`time` は実行時に変更が可能な入力 (`Param`) です。
# 構造は固定しつつ、異なる時間幅の時間発展を試すことができます。


# %%
evolve = suzuki_trotter(term_count=2, order=4, steps=FORMULA_STEPS)

@qm.qkernel
def rabi_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:
    """4次の積公式で1量子ビットを時間発展させ、測定する。"""
    state = qm.qubits(1)
    state = evolve(state, hamiltonians, time)
    return qm.measure(state)

program = qm.specialize(rabi_samples, hamiltonians=hamiltonians)

# %% [markdown]
# ### 精度と論理演算回数のトレードオフ
#
# 公式の次数や時間分割数を増やすほど、通常は近似精度が向上します。
# しかし、その反面、必要な演算回数も増えていきます。
# 今回の 2 項からなるハミルトニアンでは、1 分割あたりのパウリ時間発展の個数は、隣り合う導出の演算をまとめた公式の構造に基づく数え方から 1 次で 2 個、2 次で 3 個、4 次で 11 個などとなります。
# 今の場合、4 次・4 分割のプログラムに対しては 11 x 4 = 44 回のパウリ時間発展となるはずです。
# 実際に、Qamomile の論理リソース解析も、同じ結果となることを確認してみましょう。

# %%
study = qm.resources.logical(program)
evolution_metric = qm.resources.logical_operation_count("quantum.pauli_evolution")
evolution_count = study.metric(evolution_metric).numeric_enclosure()

print("Logical Pauli-evolution count:", evolution_count)
assert evolution_count == (11 * FORMULA_STEPS, 11 * FORMULA_STEPS)

# %% [markdown]
# ## 結果
#
# ### 積公式の収束を確かめる
#
# ここでは、1 次・2 次・4 次の積公式そのものがどのような精度で厳密解に近づくかを、独立した $2\times2$ 行列計算で調べてみましょう。
# 厳密な状態を $\lvert \psi_{\mathrm{exact}} \rangle$、積公式から得られる状態を $\lvert \psi_{\mathrm{approx}}\rangle$ として、純粋状態のフィデリティとインフィデリティを次のように定義します。
#
# $$
# F
# =\lvert \langle\psi_{\mathrm{exact}} \vert \psi_{\mathrm{approx}}\rangle \rvert^2,
# \qquad \epsilon_F
# =1-F.
# $$
#
# 正規化した状態（長さを1にそろえた状態）$a$と$b$では、同じインフィデリティを$\epsilon_F = \|b - \langle a|b\rangle a\|^2$と表せます。
#
# フィデリティは状態の全体位相（global phase）には依存しません。
# 大域的な状態誤差が $\mathcal{O}(\Delta t^p)$ の場合、インフィデリティは高々 $\mathcal{O}(\Delta t^{2p})$ で小さくなります。
#
# | 積公式 | 1ステップの局所誤差 | 大域的な状態誤差 | インフィデリティの上界の次数 |
# |---|---|---|---|
# | 1次 $S_1$ | $O(\Delta t^2)$ | $O(\Delta t)$ | $O(\Delta t^2)$ |
# | 2次 $S_2$ | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$ |
# | 4次 $S_4$ | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$ |
#
# 特定の初期状態では、先頭の誤差項が消えてさらに小さくなる場合があります。
# 今回は、この表に対応する傾きが現れるかを確認してみましょう。

# %%
drift_matrix = 0.5 * omega * pauli_z
drive_matrix = 0.5 * drive * pauli_x

def suzuki_step(order: int, duration: float) -> np.ndarray:
    """NumPyだけで1ステップのSuzuki積公式を構成する。"""
    if order == 1:
        return matrix_evolution(drive_matrix, duration) @ matrix_evolution(
            drift_matrix, duration
        )
    if order == 2:
        half_drift = matrix_evolution(drift_matrix, duration / 2.0)
        return half_drift @ matrix_evolution(drive_matrix, duration) @ half_drift
    if order < 4 or order % 2 != 0:
        raise ValueError("order must be 1 or a positive even integer (at least 2)")

    k = order // 2
    coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))
    outer = suzuki_step(order - 2, coefficient * duration)
    middle = suzuki_step(order - 2, (1.0 - 4.0 * coefficient) * duration)
    return outer @ outer @ middle @ outer @ outer

step_counts = np.array([1] if docs_test_mode else [2, 4, 8, 16], dtype=np.int64)
step_widths = evolution_time / step_counts
infidelities: dict[int, list[float]] = {1: [], 2: [], 4: []}
reference_state = exact_state / np.linalg.norm(exact_state)

for steps in step_counts:
    for order in infidelities:
        step = suzuki_step(order, evolution_time / int(steps))
        approximate_state = np.linalg.matrix_power(step, int(steps)) @ initial_state
        approximate_state /= np.linalg.norm(approximate_state)
        residual = (
            approximate_state
            - np.vdot(reference_state, approximate_state) * reference_state
        )
        error = float(np.vdot(residual, residual).real)
        infidelities[order].append(error)

slopes = {}
if not docs_test_mode:
    for order, errors in infidelities.items():
        slope = np.polyfit(
            np.log(step_widths),
            np.log(np.asarray(errors)),
            1,
        )[0]
        slopes[order] = float(slope)

print("Convergence slopes (expected: order 1: 2, order 2: 4, order 4: 8):", slopes)

# %% [markdown]
# 測定された傾きが、期待される $\Delta t^2$、$\Delta t^4$、$\Delta t^8$ の フィデリティ誤差の振る舞いを再現していることがわかります。\
# 次に、横軸をステップ幅 $\Delta t=T/N$、縦軸をインフィデリティ $1-F$ として、両対数グラフに表示してみましょう。

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for order, values in infidelities.items():
    ax.loglog(step_widths, values, "o-", label=f"Order {order}")

ax.set_xlabel(r"Time-step width $\Delta t = T/N$")
ax.set_ylabel(r"Infidelity $1-F$")
ax.set_title("Suzuki–Trotter convergence (NumPy)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 今の場合、傾きはおよそ 2, 4, 8 となり、理論的な次数と良い一致が確認できます。
#
# ### Qamomile のシミュレータでの実行
#
# ここからは、先ほどの `suzuki_trotter(order=4, steps=4)` で定義した Qamomile の量子プログラムを実行していきましょう。
# `qm.Reference()` を実行先に指定し、総発展時間 $T = 1.5$ で 2048 回測定を行います。
# NumPy の結果は理想的な状態ベクトルから得られる結果ですが、以降で得られるのは有限回の測定による出現回数です。
# そのため、サンプリングによるばらつきが存在します。

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
record = qm.run(
    artifact,
    on=reference,
    parameters={"time": evolution_time},
    shots=SAMPLE_SHOTS,
    seed=2,
)

assert artifact.source_digest.hex() == program.semantic_digest
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("time",)
assert record.result.shots == SAMPLE_SHOTS
assert sum(record.result.counts.values()) == SAMPLE_SHOTS
print("Measurement counts:", record.result.counts)

# %% [markdown]
# Qamomile の測定結果から得た励起確率、そして同じ 4 次公式を NumPy で計算した励起確率と比較してみましょう。

# %%
energy = np.hypot(omega, drive)
exact_probability = (drive / energy) ** 2 * np.sin(energy * evolution_time / 2.0) ** 2

one_step = suzuki_step(4, evolution_time / FORMULA_STEPS)
numerical_state = np.linalg.matrix_power(one_step, FORMULA_STEPS) @ initial_state
numerical_probability = float(abs(numerical_state[1]) ** 2)

observed_probability = record.result.probability((1,))
shots = record.result.shots
sample_standard_error = np.sqrt(observed_probability * (1.0 - observed_probability) / shots)

print(f"Excitation probability (exact):          {exact_probability:.4f}")
print(f"Excitation probability (NumPy, order 4): {numerical_probability:.4f}")
print(f"Excitation probability (Qamomile):       {observed_probability:.4f}")
print(f"Estimated sampling standard error:      {sample_standard_error:.4f}")

# %% [markdown]
# ### 発展: ラビ振動の時間依存性を可視化する
#
# 最後に、時刻 $t$ を変化させたときの励起確率を図示してみましょう。
# 比較のため、厳密解と 1 次・ 4 次公式（ともに時間分割数 4）の NumPy 計算結果も図示します。
# この図は Rabi 振動と近似精度を直感的に示すものであり、上の `qm.Reference()` によるサンプリングとは別の計算です。

# %%
times = np.linspace(0.0, 6.0, 2 if docs_test_mode else 121)
exact_probabilities = (drive / energy) ** 2 * np.sin(energy * times / 2.0) ** 2

first_probabilities = []
fourth_probabilities = []
for current_time in times:
    for order, series in ((1, first_probabilities), (4, fourth_probabilities)):
        step = suzuki_step(order, current_time / 4.0)
        state = np.linalg.matrix_power(step, 4) @ initial_state
        series.append(float(abs(state[1]) ** 2))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, exact_probabilities, label="Exact", linewidth=2)
ax.plot(times, first_probabilities, "--", label="Order 1 (4 steps)")
ax.plot(times, fourth_probabilities, ":", label="Order 4 (4 steps)")
ax.set_xlabel("Evolution time t")
ax.set_ylabel("Excitation probability P(1)")
ax.set_title("Detuned Rabi oscillations (NumPy)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## まとめ
#
# - 時間発展: $H=H_z+H_x$ の各項が非可換な場合、厳密な時間発展を各項の指数関数の単純な積には分解できません。
# - 積公式: 1 次、2 次、4 次と次数を上げると、固定した時間幅での近似精度が一般に改善します。その代わり、必要な演算回数は増えます。
# - 高次の再帰: $S_{2k}$ は $S_{2k-2}$ を 5 つ組み合わせて構成します。係数 $p_k$ は次数ごとに計算します。
# - Qamomile: `suzuki_trotter(term_count=2, order=4, steps=4)` で積公式を定義し、量子カーネルから利用できます。
# - 数値検証: NumPy による状態ベクトルの比較では、インフィデリティ $1-F$ の収束の傾きを確認できます。Qamomile による有限ショットの測定では、励起確率とその統計的なばらつきを調べられます。

# %% [markdown]
#
