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
# tags: [algorithm, block-encoding, amplitude-amplification]
# ---
#
# # ブロックエンコーディングの事後選択の増幅
#
# ブロックエンコーディングは、非ユニタリ演算子 $A$ をより大きなユニタリ演算子に埋め込み、
# シグナル（補助）レジスタが全て $\lvert 0 \rangle$ と測定されることを条件とする
# **事後選択**によって適用します。この測定の成功確率は次式で与えられます。
#
# $$
# a \;=\; \frac{\lVert A\lvert\psi\rangle\rVert^2}{\alpha^2},
# $$
#
# ここで $\alpha$ は部分正規化定数です。フィルタが強くなるほど $a$ は小さくなり、
# 必要な試行回数は $1/a$ に比例して増加します。振幅増幅はこれを $O(1/\sqrt{a})$ に削減します。
#
# 回路を出力する関数は`qmc.amplitude_amplification`の1つだけで、**スケジュール**オブジェクトを
# `(signal, system)` のABI（`LCUBlockEncoding`が既に採用しているもの）を持つ任意の状態準備に
# 適用します。利用できるスケジュールは2種類です。
#
# * `qmc.standard_amplification_schedule(k)` — Brassard-Hoyer-Mosca-Tappの反復演算子
#   $Q = -A S_0 A^\dagger S_{\mathrm{good}}$ です
#   {cite:p}`10.1090/conm/305/05215`。$k$ 回適用した後の成功確率は
#   $\sin^2\!\big((2k+1)\theta\big)$（ただし $\sin^2\theta = a$）と厳密に一致します。
#   $a$ が既知であれば最適ですが、未知の場合は行き過ぎ（オーバーシュート）が生じます。
# * `qmc.fixed_point_amplification_schedule(lambda, delta)` — Yoder-Low-Chuangの
#   不動点スケジュールです {cite:p}`10.1103/PhysRevLett.113.210501`。必要なのは
#   *下界* $\lambda \le a$ のみで、$a \ge \lambda$ を満たす**すべての** $a$ に対して
#   成功確率 $1 - \delta^2$ 以上を
#   $L = O\!\big(\log(2/\delta)/\sqrt{\lambda}\big)$ 回のクエリで達成します。
#   オーバーシュートは起こりません。
#
# スケジュールは自身の解析的な保証も保持します。`schedule.query_count`、
# `schedule.guaranteed_success_probability`、`schedule.success_probability(a)` により、
# ゲートを1つも出力する前にその系列の振る舞いを知ることができます。
#
# このページでは、ブロックエンコーディングを構築して増幅前の成功確率を測定し、
# 2つのスケジュールを適用して解析解と比較します。

# %%
# pipで最新のQamomileをインストールしてください！
# # !pip install "qamomile[qiskit,visualization]"

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
executor = transpiler.executor()

# %% [markdown]
# ## 成功確率を調整できるブロックエンコーディング
#
# ここでは2項からなる対角ブロックエンコーディングを使用します。
#
# $$
# A \;=\; c_0 I + c_1 Z_0, \qquad \alpha = |c_0| + |c_1|.
# $$
#
# システムレジスタが全て $\lvert 0 \rangle$ の状態に作用させると $c_0 + c_1$ 倍となるため、
# 事後選択の成功確率は $a = |c_0 + c_1|^2 / \alpha^2$ となります。$c_1$ を $-c_0$ に
# 近づけるほどフィルタは鋭くなり成功確率は小さくなります。これは事後選択のコストが
# 問題となる領域そのものです。


# %%
def build_encoding(single: float) -> qmc.IsingZBlockEncoding:
    """Build the two-term diagonal encoding for a given Z coefficient.

    Args:
        single (float): Coefficient of the ``Z_0`` word; the identity
            coefficient is fixed to one.

    Returns:
        qmc.IsingZBlockEncoding: Two-term diagonal block encoding.
    """
    return qmc.ising_z_block_encoding(
        {(): 1.0 + 0.0j, (0,): complex(single)},
        num_system_qubits=1,
    )


def analytic_success_probability(single: float) -> float:
    """Return the unamplified post-selection success probability.

    Args:
        single (float): Coefficient of the ``Z_0`` word.

    Returns:
        float: Probability of measuring the signal register all-zero.
    """
    return abs(1.0 + single) ** 2 / (1.0 + abs(single)) ** 2


def zero_projector(num_qubits: int) -> qm_o.Hamiltonian:
    """Return the projector onto an all-zero register.

    Args:
        num_qubits (int): Register width.

    Returns:
        qm_o.Hamiltonian: Product of ``(I + Z_i) / 2`` over every qubit.
    """
    projector = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    identity = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    for index in range(num_qubits):
        projector = projector * (0.5 * (identity + qm_o.Z(index)))
    return projector


SINGLE = -0.6
encoding = build_encoding(SINGLE)
initial_probability = analytic_success_probability(SINGLE)
theta = math.asin(math.sqrt(initial_probability))

print(f"subnormalization alpha = {encoding.normalization:.4f}")
print(f"signal qubits          = {encoding.num_signal_qubits}")
print(f"system qubits          = {encoding.num_system_qubits}")
print(f"success probability a  = {initial_probability:.4f}")
print(f"expected repetitions   = {1 / initial_probability:.1f}")

# %% [markdown]
# ## 回路上での成功確率の測定
#
# 成功事象は「シグナルレジスタが全て $\lvert 0 \rangle$ であること」なので、その確率は
# シグナルレジスタ上の射影演算子 $\prod_i (I + Z_i)/2$ の期待値に等しくなります。
# 2つの増幅関数はいずれも外側のqkernelに回路を出力する通常のPython関数なので、
# スケジュールごとにqkernelを構築します。


# %%
def standard_kernel(iterations: int) -> qmc.QKernel:
    """Build a kernel measuring success after standard amplification.

    Args:
        iterations (int): Number of amplification rounds ``k``.

    Returns:
        qmc.QKernel: Kernel returning the all-zero-signal probability.
    """

    @qmc.qkernel
    def kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero-signal probability.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Success probability after amplification.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            encoding.unitary,
            qmc.standard_amplification_schedule(iterations),
        )
        return qmc.expval(signal, observable)

    return kernel


def run(kernel: qmc.QKernel) -> float:
    """Transpile and evaluate a success-probability kernel.

    Args:
        kernel (qmc.QKernel): Kernel returning an expectation value.

    Returns:
        float: Estimated success probability.
    """
    executable = transpiler.transpile(
        kernel,
        bindings={"observable": zero_projector(encoding.num_signal_qubits)},
    )
    return float(executable.run(executor).result())


measured = run(standard_kernel(0))
print(f"measured (k=0): {measured:.6f}")
print(f"analytic      : {initial_probability:.6f}")
assert abs(measured - initial_probability) < 1e-8

# %% [markdown]
# ## 標準的な振幅増幅：厳密だがオーバーシュートする
#
# 1回の適用ごとに、状態はgood/badの2次元平面内で $2\theta$ だけ回転します。
# そのため成功確率は $\sin^2\!\big((2k+1)\theta\big)$ を描き、1に近づいた後に再び減少します。
# `qmc.amplitude_amplification_iteration_count`はピークに最も近い $k$ を返しますが、
# その計算には $a$ の値が必要です。

# %%
optimal_k = qmc.amplitude_amplification_iteration_count(initial_probability)
rounds = list(range(13))
standard_measured = [run(standard_kernel(k)) for k in rounds]
standard_analytic = [math.sin((2 * k + 1) * theta) ** 2 for k in rounds]

for k, observed, expected in zip(rounds, standard_measured, standard_analytic):
    assert abs(observed - expected) < 1e-8

print(f"optimal k = {optimal_k}")
print(f"success at optimal k     = {standard_measured[optimal_k]:.4f}")
print(f"success at k = {rounds[-1]:<2}        = {standard_measured[-1]:.4f}")

# %%
figure, axes = plt.subplots(figsize=(7, 4))
axes.plot(rounds, standard_analytic, "-", color="0.6", label=r"$\sin^2((2k+1)\theta)$")
axes.plot(rounds, standard_measured, "o", color="tab:blue", label="simulated")
axes.axvline(
    optimal_k, color="tab:red", linestyle="--", label=f"optimal $k$={optimal_k}"
)
axes.axhline(initial_probability, color="0.8", linestyle=":", label="unamplified")
axes.set_xlabel("amplification rounds $k$")
axes.set_ylabel("success probability")
axes.set_title("Standard amplitude amplification overshoots")
axes.set_ylim(0.0, 1.05)
axes.legend(loc="lower right", fontsize=8)
figure.tight_layout()
plt.show()

# %% [markdown]
# シミュレーション結果は解析曲線と完全に一致し、$k=3$ を超えると成功確率は再び低下します。
# したがって $a$ そのものではなく $a$ の*下界*から $k$ を選ぶことは安全ではありません。
# 適用回数が多すぎることは、少なすぎることと同じくらい問題になります。
#
# ## 不動点振幅増幅：下界だけで十分
#
# Yoder-Low-Chuangのスケジュールは、各ラウンドの2つの完全な反転を位相付きの反転に置き換えます。
# $\lambda \le a$ と目標の失敗許容度 $\delta$ を与えると、
# `qmc.fixed_point_amplification_schedule`は $a \ge \lambda$ を満たすすべての $a$ に対して
# 成功確率が $1 - \delta^2$ 以上となる最小の奇数クエリ回数 $L$ を導出し、
# それを保持するスケジュールオブジェクトを返します。

# %%
BOUND = 0.05
TOLERANCE = 0.3

schedule = qmc.fixed_point_amplification_schedule(BOUND, TOLERANCE)
query_count = schedule.query_count

print(f"lower bound lambda  = {BOUND}")
print(f"failure tolerance   = {TOLERANCE}")
print(f"guarantee           = {schedule.guaranteed_success_probability:.2f}")
print(f"query count L       = {query_count}")
print(f"phase schedule      = {len(schedule.phases)} phases "
      f"({schedule.num_rounds} rounds)")
print(f"covers a >= {schedule.admissible_success_probability_bound:.4f}")

# %% [markdown]
# スケジュールは $L - 1$ 個の位相を保持し、`(good_1, zero_1, good_2, zero_2, ...)`
# の順に並びます。good部分空間の位相はゼロ状態の位相を逆順に並べたものであり、
# この位相整合条件によって系列が不動点となります。
#
# なお $L$ を次の奇数に切り上げることで、実際には指定した $\lambda$ よりも小さい
# `admissible_success_probability_bound` 以上のすべての $a$ が保証の対象になります。


# %%
def scheduled_kernel(
    target: qmc.IsingZBlockEncoding,
    applied_schedule: qmc.AmplificationSchedule,
) -> qmc.QKernel:
    """Build a kernel measuring success after one amplification schedule.

    Args:
        target (qmc.IsingZBlockEncoding): Block encoding to amplify.
        applied_schedule (qmc.AmplificationSchedule): Schedule to apply.

    Returns:
        qmc.QKernel: Kernel returning the all-zero-signal probability.
    """

    @qmc.qkernel
    def kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero-signal probability.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Success probability after amplification.
        """
        signal = qmc.qubit_array(target.num_signal_qubits, "signal")
        system = qmc.qubit_array(target.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            target.unitary,
            applied_schedule,
        )
        return qmc.expval(signal, observable)

    return kernel


fixed_point_measured = run(scheduled_kernel(encoding, schedule))
print(f"fixed-point success = {fixed_point_measured:.4f}")
print(f"schedule predicts   = {schedule.success_probability(initial_probability):.4f}")
print(f"guarantee           = {schedule.guaranteed_success_probability:.4f}")
assert fixed_point_measured >= schedule.guaranteed_success_probability - 1e-8
assert abs(
    fixed_point_measured - schedule.success_probability(initial_probability)
) < 1e-8

# %% [markdown]
# ## 不動点の保証は許容範囲全体で成立する
#
# $\lambda$ のみから構築した1つのスケジュールを、真の成功確率が1桁以上にわたって
# 異なる複数のブロックエンコーディングに適用します。比較のため、$\lambda$ に合わせて
# 適用回数を調整した標準スケジュール（$a$ を知らない利用者が取りうる最善の選択）も
# 併せて評価します。

# %%
tuned_k = qmc.amplitude_amplification_iteration_count(BOUND)
tuned_schedule = qmc.standard_amplification_schedule(tuned_k)
sweep = [-0.6, -0.5, -0.45, -0.3, -0.15, 0.0]
true_probabilities = []
fixed_point_values = []
tuned_values = []

for coefficient in sweep:
    swept = build_encoding(coefficient)
    probability = analytic_success_probability(coefficient)
    true_probabilities.append(probability)

    executable = transpiler.transpile(
        scheduled_kernel(swept, schedule),
        bindings={"observable": zero_projector(swept.num_signal_qubits)},
    )
    fixed_point_values.append(float(executable.run(executor).result()))
    tuned_values.append(tuned_schedule.success_probability(probability))

for probability, value in zip(true_probabilities, fixed_point_values):
    assert value >= schedule.guaranteed_success_probability - 1e-8

print(f"standard schedule tuned for lambda: k = {tuned_k}")
for coefficient, probability, fixed, tuned in zip(
    sweep, true_probabilities, fixed_point_values, tuned_values
):
    print(
        f"  c1={coefficient:+.2f}  a={probability:.4f}"
        f"  fixed-point={fixed:.4f}  tuned-standard={tuned:.4f}"
    )

# %%
order = np.argsort(true_probabilities)
sorted_probabilities = np.asarray(true_probabilities)[order]
figure, axes = plt.subplots(figsize=(7, 4))
axes.plot(
    sorted_probabilities,
    np.asarray(fixed_point_values)[order],
    "o-",
    color="tab:green",
    label=f"fixed point ($L$={query_count})",
)
axes.plot(
    sorted_probabilities,
    np.asarray(tuned_values)[order],
    "s--",
    color="tab:orange",
    label=rf"standard, $k$={tuned_k} tuned for $\lambda$",
)
axes.axhline(
    schedule.guaranteed_success_probability,
    color="tab:green",
    linestyle=":",
    label=r"guarantee $1-\delta^2$",
)
axes.axvline(BOUND, color="0.7", linestyle=":", label=r"bound $\lambda$")
axes.set_xscale("log")
axes.set_xlabel("true initial success probability $a$")
axes.set_ylabel("amplified success probability")
axes.set_title("The fixed-point schedule never overshoots")
axes.set_ylim(0.0, 1.05)
axes.legend(loc="lower left", fontsize=8)
figure.tight_layout()
plt.show()

# %% [markdown]
# 不動点スケジュールの曲線は $\lambda$ より大きい全領域で保証値を上回ります。
# 一方、調整済みの標準スケジュールは $\lambda$ 付近でのみ正しく、$a$ が大きくなると
# 目標を通り過ぎて回転してしまいます。
#
# ## コスト
#
# 振幅増幅が追加するのは、すべての反転で再利用されるクリーンな補助量子ビット1個のみです。
# したがって回路全体では `num_signal_qubits + num_system_qubits + 1` 個の量子ビットを使用します。
# クエリ回数 $L = 2\ell + 1$ の不動点系列は、状態準備を $\ell + 1$ 回、その逆を $\ell$ 回、
# および $\ell$ 組の位相回転を適用します。

# %%
executable = transpiler.transpile(
    scheduled_kernel(encoding, schedule),
    bindings={"observable": zero_projector(encoding.num_signal_qubits)},
)
circuit = executable.quantum_circuit
operation_names = [
    instruction.operation.name
    for instruction in circuit.data
    if instruction.operation.name != "measure"
]
half_count = schedule.num_rounds

print(f"circuit qubits          = {circuit.num_qubits}")
print(
    f"register qubits + 1     = {encoding.num_signal_qubits + encoding.num_system_qubits + 1}"
)
print(f"encoding applications   = {operation_names.count('ising_z_block_encoding')}")
print(
    f"good phase rotations    = {operation_names.count('amplitude_amplification_good_phase_rotation')}"
)
print(
    f"zero phase rotations    = {operation_names.count('amplitude_amplification_zero_phase_rotation')}"
)

assert circuit.num_qubits == (
    encoding.num_signal_qubits + encoding.num_system_qubits + 1
)
assert operation_names.count("ising_z_block_encoding") == half_count + 1
assert (
    operation_names.count("amplitude_amplification_good_phase_rotation") == half_count
)

# %% [markdown]
# ## まとめ
#
# * ブロックエンコーディングの事後選択は確率 $a = \lVert A\lvert\psi\rangle\rVert^2/\alpha^2$
#   で成功し、エンコードするフィルタが鋭くなるほどこの値は小さくなります。
# * `qmc.standard_amplification_schedule(k)`は $\sin^2((2k+1)\theta)$ を厳密に実現しますが、
#   最適となるのは $a$ が既知の場合のみで、そうでなければオーバーシュートします。
# * `qmc.fixed_point_amplification_schedule(lambda, delta)`は下界 $\lambda \le a$ のみを
#   必要とし、$a \ge \lambda$ を満たすすべての $a$ に対して
#   $L = O(\log(2/\delta)/\sqrt{\lambda})$ 回のクエリで成功確率 $\ge 1 - \delta^2$ を
#   保証します。
# * すべてのスケジュールが`success_probability(a)`に応答するため、ゲートを出力する前に
#   結果を予測できます。
# * どちらも追加コストは補助量子ビット1個のみで、`(signal, system)` のブロックエンコーディングABIを
#   持つ任意の状態準備を受け付けます。全て $\lvert 0 \rangle$ でないシステム入力を
#   エンコーディングと組み合わせて`AmplificationPreparation`にまとめるには
#   `qmc.amplification_preparation`を使用してください。この記述子はシリアライズ済みの
#   テンプレートにバインドすることもできます。
