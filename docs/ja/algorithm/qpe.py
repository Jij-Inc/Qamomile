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
# tags: [algorithm, primitive, resource-estimation]
# ---
#
# # 量子位相推定（QPE）
#
# 量子位相推定（Quantum Phase Estimation; QPE）は、$U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$を満たすユニタリ行列$U$と固有状態$|\psi\rangle$から、固有位相$\phi$を推定するアルゴリズムです。Shorのアルゴリズムなど、ユニタリ行列の固有値に埋め込まれた位相を使うアルゴリズムで中心的なプリミティブとして使われます{cite:p}`10.48550/arXiv.quant-ph/9511026,10.1098/rspa.1998.0164`。
#
# このノートブックでは、QPEの手順をQamomileの量子カーネルとして実装し、組み込みの`qmc.qpe`関数による実装と比較します。さらに、カウント用量子ビット数と推定精度、必要なゲート数の関係を確認します。

# %%
# 最新のQamomileと、このノートブックで使う追加機能をインストールします。
# # !pip install "qamomile[qiskit,visualization]"

# %%
# 数値計算、プロット、シミュレータ、Qamomileのユーティリティを読み込みます。
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 背景: 位相キックバックと量子フーリエ変換
#
# QPEは、制御$U$ゲートによる位相キックバック{cite:p}`10.1098/rspa.1998.0164`と、量子フーリエ変換{cite:p}`10.48550/arXiv.quant-ph/0201067`を組み合わせます。
#
# ### 位相キックバック
#
# 位相キックバックは、ユニタリ行列$U$の固有状態$|\psi\rangle$に制御$U$ゲートを適用したとき、$U$の固有位相が制御量子ビットの相対位相として現れる仕組みです。次を仮定します。
#
# $$
# U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle.
# $$
#
# 1つの制御量子ビットを重ね合わせ状態にし、対象となる量子状態を$|\psi\rangle$に準備します。
#
# $$
# \frac{|0\rangle + |1\rangle}{\sqrt{2}}|\psi\rangle.
# $$
#
# 制御$U$ゲートは、制御量子ビットが$|1\rangle$である成分にだけ$U$を適用します。$|\psi\rangle$は$U$の固有状態なので、その成分に固有位相が現れます。
#
# $$
# \frac{|0\rangle|\psi\rangle + |1\rangle U|\psi\rangle}{\sqrt{2}}
# =
# \frac{|0\rangle + e^{2\pi i\phi}|1\rangle}{\sqrt{2}}|\psi\rangle.
# $$
#
# この変換では、対象となる量子状態は$|\psi\rangle$のまま保たれ、固有位相は制御量子ビットの相対位相に符号化されます。QPEでは、複数の制御量子ビットに対してこの仕組みを意図的に利用します。具体的な符号化方法は「アルゴリズム」節で説明します。
#
# ### 量子フーリエ変換
#
# 量子フーリエ変換（QFT）は、計算基底状態を、その値に応じた相対位相を持つ重ね合わせへ写す変換です。$M=2^m$個の基底状態に対するQFTは次のように定義されます。
#
# $$
# \mathrm{QFT}_M|x\rangle
# = \frac{1}{\sqrt{M}}
#   \sum_{y=0}^{M-1} e^{2\pi i xy/M}|y\rangle.
# $$
#
# これは整数$x$を、計算基底全体にわたる規則的な位相パターンへ写します。このように、値を各計算基底の相対位相として保持した状態を、ここでは「フーリエ符号化された状態」と呼びます。詳しい説明はQFTチュートリアルをご覧ください。逆QFT $\mathrm{QFT}_M^{-1}$は、この変換の逆操作です。$\phi=a/M$が$m$ビットで正確に表せる場合、制御$U^{2^0}, U^{2^1}, \ldots, U^{2^{m-1}}$ゲートによってカウント用量子ビットは
#
# $$
# \frac{1}{\sqrt{M}}\sum_{k=0}^{M-1} e^{2\pi i a k/M}|k\rangle
# = \mathrm{QFT}_M|a\rangle.
# $$
#
# ここに$\mathrm{QFT}_M^{-1}$を適用すると$|a\rangle$が返ります。重要なことは、逆QFTによって位相の情報を計算基底のビット列に変換できることです。
#
# $$
# \underbrace{
# \frac{1}{\sqrt{M}}
# \begin{pmatrix}
# 1 \\
# e^{2\pi i a/M} \\
# \vdots \\
# e^{2\pi i a(M-1)/M}
# \end{pmatrix}
# }_{\text{位相の値のベクトル}}
# \xrightarrow{\mathrm{QFT}_M^{-1}}
# \underbrace{|a\rangle = |a_{m-1}\cdots a_0\rangle}_{\text{計算基底のビット列}}.
# $$

# %% [markdown]
# ## アルゴリズム
#
# QPEは、位相を読み出すための$m$個のカウント用量子ビットと、$U$の固有状態$|\psi\rangle$を保持する対象量子ビットから構成されます。カウント用量子ビットとは、位相を$m$ビットの2進小数として読み出すための補助量子ビットです。対象量子ビットへの入力は、$U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$を満たす固有状態$|\psi\rangle$とし、カウント用量子ビットは$|0\rangle^{\otimes m}$から始めます。ここで$M=2^m$とおきます。
#
# :::{note} 固有状態の重ね合わせ
# より一般には、対象量子ビットへの入力は1つの固有状態に限られません。固有状態の重ね合わせを入力した場合、QPEは入力状態に含まれる各固有状態の重みに応じた確率で、対応する固有位相を測定します。このチュートリアルでは、位相キックバックの式とサンプリング結果を読みやすくするため、既知の固有状態を使います。
# :::
#
# ### ステップ1：カウント用量子ビットを重ね合わせにする
#
# カウント用量子ビットにアダマールゲートを適用します。これにより、$M$個の値の一様重ね合わせが作られます。
#
# $$
# |\Psi_1\rangle =
# H^{\otimes m}|0\rangle^{\otimes m}|\psi\rangle
# =
# \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1}|r\rangle|\psi\rangle.
# $$
#
# ### ステップ2：制御$U^{2^k}$ゲートを適用する
#
# 各カウント用量子ビット$k$を制御量子ビットとして、制御$U^{2^k}$ゲートを適用します。$r=\sum_{k=0}^{m-1} r_k2^k$と書くと、対象の固有状態には位相$e^{2\pi i\phi r}$が乗ります。
#
# $$
# |\Psi_2\rangle =
# \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1}
# e^{2\pi i \phi r}|r\rangle|\psi\rangle.
# $$
#
# ### ステップ3：逆QFTで位相情報をビット列へ変換する
#
# $\phi=a/M$が正確に表せる場合、カウント用量子ビットは$\mathrm{QFT}_M|a\rangle$になります。逆QFTを適用すると$|a\rangle$が返ります。
#
# $$
# |\Psi_2\rangle =
# \left(
#   \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1} e^{2\pi i ar/M}|r\rangle
# \right)|\psi\rangle
# =
# \mathrm{QFT}_M|a\rangle|\psi\rangle,
# $$
#
# $$
# |\Psi_3\rangle =
# \mathrm{QFT}_M^{-1}\mathrm{QFT}_M|a\rangle|\psi\rangle
# =
# |a\rangle|\psi\rangle.
# $$
#
# ### ステップ4：ビット列を測定し、位相を復号する
#
# カウント用量子ビットを測定します。正確に表せる場合、測定結果は$a$となり、このビット列を復号した位相の推定値は次のようになります。
#
# $$
# \tilde{\phi} = \frac{a}{M}.
# $$
#
# $\phi$が$m$ビットで正確に表せない場合、分布は最も近い$m$ビット近似の周辺に集中します。$m$を増やすと表現できる位相の間隔$1/2^m$が小さくなるため、推定値を真の固有位相へ近づけられます。
#
# :::{note} **2進小数と精度**
# 2進小数では、小数部に使えるビット数が位相推定値の精度を左右します。たとえば$\phi=0.6$は2進小数では正確に表せません。2ビットで表すと、最も近い2進小数は
#
# $$
# 0.10_2 = \frac{1}{2^1} + \frac{0}{2^2} = 0.5
# $$
#
# です。一方、3ビットまで使うと表現できる値の間隔が小さくなり、
#
# $$
# 0.101_2
# = \frac{1}{2^1} + \frac{0}{2^2} + \frac{1}{2^3}
# = 0.625
# $$
#
# となって、$0.6$により近い表現が得られます。そのため、カウント用量子ビット数を増やすことは、位相推定値として読み出せる2進小数のビット数を増やすことに対応します。
# :::
#
# ```{figure} assets/qpe_circuit.png
# :alt: カウント用量子ビット、制御Uゲート、逆QFT、測定からなる量子位相推定回路。
# :width: 720px
#
# QPE回路の模式図です。カウント用量子ビットが制御$U$ゲートを通じて位相を蓄積し、逆QFTが蓄積された位相パターンを測定ビット列に変換します。
# ```

# %% [markdown]
# ## Qamomileでの実装
#
# ここでは対角な4x4ユニタリ行列を使います。
#
# $$
# U =
# \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & e^{i\theta_{01}} & 0 & 0 \\
# 0 & 0 & e^{i\theta_{10}} & 0 \\
# 0 & 0 & 0 & e^{i\theta_{11}}
# \end{pmatrix}.
# $$
#
# この行列では、すべての計算基底状態が固有状態です。対象状態$|01\rangle$を準備し、推定したい位相$\theta_{01} / 2\pi$を$0.6$に設定します。$0.6$は有限の2進小数では正確に表せないため、カウント用量子ビットを増やす効果が見えやすくなります。

# %%
# サンプリング設定と対象固有状態を決めます。
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
SHOTS = 512 if docs_test_mode else 4096
SAMPLER_SEED = 42

# 対角なユニタリ行列の位相と対象位相を設定します。
TARGET_PHASE_FRACTION = 0.6
phase_fractions = np.array([0.0, TARGET_PHASE_FRACTION, 0.23, 0.81])
phase_angles = 2 * math.pi * phase_fractions

# 位相をユニタリ行列に変換し、ユニタリ性を確認します。
unitary = np.diag(np.exp(1j * phase_angles))
assert np.allclose(unitary.conj().T @ unitary, np.eye(4))

# 量子カーネルに渡す具体的な位相パラメータを保存します。
PHI_01 = float(phase_angles[1])
PHI_10 = float(phase_angles[2])
PHI_11 = float(phase_angles[3])

# 後続セルで使う問題設定を表示します。
print("phase fractions:", np.round(phase_fractions, 6))
print("target phase fraction:", f"{TARGET_PHASE_FRACTION:.8f}")
print("U =")
print(np.round(unitary, 3))
assert 0.0 <= TARGET_PHASE_FRACTION < 1.0

# %% [markdown]
# ### スクラッチ実装
#
# まず、位相を推定したい4x4ユニタリ行列を定義します。位相ゲート$P(\theta)$は、量子ビットの$|1\rangle$成分に$e^{i\theta}$を掛けます。Qamomileでは、このゲートを`qmc.p(q, theta)`と記述します。そのため、`qmc.p(q[0], phi10)`は対象量子ビットの最初のビットが1であるすべての基底状態に位相$e^{i\theta_{10}}$を与え、`qmc.p(q[1], phi01)`は2つ目のビットが1であるすべての基底状態に位相$e^{i\theta_{01}}$を与えます。この時点で$|11\rangle$には$e^{i(\theta_{10}+\theta_{01})}$が乗っているため、制御位相シフトゲートでは補正分
#
# $$
# \theta_{11} - \theta_{10} - \theta_{01}
# $$
#
# だけを$|11\rangle$に追加します。これにより、順序付き基底$|00\rangle, |01\rangle, |10\rangle, |11\rangle$に対する対角成分は、ちょうど$\operatorname{diag}(1, e^{i\theta_{01}}, e^{i\theta_{10}}, e^{i\theta_{11}})$になります。


# %%
# 位相ゲートで対角な4x4ユニタリ行列を実装します。
@qmc.qkernel
def diagonal_4x4(
    q: qmc.Vector[qmc.Qubit],
    phi01: qmc.Float,
    phi10: qmc.Float,
    phi11: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    # 対象ビットごとに位相を付与します。
    q[0] = qmc.p(q[0], phi10)
    q[1] = qmc.p(q[1], phi01)
    # |11>成分が行列の対角要素と一致するよう補正します。
    q[0], q[1] = qmc.cp(q[0], q[1], phi11 - phi10 - phi01)
    return q


# 具体的な位相パラメータを使って対象ユニタリ行列を描画します。
diagonal_4x4.draw(
    q=2,
    phi01=PHI_01,
    phi10=PHI_10,
    phi11=PHI_11,
    fold_loops=False,
)

# %% [markdown]
# QPEの4つのステップをQamomileの基本操作で実装します。`qmc.control(diagonal_4x4)`で制御$U$ゲートを作り、`power=2**k`で制御$U^{2^k}$ゲートを適用します。逆QFTの後、量子ビット列を小数部だけを持つ`QFixed`へ変換します。`qmc.measure`は、この量子ビット列を測定し、得られたビット列を浮動小数点の位相推定値へ復号します。


# %%
# QPEを基本操作から実装します。
controlled_diagonal_4x4 = qmc.control(diagonal_4x4)


@qmc.qkernel
def qpe_from_scratch(
    counting_bits: qmc.UInt,
    phi01: qmc.Float,
    phi10: qmc.Float,
    phi11: qmc.Float,
) -> qmc.Float:
    counting = qmc.qubit_array(counting_bits, name="counting")
    target = qmc.qubit_array(2, name="target")
    target[1] = qmc.x(target[1])

    for k in qmc.range(counting_bits):
        counting[k] = qmc.h(counting[k])
    for k in qmc.range(counting_bits):
        counting[k], target = controlled_diagonal_4x4(
            counting[k],
            target,
            phi01=phi01,
            phi10=phi10,
            phi11=phi11,
            power=2**k,
        )

    counting = qmc.iqft(counting)
    phase = qmc.cast(counting, qmc.QFixed, int_bits=0)
    return qmc.measure(phase)


# 3個のカウント用量子ビットを持つスクラッチ実装を描画します。
qpe_from_scratch.draw(
    counting_bits=3,
    phi01=PHI_01,
    phi10=PHI_10,
    phi11=PHI_11,
    fold_loops=False,
)

# %% [markdown]
# ### 組み込み関数: `qpe`
#
# `qmc.qpe`関数は、スクラッチ実装で記述したアダマールゲート、制御$U^{2^k}$ゲート、逆QFT、`QFixed`への変換をまとめて適用します。`qmc.qpe`が返す`QFixed`値を`qmc.measure`に渡すと、量子ビット列の測定と固定小数点値への復号が行われ、浮動小数点の位相推定値が得られます。


# %%
# 組み込みのqpe関数を使って同じQPEを実装します。
@qmc.qkernel
def qpe_with_stdlib(
    counting_bits: qmc.UInt,
    phi01: qmc.Float,
    phi10: qmc.Float,
    phi11: qmc.Float,
) -> qmc.Float:
    counting = qmc.qubit_array(counting_bits, name="counting")
    target = qmc.qubit_array(2, name="target")
    target[1] = qmc.x(target[1])

    phase = qmc.qpe(
        target,
        counting,
        diagonal_4x4,
        phi01=phi01,
        phi10=phi10,
        phi11=phi11,
    )
    return qmc.measure(phase)


# 3個のカウント用量子ビットを持つ組み込み実装を描画します。
qpe_with_stdlib.draw(
    counting_bits=3,
    phi01=PHI_01,
    phi10=PHI_10,
    phi11=PHI_11,
    fold_loops=False,
)

# %% [markdown]
# ## 実行結果
#
# 目標位相$0.6$は少ないビット数では正確に表現できないため、QPEは近い$m$ビット近似に対応する分布を返します。ここではカウント用量子ビット数を3から9まで変化させ、組み込みの`qpe`関数による推定値と厳密な位相値を比較します。

# %%
# 対角なユニタリ行列の位相をトランスパイル時に固定します。
phase_bindings = {"phi01": PHI_01, "phi10": PHI_10, "phi11": PHI_11}


# 2つの位相小数の循環距離を計算します。
def phase_distance(a: float, b: float) -> float:
    raw_distance = abs(a - b)
    return min(raw_distance, 1.0 - raw_distance)


# 指定したQPE量子カーネルをトランスパイルし、サンプリングします。
def run_qpe_experiment(qpe_kernel, counting_bits: int) -> float:
    # 量子ビット数と位相をトランスパイル時に固定します。
    bindings = {"counting_bits": counting_bits, **phase_bindings}
    executable = transpiler.transpile(qpe_kernel, bindings=bindings)
    # ドキュメント出力が再現可能になるようシミュレータのseedを固定します。
    executor = transpiler.executor(
        backend=AerSimulator(
            seed_simulator=SAMPLER_SEED + counting_bits,
            max_parallel_threads=1,
        )
    )
    # 測定されたQFixedの位相推定値をサンプリングします。
    sample_result = executable.sample(
        executor,
        shots=SHOTS,
        bindings={},
    ).result()

    # 最も多く観測された復号済み位相を取り出します。
    most_observed_result = max(sample_result.results, key=lambda item: item[1])
    print(most_observed_result)
    qpe_output, most_observed_shots = most_observed_result

    assert 0 < most_observed_shots <= SHOTS
    return qpe_output

# すべてのカウント用量子ビット数でQPEを実行します。
bits = list(range(3, 6) if docs_test_mode else range(3, 10))
estimated_phases = [
    run_qpe_experiment(qpe_with_stdlib, counting_bits) for counting_bits in bits
]
# スクラッチ実装と組み込み関数が同じ位相推定値を返すことを確認します。
scratch_phase = run_qpe_experiment(qpe_from_scratch, bits[0])
assert np.isclose(scratch_phase, estimated_phases[0])
# 推定値を厳密な位相と比較し、循環距離で誤差を計算します。
exact_phases = [TARGET_PHASE_FRACTION for _ in bits]
phase_errors = [
    phase_distance(estimated_phase, TARGET_PHASE_FRACTION)
    for estimated_phase in estimated_phases
]

# 位相推定値を厳密な位相と並べてプロットします。
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(bits, estimated_phases, marker="o", color="#2696EB", label="QPE estimate")
ax.plot(bits, exact_phases, linestyle="--", color="#DB4D3F", label="exact phase")
ax.set_xlabel("counting qubits")
ax.set_ylabel("phase fraction")
ax.set_xticks(bits)
phase_margin = max(0.02, max(phase_errors) + 0.01)
ax.set_ylim(
    max(0.0, TARGET_PHASE_FRACTION - phase_margin),
    min(1.0, TARGET_PHASE_FRACTION + phase_margin),
)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

# この例でカウント用量子ビットを増やすと推定が改善することを確認します。
assert phase_errors[-1] < phase_errors[0]
for counting_bits, phase_error in zip(bits, phase_errors):
    assert phase_error <= 1 / 2**counting_bits

# %% [markdown]
# ## リソース推定
#
# 前のセクションでは、カウント用量子ビットを増やすと精度が上がることを確認しました。ここでは、上の`run_qpe_experiment()`で使ったものと同じQPE量子カーネルに、`estimate_resources()`を直接適用します。この推定には、アダマールゲート、`qmc.qpe`による制御$U^{2^k}$ゲート、逆QFT、最後の固定小数点測定が含まれます。

# %%
# 具体的なカウント用量子ビット数を代入し、総ゲート数を集めます。
resource_gate_counts: list[int] = []
for counting_bits in bits:
    bindings = {"counting_bits": counting_bits, **phase_bindings}
    concrete_estimate = qpe_with_stdlib.estimate_resources(inputs=bindings).simplify()
    resource_gate_counts.append(int(concrete_estimate.gates.total))

# 最初の推定点に合わせて2^mの参照曲線を作ります。
scaling_reference = [
    resource_gate_counts[0] * 2 ** (counting_bits - bits[0])
    for counting_bits in bits
]

# リソース推定とmに対する指数的な傾向をプロットします。
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(
    bits,
    resource_gate_counts,
    marker="o",
    color="#2696EB",
    label="QPE gate count",
)
ax.plot(
    bits,
    scaling_reference,
    linestyle="--",
    color="#DB4D3F",
    label=r"$O(2^m)$",
)
ax.set_xlabel(r"counting qubits $m$")
ax.set_ylabel("total gates")
ax.set_yscale("log")
ax.set_xticks(bits)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

# 直接推定したゲート数が、この範囲で増加することを確認します。
assert all(
    later > earlier
    for earlier, later in zip(resource_gate_counts, resource_gate_counts[1:])
)

# %% [markdown]
# プロットは、QPEの精度を上げるために必要なカウント用量子ビット数$m$と、そのときの総ゲート数を、$2^m$に比例する参照線とともに示しています。目標とする加法誤差を$\epsilon$とすると、表現できる位相の間隔はおおよそ
#
# $$
# 2^{-m} \lesssim \epsilon
# $$
#
# を満たす必要があります。そのため、必要なカウント用量子ビット数は
#
# $$
# m = O\!\left(\log\frac{1}{\epsilon}\right)
# $$
#
# です。
#
# 今回の実装では、制御$U^{2^k}$ゲートを$U$の反復として構成します。したがって、$m$個のカウント用量子ビットに対する$U$の適用回数は
#
# $$
# \sum_{k=0}^{m-1} 2^k = 2^m - 1
# =
# O\!\left(\frac{1}{\epsilon}\right)
# $$
#
# です。つまり、カウント用量子ビット数は$1/\epsilon$に対して対数的ですが、$U$の適用回数は$O(1/\epsilon)$で増えます{cite:p}`10.1017/CBO9780511976667`。
#
# このような場合、ゲート数は先ほど見た通り$O(1/\epsilon)$となり，高精度なQPEを実行するには効率的ではありません。より一般には、あるユニタリ行列$V$を実行するために必要なゲート数を$G(V)$とすると、QPE本体に必要なゲート数は次のように書けます。
#
# $$
# G_{\mathrm{QPE}}(m)
# =
# \sum_{k=0}^{m-1} G\!\left(\mathrm{controlled}\text{-}U^{2^k}\right)
# + O(m^2)
# $$
#
# :::{note}
# $m$個の量子ビットに対する逆QFTは、$O(m^2)$個のゲートに分解できます。詳しくはQFTのチュートリアルを参照してください。
# :::
#
# しかし、実際にはこの見積もりは大きく変わる可能性があります。上式はQPE本体のゲート数を表しており、初期状態の準備コストは含みません。また、各制御$U^{2^k}$ゲートのゲート数は、その実装方法に依存します。以下では、この2点を分けて説明します。
#
# 1. **制御$U$ゲートの実装コスト**
#
#    QPEは、Shorのアルゴリズムでは位数発見に、量子系のシミュレーションでは時間発展演算子の固有位相からエネルギーを推定するために使われます。Shorのアルゴリズムでは、モジュラー乗算に対応する制御$U^{2^k}$ゲートを問題構造から構成できるため、$U$を指数回反復する必要はありません。ハミルトニアンシミュレーションでも、ハミルトニアンの形やシミュレーション手法によっては、制御された時間発展を多項式程度のリソースで実装できる場合があります。そのため、QPEのリソースを見積もるときは、制御$U^{2^k}$ゲートを$U$の反復として数えるのか、直接合成した回路として数えるのか、あるいは問題固有の算術回路やシミュレーション回路として与えるのかを明示する必要があります。
#
# 2. **初期状態の準備コスト**
#
#    QPEで特定の固有位相が得られる確率は、準備した状態と対応する固有状態との重なりの二乗で決まります。十分な重なりを持つ近似状態の準備には非自明な量子回路が必要になる場合があるため、実際の応用ではそのゲート数も別途見積もる必要があります。
#
# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを学びました。
#
# - QPEは、制御$U^{2^k}$ゲートによる位相キックバックと逆QFTを使って、ユニタリ行列の固有位相を2進数として読み出します。
# - Qamomileでは、`qmc.control`、`qmc.iqft`、`qmc.cast`を使ってQPEを構成でき、組み込みの`qmc.qpe`で同じ処理を簡潔に記述できます。
# - $m$個のカウント用量子ビットによる位相分解能は$O(2^{-m})$ですが、必要なゲート数は制御$U^{2^k}$ゲートの実装方法と初期状態の準備方法に依存します。
