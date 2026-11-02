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
# このノートブックでは、QPEの手順をQamomileの量子カーネルとして実装し、組み込みの`phase_estimation`関数による実装と比較します。さらに、カウント用量子ビット数と推定精度、必要なゲート数の関係を確認します。

# %%
# 最新のQamomileと、このノートブックで使う追加機能をインストールします。
# # !pip install "qamomile[qiskit,visualization]"

# %%
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

import qamomile as qm
from qamomile.qiskit import Aer
from qamomile.stdlib import inverse_qft, phase_estimation
from qamomile.visualization import draw

endpoint = Aer()

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
# これは整数$x$を、計算基底全体にわたる規則的な位相パターンへ写します。このように、値を各計算基底の相対位相として保持した状態を、ここでは「フーリエ符号化された状態」と呼びます。Qamomileでの応用例は、[多次元QFTの記事](multidimensional_qft.ipynb)を参照してください。逆QFT $\mathrm{QFT}_M^{-1}$は、この変換の逆操作です。$\phi=a/M$が$m$ビットで正確に表せる場合、制御$U^{2^0}, U^{2^1}, \ldots, U^{2^{m-1}}$ゲートによってカウント用量子ビットは
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
# まず、位相を推定したい4x4ユニタリ行列を定義します。位相ゲート$P(\theta)$は、量子ビットの$|1\rangle$成分に$e^{i\theta}$を掛けます。Qamomileでは、このゲートを`qm.p(q, theta, target=index)`と記述します。そのため、`qm.p(q, PHI_10, target=0)`は対象量子ビットの最初のビットが1であるすべての基底状態に位相$e^{i\theta_{10}}$を与え、`qm.p(q, PHI_01, target=1)`は2つ目のビットが1であるすべての基底状態に位相$e^{i\theta_{01}}$を与えます。この時点で$|11\rangle$には$e^{i(\theta_{10}+\theta_{01})}$が乗っているため、制御位相シフトゲートでは補正分
#
# $$
# \theta_{11} - \theta_{10} - \theta_{01}
# $$
#
# だけを$|11\rangle$に追加します。これにより、順序付き基底$|00\rangle, |01\rangle, |10\rangle, |11\rangle$に対する対角成分は、ちょうど$\operatorname{diag}(1, e^{i\theta_{01}}, e^{i\theta_{10}}, e^{i\theta_{11}})$になります。


# %%
@qm.qkernel
def diagonal_4x4(q: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Apply the diagonal unitary with the phase constants defined above."""
    q = qm.p(q, PHI_10, target=0)
    q = qm.p(q, PHI_01, target=1)
    q = qm.cp(q, PHI_11 - PHI_10 - PHI_01, control=0, target=1)
    return q


display(HTML(draw(qm.specialize(diagonal_4x4), loops="expanded").to_html()))

# %% [markdown]
# QPEの4つのステップをQamomileの基本操作で実装します。`qm.control(diagonal_4x4, counting, target, at=k)`は、カウント用量子ビット$k$を使って制御$U$ゲートを適用します。これを`2**k`回繰り返すことで、制御$U^{2^k}$ゲートを適用します。逆QFTの後、リトルエンディアン（下位ビットが先頭の順序）の量子ビット列を上位ビットが先頭の固定小数点の桁順へ反転し、`qm.as_fixed`で整数部を持たない`UFixed[0, counting_bits]`として扱います。`qm.measure`は、この量子ビット列を測定し、得られたビット列を数値の位相推定値へ復号します。すべての量子値を消費するため、保持されている対象レジスターも測定します。その測定結果は返しません。


# %%
@qm.qkernel
def qpe_from_scratch(
    counting_bits: qm.Const[qm.Int],
) -> qm.Shot[qm.UFixed[0, counting_bits]]:
    """Read the eigenphase using explicit phase kickback and inverse QFT."""
    counting = qm.qubits(counting_bits)
    target = qm.prepare((0, 1))
    for k in range(counting_bits):
        counting = qm.h(counting, target=k)

    system = (counting, target)
    for k in range(counting_bits):
        for _ in range(2**k):
            counting, target = system
            counting, target = qm.control(diagonal_4x4, counting, target, at=k)
            system = (counting, target)

    counting, target = system
    counting = inverse_qft(counting, counting_bits)
    for left in range(counting_bits // 2):
        counting = qm.swap(counting, left=left, right=counting_bits - left - 1)
    phase = qm.as_fixed(counting, integer_bits=0)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


display(
    HTML(
        draw(
            qm.specialize(qpe_from_scratch, counting_bits=3), loops="expanded"
        ).to_html()
    )
)

# %% [markdown]
# ### 組み込み関数: `phase_estimation`
#
# `phase_estimation`関数は、スクラッチ実装で記述したアダマールゲート、制御$U^{2^k}$ゲート、逆QFT、固定小数点値への変換をまとめて適用します。この関数は、位相と保持されている対象レジスターの両方を返します。返された位相を`qm.measure`に渡すと、量子ビット列の測定と固定小数点値への復号が行われ、数値の位相推定値が得られます。


# %%
@qm.qkernel
def qpe_with_stdlib(
    counting_bits: qm.Const[qm.Int],
) -> qm.Shot[qm.UFixed[0, counting_bits]]:
    """Read the same eigenphase with the standard-library QPE component."""
    target = qm.prepare((0, 1))
    phase, target = phase_estimation(diagonal_4x4, target, counting_bits)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


display(
    HTML(
        draw(
            qm.specialize(qpe_with_stdlib, counting_bits=3), loops="expanded"
        ).to_html()
    )
)

# %% [markdown]
# ## 実行結果
#
# 目標位相$0.6$は少ないビット数では正確に表現できないため、QPEは近い$m$ビット近似に対応する分布を返します。ここではカウント用量子ビット数を3から9まで変化させ、組み込みの`phase_estimation`関数による推定値と厳密な位相値を比較します。


# %%
def phase_distance(a: float, b: float) -> float:
    """Compute cyclic distance between two phase fractions."""
    raw_distance = abs(a - b)
    return min(raw_distance, 1.0 - raw_distance)


def run_qpe_experiment(qpe_kernel, counting_bits: int) -> float:
    """Specialize, compile, and sample one QPE register width."""
    program = qm.specialize(qpe_kernel, counting_bits=counting_bits)
    artifact = qm.compile(program, to=endpoint.target)
    sample_result = qm.run(
        artifact,
        on=endpoint,
        shots=SHOTS,
        seed=SAMPLER_SEED + counting_bits,
    ).result
    assert sample_result.shots == SHOTS
    assert sum(sample_result.values.values()) == SHOTS
    most_observed_result = max(
        sample_result.values.items(), key=lambda item: (item[1], -item[0])
    )
    print(most_observed_result)
    qpe_output, most_observed_shots = most_observed_result
    assert 0 < most_observed_shots <= SHOTS
    return float(qpe_output)


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
# 前のセクションでは、カウント用量子ビットを増やすと精度が上がることを確認しました。ここでは、上の`run_qpe_experiment()`で使ったものと同じ、量子ビット数を固定したQPE量子カーネルに`qm.resources.logical`を適用します。この推定には、アダマールゲート、`phase_estimation`による制御$U^{2^k}$ゲート、逆QFT、測定が含まれます。
# プロットでは、`LOGICAL_GATES`の下限を使います。このプログラムの制御演算と随伴演算（逆演算）について、nativeの推定結果には未確定の上限が残るため、これは正確なゲート数ではなく、保証される下限を示します。制御$U$の呼び出し回数は正確に求まり、$2^m-1$であることを別途確認します。測定回数は`LOGICAL_MEASUREMENTS`に別途計上されます。エンジン向けのコンパイルや最適化によって、出力回路のゲート数は変わる場合があります。リソース推定と上下限の詳細は、[リソース推定のチュートリアル](../tutorial/04_resource_estimation.ipynb)を参照してください。

# %%
resource_gate_counts: list[int] = []
for counting_bits in bits:
    program = qm.specialize(qpe_with_stdlib, counting_bits=counting_bits)
    logical_study = qm.resources.logical(program)
    controlled_queries = logical_study.metric(
        qm.resources.logical_operation_count("qcore.control_call")
    ).numeric_enclosure()
    assert controlled_queries == (2**counting_bits - 1, 2**counting_bits - 1)
    gate_bounds = logical_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
    assert gate_bounds[0] is not None
    assert logical_study.metric(
        qm.resources.LOGICAL_MEASUREMENTS
    ).numeric_enclosure() == (counting_bits + 2, counting_bits + 2)
    resource_gate_counts.append(int(gate_bounds[0]))

# 最初の推定点に合わせて2^mの参照曲線を作ります。
scaling_reference = [
    resource_gate_counts[0] * 2 ** (counting_bits - bits[0]) for counting_bits in bits
]

# リソース推定とmに対する指数的な傾向をプロットします。
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(
    bits,
    resource_gate_counts,
    marker="o",
    color="#2696EB",
    label="QPE logical gate lower bound",
)
ax.plot(
    bits,
    scaling_reference,
    linestyle="--",
    color="#DB4D3F",
    label=r"$O(2^m)$",
)
ax.set_xlabel(r"counting qubits $m$")
ax.set_ylabel("logical gate lower bound")
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
# プロットは、QPEの精度を上げるために必要なカウント用量子ビット数$m$と、そのときの総ゲート数の下限を、$2^m$に比例する参照線とともに示しています。目標とする加法誤差を$\epsilon$とすると、表現できる位相の間隔はおおよそ
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
# $m$個の量子ビットに対する逆QFTは、$O(m^2)$個のゲートに分解できます。これが上式の$O(m^2)$項に相当します。
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
# - Qamomileでは、`qm.control`、`inverse_qft`、`qm.as_fixed`を使ってQPEを構成でき、組み込みの`phase_estimation`で同じ処理を簡潔に記述できます。
# - $m$個のカウント用量子ビットによる位相分解能は$O(2^{-m})$ですが、必要なゲート数は制御$U^{2^k}$ゲートの実装方法と初期状態の準備方法に依存します。
