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
# tags: [algorithm, primitive, resource-estimation]
# ---
#
# # 量子フーリエ変換（QFT）
#
# 量子フーリエ変換（Quantum Fourier Transformation; QFT）は、離散フーリエ変換の量子版です。量子位相推定やShorのアルゴリズム{cite:p}`10.1109/SFCS.1994.365700`など、量子振幅に埋め込まれた位相情報を使うアルゴリズムで、重要なサブルーチンとして使われます{cite:p}`10.48550/arXiv.quant-ph/0201067`。
#
# このノートブックでは、古典のフーリエ変換から始めてQFT回路を説明し、4量子ビットの周波数推定をQamomileで実装します。スクラッチ実装と組み込みの`qft`関数を比較し、サンプリング結果から主要な周波数を推定して、必要なゲート数を確認します。

# %%
# 最新のQamomileと、このノートブックで使う追加機能をインストールします。
# # !pip install qamomile matplotlib

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import qft

# %% [markdown]
# ## 背景: フーリエ変換
#
# フーリエ変換は、データを周波数成分として表し、各周波数がどれだけ含まれているかを調べる方法です。有限長のベクトルに対してよく使うのが**離散フーリエ変換**（Discrete Fourier Transform、DFT）です。
#
# ベクトル$x = (x_0, x_1, \ldots, x_{N-1})$に対して、このノートブックでは次の正規化を使います。
#
# $$
# y_k = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} x_j e^{2\pi i jk / N},
# \qquad k = 0, 1, \ldots, N-1.
# $$
#
# 出力の添字$k$は周波数成分のインデックスです。このインデックスに対応する角周波数は$2\pi k/N$であり、位相因子$e^{2\pi i jk / N}$により、入力の各位置が異なる位相で足し合わされます。

# %% [markdown]
# ## アルゴリズム
#
# QFTは、DFTと同じ変換を量子状態の振幅に適用します。$N = 2^n$のとき、整数$x$に対応する計算基底状態$\lvert x\rangle$への作用は次の通りです。
#
# $$
# \mathrm{QFT}_N \lvert x \rangle =
# \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N}\lvert k \rangle.
# $$
#
# 一般の量子状態$\lvert\psi\rangle = \sum_{j=0}^{N-1} a_j\lvert j\rangle$に対しては、QFTは線形性により各計算基底状態$\lvert j\rangle$への作用を重ね合わせた状態を返します。
#
# 古典のDFTでは、出力ベクトル全体を得られます。一方で、QFTは量子状態の振幅を変換し、変換後の量子状態を返します。このため、QFTの直後に測定しても、得られるのは変換後の確率分布に従う計算基底の測定結果だけです。ただし、位相推定のようにサブルーチンとして使用する場合は、変換後の位相情報をそのまま利用することができます。
#
# 標準的なQFT回路は、アダマールゲート、制御付き位相回転、最後のスワップで構成されます。$n$個の量子ビットに対して、$O(n^2)$個のゲートで実装できます。位相を表すために、次の2進小数表記を使います。
#
# $$
# [0.x_jx_{j+1}\ldots x_n] =
# \frac{x_j}{2}
# + \frac{x_{j+1}}{2^2}
# + \cdots
# + \frac{x_n}{2^{n-j+1}}
# =
# \sum_{m=j}^{n} \frac{x_m}{2^{m-j+1}}.
# $$
#
# ### ステップ1：対象量子ビットを1つ選ぶ
#
# 最上位ビットから$x_1,x_2,\ldots,x_n$の順に対象量子ビットを処理します。まず$x_1$にアダマールゲートを適用します。
#
# $x_1=0$なら$H\lvert0\rangle = (\lvert0\rangle + \lvert1\rangle)/\sqrt{2}$です。一方、$x_1=1$なら$H\lvert1\rangle = (\lvert0\rangle - \lvert1\rangle)/\sqrt{2}$であり、これは$e^{2\pi i[0.1]} = e^{\pi i} = -1$を使って同じ形にまとめられます。
#
# $$
# \lvert x_1\rangle
# \xrightarrow{H}
# \frac{1}{\sqrt{2}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_1]}\lvert 1\rangle\right).
# $$
#
# ### ステップ2：制御付き位相回転を加える
#
# 残りの量子ビットからの制御付き位相回転により、2進小数に必要なビットが加わります。対象量子ビット$x_j$に対して、アダマールゲートと制御付き位相回転は次の状態を作ります。
#
# $$
# \lvert x_j\rangle
# \longmapsto
# \frac{1}{\sqrt{2}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_jx_{j+1}\ldots x_n]}\lvert 1\rangle\right).
# $$
#
# この位相回転は、一般に$R_k$として次のように書きます。
#
# $$
# R_k =
# \begin{pmatrix}
# 1 & 0 \\
# 0 & e^{2\pi i / 2^k}
# \end{pmatrix}.
# $$
#
# 制御付き$R_k$は、制御量子ビットが$\lvert 1\rangle$のときだけ対象量子ビットにこの回転を適用します。制御位置と対象位置の距離を$d$とすると、QFT回路では制御付き$R_{d+1}$を使います。その回転角は次の通りです。
#
# $$
# \theta = \frac{2\pi}{2^{d+1}} = \frac{\pi}{2^d}.
# $$
#
# ### ステップ3：すべての量子ビットで繰り返す
#
# $x_1,x_2,\ldots,x_n$の順にこのパターンを繰り返します。次に説明する最後のスワップを適用すると、QFTの出力は次の積の形になります。
#
# $$
# \mathrm{QFT}\lvert x_1x_2\ldots x_n\rangle =
# \frac{1}{\sqrt{2^n}}
# \bigotimes_{j=n}^{1}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_jx_{j+1}\ldots x_n]}\lvert 1\rangle\right).
# $$
#
# 4量子ビットの場合は次のようになります。
#
# $$
# \mathrm{QFT}_{16}\lvert x_1x_2x_3x_4\rangle =
# \frac{1}{\sqrt{16}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_4]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_3x_4]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_2x_3x_4]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_1x_2x_3x_4]}\lvert 1\rangle\right).
# $$
#
# ### ステップ4：出力順序を反転する
#
# 標準的なQFT回路では、出力量子ビットの順序が逆になります。最後にスワップ層を置くことで、量子ビットを元の順序に戻します。一部のアルゴリズムではこのスワップを省き、逆順であることを古典側で管理します。
#
# ```{figure} assets/qft_circuit.png
# :alt: q3、q2、q1、q0へのHゲート、6個の制御付き位相回転、最後の2個のスワップからなる4量子ビットQFT
# :width: 720px
#
# n=4のときのQFT回路
# ```

# %% [markdown]
# ## Qamomileでの実装
#
# はじめにQamomileのゲートを使ってQFT回路をスクラッチ実装し、その後、この実装を組み込みの`qft`関数へ置き換えます。
#
# ### 問題設定
#
# $N=16$個のサンプルを使うので、4量子ビットで表せます。この例では、計算基底状態$\lvert j\rangle$の振幅を$s_j$とする量子状態を準備します。ここで、$n=4$、$N=16$、$\omega = e^{2\pi i/16}$です。
#
# $$
# \lvert \psi_f\rangle =
# \sum_{j=0}^{N-1} s_j \lvert j\rangle,
# \qquad
# s_j = \frac{1}{\sqrt{N}} e^{-2\pi i f j/N}
# = \frac{1}{\sqrt{N}}\omega^{-fj},
# \qquad f=5,\quad j=0,1,\ldots,N-1.
# $$
#
# つまり、$e^{-2\pi i f j/N}$の値を、正規化係数$1/\sqrt{N}$付きで$\lvert \psi_f\rangle$の振幅に埋め込んでいます。この状態にQFTを適用すると、各計算基底状態$\lvert j\rangle$からの寄与が足し合わされ、周波数インデックス$f$だけが残ります。
#
# $$
# \mathrm{QFT}_{16}\lvert \psi_f \rangle =
# \frac{1}{16}\sum_{j=0}^{15}\sum_{k=0}^{15}\omega^{j(k-f)}\lvert k\rangle
# = \lvert f\rangle.
# $$
#
# したがって、$f=5$のとき、出力は周波数インデックス$k=5$に集中するはずです。実際に古典DFTを実行して確かめてみましょう。NumPyの`np.fft.ifft`を利用して計算します。

# %%
EXAMPLE_INPUTS = {"num_qubits": 4, "frequency": 5}
dimension = 2 ** EXAMPLE_INPUTS["num_qubits"]
positions = np.arange(dimension)

signal = np.exp(
    -2j * np.pi * EXAMPLE_INPUTS["frequency"] * positions / dimension
) / np.sqrt(dimension)
spectrum = np.fft.ifft(signal, norm="ortho")
expected_spectrum = np.zeros(dimension, dtype=complex)
expected_spectrum[EXAMPLE_INPUTS["frequency"]] = 1.0

print(np.round(np.abs(spectrum), 3))
assert np.allclose(spectrum, expected_spectrum, rtol=1e-10, atol=1e-10)

# %% [markdown]
# ### スクラッチ実装
#
# 次の量子カーネルは、前節で説明した4つのステップをQamomileのゲートで直接実装します。制御付き位相回転は`qubits = qm.cp(qubits, angle, control=control, target=target)`と記述し、所有する量子状態を更新します。外側のループでは、`num_qubits - 1`から`0`へ添字の大きい順に対象量子ビットを選び、内側のループでは添字の小さい量子ビットから必要な制御付き位相回転を適用します。上の図では、対象量子ビットが最下段から最上段へ進みます。最後のループでスワップを適用し、量子ビットを元の順序に戻します。


# %%
@qm.qkernel
def qft_from_scratch(
    qubits: qm.Q[qm.Bits[num_qubits]],
    num_qubits: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[num_qubits]]:
    for offset in range(num_qubits):
        target = num_qubits - 1 - offset
        qubits = qm.h(qubits, target=target)
        for delta in range(target):
            control = target - 1 - delta
            angle = qm.pi / (2 ** (target - control))
            qubits = qm.cp(qubits, angle, control=control, target=target)
    for index in range(num_qubits // 2):
        mirror = num_qubits - index - 1
        qubits = qm.swap(qubits, left=index, right=mirror)
    return qubits


# %% [markdown]
# 次の状態準備用量子カーネルは、振幅に$e^{-2\pi i f j/N}$を符号化します。0番目の量子ビットをサンプルインデックス$j$の最下位ビットとして扱うため、量子ビットが1つ進むごとに位相角を2倍します。2つの周波数推定用量子カーネルは、量子ビット数と周波数を明示的な引数として受け取り、この例の具体的な値は`Const[Int]`の引数として`qm.specialize`で与えます。


# %%
@qm.qkernel
def prepare_frequency_state(
    qubits: qm.Q[qm.Bits[num_qubits]],
    num_qubits: qm.Const[qm.Int],
    frequency: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[num_qubits]]:
    dimension = 2**num_qubits
    qubits = qm.h(qubits)
    for index in range(num_qubits):
        angle = -2 * qm.pi * frequency * (2**index) / dimension
        qubits = qm.p(qubits, angle, target=index)
    return qubits


@qm.qkernel
def qft_frequency_estimator_from_scratch(
    num_qubits: qm.Const[qm.Int], frequency: qm.Const[qm.Int]
) -> qm.Shot[qm.Bits[num_qubits]]:
    qubits = qm.qubits(num_qubits)
    qubits = prepare_frequency_state(qubits, num_qubits, frequency)
    qubits = qft_from_scratch(qubits, num_qubits)
    return qm.measure(qubits)


# %% [markdown]
# ### 組み込み関数：`qft`
#
# Qamomileでは、同じ変換を組み込みの量子カーネル`qft`で利用できます。所有する量子状態`qm.Q[qm.Bits[n]]`とコンパイル時の量子ビット数を受け取り、アダマールゲート、制御付き位相回転、スワップを適用して、変換後の量子ビット列を返します。周波数推定用量子カーネルでは、`qft_from_scratch`の呼び出しを`qft`へ置き換えるだけです。


# %%
@qm.qkernel
def qft_frequency_estimator_with_stdlib(
    num_qubits: qm.Const[qm.Int], frequency: qm.Const[qm.Int]
) -> qm.Shot[qm.Bits[num_qubits]]:
    qubits = qm.qubits(num_qubits)
    qubits = prepare_frequency_state(qubits, num_qubits, frequency)
    qubits = qft(qubits, num_qubits)
    return qm.measure(qubits)


# %% [markdown]
# ### 実行結果
#
# 2つの量子カーネルを同じ`Const`の値で特殊化し、Qamomile組み込みの実行先`qm.Reference()`向けにコンパイルして、`qm.execute`で実行します。測定された周波数分布を比較します。下の変換では、上の状態準備に合わせて`result.counts`の各測定結果の先頭ビットを最下位ビットとして扱います。

# %%
endpoint = qm.Reference()
shots = 512
results = {}
for implementation, kernel in {
    "from scratch": qft_frequency_estimator_from_scratch,
    "built-in qft": qft_frequency_estimator_with_stdlib,
}.items():
    implementation_program = qm.specialize(kernel, **EXAMPLE_INPUTS)
    implementation_artifact = qm.compile(implementation_program, to=endpoint)
    results[implementation] = qm.execute(
        implementation_artifact, on=endpoint, shots=shots, seed=42
    )

probabilities_by_implementation = {}
for implementation, result in results.items():
    probabilities = np.zeros(dimension)
    for outcome, count in result.counts.items():
        frequency_index = sum(bit << index for index, bit in enumerate(outcome))
        probabilities[frequency_index] = count / shots
    probabilities_by_implementation[implementation] = probabilities

fig, ax = plt.subplots(figsize=(7, 3))
indices = np.arange(dimension)
bar_width = 0.4
ax.bar(
    indices - bar_width / 2,
    probabilities_by_implementation["from scratch"],
    width=bar_width,
    color="#2696EB",
    label="from scratch",
)
ax.bar(
    indices + bar_width / 2,
    probabilities_by_implementation["built-in qft"],
    width=bar_width,
    color="#FF6B6B",
    label="built-in qft",
)
ax.set_xlabel("frequency index")
ax.set_ylabel("probability")
ax.set_xticks(indices)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.show()

for implementation, probabilities in probabilities_by_implementation.items():
    estimated_frequency = int(np.argmax(probabilities))
    print(f"{implementation}: estimated frequency = {estimated_frequency}")
    assert estimated_frequency == EXAMPLE_INPUTS["frequency"]
    assert probabilities[EXAMPLE_INPUTS["frequency"]] > 0.95

for result in results.values():
    assert result.shots == shots
    assert sum(result.counts.values()) == shots
assert all(
    isinstance(outcome, tuple) and len(outcome) == EXAMPLE_INPUTS["num_qubits"]
    for result in results.values()
    for outcome in result.counts
)
assert np.allclose(
    probabilities_by_implementation["from scratch"],
    probabilities_by_implementation["built-in qft"],
    rtol=0.0,
    atol=0.0,
)

# %% [markdown]
# ## リソース推定
#
# 厳密なQFT回路では、次のゲートを使います。
#
# - $n$個のアダマールゲート
# - $\frac{n(n - 1)}{2}$個の制御付き位相回転
# - $\left\lfloor n / 2 \right\rfloor$個のスワップ
#
# したがって総ゲート数は$n + \frac{n(n - 1)}{2} + \left\lfloor n / 2 \right\rfloor$であり、$O(n^2)$で増えます。
#
# 量子ビット数は回路構造を決めるため、リソース推定用量子カーネルの明示的な引数にします。これにより、量子カーネルをサイズごとに生成するファクトリ関数を使わず、`qm.resources.logical`による1つのシンボリックな推定へ`evaluate`で具体的な値を代入できます。各指標は下限と上限を返し、両者が一致するときは厳密な個数です。


# %%
@qm.qkernel
def qft_resource_kernel(
    num_qubits: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[num_qubits]]:
    qubits = qm.qubits(num_qubits)
    qubits = qft(qubits, num_qubits)
    return qubits


# %%
qft_study = qm.resources.logical(qft_resource_kernel)
estimate_4 = qft_study.evaluate(num_qubits=4)


def exact_count(study, metric):
    """Return a resource count only when its lower and upper bounds agree."""
    lower, upper = study.metric(metric).numeric_enclosure()
    assert lower is not None and lower == upper
    return int(lower)


resource_metrics = {
    "qubits": qm.resources.LOGICAL_PEAK_QUBITS,
    "total": qm.resources.LOGICAL_GATES,
    "single_qubit": qm.resources.LOGICAL_SINGLE_QUBIT_GATES,
    "two_qubit": qm.resources.LOGICAL_TWO_QUBIT_GATES,
    "rotation_gates": qm.resources.LOGICAL_ROTATION_GATES,
    "clifford_gates": qm.resources.LOGICAL_CLIFFORD_GATES,
}
counts_4 = {
    name: exact_count(estimate_4, metric) for name, metric in resource_metrics.items()
}
print("qubits:", counts_4["qubits"])
print("total gates:", counts_4["total"])
print("single-qubit gates:", counts_4["single_qubit"])
print("two-qubit gates:", counts_4["two_qubit"])
print("rotation gates:", counts_4["rotation_gates"])
print("Clifford gates:", counts_4["clifford_gates"])

assert counts_4["qubits"] == 4
assert counts_4["total"] == 12
assert counts_4["single_qubit"] == 4
assert counts_4["two_qubit"] == 8
assert counts_4["rotation_gates"] == 6
assert counts_4["clifford_gates"] == 6

# %% [markdown]
# 長さ$N$のベクトルを古典DFTで直接計算すると、$O(N^2)$回の演算が必要です。高速フーリエ変換（FFT）を使うと、これを$O(N\log N)$まで減らせます。一方、$N = 2^n$と書くと、厳密QFTは$O(n^2)=O((\log N)^2)$個のゲートで実装できます。したがって、直接計算する古典DFTと比べると、量子ビット数$n$に対して指数的に少ないゲートで同じ変換を状態に適用できます。ただし、測定だけで$N$個すべてのフーリエ係数を読み出せるわけではありません。QFTの利点は、変換後の振幅を後続の量子演算でそのまま使える場合に現れます。
#
# 次のプロットでは、`qm.resources.logical`が返す`qft`のゲート数を上の厳密式と比較します。また、2つの増加率の違いを示すため、$n=3$でQFTのゲート数と一致するように正規化した古典FFTの$O(N\log N)$参照線も表示します。

# %%
qft_qubit_counts = np.arange(3, 10)
qft_total_gates = []

for num_qubits in qft_qubit_counts:
    estimate_n = qft_study.evaluate(num_qubits=int(num_qubits))
    qft_total_gates.append(exact_count(estimate_n, qm.resources.LOGICAL_GATES))

theoretical_qft_gate_counts = [
    num_qubits + num_qubits * (num_qubits - 1) // 2 + num_qubits // 2
    for num_qubits in qft_qubit_counts
]

dimension_counts = 2**qft_qubit_counts
nlogn_reference = dimension_counts * qft_qubit_counts
nlogn_reference = nlogn_reference / nlogn_reference[0] * qft_total_gates[0]

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(
    qft_qubit_counts,
    qft_total_gates,
    marker="o",
    color="#2696EB",
    label="Qamomile qft",
)
ax.plot(
    qft_qubit_counts,
    theoretical_qft_gate_counts,
    linestyle="--",
    color="#FF6B6B",
    label="exact QFT formula",
)
ax.plot(
    qft_qubit_counts,
    nlogn_reference,
    linestyle="--",
    color="#4ECDC4",
    label=r"FFT $O(N\log N)$ (scaled)",
)
ax.set_xlabel(r"number of qubits $n$")
ax.set_ylabel("total gates")
ax.set_yscale("log")
ax.set_xticks(qft_qubit_counts)
ax.grid(alpha=0.3)
ax.legend()
plt.show()

assert qft_total_gates == theoretical_qft_gate_counts
assert len(theoretical_qft_gate_counts) == len(qft_total_gates)
assert len(nlogn_reference) == len(qft_total_gates)

# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを学びました。
#
# - QFTは量子振幅にDFTを適用して変換後の量子状態を返します。この状態を測定して得られるのは、フーリエ係数のベクトル全体ではなくサンプルです。
# - QFT回路はアダマールゲート、制御付き位相回転、スワップから構成でき、Qamomileでは同じ変換を組み込みの`qft`で簡潔に記述できます。
# - $n$量子ビットの厳密QFT回路には$n + n(n-1)/2 + \lfloor n/2\rfloor=O(n^2)$個のゲートが必要であり、Qamomileの`qm.resources.logical`でこの増加を直接評価できます。
