# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# # !pip install "qamomile[qiskit,visualization]"

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

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
# 対象量子ビットを1つずつ処理します。最後の量子ビットにアダマールゲートを適用すると、QFTの出力に現れる最初の因子が得られます。
#
# $x_n=0$なら$H\lvert0\rangle = (\lvert0\rangle + \lvert1\rangle)/\sqrt{2}$です。一方、$x_n=1$なら$H\lvert1\rangle = (\lvert0\rangle - \lvert1\rangle)/\sqrt{2}$であり、これは$e^{2\pi i[0.1]} = e^{\pi i} = -1$を使って同じ形にまとめられます。
#
# $$
# \lvert x_n\rangle
# \xrightarrow{H}
# \frac{1}{\sqrt{2}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_n]}\lvert 1\rangle\right).
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
# アダマールゲートと制御付き位相回転のパターンを繰り返すと、QFTは次の積の形で書けます。
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
# :alt: 標準QFT回路
# :width: 720px
#
# n=4のときのQFTの量子回路。
# ```

# %% [markdown]
# ## Qamomileでの実装
#
# はじめにQamomileのゲートを使ってQFT回路をスクラッチ実装し、その後、この実装を組み込みの`qmc.qft`関数へ置き換えます。
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
# 次の量子カーネルは、前節で説明した4つのステップをQamomileのゲートで直接実装します。制御付き位相回転は`qmc.cp(control, target, angle)`と記述します。外側のループでは右から順に対象量子ビットを選び、内側のループでは左側の量子ビットから必要な位相回転を適用します。最後のループでスワップを適用し、量子ビットを元の順序に戻します。


# %%
@qmc.qkernel
def qft_from_scratch(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    num_qubits = qubits.shape[0]
    for offset in qmc.range(num_qubits):
        target = num_qubits - 1 - offset
        qubits[target] = qmc.h(qubits[target])
        for delta in qmc.range(target):
            control = target - 1 - delta
            angle = math.pi / (2 ** (target - control))
            qubits[control], qubits[target] = qmc.cp(
                qubits[control], qubits[target], angle
            )
    for index in qmc.range(num_qubits // 2):
        mirror = num_qubits - index - 1
        qubits[index], qubits[mirror] = qmc.swap(qubits[index], qubits[mirror])
    return qubits


# %%
qft_from_scratch.draw(qubits=4)

# %% [markdown]
# 次の状態準備用量子カーネルは、振幅に$e^{-2\pi i f j/N}$を符号化します。`qubits[0]`をサンプルインデックス$j$の最下位ビットとして扱うため、量子ビットが1つ進むごとに位相角を2倍します。2つの周波数推定用量子カーネルは、量子ビット数と周波数を明示的な引数として受け取り、この例の具体的な値はコンパイル時の`bindings`で与えます。


# %%
@qmc.qkernel
def prepare_frequency_state(
    qubits: qmc.Vector[qmc.Qubit], frequency: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    num_qubits = qubits.shape[0]
    dimension = 2**num_qubits
    qubits = qmc.h(qubits)
    for index in qmc.range(num_qubits):
        angle = -2 * math.pi * frequency * (2**index) / dimension
        qubits[index] = qmc.p(qubits[index], angle)
    return qubits


@qmc.qkernel
def qft_frequency_estimator_from_scratch(
    num_qubits: qmc.UInt, frequency: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    qubits = qmc.qubit_array(num_qubits, name="qubits")
    qubits = prepare_frequency_state(qubits, frequency)
    qubits = qft_from_scratch(qubits)
    return qmc.measure(qubits)


# %% [markdown]
# ### 組み込み関数：`qft`
#
# Qamomileでは、同じ変換を組み込みの量子カーネル`qmc.qft`で利用できます。`Vector[Qubit]`を受け取り、アダマールゲート、制御付き位相回転、スワップを適用して、変換後の量子ビット列を返します。周波数推定用量子カーネルでは、`qft_from_scratch`の呼び出しを`qmc.qft`へ置き換えるだけです。


# %%
@qmc.qkernel
def qft_frequency_estimator_with_stdlib(
    num_qubits: qmc.UInt, frequency: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    qubits = qmc.qubit_array(num_qubits, name="qubits")
    qubits = prepare_frequency_state(qubits, frequency)
    qubits = qmc.qft(qubits)
    return qmc.measure(qubits)


# %%
qft_frequency_estimator_with_stdlib.draw(**EXAMPLE_INPUTS)

# %% [markdown]
# `draw()`では、組み込みQFTが1つの演算として表示されます。Qiskit向けに出力された回路を確認するには、同じコンパイル時の`bindings`を指定して`to_circuit`で変換します。

# %%
qiskit_circuit = transpiler.to_circuit(
    qft_frequency_estimator_with_stdlib,
    bindings=EXAMPLE_INPUTS,
)
print(qiskit_circuit.draw())

# %% [markdown]
# ### 実行結果
#
# 2つの量子カーネルを同じ`bindings`で実行し、測定された周波数分布を比較します。下の変換では、上の状態準備に合わせて`qubits[0]`を最下位ビットとして扱います。

# %%
backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
shots = 512
results = {}
for implementation, kernel in {
    "スクラッチ実装": qft_frequency_estimator_from_scratch,
    "組み込みqft": qft_frequency_estimator_with_stdlib,
}.items():
    executable = transpiler.transpile(kernel, bindings=EXAMPLE_INPUTS)
    results[implementation] = executable.sample(
        transpiler.executor(backend), shots=shots
    ).result()

probabilities_by_implementation = {}
for implementation, result in results.items():
    probabilities = np.zeros(dimension)
    for outcome, count in result.results:
        frequency_index = sum(bit << index for index, bit in enumerate(outcome))
        probabilities[frequency_index] = count / shots
    probabilities_by_implementation[implementation] = probabilities

fig, ax = plt.subplots(figsize=(7, 3))
indices = np.arange(dimension)
bar_width = 0.4
ax.bar(
    indices - bar_width / 2,
    probabilities_by_implementation["スクラッチ実装"],
    width=bar_width,
    color="#2696EB",
    label="from scratch",
)
ax.bar(
    indices + bar_width / 2,
    probabilities_by_implementation["組み込みqft"],
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
    assert sum(count for _, count in result.results) == shots
assert all(
    isinstance(outcome, tuple) and len(outcome) == EXAMPLE_INPUTS["num_qubits"]
    for result in results.values()
    for outcome, _ in result.results
)
assert np.allclose(
    probabilities_by_implementation["スクラッチ実装"],
    probabilities_by_implementation["組み込みqft"],
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
# 量子ビット数は回路構造を決めるため、リソース推定用量子カーネルの明示的な引数にします。これにより、量子カーネルをサイズごとに生成するファクトリ関数を使わず、1つのシンボリックな推定へ`inputs`で具体的な値を代入できます。


# %%
@qmc.qkernel
def qft_resource_kernel(num_qubits: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qubits = qmc.qubit_array(num_qubits, name="qubits")
    qubits = qmc.qft(qubits)
    return qubits


# %%
estimate_4 = qft_resource_kernel.estimate_resources(inputs={"num_qubits": 4}).simplify()
print("量子ビット数:", estimate_4.qubits)
print("総ゲート数:", estimate_4.gates.total)
print("単一量子ビットゲート数:", estimate_4.gates.single_qubit)
print("2量子ビットゲート数:", estimate_4.gates.two_qubit)
print("回転ゲート数:", estimate_4.gates.rotation_gates)
print("クリフォードゲート数:", estimate_4.gates.clifford_gates)

assert estimate_4.qubits == 4
assert estimate_4.gates.total == 12
assert estimate_4.gates.single_qubit == 4
assert estimate_4.gates.two_qubit == 8
assert estimate_4.gates.rotation_gates == 6
assert estimate_4.gates.clifford_gates == 6

# %% [markdown]
# 長さ$N$のベクトルを古典DFTで直接計算すると、$O(N^2)$回の演算が必要です。高速フーリエ変換（FFT）を使うと、これを$O(N\log N)$まで減らせます。一方、$N = 2^n$と書くと、厳密QFTは$O(n^2)=O((\log N)^2)$個のゲートで実装できます。したがって、直接計算する古典DFTと比べると、$N$に対して指数的に少ないゲートで同じ変換を状態に適用できます。ただし、測定だけで$N$個すべてのフーリエ係数を読み出せるわけではありません。QFTの利点は、変換後の振幅を後続の量子演算でそのまま使える場合に現れます。
#
# 次のプロットでは、`.estimate_resources()`が返す`qmc.qft`のゲート数を上の厳密式と比較します。また、2つの増加率の違いを示すため、$n=3$でQFTのゲート数と一致するように正規化した古典FFTの$O(N\log N)$参照線も表示します。

# %%
qft_qubit_counts = np.arange(3, 10)
qft_total_gates = []

for num_qubits in qft_qubit_counts:
    estimate_n = qft_resource_kernel.estimate_resources(
        inputs={"num_qubits": int(num_qubits)}
    ).simplify()
    qft_total_gates.append(int(estimate_n.gates.total))

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
# - QFT回路はアダマールゲート、制御付き位相回転、スワップから構成でき、Qamomileでは同じ変換を組み込みの`qmc.qft`で簡潔に記述できます。
# - $n$量子ビットの厳密QFT回路には$n + n(n-1)/2 + \lfloor n/2\rfloor=O(n^2)$個のゲートが必要であり、Qamomileの`.estimate_resources()`でこの増加を直接評価できます。
