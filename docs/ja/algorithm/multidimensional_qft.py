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
# tags: [algorithm, chemistry, simulation]
# ---
#
# # 多次元量子フーリエ変換によるナノシート物質特性推定
#
# 量子フーリエ変換（QFT）は、量子アルゴリズムの中核的なプリミティブとして広く使われており、さまざまな応用が提案されています。
# しかし従来のQFTに基づく構成では、入力データが2のべき乗のグリッドサイズに自然に定義されない場合に制約がありました。
# そこで本記事では、[](https://doi.org/10.1039/d4cp04399e)で提案された、任意のグリッドサイズの入力に対する多次元量子フーリエ変換の実装についてまとめました。
# 多次元量子フーリエ変換実装を通して、Qamomileの使い方を学ぶことができます。
#
# ## 背景
#
# ### 問題: ナノシート材料の特性評価
#
# ナノシート材料は、幅広い化学応用のために注目を集めています。
# その物理化学的な特性は、その厚さに応じて劇的に変化します。
# そのため、ナノシート材料の多様な特性を自在に制御するには、厚さを制御した合成や、ナノシート材料の層数の特定が極めて重要となります。
# 層数を特定する有望な手段として、透過型電子顕微鏡（TEM）による観測があります。
# TEMを用いた観測からは、透過イメージとSelected Area Electron Diffraction（SAED:制限視野電子回折）パターンの両方が得られます。
# そこで、計算機によるナノシート材料のSAEDパターンシミュレーションが、有望なアプローチとなっています。
#
# その手法の一つとして、古典的な高速フーリエ変換（FFT）により離散フーリエ変換を計算する方法が活用されてきました。
# FFTは非常に高速な手法であり、その計算量は$\mathcal{O} (N \log_2 N)$（$N$:グリッド数）であることが知られています。
# しかし、$N = 2^n$ のようにグリッド数が2のべき乗で表されているとすると、$\mathcal{O} (n 2^n)$ となり、指数 $n$ の増大に伴って計算困難になることがわかります。
# この有望な解決手法として、近年では量子コンピュータによる量子フーリエ変換（QFT）が脚光を浴びるようになりました。
# 厳密な標準QFTユニタリは、$n$量子ビット上で$\mathcal{O} (n^2)$ゲートにより実装できます。このことが、QFTを量子フーリエ変換プリミティブとして利用する動機になります。ただし、状態準備と読み出しのコストは別途考慮する必要があります。
# そこでこの論文では多次元QFTの応用可能性を探求し、その利用可能な入力データをより詳細に特性評価することを目的として、QFTを用いた$\mathrm{CeO_2}$ナノシートの近似的なSAEDパターンのシミュレーションを行いました。
#
# ### 多次元QFT
#
# QFTは、Shorのアルゴリズムや量子位相推定のコア技術としてよく知られています。
# QFTに関連する拡張として、[](https://doi.org/10.48550/arXiv.2301.13835)が挙げられます。
# この研究では、既知のQFT回路をベースに、多次元QFTを実現する効率的な回路を導出しました。
#
# ```{figure} assets/multidimensional_qft_01.png
# :width: 100%
# :alt: quantum circuit for implementing a d-dimensional Fourier transform from Pfeffer (2023)
#
# [](https://doi.org/10.48550/arXiv.2301.13835)より、$d$次元フーリエ変換を実装するための量子回路図。
# ```
#
# 上図の${\mathrm{initialize}}(\vert v \rangle)$は、入力データに応じて量子状態を準備するサブルーチンを表しています。
# 状態準備の手法にはいくつかありますが、以降に示す実装では[Möttönen状態準備ルーチンによる振幅エンコーディング](mottonen_amplitude_encoding)を用いています。
# しかしこの手法は、各次元サイズが $N_i = 2^{n_i}$ でなければならないという制限がありました。
# そこで、[](https://doi.org/10.1039/d4cp04399e)ではこの制限を解消する手法を提案しました。
# Qamomileには、1次元QFTが標準で備わっています。
# Qamomileの機能を活用することで、多次元QFTも容易に実装することが可能です。
#

# %% [markdown]
# ## インストール
#
# Qamomile distributionをインストールします。
#
# ```bash
# pip install qamomile
# ```

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
from __future__ import annotations

import numpy as np

import qamomile as qm
from qamomile.stdlib import amplitude_encoding, qft

# %% [markdown]
# ## 小さな2次元入力のencoding
#
# 次のlittle-endianなflat indexを持つ$4\times4$ gridを使います。
#
# $$
# j=x+4y
# $$
#
# 入力は$x$軸に沿って一様で、$y=1$に局在しています。したがって正規化後の状態は
#
# $$
# \frac{1}{2}\sum_{x=0}^{3}|x\rangle|1\rangle
# $$
#
# です。$x$軸のQFTでは逆格子座標0が得られます。局在した$y$座標は、4つすべての逆格子値に等しいsupportを持つ状態へ変換されます。

# %%
side = 4
input_amplitudes = tuple(
    1.0 if side <= index < 2 * side else 0.0 for index in range(side * side)
)
prepare = amplitude_encoding(input_amplitudes)

assert len(input_amplitudes) == 16
assert sum(value * value for value in input_amplitudes) == 4.0

# %% [markdown]
# `amplitude_encoding`は有限vectorを正規化し、Rustが導出したexact-phase座標を通常のunitary Qamomile関数に閉じ込めます。入力がSDKのstate-preparation objectになることはありません。

# %% [markdown]
# ## アルゴリズム: 任意の周期性をもつ多次元入力のためのQFT
#
# 結晶やナノシートの実空間構造は、一般に2のべき乗のグリッド上でサンプリングされるとは限りません。
# その格子周期性は、任意の整数個のグリッド点に対応しえます。
# そこで、[](https://doi.org/10.1039/d4cp04399e)では、結晶の周期性をそのまま量子状態に符号化するための、前処理手法を提案しました。
#
# ### 領域の切り捨て/ゼロ埋め
#
# 実際に得られる結晶構造のデータは、2のべき乗のグリッド上に定義されているとは限りません。
# ある次元方向のサンプリング点数が$2^p$より大きい場合、最初の$2^p$点のみを抜き出すことで、QFTを適用可能にします。
# またある次元方向のサンプリング点数が$2^p$より小さい場合、その前後にゼロを追加することでも、QFTを適用可能にします。
#
# ### フラットトップ窓関数の適用
#
# 領域の切り捨て/ゼロ埋めだけを適用したデータの周波数解析を行うと、本来のスペクトルの横に余分な成分が生じます。
# これをサイドローブと呼び、これは領域を切り取ったことで生まれる人工的なスペクトルです。
# そこでこれを抑えるために、切り捨て/ゼロ埋めされた入力データ区間の両端を滑らかにゼロに近づける重みづけ関数 (窓関数) を導入しましょう。
# この論文では、次のようなフラットトップ関数を用いています。
#
# $$
# w_5(n) = \sum_{k=0}^4 (-1)^k a_k \cos \left( \frac{2\pi k n}{N}\right)
# $$
#
# ここで$(a_0, a_1, a_2, a_3, a_4) = (0.1881, 0.36923, 0.28702, 0.13077, 0.02488)$であり、$N$は考えている次元方向の点の総数、$n$はその点の位置$(n=0, 1, \dots, N-1)$を表します。
# 2次元の場合、$w_5(x) w_5(y)$ のような積を、入力に掛け合わせます。


# %%
def pad_center(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-pad a two-dimensional array to one power-of-two shape."""

    padding = []
    for size, target in zip(array.shape, target_shape, strict=True):
        total = target - size
        padding.append(((total + 1) // 2, total // 2))
    return np.pad(array, tuple(padding))


def flat_top(size: int) -> np.ndarray:
    """Return the five-term flat-top window used by the workflow."""

    coefficients = np.array([0.1881, 0.36923, 0.28702, 0.13077, 0.02488])
    positions = np.arange(size)
    return sum(
        (-1) ** order * coefficient * np.cos(2.0 * np.pi * order * positions / size)
        for order, coefficient in enumerate(coefficients)
    )


source_side = 6
x, y = np.meshgrid(
    np.arange(source_side),
    np.arange(source_side),
    indexing="ij",
)
periodic_data = np.cos(np.pi * x / 2.0) * np.cos(np.pi * y / 2.0)
padded_data = pad_center(periodic_data, (8, 8))
window_2d = flat_top(8)[:, None] * flat_top(8)[None, :]
windowed_data = padded_data * window_2d

assert periodic_data.shape == (6, 6)
assert padded_data.shape == (8, 8)
assert windowed_data.shape == (8, 8)

# %% [markdown]
# ## Qamomileによる実装
#
# ### 軸の分割、変換、再結合
#
# `split_at(state, 2)`は`Q[Bits[4]]`を消費し、source順に2つのownedな`Q[Bits[2]]`を返します。各componentは同じ`qft` callによって消費されます。その後、`concat`が両方の結果を消費して`Q[Bits[4]]`を再構築します。


# %%
@qm.qkernel
def two_axis_qft_state() -> qm.Q[qm.Bits[4]]:
    """Prepare the grid and apply one QFT to each logical axis.

    Returns:
        qm.Q[qm.Bits[4]]: Reciprocal-grid state with two two-bit axes.
    """

    state = prepare(qm.qubits(4))
    x_axis, y_axis = qm.split_at(state, 2)
    x_axis = qft(x_axis, 2)
    y_axis = qft(y_axis, 2)
    return qm.concat(x_axis, y_axis)


@qm.qkernel
def two_axis_qft_samples() -> qm.Shot[qm.Bits[4]]:
    """Measure the two-dimensional reciprocal coordinate.

    Returns:
        qm.Shot[qm.Bits[4]]: Two little-endian two-bit coordinates.
    """

    return qm.measure(two_axis_qft_state())


# %% [markdown]
# slice aliasや重複するviewは存在しません。returnされないcomponent、componentの複製、誤った幅の式、owned stateの2回目の使用は、target lowering前に型またはownership errorになります。

# %% [markdown]
# ## 同じprogramのinspectionと推定
#
# specialized QCore programは1つの`qft`定義と2本のcall edgeを保持します。構造的なfactorとtensorは検証済みのzero-cost reshapeになります。多次元QFT opcodeやbackend algorithm caseはありません。

# %%
program = qm.specialize(two_axis_qft_samples)
inspection = program.inspect()
qft_symbols = {
    function.symbol
    for function in program.functions
    if function.symbol.rsplit(".", 1)[-1] == "qft"
}

assert len(qft_symbols) == 1
assert inspection.count(f"callee='{next(iter(qft_symbols))}'") == 2
assert "quantum.multidimensional_qft" not in inspection

study = qm.resources.logical(program)
assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (4, 4)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cp")
).numeric_enclosure() == (2, 2)
assert study.metric(
    qm.resources.logical_operation_count("quantum.swap")
).numeric_enclosure() == (2, 2)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# QFT部分の厳密なresourceは、実行用にloweringされるものと同じcall graphから導出されます。状態準備のworkも別に見えるため、backend-native boxによってlogical estimateが0になることはありません。

# %% [markdown]
# ### サーベイにあるQFT scalingの再現
#
# [](https://doi.org/10.48550/arXiv.2310.03011)のSection 12では、$N=2^n$
# 振幅に対する教科書的QFTは$O(\log N)$量子ビットと$O(\log^2 N)$量子ゲートを
# 使うとされています。その回路は$n$個のHadamard、$n(n-1)/2$個のcontrolled
# phase shift、最後の$\lfloor n/2\rfloor$個のswapからなります。ここでは集約値の
# slopeをfitするのではなく、この有限サイズの係数を直接検証します。compilerが
# QFT本体を1度解析する間、幅はsymbolicなままです。`evaluate`は下のtableに
# 表示する幅を代入するだけです。


# %%
@qm.qkernel
def surveyed_qft(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:
    """Resource sweepに使う教科書的QFT familyを構築する。"""

    return qm.measure(qft(qm.qubits(width), width))


qft_scaling_study = qm.resources.logical(surveyed_qft)


def qft_resource_row(width: int) -> tuple[int, int, int, int]:
    """1つのregister幅について教科書的QFTの厳密なresource countを返す。"""

    qft_study = qft_scaling_study.evaluate(width=width)
    hadamards = qft_study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure()
    controlled_phases = qft_study.metric(
        qm.resources.logical_operation_count("quantum.cp")
    ).numeric_enclosure()
    swaps = qft_study.metric(
        qm.resources.logical_operation_count("quantum.swap")
    ).numeric_enclosure()
    peak = qft_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
    assert hadamards[0] == hadamards[1]
    assert controlled_phases[0] == controlled_phases[1]
    assert swaps[0] == swaps[1]
    assert peak[0] == peak[1]
    return hadamards[0], controlled_phases[0], swaps[0], peak[0]


qft_scaling = {width: qft_resource_row(width) for width in range(2, 9)}
for width, (hadamards, controlled_phases, swaps, peak) in qft_scaling.items():
    assert hadamards == width
    assert controlled_phases == width * (width - 1) // 2
    assert swaps == width // 2
    assert peak == width

print("QFT scaling (n: H, CP, SWAP, peak qubits):", qft_scaling)

# %% [markdown]
# Qamomileは有限サイズの教科書的回路を厳密に再現します。したがって主要な
# controlled-phase数は$\Theta(n^2)=\Theta(\log^2 N)$、peak widthは
# $n=\log_2N$となり、サーベイと一致します。これはlogical circuitの比較です。
# サーベイのcaveatと同様、amplitude loadingとtomographyはQFT本体の外です。

# %% [markdown]
# ## compileと実行
#
# `ResultABI`は4つの論理bitをsource順に保持します。backend固有のbit orderによらず、先頭2bitがlittle-endianな$x$座標、末尾2bitがlittle-endianな$y$座標です。

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
record = qm.run(
    artifact,
    on=reference,
    shots=32,
    seed=2,
)

observed_y = {
    sum(bit << index for index, bit in enumerate(bits[2:]))
    for bits in record.result.counts
}

assert record.result.shots == 32
assert sum(record.result.counts.values()) == 32
assert all(bits[:2] == (0, 0) for bits in record.result.counts)
assert observed_y == set(range(side))
print("reciprocal-grid samples:", record.result.counts)

# %% [markdown]
# すべてのsampleで$k_x=0$となり、一様な$x$入力と一致します。基底に局在した$y$入力は一様なFourier振幅を持つため、4つすべての$k_y$値が現れます。この構造的な性質により、大規模な統計比較なしで軸の入れ替えやresult bitの反転を検出できます。

# %% [markdown]
# ## 結果
#
# ### classical methodであるFFTとの比較
#
# 同じcompositionをpadded 6×6 periodic inputに適用します。以下のhelperは1つのnormalized amplitude vectorをcloseし、6量子ビットを2つの3量子ビット軸へfactorし、再利用可能なQFTを両方へ適用します。classical inverse FFTはこのQFTと同じpositive-exponent conventionを使います。残るphase conventionはmagnitudeに影響しません。


# %%
def make_grid_qft(amplitudes: tuple[float, ...]) -> qm.QKernel:
    """Build one fixed six-qubit two-axis QFT program."""

    prepare_grid = amplitude_encoding(amplitudes)

    @qm.qkernel
    def grid_qft() -> qm.Shot[qm.Bits[6]]:
        state = prepare_grid(qm.qubits(6))
        x_axis, y_axis = qm.split_at(state, 3)
        x_axis = qft(x_axis, 3)
        y_axis = qft(y_axis, 3)
        return qm.measure(qm.concat(x_axis, y_axis))

    return grid_qft


def classical_probabilities(array: np.ndarray) -> np.ndarray:
    """Return normalized two-dimensional inverse-FFT probabilities."""

    normalized = array / np.linalg.norm(array)
    transformed = np.fft.ifft2(normalized) * np.sqrt(array.size)
    return np.abs(transformed) ** 2


def sampled_grid(array: np.ndarray, seed: int) -> np.ndarray:
    """Execute the QFT program and return an 8×8 sampled distribution."""

    kernel = make_grid_qft(tuple(array.reshape(-1)))
    result = qm.run(
        qm.specialize(kernel),
        on=qm.Reference(),
        shots=512,
        seed=seed,
    ).result
    probabilities = np.zeros((8, 8))
    for bits, count in result.counts.items():
        kx = sum(bit << index for index, bit in enumerate(bits[:3]))
        ky = sum(bit << index for index, bit in enumerate(bits[3:]))
        probabilities[kx, ky] += count / result.shots
    return probabilities


padded_classical = classical_probabilities(padded_data)
padded_quantum = sampled_grid(padded_data, seed=5)
windowed_classical = classical_probabilities(windowed_data)
windowed_quantum = sampled_grid(windowed_data, seed=7)

padded_distance = np.abs(padded_quantum - padded_classical).sum()
windowed_distance = np.abs(windowed_quantum - windowed_classical).sum()
print("distribution distances:", padded_distance, windowed_distance)
assert padded_distance < 0.5
assert windowed_distance < 0.5
print("padded peak:", np.unravel_index(padded_quantum.argmax(), (8, 8)))
print("windowed peak:", np.unravel_index(windowed_quantum.argmax(), (8, 8)))

# %% [markdown]
# sampled QFT distributionはfinite-shot variationの範囲でclassical FFT referenceと一致します。windowを適用するとmain lobeは広がりますがboundary由来のsidelobeが局在し、nanosheet workflowのpreprocessing tradeoffを再現します。

# %% [markdown]
# ## まとめ
#
# ここでは、2のべき乗でない入力データに対する前処理手法を開発した論文と、Qamomileを用いたQFTの実装方法についてご紹介しました。
# 以下にこのページで紹介した重要な情報をまとめます。
#
# * Qamomileには標準でQFTを実装する機能が備わっています。
# * 入力が2のべき乗でない場合、領域を切り捨て/ゼロ埋めすることでQFTを適用可能にします。
# * 領域の切り捨て/ゼロ埋めにより生じるスペクトル漏れやサイドローブを、窓関数により抑制します。
