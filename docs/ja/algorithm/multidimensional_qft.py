# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
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

# %%
# From the repository root, install this Qamomile checkout and Matplotlib.
# # pip install -e . matplotlib

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import amplitude_encoding, qft

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 1 if docs_test_mode else 2**14

# %% [markdown]
# ## 背景
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

# %% [markdown]
# ## 問題設定
#
# 入力データを作成しましょう。
# ここでは、次のような仮想的な周期データを用いることにします。

# %%
Nx = 6
Ny = 6
xs = np.arange(Nx)
ys = np.arange(Ny)
X, Y = np.meshgrid(xs, ys, indexing="ij")
f = np.cos(np.pi * X / 2) * np.cos(np.pi * Y / 2)

# %% [markdown]
# 作成された入力データは、次のような単純な周期性を持つ2次元データです。

# %%
im1 = plt.pcolormesh(f.T, cmap="viridis")
cbar = plt.colorbar(im1)
cbar.set_label("f", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()


# %% [markdown]
# ## アルゴリズム
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
# そこで、[](https://doi.org/10.1039/d4cp04399e)では、QFTの前に入力を2のべき乗のグリッドに合わせる前処理を提案しました。
# Qamomileには、1次元QFTが標準で備わっています。
# Qamomileの機能を活用することで、多次元QFTも容易に実装することが可能です。
#
# 結晶やナノシートの実空間構造は、一般に2のべき乗のグリッド上でサンプリングされるとは限りません。
# その格子周期性は、任意の整数個のグリッド点に対応しえます。
# そこで、[](https://doi.org/10.1039/d4cp04399e)では、結晶の周期性をそのまま量子状態に符号化するための、前処理手法を提案しました。
#
# 実際に得られる結晶構造のデータは、2のべき乗のグリッド上に定義されているとは限りません。
# ある次元方向のサンプリング点数が$2^p$より大きい場合、最初の$2^p$点のみを抜き出すことで、QFTを適用可能にします。
# またある次元方向のサンプリング点数が$2^p$より小さい場合、その前後にゼロを追加することでも、QFTを適用可能にします。
#
# 領域の切り捨てによって、本来のスペクトルの横に余分な成分が生じることがあります。
# これをサイドローブと呼び、これは領域を切り取ったことで生まれる人工的なスペクトルです。
# ゼロ埋めをすると、このスペクトルをより細かい間隔でサンプリングできます。
# そこでこれを抑えるために、切り捨て/ゼロ埋めされた入力データ区間の両端を滑らかにゼロに近づける重みづけ関数 (窓関数) を導入しましょう。
# この論文では、次のようなフラットトップ関数を用いています。
#
# $$
# w_5(n) = \sum_{k=0}^4 (-1)^k a_k \cos \left( \frac{2\pi k n}{N}\right)
# $$
#
# ここで$(a_0, a_1, a_2, a_3, a_4) = (0.1881, 0.36923, 0.28702, 0.13077, 0.02488)$であり、$N$は考えている次元方向の点の総数、$n$はその点の位置$(n=0, 1, \dots, N-1)$を表します。
# 2次元の場合、$w_5(x) w_5(y)$ のような積を、入力に掛け合わせます。

# %% [markdown]
# ## 実装
#
# Qamomileを用いて、多次元QFTのための量子回路を実装しましょう。
#
# このデータはサイズが2のべき乗でないために、そのままでは実装されたQFTを適用することができません。
# そこでゼロ埋めを施すことで、サイズを2のべき乗に拡張しましょう。
# ここでは、$x$方向と$y$方向の座標レジスタそれぞれに3量子ビットを使います。
# すなわち、各次元が$2^3 = 8$個のグリッド点を持つようにゼロ埋めします。

# %%
def crop_and_pad(
    array_2d: np.ndarray, num_x_target: int, num_y_target: int
) -> np.ndarray:
    array_2d = array_2d[:num_x_target, :num_y_target]
    num_x, num_y = array_2d.shape
    pad_width_x = num_x_target - num_x
    pad_width_y = num_y_target - num_y
    pad_x_left = int(np.ceil(pad_width_x / 2.0))
    pad_x_right = int(np.floor(pad_width_x / 2.0))
    pad_y_left = int(np.ceil(pad_width_y / 2.0))
    pad_y_right = int(np.floor(pad_width_y / 2.0))
    array_2d_padding = np.pad(
        array_2d, ((pad_x_left, pad_x_right), (pad_y_left, pad_y_right))
    )
    return array_2d_padding


Nqx = 3
Nqy = 3
Nx_target = 2**Nqx
Ny_target = 2**Nqy
f_padding = crop_and_pad(f, Nx_target, Ny_target)

# %% [markdown]
# ゼロ埋めされたデータを見てみましょう。

# %%
im2 = plt.pcolor(f_padding.T, cmap="viridis")
cbar = plt.colorbar(im2)
cbar.set_label("f", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()


# %% [markdown]
# 続いてはサイドローブを抑えるために、窓関数の実装を行いましょう。

# %%
def w5(N: int) -> np.ndarray:
    a = np.array([0.1881, 0.36923, 0.28702, 0.13077, 0.02488])
    n = np.arange(N)
    results = [(-1) ** k * a[k] * np.cos(2 * np.pi * k * n / N) for k in range(5)]
    return sum(results)


wx = w5(Nx_target)
wy = w5(Ny_target)
w2d = wx[:, None] * wy[None, :]

# %% [markdown]
# 窓関数の概形を描画してみましょう。

# %%
im3 = plt.pcolor(w2d.T, cmap="plasma")
cbar = plt.colorbar(im3)
cbar.set_label("Flat-top window function", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()


# %% [markdown]
# 窓関数の重みは中心領域で最も大きく、端の近くでは小さくなっていることがわかります。
# この窓関数をゼロ埋めされたデータに適用し、2次元QFTを実行しましょう。

# %% [markdown]
# Qamomileの`qft`関数を用いて、多次元QFTを実装しましょう。

# %%
def make_multidimensional_qft(inputs: np.ndarray, nqx: int, nqy: int) -> qm.QKernel:
    if nqx < 1 or nqy < 1 or inputs.shape != (2**nqx, 2**nqy):
        raise ValueError("The input shape must match the two nonempty qubit registers.")
    width = nqx + nqy
    prepare = amplitude_encoding(tuple(inputs.ravel(order="F")))

    @qm.qkernel
    def qft_for_multidimension() -> qm.Shot[qm.Bits[width]]:
        state = prepare(qm.qubits(width))
        x_axis, y_axis = qm.split_at(state, nqx)
        x_axis = qft(x_axis, nqx)
        y_axis = qft(y_axis, nqy)
        return qm.measure(qm.concat(x_axis, y_axis))

    return qft_for_multidimension

# %% [markdown]
# これでQFTの適用が可能になりました。

# %% [markdown]
# 先ほど実装したQFT回路をQamomileの組み込みシミュレーター `qm.Reference()` 向けにコンパイルしましょう。
# 下位の量子ビットが $x$ を表すよう、$x$ が最も速く変わる順に入力を平坦化します。

# %%
reference = qm.Reference()
program = qm.specialize(make_multidimensional_qft(f_padding, Nqx, Nqy))
exe = qm.compile(program, to=reference)

# %% [markdown]
# ## 結果
#
# コンパイルした回路を実行し、サンプリングされたビット列を2次元の逆格子空間グリッド上の確率にマッピングしましょう。ここで $k_x$ と $k_y$ は、それぞれ $x$ 方向と $y$ 方向の周波数の番号です。

# %%
def compute_prob(result: qm.SampleResult, nqx: int, nqy: int) -> np.ndarray:
    prob = np.zeros((2**nqx, 2**nqy))
    for bits, count in result.counts.items():
        kx = sum(bits[i] << i for i in range(nqx))
        ky = sum(bits[nqx + i] << i for i in range(nqy))
        prob[kx, ky] += count / result.shots
    return prob


result1 = qm.run(exe, on=reference, shots=sample_shots, seed=5).result
prob = compute_prob(result1, Nqx, Nqy)

# %% [markdown]
# 比較のため、同じ入力に対して古典FFTを適用した場合の結果も計算しましょう。

# %%
def compute_classical_fft(f: np.ndarray) -> np.ndarray:
    Nx_target, Ny_target = f.shape
    f_normalized = f / np.linalg.norm(f)
    fft = np.fft.ifft2(f_normalized) * np.sqrt(Nx_target * Ny_target)
    prob = np.abs(fft) ** 2
    return prob


prob_classical = compute_classical_fft(f_padding)
assert np.allclose(prob, prob_classical, atol=3 / np.sqrt(sample_shots))

# %% [markdown]
# 結果を可視化しましょう。

# %%
vmin = min(prob_classical.min(), prob.min())
vmax = max(prob_classical.max(), prob.max())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im4 = axes[0].pcolor(prob_classical.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[0].set_title("Classical")
axes[0].set_xlabel("kx")
axes[0].set_ylabel("ky")
im5 = axes[1].pcolor(prob.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[1].set_title("Quantum")
axes[1].set_xlabel("kx")
print("Classical peak value: {}".format(prob_classical.max()))
print("Quantum peak value: {}".format(prob.max()))
fig.colorbar(im5, ax=axes, fraction=0.046, pad=0.04)
plt.show()

# %% [markdown]
# FFT参照結果とQFTサンプリング結果のどちらにも、同様のサイドローブ構造が見られます。

# %% [markdown]
# 窓関数を適用した入力を用い、同じ計算を行いましょう。

# %%
fw = w2d * f_padding
program = qm.specialize(make_multidimensional_qft(fw, Nqx, Nqy))
exe = qm.compile(program, to=reference)
result2 = qm.run(exe, on=reference, shots=sample_shots, seed=7).result
prob2 = compute_prob(result2, Nqx, Nqy)

# %% [markdown]
# また比較のため、同じく窓関数を適用した入力に古典FFTを適用しましょう。

# %%
prob_classical2 = compute_classical_fft(fw)
assert np.allclose(prob2, prob_classical2, atol=3 / np.sqrt(sample_shots))

# %% [markdown]
# これら2つの結果を可視化しましょう。

# %%
vmin = min(prob_classical2.min(), prob2.min())
vmax = max(prob_classical2.max(), prob2.max())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im6 = axes[0].pcolor(prob_classical2.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[0].set_title("Classical")
axes[0].set_xlabel("kx")
axes[0].set_ylabel("ky")
im7 = axes[1].pcolor(prob2.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[1].set_title("Quantum")
axes[1].set_xlabel("kx")
print("Classical peak value: {}".format(prob_classical2.max()))
print("Quantum peak value: {}".format(prob2.max()))
fig.colorbar(im7, ax=axes, fraction=0.046, pad=0.04)
plt.show()

# %% [markdown]
# この例では、窓関数を適用するとピークの確率が下がり、その周辺の周波数にも確率が広がります。

# %% [markdown]
# ## まとめ
#
# ここでは、2のべき乗でない入力データに対する前処理手法を開発した論文と、Qamomileを用いたQFTの実装方法についてご紹介しました。
# 以下にこのページで紹介した重要な情報をまとめます。
#
# * Qamomileには標準でQFTを実装する機能が備わっています。
# * 入力が2のべき乗でない場合、領域を切り捨て/ゼロ埋めすることでQFTを適用可能にします。
# * 領域の切り捨てにより生じるサイドローブを、窓関数により抑制できます。
