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
# tags: [tutorial]
# ---
#
# # 量子プログラムの実行と期待値の推定
#
# Qamomile で量子プログラムを作成したら、次は実行して結果を確認しましょう。
# このチュートリアルでは、1 回の実行結果を受け取る方法だけでなく、複数回測定して結果を集める方法、そしてオブザーバブルの期待値を推定する方法を学びます。
# この章で学ぶ内容を、以下にまとめます。
#
# - `qm.evaluate(...)` を用い、1 回の実行結果を取得します。
# - `qm.run(...)` で量子回路を繰り返し実行し、測定結果を読み取ります。
# - `qm.statistics` により、測定結果から平均値と標準誤差を求めます。
# - `Estimator` で、ハミルトニアンの期待値を推定します。
#
# このチュートリアルでは、Qamomile に組み込まれた `Reference` シミュレーターを使い、手元のコンピューターで実行します。
# `qm.evaluate(...)` は `Reference` を自動で用います。
# `qm.run(...)` と `estimator.estimate(...)` の場合には、引数を `on=reference` とすることで明示的に指定する必要があります。

# %%
from __future__ import annotations

import numpy as np

import qamomile as qm
from qamomile.measurement import Estimator

reference = qm.Reference()


# %% [markdown]
# ## 1. `qm.evaluate(...)`：1 回の実行結果の取得
#
# まず、2 量子ビットの回路を作成しましょう。
# `prepare_parity_state(...)` は、量子ビット 0 に H ゲート、量子ビット 1 に角度 $\theta$ の RY ゲートを適用します。
# そしてその後、量子ビット 0 を制御、量子ビット 1 をターゲットとして CX ゲートを適用します。
# `parity_probe(...)` は、この状態の両量子ビットを計算基底で測定します。\
# 回転角 `theta` は `qm.Param[qm.Real]` として宣言しているため、その値を指定せずにカーネルを特殊化することができます。
# 値は、プログラムを実行するときに `qm.evaluate(...)` へ渡します。
# 戻り値の型注釈 `qm.Shot[qm.Bits[2]]` は、各実行で 2 ビットの測定結果を返すことを表しています。


# %%
@qm.qkernel
def prepare_parity_state(theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.ry(state, theta, target=1)
    state = qm.cx(state, control=0, target=1)
    return state


@qm.qkernel
def parity_probe(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(prepare_parity_state(theta))


theta = 0.7
parity_program = qm.specialize(parity_probe)
one_result = qm.evaluate(parity_program, theta=theta)
print("One-shot result:", one_result)

# %% [markdown]
# `qm.specialize(parity_probe)` でプログラムを準備し、`qm.evaluate(parity_program, theta=theta)` で 1 回実行します。
# 結果は測定された 2 ビットを含む Python のタプルで、先頭が量子ビット0、次が量子ビット1に対応します。\
# 量子測定は確率的ですが、`qm.evaluate(...)` では既定で `seed=0` を使います。
# そのため、この例は入力を変えずに再実行すると、同じ結果が得られます。
#
# ## 2. `qm.run(...)`：量子回路を繰り返し実行する
#
# 次に、同じ `parity_program` を `qm.run(...)`で 128 回実行しましょう。
# 回転角は引き続き $\theta = 0.7$ ラジアンとし、各ショットで 2 ビットの測定結果を取得します。\
# `on=reference` でシミュレーターを指定し、`parameters={"theta": theta}` で回転角を渡します。
# `shots=128` は測定の実行回数です。
# シミュレーションの結果を再現できるよう、`seed=7` を指定しています。

# %%
record = qm.run(
    parity_program,
    on=reference,
    parameters={"theta": theta},
    shots=128,
    seed=7,
)
samples = record.result
counts = samples.values

print("Shots:", samples.shots)
print("Outcome counts:", counts)
assert sum(counts.values()) == samples.shots

# %% [markdown]
# `samples.values`は読み取り専用の辞書形式で、要素の追加・変更・削除はできません。
# ここでは出現回数の表示と合計にそのまま使います。
# キーは測定結果を表す 2 ビットのタプルで、値はその結果の出現回数です。
# 例えば `(0, 1): 7` は、量子ビット 0 の測定結果が 0、量子ビット 1 の測定結果が 1 となったショットが 7 回あったことを表します。
# `shots` は量子回路の測定回数です。
# `seed` は乱数の再現性を確認するために指定していますが、測定結果は一般にショット数や乱数によって変化します。
#
# ## 3. 期待値の計算
#
# 量子計算では、繰り返し測定した結果の平均から、観測量の期待値を推定することがあります。
# `qm.statistics` は、量子プログラムを再実行せずに、取得済みの結果から統計量を求めるモジュールです。
# ここでは、$ZZ = Z_0 Z_1$ の期待値を推定します。\
# 計算基底（Z基底）での測定では、ビット 0 は Z の固有値 +1 に、ビット 1 は −1 に対応します。
# この 2 つの固有値の積は、00 と 11 では +1、01 と 10 では −1 になります。\
# この積は、測定結果に含まれる 1 の個数の偶奇（パリティ）を、偶数なら +1、奇数なら −1 で表したものです。
# `qm.statistics.parity_mean(samples, indices=(0, 1))` は、各結果の出現回数を用い、この値の平均を求めます。
# これらの値は ZZ の測定値に対応するため、その平均が $\langle ZZ\rangle$ の推定値になります。
# 推定値の標準誤差も取得することができます。

# %%
parity = qm.statistics.parity_mean(samples, indices=(0, 1))

print("Estimated ZZ:", parity.value)
print("Standard error:", parity.standard_error)
print("Theoretical value (cos(theta)):", np.cos(theta))


# %% [markdown]
# 計算基底 (Z 基底) を用い、量子ビット 0 を左側に書くと、この回路が準備する状態は次のようになります。
#
# $$
# \lvert \psi \rangle
# = \frac{\cos (\theta / 2)}{\sqrt{2} } (\lvert 00 \rangle + \lvert 11 \rangle) + \frac{\sin (\theta / 2)}{\sqrt{2} } (\lvert 01 \rangle + \lvert 10 \rangle) \tag{1}
# $$
#
# 2 つのビットが一致する確率は $\cos^2 (\theta / 2)$、異なる確率は $\sin^2 (\theta / 2)$ となります。
# それぞれの場合の ZZ の値は $\pm 1$ なので、期待値は次のようになります。
#
# $$
# \langle ZZ \rangle
# = \cos^2 (\theta / 2) - \sin^2 (\theta / 2)
# = \cos \theta \tag{2}
# $$
#
# 有限回の測定から得た推定値は、理論値と必ずしも一致しません。
# 標準誤差は、測定結果のばらつきによる推定の不確かさを表します。
# ショット数を増やすことで、一般にこの不確かさを小さくすることができます。
#
# :::{note}
# `qm.statistics` は、すでに取得した測定結果を集計するライブラリです。
# 今回は Z 基底で測定したので ZZ を求めることができますが、X の期待値を同じ Z 基底の測定結果から計算することはできないことに注意しましょう。
# :::
#
#
# ## 4. `Estimator`： ハミルトニアンの期待値を推定する
#
# Z と X のように、異なる測定基底を必要とする項を含むハミルトニアンを扱う場合は、`Estimator` が便利です。
#
# `Estimator` には、次の2つを渡します。
#
# 1. 状態準備：量子状態を作る QKernel
# 2. オブザーバブル：期待値を知りたいハミルトニアン
#
# ここでは、1 量子ビットを Y 軸まわりに回転させる状態準備を定義しましょう。
# この関数は測定せずに量子状態を返すため、単独での測定プログラムではなく、`Estimator` のための状態準備の定義として利用します。


# %%
@qm.qkernel
def rotated_state(theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.ry(qm.qubits(1), theta)


preparation = rotated_state.specialize()

# %% [markdown]
# 次に、期待値を求めたいハミルトニアンを設定しましょう。
#
# $$
# H
# = Z_0 + 0.5 X_0 \tag{3}
# $$
#
# $Z_0, X_0$ はそれぞれ、0 番目の量子ビットに作用するパウリ演算子です。
# これらは `qm.Z(0)` と `qm.X(0)` のように書くことができます。\
# `Estimator` は、各ショットで状態準備用の量子カーネルを用い、量子状態を新たに準備します。
# 一部のショットでは Z 基底、残りでは X 基底で測定し、それぞれの推定値から $\langle Z_0\rangle + 0.5\langle X_0\rangle$ を求めます。

# %%
hamiltonian = qm.Z(0) + 0.5 * qm.X(0)
estimator = Estimator(preparation, hamiltonian, grouping="qwc")

energy = estimator.estimate(
    on=reference,
    parameters={"theta": theta},
    shots=128,
    seed=11,
)

expected = np.cos(theta) + 0.5 * np.sin(theta)

assert energy.shots == 128
assert energy.standard_error is not None
assert abs(energy.value - expected) <= max(6 * energy.standard_error, 0.02)

print("Estimated expectation value:", energy.value)
print("Standard error:", energy.standard_error)
print("Total shots:", energy.shots)
print("Theoretical value:", expected)

# %% [markdown]
# ここで、`energy.value` は推定された期待値、`energy.standard_error` はサンプリングによる標準誤差、`energy.shots` は使用した総ショット数です。
#
# `grouping="qwc"` では、各量子ビットについて同じ測定基底を使える項をまとめます。
# この例の $Z_0$ と $X_0$ は別々に測定を行う必要があるため、総ショット数 128 回を、Z 測定 64 回・X 測定 64 回に均等に分けます。\
# この例の状態準備に対しては、理論値が
#
# $$
# \langle H \rangle
# = \cos(\theta) + 0.5\sin(\theta) \tag{4}
# $$
#
# となります。\
# 有限回の測定から求めた推定値は、理論値と一致しないことがあります。
# ショット数を増やすと、一般に推定の不確かさが小さくなります。\
# QKernel には 1 回の実行で行う処理を記述し、複数ショットの結果はカーネルの外で集計します。
# 前の例では `qm.run(...)` で取得した結果を `qm.statistics` で集計しました。
# `Estimator` を使う場合は、測定の実行から結果の集計までを任せることができます。
#
# :::{note}
# 各結果に計算を行ってから平均する場合と、平均してから計算する場合では、答えが異なることがあります。
# 例えば、 +1 と −1 が同じ回数だけ得られた場合、それぞれを二乗してから平均すると 1 となります。
# しかし、平均してから二乗すると 0 です。
# どちらも、`SampleResult` に保存された結果と出現回数を使い、Python側で計算できます。
# 各実行で行う処理の一部として、量子カーネル内に計算を書くこともできます。
# :::

# %% [markdown]
# ## まとめ
#
# | やりたいこと | 使うもの |
# |---|---|
# | 1 回だけ実行して結果を受け取る | `qm.evaluate(...)` |
# | 量子回路を繰り返し測定する | `qm.run(...)` |
# | 取得したサンプルから平均や誤差を求める | `qm.statistics` |
# | 状態準備とハミルトニアンから期待値を推定する | `Estimator` |
#
# まずは `qm.run(...)` で測定結果を取得し、`qm.statistics` で集計する流れを確認しましょう。
# 複数のパウリ項に対して必要な測定を自動で行いたい場合には、`Estimator` を用いるのが良いでしょう。
#
# **次へ：** [古典制御フロー](06_classical_flow_patterns.ipynb)では、`for` や `if`、測定結果に応じた条件分岐を学びます。

# %% [markdown]
#
