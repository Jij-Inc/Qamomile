# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # 実行モデル：`sample()` と `run()`
#
# Qamomile はカーネルの戻り値に応じて 2 つの実行メソッドを提供します：
#
# | カーネルの戻り値 | 使用メソッド | 返される結果 |
# |----------------|-----|-------------|
# | `Bit`, `Vector[Bit]`, `tuple[Bit, ...]` | `sample()` | `SampleResult` — カウント付き測定結果 |
# | `Float` (`expval` から) | `run()` | `float` — 期待値 |
#
# この章では両方のモードを説明し、期待値計算のための**オブザーバブル**を紹介します。

# %%
import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 複数量子ビットの `sample()`
#
# チュートリアル 01 では単一の `Bit` をサンプリングしました。カーネルが複数ビットを返す場合、
# 各測定結果は整数値（`0` または `1`）の**タプル**になります。


# %%
@qmc.qkernel
def parity_probe(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    q0 = qmc.h(q0)
    q1 = qmc.ry(q1, theta)
    q0, q1 = qmc.cx(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


# %%
parity_probe.draw(theta=0.7)

# %%
exe_sample = transpiler.transpile(parity_probe, parameters=["theta"])
sample_result = exe_sample.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": 0.7},
).result()

for outcome, count in sample_result.results:
    print(f"  outcome={outcome}, count={count}")

# %% [markdown]
# 各 `outcome` は `(0, 1)` や `(1, 0)` のようなタプルです。
# 最初の要素は `q0` に、2番目の要素は `q1` に対応し、
# `return` 文での順序と一致します。

# %% [markdown]
# ## ビット順序の規約
#
# Qamomile の出力は**ビッグエンディアン**順序を使用します: **最も左**の位置が
# 戻り値タプルの**最初の**量子ビットに対応します。
#
# `(measure(q0), measure(q1), measure(q2))` を返すカーネルの場合:
#
# | 結果タプル | q0 | q1 | q2 |
# |--------------|----|----|-----|
# | `(0, 1, 1)` | 0 | 1 | 1 |
# | `(1, 0, 0)` | 1 | 0 | 0 |
#
# これは直感的です — タプルの位置 `i` が戻り値の量子ビット `i` に対応します。
#
# > **注意**: Qiskit は内部的にリトルエンディアンを使用しますが、Qamomile が
# > 変換を処理します。結果は常に記述した順序で得られます。

# %% [markdown]
# ## 期待値が必要な場合
#
# 個々の測定結果ではなく、量子オブザーバブルの**期待値**（平均値）が必要な場面があります。
# 例えば：
#
# - **VQE**（変分量子固有値ソルバー）：`<psi|H|psi>` の最小化
# - **QAOA**：コスト関数の期待値の評価
# - 量子パラメータに対するあらゆる最適化ループ
#
# このために Qamomile は `expval()` と `run()` を提供しています。

# %% [markdown]
# ## Observable 型
#
# 関連する 2 つの概念があります：
#
# 1. **`qmc.Observable`** — カーネルのシグネチャで使用する**ハンドル型**です。
#    `qmc.Float` が数値のプレースホルダーであるのと同様に、
#    トレーシング中のプレースホルダーとして機能します。
#
# 2. **`qamomile.observable` モジュール** — バインディングで渡す**具体的な**
#    オブザーバブル値を構築する場所です。例えば：
#
# ```python
# import qamomile.observable as qmo
#
# H = qmo.Z(0)                         # 量子ビット0のパウリZ
# H = qmo.Z(0) * qmo.Z(1)             # ZZ相互作用
# H = 0.5 * qmo.X(0) + 0.3 * qmo.Y(1) # 線形結合
# ```
#
# つまり `qmc.Observable` はカーネル内の**型アノテーション**、
# `qmo.Z(0)` はトランスパイル時にバインドする**具体的な値**です。

# %% [markdown]
# ## `expval()`: オブザーバブルの測定
#
# `expval(qubit, hamiltonian)` は期待値 `<psi|hamiltonian|psi>` を計算し、
# `Float` を返します。
# `expval` から `Float` を返すカーネルは `run()` で実行する必要があります。


# %%
@qmc.qkernel
def z_expectation(theta: qmc.Float, hamiltonian: qmc.Observable) -> qmc.Float:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.expval(q, hamiltonian)


H = qmo.Z(0)

# %%
z_expectation.draw(theta=0.7, hamiltonian=H)

# %% [markdown]
# ## `run()` による実行
#
# 期待値カーネルには `sample()` の代わりに `run()` を使います。
# オブザーバブルはトランスパイル時にバインドし（測定回路に影響するため）、
# `theta` はスイープ可能なランタイムパラメータとして残します。

# %%
exe_run = transpiler.transpile(
    z_expectation,
    bindings={"hamiltonian": H},  # Observable bound at transpile time
    parameters=["theta"],  # theta remains sweepable
)

run_result = exe_run.run(
    transpiler.executor(),
    bindings={"theta": 0.7},
).result()

print("expectation value:", run_result)
print("python type:", type(run_result))

# %% [markdown]
# `run().result()` はプレーンな `float` を返します — 推定された `<psi|Z|psi>` の値です。
# `theta = 0.7` の場合、RY ゲートが量子ビットを回転させ、Z の期待値は
# `cos(0.7) ≈ 0.765` となります。

# %% [markdown]
# ## 判断ガイド: `sample()` と `run()`
#
# | カーネルの戻り値 | 実行メソッド | `.result()` の返り値 |
# |----------------|-----------------|-------------------|
# | `Bit` | `sample()` | `SampleResult` (`.results: list[tuple[int, int]]`) |
# | `tuple[Bit, Bit]` | `sample()` | `SampleResult` (`.results: list[tuple[tuple[int, int], int]]`) |
# | `Vector[Bit]` | `sample()` | `SampleResult` (`.results: list[tuple[tuple[int, ...], int]]`) |
# | `Float` (`expval` から) | `run()` | `float` |
#
# **経験則**: カーネルが `measure()` で終わる場合は `sample()` を使用します。
# `expval()` で終わる場合は `run()` を使用します。

# %% [markdown]
# ## まとめ
#
# - `sample()` は測定ビットを返すカーネル用 — カウント付きの測定結果分布が得られます。
# - `run()` は `expval()` で `Float` を返すカーネル用 — 単一の期待値が得られます。
# - `qmc.Observable` はハンドル型、`qamomile.observable.Z(0)` 等が具体的な値です。
#   オブザーバブルはトランスパイル時にバインドします。
# - ビット順序はビッグエンディアン：戻り値タプルの位置が量子ビットの順序に対応します。
#
# **次へ**：[古典フローパターン](05_classical_flow_patterns.ipynb) — `qmc.range` によるループ、
# `qmc.items` によるスパースデータ、条件分岐。
