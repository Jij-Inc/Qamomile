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
# # パラメータ付きカーネル
#
# チュートリアル 01 では、固定数の量子ビットを持つカーネルを構築しました。
# 実際の量子プログラムでは、**回路構造**（量子ビット数、レイヤー数）と
# **ランタイム値**（回転角度）の両方を変化させる必要があります。
#
# この章では以下を学びます：
#
# - カーネル入力における `UInt` と `Float` の役割
# - パラメータ化された回路のための `qubit_array()` と `qmc.range()`
# - **バインド/スイープパターン**：一度コンパイルし、複数回実行する

# %% [markdown]
# ## `UInt` と `Float` の役割
#
# カーネルのパラメータは大きく 2 種類に分かれます：
#
# | 型 | 役割 |
# |------|-------------|
# | `qmc.UInt` | 回路構造（量子ビット数、ループ上限） |
# | `qmc.Float` | 連続値（回転角度、重み） |
#
# `qubit_array` のサイズや `qmc.range` の上限を制御する `UInt` はバックエンドが
# 固定の回路構造を必要とするため、トランスパイル時にバインドする**必要があります**。
# 一方 `Float` はスイープ可能なパラメータとして残せます。
#
# 典型的なパターンは、構造をトランスパイル時に固定し、連続値を実行時に
# スイープすることです。

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## `qubit_array` と `qmc.range`
#
# 量子ビット数がパラメータ `n` に依存する場合、`qubit_array(n)` を使用します。
# 配列をループするには、Python 組み込みの `range()` ではなく `qmc.range(n)` を使用します。
#
# > **なぜ Python の `range()` ではダメなのか？** トレース時に `n` はシンボルであり、
# > Python の整数ではありません。Python の `range()` はシンボルに対して反復できないため、
# > 代わりに `qmc.range()` を使います。`qmc.range()` は IR 内に**ループノード**を生成し、
# > `n` が具体的な値にバインドされた時点でトランスパイラが展開します。


# %%
@qmc.qkernel
def rotation_layer(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
        q[i] = qmc.ry(q[i], theta)

    return qmc.measure(q)


# %% [markdown]
# `draw()` はパラメータを具体的な値にバインドするキーワード引数を受け取ります。
# ここでは `n=4` で回路を4量子ビットに固定し、`theta=0.3` でプレースホルダーの
# 角度を指定しています。

# %%
rotation_layer.draw(n=4, theta=0.3)

# %% [markdown]
# ## インデックスベースの更新
#
# パターン `q[i] = qmc.h(q[i])` に注目してください。
# 配列にインデックスでアクセスし、ゲートを適用して、結果を同じインデックスに
# 書き戻します。これはチュートリアル 01 のアフィンルールに従っています。
# `q[i]` の古いハンドルは消費され、返されたハンドルがその場所に入ります。
#
# **アンチパターン：配列を直接イテレートする。**
# `for qi in q:` と書くと、Qamomile がアフィンルールの適用に使う
# 所有権追跡を迂回してしまうため、禁止しています。

# %%
try:

    @qmc.qkernel
    def bad_iteration(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        q = qmc.qubit_array(n, name="q")
        for qi in q:  # 直接イテレーション — ダメ！
            qi = qmc.h(qi)
        return q

    bad_iteration.draw(n=4)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# 常にインデックスベースのアクセスを使用してください：`for i in qmc.range(n): q[i] = qmc.h(q[i])`。

# %% [markdown]
# ## バインド/スイープパターン
#
# パラメータ付きカーネルの中心となるワークフローパターンです：
#
# 1. **一度トランスパイル**：構造パラメータ（`n`）をバインドし、ランタイム
#    パラメータ（`theta`）をスイープ可能として宣言します。
# 2. **複数回実行**：コンパイル済みのオブジェクトを異なるランタイム
#    バインディングで再利用します。
#
# これにより、パラメータ値を変えるたびに回路を再コンパイルする必要がなくなります。

# %%
# トランスパイル：n=4 を固定し、theta をスイープ可能なパラメータとして保持
exe = transpiler.transpile(
    rotation_layer,
    bindings={"n": 4},
    parameters=["theta"],
)

# スイープ：同じ実行可能オブジェクトを異なる theta 値で実行
for theta in [0.1, 0.5, 1.0]:
    result = exe.sample(
        transpiler.executor(),
        shots=128,
        bindings={"theta": theta},
    ).result()
    print(f"theta={theta:.1f} -> {result.results}")

# %% [markdown]
# トランスパイル済みオブジェクトは 3 回の実行すべてで再利用されます。
# 変わるのはランタイムバインディング `{"theta": theta}` だけです。
#
# まとめると：
#
# - **`bindings={"n": 4}`**（トランスパイル時）：回路構造を固定。
# - **`parameters=["theta"]`**（トランスパイル時）：`theta` をスイープ可能として宣言。
# - **`bindings={"theta": ...}`**（実行時）：具体的な値を指定。

# %% [markdown]
# ## まとめ
#
# - 回路構造（量子ビット数、ループ上限）を制御する `qmc.UInt` は通常
#   トランスパイル時にバインドします。`qmc.Float`（回転角度）はランタイムスイープの
#   候補です。
# - パラメータ化された回路には `qmc.qubit_array(n)` と `qmc.range(n)` を使います。
#   常にインデックスベースの更新 `q[i] = qmc.gate(q[i])` を使ってください。
# - バインド/スイープパターン — `transpile(bindings=..., parameters=...)` → ループ —
#   で一度コンパイルして複数回実行できます。
#
# **次へ**：[リソース推定](03_resource_estimation.ipynb) — シンボリックなコスト
# 分析、ゲート内訳、スケーリング分析。
