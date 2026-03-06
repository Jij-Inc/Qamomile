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
# **実行時の値**（回転角度）の両方を変化させる必要があることがよくあります。
#
# この章では以下を学びます：
#
# - カーネル入力における `UInt` と `Float` の典型的な役割
# - パラメータ化された回路のための `qubit_array()` と `qmc.range()`
# - **バインド/スイープパターン**：一度コンパイルし、複数回実行する

# %% [markdown]
# ## `UInt` と `Float` の典型的な役割
#
# カーネルのパラメータには2種類あります：
#
# | 型 | 典型的な役割 |
# |------|-------------|
# | `qmc.UInt` | 回路構造（量子ビット数、ループ上限） |
# | `qmc.Float` | 連続値（回転角度、重み） |
#
# 実際には、`qubit_array` のサイズや `qmc.range` の上限を制御する `UInt` の値は、
# バックエンドが固定された回路構造を必要とするため、トランスパイル時にバインド
# する**必要があります**。`Float` の値はスイープ可能なパラメータとして残すことができます。
#
# 一般的なパターンは、構造をトランスパイル時にバインドし、連続値を実行時に
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
# > Python の整数ではありません。カーネル本体は IR を構築するためにトレースされますが、
# > Python の `range()` はシンボルを反復処理できません。`qmc.range()` は IR 内に
# > **ループノード**を生成し、`n` が具体的な値にバインドされた時点でトランスパイラが展開します。


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
# `for qi in q:` と書くと、Qamomile がアフィンルールを適用するために使用する
# 所有権追跡をバイパスしてしまうため、動作しません。

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
# これはパラメータ付きカーネルの中心的なワークフローパターンです：
#
# 1. **一度トランスパイル**：構造パラメータ（`n`）をバインドし、実行時
#    パラメータ（`theta`）をスイープ可能として宣言します。
# 2. **複数回実行**：コンパイル済みの実行可能オブジェクトを異なる実行時
#    バインディングで再利用します。
#
# これにより、パラメータ値ごとに回路を再コンパイルすることを回避できます。

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
# トランスパイルされた実行可能オブジェクトは3回の実行すべてで再利用されます。
# 変わるのは実行時バインディング `{"theta": theta}` だけです。
#
# まとめると：
#
# - **`bindings={"n": 4}`**（トランスパイル時）：回路構造を固定します。
# - **`parameters=["theta"]`**（トランスパイル時）：`theta` をスイープ可能として宣言します。
# - **`bindings={"theta": ...}`**（実行時）：具体的な値を提供します。

# %% [markdown]
# ## まとめ
#
# - 回路構造（量子ビット数、ループ上限）を制御する `qmc.UInt` の値は、通常
#   トランスパイル時にバインドする必要があります。`qmc.Float` の値（回転角度）は
#   実行時スイープの自然な候補です。
# - パラメータ化された回路には `qmc.qubit_array(n)` と `qmc.range(n)` を使用してください。
#   常にインデックスベースの更新を使用してください：`q[i] = qmc.gate(q[i])`。
# - バインド/スイープパターン — `transpile(bindings=..., parameters=...)` の後にループ —
#   は一度コンパイルして複数回実行します。
#
# **次へ**：[リソース推定](03_resource_estimation.ipynb) — シンボリックなコスト
# 分析、ゲート分解、設計候補の比較。
