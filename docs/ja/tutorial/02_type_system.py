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
# # Qamomileの型システム
#
# Qamomileは、量子データと古典データを区別し、実行前に正しさを保証し、
# 回路を自己文書化するための豊富な型システムを備えています。
# このチュートリアルでは、Qamomileが提供するすべての型と、
# それぞれをどのような場面で使うかを学びます。
#
# ## このチュートリアルで学ぶこと
# - Qamomileの型の全カタログ
# - 量子型と古典型の違い
# - 量子ビットおよび量子ビット配列の作成方法
# - 線形型エラー：どのようなエラーが出るか、どう修正するか
# - 古典スカラー型：`Float`、`UInt`、`Bit`（型アノテーションとして使用）
# - シンボリック値と `qmc.range()` および `qmc.items()` による反復
# - コンテナ型：`Vector`、`Dict`、`Tuple`、`Matrix`、`Tensor`
# - 特殊型：`QFixed`、`Observable`

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. 型の概要
#
# `@qmc.qkernel` 関数内のすべての値にはQamomileの型が割り当てられます。
# これらの型は、関数シグネチャにおける**型アノテーション**
# （例：`theta: qmc.Float`）として使われるとともに、
# 回路内のデータフローを追跡する**ランタイムハンドル**としても機能します。
#
# ### 量子ビット型
#
# 最もよく使われるコンストラクタは量子ビットのコンストラクタです：
#
# | 型 | コンストラクタ | 説明 |
# |------|-------------|-------------|
# | `Qubit` | `qmc.qubit(name=)` | 単一量子ビット、$\|0\rangle$ に初期化 |
# | `Vector[Qubit]` | `qmc.qubit_array(n, name=)` | 1次元量子ビットレジスタ |
#
# ### 古典型
#
# 古典型にもコンストラクタ（`qmc.uint()`、`qmc.float_()`、
# `qmc.bit()`）がありますが、通常は直接構築するのではなく、
# **関数パラメータやバインディングとして提供**されます。
# 値の供給元は以下の通りです：
# 型アノテーション付きの関数引数（`theta: qmc.Float`）、
# トランスパイル時の `bindings` 辞書、
# または操作の戻り値です。
#
# | 型 | カテゴリ | 説明 |
# |------|----------|-------------|
# | `Float` | 古典 | 浮動小数点パラメータ（回転角度など） |
# | `UInt` | 古典 | 符号なし整数（ループ上限、配列インデックス） |
# | `Bit` | 古典 | 測定結果（`qmc.measure()` の戻り値） |
# | `Vector[T]` | コンテナ | 1次元配列（`Vector[Float]`、`Vector[Bit]` など） |
# | `Dict[K, V]` | コンテナ | キー・バリューマッピング（`bindings` 経由で渡す） |
# | `Tuple[K, V]` | コンテナ | 固定サイズのペア（Dictキーのアンパックに使用） |
# | `Matrix[T]` | コンテナ | 2次元配列 |
# | `Tensor[T]` | コンテナ | N次元配列（3次元以上） |
# | `QFixed` | 量子 | 固定小数点量子数（`qmc.qpe()` の戻り値） |
# | `Observable` | 特殊 | ハミルトニアンへの参照（`bindings` 経由で渡し、`qmc.expval()` で使用） |
#
# ### 量子型と古典型
#
# 最も重要な区別は**量子型**と**古典型**の違いです：
#
# - **量子型**（`Qubit`、`QFixed`、`Vector[Qubit]`）は
#   **線形型規則**に従います。各ハンドルは一度しか使用できず、
#   ゲートを適用するたびに再代入が必要です（`q = qmc.h(q)`）。
# - **古典型**（`Float`、`UInt`、`Bit`）は自由に再利用・コピーが
#   可能で、通常のPythonの値と同様に扱えます。

# %% [markdown]
# ## 2. 量子ビット型
#
# `Qubit` と `Vector[Qubit]` は**最もよく構築される型**です。
# 古典型も `qmc.uint()`、`qmc.float_()`、`qmc.bit()` で構築できますが、
# 実際にはこれらの値は通常、以下の方法で提供されます：
#
# - 型アノテーション付きの関数引数（`theta: qmc.Float`）
# - トランスパイル時の `bindings` 辞書
# - 操作の戻り値（`qmc.measure()` は `Bit` を返す）
#
# ### 単一量子ビット：`qmc.qubit()`
#
# $|0\rangle$ に初期化された量子ビットを1つ作成します。


# %%
@qmc.qkernel
def single_qubit_demo() -> qmc.Bit:
    """Create a single qubit, apply H, and measure."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


single_qubit_demo.draw()

# %% [markdown]
# ### 量子ビット配列：`qmc.qubit_array()`
#
# `Vector[Qubit]`（1次元の量子ビットレジスタ）を作成します。
# 個々の量子ビットにはインデックスでアクセスし、`.shape[0]` でサイズを取得できます。


# %%
@qmc.qkernel
def qubit_array_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Create an array of n qubits, apply H to each, and measure."""
    qubits = qmc.qubit_array(n, name="q")

    # qubits.shape[0] gives the symbolic array size
    size = qubits.shape[0]

    for i in qmc.range(size):
        qubits[i] = qmc.h(qubits[i])

    return qmc.measure(qubits)


qubit_array_demo.draw(n=3)

# %% [markdown]
# デフォルトでは、`qmc.range()` で作成されたループはコンパクトなブロックとして表示されます。
# `fold_loops=False` を指定するとループが展開され、各イテレーションが
# 個別のゲートとして表示されます。

# %%
qubit_array_demo.draw(n=3, fold_loops=False)

# %% [markdown]
# #### パラメトリック回路での `draw()`
#
# `qmc.qubit_array(n, ...)` を使った回路では、量子ビット数を
# **必ず指定する**必要があります。指定しないと `draw()` が回路の
# レイアウトを決定できず、エラーが発生します。

# %%
# What happens when you don't specify n?
try:
    qubit_array_demo.draw()  # No n specified — error!
except Exception as e:
    print(f"Error (expected): {type(e).__name__}: {e}")

# %% [markdown]
# ### 線形型エラー
#
# 量子ビットは線形型規則に従うため、Qamomileはトレース時
# （回路がバックエンドに渡される前）によくある間違いを検出します。
# エラーには3種類あります：
#
# | エラー | 原因 |
# |-------|-------|
# | `QubitConsumedError` | ゲートで消費済みの量子ビットハンドルを再利用した |
# | `QubitAliasError` | 2量子ビットゲートの両方の入力に同じ量子ビットを使用した |
# | `UnreturnedBorrowError` | 最初の配列要素を返す前に2番目の要素を借用した |
#
# それぞれの動作を確認してみましょう。


# %%
# QubitConsumedError: reusing a consumed qubit
@qmc.qkernel
def consumed_error_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit(name="q")
    q1 = qmc.h(q)  # consumes q
    q2 = qmc.x(q)  # ERROR: q was already consumed
    return qmc.measure(q1), qmc.measure(q2)


try:
    consumed_error_demo.draw()
except Exception as e:
    print(f"QubitConsumedError (expected): {type(e).__name__}: {e}")


# %%
# QubitAliasError: same qubit as both control and target
@qmc.qkernel
def alias_error_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit(name="q")
    q1, q2 = qmc.cx(q, q)  # ERROR: same qubit twice
    return qmc.measure(q1), qmc.measure(q2)


try:
    alias_error_demo.draw()
except Exception as e:
    print(f"QubitAliasError (expected): {type(e).__name__}: {e}")


# %%
# UnreturnedBorrowError: borrowing without returning
@qmc.qkernel
def borrow_error_demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")
    q0 = q[0]  # borrow q[0]
    q0 = qmc.h(q0)
    q1 = q[1]  # ERROR: q[0] not returned yet
    q1 = qmc.x(q1)
    q[0] = q0
    q[1] = q1
    return qmc.measure(q)


try:
    borrow_error_demo.draw()
except Exception as e:
    print(f"UnreturnedBorrowError (expected): {type(e).__name__}: {e}")

# %% [markdown]
# これらのエラーは `@qkernel` のトレース時（`draw()` や `transpile()` の呼び出し時）に
# 即座に検出されます。実行時ではなくトレース時に検出されるため、
# 量子ハードウェアを使用する前に、明確で対処しやすいエラーメッセージが得られます。

# %% [markdown]
# ## 3. 古典スカラー型
#
# 古典スカラーは、回路のトランスパイル時に値が決定される（または決定予定の）
# データを保持します。ゲートパラメータ、ループ上限、配列インデックスとして使用されます。
#
# 古典型にはコンストラクタ（`qmc.uint()`、`qmc.float_()`、`qmc.bit()`）が
# ありますが、一般的なパターンは `@qkernel` シグネチャで**型アノテーション**として
# 宣言し、トランスパイル時に `bindings` や `parameters` で値を提供することです。
#
# ### `Float` -- 浮動小数点値
#
# `Float` は回転角度などの連続的なゲートパラメータのための型です。
# `@qkernel` シグネチャで型アノテーションとして使用します：`theta: qmc.Float`。
# （Python の `float` もエイリアスとして使用でき、自動的に `Float` に
# 昇格されますが、このチュートリアルでは明確さのために `qmc.Float` を
# 一貫して使用します。）
#
# `Float` は通常の算術演算子（`+`、`-`、`*`、`/`）をサポートしており、
# IR にシンボリックな演算として記録され、トランスパイル時に式全体が評価されます。


# %%
@qmc.qkernel
def float_arithmetic(theta: qmc.Float) -> qmc.Bit:
    """Demonstrate Float arithmetic inside a qkernel."""
    q = qmc.qubit(name="q")

    # Arithmetic on Float values produces new Float handles
    half_theta = theta / 2
    q = qmc.rx(q, half_theta)

    double_theta = theta * 2
    q = qmc.ry(q, double_theta)

    return qmc.measure(q)


float_arithmetic.draw()

# %% [markdown]
# ### `UInt` -- 符号なし整数
#
# `UInt` は非負整数のための型です。以下の用途で使用されます：
#
# - **配列インデックス**：`Vector[Qubit]` へのインデックスアクセス
# - **ループ上限**：`qmc.range()` の引数
# - **シンボリックサイズ**：`qubits.shape[0]` は `UInt` を返す
#
# `Float` と同様に、算術演算（`+`、`-`、`*`、`//`、`**`）や
# 比較演算（`<`、`>`、`<=`、`>=`）をサポートしています。
#
# `@qkernel` シグネチャで型アノテーションとして `n: qmc.UInt` を使用します。
# （Python の `int` もエイリアスとして使用でき、自動的に `UInt` に
# 昇格されますが、このチュートリアルでは明確さのために `qmc.UInt` を
# 一貫して使用します。）


# %%
@qmc.qkernel
def uint_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Use UInt for symbolic array size and loop bounds."""
    qubits = qmc.qubit_array(n, name="q")

    # n is available as a UInt inside the kernel
    for i in qmc.range(n):
        qubits[i] = qmc.h(qubits[i])

    return qmc.measure(qubits)


uint_demo.draw(n=4, fold_loops=False)

# %% [markdown]
# ### `Bit` -- 測定結果
#
# `Bit` は `qmc.measure()` が返す型です。量子ビットを測定した結果の
# 古典ビットを表します。
#
# - 単一量子ビットの測定は `Bit` を返す
# - `Vector[Qubit]` の測定は `Vector[Bit]` を返す
# - `Bit` は通常、戻り値の型としてのみ使用される


# %%
@qmc.qkernel
def bit_demo() -> tuple[qmc.Bit, qmc.Bit]:
    """Measure two qubits independently."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    q0 = qmc.h(q0)
    q1 = qmc.x(q1)

    # Each qmc.measure() returns a Bit
    b0 = qmc.measure(q0)
    b1 = qmc.measure(q1)
    return b0, b1


bit_demo.draw()

# %% [markdown]
# ### 古典型は自由に再利用可能
#
# 量子型とは異なり、古典ハンドル（`Float`、`UInt`、`Bit`）は
# 消費されることなく何度でも読み取ることができます。


# %%
@qmc.qkernel
def reuse_classical(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """The same Float handle can be used in multiple gates."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    # theta is used twice -- this is perfectly fine for classical types
    q0 = qmc.rx(q0, theta)
    q1 = qmc.ry(q1, theta)

    return qmc.measure(q0), qmc.measure(q1)


reuse_classical.draw()

# %% [markdown]
# ## 4. シンボリック値と反復
#
# `@qkernel` シグネチャで `n: qmc.UInt` や `theta: qmc.Float` と書くと、
# トレース時にはこれらの値は**シンボリック**です。実際の値はトランスパイル時に
# `bindings` を通じて提供されます。
#
# この設計により、回路を一度定義すれば、異なるサイズやパラメータで
# 再利用することができます。
#
# ### `qmc.range()` -- シンボリック上限によるループ
#
# Python 組み込みの `range()` はシンボリックな `UInt` 値では動作しません。
# Qamomile は `qmc.range()` を提供しており、`int` と `UInt` の
# 両方の引数を受け付け、トランスパイル時に量子対応のループに展開されます。
#
# ```python
# for i in qmc.range(n):           # 0 から n-1 まで
# for i in qmc.range(start, stop): # start から stop-1 まで
# ```
#
# ループ変数 `i` は `UInt` であり、配列のインデックスとして使用できます。


# %%
@qmc.qkernel
def range_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Apply H gates to all qubits using qmc.range()."""
    qubits = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        qubits[i] = qmc.h(qubits[i])

    return qmc.measure(qubits)


range_demo.draw(n=4, fold_loops=False)

# %% [markdown]
# ### `qmc.items()` -- 辞書の反復
#
# `qmc.items()` は `Dict` ハンドルのキーと値のペアを反復処理します。
# 回路の構造がデータに依存する問題（例えばイジングモデルの結合係数など）で
# 不可欠な機能です。
#
# Dict とその内容はトランスパイル時に `bindings` で提供され、
# ループは**アンロール**されます。各イテレーションが最終回路の
# 具体的なゲートになります。
#
# ```python
# for (i, j), Jij in qmc.items(ising):
#     q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
# ```


# %%
@qmc.qkernel
def items_demo(
    n_qubits: qmc.UInt,
    ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Apply RZZ gates based on Ising coefficients from a Dict."""
    q = qmc.qubit_array(n_qubits, name="q")
    for (i, j), Jij in qmc.items(ising):
        q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
    return qmc.measure(q)


items_demo.draw(n_qubits=1)


# %% [markdown]
# `ising` 辞書は量子ビットペアのインデックスを結合強度に対応付けます。
# トランスパイル時に `bindings` でデータを提供します。

# %%
# Define Ising coefficients: J_{01} = 1.0, J_{12} = -0.5
ising_data = {(0, 1): 1.0, (1, 2): -0.5}

exec_items = transpiler.transpile(
    items_demo,
    bindings={"n_qubits": 3, "ising": ising_data, "gamma": 0.5},
)

# %% [markdown]
# トランスパイルされた回路を確認し、ループが2つの具体的な RZZ ゲートに
# アンロールされていることを検証しましょう。

# %%
qiskit_circuit = exec_items.get_first_circuit()
print("=== Transpiled Circuit ===")
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ## 5. コンテナ型
#
# Qamomileは値をグループ化するためのコンテナ型を提供しています。
#
# ### `Vector[T]` -- 1次元配列
#
# `Vector` は要素型でパラメータ化された1次元配列です。
# 最も一般的な使い方は `Vector[Qubit]`（量子ビットレジスタ）と
# `Vector[Bit]`（測定結果）です。
#
# ```python
# qubits: qmc.Vector[qmc.Qubit] = qmc.qubit_array(n, name="q")
# q = qubits[i]       # 要素を取得（量子ビットを借用）
# qubits[i] = q       # 要素を返却（量子ビットを返す）
# qubits.shape[0]     # シンボリックサイズ（UInt として取得）
# ```
#
# `Vector[Float]` は古典配列（例：パラメータベクトル）の保持にも使えます。

# %% [markdown]
# ### `Dict[K, V]` -- キー・バリューマッピング
#
# `Dict` はキーを値に対応付けます。Qamomileでは主に、
# イジング結合係数などの問題データを回路に渡すために使用されます。
#
# - キー型 `K` は `Tuple[UInt, UInt]`（量子ビットペアのインデックス）
#   または単に `UInt`（単一インデックス）であることが多い
# - 値型 `V` は通常 `Float`（係数）
# - `qmc.items(dict_handle)` で反復処理
# - 実際のデータは `bindings` で通常の Python `dict` として提供
#
# ```python
# # 型アノテーション
# ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]
#
# # 反復処理
# for (i, j), Jij in qmc.items(ising):
#     ...
#
# # トランスパイル時のデータ提供
# transpiler.transpile(kernel, bindings={"ising": {(0, 1): 1.0, (1, 2): -0.5}})
# ```

# %% [markdown]
# ### `Tuple[K, V]` -- 固定サイズのペア
#
# `Tuple` は2つの値のペアを表します。主に `Dict` のキー型として
# マルチインデックスのエントリ（例：量子ビットペア `(i, j)`）に使用されます。
#
# `for ... in qmc.items(d):` ループ内では、タプルキーは自動的に
# アンパックされます：
#
# ```python
# for (i, j), Jij in qmc.items(ising):
#     # i と j は UInt ハンドル
#     # Jij は Float ハンドル
# ```

# %% [markdown]
# ## 6. 特殊型
#
# ### `QFixed` -- 量子固定小数点数
#
# `QFixed` は固定小数点2進数として解釈される量子レジスタを表します。
# **量子位相推定（QPE）**において、測定結果から推定された位相を
# 自動的にデコードするために使用されます。
#
# `QFixed` は量子型（線形型規則の対象）です。
# `qmc.qpe()` の戻り値として得られ、直接構築することはありません。
#
# ```python
# # QFixed は qmc.qpe() の戻り値
# phase: qmc.QFixed = qmc.qpe(target, counting, unitary, **params)
# result: qmc.Float = qmc.measure(phase)
# ```
#
# 実際の使用例は [QPEチュートリアル](05_stdlib.ipynb) を参照してください。
#
# ### `Observable` -- ハミルトニアン参照
#
# `Observable` はハミルトニアン演算子を参照するハンドルです。
# **型アノテーション専用**の型であり、`@qkernel` 内部で構築することはできません。
# 代わりに、Python で `qamomile.observable` を使ってハミルトニアンを構築し、
# `bindings` 経由でカーネルに渡します。
#
# `Observable` は `qmc.expval()` と組み合わせて期待値を計算するために使用されます。
# これは変分量子アルゴリズム（VQE、QAOA）における重要な操作です。
#
# ```python
# import qamomile.observable as qm_o
#
# # ハミルトニアンを Python で構築（@qkernel の外部で）
# H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))
#
# @qmc.qkernel
# def vqe_step(
#     q: qmc.Vector[qmc.Qubit],
#     H: qmc.Observable,
#     theta: qmc.Float,
# ) -> qmc.Float:
#     q[0] = qmc.ry(q[0], theta)
#     q[0], q[1] = qmc.cx(q[0], q[1])
#     return qmc.expval(q, H)  # <psi|H|psi>
#
# # ハミルトニアンを bindings で渡す（変更時は再トランスパイルが必要）
# executable = transpiler.transpile(
#     vqe_step,
#     bindings={"H": H},
#     parameters=["theta"],
# )
# ```
#
# `Observable` は最適化チュートリアルで使用します。

# %% [markdown]
# ## 7. まとめ：各型の使いどころ
#
# | やりたいこと | 使用する型 | 例 |
# |-------------|--------------|---------|
# | 単一量子ビットの作成 | `Qubit` | `q = qmc.qubit(name="q")` |
# | 量子ビットレジスタの作成 | `Vector[Qubit]` | `qubits = qmc.qubit_array(n, name="q")` |
# | 回転角度の受け渡し | `Float` | `theta: qmc.Float` |
# | 配列のインデックスアクセス | `UInt` | `i: qmc.UInt` または `qmc.range()` のループ変数 |
# | 測定結果の格納 | `Bit` | `b = qmc.measure(q)` |
# | 複数の測定結果の格納 | `Vector[Bit]` | `bits = qmc.measure(qubits)` |
# | 問題データ（係数）の受け渡し | `Dict[K, V]` | `ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` |
# | マルチインデックスキーの表現 | `Tuple[K, V]` | `qmc.Tuple[qmc.UInt, qmc.UInt]` |
# | 問題データの反復処理 | `qmc.items(d)` | `for (i, j), Jij in qmc.items(ising):` |
# | シンボリック上限によるループ | `qmc.range(n)` | `for i in qmc.range(n):` |
# | QPE 位相のデコード | `QFixed` | （[QPEチュートリアル](05_stdlib.ipynb) を参照） |
# | 期待値の計算 | `Observable` | （[最適化チュートリアル](../optimization/qaoa.ipynb) を参照） |
#
# ### 覚えておくべきルール
#
# 1. 最もよく使うコンストラクタは `qmc.qubit()` と `qmc.qubit_array()` です。
#    古典コンストラクタ（`qmc.uint()`、`qmc.float_()`、`qmc.bit()`）も
#    ありますが、直接使う必要があることは稀です。
# 2. **量子型**（`Qubit`、`QFixed`）は線形型規則に従います：
#    ゲート適用後は必ず再代入してください。
# 3. **古典型**（`Float`、`UInt`、`Bit`）は自由に再利用できます。
# 4. `@qkernel` 内のループには `qmc.range()` を使います（Python の `range()` ではなく）。
# 5. `Dict` ハンドルの反復には `qmc.items()` を使います。
# 6. データに依存するすべての値（`Dict`、`UInt`、`Float`）は
#    トランスパイル時に `bindings` で提供します。
#
# ### クイックリファレンス：コンストラクタ
#
# ```python
# import qamomile.circuit as qmc
#
# # 量子ビットコンストラクタ（最もよく使う）：
# q = qmc.qubit(name="q")               # Qubit
# qubits = qmc.qubit_array(n, name="q")  # Vector[Qubit]
#
# # 古典コンストラクタ（利用可能だが直接使うことは稀）：
# # qmc.uint(3)                — int または str から UInt を作成
# # qmc.float_(1.5)            — float または str から Float を作成
# # qmc.bit(True)              — bool、str、または int から Bit を作成
#
# # 一般的なパターン — 型アノテーションとして宣言：
# # theta: qmc.Float           — 回転角度（bindings 経由）
# # n: qmc.UInt                — 整数パラメータ（bindings 経由）
# # b: qmc.Bit                 — 測定結果（qmc.measure() の戻り値）
# # ising: qmc.Dict[K, V]      — 問題データ（bindings 経由）
# # H: qmc.Observable           — ハミルトニアン（bindings 経由）
#
# # @qkernel 内での反復：
# for i in qmc.range(n):                      # シンボリック for ループ
# for (i, j), v in qmc.items(dict_handle):    # 辞書の反復
# ```
#
# 次のチュートリアルでは、これらの型を実際に使います：
# - **[03_gates](03_gates.ipynb)**：ゲートの完全リファレンス（全11ゲート）
# - **[04_superposition_entanglement](04_superposition_entanglement.ipynb)**：重ね合わせ、干渉、Bell/GHZ状態
# - **[05_stdlib](05_stdlib.ipynb)**：QFT、QPE と `QFixed`
# - **[optimization/qaoa](../optimization/qaoa.ipynb)**：`Dict`、`Tuple`、`qmc.items()` を使った QAOA

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **Qamomileの型の全カタログ** -- Qamomileは量子型（`Qubit`、`QFixed`）、古典スカラー（`Float`、`UInt`、`Bit`）、コンテナ（`Vector`、`Dict`、`Tuple`、`Matrix`、`Tensor`）、特殊型（`Observable`）を提供します。
# - **量子型と古典型** -- 量子型は線形所有権（消費して返す）を強制し、古典型は自由に再利用できます。
# - **量子ビットと量子ビット配列の作成方法** -- `qmc.qubit(name=...)` と `qmc.qubit_array(n, name=...)` が主要なコンストラクタです。古典コンストラクタ（`qmc.uint()`、`qmc.float_()`、`qmc.bit()`）もありますが、通常は型アノテーションと bindings で提供されます。
# - **線形型エラー：どのようなエラーが出るか、どう修正するか** -- `QubitConsumedError`、`QubitAliasError`、`UnreturnedBorrowError` はトレース時に明確なメッセージで検出されます。修正方法は常に、ゲートの戻り値をキャプチャし、消費済みハンドルの再利用を避けることです。
# - **古典スカラー型：`Float`、`UInt`、`Bit`** -- 通常は関数パラメータとして宣言されるか、操作の戻り値として取得されます。単体のコンストラクタ（`qmc.uint()`、`qmc.float_()`、`qmc.bit()`）は利用可能ですが、直接使うことは稀です。
# - **`qmc.range()` と `qmc.items()` によるシンボリック値と反復** -- `qmc.range(n)` は `UInt` 上限のシンボリックループを作成し、`qmc.items(d)` は `@qkernel` 内で `Dict` ハンドルを反復処理します。
# - **コンテナ型：`Vector`、`Dict`、`Tuple`、`Matrix`、`Tensor`** -- `Vector` は量子ビットレジスタと測定配列を保持し、`Dict` と `Tuple` は `bindings` 経由で構造化された問題データを渡します。
# - **特殊型：`QFixed`、`Observable`** -- `QFixed` はQPEからの固定小数点量子値を表し、`Observable` は変分アルゴリズムでの期待値計算に使用されます。
