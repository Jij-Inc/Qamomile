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
# tags: [tutorial]
# ---
#
# # Vectorのスライシング
#
# [チュートリアル02](02_parameterized_kernels.ipynb)ではパラメータ付き量子カーネルの構築方法と、単一量子ビットゲートのブロードキャスト機能を紹介しました。本章では、より複雑な量子カーネルを記述するのに役立つQamomileの機能、**スライシング**を紹介します。

# %%
# 最新のQamomileをpipからインストールします。
# # !pip install "qamomile[qiskit,visualization]"

# %%
import qamomile.circuit as qmc

from qamomile.circuit.transpiler.errors import (
    AffineTypeError,
    QubitBorrowConflictError,
    UnreturnedBorrowError,
)

# %% [markdown]
# ## スライスの基本構文
#
# `Vector[Qubit]`をPythonの`slice`(`start:stop:step`)でインデックスすると、**VectorView**が返されます。これは親Vectorの部分範囲を指すハンドルですが、ほとんどのケースでは通常の`Vector[Qubit]`と同じように扱えます。


# %%
@qmc.qkernel
def demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    # スライスで偶数番目の量子ビットを取り出す
    evens = q[0::2]
    # スライスで奇数番目の量子ビットを取り出す
    odds = q[1::2]
    # 取り出した量子ビット配列のshapeをもとにループし、Hゲートを適用する
    for i in qmc.range(evens.shape[0]):
        evens[i] = qmc.h(evens[i])
    # ブロードキャスト機能によりスライス全体にXゲート(単一量子ビットゲート)を適用する
    odds = qmc.x(odds)
    # 元の量子ビット配列にスライスを戻す
    q[0::2] = evens
    q[1::2] = odds
    return qmc.measure(q)


demo.draw(fold_loops=False)

# %% [markdown]
# `demo`で注目すべき点:
#
# - `evens = q[0::2]`は`q`の偶数番目の量子ビットを覆う`VectorView`を作成します。同様に`odds = q[1::2]`は`q`の奇数番目の量子ビットを覆う`VectorView`を作成します。
# - ループ内の`evens[i] = qmc.h(evens[i])`はチュートリアル02で見たアフィン型のパターンと同じで、各要素は消費され、新しい要素ハンドルがその位置に格納されます。
# - `VectorView`は`Vector[Qubit]`とほとんど同じように使用できます。

# %% [markdown]
# ## インライン短縮形
#
# 本体がブロードキャスト可能な単一の操作なら、`VectorView`に名前を付ける必要はありません。借りる、変換する、返すを1ステートメントで完結できます。


# %%
@qmc.qkernel
def demo_inline() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    q[0::2] = qmc.h(q[0::2])
    q[1::2] = qmc.x(q[1::2])
    return qmc.measure(q)


demo_inline.draw(fold_loops=False)

# %% [markdown]
# ## ネストしたスライス
#
# `VectorView`そのものをさらにスライスし、新たな`VectorView`を取り出すこともできます。結果は親`VectorView`の部分範囲を覆う`VectorView`となります。各階層は1つ上の階層から借り、各階層は祖父にあたる階層に触る前に直接の親に返さなければなりません。
#
# 例えば、`outer = q[0::2]`と`inner = outer[1:3]`があるとき、返却順序は**inner → outer → root**です。まず`inner`を`outer`に、次に`outer`を`q`に返します。


# %%
@qmc.qkernel
def nested_slice() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    outer = q[0::2]
    inner = outer[1:3]
    for i in qmc.range(inner.shape[0]):
        inner[i] = qmc.h(inner[i])
    outer[1:3] = inner
    q[0::2] = outer
    return qmc.measure(q)


nested_slice.draw(fold_loops=False)

# %% [markdown]
# ## `VectorView`をヘルパーカーネルに渡す
#
# `VectorView`は外部の量子カーネルに渡されるときは`Vector[Qubit]`と同様に扱われます。


# %%
@qmc.qkernel
def h_all(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    return qmc.h(v)


@qmc.qkernel
def x_all(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    return qmc.x(v)


@qmc.qkernel
def demo_helper() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    evens = q[0::2]
    odds = q[1::2]
    evens = h_all(evens)
    odds = x_all(odds)
    q[0::2] = evens
    q[1::2] = odds
    return qmc.measure(q)


demo_helper.draw(inline=True, fold_loops=False)

# %% [markdown]
# これは`qamomile.circuit.stdlib`の組み込み関数にも適用されます。`qmc.qft`、`qmc.iqft`、`qmc.qpe`はいずれも`Vector[Qubit]`を取り、`VectorView`に対しても同様に動きます。


# %%
@qmc.qkernel
def qft_on_window() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    q[1:4] = qmc.qft(q[1:4])
    return qmc.measure(q)


qft_on_window.draw()

# %% [markdown]
# ## スライシングによる量子ビット配列の分割
#
# Qamomileでは、量子ビット配列`Vector[Qubit]`そのものを分割したり統合したりすることはできません。一方で、スライシング機能を駆使すると、一つの大きな量子ビット配列から複数の`VectorView`に分けて取り出すことができ、あたかも複数のレジスタがあるかのようにアルゴリズムを書けます。ここでは、量子ビット全体に量子フーリエ変換をかけたあと、その一部にだけ再度量子フーリエ変換をかけてみます。Qamomileの量子フーリエ変換`qft`は`Vector[Qubit]`を引数に取るため、もしスライシング機能がなければ、量子フーリエ変換自体を量子カーネルの中にゼロから書く必要があります。スライシングを使って量子ビット配列の一部を取り出せば、この手間を回避できます。


# %%
@qmc.qkernel
def demo_separation() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(6, name="qs")
    # 量子フーリエ変換を全体にかける
    qs = qmc.qft(qs)
    # 前半にだけ再度量子フーリエ変換をかける
    partial_qs = qs[0:3]
    partial_qs = qmc.qft(partial_qs)
    # 元の量子ビット配列にスライスを戻す
    qs[0:3] = partial_qs
    return qmc.measure(qs)


demo_separation.draw(inline=True, fold_loops=False)

# %% [markdown]
# ## `VectorView`のエラーパターン
#
# `VectorView`は、親となる量子ビット配列`Vector[Qubit]`からスライスで指定された量子ビットを**借りる**ことで作成されます。親となる`Vector[Qubit]`から見ると、指定された量子ビットは貸し出された状態になるため、対象の量子ビットが返却されるまで、親となる`Vector[Qubit]`から直接アクセスすることはできません。同様の理由で、親となる`Vector[Qubit]`全体を消費するような操作も許されません。貸し出した量子ビットを返却するには、スライス代入を行う必要があります。
#
# ここでは、貸し出した量子ビットを返さずにアクセスする例を通じてエラーパターンを見ていきます。


# %%
@qmc.qkernel
def direct_parent_access() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    v = q[1:3]
    q[1] = qmc.h(q[1])  # 貸出済みの量子ビットに直接アクセスしているためエラー
    q[1:3] = v
    return qmc.measure(q)


try:
    direct_parent_access.draw()
except QubitBorrowConflictError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError(
        "expected QubitBorrowConflictError, but draw() returned normally"
    )


# %%
@qmc.qkernel
def forgot_return() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    evens = q[0::2]
    for i in qmc.range(evens.shape[0]):
        evens[i] = qmc.h(evens[i])
    return qmc.measure(q)  # VectorViewを返却していないためエラー


try:
    forgot_return.draw()
except UnreturnedBorrowError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError("expected UnreturnedBorrowError, but draw() returned normally")


# %%
@qmc.qkernel
def overlapping_views() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    a = q[0:3]
    b = q[
        2:5
    ]  # aで貸し出した量子ビットを返却せずに二重に貸し出そうとしているためエラー
    q[0:3] = a
    q[2:5] = b
    return qmc.measure(q)


try:
    overlapping_views.draw()
except QubitBorrowConflictError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError(
        "expected QubitBorrowConflictError, but draw() returned normally"
    )


# %%
@qmc.qkernel
def invalid_nested_return() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    outer = q[0::2]
    inner = outer[1:3]
    for i in qmc.range(inner.shape[0]):
        inner[i] = qmc.h(inner[i])
    q[0::2] = outer  # innerをouterに返却せずに、outerをrootに戻そうとしているためエラー
    return qmc.measure(q)


try:
    invalid_nested_return.draw()
except AffineTypeError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError("expected AffineTypeError, but draw() returned normally")


# %% [markdown]
# ### ループや分岐の内部でviewを返却する
#
# viewを返却するスライス代入は、viewを作成したのと**同じスコープ**で行う必要があります。外側で作成したviewを`for` / `while`本体（や`if`分岐）の内部で返却することは拒否されます。コンパイラは借用状態を1つの静的なテーブルで追跡しており、本体は0回も複数回も実行され得るため、「このviewは返却されたかもしれないし、されていないかもしれない」という状態を表現できないからです。ループ本体のケースはスライス境界が解決された後のトランスパイル時に検出されるので、エラーは`draw()`ではなく`transpile()`から送出されます。

# %%
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()


@qmc.qkernel
def release_inside_loop() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(4, name="q")
    even = q[0::2]
    for _ in qmc.range(2):
        q[0::2] = even  # 外側のviewを本体内部で返却しているためエラー
    return qmc.measure(q)


try:
    transpiler.transpile(release_inside_loop)
except ValidationError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError(
        "expected ValidationError, but transpile() returned normally"
    )

# %% [markdown]
# 修正方法は、借用と返却のサイクルを1つのスコープ内に収めることです。スライス代入を制御フローの外側で行う（このページのこれまでの例はすべてこの形です）か、ループ本体自体がviewを必要とする場合は、viewを本体の**内部**で作成して、各イテレーションが局所的に借用・返却するようにします:


# %%
@qmc.qkernel
def view_per_iteration() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(4, name="q")
    for _ in qmc.range(2):
        even = q[0::2]  # 本体の内部で借用し...
        even = qmc.x(even)
        q[0::2] = even  # ...同じ本体の内部で返却する
    return qmc.measure(q)


view_per_iteration.draw()

# %% [markdown]
# **次へ**: [制御ゲート](04_controlled_gates.ipynb) — `qmc.control`によるビルトインゲートやサブカーネルの制御、concrete/symbolicの制御数の指定、合成できないパターンのカタログを扱います。
