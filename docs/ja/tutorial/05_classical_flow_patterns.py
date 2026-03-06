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
# # 古典制御フローパターン
#
# 量子回路はしばしば古典データに依存する構造を持ちます。
# 例えば、量子ビットに対するイテレーション、グラフのエッジに基づくゲート適用、
# ゲートシーケンスの選択などです。Qamomile では `qmc.range`、`qmc.items`、
# `if` 分岐、`while` ループによってこれらのパターンをサポートしています。
#
# この章では以下を扱います:
#
# - `qmc.range()` によるループ（復習とより深い使い方）
# - `qmc.items()` による辞書のイテレーション
# - 測定結果に対する `if` と `while` による回路途中での分岐

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## `qmc.range` ループ
#
# チュートリアル 02 で `qmc.range(n)` を使った単純なループを見ました。
# ここではもう少し豊富な例を示します。全ての量子ビットに H を適用し、
# 隣接するペアを CX でエンタングルします。


# %%
@qmc.qkernel
def hadamard_chain(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # 全ての量子ビットに H を適用
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])

    # 隣接するペアをエンタングル
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
hadamard_chain.draw(n=4)

# %% [markdown]
# ## `qmc.items` によるスパースな相互作用データの処理
#
# 多くの量子アルゴリズム（QAOA、VQE）では、グラフや相互作用マップで
# 決定された特定の量子ビットペアにのみゲートを適用します。全てのペアを
# ループするのではなく、相互作用の**辞書**を渡して `qmc.items()` で
# イテレーションすることができます。
#
# 辞書型は Qamomile のシンボリック型を使用します:
# `qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` — キーは量子ビット
# インデックスのペア、値は相互作用の重みです。


# %%
@qmc.qkernel
def sparse_coupling(
    n: qmc.UInt,
    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # 初期重ね合わせ
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])

    # 指定されたエッジにのみ RZZ 相互作用を適用
    for (i, j), weight in qmc.items(edges):
        q[i], q[j] = qmc.rzz(q[i], q[j], gamma * weight)

    return qmc.measure(q)


# %% [markdown]
# ## `to_circuit()` による確認
#
# `draw()` は全てのパターン（特に複雑な型を伴う `items`）にはまだ対応して
# いません。そのような場合は `to_circuit()` を使用して、全パラメータが
# バインドされた後の具体的なバックエンド回路を確認してください。

# %%
edge_data = {(0, 1): 1.0, (1, 2): -0.7, (0, 2): 0.3}

circuit = transpiler.to_circuit(
    sparse_coupling,
    bindings={"n": 3, "edges": edge_data, "gamma": 0.4},
)
print(circuit)

# %% [markdown]
# `edge_data` の 3 つのエッジのみが RZZ ゲートを生成し、無駄な操作はありません。

# %% [markdown]
# ## `if` 分岐と `while` ループ
#
# Qamomile は**回路途中での測定**とそれに続く古典的な分岐をサポートしています。
# 条件は**測定結果**（`Bit`）でなければならず、カーネルパラメータではありません。
#
# これはハードウェアレベルの条件付き実行に直接対応します:
# 量子ビットを測定し、その結果に基づいて次の操作を決定します。

# %% [markdown]
# ### 測定結果に対する `if`
#
# よくあるパターン: 一つの量子ビットを測定し、その結果に基づいて
# 別の量子ビットに条件付きでゲートを適用します。

# %%
@qmc.qkernel
def conditional_flip() -> qmc.Bit:
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")

    q0 = qmc.x(q0)  # |1⟩ を準備
    bit = qmc.measure(q0)

    # q0 の測定結果に基づいて q1 を条件付きで反転
    if bit:
        q1 = qmc.x(q1)
    else:
        q1 = q1  # 何もしない — 両方の分岐で q1 を扱う必要がある

    return qmc.measure(q1)


# %% [markdown]
# > **注意**: アフィン型システムにより、`if` と `else` の両方の分岐で
# > 同じ量子ビットハンドルを扱う必要があります。true 分岐で `q1` に
# > ゲートを適用する場合、false 分岐でも `q1` を再代入する必要があります
# > （何もしない場合でも `q1 = q1` とします）。

# %% [markdown]
# これは Qiskit の `if_else` 命令にトランスパイルされ、実行できます:

# %%
exe = transpiler.transpile(conditional_flip)
executor = transpiler.executor()
job = exe.sample(executor, bindings={}, shots=100)
result = job.result()
for value, count in result.results:
    print(f"  bit={value}: {count} shots")

# %% [markdown]
# `q0` は |1⟩ として準備されているため、測定結果は常に 1 となり、
# `q1` は常に反転されます。全てのショットで 1 が返るはずです。

# %% [markdown]
# ### 測定結果に対する `while`
#
# `while` ループは測定条件が false になるまで繰り返します。
# これは repeat-until-success プロトコルに有用です。

# %%
@qmc.qkernel
def repeat_until_zero() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.h(q)  # |0⟩ か |1⟩ が 50/50 の確率
    bit = qmc.measure(q)

    while bit:
        # 0 が得られるまで再準備と再測定を繰り返す
        q = qmc.qubit("q2")
        q = qmc.h(q)
        bit = qmc.measure(q)

    return bit


# %% [markdown]
# これは Qiskit の `while_loop` 命令にトランスパイルされます:

# %%
exe = transpiler.transpile(repeat_until_zero)
job = exe.sample(executor, bindings={}, shots=100)
result = job.result()
for value, count in result.results:
    print(f"  bit={value}: {count} shots")

# %% [markdown]
# ループは測定結果が 0 になるまで実行され続けるため、
# 最終結果は常に 0 です。

# %% [markdown]
# ### `if` と `while` の組み合わせ
#
# 両方のパターンを組み合わせることができます。以下は測定を繰り返し行い、
# 条件付きで補正ゲートを適用するプロトコルです:

# %%
@qmc.qkernel
def measure_and_correct() -> qmc.Bit:
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")

    q0 = qmc.h(q0)
    bit = qmc.measure(q0)

    while bit:
        # bit が 1 なら q1 に補正を適用
        if bit:
            q1 = qmc.x(q1)
        else:
            q1 = q1
        # 再準備と再測定
        q0 = qmc.qubit("q0_retry")
        q0 = qmc.h(q0)
        bit = qmc.measure(q0)

    return qmc.measure(q1)


# %%
exe = transpiler.transpile(measure_and_correct)
job = exe.sample(executor, bindings={}, shots=100)
result = job.result()
for value, count in result.results:
    print(f"  bit={value}: {count} shots")

# %% [markdown]
# ## まとめ
#
# - `qmc.range(n)` でシンボリックな範囲に対するループ。
# - `qmc.items(dict)` でスパースなキーバリューデータ（エッジ、重み）のイテレーション。
# - `if bit:` と `while bit:` で**測定結果**に基づく分岐。
#   両方の分岐で同じ量子ビットハンドルを扱う必要があります（アフィン規則）。
# - これらの制御フローパターンはネイティブなバックエンド命令
#   （例: Qiskit の `if_else` や `while_loop`）にトランスパイルされます。
#
# **次へ**: [再利用パターン](06_reuse_patterns.ipynb) — ヘルパーカーネル、
# コンポジットゲート、トップダウン設計のためのスタブゲート。
