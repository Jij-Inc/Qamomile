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
# # 古典制御フローパターン
#
# 量子回路の構造は古典制御フローに依存することが多くあります。量子ビットのイテレーション、グラフのエッジに基づくゲート適用、ゲート列の条件分岐などです。Qamomileでは`qmc.range`、`qmc.items`、`if`分岐、`while`ループでこれらをサポートしています。
#
# この章では以下を扱います：
#
# - `qmc.range()`によるループ
# - `qmc.items()`による辞書のイテレーション
# - `d[key]`による辞書の添字参照
# - 測定結果に対する`if` / `while`による回路途中の分岐

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import os

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## `qmc.range`ループ
#
# `qmc.range`は`start`、`stop`、`step`を引数に取ることができます。ここでは偶数番目の量子ビットにHゲートを適用し、隣接ペアをCXでエンタングルする量子カーネルを作ってみます。


# %%
@qmc.qkernel
def hadamard_chain(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # 偶数番目の量子ビットに H を適用
    q[0::2] = qmc.h(q[0::2])

    # 隣接するペアをエンタングル
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
hadamard_chain.draw(n=5, fold_loops=False)

# %% [markdown]
# :::{note}
# `qmc.range`のループ変数は**単一の変数**でなければなりません（例：`for i in qmc.range(n)`）。
# `for [i, j] in qmc.range(n)` のようなタプル・リストのアンパックはサポートされておらず、`SyntaxError`が発生します。
# :::

# %% [markdown]
# ## `qmc.items`によるスパースな相互作用データの処理
#
# 多くの変分アルゴリズムでは、グラフや相互作用マップで決まる特定の量子ビットペアにのみゲートを適用します。全ペアをループするのではなく、相互作用の**辞書**を渡して`qmc.items()`でイテレーションできます。
#
# 辞書型にはQamomileのシンボリック型を使います：`qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` — キーが量子ビットインデックスのペア、値が相互作用の重みです。


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
# :::{note}
# `qmc.items`は以下のループパターンをサポートしています：
#
# - `for key, value in qmc.items(d)` — スカラーキー
# - `for (i, j), value in qmc.items(d)` — タプルキー
# - `for key, value in d.items()` — メソッド呼び出し形式
#
# **value**側は単一の変数でなければなりません。value位置でのタプルアンパック
# （例：`for _, (i, j) in qmc.items(d)`）は**サポートされておらず**、`SyntaxError`が発生します。
# 同様に、`for pair in qmc.items(d)` のような単一ターゲットパターンもサポートされていません。
# :::

# %% [markdown]
# ## 辞書の添字参照（`d[key]`）
#
# `qmc.Dict`は`qmc.items()`でのイテレーションに加えて、`d[key]`で直接参照できます。特に有用なのは、**ある辞書のイテレーションキーで別の辞書を引く**パターンです。ある辞書のスパースな相互作用項をループしながら、エッジごとのスケール係数をもう一方の辞書から取り出せます。


# %%
@qmc.qkernel
def per_edge_angles(
    n: qmc.UInt,
    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gammas: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])

    # 各エッジに固有の角度を、同じ(i, j)キーで参照する
    for (i, j), weight in qmc.items(edges):
        q[i], q[j] = qmc.rzz(q[i], q[j], weight * gammas[(i, j)])

    return qmc.measure(q)


# %%
edge_data = {(0, 1): 1.0, (1, 2): -0.7}
gamma_data = {(0, 1): 0.3, (1, 2): 0.5}

circuit = transpiler.to_circuit(
    per_edge_angles,
    bindings={"n": 3, "edges": edge_data, "gammas": gamma_data},
)
_rzz_angles = sorted(
    float(_instr.operation.params[0])
    for _instr in circuit.data
    if _instr.operation.name == "rzz"
)
# 各RZZの角度は、そのエッジ自身のweight * gammaになる
assert _rzz_angles == sorted([1.0 * 0.3, -0.7 * 0.5])

# %% [markdown]
# :::{note}
# `d[key]`は以下をサポートしています：
#
# - **キー**：itemsループのループ変数（`gammas[i]`、`gammas[(i, j)]`）、`int`定数、両者の混在（`gammas[(0, i)]`）。キーの全要素がコンパイル時定数で辞書データがバインド済みの場合は、hashableなPythonキー（`str`など）も使えます。このとき参照はトレース時に定数へ畳み込まれます。シンボリックなキー要素は`UInt`でなければなりません。
# - **値の型**：スカラー値（`qmc.Float`、`qmc.UInt`、`qmc.Bit`）。コンテナ値（`qmc.Tuple` / `qmc.Vector`）はまだサポートされておらず、`NotImplementedError`が発生します。
# - 存在しないキーはPythonの辞書と同様に、ビルド時に`KeyError`が発生します。
# :::

# %% [markdown]
# ## `transpiler.to_circuit()`による確認
#
# `draw()`は全パターン（特に複雑な型を伴う`items`、`if`、`while`）にはまだ対応していません。そのような場合は`transpiler.to_circuit()`で全パラメータをバインドした後のトランスパイル済みの回路を確認してください。

# %%
edge_data = {(0, 1): 1.0, (1, 2): -0.7, (0, 2): 0.3}

circuit = transpiler.to_circuit(
    sparse_coupling,
    bindings={"n": 3, "edges": edge_data, "gamma": 0.4},
)
print(circuit)
assert circuit.num_qubits == 3
# n=3 → 初期 H 3個 + 測定 3個、len(edge_data)=3 → RZZ ちょうど 3個。
_ops = {}
for _instr in circuit.data:
    _ops[_instr.operation.name] = _ops.get(_instr.operation.name, 0) + 1
assert _ops == {"h": 3, "rzz": 3, "measure": 3}

# %% [markdown]
# `edge_data`の3つのエッジのみがRZZゲートを生成します。

# %% [markdown]
# ## `if`分岐と`while`ループ
#
# Qamomileは**回路途中での測定**に続く古典分岐をサポートしています。条件は**測定結果**（`Bit`）でなければならず、量子カーネルの引数は使えません。
#
# これはハードウェアレベルの条件付き実行に直接対応します：量子ビットを測定し、その結果に基づいて次の操作を決定します。

# %% [markdown]
# ### 測定結果に対する`if`
#
# よくあるパターンとして、ある量子ビットを測定し、その結果に基づいて別の量子ビットにゲートを条件付きで適用します。


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
        pass

    return qmc.measure(q1)


# %% [markdown]
# これはQiskitの`if_else`命令にトランスパイルされ、実行できます:

# %%
exe = transpiler.transpile(conditional_flip)
if os.environ.get("QAMOMILE_DOCS_TEST") == "1":
    print("docs test mode では dynamic circuit の実行を省略します。")
else:
    executor = transpiler.executor()
    job = exe.sample(executor, bindings={}, shots=100)
    result = job.result()
    for value, count in result.results:
        print(f"  bit={value}: {count} shots")
    # q0 は |1> として準備 → if 分岐で q1 を毎ショット |1> に反転。
    assert result.shots == 100
    assert result.results == [(1, 100)]

# %% [markdown]
# `q0`は |1⟩ として準備されているため、測定結果は常に1となり、`q1`は常に反転されます。全てのショットで1が返るはずです。

# %% [markdown]
# ### 測定結果に対する`while`
#
# `while`ループは測定条件がfalseになるまで繰り返します。これはrepeat-until-successプロトコルに有用です。


# %%
@qmc.qkernel
def repeat_until_zero() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.h(q)  # |0⟩ か |1⟩ が 50/50 の確率
    bit = qmc.measure(q)

    while bit:
        # 0 が得られるまで再準備と再測定を繰り返す。レジスタは body-local な
        # 名前にする。外側の `q` を本体内で確保したレジスタに再束縛する形は、
        # runtime ループが単一のレジスタをリセットなしで再実行するため拒否される。
        q2 = qmc.qubit("q2")
        q2 = qmc.h(q2)
        bit = qmc.measure(q2)

    return bit


# %% [markdown]
# これはQiskitの`while_loop`命令にトランスパイルされます。生成された回路構造を確認できます:

# %%
exe_while = transpiler.transpile(repeat_until_zero)
qc_while = exe_while.compiled_quantum[0].circuit
print(qc_while)
assert qc_while.num_qubits == 2
# `while bit:` は Qiskit の `while_loop` 命令に lower される。
assert "while_loop" in {instr.operation.name for instr in qc_while.data}

# %% [markdown]
# ### `if`と`while`の組み合わせ
#
# 両方のパターンを組み合わせることができます。以下は測定を繰り返し行い、条件付きで補正ゲートを適用するプロトコルです:


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
        # 再準備と再測定(前述と同じく body-local なレジスタ名にする)
        q0_retry = qmc.qubit("q0_retry")
        q0_retry = qmc.h(q0_retry)
        bit = qmc.measure(q0_retry)

    return qmc.measure(q1)


# %%
exe_combined = transpiler.transpile(measure_and_correct)
qc_combined = exe_combined.compiled_quantum[0].circuit
print(qc_combined)
assert qc_combined.num_qubits == 3
assert "while_loop" in {instr.operation.name for instr in qc_combined.data}

# %% [markdown]
# ## まとめ
#
# - `qmc.range(n)`でシンボリック範囲のループ。
# - `qmc.items(dict)`でスパースなキーバリューデータ（エッジ、重み）のイテレーション。
# - `d[key]`で、ある辞書のイテレーションキーによる別の辞書の参照
#   （エッジごとの係数やキャリブレーション用スケール）。
# - `if bit:` / `while bit:`で**測定結果**に基づく分岐。両分岐で同じ量子ビットハンドルを扱う必要があります（アフィンルール）。
# - これらの制御フローは対象の量子SDKのネイティブな命令（例：Qiskitの`if_else`や`while_loop`）にトランスパイルされます。
#
# **次へ**：[再利用パターン](08_reuse_patterns.ipynb) — ヘルパー量子カーネル、composite gate callable、opaque callable。
