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
# # 再利用パターン：QKernel の合成とコンポジットゲート
#
# 回路が大きくなると、ゲート列のコピー＆ペーストを避けたくなります。
# Qamomile は 2 つの再利用メカニズムを提供しています：
#
# 1. **ヘルパー QKernel** — ある `@qkernel` を別の `@qkernel` から呼び出す、
#    通常の関数合成と同じ方法です。
# 2. **`@composite_gate`** — カーネルを**名前付きゲート**に昇格させ、
#    図中で単一のボックスとして表示します。
#
# さらにトップダウン設計のための第 3 のパターンもあります：
#
# 3. **スタブコンポジット** — 実装本体を持たないゲートで、
#    分解が確定する前のリソース推定に使います。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## パターン 1: ヘルパー QKernel
#
# どの `@qkernel` 関数も別の `@qkernel` から呼び出せます。
# コンパイラが呼び出しをインライン展開するため、結果はフラットな回路です。


# %%
@qmc.qkernel
def entangle_once(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def ghz_with_helper(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = entangle_once(q[i], q[i + 1])

    return qmc.measure(q)


# %%
ghz_with_helper.draw(n=4, fold_loops=False)

# %%
result = (
    transpiler.transpile(ghz_with_helper, bindings={"n": 4})
    .sample(
        transpiler.executor(),
        shots=128,
    )
    .result()
)
print("GHZ result:", result.results)

# %% [markdown]
# ヘルパー `entangle_once` により、呼び出し側のコードが読みやすくなります。
# コンパイル後の回路ではインライン展開されるため、サブブロックではなく個々の CX ゲートが見えます。

# %% [markdown]
# ## パターン 2: `@composite_gate`
#
# 再利用可能なブロックを回路図で**名前付きボックス**として表示したい場合
# `@composite_gate` でを使うこともできます。またコンピジットゲートをにすることでその他のカスタムも可能になります。
#
# `@qkernel` の上に `@composite_gate(name="...")` を重ねます：


# %%
@qmc.composite_gate(name="entangle")
@qmc.qkernel
def entangle_link(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def ghz_with_composite(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = entangle_link(q[i], q[i + 1])

    return qmc.measure(q)


# %%
ghz_with_composite.draw(n=4, fold_loops=False)

# %% [markdown]
# ### どちらを使うべきか?
#
# | パターン | `draw()` での表示 | 使用場面 |
# |---------|-------------------|--------------------------|
# | ヘルパー `@qkernel` | インライン展開(フラット) | コードの整理 |
# | `@composite_gate` | 名前付きボックス | ドメインレベルの抽象化/高度なカスタム |

# %% [markdown]
# ## パターン 3: トップダウン設計のためのスタブコンポジット
#
# オラクルなどを想定する量子アルゴリズムを設計する場合に内部は未知のまま回路を組みたいこともあると思います。
# **スタブコンポジット**は実装本体を持たず、名前・量子ビット数・オプションのリソースメタデータだけを持ちます。
#
# オラクルあるいはサブルーチンが開発中でも、アルゴリズム全体のコストを推定できます。


# %%
@qmc.composite_gate(
    stub=True,
    name="oracle",
    num_qubits=3,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
        t_gates=40,
    ),
)
def oracle_box():
    pass


@qmc.qkernel
def algorithm_skeleton() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")
    for i in qmc.range(3):
        q[i] = qmc.h(q[i])

    q[0], q[1], q[2] = oracle_box(q[0], q[1], q[2])
    return q


# %%
algorithm_skeleton.draw(fold_loops=False)

# %% [markdown]
# ### スタブによるリソース推定
#
# `estimate_resources()` はスタブのメタデータを自動的に取得します。
# メタデータは直接参照することもできます。

# %%
est = algorithm_skeleton.estimate_resources().simplify()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %%
meta = oracle_box.get_resource_metadata()
print("oracle query complexity:", meta.query_complexity)
print("oracle T-gate count:", meta.t_gates)

# %% [markdown]
# このトップダウンアプローチにより、完全な分解を実装する前に
# アルゴリズムレベルのコスト（量子ビット数、ゲート数、オラクルクエリ数）を
# 確認できます。

# %% [markdown]
# ## まとめ
#
# - **ヘルパー `@qkernel`**：あるカーネルから別のカーネルを呼び出してコードを再利用。
#   コンパイラがインライン展開し、フラットな回路になります。
# - **`@composite_gate`**：カーネルに名前付きの識別子を与え、図で
#   可視化します。`@qkernel` の上に重ねて使います。
# - **スタブコンポジット**：`stub=True` と `ResourceMetadata` で、
#   実装なしにトップダウン設計とリソース推定が可能です。
