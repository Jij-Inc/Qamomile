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
# 2. **`@composite_gate`** — 量子カーネルをカスタム可能な**名前付きゲート**に昇格させ、
#    図中で単一のボックスとして表示します。
#
# さらにトップダウン設計のための第 3 のパターンもあります：
#
# 3. **スタブゲート** — 実装本体を持たないゲートで、リソース推定に使います。
#    例えば、グローバー探索アルゴリズムを設計しており、
#    オラクルが約 40 個の T ゲートを使用することはわかっているが、まだ実装していないとします。
#    スタブゲートを使用すると、完全なオラクル実装なしでアルゴリズムの総コストを推定できます。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## パターン 1: ヘルパー QKernel
#
# どの `@qkernel` 関数も別の `@qkernel` から呼び出せます。
# トランスパイル時にインライン展開されるため、トランスパイル結果はフラットな回路になります。


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
# トランスパイル後の回路ではインライン展開されるため、サブブロックではなく個々の CX ゲートが見えます。

# %%
qc = transpiler.to_circuit(ghz_with_helper, bindings={"n": 4})
print(qc.draw())

# %% [markdown]
# ## パターン 2: `@composite_gate`
#
# 再利用可能なブロックを回路図で**名前付きボックス**として表示したい場合
# `@composite_gate` でを使うこともできます。
# また、より高度な内容としてコンポジットゲートにすることで
# 複数の実装方式を与えるといったカスタム設定を与えることも可能です。
#
# `@qkernel` の代わりに `@composite_gate(name="...")` と書きます：


# %%
@qmc.composite_gate(name="entangle")
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
# ## パターン 3: トップダウン設計のためのスタブゲート
#
# オラクルなどを想定する量子アルゴリズムを設計する場合に内部は未知のまま回路を組みたいこともあると思います。
# **スタブゲート**は実装本体を持たず、名前・量子ビット数・オプションのリソースメタデータだけを持ちます。
#
# オラクルあるいはサブルーチンが開発中でも、アルゴリズム全体のコストを推定できます。
#
# スタブゲートを使うためには `@composite_gate` の引数として `stub=True` を指定します。
# このとき同時にリソース情報を `ResrouceMetadata` として与えられます。


# %%
@qmc.composite_gate(
    stub=True,
    name="oracle",
    num_qubits=3,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
        total_gates=40,
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
# ### スタブゲートによるリソース推定
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
# アルゴリズムレベルのコスト（量子ビット数、オラクルクエリ数等）を
# 確認できます。

# %% [markdown]
# ## まとめ
#
# - **ヘルパー `@qkernel`**：ある量子カーネルから別の量子カーネルを呼び出してコードを再利用できます。
#   トランスパイラがインライン展開し、結果はフラットな回路になります。
# - **`@composite_gate`**：量子カーネルに名前付きの識別子を与え、図で一つのゲートとして可視化します。
#   `@qkernel` の代わりに `@composite_gate` とデコレータを書きます。
# - **スタブゲート**：`stub=True` と `ResourceMetadata` で、
#   実装なしにトップダウン設計とリソース推定が可能です。
