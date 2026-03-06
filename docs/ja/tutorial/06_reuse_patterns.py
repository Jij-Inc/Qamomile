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
# # 再利用パターン: QKernel の合成とコンポジットゲート
#
# 回路が大きくなると、ゲート列のコピー&ペーストを避けたくなります。
# Qamomile は2つの補完的な再利用メカニズムを提供しています:
#
# 1. **ヘルパー QKernel** — ある `@qkernel` を別の `@qkernel` から呼び出す、
#    通常の関数合成と同様の方法です。
# 2. **`@composite_gate`** — カーネルを**名前付きゲート**に昇格させ、
#    図中では単一のボックスとして表示され、バックエンド固有の処理が可能になります。
#
# また、トップダウン設計のための第3のパターンもあります:
#
# 3. **スタブコンポジット** — 実装本体を持たないゲートで、
#    分解が確定する前のリソース推定に使用します。

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata

transpiler = QiskitTranspiler()

# %% [markdown]
# ## パターン 1: ヘルパー QKernel
#
# どの `@qkernel` 関数も、別の `@qkernel` から呼び出すことができます。
# コンパイラが呼び出しをインライン展開するため、結果はフラットな回路になります。

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
ghz_with_helper.draw(n=4)

# %%
result = transpiler.transpile(ghz_with_helper, bindings={"n": 4}).sample(
    transpiler.executor(),
    shots=128,
).result()
print("GHZ result:", result.results)

# %% [markdown]
# ヘルパー `entangle_once` により、呼び出し側のコードが読みやすくなります。
# コンパイル後の回路ではインライン展開されるため、サブブロックではなく個々の CX ゲートが見えます。

# %% [markdown]
# ## パターン 2: `@composite_gate`
#
# 再利用可能なブロックを回路図中で**名前付きボックス**として表示したい場合
# (さらにバックエンド固有のネイティブ実装を持つ可能性がある場合)、
# `@composite_gate` で昇格させます。
#
# `@qkernel` の上に `@composite_gate(name="...")` を重ねます:

# %%
@qmc.composite_gate(name="entangle_link")
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
ghz_with_composite.draw(n=4)

# %% [markdown]
# ### どちらを使うべきか?
#
# | パターン | `draw()` での表示 | バックエンド固有の処理 | 使用場面 |
# |---------|-------------------|--------------------------|----|
# | ヘルパー `@qkernel` | インライン展開(フラット) | なし | コードの整理 |
# | `@composite_gate` | 名前付きボックス | あり(エミッターがネイティブ版を提供可能) | ドメインレベルの抽象化 |
#
# 単に繰り返しを避けたい場合はプレーンなヘルパーを使います。
# ブロックに意味のある名前があり、図中で見えるべきで、
# ネイティブバックエンドのサポートの恩恵を受ける可能性がある場合
# (例えば Qiskit がネイティブに実装できる QFT など)は `@composite_gate` を使います。

# %% [markdown]
# ## パターン 3: トップダウン設計のためのスタブコンポジット
#
# すべてのサブコンポーネントを実装する前に、アルゴリズムの構造を設計したい場合があります。
# **スタブコンポジット**は実装本体を持たず、名前、量子ビット数、
# およびオプションのリソースメタデータのみを持ちます。
#
# これにより、オラクルやサブルーチンがまだ開発中の段階でも、
# アルゴリズム全体のコストを推定できます。

# %%
@qmc.composite_gate(
    stub=True,
    name="oracle_box",
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
algorithm_skeleton.draw()

# %% [markdown]
# ### スタブによるリソース推定
#
# `estimate_resources()` はスタブのメタデータを自動的に取得します。
# メタデータを直接クエリすることもできます。

# %%
est = algorithm_skeleton.estimate_resources().simplify()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %%
meta = oracle_box.get_resource_metadata()
print("oracle query complexity:", meta.query_complexity)
print("oracle T-gate count:", meta.t_gates)

# %% [markdown]
# このトップダウンアプローチにより、完全な分解にコミットする前に、
# アルゴリズムレベルのコスト(量子ビット数、オラクルクエリ数、T ゲート予算)
# について検討できます。

# %% [markdown]
# ## まとめ
#
# - **ヘルパー `@qkernel`**: コード再利用のために、あるカーネルから別のカーネルを呼び出します。
#   コンパイラが呼び出しをフラットな回路にインライン展開します。
# - **`@composite_gate`**: カーネルに、図やバックエンドで可視化される名前付きの
#   アイデンティティを与えます。`@qkernel` の上に重ねて使用します。
# - **スタブコンポジット**: `stub=True` と `ResourceMetadata` を使って、
#   完全な実装なしにトップダウン設計とリソース推定を行います。
#
# **次へ**: [デバッグとバックエンド](07_debugging_and_backend.ipynb) — カーネルを
# 動作させるための実践的チェックリスト、よくあるエラーメッセージ、クイックリファレンス。
