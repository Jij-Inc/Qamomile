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
# # 再利用パターン：QKernelの合成とコンポジットゲート
#
# 回路が大きくなると、ゲート列のコピー＆ペーストを避けたくなります。Qamomileは2つの再利用メカニズムを提供しています：
#
# 1. **ヘルパーQKernel** — ある`@qkernel`を別の`@qkernel`から呼び出す、通常の関数合成と同じ方法です。
# 2. `@composite_gate` — 量子カーネルをカスタム可能な名前付きゲートに昇格させ、図中で単一のボックスとして表示します。
#
# さらにトップダウン設計のための第3のパターンもあります：
#
# 3. **不透明オラクル** — 実装本体を持たない呼び出しで、トップダウン設計とリソース推定に使います。完全なオラクル実装がなくても、呼び出し構造を推定できます。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## パターン1:ヘルパーQKernel
#
# どの`@qkernel`関数も別の`@qkernel`から呼び出せます。トランスパイル時にインライン展開されるため、トランスパイル結果はフラットな回路になります。


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
assert result.shots == 128
assert sum(count for _, count in result.results) == 128
# 4 量子ビット GHZ 状態 → (0, 0, 0, 0) と (1, 1, 1, 1) のみ出現。
assert all(
    outcome in {(0, 0, 0, 0), (1, 1, 1, 1)}
    for outcome, _ in result.results
)

# %% [markdown]
# ヘルパー`entangle_once`により、呼び出し側のコードが読みやすくなります。トランスパイル後の回路ではインライン展開されるため、サブブロックではなく個々のCXゲートが見えます。

# %%
qc = transpiler.to_circuit(ghz_with_helper, bindings={"n": 4})
print(qc.draw())

# %% [markdown]
# ### ヘルパーへのスカラーリテラルの受け渡し
#
# ヘルパーqkernelがスカラー型(`UInt`,`Float`,`Bit`)のパラメータを宣言している場合、呼び出し側ではPythonの生のリテラルをそのまま渡せます。Qamomileが`int`を`UInt`、`float`を`Float`、`bool`を`Bit`に自動昇格します。`helper(q, 0, 0.5)`は`helper(q, qmc.uint(0), qmc.float_(0.5))`と等価です。明示的な`qmc.uint`/`qmc.float_`/`qmc.bit`コンストラクタは、値に名前を付けたい場合や複数の呼び出し箇所で共有したい場合にのみ使えば十分です。


# %%
@qmc.qkernel
def rotate_first(
    q: qmc.Vector[qmc.Qubit],
    idx: qmc.UInt,
    angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    q[idx] = qmc.ry(q[idx], angle)
    return q


@qmc.qkernel
def helper_with_literals(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    # int / float リテラルは runtime で ``qmc.UInt`` / ``qmc.Float`` に
    # auto-promote されるが、kernel の静的シグネチャは Qamomile ハンドル型を
    # そのまま要求する。下の ignore はその意図的なギャップを明示するもの。
    q = rotate_first(q, 0, 0.5)  # type: ignore[arg-type]
    return qmc.measure(q)


helper_with_literals.draw(n=3, fold_loops=False, inline=True)

# %% [markdown]
# ## パターン2：`@composite_gate`
#
# 再利用可能なブロックを回路図で**名前付きボックス**として表示したい場合`@composite_gate`でを使うこともできます。また、より高度な内容としてコンポジットゲートにすることで複数の実装方式を与えるといったカスタム設定を与えることも可能です。
#
# `@qkernel`の上に`@composite_gate(name="...")`を重ねて書きます：


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
# | パターン | `draw()`での表示 | 使用場面 |
# |---------|-------------------|--------------------------|
# | ヘルパー`@qkernel` | インライン展開(フラット) | コードの整理 |
# | `@composite_gate` | 名前付きボックス | ドメインレベルの抽象化/高度なカスタム |

# %% [markdown]
# ## パターン3:トップダウン設計のための不透明オラクル
#
# オラクル内部が未知のまま回路を組みたい場合は、名前・量子ビット数だけを持つ`qmc.Oracle`を使えます。
#
# オラクルあるいはサブルーチンが開発中でも、アルゴリズム全体のコストを推定できます。
#
# 推定時に`UnknownResourcePolicy.OPAQUE_CALL`を指定すると、未実装の呼び出しをopaque call/queryとして記録します。


# %%
oracle_box = qmc.Oracle(
    name="oracle",
    num_qubits=3,
)


@qmc.qkernel
def algorithm_skeleton() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")
    q = qmc.h(q)

    q[0], q[1], q[2] = oracle_box(q[0], q[1], q[2])
    return q


# %%
algorithm_skeleton.draw(fold_loops=False)

# %% [markdown]
# ### 不透明オラクルを含むqkernelのリソース推定
#
# `estimate_resources()`は、オラクル内部が未実装でもqkernel全体を解析できます。既知の回路部分は通常どおり集計され、未知部分は`est.calls.oracle_calls` / `est.calls.oracle_queries`として追跡されます。

# %%
est = algorithm_skeleton.estimate_resources(
    unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
).simplify()
print("qubits:", est.qubits)
assert est.qubits == 3
print("total gates:", est.gates.total)
# Hゲート3個は`gates.total`に、不透明な`oracle_box`は`calls.oracle_calls`に記録されます。
assert est.gates.total == 3

# %% [markdown]
# 次に、通常ゲートと複数の不透明オラクルを混在させたqkernelで確認します。


# %%
phase_oracle = qmc.Oracle(
    name="oracle",
    num_qubits=3,
)


mixing_oracle = qmc.Oracle(
    name="mixing",
    num_qubits=3,
)


@qmc.qkernel
def iterative_oracle_skeleton(rounds: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")

    # 既知の回路部分（非オラクル）
    q[0] = qmc.h(q[0])
    q[1] = qmc.h(q[1])
    q[0], q[1] = qmc.cx(q[0], q[1])

    # ループ外で 1 回オラクル呼び出し
    q[0], q[1], q[2] = phase_oracle(q[0], q[1], q[2])

    # 各ラウンドで既知ゲートと未知オラクルを混在
    for i in qmc.range(rounds):
        q[1] = qmc.ry(q[1], 0.3)
        q[1], q[2] = qmc.cx(q[1], q[2])
        q[0], q[1], q[2] = phase_oracle(q[0], q[1], q[2])
        q[0], q[1], q[2] = mixing_oracle(q[0], q[1], q[2])
        q[1], q[2] = qmc.cx(q[1], q[2])

    return q


iterative_oracle_skeleton.draw(rounds=4, fold_loops=False)


# %%
oracle_est = iterative_oracle_skeleton.estimate_resources(
    unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
).simplify()
print("total gates:", oracle_est.gates.total)
assert str(oracle_est.gates.total) == "3*rounds + 3"
print("two-qubit gates:", oracle_est.gates.two_qubit)
assert str(oracle_est.gates.two_qubit) == "2*rounds + 1"
print("oracle_calls:", oracle_est.calls.oracle_calls)
assert {k: str(v) for k, v in oracle_est.calls.oracle_calls.items()} == {
    "oracle": "rounds + 1",
    "mixing": "rounds",
}
print("oracle_queries:", oracle_est.calls.oracle_queries)
assert {k: str(v) for k, v in oracle_est.calls.oracle_queries.items()} == {
    "oracle": "rounds + 1",
    "mixing": "rounds",
}

# %% [markdown]
# `rounds`に具体的な値を代入して、数値的なカウントを確認します：

# %%
oracle_est_4 = oracle_est.substitute(rounds=4)
print("oracle_calls (rounds=4):", oracle_est_4.calls.oracle_calls)
assert oracle_est_4.calls.oracle_calls == {"oracle": 5, "mixing": 4}
print("oracle_queries (rounds=4):", oracle_est_4.calls.oracle_queries)
assert oracle_est_4.calls.oracle_queries == {"oracle": 5, "mixing": 4}

# %% [markdown]
# この例のように、オラクル内部が不明でも回路解析を進められます。既知部分は通常通りカウントされ、未知オラクル部分は`oracle_calls`（例: `{'phase_oracle': rounds + 1, 'mixing_oracle': rounds}`）と`oracle_queries`（`query_complexity`で重み付け）として追跡されます。

# %% [markdown]
# このように完全な分解を実装する前にアルゴリズムレベルのコスト（量子ビット数、オラクルクエリ数等）を確認できます。

# %% [markdown]
# ## まとめ
#
# - ヘルパー`@qkernel`：ある量子カーネルから別の量子カーネルを呼び出してコードを再利用できます。トランスパイラがインライン展開し、結果はフラットな回路になります。
# - `@composite_gate`：量子カーネルに名前付きの識別子を与え、図で一つのゲートとして可視化します。`@qkernel`の上に`@composite_gate`デコレータを重ねて書きます。
# - **不透明オラクル**：`qmc.Oracle`により、実装なしでトップダウン設計ができます。
# - `est.calls.oracle_calls`：`UnknownResourcePolicy.OPAQUE_CALL`を使うと、名前別の呼び出し回数を確認できます。
#
# 制御ゲート（`qmc.control`）については[チュートリアル04 — 制御ゲート](04_controlled_gates.ipynb)を参照してください。
