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
# # コンポジットゲート
#
# [前回のチュートリアル](05_stdlib.ipynb)では、QFTやIQFTが回路図上で
# 単一のラベル付きボックスとして表示されることを確認しました。これらは
# **コンポジットゲート**と呼ばれ、複数量子ビットの操作を1つの名前付き
# ユニットにまとめたものです。Qamomileでは、`CompositeGate`基底クラスや
# `@composite_gate`デコレータを使って、独自のコンポジットゲートを定義できます。
#
# ## このチュートリアルで学ぶこと
# - `CompositeGate`をサブクラス化してカスタムコンポジットゲートを作成する方法
# - 解析用のリソースメタデータの付与
# - より簡単なケースでの`@composite_gate`デコレータの使い方
# - ゲートレベルの実装なしでリソース推定を行うスタブゲートの作成
#
# ## なぜCompositeGateを使うのか
#
# - **カプセル化**: 複数のゲートを1つの名前付き操作にまとめられる
# - **再利用性**: 作成したゲートを複数のカーネルで使い回せる
# - **バックエンド最適化**: バックエンドがネイティブ実装を提供できる
# - **リソース推定**: 解析用のリソースメタデータを付与できる
# - **分解戦略**: 複数の実装方法をサポートできる

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. `CompositeGate`のサブクラス化
#
# カスタムゲートを作成するには、`CompositeGate`をサブクラス化し、
# 以下を実装します：
#
# 1. `num_target_qubits`（プロパティ）: ゲートが作用する量子ビット数
# 2. `_decompose(qubits)`: フロントエンド操作を使ったゲートロジック
# 3. `_resources()`（オプション）: リソースメタデータ

# %%
from qamomile.circuit import CompositeGate
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


class MyTwoQubitGate(CompositeGate):
    """A custom 2-qubit gate: H on first qubit, then CNOT."""

    custom_name = "my_gate"

    def __init__(self):
        pass

    @property
    def num_target_qubits(self) -> int:
        return 2

    def _decompose(self, qubits):
        q0, q1 = qubits
        q0 = qmc.h(q0)
        q0, q1 = qmc.cx(q0, q1)
        return q0, q1

    def _resources(self):
        return ResourceMetadata(
            t_gate_count=0,
            custom_metadata={
                "num_h_gates": 1,
                "num_cx_gates": 1,
                "total_gates": 2,
            },
        )


# %% [markdown]
# ## 2. QKernelでのカスタムゲートの使用
#
# ゲートをインスタンス化し、`@qkernel`内で関数のように呼び出します。
# 個々の量子ビットを引数として受け取り、量子ビットのタプルを返します。


# %%
@qmc.qkernel
def use_custom_gate() -> tuple[qmc.Bit, qmc.Bit]:
    """Use MyTwoQubitGate inside a circuit."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    gate = MyTwoQubitGate()
    q0, q1 = gate(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


use_custom_gate.draw()

# %% [markdown]
# `expand_composite=True`を指定すると、ボックスの中のゲートを確認できます：

# %%
use_custom_gate.draw(expand_composite=True)

# %% [markdown]
# ### リソースの確認

# %%
gate = MyTwoQubitGate()
resources = gate.get_resource_metadata()

print("=== MyTwoQubitGate Resources ===")
print(f"  Custom metadata: {resources.custom_metadata}")

# %% [markdown]
# ## 3. `@composite_gate`デコレータ
#
# より簡単なケースでは、Qamomileは`@composite_gate`デコレータを提供しています。
# これは`@qkernel`関数を`CompositeGate`としてラップするもので、
# クラスを完全に記述する必要がありません。


# %%
@qmc.composite_gate
@qmc.qkernel
def bell_gate(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Create a Bell state: H on q0, then CNOT."""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


# %%
@qmc.qkernel
def use_bell_gate() -> tuple[qmc.Bit, qmc.Bit]:
    """Use the decorator-based bell_gate."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    q0, q1 = bell_gate(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


use_bell_gate.draw()

# %% [markdown]
# ### クラスとデコレータの使い分け
#
# | 機能 | `CompositeGate`サブクラス | `@composite_gate`デコレータ |
# |---------|--------------------------|----------------------------|
# | リソースメタデータ | `_resources()`で完全に制御可能 | 非対応 |
# | 分解戦略 | `_strategies`レジストリで対応 | 非対応 |
# | パラメータ付きの構築 | `__init__`の引数で対応 | クロージャ/定数で対応 |
# | 簡潔さ | ボイラープレートが多い | 最小限のコード |
# | 適した用途 | ライブラリ用ゲート、設定可能なゲート | 単発の簡易ゲート |

# %% [markdown]
# ## 4. リソース推定用のスタブゲート
#
# ゲートレベルの実装がまだ用意できていないコンポーネント（例えばGroverの
# アルゴリズムにおけるオラクル）のリソースを推定したい場合があります。
# `@composite_gate(stub=True, ...)`を使うと、リソースのアノテーション付きの
# プレースホルダーゲートを作成できます。**分解は行われません**。


# %%
@qmc.composite_gate(
    stub=True,
    name="oracle",
    num_qubits=5,
    query_complexity=1,
    t_gate_count=100,
)
def oracle():
    pass


# %%
@qmc.qkernel
def grover_iteration() -> qmc.Vector[qmc.Bit]:
    """A single Grover iteration using a stub oracle."""
    q = qmc.qubit_array(5, name="q")

    # Superposition
    for i in qmc.range(5):
        q[i] = qmc.h(q[i])

    # Oracle (stub — no gate-level implementation)
    q[0], q[1], q[2], q[3], q[4] = oracle(q[0], q[1], q[2], q[3], q[4])

    return qmc.measure(q)


grover_iteration.draw()

# %%
stub_resources = oracle.get_resource_metadata()

print("=== Stub Oracle Resources ===")
print(f"  Query complexity: {stub_resources.query_complexity}")
print(f"  T-gate count:     {stub_resources.t_gate_count}")

# %% [markdown]
# スタブゲートは回路図上でラベル付きボックスとして表示され、推定用の
# リソースメタデータを保持していますが、ゲートレベルの分解は持ちません。
# これはトップダウンの回路設計に便利で、まずアルゴリズムの構造を定義し、
# 後から実装を埋めていくことができます。

# %% [markdown]
# ## 5. まとめ
#
# ### 主要なクラス
#
# | クラス | モジュール | 用途 |
# |-------|--------|---------|
# | `CompositeGate` | `qamomile.circuit` | カスタムゲートの基底クラス |
# | `ResourceMetadata` | `qamomile.circuit.ir.operation.composite_gate` | リソース推定データ |
#
# ### カスタムCompositeGateのパターン
#
# ```python
# class MyGate(CompositeGate):
#     custom_name = "my_gate"
#
#     def __init__(self, ...):
#         ...
#
#     @property
#     def num_target_qubits(self) -> int:
#         return N
#
#     def _decompose(self, qubits):
#         q0, q1 = qubits
#         # ... gate operations ...
#         return q0, q1
#
#     def _resources(self):
#         return ResourceMetadata(t_gate_count=0)
# ```
#
# ### 次のチュートリアル
#
# - [初めての量子アルゴリズム](07_first_algorithm.ipynb): Deutsch-Jozsaアルゴリズム
# - [リソース推定](09_resource_estimation.ipynb): ゲート数と回路の深さの推定
# - [QAOA](../optimization/qaoa.ipynb): QAOAによる組合せ最適化問題の解法

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **`CompositeGate`によるカスタムコンポジットゲートの作成** — `CompositeGate`をサブクラス化して、プラグイン可能な分解戦略を持つ再利用可能な多量子ビット操作を定義する方法。
# - **リソースメタデータの付与** — `_resources()`をオーバーライドして`ResourceMetadata`を返すことで、ゲート数の解析が可能になります。
# - **`@composite_gate`デコレータ** — クラスによるアプローチの軽量な代替手段。単発のコンポジットゲートを素早く作成できます。
# - **リソース推定用のスタブゲート** — `@composite_gate(stub=True, ...)`を使って、リソースアノテーション付きのプレースホルダーゲートを作成し、分解なしでトップダウンの回路設計が可能になります。
