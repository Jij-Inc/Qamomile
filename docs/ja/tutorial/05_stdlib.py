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
# # Qamomileの標準ライブラリ
#
# Qamomileは、よく使われる量子サブルーチンを集めた**標準ライブラリ**（`stdlib`）と、
# 再利用可能な回路パターンを提供する`algorithm`モジュールを備えています。
#
# `stdlib`には、最も基本的で広く使われるビルディングブロックが含まれています。
# `algorithm`モジュールには、実用上よく登場する追加の回路パターンが収められています。
# エコシステムの発展に伴い、`algorithm`内のよく使われるパターンは`stdlib`に
# 昇格される可能性があり、両者の境界はバージョン間で変化することがあります。
#
# ## このチュートリアルで学ぶこと
# - 標準ライブラリの `qmc.qft()`、`qmc.iqft()`、`qmc.qpe()` の使い方
# - 精度とゲート数のトレードオフを制御する分解戦略
# - IR レベルとバックエンドレベルの回路の違い（なぜ異なりうるのか）
# - `qmc.qpe()` を使った量子位相推定の構築
# - `qamomile.circuit.algorithm` モジュールの概要
#
# ## 現在の標準ライブラリ
#
# | 関数 | 説明 |
# |----------|-------------|
# | `qmc.qft()` | 量子フーリエ変換 |
# | `qmc.iqft()` | 逆量子フーリエ変換 |
# | `qmc.qpe()` | 量子位相推定 |
#
# QFT と IQFT は、プラグイン可能な分解戦略を持つ `CompositeGate` として
# 実装されています。バックエンドはネイティブ実装（例：Qiskit 組み込みの QFT）を
# 提供することも、ゲートレベルの分解にフォールバックすることもできます。
# QPE は、制御ユニタリ演算と IQFT を内部的に組み合わせた、より高レベルの関数です。
#
# 標準ライブラリは、量子コンピューティングのエコシステムが成熟し、
# 新しいパターンが確立されるにつれて拡充されていきます。

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. QFT と IQFT
#
# **量子フーリエ変換**（QFT）は、離散フーリエ変換の量子版です。
# Shorのアルゴリズムや量子位相推定など、多くの量子アルゴリズムで
# 重要なサブルーチンとして使われています。
#
# Qamomile では、`qmc.qft()` と `qmc.iqft()` をフロントエンド関数として
# 提供しています。これらは `Vector[Qubit]` を受け取り `Vector[Qubit]` を返します。
# 他のゲートと同じファクトリ関数パターンに従います：
#
# ```python
# qubits = qmc.qft(qubits)
# ```
#
# ### QFT 回路


# %%
@qmc.qkernel
def qft_demo() -> qmc.Vector[qmc.Bit]:
    """Apply QFT to a 3-qubit register."""
    qubits = qmc.qubit_array(3, name="q")

    # Prepare a non-trivial input state: |101>
    # Ket notation uses big-endian: |q[2] q[1] q[0]>
    # So |101> means q[0]=1, q[1]=0, q[2]=1
    qubits[0] = qmc.x(qubits[0])
    qubits[2] = qmc.x(qubits[2])

    # Apply QFT
    qubits = qmc.qft(qubits)

    return qmc.measure(qubits)


qft_demo.draw()

# %% [markdown]
# デフォルトでは、`draw()` は `CompositeGate`（QFT など）を単一のラベル付き
# ボックスとして表示します。`expand_composite=True` を渡すと、各ボックスが
# 「展開」され、コンポジット演算を構成する個々のゲートが表示されます。
#
# **重要**: `expand_composite=True` は Qamomile の **IR レベル**の分解を
# 表示します。これは最終的なバックエンド回路ではありません。特定のバックエンド
# （例：Qiskit）に実際に出力されるゲートは、バックエンドトランスパイラが
# 自由にゲートの置換・最適化・再配置を行えるため、異なる場合があります。
# この違いの具体例をセクション 2 で確認します。

# %%
qft_demo.draw(expand_composite=True)

# %%
exec_qft = transpiler.transpile(qft_demo)
result_qft = exec_qft.sample(transpiler.executor(), shots=1000).result()

print("=== QFT on |101> ===")
print("After QFT, the state is spread across all basis states\n")
sorted_results = sorted(result_qft.results, key=lambda x: str(x[0]))
for value, count in sorted_results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### IQFT 回路
#
# 逆 QFT は QFT を元に戻す操作です。QFT に続けて IQFT を適用すると、
# 元の状態が復元されるはずです。


# %%
@qmc.qkernel
def qft_iqft_roundtrip() -> qmc.Vector[qmc.Bit]:
    """Apply QFT then IQFT: should recover original state."""
    qubits = qmc.qubit_array(3, name="q")

    # Prepare |110>
    qubits[1] = qmc.x(qubits[1])
    qubits[2] = qmc.x(qubits[2])

    # QFT followed by IQFT
    qubits = qmc.qft(qubits)
    qubits = qmc.iqft(qubits)

    return qmc.measure(qubits)


qft_iqft_roundtrip.draw()

# %%
exec_roundtrip = transpiler.transpile(qft_iqft_roundtrip)
result_roundtrip = exec_roundtrip.sample(transpiler.executor(), shots=1000).result()

print("=== QFT -> IQFT Roundtrip (should recover |110>) ===")
for value, count in result_roundtrip.results:
    print(f"  {value}: {count}")

# %% [markdown]
# 結果は `(0, 1, 1)` が確実に得られ、IQFT が QFT を完全に元に戻すことが
# 確認できます。

# %% [markdown]
# ## 2. 分解戦略
#
# QFT と IQFT の実装は、複数の**分解戦略**をサポートしています。
# 標準の QFT は $O(n^2)$ 個のゲートを使用しますが、大規模な回路では
# $O(nk)$ 個のゲートで済む近似版が好ましい場合があります。
#
# ### 利用可能な戦略の一覧
#
# `qamomile.circuit.stdlib.qft` から `QFT` クラスをインポートして、
# 戦略管理メソッドにアクセスします。

# %%
from qamomile.circuit.stdlib.qft import IQFT, QFT

# List available strategies
print("QFT strategies:", QFT.list_strategies())
print("IQFT strategies:", IQFT.list_strategies())

# %% [markdown]
# ### リソース数の比較
#
# `get_resources_for_strategy()` を使うと、完全な回路を構築せずに
# 異なる戦略のゲート数を比較できます。

# %%
# Compare resources for an 8-qubit QFT
qft_8 = QFT(8)

print("=== Resource Comparison: 8-qubit QFT ===\n")

for strategy_name in QFT.list_strategies():
    resources = qft_8.get_resources_for_strategy(strategy_name)
    meta = resources.custom_metadata
    print(f"Strategy: {strategy_name}")
    print(f"  H gates:    {meta['num_h_gates']}")
    print(f"  CP gates:   {meta['num_cp_gates']}")
    print(f"  SWAP gates: {meta['num_swap_gates']}")
    print(f"  Total:      {meta['total_gates']}")
    print()

# %% [markdown]
# 近似戦略では、小さな角度の回転を打ち切ることで制御位相ゲートの数が
# 大幅に削減されます。パラメータ $k$（打ち切り深度）がトレードオフを制御し、
# $k$ が大きいほど精度は高くなりますが、ゲート数も増えます。
#
# | 戦略 | 打ち切り深度 (k) | CP ゲート数 (n=8) | 誤差 |
# |----------|---------------------|---------------|-------|
# | standard | -- (全結合) | n(n-1)/2 = 28 | 0 |
# | approximate_k2 | 2 | 13 | O(n/2^2) |
# | approximate (k=3) | 3 | ~18 | O(n/2^3) |
# | approximate_k4 | 4 | ~22 | O(n/2^4) |

# %% [markdown]
# ### 特定の戦略を使用する
#
# 特定の戦略で QFT を適用するには、`QFT` クラスを直接使用し、
# `strategy` キーワード引数を渡します。
#
# standard と approximate_k2 の分解を視覚的に比較してみましょう：


# %%
@qmc.qkernel
def qft_standard_4() -> qmc.Vector[qmc.Bit]:
    """QFT with standard (full) strategy."""
    qubits = qmc.qubit_array(4, name="q")
    qft_gate = QFT(4)
    qubits[0], qubits[1], qubits[2], qubits[3] = qft_gate(
        qubits[0],
        qubits[1],
        qubits[2],
        qubits[3],
        strategy="standard",
    )
    return qmc.measure(qubits)


@qmc.qkernel
def qft_approx_k2_4() -> qmc.Vector[qmc.Bit]:
    """QFT with approximate_k2 strategy."""
    qubits = qmc.qubit_array(4, name="q")
    qft_gate = QFT(4)
    qubits[0], qubits[1], qubits[2], qubits[3] = qft_gate(
        qubits[0],
        qubits[1],
        qubits[2],
        qubits[3],
        strategy="approximate_k2",
    )
    return qmc.measure(qubits)


# %% [markdown]
# `expand_composite=True` を使うと、各戦略が **Qamomile IR レベル**で QFT を
# どのように分解するか確認できます。標準版は 4 量子ビットに対して 6 個の
# 制御位相ゲートをすべて含みますが、近似 $k=2$ 版は 5 個のみです。
# これは **IR レベルの表示**であることに注意してください。特定のバックエンドに
# 実際に出力される回路は、バックエンドが独自のネイティブ実装に置き換える
# ことができるため、異なる場合があります（以下で確認します）。

# %%
print("Qamomile IR — Standard QFT (all CP gates):")
qft_standard_4.draw(expand_composite=True)

# %%
print("Qamomile IR — Approximate QFT k=2 (fewer CP gates):")
qft_approx_k2_4.draw(expand_composite=True)

# %% [markdown]
# ### IR レベルとバックエンドレベルの回路
#
# 両方のカーネルを Qiskit にトランスパイルして、バックエンドが実際に
# 何を出力するか見てみましょう。

# %%
qiskit_standard = transpiler.to_circuit(qft_standard_4)
print("Qiskit — Standard QFT:")
print(qiskit_standard.draw(output="text"))

# %%
qiskit_approx = transpiler.to_circuit(qft_approx_k2_4)
print("Qiskit — Approximate QFT k=2:")
print(qiskit_approx.draw(output="text"))

# %% [markdown]
# トランスパイル後の回路はどちらも同じ `QFT` ボックスを示しています。
# これは、Qiskit バックエンドが**ネイティブ QFT エミッタ**を提供しており、
# 要求された戦略に関係なく、すべての QFT 演算を Qiskit 独自の
# `qiskit.circuit.library.QFT` ゲートに置き換えるためです。
#
# これは IR とバックエンドの違いの具体例です。Qamomile の IR は
# *どの戦略が要求されたか*を記録し、`expand_composite=True` の表示では
# 対応する分解が表示されますが、Qiskit バックエンドは自由にそれを
# 独自の最適化された実装に置き換えます。
#
# 戦略の違いは依然として意味があります：
#
# - **リソース推定**: `get_resources_for_strategy()` は、各戦略が使用する
#   ゲート数を正確に報告します。
# - **他のバックエンド**: ネイティブ QFT エミッタを提供しないバックエンドは、
#   Qamomile の分解にフォールバックし、戦略を尊重します。
# - Qiskit の `QFTGate` はすでに `approximation_degree` パラメータを
#   サポートしていますが、Qamomile の Qiskit エミッタはまだ戦略を
#   それに転送していません。
#
# **重要なポイント**: 近似戦略はゲート数と精度のトレードオフを制御します。
# `get_resources_for_strategy()` API（上記参照）は、バックエンドに依存せず
# 戦略を比較する確実な方法です。

# %% [markdown]
# ## 3. `qmc.qpe()` を使った QPE
#
# **量子位相推定**（QPE）は、最も重要な量子サブルーチンの一つです。
# ユニタリ $U$ とその固有状態 $|\psi\rangle$（ただし
# $U|\psi\rangle = e^{2\pi i \varphi}|\psi\rangle$）が与えられたとき、
# QPE は位相 $\varphi$ を推定します。
#
# Qamomile の `qmc.qpe()` は以下を処理します：
# 1. カウンティングレジスタへのアダマールゲート
# 2. 制御 $U^{2^k}$ 演算（自動的なべき乗の繰り返し）
# 3. カウンティングレジスタへの逆 QFT
# 4. 自動デコーディングのための `QFixed` 型への変換
#
# ### シグネチャ
#
# ```python
# qmc.qpe(target, counting, unitary, **params) -> QFixed
# ```
#
# - `target`: 固有状態の量子ビット
# - `counting`: 位相推定用の `Vector[Qubit]`
# - `unitary`: $U$ の1回適用を定義する `@qkernel`
# - `**params`: ユニタリに渡すパラメータ
#
# **重要**: ユニタリは $U$ の**1回**の適用を定義するものです。
# `qmc.qpe()` は内部で `qmc.controlled()` の `power` パラメータを使って
# $U^{2^k}$ の繰り返しを自動的に行います。
#
# ### ユニタリの定義


# %%
@qmc.qkernel
def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Phase gate: P(theta)|1> = e^{i*theta}|1>."""
    return qmc.p(q, theta)


# %% [markdown]
# ### QPE の構築
#
# `qmc.qpe()` が返す `QFixed` 型は、量子固定小数点数を表します。
# `qmc.measure()` で測定すると、推定された位相
# $\varphi = \theta / (2\pi)$ を表す `Float` 値に自動的にデコードされます。


# %%
@qmc.qkernel
def qpe_demo(theta: qmc.Float) -> qmc.Float:
    """Estimate the phase of P(theta)."""
    counting = qmc.qubit_array(3, name="counting")
    target = qmc.qubit(name="target")

    # Prepare eigenstate |1>
    target = qmc.x(target)

    # Run QPE
    phase: qmc.QFixed = qmc.qpe(target, counting, p_gate, theta=theta)

    # Measure and decode
    return qmc.measure(phase)


qpe_demo.draw(fold_loops=False, inline=True)

# %% [markdown]
# ### QPE の結果検証
#
# 位相ゲート $P(\theta)$ の固有値は $e^{i\theta}$ なので、
# 位相は $\varphi = \theta / (2\pi)$ となります。
#
# - $\theta = \pi/2$: 期待値 $\varphi = 0.25$
# - $\theta = \pi/4$: 期待値 $\varphi = 0.125$

# %%
# Test 1: theta = pi/2, expected phase = 0.25
test_theta_1 = math.pi / 2
exec_qpe1 = transpiler.transpile(qpe_demo, bindings={"theta": test_theta_1})
result_qpe1 = exec_qpe1.sample(transpiler.executor(), shots=1024).result()

print("=== QPE with theta = pi/2 (expected phase = 0.25) ===")
for value, count in result_qpe1.results:
    print(f"  Measured phase: {value}, Count: {count}")

# %%
# Test 2: theta = pi/4, expected phase = 0.125
test_theta_2 = math.pi / 4
exec_qpe2 = transpiler.transpile(qpe_demo, bindings={"theta": test_theta_2})
result_qpe2 = exec_qpe2.sample(transpiler.executor(), shots=1024).result()

print("=== QPE with theta = pi/4 (expected phase = 0.125) ===")
for value, count in result_qpe2.results:
    print(f"  Measured phase: {value}, Count: {count}")

# %% [markdown]
# どちらの結果も期待される位相と正確に一致しています。`QFixed` 型と
# `qmc.measure()` が二進小数のデコーディングをすべて自動的に処理するため、
# 手動のビット操作なしに位相の値を直接得ることができます。

# %% [markdown]
# ## 4. アルゴリズムのビルディングブロック
#
# `qamomile.circuit.algorithm` モジュールは、一般的な回路パターンのための
# `@qkernel` ビルディングブロックを提供します。これらは通常のカーネルであり、
# 自分のカーネル内で組み合わせることも、**単独で実行する**こともできます。
#
# | モジュール | 例 |
# |--------|---------|
# | `qaoa` | `superposition_vector`, `qaoa_circuit`, `qaoa_state`, ... |
# | `basic` | `rx_layer`, `ry_layer`, `rz_layer`, `cz_entangling_layer` |
# | `fqaoa` | `fqaoa_layers`, `fqaoa_state`, ... |
#
# 完全な API については、API リファレンスドキュメントを参照してください。
#
# 内容はバージョン間で変更される可能性があります。よく使われるパターンは
# `stdlib` に昇格される場合があり、エコシステムの発展に伴って境界は変化します。

# %% [markdown]
# ### 例：`superposition_vector` の使用
#
# `superposition_vector` は $n$ 量子ビットの均一な重ね合わせ状態を作成します。
# これ自体が `@qkernel` なので、**トランスパイルして単独で実行する**ことも、
# **別のカーネル内から呼び出す**こともできます。

# %%
from qamomile.circuit.algorithm import superposition_vector


@qmc.qkernel
def superposition_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Create uniform superposition using the algorithm module."""
    q = superposition_vector(n)
    return qmc.measure(q)


# %%
exec_sup = transpiler.transpile(superposition_demo, bindings={"n": 4})
result_sup = exec_sup.sample(transpiler.executor(), shots=1000).result()

print("=== Superposition Vector (n=4) ===")
print("All 2^4=16 states should appear with roughly equal probability\n")
sorted_results = sorted(result_sup.results, key=lambda x: str(x[0]))
for value, count in sorted_results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# すべてのアルゴリズムビルディングブロックは同じパターンに従います。
# 通常の `@qkernel` 関数であり、自分の回路に組み合わせることができます。
# これらの実際の活用例は
# [QAOA 最適化チュートリアル](../optimization/qaoa.ipynb) で確認できます。

# %% [markdown]
# ## 5. まとめ
#
# ### 標準ライブラリ関数
#
# | 関数 | モジュール | 入力 | 出力 | 説明 |
# |----------|--------|-------|--------|-------------|
# | `qmc.qft()` | `qamomile.circuit` | `Vector[Qubit]` | `Vector[Qubit]` | 量子フーリエ変換 |
# | `qmc.iqft()` | `qamomile.circuit` | `Vector[Qubit]` | `Vector[Qubit]` | 逆 QFT |
# | `qmc.qpe()` | `qamomile.circuit` | target, counting, unitary | `QFixed` | 量子位相推定 |
#
# ### 主要クラス
#
# | クラス | モジュール | 用途 |
# |-------|--------|---------|
# | `QFT` | `qamomile.circuit.stdlib.qft` | 戦略サポート付き QFT |
# | `IQFT` | `qamomile.circuit.stdlib.qft` | 戦略サポート付き IQFT |
#
# ### 分解戦略
#
# ```python
# from qamomile.circuit.stdlib.qft import QFT
#
# # List strategies
# QFT.list_strategies()  # ['standard', 'approximate', 'approximate_k2', ...]
#
# # Compare resources
# qft = QFT(8)
# resources = qft.get_resources_for_strategy("approximate")
#
# # Use a specific strategy
# qft_gate = QFT(n)
# results = qft_gate(q0, q1, ..., strategy="approximate")
# ```
#
# ### 次のチュートリアル
#
# - [コンポジットゲート](06_composite_gate.ipynb): カスタム `CompositeGate` と `@composite_gate` の作成
# - [初めての量子アルゴリズム](07_first_algorithm.ipynb): Deutsch-Jozsa アルゴリズム
# - [リソース推定](09_resource_estimation.ipynb): ゲート数と回路の深さの推定
# - [QAOA](../optimization/qaoa.ipynb): QAOA で組合せ最適化問題を解く

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **標準ライブラリの `qmc.qft()`、`qmc.iqft()`、`qmc.qpe()` の使い方** -- これらの既製ビルディングブロックにより、量子フーリエ変換と位相推定を1つの関数呼び出しで実行できます。
# - **精度とゲート数のトレードオフを制御する分解戦略** -- `QFT.list_strategies()` と `get_resources_for_strategy()` で、`"standard"` と `"approximate"` などのトレードオフを比較・選択できます。
# - **IR レベルとバックエンドレベルの回路** -- `expand_composite=True` は Qamomile の IR 分解を表示しますが、実際のトランスパイル後の回路とは異なる場合があります。バックエンドレベルの結果を確認するには `transpiler.to_circuit()` を使用します。
# - **`qmc.qpe()` を使った量子位相推定の構築** -- `qmc.qpe()` はユニタリカーネル、ターゲット量子ビット、カウンティング量子ビットを受け取り、自動的にデコードされる `QFixed` 値を返します。
# - **`qamomile.circuit.algorithm` モジュールの概要** -- QAOA レイヤー、回転レイヤー、エンタングリングレイヤーなど、より大きなアルゴリズムに組み合わせられる変分ビルディングブロックが事前に用意されています。
