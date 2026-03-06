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
# # リソース推定
#
# 量子回路を実際のハードウェアで実行する前に、そのコストを知りたいものです。
# 量子ビット数はいくつか、ゲート数はいくつか、どのような種類のゲートがあるか。
# Qamomile の `estimate_resources()` は、**回路を実行せずに**これらの
# 質問に答えます。具体的なカーネルでもシンボリック（パラメータ付き）な
# カーネルでも動作します。
#
# この章では以下を扱います:
#
# - 固定カーネルの基本的なリソース推定
# - パラメータ付きカーネルのシンボリック推定
# - `ResourceEstimate` フィールドの完全なリファレンス
# - シンボリック式を扱うための `.substitute()` と `.simplify()`
# - リソースコストによる設計候補の比較

# %%
import qamomile.circuit as qmc

# %% [markdown]
# ## 固定カーネルの推定
#
# パラメータを持たないカーネルに対しては、`estimate_resources()` は
# 具体的な数値を返します。


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)
print("single-qubit gates:", est.gates.single_qubit)
print("two-qubit gates:", est.gates.two_qubit)

# %% [markdown]
# ## シンボリック推定
#
# カーネルが未束縛のパラメータ（例: `n: qmc.UInt`）を持つ場合、
# `estimate_resources()` は **SymPy 式**を返し、パラメータに対する
# コストのスケーリングを示します。これにより、特定の値を選ばなくても
# スケーリングを分析できます。


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
        q[i] = qmc.ry(q[i], theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)
print("single-qubit gates:", est.gates.single_qubit)
print("two-qubit gates:", est.gates.two_qubit)
print("rotation gates:", est.gates.rotation_gates)
print("parameters:", est.parameters)

# %% [markdown]
# 出力には、量子ビット数を表す `n` や総ゲート数を表す `3*n - 1` のような
# SymPy 式が含まれます。これらは近似ではなく厳密な値です。

# %% [markdown]
# ## `ResourceEstimate` フィールドリファレンス
#
# | フィールド | 説明 |
# |-------|------------|
# | `est.qubits` | 論理量子ビット数 |
# | `est.gates.total` | 総ゲート数 |
# | `est.gates.single_qubit` | 単一量子ビットゲート数 |
# | `est.gates.two_qubit` | 2量子ビットゲート数 |
# | `est.gates.multi_qubit` | 多量子ビットゲート数（3量子ビット以上） |
# | `est.gates.t_gates` | Tゲート数 |
# | `est.gates.clifford_gates` | Cliffordゲート数 |
# | `est.gates.rotation_gates` | 回転ゲート数 |
# | `est.gates.oracle_calls` | オラクル呼び出し回数（名前別の辞書） |
# | `est.parameters` | シンボル名から SymPy シンボルへの辞書 |
#
# すべてのフィールドは SymPy 式です。固定カーネルの場合は
# 通常の整数に評価されます。

# %% [markdown]
# ## `.substitute()` によるスケーリング分析
#
# シンボリック式は*数式*を教えてくれますが、特定のサイズでの
# 具体的な数値が欲しいことも多いでしょう。`.substitute()` を使って
# 評価できます:

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit")

# %% [markdown]
# これにより、実行を開始する前に、目標とするスケールで回路が
# 実行可能かどうかを確認できます。

# %% [markdown]
# ## まとめ
#
# - `estimate_resources()` は実行せずに量子ビット数とゲートコストを報告します。
# - パラメータ付きカーネルの場合、結果は正確なスケーリングを示す SymPy 式です。
# - `.substitute(n=...)` を使って特定のサイズで評価し、実行可能性を確認できます。
#
# **次**: [実行モデル](04_execution_models.ipynb) --- `sample()` と `run()`、
# オブザーバブル、ビット順序について。
