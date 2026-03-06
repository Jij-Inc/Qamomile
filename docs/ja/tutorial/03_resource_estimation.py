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
# 量子回路を実機で実行する前に、そのコストを把握しておきたいところです。
# 量子ビット数、ゲート数、ゲートの種類 — Qamomile の `estimate_resources()` は
# **回路を実行せずに**これらを算出します。具体的な（パラメータ固定の）カーネルにも、
# シンボリック（パラメータ付き）なカーネルにも対応しています。
#
# この章では以下を扱います：
#
# - 固定カーネルの基本的なリソース推定
# - パラメータ付きカーネルのシンボリック推定
# - `ResourceEstimate` フィールドリファレンス
# - `.substitute()` によるスケーリング分析

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
# カーネルに未バインドのパラメータ（例: `n: qmc.UInt`）がある場合、
# `estimate_resources()` は **SymPy 式**を返します。特定の値を選ばなくても
# コストのスケーリングが分かります。


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
# シンボリック式は*数式*を示してくれますが、特定のサイズでの
# 具体的な数値も確認したい場合`.substitute()` で評価できます：

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )

# %% [markdown]
# 実行前に、目標のスケールで回路が現実的かどうかを確認できます。

# %% [markdown]
# ## まとめ
#
# - `estimate_resources()` は実行せずに量子ビット数とゲートコストを算出します。
# - パラメータ付きカーネルでは、結果は厳密なスケーリングを示す SymPy 式になります。
# - `.substitute(n=...)` で特定のサイズに代入し、実行可能性を確認できます。
#
# **次へ**：[実行モデル](04_execution_models.ipynb) — `sample()` と `run()`、
# オブザーバブル、ビット順序について。
