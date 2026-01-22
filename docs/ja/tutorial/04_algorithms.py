# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # 量子アルゴリズム入門：Deutsch-Jozsaアルゴリズム
#
# このチュートリアルでは、最初の本格的な量子アルゴリズムである
# **Deutsch-Jozsaアルゴリズム**を学びます。
# このアルゴリズムは、量子コンピュータが古典コンピュータより優れていることを
# 示す最初の例として歴史的に重要です。
#
# ## このチュートリアルで学ぶこと
# - 量子アルゴリズムとは何か
# - オラクル（ブラックボックス関数）の概念
# - Deutsch-Jozsa問題と量子的な解法
# - 量子並列性と干渉の活用
# - 次のステップ：より高度なアルゴリズムへ

# %%
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. 量子アルゴリズムとは
#
# **量子アルゴリズム**は、量子コンピュータの特性を活用して問題を解く手順です。
#
# 量子アルゴリズムが活用する主な量子特性：
#
# 1. **重ね合わせ**: 複数の状態を同時に処理
# 2. **エンタングルメント**: 量子ビット間の強い相関
# 3. **量子干渉**: 正しい答えの確率を高め、間違った答えの確率を下げる
#
# Deutsch-Jozsaアルゴリズムは、これらすべてを使う最もシンプルな例です。

# %% [markdown]
# ## 2. Deutsch-Jozsa問題
#
# ### 問題設定
#
# ある関数 $f: \{0,1\}^n \rightarrow \{0,1\}$ があるとします。
# この関数は以下のどちらかであることが保証されています：
#
# - **定数関数（Constant）**: すべての入力に対して同じ値を返す
#   - 例: f(00)=0, f(01)=0, f(10)=0, f(11)=0（すべて0）
#   - 例: f(00)=1, f(01)=1, f(10)=1, f(11)=1（すべて1）
#
# - **均衡関数（Balanced）**: 入力の半分に0、半分に1を返す
#   - 例: f(00)=0, f(01)=0, f(10)=1, f(11)=1
#   - 例: f(00)=0, f(01)=1, f(10)=0, f(11)=1
#
# **問題**: 関数が「定数」か「均衡」かを判定せよ
#
# ### 古典的なアプローチ
#
# 古典コンピュータでは、最悪の場合 $2^{n-1} + 1$ 回の関数呼び出しが必要です。
# （半分+1個の入力を試して、すべて同じなら定数、異なる結果が出たら均衡）
#
# ### 量子的なアプローチ
#
# 量子コンピュータでは、**たった1回**の関数呼び出しで判定できます！

# %% [markdown]
# ## 3. オラクル（Oracle）とは
#
# 量子アルゴリズムでは、関数 $f$ を「オラクル」として扱います。
# オラクルは**ブラックボックス**として与えられ、内部構造は不明です。
#
# ### 量子オラクルの構造
#
# 量子オラクルは、入力レジスタと補助ビット（アンシラ）に作用します：
#
# $$U_f |x\rangle |y\rangle = |x\rangle |y \oplus f(x)\rangle$$
#
# - $|x\rangle$: 入力（変化しない）
# - $|y\rangle$: 補助ビット
# - $\oplus$: XOR演算
# - $f(x)$: 関数の出力

# %% [markdown]
# ## 4. アルゴリズムの実装
#
# まず、様々なオラクルを定義します。

# %%
# === 定数オラクル（常に0を返す）===
@qm.qkernel
def oracle_constant_0(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """定数関数: f(x) = 0 for all x"""
    # 何もしない（常に0を返す）
    return inputs, ancilla


# === 定数オラクル（常に1を返す）===
@qm.qkernel
def oracle_constant_1(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """定数関数: f(x) = 1 for all x"""
    # アンシラを反転（常に1を返す効果）
    ancilla = qm.x(ancilla)
    return inputs, ancilla


# === 均衡オラクル（XORパリティ）===
@qm.qkernel
def oracle_balanced_xor(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """均衡関数: f(x) = x_0 XOR x_1 XOR ... XOR x_{n-1}"""
    n = inputs.shape[0]
    for i in qm.range(n):
        inputs[i], ancilla = qm.cx(inputs[i], ancilla)
    return inputs, ancilla


# === 均衡オラクル（最初のビットのみ）===
@qm.qkernel
def oracle_balanced_first_bit(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """均衡関数: f(x) = x_0（最初のビットの値）"""
    inputs[0], ancilla = qm.cx(inputs[0], ancilla)
    return inputs, ancilla


# %% [markdown]
# ### Deutsch-Jozsaアルゴリズムの本体
#
# アルゴリズムの手順：
#
# 1. 入力レジスタを $|0\rangle^{\otimes n}$、アンシラを $|1\rangle$ に初期化
# 2. すべてに Hadamard ゲートを適用
# 3. オラクルを適用
# 4. 入力レジスタに再び Hadamard を適用
# 5. 入力レジスタを測定
#
# **結果の解釈**:
# - すべて $|0\rangle$ → 定数関数
# - それ以外 → 均衡関数

# %%
@qm.qkernel
def deutsch_jozsa_constant_0(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with constant oracle (f=0)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    # Step 1: 初期化
    ancilla = qm.x(ancilla)  # アンシラを |1⟩ に

    # Step 2: Hadamard をすべてに適用
    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    # Step 3: オラクルを適用
    inputs, ancilla = oracle_constant_0(inputs, ancilla)

    # Step 4: 入力レジスタに Hadamard を適用
    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    # Step 5: 入力レジスタを測定
    return qm.measure(inputs)


@qm.qkernel
def deutsch_jozsa_constant_1(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with constant oracle (f=1)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    ancilla = qm.x(ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    inputs, ancilla = oracle_constant_1(inputs, ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    return qm.measure(inputs)


@qm.qkernel
def deutsch_jozsa_balanced_xor(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with balanced oracle (XOR)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    ancilla = qm.x(ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    inputs, ancilla = oracle_balanced_xor(inputs, ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    return qm.measure(inputs)


@qm.qkernel
def deutsch_jozsa_balanced_first(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with balanced oracle (first bit)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    ancilla = qm.x(ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    inputs, ancilla = oracle_balanced_first_bit(inputs, ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    return qm.measure(inputs)


# %% [markdown]
# ## 5. 実行と結果

# %%
n = 3  # 入力ビット数

test_cases = [
    ("定数 (f=0)", deutsch_jozsa_constant_0),
    ("定数 (f=1)", deutsch_jozsa_constant_1),
    ("均衡 (XOR)", deutsch_jozsa_balanced_xor),
    ("均衡 (最初のビット)", deutsch_jozsa_balanced_first),
]

print(f"=== Deutsch-Jozsa アルゴリズム（n={n}）===\n")
print("判定基準: すべて0 → 定数、それ以外 → 均衡\n")

for name, circuit in test_cases:
    exec_dj = transpiler.transpile(circuit, bindings={"n": n})
    result_dj = exec_dj.sample(transpiler.executor(), shots=1000).result()

    print(f"{name}:")
    for value, count in result_dj.results:
        # 判定
        all_zero = all(v == 0 for v in value)
        judgment = "定数" if all_zero else "均衡"
        print(f"  結果: {value}, 回数: {count}, 判定: {judgment}")
    print()

# %% [markdown]
# ### 結果の解釈
#
# - **定数オラクル**: 測定結果は常に `(0, 0, 0)` → 「定数」と正しく判定
# - **均衡オラクル**: 測定結果は `(0, 0, 0)` 以外 → 「均衡」と正しく判定
#
# 重要なのは、**1回の測定だけ**で判定できていることです！
# 古典的には最悪 $2^{n-1}+1 = 5$ 回の関数呼び出しが必要でした。

# %% [markdown]
# ## 6. 回路の可視化

# %%
# 均衡オラクル（XOR）の回路を可視化
qiskit_dj = transpiler.to_circuit(deutsch_jozsa_balanced_xor, bindings={"n": 3})
print("=== Deutsch-Jozsa 回路（均衡オラクル XOR、n=3）===")
print(qiskit_dj.draw(output="text"))

# %% [markdown]
# ### 回路の構造
#
# 1. 最初の列: Hadamard ゲート（重ね合わせを作る）
# 2. 中央: オラクル（CNOT ゲートで実装）
# 3. 最後の列: Hadamard ゲート（干渉を起こす）
# 4. 測定

# %% [markdown]
# ## 7. なぜ動くのか？（直感的な説明）
#
# ### 量子並列性
#
# Hadamard ゲートで重ね合わせを作ると、入力レジスタは
# すべての可能な入力 $|00...0\rangle, |00...1\rangle, ..., |11...1\rangle$ の
# 重ね合わせになります。
#
# オラクルを1回呼び出すだけで、**すべての入力に対する $f(x)$ の情報**が
# 量子状態に書き込まれます。
#
# ### 量子干渉
#
# 2回目の Hadamard で、正しい答えの確率振幅が**強め合い**、
# 間違った答えの確率振幅が**打ち消し合います**。
#
# - 定数関数: すべての振幅が同じ位相 → $|00...0\rangle$ に集中
# - 均衡関数: 半分が逆位相 → $|00...0\rangle$ がキャンセル

# %% [markdown]
# ## 8. 量子アルゴリズムの威力
#
# Deutsch-Jozsa は「おもちゃの問題」ですが、重要な教訓を与えてくれます：
#
# | 指標 | 古典 | 量子 |
# |------|------|------|
# | クエリ数 | $O(2^n)$ | $O(1)$ |
# | 確実性 | 確定的 | 確定的（100%正しい） |
#
# この**指数関数的な高速化**は、より実用的なアルゴリズムでも実現されています：
#
# - **Grover のアルゴリズム**: データベース検索で $\sqrt{N}$ 倍高速化
# - **Shor のアルゴリズム**: 整数の素因数分解で指数関数的高速化
# - **QAOA**: 組合せ最適化問題の近似解

# %% [markdown]
# ## 9. 次のステップ
#
# おめでとうございます！ここまでで、Qamomile を使った量子プログラミングの基礎を習得しました。
#
# ### 学んだこと
#
# 1. **01_introduction.py**: Qamomile の基本、線形型システム
# 2. **02_single_qubit.py**: 重ね合わせ、回転ゲート、パラメータ化
# 3. **03_entanglement.py**: エンタングルメント、ベル状態、GHZ状態
# 4. **04_algorithms.py**: Deutsch-Jozsa アルゴリズム
#
# ### 次に学ぶべきチュートリアル
#
# より高度な量子アルゴリズムを学ぶ準備ができました：
#
# - **`qpe.py`（量子位相推定）**: 固有値を推定する重要なサブルーチン
#   - `qm.controlled()` で制御ゲートを作る
#   - `QFixed` 型で自動的に位相をデコード
#   - Shor のアルゴリズムや量子化学シミュレーションの基盤
#
# - **`qaoa.py`（量子近似最適化アルゴリズム）**: 組合せ最適化問題を解く
#   - パラメータ化された回路の高度な使い方
#   - 古典最適化との組み合わせ（ハイブリッドアルゴリズム）
#   - `rzz` ゲートを使ったコスト関数のエンコード

# %% [markdown]
# ## 10. まとめ
#
# ### Qamomile プログラミングのパターン
#
# ```python
# import qamomile.circuit as qm
# from qamomile.qiskit import QiskitTranspiler
#
# # 1. 量子カーネルを定義
# @qm.qkernel
# def my_algorithm(n: int) -> qm.Vector[qm.Bit]:
#     qubits = qm.qubit_array(n, name="q")
#
#     # 量子操作（線形型: 必ず再代入！）
#     for i in qm.range(n):
#         qubits[i] = qm.h(qubits[i])
#
#     return qm.measure(qubits)
#
# # 2. トランスパイルして実行
# transpiler = QiskitTranspiler()
# executable = transpiler.transpile(my_algorithm, bindings={"n": 3})
# result = executable.sample(transpiler.executor(), shots=1000).result()
#
# # 3. 結果を解析
# for value, count in result.results:
#     print(f"{value}: {count}")
# ```
#
# ### 重要なポイント
#
# 1. **`@qm.qkernel`**: 量子回路を定義するデコレータ
# 2. **線形型**: `q = qm.h(q)` のように必ず再代入
# 3. **2量子ビットゲート**: `q0, q1 = qm.cx(q0, q1)` で両方受け取る
# 4. **配列**: `qm.qubit_array(n, name="q")` と `qm.range(n)`
# 5. **トランスパイル**: `QiskitTranspiler` でバックエンドに変換
#
# これらの知識を使って、`qpe.py` と `qaoa.py` に挑戦してみてください！
