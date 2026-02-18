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
# # 初めての量子アルゴリズム：Deutsch-Jozsaアルゴリズム
#
# このチュートリアルでは、初めての本格的な量子アルゴリズムである
# **Deutsch-Jozsaアルゴリズム**について学びます。
# Deutsch-Jozsaアルゴリズムは最も初期の量子アルゴリズムの一つで、
# Deutschが提案した1ビットのアルゴリズム（1985年）を$n$ビットに一般化したものです。
# 量子コンピュータが特定の問題を古典コンピュータより指数関数的に速く
# 解けることを初めて明確に示したアルゴリズムの一つです。
#
# ## このチュートリアルで学ぶこと
# - 量子アルゴリズムとは何か
# - オラクル（ブラックボックス関数）の概念
# - Deutsch-Jozsa問題とその量子的な解法
# - オラクルを引数として渡して再利用可能な量子アルゴリズムを構築する方法
# - 量子並列性と干渉の活用

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. 量子アルゴリズムとは
#
# **量子アルゴリズム**とは、量子コンピュータの性質を活用して問題を解く手順のことです。
#
# 量子アルゴリズムが利用する主な量子的性質：
#
# 1. **重ね合わせ**: 複数の状態を同時に処理する
# 2. **量子もつれ**: 量子ビット間の強い相関
# 3. **量子干渉**: 正しい答えの確率を増幅し、誤った答えの確率を減少させる
#
# Deutsch-Jozsaアルゴリズムは、これらすべてを使う最もシンプルな例です。

# %% [markdown]
# ## 2. Deutsch-Jozsa問題
#
# ### 問題設定
#
# 関数 $f: \{0,1\}^n \rightarrow \{0,1\}$ があるとします。
# この関数は、以下のいずれかであることが保証されています：
#
# **定値関数** — すべての入力に対して同じ出力を返す：
#
# | x  | f(x)=0 | f(x)=1 |
# |----|--------|--------|
# | 00 | 0      | 1      |
# | 01 | 0      | 1      |
# | 10 | 0      | 1      |
# | 11 | 0      | 1      |
#
# **均等関数** — 入力の半分に対して0を、残り半分に対して1を返す：
#
# | x  | f₁(x) | f₂(x) |
# |----|-------|-------|
# | 00 | 0     | 0     |
# | 01 | 0     | 1     |
# | 10 | 1     | 0     |
# | 11 | 1     | 1     |
#
# **問題**: 関数が「定値」か「均等」かを判定する
#
# ### 古典的なアプローチ
#
# 古典コンピュータでは、最悪の場合 $2^{n-1} + 1$ 回の関数呼び出しが必要です。
# （半分+1個の入力を試し、すべて同じなら定値、異なる結果が出れば均等）
#
# ### 量子的なアプローチ
#
# 量子コンピュータでは、**たった1回**の関数呼び出しで判定できます！

# %% [markdown]
# ## 3. オラクルとは
#
# 量子アルゴリズムでは、関数 $f$ を「オラクル」として扱います。
# オラクルは**ブラックボックス**として与えられ、内部構造は不明です。
#
# ### 量子オラクルの構造
#
# 量子オラクルは入力レジスタと補助ビット（アンシラ）に作用します：
#
# $$U_f |x\rangle |y\rangle = |x\rangle |y \oplus f(x)\rangle$$
#
# - $|x\rangle$: 入力（変化しない）
# - $|y\rangle$: 補助ビット
# - $\oplus$: XOR演算
# - $f(x)$: 関数の出力

# %% [markdown]
# ## 4. オラクルの定義
#
# 様々なオラクルをQKernel関数として定義しましょう。


# %%
# === Constant oracle (always returns 0) ===
@qmc.qkernel
def oracle_constant_0(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Constant function: f(x) = 0 for all x."""
    # Do nothing (always returns 0)
    return inputs, ancilla


oracle_constant_0.draw(inputs=2)


# %%
# === Constant oracle (always returns 1) ===
@qmc.qkernel
def oracle_constant_1(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Constant function: f(x) = 1 for all x."""
    # Flip the ancilla (effect of always returning 1)
    ancilla = qmc.x(ancilla)
    return inputs, ancilla


oracle_constant_1.draw(inputs=2)


# %%
# === Balanced oracle (XOR parity) ===
@qmc.qkernel
def oracle_balanced_xor(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Balanced function: f(x) = x_0 XOR x_1 XOR ... XOR x_{n-1}."""
    n = inputs.shape[0]
    for i in qmc.range(n):
        inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
    return inputs, ancilla


oracle_balanced_xor.draw(inputs=2, fold_loops=False)


# %%
# === Balanced oracle (first bit only) ===
@qmc.qkernel
def oracle_balanced_first_bit(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Balanced function: f(x) = x_0 (value of first bit)."""
    inputs[0], ancilla = qmc.cx(inputs[0], ancilla)
    return inputs, ancilla


oracle_balanced_first_bit.draw(inputs=2)

# %% [markdown]
# ## 5. Deutsch-Jozsaアルゴリズム
#
# オラクルごとに別々の関数を書く代わりに、任意のオラクルを引数として受け取る
# **再利用可能な** Deutsch-Jozsa関数を作成します。
#
# ### アルゴリズムの手順
#
# 1. 入力レジスタを $|0\rangle^{\otimes n}$ に、アンシラを $|1\rangle$ に初期化
# 2. すべてにアダマールゲートを適用
# 3. オラクルを適用
# 4. 入力レジスタに再度アダマールゲートを適用
# 5. 入力レジスタを測定
#
# **結果の解釈**:
# - すべて $|0\rangle$ -> 定値関数
# - それ以外 -> 均等関数


# %%
def deutsch_jozsa(oracle):
    """Create a Deutsch-Jozsa circuit with the given oracle.

    This is a factory function that returns a QKernel circuit.
    The oracle is captured in the closure and called during tracing.
    """

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        inputs = qmc.qubit_array(n, name="input")
        ancilla = qmc.qubit(name="ancilla")

        # Step 1: Initialize ancilla to |1⟩
        ancilla = qmc.x(ancilla)

        # Step 2: Apply Hadamard to all
        for i in qmc.range(n):
            inputs[i] = qmc.h(inputs[i])
        ancilla = qmc.h(ancilla)

        # Step 3: Apply oracle
        inputs, ancilla = oracle(inputs, ancilla)

        # Step 4: Apply Hadamard to input register
        for i in qmc.range(n):
            inputs[i] = qmc.h(inputs[i])

        # Step 5: Measure input register
        return qmc.measure(inputs)

    return circuit


# %% [markdown]
# これで、任意のオラクルに対してDeutsch-Jozsa回路を作成できます：

# %%
dj_constant_0 = deutsch_jozsa(oracle_constant_0)
dj_constant_1 = deutsch_jozsa(oracle_constant_1)
dj_balanced_xor = deutsch_jozsa(oracle_balanced_xor)
dj_balanced_first = deutsch_jozsa(oracle_balanced_first_bit)

# %% [markdown]
# `draw()` メソッドには `inline` パラメータがあり、呼び出されたカーネル
# （オラクルなど）の表示方法を制御できます：
#
# - `inline=True`: オラクルのゲートを回路図に直接展開し、
#   すべてのゲートをフラットなビューで表示します。
# - `inline=False`（デフォルト）: オラクルをラベル付きのボックスとして表示し、
#   モジュール構造を保持します。実装の詳細に惑わされず、
#   アルゴリズムの全体像を把握するのに便利です。

# %%
# With inlining — all gates visible:
dj_balanced_xor.draw(n=2, fold_loops=False, inline=True)

# %%
# Without inlining — the oracle appears as a named box:
dj_balanced_xor.draw(n=2, fold_loops=False)

# %% [markdown]
# ## 6. 実行と結果

# %%
n = 2  # Number of input bits

test_cases = [
    ("Constant (f=0)", dj_constant_0),
    ("Constant (f=1)", dj_constant_1),
    ("Balanced (XOR)", dj_balanced_xor),
    ("Balanced (first bit)", dj_balanced_first),
]

print(f"=== Deutsch-Jozsa Algorithm (n={n}) ===\n")
print("Decision rule: all 0 -> constant, otherwise -> balanced\n")

for name, circuit in test_cases:
    exec_dj = transpiler.transpile(circuit, bindings={"n": n})
    result_dj = exec_dj.sample(transpiler.executor(), shots=1000).result()

    print(f"{name}:")
    for value, count in result_dj.results:
        # Judgment
        all_zero = all(v == 0 for v in value)
        judgment = "constant" if all_zero else "balanced"
        print(f"  Result: {value}, Count: {count}, Judgment: {judgment}")
    print()

# %% [markdown]
# ### 結果の解釈
#
# - **定値オラクル**: 測定結果が常に `(0, 0)` -> 正しく「定値」と判定
# - **均等オラクル**: 測定結果が `(0, 0)` 以外 -> 正しく「均等」と判定
#
# 重要なのは、**たった1回の測定**でこれを判定できるということです！
# 古典的には、最悪の場合 $2^{n-1}+1 = 3$ 回の関数呼び出しが必要です。

# %% [markdown]
# ## 7. なぜうまくいくのか？（直感的な説明）
#
# ### 量子並列性
#
# アダマールゲートで重ね合わせを作ると、入力レジスタは
# すべての可能な入力 $|00...0\rangle, |00...1\rangle, ..., |11...1\rangle$ の重ね合わせになります。
#
# たった1回のオラクル呼び出しで、**すべての入力に対する $f(x)$ の情報**が
# 量子状態にエンコードされます。
#
# ### 位相キックバック（量子もつれの役割）
#
# アンシラ量子ビットは $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$ に準備されます。
# オラクルが $U_f$ を適用すると、入力レジスタとアンシラが一時的にもつれます。
# ここでの重要な恒等式は：
#
# $$U_f |x\rangle |-\rangle = (-1)^{f(x)} |x\rangle |-\rangle$$
#
# アンシラは $|-\rangle$ のまま（変化なし）ですが、入力レジスタには
# 位相因子 $(-1)^{f(x)}$ が付きます。これを**位相キックバック**と呼びます。
# アンシラとの一時的なもつれを通じて、$f(x)$ の情報が入力レジスタの位相に
# 「蹴り返される」のです。
#
# このもつれを介した位相の転送がなければ、アルゴリズムは機能しません。
#
# ### 量子干渉
#
# 2回目のアダマールにより、正しい答えの確率振幅が**強め合い**、
# 誤った答えの確率振幅が**打ち消し合います**。
#
# - 定値関数: すべての振幅が同じ位相 -> $|00...0\rangle$ に集中
# - 均等関数: 半分が逆位相 -> $|00...0\rangle$ が相殺される

# %% [markdown]
# ## 8. 量子アルゴリズムの威力
#
# Deutsch-Jozsaは「おもちゃの問題」ですが、重要な教訓を与えてくれます：
#
# | 指標 | 古典 | 量子 |
# |--------|-----------|---------|
# | クエリ回数 | $O(2^n)$ | $O(1)$ |
# | 確実性 | 決定的 | 決定的（100%正確） |
#
# この**量子スピードアップ**は、より実用的なアルゴリズムでも実現されています：
#
# - **Groverのアルゴリズム**: データベース検索で $\sqrt{N}$ の高速化
# - **Shorのアルゴリズム**: 整数の素因数分解で指数関数的な高速化

# %% [markdown]
# ## 9. まとめ
#
# このチュートリアルでは、Qamomileで構築する初めての完全な量子アルゴリズムとして
# **Deutsch-Jozsaアルゴリズム**を扱いました。
#
# ### 重要なポイント
#
# 1. **オラクルパターン**: オラクルは `@qkernel` 関数として定義し、
#    `(inputs, ancilla) -> (inputs, ancilla)` という標準的なシグネチャを持たせることで、
#    相互に交換可能になります。
#
# 2. **引数としてのオラクル**: アルゴリズムをオラクルを受け取るファクトリ関数で
#    ラップすることで、アルゴリズムの構造と問題固有のロジックを分離できます：
#
#    ```python
#    def deutsch_jozsa(oracle):
#        @qmc.qkernel
#        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
#            ...
#            inputs, ancilla = oracle(inputs, ancilla)
#            ...
#            return qmc.measure(inputs)
#        return circuit
#    ```
#
# 3. **量子スピードアップ**: このアルゴリズムは、古典的な $O(2^n)$ に対して
#    $O(1)$ 回のオラクルクエリで定値か均等かを判定します。
#
# 4. **3つの量子的要素**: 重ね合わせ（並列評価）、もつれを介した位相キックバック
#    （情報転送）、干渉（答えの抽出）。
#
# ### 次のチュートリアル
#
# - [標準ライブラリ](05_stdlib.ipynb): QFT、IQFT、QPE と `qmc.controlled()` および `QFixed`
# - [パラメトリック回路](08_parametric_circuits.ipynb): 変分量子アルゴリズムとハイブリッド最適化

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **量子アルゴリズムとは何か** — 重ね合わせと干渉を活用して古典的なアプローチに対する計算上の優位性を得るために設計された、構造化された量子ゲートの手順。
# - **オラクル（ブラックボックス関数）の概念** — オラクルは問題固有のロジックを `@qkernel` 関数としてエンコードし、アルゴリズムのテンプレートに引数として渡すことができる。
# - **Deutsch-Jozsa問題とその量子的な解法** — 関数が定値か均等かを1回のクエリで判定する。古典的には $2^{n-1}+1$ 回必要。
# - **オラクルを引数として渡して再利用可能な量子アルゴリズムを構築する方法** — 引数としてのオラクルパターンにより、アルゴリズムの構造と問題固有のロジックを分離し、異なるオラクルでの再利用を可能にする。
# - **量子並列性と干渉の活用** — アダマールゲートがすべての入力の重ね合わせを作成し、オラクルが位相をマークし、最後のアダマールが位相の差を測定可能な結果に変換する。
