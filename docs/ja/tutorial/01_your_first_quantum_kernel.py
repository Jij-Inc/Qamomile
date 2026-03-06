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
# # はじめての量子カーネル
#
# この章では、ゼロから量子プログラムを実行するまでを解説します。
# 最後まで読めば、Qamomile カーネルの記述・可視化・実行に必要な
# すべての行とすべての概念を理解できるようになります。
#
# ## Qamomile とは？
#
# Qamomile は量子回路 SDK です。Python で量子プログラムを記述し、
# サポートされている任意のバックエンド（Qiskit、QuriParts、その他今後対応予定）で
# 実行できます。
# **型付きでシンボリック**なアプローチを採用しており、`@qkernel` デコレータを
# 付けた Python 関数を書くと、Qamomile がそれを中間表現にトレースし、
# 解析・可視化・コンパイルを行います。
#
# 基本的なワークフローは以下のパイプラインです：
#
# ```
# @qkernel 定義  →  draw() / estimate_resources()  →  transpile()  →  sample() / run()  →  .result()
# ```
#
# - **定義**: 型アノテーション付きのカーネル関数を記述します。
# - **検査**: `draw()` で回路図を可視化したり、`estimate_resources()` でコストを見積もります。
# - **トランスパイル**: カーネルをバックエンド固有の実行可能形式にコンパイルします。
# - **実行**: `sample()`（測定ビットの取得）または `run()`（期待値の計算）で実行します。
# - **結果の読み取り**: `.result()` を呼び出して出力を取得します。
#
# 毎回すべてのステップが必要なわけではありません — タスクに合ったものを使ってください。

# %% [markdown]
# ## インストール
#
# 通常の使用：
#
# ```bash
# pip install qamomile
# ```
#
# リポジトリ内での開発：
#
# ```bash
# uv sync
# ```
#
# このチュートリアルでは、具体的なバックエンドとして Qiskit を使用します。
# QuriParts もサポートされており、バックエンドの選択肢は今後も増えていく予定です。

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 最初のカーネル：偏ったコイン
#
# **カーネル**とは、`@qmc.qkernel` デコレータを付けた Python 関数です。
# 型付きハンドルとゲート操作を使って量子回路を記述します。
#
# 最もシンプルな例を作りましょう：1 つの量子ビットを角度 `theta` で回転させ、
# 測定します。`theta` の値に応じて、量子ビットは `0` または `1` に偏ります
# — ちょうど偏ったコインのようなものです。


# %%
@qmc.qkernel
def biased_coin(theta: qmc.Float) -> qmc.Bit:
    # "q" という名前の量子ビットハンドルを作成
    q = qmc.qubit(name="q")

    # RY 回転を適用 — 量子ビットにバイアスをかける
    q = qmc.ry(q, theta)

    # 測定し、結果を古典ビットとして返す
    return qmc.measure(q)


# %% [markdown]
# 注目すべきポイント：
#
# - **型アノテーションは必須です**: `theta: qmc.Float` は theta が浮動小数点
#   パラメータであることを示します。戻り値の型 `qmc.Bit` は、このカーネルが
#   1 つの古典ビットを出力することを示します。
# - **`qmc.qubit(name="q")`** は量子ビットハンドルを作成します。`name` は
#   回路図に表示されます。
# - **`q = qmc.ry(q, theta)`** は RY ゲートを適用し、**`q` を再代入**します。
#   この再代入が重要です — 理由はすぐ後で説明します。
# - **`qmc.measure(q)`** は量子ビットの状態を測定し、`Bit` を返します。

# %% [markdown]
# ## 実行前の検査
#
# 実行する前に、カーネルを検査できます。`draw()` は回路図を表示します：
#
# > **注意**: `draw()` は Qamomile の IR レベルで回路を可視化します。
# > バックエンド（例: Qiskit）にトランスパイルする際、バックエンドがゲートを
# > 分解・最適化する場合があるため、実際に実行される回路は `draw()` の表示と
# > 異なることがあります。バックエンド固有の回路を確認するには `to_circuit()` を
# > 使用してください。

# %%
biased_coin.draw(theta=0.6)

# %% [markdown]
# 実行前にカーネルのコストも確認できます。
# `estimate_resources()` は量子ビット数とゲート数を報告します：

# %%
est = biased_coin.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %% [markdown]
# このシンプルなカーネルでは数値は具体的ですが、パラメータ化されたカーネルでは
# SymPy のシンボリック式になります — これについては
# [チュートリアル 02](02_parameterized_kernels.ipynb) で詳しく説明します。

# %% [markdown]
# ## 実行パイプライン
#
# それでは実際にこのカーネルを実行しましょう。プロセスは 3 つのステップです：
#
# 1. **トランスパイル**: カーネルをバックエンドで実行可能な形式にコンパイルします。
# 2. **実行**: `sample()` を呼び出し、具体的なパラメータ値で実行します。
# 3. **結果の読み取り**: 返された Job に対して `.result()` を呼び出します。
#
# 以下がコードです。各パートを順に説明します：

# %%
# ステップ 1: トランスパイル
# parameters=["theta"] はトランスパイラに「theta は後で指定するので、
# コンパイル済み回路ではスイープ可能なパラメータとして保持せよ」と伝えます。
exe = transpiler.transpile(biased_coin, parameters=["theta"])

# ステップ 2: 実行
# bindings={"theta": ...} は theta の具体的な値を指定します。
# shots=256 は回路を 256 回実行することを意味します。
job = exe.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": math.pi / 4},
)

# ステップ 3: 結果の読み取り
# .result() はジョブが完了するまでブロックし、SampleResult を返します。
# デフォルトのエグゼキュータはローカルシミュレータを使用しますが、
# 独自のカスタムエグゼキュータ（例: 実機やクラウドサービス）を接続することもできます。
result = job.result()

print("sample results:", result.results)

# %% [markdown]
# 3 つの概念を詳しく見てみましょう：
#
# - **`parameters=["theta"]`**（トランスパイル時）は、カーネル入力のうちどれを
#   コンパイル済みプログラム内の調整可能なパラメータとして残すかを宣言します。
#   ここに列挙されて*いない*入力は、トランスパイル時に `bindings` で指定する
#   必要があります（チュートリアル 02 で説明します）。
#
# - **`bindings={"theta": math.pi / 4}`**（実行時）は、パラメータに具体的な値を
#   設定します。
#
# - **`.result()`**: `sample()` は結果を直接返すのではなく、**Job** オブジェクトを
#   返します。`.result()` を呼ぶとジョブの完了を待ち、`SampleResult` を返します。
#   デフォルトのエグゼキュータはローカルシミュレータを使用しますが、Job パターンにより、
#   カーネルコードを変更せずにカスタムエグゼキュータ（例: 実機やクラウドサービス）に
#   差し替えることができます。

# %% [markdown]
# ## `SampleResult` の読み方
#
# `result.results` は `list[tuple[T, int]]` で、以下の構造です：
#
# - `T` は測定出力の型（ここでは `int` — `Bit` の場合 `0` または `1`）
# - `int` はカウント：その結果が何回出現したか
#
# 例えば、`[(0, 150), (1, 106)]` は、256 ショットのうち結果 `0` が 150 回、
# 結果 `1` が 106 回出現したことを意味します。

# %%
for value, count in result.results:
    print(f"  outcome={value}, count={count}")

# %% [markdown]
# `SampleResult` には便利なメソッドも用意されています：

# %%
# 最も多い結果
print("most common:", result.most_common(1))

# 確率分布
print("probabilities:", result.probabilities())

# %% [markdown]
# ## バックエンド回路の確認
#
# `to_circuit()` は**すべての**パラメータをバインドした状態でカーネルをコンパイルし、
# バックエンド固有の回路（例: Qiskit の `QuantumCircuit`）を返します。
# デバッグに便利です — バックエンドが受け取る回路を正確に確認できます。

# %%
qiskit_circuit = transpiler.to_circuit(
    biased_coin,
    bindings={"theta": math.pi / 4},
)
print(qiskit_circuit)

# %% [markdown]
# ## 複数量子ビットの例
#
# 量子ビットを複数使ってみましょう。この例では 2 つの新しい要素を導入します：
#
# - **`qubit_array(n)`**: 複数の量子ビットを一度に確保する
# - **`cx()`**（CNOT ゲート）: 2 量子ビットゲートで、**両方の**ハンドルを返す


# %%
@qmc.qkernel
def two_qubit_demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, name="q")

    q[0] = qmc.h(q[0])  # q[0] に H ゲートを適用
    q[0], q[1] = qmc.cx(q[0], q[1])  # q[0], q[1] に CNOT を適用

    return qmc.measure(q)


# %%
two_qubit_demo.draw()

# %%
demo_result = (
    transpiler.transpile(two_qubit_demo)
    .sample(
        transpiler.executor(),
        shots=256,
    )
    .result()
)

for outcome, count in demo_result.results:
    print(f"  outcome={outcome}, count={count}")

# %% [markdown]
# ここで 2 つのパターンに注目してください：
#
# - **`qubit_array(2)`** は 1 回の呼び出しで複数の量子ビットを作成します。
#   インデックスでアクセスします：`q[0]`、`q[1]`。チュートリアル 02 では
#   `qubit_array(n)` でサイズをシンボリックにする方法を紹介します。
# - **2 量子ビットゲートは両方のハンドルを返します**: `q[0], q[1] = qmc.cx(q[0], q[1])`。
#   両辺を再代入する必要があります。
#
# これは重要なルールにつながります。

# %% [markdown]
# ## アフィンルール
#
# Qamomile では、量子ハンドルは**アフィン**です：ゲートがハンドルを消費したら、
# 以降の操作では返されたハンドルを**必ず**使わなければなりません。
# これは「ムーブ」のようなもので、古いハンドルは無効化されます。
#
# - 1 量子ビットゲート: `q = qmc.h(q)` — 同じ変数に再代入する。
# - 2 量子ビットゲート: `q0, q1 = qmc.cx(q0, q1)` — 両方の変数に再代入する。
#
# ### なぜリニアではなくアフィンなのか？
#
# 量子コンピューティングでは、一時的な計算に使った量子ビットを、
# システムの残りの部分ともつれたまま片付けずに放置すると、
# 他の量子ビットへの操作が予期しない影響を受ける可能性があります。
# 厳密に言えば、**リニア**型システム（すべてのハンドルをちょうど 1 回使用する必要がある）
# が最も安全なモデルです — 一時的な量子ビットを破棄する前に必ず「逆計算」
# （アンコンピュート）することを強制できるからです。
#
# しかし、Python でリニア型を強制すると、単純なプログラムでも書きづらくなります。
# Qamomile では代わりに**アフィン**型を採用しています：ハンドルは**最大 1 回**
# 使用できますが、ドロップ（使わずに捨てること）は許可されます。
# これにより API が Python らしくなり、儀式的なコードなしに自然に書けます。
#
# > **トレードオフ**: 一時的な量子ビットを確保し、メインレジスタともつれさせた後、
# > そのまま忘れてしまった場合でも物理法則は変わりません — その残った量子ビットは
# > 結果を汚染します。コンパイラはこれを検出してくれません。
# > したがって覚えておいてください：**一時的な量子ビットをもつれさせたら、
# > 使い終わる前にアンコンピュートしてください。**
#
# 再代入を忘れるとエラーになります。以下はその例です：

# %%
try:

    @qmc.qkernel
    def bad_rebind() -> qmc.Bit:
        q = qmc.qubit(name="q")
        qmc.h(q)  # 間違い：q を消費したのに結果をキャプチャしていない
        q = qmc.x(q)  # 古い（既に消費された）ハンドルを使っている
        return qmc.measure(q)

    bad_rebind.draw()
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# 修正は簡単です：`qmc.h(q)` ではなく、常に `q = qmc.h(q)` と書いてください。

# %% [markdown]
# ## まとめ
#
# ここまでで以下のことができるようになりました：
#
# - `@qmc.qkernel` でカーネルを定義する
# - 量子ビットの作成、ゲートの適用、測定
# - `draw()` で可視化
# - `transpile()` → `sample()` → `.result()` で実行
# - `SampleResult` の結果を読み取る
# - `to_circuit()` でバックエンド回路を確認する
# - アフィンルール（`q = qmc.gate(q)`）に従う
# - `estimate_resources()` でリソースを見積もる
#
# ## サポートされているバックエンド
#
# Qamomile は同じ `@qkernel` を異なる量子フレームワークにコンパイルします。
# 現在のバックエンドサポート状況：
#
# | バックエンド | ステータス | 備考 |
# |---------|--------|-------|
# | **Qiskit** | サポート済み | 全ゲートセット、制御フロー、オブザーバブル |
# | **QuriParts** | サポート済み | 全ゲートセット、オブザーバブル |
# | **CUDA-Q** | 近日対応予定 | GPU アクセラレーテッドシミュレーション |
#
# > **重要**: すべてのカーネル機能がすべてのバックエンドで利用できるわけではありません。
# > 例えば、カーネル内の `if` 分岐は Qiskit ではサポートされていますが、
# > 他のバックエンドではまだサポートされていない場合があります。選択したバックエンドで
# > 機能が利用できない場合は、トランスパイル時に明確なエラーが表示されます。
# > 迷った場合は、開発時には Qiskit を使い、デプロイ準備ができたら
# > バックエンドを切り替えてください。
#
# ## 次の章
#
# 1. [パラメータ化カーネル](02_parameterized_kernels.ipynb) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
# 2. [リソース推定](03_resource_estimation.ipynb) — シンボリックなコスト分析、ゲート内訳、設計の比較
# 3. [実行モデル](04_execution_models.ipynb) — `sample()` と `run()`、オブザーバブル、ビット順序
# 4. [古典フローパターン](05_classical_flow_patterns.ipynb) — ループ、スパースデータ、分岐
# 5. [再利用パターン](06_reuse_patterns.ipynb) — ヘルパーカーネル、複合ゲート、スタブ
# 6. [デバッグとバックエンド](07_debugging_and_backend.ipynb) — デバッグチェックリスト、エラーメッセージ、クイックリファレンス
