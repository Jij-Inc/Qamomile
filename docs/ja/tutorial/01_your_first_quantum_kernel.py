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
# ゼロから量子プログラムを実行するまでの流れを、
# カーネルの記述・可視化・実行を通して一通り学びます。
#
# ## Qamomile とは？
#
# Qamomile は端的に言えば量子プログラミング SDK です。Python で量子プログラムを記述し、
# 記述したプログラムはサポートされている任意の量子SDK
# （Qiskit、QuriParts、その他今後対応予定）で実行できます。
# **型アノテーション付きのシンボリック**なアプローチを採用しており、`@qkernel` デコレータを
# 付けた Python 関数を書くと、Qamomile がそれを中間表現にトレースし、
# その中間表現を通じて各バックエンドで実行します。
#
# 基本的なワークフローは以下のパイプラインです：
#
# ```
# @qkernel 定義  →  draw() / estimate_resources()  →  transpile()  →  sample() / run()  →  .result()
# ```
#
# - **定義**: 型アノテーション付きのカーネル関数を記述します。
# - **検査**: `draw()` で回路図を可視化、`estimate_resources()` で量子ビット数やゲート数を見積もり。
# - **トランスパイル**: カーネルをユーザーが指定した量子SDKで実行可能形式にコンパイル。
# - **実行**: `sample()`（測定ビットの取得）または `run()`（期待値の計算）で実行。
# - **結果の読み取り**: `.result()` で出力を取得。
#
# 毎回すべてのステップが必要なわけではありません — タスクに応じて使い分けてください。

# %% [markdown]
# ## インストール
#
# 通常の使用：
#
# ```bash
# pip install qamomile
# ```
#
# このチュートリアルでは、具体的な量子SDKとして Qiskit を使用します。
# QuriParts もサポートされており、トランスパイル可能な量子SDKは今後も増えていく予定です。

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 最初のカーネル：偏りのあるコイン
#
# **カーネル**とは、`@qmc.qkernel` デコレータを付けた Python 関数です。
# 型付きハンドルとゲート操作を使って量子回路を記述します。
#
# 最もシンプルな例を作りましょう：1 つの量子ビットを角度 `theta` で回転させ、
# 測定します。`theta` の値に応じて、量子ビットは `0` または `1` に偏ります
# 偏りのあるコイントスのようなイメージです。


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
# - **型アノテーションは必須**: `theta: qmc.Float` は theta が浮動小数点
#   パラメータであることを、戻り値の `qmc.Bit` はこのカーネルが
#   1 つの古典ビットを出力することを示します。
# - **`qmc.qubit(name="q")`** は量子ビットハンドルを作成します。`name` は
#   回路図に表示されます。
# - **`q = qmc.ry(q, theta)`** は RY ゲートを適用し、**`q` を再代入**します。
#   この再代入が重要です — 理由は後述します。
# - **`qmc.measure(q)`** は量子ビットの状態を測定し、`Bit` を返します。

# %% [markdown]
# ## 実行前の検査
#
# 実行前にカーネルを検査できます。`draw()` で回路図を確認しましょう：
#
# > **注意**: `draw()` は Qamomile の IR レベルで回路を可視化します。
# > 対象の量子SDKへのトランスパイル時にゲートが分解・最適化されることがあるため、
# > 実際に実行される回路は `draw()` の表示と異なる場合があります。
# > 対象量子SDK固有の回路を確認するには `to_circuit()` を使ってください。

# %%
biased_coin.draw(theta=0.6)

# %% [markdown]
# コストも事前に確認できます。
# `estimate_resources()` で量子ビット数やゲート数を推定します：

# %%
est = biased_coin.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %% [markdown]
# このカーネルでは具体的な数値が返りますが、パラメータ付きカーネルでは
# SymPy を用いた代数的なリソース推定も可能です —
# [チュートリアル 02](02_parameterized_kernels.ipynb) で詳しく扱います。

# %% [markdown]
# ## 実行パイプライン
#
# 実際にカーネルを実行してみましょう。3 ステップです：
#
# 1. **トランスパイル**: カーネルをユーザーが指定した量子SDKで実行可能形式にコンパイル。
# 2. **実行**: `sample()` で具体的なパラメータ値を与えて実行。
# 3. **結果の読み取り**: Job の `.result()` で出力を取得。
#
# 各パートを順に見ていきます：

# %%
# ステップ 1: トランスパイル
# parameters=["theta"] はトランスパイラに「theta は後で指定するので、
# コンパイル済み回路では調整可能なパラメータとして保持せよ」と伝えます。
exe = transpiler.transpile(biased_coin, parameters=["theta"])

# ステップ 2: 実行
# bindings={"theta": ...} は theta の具体的な値を指定します。
# shots=256 は回路を 256 回実行することを意味します。
# デフォルトのexecutor（transpiler.executor()）はローカルシミュレータを使用しますが、
# 独自のカスタムexecutor（例: 実機やクラウドサービス）を接続することもできます。
job = exe.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": math.pi / 4},
)

# ステップ 3: 結果の読み取り
# .result() はジョブが完了するまで待ち、SampleResult を返します。
result = job.result()

print("sample results:", result.results)

# %% [markdown]
# 3 つの概念を押さえておきましょう：
#
# - **`parameters=["theta"]`**（トランスパイル時）— カーネル入力のうち、
#   コンパイル後もスイープ可能なパラメータとして残すものを宣言します。
#   ここに列挙されていない入力は、トランスパイル時に `bindings` で
#   値を与える必要があります（チュートリアル 02 で詳しく扱います）。
#
# - **`bindings={"theta": math.pi / 4}`**（実行時）— パラメータに
#   具体的な値を設定します。
#
# - **`.result()`** — `sample()` は結果を直接返すのではなく **Job** を返します。
#   `.result()` でジョブの完了を待ち、`SampleResult` を取得します。
#   デフォルトの executor はローカルシミュレータですが、Job パターンのおかげで
#   カーネルコードを変えずにカスタム executor（実機やクラウドサービス）に
#   差し替えられます。

# %% [markdown]
# ## `SampleResult`
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
# ## トランスパイル後の回路の確認
#
# `to_circuit()` はすべてのパラメータをバインドした状態でコンパイルし、
# 量子SDK固有の回路（例: Qiskit の `QuantumCircuit`）を返します。
# 量子SDKの形式で回路を確認できるので、デバッグに便利です。

# %%
qiskit_circuit = transpiler.to_circuit(
    biased_coin,
    bindings={"theta": math.pi / 4},
)
print(qiskit_circuit)

# %% [markdown]
# ## 複数量子ビットの例
#
# 量子ビットを複数使ってみましょう。ここでは 2 つの新しい要素が登場します：
#
# - **`qubit_array(n)`** — 複数の量子ビットをまとめて確保
# - **`cx()`**（CNOT ゲート）— 2 量子ビットゲートで、**両方の**ハンドルを返す


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
# 2 つのパターンに注目してください：
#
# - **`qubit_array(2)`** で複数の量子ビットをまとめて作成し、
#   `q[0]`、`q[1]` のようにインデックスでアクセスします。チュートリアル 02 では
#   `qubit_array(n)` でサイズをシンボリックにする方法を扱います。
# - **2 量子ビットゲートは両方のハンドルを返す**: `q[0], q[1] = qmc.cx(q[0], q[1])`。
#   両辺を再代入する必要があります。
#
# これは次に説明する重要なルールにつながります。

# %% [markdown]
# ## アフィンルール
#
# Qamomile の量子ハンドルは**アフィン**です。ゲートがハンドルを消費したら、
# 以降は返された新しいハンドルを**必ず**使う必要があります。
# Rust のムーブセマンティクスに近く、古いハンドルは無効化されます。
#
# - 1 量子ビットゲート: `q = qmc.h(q)` — 同じ変数に再代入する。
# - 2 量子ビットゲート: `q0, q1 = qmc.cx(q0, q1)` — 両方の変数に再代入する。
#
# ### なぜ線型ではなくアフィンなのか？
#
# 量子コンピューティングでは、一時的な量子ビットをシステムともつれたまま放置すると、
# 他の量子ビットに予期しない影響を与える可能性があります。
# 厳密には **線型**型（すべてのハンドルをちょうど 1 回使う）が最も安全ですが、
# Python で線型型を強制すると書きづらくなるため、
# Qamomile では**アフィン**型を採用しています：ハンドルは**最大 1 回**
# 使用でき、ドロップ（使わずに捨てること）も許可されます。
#
# > **トレードオフ**: 一時的な量子ビットをメインレジスタともつれさせた後、
# > そのまま放置するとその量子ビットが結果に影響を与えることがあります。
# > コンパイラはこれを検出できないので、**一時的な量子ビットをもつれさせたら
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
# ここまでで以下ができるようになりました：
#
# - `@qmc.qkernel` でカーネルを定義
# - 量子ビットの作成、ゲートの適用、測定
# - `draw()` で可視化、`estimate_resources()` でコスト見積もり
# - `transpile()` → `sample()` → `.result()` で実行
# - `SampleResult` から結果を読み取る
# - `to_circuit()` でトランスパイル後の回路を確認
# - アフィンルール（`q = qmc.gate(q)`）に従う
#
# ## サポートされている量子SDK
#
# Qamomile は同じ `@qkernel` を異なる量子フレームワークにコンパイルします。
# 現在のサポート状況：
#
# | 量子SDK | ステータス | 備考 |
# |---------|--------|-------|
# | **Qiskit** | サポート済み | 全ゲートセット、制御フロー、オブザーバブル |
# | **QuriParts** | サポート済み | 全ゲートセット、オブザーバブル |
# | **CUDA-Q** | 近日対応予定 | GPU アクセラレーテッドシミュレーション |
#
# > **注意**: すべての機能がすべての量子SDKで使えるわけではありません。
# > 例えば `if` 分岐は Qiskit では対応済みですが、他のSDKでは未対応の
# > 場合があります。未対応の場合はトランスパイル時にエラーが出ます。
#
# ## 次の章
#
# 1. [パラメータ付きカーネル](02_parameterized_kernels.ipynb) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
# 2. [リソース推定](03_resource_estimation.ipynb) — シンボリックなコスト分析、ゲート内訳、スケーリング分析
# 3. [実行モデル](04_execution_models.ipynb) — `sample()` と `run()`、オブザーバブル、ビット順序
# 4. [古典フローパターン](05_classical_flow_patterns.ipynb) — ループ、スパースデータ、条件分岐
# 5. [再利用パターン](06_reuse_patterns.ipynb) — ヘルパーカーネル、コンポジットゲート、スタブ
