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
# # Qamomile入門：量子コンピューティングの第一歩
#
# このチュートリアルでは、Qamomileを使って量子コンピューティングの基礎を学びます。
# プログラミング経験があれば、量子コンピューティングの予備知識は必要ありません。
#
# ## このチュートリアルで学ぶこと
# - Qamomileとは何か、その特徴
# - 量子ビット（qubit）の基本概念
# - 最初の量子回路の作成と実行
# - Qamomileの重要な設計思想：線形型システム

# %% [markdown]
# ## 1. Qamomileとは
#
# **Qamomile**は、Pythonで量子回路を記述するためのライブラリです。
# 以下のような特徴があります：
#
# 1. **Pythonライクな記法**: `@qkernel`デコレータを使って、普通のPython関数のように量子回路を書けます
# 2. **型安全**: 型アノテーションにより、コンパイル時にエラーを検出できます
# 3. **マルチバックエンド**: 同じコードをQiskit、QuriParts、PennyLaneなど様々なSDKで実行できます
# 4. **線形型システム**: 量子ビットの状態を安全に追跡し、よくあるバグを防ぎます

# %% [markdown]
# ## 2. 量子ビット（Qubit）とは
#
# ### 古典ビットと量子ビット
#
# 普通のコンピュータでは、**ビット**（bit）が情報の最小単位です。
# ビットは **0** か **1** のどちらか一方の値しか持てません。
#
# ```
# 古典ビット: 0 または 1
# ```
#
# 一方、量子コンピュータでは**量子ビット**（qubit）を使います。
# 量子ビットは、0と1の**重ね合わせ状態**を取ることができます。
#
# ```
# 量子ビット: |0⟩ と |1⟩ の重ね合わせ（測定するまでどちらか決まらない）
# ```
#
# 記号の説明：
# - `|0⟩`（ケット0）: 量子ビットの「0」状態
# - `|1⟩`（ケット1）: 量子ビットの「1」状態
#
# ### 測定
#
# 量子ビットを**測定**すると、重ね合わせ状態が崩れ、0か1のどちらかの値が得られます。
# どちらが出るかは確率的に決まります。

# %% [markdown]
# ## 3. 最初の量子回路
#
# それでは、Qamomileを使って最初の量子回路を作りましょう。
# まずは必要なモジュールをインポートします。

# %%
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ### X ゲート（NOTゲート）
#
# 最も基本的な量子ゲートの1つが **Xゲート** です。
# Xゲートは、量子ビットの状態を反転させます：
# - `|0⟩` → `|1⟩`
# - `|1⟩` → `|0⟩`
#
# 古典コンピュータの「NOTゲート」と同じ働きです。

# %%
@qm.qkernel
def x_gate_circuit() -> qm.Bit:
    """Xゲートを適用する最初の量子回路"""
    # 量子ビットを作成（初期状態は |0⟩）
    q = qm.qubit(name="q")

    # Xゲートを適用して |0⟩ → |1⟩ に変換
    q = qm.x(q)

    # 測定して結果を返す
    return qm.measure(q)


# %% [markdown]
# ### コードの解説
#
# 1. **`@qm.qkernel`**: この関数が量子カーネル（量子回路を定義する関数）であることを示すデコレータ
# 2. **`-> qm.Bit`**: 戻り値の型。測定結果は古典ビット（`Bit`）になります
# 3. **`qm.qubit(name="q")`**: 量子ビットを1つ作成。初期状態は `|0⟩`
# 4. **`q = qm.x(q)`**: Xゲートを適用。**重要：結果を必ず再代入します**
# 5. **`qm.measure(q)`**: 量子ビットを測定し、古典ビットとして結果を取得

# %% [markdown]
# ## 4. 線形型システム：Qamomileの重要な設計思想
#
# 上のコードで、なぜ `q = qm.x(q)` と書くのか疑問に思ったかもしれません。
# これはQamomileの**線形型システム**によるものです。
#
# ### なぜ再代入が必要か
#
# 量子ビットは「コピー」できない性質があります（量子力学の「複製不可能定理」）。
# Qamomileでは、この性質を型システムで表現しています。
#
# ```python
# # 正しい書き方
# q = qm.x(q)  # ゲート適用後、新しい状態を q に再代入
#
# # 間違った書き方（エラーになります）
# qm.x(q)      # 再代入せずに q を使い続けると、古い状態を参照してしまう
# ```
#
# この設計により、「同じ量子ビットを2回使ってしまう」といったバグを防げます。

# %% [markdown]
# ### 線形型のエラー例：実際に試してみよう
#
# 実際に間違った書き方をするとどうなるか見てみましょう。

# %%
# ダメな例1: 同じ量子ビットを2回使う
@qm.qkernel
def bad_example_reuse() -> tuple[qm.Bit, qm.Bit]:
    q = qm.qubit(name="q")
    q1 = qm.h(q)   # q を消費して q1 に
    q2 = qm.x(q)   # ダメ！q は既に q1 で使われている
    return qm.measure(q1), qm.measure(q2)


# %%
# この回路をトランスパイルしようとするとエラーになります
try:
    transpiler_test = QiskitTranspiler()
    transpiler_test.transpile(bad_example_reuse)
    print("エラーが発生しませんでした（予期しない動作）")
except Exception as e:
    print(f"エラーが発生しました（これが正しい動作です）:")
    print(f"  {type(e).__name__}: {e}")

# %% [markdown]
# ### ダメな例2: 戻り値を無視する
#
# ゲートの戻り値を無視して、古い変数を使い続けるのもダメです。

# %%
@qm.qkernel
def bad_example_ignore_return() -> qm.Bit:
    q = qm.qubit(name="q")
    qm.h(q)        # ダメ！戻り値を無視している
    qm.x(q)        # ダメ！古い q を使っている
    return qm.measure(q)  # これも古い q


# %%
try:
    transpiler_test = QiskitTranspiler()
    transpiler_test.transpile(bad_example_ignore_return)
    print("エラーが発生しませんでした（予期しない動作）")
except Exception as e:
    print(f"エラーが発生しました（これが正しい動作です）:")
    print(f"  {type(e).__name__}: {e}")

# %% [markdown]
# ### 正しい書き方
#
# 常にゲートの戻り値を同じ変数に再代入しましょう。

# %%
@qm.qkernel
def good_example() -> qm.Bit:
    q = qm.qubit(name="q")
    q = qm.h(q)    # 正しい！結果を q に再代入
    q = qm.x(q)    # 正しい！更新された q を使う
    return qm.measure(q)


# %%
# 正しい回路は問題なくトランスパイルできます
transpiler_test = QiskitTranspiler()
executable_good = transpiler_test.transpile(good_example)
result_good = executable_good.sample(transpiler_test.executor(), shots=100).result()
print("正しい回路は問題なく実行できます:")
for value, count in result_good.results:
    print(f"  測定結果: {value}, 回数: {count}")

# %% [markdown]
# ### まとめ：線形型のルール
#
# | 書き方 | 正しい？ | 理由 |
# |--------|---------|------|
# | `q = qm.h(q)` | OK | 戻り値を再代入 |
# | `qm.h(q)` | NG | 戻り値を無視 |
# | `q1 = qm.h(q); q2 = qm.x(q)` | NG | 同じ q を2回使用 |
# | `q = qm.h(q); q = qm.x(q)` | OK | 順番に更新 |

# %% [markdown]
# ## 5. 量子回路の実行
#
# 作成した量子回路を実行してみましょう。
# Qamomileでは、**トランスパイラ**を使って回路をバックエンド（今回はQiskit）に変換し、実行します。

# %%
# トランスパイラを作成
transpiler = QiskitTranspiler()

# 量子回路をコンパイル
executable = transpiler.transpile(x_gate_circuit)

# シミュレータで実行（1000回測定）
job = executable.sample(transpiler.executor(), shots=1000)
result = job.result()

# 結果を表示
print("=== Xゲート回路の実行結果 ===")
for value, count in result.results:
    print(f"  測定結果: {value}, 回数: {count}")

# %% [markdown]
# ### 結果の解説
#
# Xゲートを `|0⟩` に適用すると `|1⟩` になります。
# したがって、測定結果は常に **1** になるはずです。
#
# 上の結果で、1000回の測定全てで `1` が得られていることを確認してください。

# %% [markdown]
# ## 6. 量子回路の可視化
#
# 作成した回路がどのような構造になっているか、図で確認できます。

# %%
# Qiskit形式の回路を取得
qiskit_circuit = transpiler.to_circuit(x_gate_circuit)

# 回路を表示
print("=== 量子回路の構造 ===")
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ### 回路図の読み方
#
# ```
#      ┌───┐┌─┐
#   q: ┤ X ├┤M├
#      └───┘└╥┘
# c: 1/══════╩═
#            0
# ```
#
# - `q`: 量子ビットのライン
# - `X`: Xゲート
# - `M`: 測定
# - `c`: 古典ビット（測定結果が格納される）

# %% [markdown]
# ## 7. もう一つの例：何もしない回路
#
# 比較のため、何もゲートを適用しない回路も作ってみましょう。

# %%
@qm.qkernel
def identity_circuit() -> qm.Bit:
    """何もしない回路（初期状態をそのまま測定）"""
    q = qm.qubit(name="q")
    # 何もゲートを適用しない
    return qm.measure(q)


# %%
# 実行
executable_id = transpiler.transpile(identity_circuit)
job_id = executable_id.sample(transpiler.executor(), shots=1000)
result_id = job_id.result()

print("=== 何もしない回路の実行結果 ===")
for value, count in result_id.results:
    print(f"  測定結果: {value}, 回数: {count}")

# %% [markdown]
# 量子ビットの初期状態は `|0⟩` なので、何もしなければ測定結果は常に **0** になります。

# %% [markdown]
# ## 8. まとめ
#
# このチュートリアルでは、以下のことを学びました：
#
# 1. **Qamomileの基本**: `@qm.qkernel` デコレータで量子回路を定義
# 2. **量子ビット**: `qm.qubit()` で作成、初期状態は `|0⟩`
# 3. **量子ゲート**: `qm.x()` などのゲートで状態を操作
# 4. **線形型システム**: ゲート適用後は **必ず再代入** (`q = qm.x(q)`)
# 5. **測定**: `qm.measure()` で量子状態を古典ビットに変換
# 6. **実行**: `QiskitTranspiler` でコンパイルし、`sample()` で実行
#
# ### 重要なポイント
#
# ```python
# @qm.qkernel
# def my_circuit() -> qm.Bit:
#     q = qm.qubit(name="q")  # 量子ビット作成
#     q = qm.x(q)              # ゲート適用（再代入必須！）
#     return qm.measure(q)     # 測定
# ```
#
# 次のチュートリアル（`02_single_qubit.py`）では、重ね合わせ状態を作る**アダマールゲート**や、
# 回転角度を指定できる**回転ゲート**について学びます。
