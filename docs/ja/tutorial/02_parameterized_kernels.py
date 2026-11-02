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
# # パラメータ付き量子カーネル
#
# [「はじめての量子カーネル」](01_your_first_quantum_kernel.ipynb)では、量子ビット数が固定された量子カーネルを定義しました。Qamomileでは、量子ビット数や回路のレイヤー数などをシンボリックなパラメータとして表し、それらの値に応じて構造が決まるカーネルも記述できます。例えば、`width`個の量子ビットにアダマール（H）ゲートを適用するカーネルや、一連のゲート操作を`layers`回繰り返すカーネルを記述できます。
#
# `Const`と`Param`は、カーネルの入力に対して、いつ値を指定する必要があるかを示します。`Const`の値は、カーネルを具体化する特殊化（specialization）までに指定しておく必要があります。量子カーネルを直接`qm.compile(...)`に渡す場合は、その呼び出しの中で特殊化も行われるため、`Const`の値を`qm.compile(...)`の引数として渡します。`Param`の値は実行前に指定します。
#
# :::{note}
# 特殊化をコンパイルとは別に行う場合は、`Const`の値を`qm.specialize(...)`の引数として渡します。返された`Program`を`qm.compile(...)`に渡してコンパイルできます。
# :::
#
# この章の例では、量子ビット数を`Const`、回転角を`Param`として宣言します。
#
# `Shot`は、1回の実行（ショット）の中で、使う時点までに値がわかっている必要があることを示します。
#
# 回転角を`Param`として宣言しておくことで、量子ビット数を固定したコンパイル結果を、回転角を変えながら再利用できます。
#
# この章では、次の内容を扱います。
#
# - 量子カーネルにおける`Const`、`Param`、`Shot`の役割
# - `qm.qubits()`と`range()`を使った、パラメータ付き回路の定義
# - 量子状態に含まれるすべての量子ビットへ単一量子ビットゲートを適用する**ブロードキャスト**
# - 一度コンパイルし、パラメータの値を変えながら同じコンパイル結果を再利用する方法

# %% [markdown]
# ## セットアップ

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
from __future__ import annotations

import os

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 1 if docs_test_mode else 128

# %% [markdown]
# ## `Const`、`Param`、`Shot`の役割
#
# `Const`、`Param`、`Shot`は、古典データの値がどの時点までにわかっている必要があるかを示します。値をどのように得たかを区別するものではありません。以下に、それぞれの要件と、この章での使用例をまとめます。
#
# | 型アノテーション | 値の扱いと使用例 |
# |---|---|
# | `qm.Const[T]` | 特殊化までに値を指定する入力。例：量子ビット数、繰り返し回数 |
# | `qm.Param[T]` | 実行前に値を指定する入力。例：回転角 |
# | `qm.Shot[T]` | 各ショットで使う時点までにわかっている必要がある値。例：測定結果 |
#
# `T`は、`qm.Int`、`qm.Real`、`qm.Bits[4]`などの古典データの型を表す仮の記号です。例えば、`qm.Const[qm.Int]`は特殊化までに値を指定する整数の入力を表し、`qm.Bits[4]`は4ビットの値を表します。
#
# `Shot[T]`の値は、実行開始前にわかっている必要はありませんが、そのショット内で使う時点までにはわかっている必要があります。代表的な例は、`qm.measure(...)`が返す測定結果や、それを使って計算した値です。実行前からわかっている値を`Shot[T]`として扱うこともできます。
#
# 以下の例では、`width`が量子ビット数とループの繰り返し回数を決めます。実行例では、その値を`qm.compile(...)`の引数として渡します。回転角`theta`は`Param`のまま残るため、実行ごとに異なる値を指定できます。
#
# `Const`は回路の構造を決める値に限らず、回転角などにも使えます。回転角を`qm.Const[qm.Real]`と宣言した場合、その値を特殊化までに指定する必要があります。一方、`qm.Param[qm.Real]`と宣言すると、実行ごとに値を指定できます。

# %% [markdown]
# ## `qm.qubits`と`range`
#
# `qm.qubits(width)`を使うと、`width`個の量子ビットを含む量子状態を作成できます。個々の量子ビットにゲートを適用するには、`range(width)`で量子ビットのインデックスを順に取り出します。`target`引数には、ゲートを適用する量子ビットのインデックスを指定します。
#
# `range(start, stop, step)`も使えます。例えば、`range(0, width, 2)`を使うと、インデックス0から1つおきに量子ビットを選択できます。
#
# 次のカーネルは、各量子ビットにHゲートとRYゲートを順に適用し、その後で量子状態を測定します。


# %%
@qm.qkernel
def rotation_layer(
    width: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    for target in range(width):
        state = qm.h(state, target=target)
        state = qm.ry(state, theta, target=target)
    return qm.measure(state)


# %% [markdown]
# :::{note}
# ` # noqa: F821 `は、戻り値の型アノテーションにある`width`について、リンターが出す警告を抑制するコメントです。実行やQamomile自身の型チェックには影響しません。
# :::
#
# `width`に具体的な値を指定する前でも、カーネルの論理リソースを解析できます。このカーネルでは各量子ビットに2つのゲートを適用するため、論理ゲート数は`2 * width`です。
#
# この式を`width=4`で評価すると、論理ゲート数は8になります。ここではリソースを表す式を評価しており、カーネル自体の特殊化は行いません。

# %%
symbolic_study = qm.resources.logical(rotation_layer)
symbolic_gates = symbolic_study.metric(qm.resources.LOGICAL_GATES)

print("symbolic gates:", symbolic_gates.expression)
assert symbolic_gates.evaluate({"width": 4}).numeric_enclosure() == (8, 8)
assert symbolic_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).evaluate(
    {"width": 4}
).numeric_enclosure() == (4, 4)

# %% [markdown]
# `numeric_enclosure()`は、リソース量の下限と上限を返します。論理ゲート数の下限と上限は両方とも8なので、ゲート数は正確に8とわかります。

# %% [markdown]
# ## インデックスを指定してゲートを適用する
#
# `state = qm.h(state, target=target)`では、`qm.h`が量子状態を表す入力値を受け取り、ゲート適用後の状態を表す値を返します。入力として渡した元の値は、その後の操作では再利用できません。この例では、返された値を再び`state`という名前で受け取り、次の操作に渡しています。
#
# Qamomileは、`state`に含まれる量子ビットを直接ループで取り出す書き方には対応していません。次の例では、この書き方がエラーになることを確認します。


# %%
@qm.qkernel
def bad_iteration(
    width: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    for qubit in state:  # 量子ビットを直接ループで取り出す書き方は未対応です。
        qubit = qm.h(qubit)
    return state


# %%
try:
    qm.specialize(bad_iteration, width=4)
except qm.FrontendError as error:
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {error}")
    assert "for expects range" in str(error)

# %% [markdown]
# 代わりに、`rotation_layer`と同じように、`range(width)`で量子ビットのインデックスを順に取り出します。各操作で更新された`state`を、次のゲートに渡してください。

# %% [markdown]
# ## 単一量子ビットゲートを量子状態全体にブロードキャストする
#
# HやRYなどの単一量子ビットゲートでは、`target=`を省略すると、量子状態に含まれるすべての量子ビットにそのゲートを適用します。これを**ゲートのブロードキャスト**と呼びます。
#
# 例えば、`qm.h(state)`はすべての量子ビットにHゲートを適用し、`qm.ry(state, theta)`は同じ回転角`theta`で、すべての量子ビットにRYゲートを適用します。1つの量子ビットを選ぶ場合は、`target`引数にインデックスを指定します。一部の量子ビットだけに適用したり、量子ビットごとに異なる回転角を使ったりする場合は、ループを使います。


# %%
@qm.qkernel
def rotation_layer_broadcast(
    width: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    state = qm.h(state)
    state = qm.ry(state, theta)
    return qm.measure(state)


# %% [markdown]
# `rotation_layer`と`rotation_layer_broadcast`は、どちらも各量子ビットにHゲート、RYゲートの順で操作を適用します。`width=4`の場合、どちらも4個の量子ビットを使い、論理ゲート数は8です。以下では、ブロードキャスト版のリソース量がこの値になることを確認します。

# %%
all_program = qm.specialize(rotation_layer_broadcast, width=4)
all_study = qm.resources.logical(all_program)

assert all_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (4, 4)
assert all_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
    8,
    8,
)

# %% [markdown]
# ## 一度コンパイルして、パラメータの値を変えながら実行する
#
# パラメータ付き量子カーネルでは、回路の構造を固定してコンパイルし、パラメータの値を変えながら繰り返し実行する方法が便利です。
#
# 1. **一度コンパイルする**：`width`の値を固定してコンパイルし、`theta`の値は実行時に指定できるようにします。
# 2. **値を変えて実行する**：コンパイル結果を再利用し、実行ごとに異なる`theta`の値を渡します。
#
# この例では、Qamomileに組み込まれたReferenceシミュレーターを使って、ローカル環境で実行します。

# %%
# width=4を固定してコンパイルし、thetaは実行時パラメータとして残します。
endpoint = qm.Reference()
artifact = qm.compile(rotation_layer, to=endpoint.target, width=4)

parameter_names = tuple(slot.name for slot in artifact.parameter_abi.slots)
print("runtime parameters:", parameter_names)

assert parameter_names == ("theta",)

# %% [markdown]
# コンパイル結果`artifact`には、`theta`だけが実行時パラメータとして残っています。
#
# `width=4`を指定すると、`qm.compile(...)`の内部で行われる特殊化で、量子ビット数とループの繰り返し回数が固定されます。`qm.specialize(...)`を別途呼ぶ必要はありません。
#
# 次に、同じ`artifact`を使って、3つの異なる回転角で実行します。

# %%
# 同じartifactを使い、thetaの値を変えて実行します。
for theta in (0.1, 0.5, 1.0):
    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"theta": theta},
        shots=sample_shots,
        seed=7,
    )
    result = record.result
    print(f"theta={theta:.1f} -> {result.counts}")
    assert record.artifact is artifact
    assert result.shots == sample_shots
    assert sum(result.counts.values()) == sample_shots
    # 4個の量子ビットを測定すると、古典ビットを要素とする長さ4のタプルが得られます。
    assert all(len(outcome) == 4 for outcome in result.counts)

# %% [markdown]
# 3回の実行すべてで、同じコンパイル結果を再利用しています。変えているのは、`parameters={"theta": theta}`で渡す値だけです。
#
# それぞれの指定の役割を整理すると、次のようになります。
#
# - **`qm.compile(...)`の`width=4`** は、量子ビット数とループの繰り返し回数を固定します。
# - **`theta: qm.Param[qm.Real]`** は、`theta`を実行時に値を指定する実数の入力として宣言しています。この例では、コンパイル時には回転角を固定せず、`qm.run(...)`の`parameters={"theta": ...}`で実行ごとの値を指定します。
# - **`qm.run(...)`の`parameters={"theta": ...}`** は、その実行で使う回転角を指定します。
#
# この例では、`theta`を変更しても再コンパイルは不要ですが、`width`を変更する場合は、カーネルを再び特殊化してコンパイルする必要があります。

# %% [markdown]
# ## まとめ
#
# - `Const`と`Param`は、カーネルの入力値をいつ指定するかを示します。`Const`の値は特殊化までに、`Param`の値は実行前に指定します。この章では、`qm.measure(...)`が返す測定結果を`Shot`の値として扱いました。
# - `qm.qubits(width)`で、量子ビット数をパラメータで指定した量子状態を作成できます。`range(width)`を使って、量子ビットのインデックスを順に取り出します。
# - 各ゲートは更新後の状態を表す値を返します。以降の操作では、入力として渡した古い値ではなく、返された値を使います。
# - `target=`を省略すると、量子状態に含まれるすべての量子ビットに単一量子ビットゲートをブロードキャストできます。個々の量子ビットを選ぶ場合は、インデックスやループを使います。
# - 回路の構造を固定してコンパイルし、実行時パラメータの値を変えながら、同じコンパイル結果を再利用します。
#
# **次へ**：次の章では、[`qm.control`による制御演算](03_controlled_gates.ipynb)を扱います。
#
# 量子ビットを選択したり、量子ビット列を分割したりする方法は、使い方ガイドの[量子ビット列の操作](../usage/qubit_arrays.ipynb)を参照してください。
