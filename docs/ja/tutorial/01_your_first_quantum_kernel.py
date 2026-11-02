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
# # はじめての量子カーネル
#
# この章では、Qamomileを初めて使う方に向けて、量子カーネルを定義し、実行するまでの基本的なワークフローを紹介します。量子計算の基礎や量子アルゴリズムの詳細には踏み込みません。
#
# ## Qamomileとは？
#
# Qamomileは、Pythonに埋め込まれた、ソースコンパイル型の関数型量子プログラミング言語です。
#
# Python関数に`@qm.qkernel`デコレータを付けて、量子カーネルを定義します。量子カーネルの引数と戻り値には、型アノテーションが必要です。Qamomileは、その型情報に基づいて実行前に型の整合性を検査します。
#
# Qamomileはカーネルのソースコードを読み取り、検証して中間表現に変換します。この中間表現は解析や保存ができ、Qiskit、QURI Parts、CUDA-Qなどの対応する量子SDK向けにコンパイルできます。
#
# また、Qamomileは、値がまだ確定していないパラメータを含む式をシンボリックに扱えます。
#
# この章では、量子カーネルを定義し、実行して結果を取り出すために必要なワークフローを紹介します。
#
# - **定義**：型アノテーションと`@qm.qkernel`デコレータを付けたPython関数を書きます。
# - **実行**：`record = qm.run(...)`で量子カーネルを実行し、実行結果を含む`RunRecord`を受け取ります。
# - **結果の読み取り**：`record.result`から実行結果を取得します。
#
# Qamomileでは、同じ量子カーネルを使って、計算の実行だけでなく、必要な量子ビット数や論理ゲート数などを実行前に見積もることもできます。コンパイルを個別に行う機能や、コンパイル前にプログラムの内容を確認する機能もあります。まず基本のワークフローを試してから、これらの機能を紹介します。

# %% [markdown]
# ## インストール
#
# このチュートリアルでは、Qiskit Aerを使って量子計算をシミュレーションします。量子コンピューターの実機は必要ありません。
#
# 次の行の先頭の`#`を外して実行すると、Qamomile、Qiskit、Aerをまとめてインストールできます。

# %%
# # !pip install "qamomile[qiskit]"

# %%
import math
import os

import qamomile as qm
from qamomile.qiskit import Aer, load_circuit

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 1 if docs_test_mode else 256

# %% [markdown]
# ## 基本のワークフロー
#
# ### 最初の量子カーネル：偏りのあるコイン
#
# Qamomileでは、Python関数に`@qm.qkernel`デコレータを付けて**量子カーネル**（QKernel）を定義します。量子カーネルでは、型付きの値とゲート操作を使って量子計算を記述します。
#
# まずは、1つの量子ビットを角度`theta`で回転させて測定する、簡単な例から始めましょう。`theta`を変えると、測定で`0`や`1`が得られる確率を変えられます。コイントスで表と裏の出やすさを調整するようなイメージです。


# %%
@qm.qkernel
def biased_coin(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    # 1つの量子ビットからなる状態を作成します。
    state = qm.qubits(1)

    # RY回転を適用して、測定結果に偏りを持たせます。
    state = qm.ry(state, theta)

    # 測定し、1つの古典ビットを返します。
    return qm.measure(state)


# %% [markdown]
# :::{note}
# Pythonの型チェッカーによっては、`qm.Shot[qm.Bits[1]]`が有効な型アノテーションとして認識されないことがあります。`# type: ignore[valid-type]`は、この行に対する指摘を抑制するためのコメントです。カーネルの動作を変えたり、Qamomile自身の型チェックを無効にしたりするものではありません。
# :::
#
# **ここでのポイント：**
#
# - **型アノテーションは必須です**。 `theta: qm.Param[qm.Real]`では、`qm.Real`が値の型（浮動小数点数）を、`qm.Param[...]`がその値を実行時に指定することを表します。この例では、実行ごとに回転角`theta`を指定できます。戻り値の型`qm.Shot[qm.Bits[1]]`は、このカーネルが1回の実行（ショット）で1ビットの値を返すことを表します。この例では、`qm.measure(state)`で得た測定結果を返します。
# - `qm.qubits(1)`は、1つの量子ビットからなる状態を作成します。
# - `state = qm.ry(state, theta)`は、RY回転を適用し、返された更新後の状態を同じ変数名`state`で受け取ります。以降の操作では、この更新後の値を使います。ゲートに渡した古い値を再び使うことはできません。
# - `qm.measure(state)`は、量子ビットを測定し、古典ビットを返します。

# %% [markdown]
# ### 量子カーネルを実行する
#
# この例では、`parameters`で回転角`theta`の値を、`shots`で繰り返し実行する回数を指定します。
#
# `biased_coin`を`qm.run(...)`に直接渡して実行できます。必要な検証とコンパイルは内部で行われます。
#
# ここでは、`Aer()`でQiskit Aerシミュレーターの実行先（エンドポイント）を作成します。エンドポイントは、計算をどこで実行するかを指定するものです。

# %% [markdown]
# :::{note}
# `qm.run(...)`に渡す量子カーネルは、入れ子になった値も含め、入力も出力も古典データである必要があります。呼び出し元が受け取る計算結果は、測定結果などの古典データに限られます。カーネル内で別の量子カーネルを呼び出す場合は、量子状態を引数や戻り値として受け渡せます。
#
# 1回の実行（ショット）の中で行う古典計算や、測定結果に応じた条件分岐は、量子カーネルに記述できます。複数のショットで得られた結果の集計は、量子カーネルの外で行います。[チュートリアル05](05_execution_models.ipynb)では、`statistics`と`Estimator`を使った統計処理と、観測量の期待値の推定を紹介します。
# :::

# %%
# 実行先を選びます。
endpoint = Aer()

# 回路をsample_shots回実行します。
record = qm.run(
    biased_coin,
    on=endpoint,
    parameters={"theta": math.pi / 4},
    shots=sample_shots,
    seed=7,
)

# 測定結果を保持するSampleResultを取得します。
result = record.result

print("sample counts:", result.counts)
assert result.shots == sample_shots
assert sum(result.counts.values()) == sample_shots

# %% [markdown]
# 各部分の役割を確認しましょう。
#
# - **`parameters={"theta": math.pi / 4}`** は、今回の実行で使う回転角をπ/4ラジアンに指定します。
#
# - **`shots=sample_shots`** は、計算を繰り返す回数を指定します。`sample_shots`が256の場合、この例では1ビットの測定結果が256回分得られます。
#
# - **`seed=7`** は、シミュレーターの乱数シードを設定します。同じ設定で繰り返し実行したときに、結果を再現できるようにします。
#
# - **`record.result`** から実行結果を取得できます。`qm.run(...)`は実行の完了を待ってから`RunRecord`を返します。この例では、その`result`に、測定結果をQamomileのデータ形式で保持する`SampleResult`が入っています。
#
# この例では、実行ごとに`theta`の値を指定できます。パラメータの種類と値を指定するタイミングについては、[チュートリアル02](02_parameterized_kernels.ipynb)で詳しく説明します。

# %% [markdown]
# ### `SampleResult`から結果を読み取る
#
# 前の節では、`record.result`を変数`result`に代入しました。この例の`result`は`SampleResult`のインスタンスで、測定結果をQamomileのデータ形式で保持しています。そのプロパティやメソッドを使って、各結果の出現回数や総ショット数を確認したり、測定結果の確率を推定したりできます。
#
# まず、`result.counts`を見てみましょう。これは、測定結果と出現回数を対応付けた読み取り専用のマッピングです。
#
# - **キー**は、ビットのタプルで表した測定結果です。この例では、1ビットの結果を`(0,)`または`(1,)`で表します。
# - **値**は、その測定結果が得られたショット数です。
#
# 例えば、`{(0,): 214, (1,): 42}`は、256ショットのうち`0`が214回、`1`が42回得られたことを意味します。

# %%
for value, count in result.counts.items():
    print(f"  outcome={value}, count={count}")

# %% [markdown]
# 次のプロパティやメソッドも使えます。
#
# - **`result.most_likely`** は、最も多く得られた測定結果を返します。
# - **`result.probability((1,))`** は、`1`の出現回数を総ショット数で割り、測定で`1`が得られる確率を推定します。
# - **`result.values`** は、カーネルで宣言した戻り値の型に従って復元した測定結果と、その出現回数を対応付けた読み取り専用のマッピングを返します。このカーネルでは、`result.counts`と同じ内容になります。

# %%
# 最も多く得られた結果。同数の場合も一定の規則で選びます。
print("most likely:", result.most_likely)

# 測定で1が得られる確率の推定値。
print("P(1):", result.probability((1,)))

# カーネルの戻り値の型に従って復元した測定結果と出現回数。
print("typed values:", result.values)

# %% [markdown]
# ## 追加機能を使う
#
# ここまでで、定義から実行、結果の読み取りまでを試せました。ここからは、同じカーネルを使って、コンパイルを個別に行う方法、コンパイル前の確認、生成された回路の確認を紹介します。
#
# ### コンパイルを個別に行う
#
# `qm.compile(...)`は、量子カーネルを受け取り、繰り返し実行できるコンパイル結果を返します。以下の例では、`biased_coin`を渡し、返された`CompiledProgram`のインスタンスを`artifact`として受け取ります。この`artifact`は、`qm.run(...)`に渡して実行できます。
#
# `endpoint.target`は、その実行先に必要な形式と設定を表します。ここでは、Qiskit Aerで実行するためのコンパイルに使用します。

# %%
artifact = qm.compile(biased_coin, to=endpoint.target)
compiled_record = qm.run(
    artifact,
    on=endpoint,
    parameters={"theta": math.pi / 4},
    shots=sample_shots,
    seed=7,
)
assert compiled_record.result.counts == result.counts

# %% [markdown]
# コンパイル結果`artifact`には、`theta`がパラメータとして残ります。`qm.run(...)`の`parameters`で回転角を変えながら、同じ`artifact`を再利用できます。
#
# 別のエンドポイントで必要な形式やコンパイル設定が異なる場合は、その設定でカーネルをコンパイルしてから実行します。

# %% [markdown]
# ### コンパイル前の確認
#
# 量子カーネルをコンパイルする前に、カーネルの検証、プログラムの内容の確認、必要なリソースの見積もりを行えます。バックエンドを選んだり、量子計算を実行したりする必要はありません。
#
# #### 量子カーネルを検証する
#
# `qm.specialize(...)`は、指定された`Const`入力の値をカーネルに反映し、検証済みの`Program`を作成します。このとき、型や量子状態の使い方を検査します。コンパイル前にその内容を確認したい場合などに利用できます。

# %%
program = qm.specialize(biased_coin)

# %% [markdown]
# `biased_coin`には`Const`入力がないため、ここでは値を追加で指定する必要はありません。量子カーネルを`qm.compile(...)`に直接渡す場合は、特殊化と検証が内部で行われるため、`qm.specialize(...)`を別途呼ぶ必要はありません。先に作成した`Program`をコンパイルする場合は、`qm.compile(program, to=endpoint.target)`と書けます。
#
# #### プログラムの内容を確認する
#
# `program.inspect()`では、プログラムに含まれる演算や型をテキストで確認できます。`qm.resources.logical(program)`では、必要な量子ビット数や論理ゲート数などを見積もれます。以下の例では、リソース推定に注目します。

# %% [markdown]
# #### 必要なリソースを見積もる
#
# `qm.resources.logical(program)`で、先ほど検証した`biased_coin`の論理リソースを推定します。ここでは、同時に使用する量子ビット数の最大値と、論理ゲート数を確認します。

# %%
study = qm.resources.logical(program)
peak_qubits = study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
logical_gates = study.metric(qm.resources.LOGICAL_GATES)

print("peak logical qubits:", peak_qubits)
assert peak_qubits.numeric_enclosure() == (1, 1)
print("logical gates:", logical_gates)
assert logical_gates.numeric_enclosure() == (1, 1)

# %% [markdown]
# このカーネルは、1つの量子ビットと1つの論理ゲート（RY回転）を使用しています。ここで見積もっているのはカーネルの論理リソースであり、特定の量子コンピューターで実行するための物理リソースは含みません。詳しくは[リソース推定](04_resource_estimation.ipynb)の章で扱います。

# %% [markdown]
# ### コンパイル済みの回路を確認する
#
# コンパイルで生成された回路も確認できます。冒頭で`qamomile.qiskit`からインポートした`load_circuit(artifact)`は、コンパイル結果`artifact`に保持されているQiskit回路のコピーを返します。
#
# 以下の`qiskit_circuit`は、Qiskitの`QuantumCircuit`のインスタンスです。`print(...)`でゲートや測定を含む回路図を表示できるため、回路の構成の確認やデバッグに役立ちます。

# %%
qiskit_circuit = load_circuit(artifact)
print(qiskit_circuit)
assert qiskit_circuit.num_qubits == 1
assert qiskit_circuit.num_clbits == 1

# %% [markdown]
# 表示された回路の`p0`は、カーネルのパラメータ`theta`に対応します。先ほどの実行ではπ/4を指定しましたが、`artifact`内の回路には、実行後も具体的な値に置き換えられていないパラメータが残っています。`qm.run(...)`は、実行のたびに指定された値を使います。

# %% [markdown]
# ### 複数量子ビットの例
#
# 次は、2つの量子ビットを使ってみましょう。1つ目の量子ビットにアダマール（Hadamard）ゲートを適用し、続いてCNOTゲートを適用してから、両方の量子ビットを測定します。


# %%
@qm.qkernel
def two_qubit_demo() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %% [markdown]
# `qm.qubits(2)`は、2つの量子ビットからなる状態を作成します。量子ビットのインデックスは0から始まります。
#
# - `target=0`は、アダマールゲートの対象として1つ目の量子ビットを指定します。
# - `control=0, target=1`は、CNOTゲートの制御量子ビットとして1つ目を、標的量子ビットとして2つ目を指定します。
#
# `qm.measure(state)`は、両方の量子ビットを測定します。戻り値の型`qm.Shot[qm.Bits[2]]`は、各ショットで2つの古典ビットを返すことを示しています。
#
# このカーネルでは、`two_qubit_demo`を`qm.run(...)`に直接渡せます。先ほどと同じAerのエンドポイントを使い、検証・コンパイル・実行をまとめて行います。

# %%
demo_record = qm.run(
    two_qubit_demo,
    on=endpoint,
    shots=sample_shots,
    seed=11,
)
demo_result = demo_record.result

for outcome, count in demo_result.counts.items():
    print(f"  outcome={outcome}, count={count}")
assert demo_result.shots == sample_shots
assert sum(demo_result.counts.values()) == sample_shots
# ベル状態|Phi+>では、(0, 0)と(1, 1)だけが得られます。
assert set(demo_result.counts) <= {(0, 0), (1, 1)}

# %% [markdown]
# この回路で作られるベル状態を測定すると、理想的には`(0, 0)`または`(1, 1)`だけが得られ、それぞれの確率は1/2です。ここで使うノイズのないシミュレーターは、この確率分布に従って測定結果を生成します。
#
# 実行回数が有限なので、両者の出現回数がちょうど同じになるとは限りません。
#
# この例では、量子ビット数を2に固定しています。チュートリアル02では、`Const[Int]`の入力で量子ビット数を指定する方法を扱います。
#
# アダマールゲートもCNOTゲートも更新後の状態を返し、それを同じ変数名`state`で受け取っている点に注目してください。次の節では、以降の操作で更新後の状態を使う必要がある理由を説明します。

# %% [markdown]
# ### 線形型システム
#
# Qamomileは、**線形型システム**（linear type system）を使って量子状態を管理します。ゲートは、受け取った状態を表す値を消費し、更新後の状態を表す値を返します。一度消費された古い値は、再び使うことができません。
#
# これまでの例では、更新後の状態を同じ変数名で受け取っています。
#
# - 1量子ビットゲート：`state = qm.h(state)`
# - 2量子ビットゲート：`state = qm.cx(state, control=0, target=1)`
#
# 返された状態を別の変数名で受け取ることもできます。重要なのは、以降の操作で更新後の値を使うことです。
#
# #### なぜ線形なのか？
#
# Qamomileの線形型システムは、量子状態を表す値の受け渡しを追跡し、消費済みの値の再利用を実行前に検出します。これにより、更新前と更新後の状態を別々に使えるかのような記述を防ぎます。また、使われなくなった量子状態が適切に処理されることも確認します。これまでの例では、最後に`qm.measure(state)`が量子状態を消費し、古典データを返しています。
#
# 一時的に使う量子ビットには注意が必要です。他の量子ビットともつれたままになり、干渉や最終的な測定結果の確率に影響することがあります。**アンコンピュート**（uncomputation）は、一時的な計算を逆向きに実行して、不要なもつれを解消する操作です。`qm.adjoint(...)`を使うと、ユニタリ演算を行うヘルパーカーネルの逆演算を明示的に適用できます。
#
# Qamomileは、**自動アンコンピュート**にも対応しています。変数に代入された状態が以降で使われない場合、コンパイラは、その状態をアンコンピュートして解放する処理の生成を試みます。必要なアンコンピュートと解放を検証できない場合は、エラーになります。
#
# 意図的に状態を破棄する場合は、`qm.discard(state)`を使います。これは、アンコンピュートを行わずに状態を明示的に消費します。量子ビットを破棄しても、他の量子ビットともつれさせた操作が取り消されるわけではありません。
#
# 例えば、次のカーネルでは、アダマールゲートが返した状態を無視しています。


# %%
@qm.qkernel
def bad_rebind() -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    state = qm.qubits(1)
    qm.h(state)  # 誤り：返された状態を使っていません。
    state = qm.x(state)  # ここでは、まだ古い状態を参照しています。
    return qm.measure(state)


# %% [markdown]
# このカーネルは、実行せずに`qm.specialize(...)`で検証できます。

# %%
try:
    qm.specialize(bad_rebind)
except qm.FrontendError as error:
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {error}")
    assert "unused expression has non-Unit type" in str(error)

# %% [markdown]
# `unused expression`というエラーは、`qm.h(state)`の戻り値が無視されていることを指しています。この行を`state = qm.h(state)`に変えると、次の`qm.x(state)`で更新後の状態を使えます。

# %% [markdown]
# ## サポートされている量子SDK
#
# Qamomileは、次の量子SDKや中間表現に対応しています。
#
# | SDK・中間表現 | 対応状況 | 備考 |
# |---|---|---|
# | **Qiskit** | 対応 | Qiskit回路の生成、Aerによるシミュレーション、対応するQiskitバックエンドでの実行 |
# | **QURI Parts** | 対応 | QURI Parts回路の生成、Qulacsによるシミュレーション、QURI Partsのサンプラーを使った実行 |
# | **CUDA-Q** | 対応 | CUDA-Qカーネルの生成と、利用可能なCUDA-Qシミュレーターでの実行 |
# | **Cirq** | 対応 | Cirq回路の生成、ローカルシミュレーション、対応するサンプラーやGoogle Quantum Engineを使った実行 |
# | **Amazon Braket** | 対応 | ローカルシミュレーションと、対応するAmazon Braketデバイスでの実行 |
# | **Quration（PyQret）** | 対応 | ローカルシミュレーションと、誤り耐性量子計算（FTQC）のリソース推定 |
# | **HUGR** | 対応 | HUGR形式へのコンパイル、Seleneによるローカルシミュレーション、Quantinuum Nexus経由のHeliosでの実行 |
# | **qBraid** | Qiskit経由で対応 | コンパイル済みのQiskit回路を、対応するqBraidデバイスで実行 |
# | **QIR** | 出力のみ | QIR 1.0 Base Profile形式で出力。実行には外部のQIR対応システムを使用 |
#
# :::{note}
# 利用できる機能は、選択したバックエンドによって異なります。例えば、測定結果に応じて次に適用するゲートを変える処理には、測定結果に基づく制御フローへの対応が必要です。`Const`入力で決まる条件は、コンパイル時に解決されます。
# :::
#
# ### CUDA-Qの対応環境
#
# CUDA-Qは、次の環境に対応しています。
#
# | 環境 | 対応状況 | 備考 |
# |---|---|---|
# | **Linux** | 対応 | CPUによるシミュレーション。GPUによる高速化には、対応するNVIDIA GPUと互換性のあるCUDA環境が必要 |
# | **macOS ARM64（Apple silicon）** | 対応 | ローカルシミュレーションはCPUのみ |
# | **Windows（WSL2経由）** | 対応 | WSL2のLinux環境内でインストール・実行 |
# | **ネイティブWindows** | 非対応 | WSL2を使用 |
# | **macOS x86_64（Intel）** | 非対応 | macOSではApple siliconが必要 |
#
# 対応環境と要件は、[CUDA-Qのインストールドキュメント](https://nvidia.github.io/cuda-quantum/0.14.0/using/install/local_installation.html)に基づいています。
#
# ## まとめ
#
# 基本のワークフローは、`@qm.qkernel`と型アノテーションで量子カーネルを定義し、`qm.run(...)`で実行して、`record.result`から結果を取り出すことです。この章のサンプリングの例では、`SampleResult`から出現回数や確率の推定値を確認できます。
#
# さらに、次の機能や考え方を紹介しました。
#
# - カーネルを個別にコンパイルし、パラメータの値を変えながら再利用する。
# - 必要に応じて、コンパイル前に特殊化、内容の確認、論理リソースの推定を行う。
# - `load_circuit(...)`で、コンパイル済みのQiskit回路を確認する。
# - 複数の量子ビットを扱い、更新後の量子状態を正しく使う。
# - アンコンピュートと明示的な破棄の違いを理解する。
#
# ## 次の章
#
# 以降の章では、この章で紹介したワークフローをもとに、次の内容を学びます。
#
# 2. [パラメータ付き量子カーネル](02_parameterized_kernels.ipynb) — 回路の構造を決めるパラメータと実行時パラメータを区別し、実行時の値を変えながらコンパイル済みのプログラムを再利用します。
# 3. [qm.controlによるコヒーレント制御](03_controlled_gates.ipynb) — 1つ以上の制御量子ビットを使って、ユニタリな量子カーネルを制御付きで適用する方法を学びます。
# 4. [リソース推定](04_resource_estimation.ipynb) — 量子ビット数、ゲート数、回路の深さなどの論理リソースを見積もり、シンボリックな推定結果の読み方と、その限界を学びます。
# 5. [実行・統計・オブザーバブルの推定](05_execution_models.ipynb) — 単一ショットの実行と繰り返しサンプリング、測定結果の集計、観測量の期待値の推定を学びます。
# 6. [古典制御フローのパターン](06_classical_flow_patterns.ipynb) — ループや条件分岐、特定の量子ビット対の相互作用を表すデータを扱います。測定結果に応じて処理を分岐する方法も紹介します。
# 7. [再利用パターン：量子カーネルの合成と複合ゲート](07_reuse_patterns.ipynb) — ヘルパーカーネルや複合ゲートを組み合わせる方法、カーネルを引数として渡す方法、実装本体のない関数を宣言してリソースを見積もる方法を学びます。
#
# 量子ビット列の一部を操作する方法は、使い方ガイドの[量子ビット列の操作](../usage/qubit_arrays.ipynb)を参照してください。
