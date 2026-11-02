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
# tags: [usage, resource-estimation]
# ---
#
# # リソース推定の詳細ガイド
#
# このガイドでは、個々のリソース指標の意味と数え方、推定結果の保存方法、推定の根拠や仮定の読み方を説明します。基本的な使い方、パラメータを含む推定式、制御演算のモデルについては、[リソース推定](../tutorial/04_resource_estimation.ipynb)を参照してください。
#
# 以下の例には必要な定義をすべて含めているため、先にチュートリアルを実行しなくても、このページの先頭から順に実行できます。

# %% [markdown]
# ## セットアップ
#
# 必要に応じてQamomileをインストールし、インポートします。リソース推定では、カーネルを実行せずに解析します。後の保存・復元の例では、Qamomileに組み込まれたシミュレーターで、復元した推定結果に含まれるプログラムを実行します。

# %%
# Install the latest Qamomile through pip.
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm

# %% [markdown]
# ## 確認に使う推定結果を用意する
#
# まず、2つの量子ビットがエンタングルしたベル状態を準備するカーネルを定義します。アダマール（H）ゲート、制御X（CX）ゲートの順に適用し、両方の量子ビットを測定します。`qm.resources.logical(...)`はこのカーネルを解析し、推定結果を`ResourceStudy`として返します。


# %%
@qm.qkernel
def bell_pair() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %%
bell_study = qm.resources.logical(bell_pair)

# %% [markdown]
# ## `ResourceStudy`から推定結果を取り出す
#
# `ResourceStudy`は、解析によって得られた推定結果を保持しています。`metric(...)`メソッドは、指定した指標の結果を`ResourceValue`として取り出します。このとき、解析をやり直すわけではありません。

# %%
gate_estimate = bell_study.metric(qm.resources.LOGICAL_GATES)
assert gate_estimate.numeric_enclosure() == (2, 2)

# %% [markdown]
# ここでの`gate_estimate`には、ゲート数の推定結果と、その前提条件や導出方法についての情報が含まれています。`gate_estimate.numeric_enclosure()`を呼び出すと、数値で表された上下限が返ります。この例では`(2, 2)`です。
#
# `ResourceRule`を渡すと、そこに指定した情報が解析に使われます。解析結果の`ResourceValue`には、関連する仮定と根拠が記録されるため、推定結果が何に依存しているかを確認できます。
#
# `ResourceValue`からは、次の情報を取得できます。
#
# - `numeric_enclosure()`：数値で表された下限と上限を返します。数値として得られない側の値は`None`になります。
# - `relation`：結果が等式、上限・下限、上下限の範囲、不完全な推定などのどれに当たるかを示します。
# - `assumptions`：リソースのルールで指定した条件や、モデルが導入した条件など、推定の前提を記録します。
# - `evidence`：カーネルの解析や、指定されたコストの仮定など、推定結果がどのように得られたかを記録します。
# - `unknown_contributors`：推定のどの部分が未確定かを示します。
#
# どの指標が含まれているかは、`study.metrics`で確認できます。これは、指標の識別子と`ResourceValue`の組を保持しています。含まれる指標は、行った解析や追加で適用したモデルによって異なります。含まれていない指標を取得しようとすると`KeyError`が発生します。指標がないことは、値が0であることを意味しません。
#
# 次のメソッドを使うと、個々の推定結果を詳しく確認できます。
#
# - `study.contributions(metric)`：寄与する演算、それぞれの推定コスト、ソースコード上の位置を返します。
# - `study.hotspots(metric)`：数値で表された上限が大きい順に寄与を並べます。数値の上限が得られないものは、その後に並びます。
# - `study.explain(metric)`：推定結果とその内訳の説明を文字列で返します。
#
# また、`to_bytes()`で推定結果を保存し、`ResourceStudy.from_bytes(...)`で復元できます。具体例は、このガイドの後半で紹介します。

# %% [markdown]
# ## 量子ビット数
#
# Qamomileでは、引数として受け取る量子ビット数、内部で確保する量子ビット数、同時に使用する量子ビット数を区別します。
#
# - `LOGICAL_INPUT_QUBITS`：引数を通じてカーネルに渡される量子ビット数です。
# - `LOGICAL_ALLOCATED_QUBITS`：解析対象のカーネル内で確保する量子ビット数です。そこから呼び出すカーネル内で確保する分も含みます。
# - `LOGICAL_PEAK_QUBITS`：同時に使用する論理量子ビット数の最大値です。
# - `LOGICAL_CIRCUIT_QUBITS`：入力として受け取る量子ビット数と、内部で確保する量子ビット数を足した、静的な回路幅です。ある量子ビットの使用が終わっても、この数は減りません。
#
# 一部の量子ビットの使用を終えてから別の量子ビットを確保する場合、同時に使用する数の最大値は、静的な回路幅より小さくなることがあります。この違いは、後の「指標の具体例」で確認します。
#
# カーネル内で明示的に確保したアンシラは、これらの論理リソースの指標に含まれます。一方、`clean_ancilla_controls(...)`のモデルが追加するアンシラは、`CIRCUIT_CLEAN_ANCILLA_QUBITS`で報告され、モデルによる量子ビット数の合計`CIRCUIT_VIRTUAL_QUBITS`にも含まれます。

# %% [markdown]
# ## ゲート数
#
# `LOGICAL_GATES`は、論理ゲート数の合計を表します。測定とリセットは別に数えます。
#
# 作用する量子ビット数ごとの内訳には、次の指標を使います。
#
# - `LOGICAL_SINGLE_QUBIT_GATES`：1つの量子ビットに作用するゲート数。
# - `LOGICAL_TWO_QUBIT_GATES`：2つの量子ビットに作用するゲート数。
# - `LOGICAL_MULTI_QUBIT_GATES`：3つ以上の量子ビットに作用するゲート数。
#
# :::{note}
# クリフォードゲートは、全体にかかる位相の違いを除いて、H・S・CXゲートを組み合わせて表せるユニタリー演算です。Sゲートは$|0\rangle$を変えず、$|1\rangle$を$i|1\rangle$に変換します。
#
# 非クリフォードゲートもユニタリーですが、これらのゲートだけでは構成できません。一例がTゲートで、$|0\rangle$を変えず、$|1\rangle$を$e^{i\pi/4}|1\rangle$に変換します。
# :::
#
# Qamomileでは、ゲートの種類ごとにも集計できます。
#
# - `LOGICAL_CLIFFORD_GATES`：HやCXなどのクリフォードゲート数。
# - `LOGICAL_ROTATION_GATES`：RX、RY、RZなど、パラメータを持つ回転ゲートや位相ゲートの数。
# - `LOGICAL_T_GATES`：Tゲートと、その逆演算である$T^\dagger$ゲートの数。
# - `LOGICAL_TOFFOLI_GATES`：トフォリ（CCX）ゲート数。
# - `LOGICAL_NON_CLIFFORD_GATES`：論理リソース解析で非クリフォードに分類される演算の数。T、トフォリ、パラメータを持つ回転ゲートなどが含まれます。
#
# Qamomileのリソース解析では、RX・RY・RZなどの回転ゲートを、指定した角度にかかわらず`LOGICAL_NON_CLIFFORD_GATES`に数えます。角度によってクリフォード演算になる場合もありますが、この集計ではその違いを判定しません。
#
# 種類別の集計には重複があります。たとえばTゲートは、`LOGICAL_T_GATES`と`LOGICAL_NON_CLIFFORD_GATES`の両方に数えられるため、種類別のゲート数をすべて足し合わせて総数を求めることはできません。
#
# 特定の演算を数えるには、その内部名を`qm.resources.logical_operation_count(...)`に渡します。たとえば、`"quantum.h"`は`qm.h`で適用するHゲートの演算を識別します。

# %%
h_estimate = bell_study.metric(qm.resources.logical_operation_count("quantum.h"))

# %% [markdown]
# `bell_pair`にはHゲートが1つ含まれるため、`h_estimate.numeric_enclosure()`は`(1, 1)`を返します。

# %% [markdown]
# ## 測定とリセット
#
# 次の指標は、量子ビットごとの測定回数とリセット回数を表します。
#
# - `LOGICAL_MEASUREMENTS`：個々の量子ビットを測定する回数。
# - `LOGICAL_RESETS`：個々の量子ビットを$|0\rangle$にリセットする回数。
#
# 関数を1回呼び出してレジスタ全体を操作する場合でも、測定またはリセットする量子ビット1つにつき、それぞれ1回と数えます。
#
# これらの値は、カーネル1回の実行に対応します。`shots`を増やしても、リソース解析が返す値がその回数倍になるわけではありません。

# %% [markdown]
# ## 深さ
#
# 次の指標は、演算の依存関係に沿った深さを表します。
#
# - `LOGICAL_SPAN`：論理解析の対象となる演算全体の依存関係の深さ。
# - `LOGICAL_GATE_DEPTH`：ゲートだけを数えた深さ。
# - `LOGICAL_CLIFFORD_DEPTH`：クリフォードゲートだけを数えた層数。
# - `LOGICAL_NON_CLIFFORD_DEPTH`：論理解析で非クリフォードに分類される演算だけを数えた層数。
# - `LOGICAL_ROTATION_DEPTH`、`LOGICAL_T_DEPTH`、`LOGICAL_TOFFOLI_DEPTH`：それぞれ、回転ゲート、Tゲートとその逆演算、トフォリゲートだけを数えた深さ。
# - `LOGICAL_MEASUREMENT_DEPTH`、`LOGICAL_RESET_DEPTH`：それぞれ、測定だけ、リセットだけを数えた深さ。
#
# 特定の種類の深さを求める場合、それ以外の演算は層数に加算されません。ただし、それらの演算との依存関係は保たれます。たとえば、同じ量子ビットに対して、2つのクリフォードゲートの間に回転ゲートを適用する場合、クリフォード深さでは2つの層として数えます。

# %% [markdown]
# ## カーネルの呼び出し回数とオラクルへの問い合わせ回数
#
# 以下の`oracle`は、本体を`...`として宣言した、実装のないユニタリーな量子カーネルです。そのため、内部のゲート列は解析できません。`use_oracle`は、このカーネルを`rounds`回呼び出してから量子ビットを測定します。


# %%
@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[2]], rounds: qm.Int) -> qm.Q[qm.Bits[2]]: ...


@qm.qkernel
def use_oracle(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    for _ in range(rounds):
        state = oracle(state, rounds)
    return qm.measure(state)


# %% [markdown]
# この例では、`oracle`を1回呼び出すと`2 * rounds`個の論理ゲートを使うと仮定します。`rule_parameter(...)`は、オラクルの引数`rounds`をルールの式に含めるために使います。`Relation.EQUALITY`は、この仮定の下でゲート数がその式に等しいことを表します。Qamomileが実装を調べて仮定を検証するわけではありません。
#
# `rounds=2`で`use_oracle`を特殊化し、このルールを使って解析します。仮定の下では1回の呼び出しで4ゲートを使うため、2回の呼び出しで合計8ゲートになります。`rounds >= 0`という条件も、推定結果に記録されます。

# %%
oracle_rounds = qm.resources.rule_parameter(
    oracle,
    "rounds",
    domain=qm.resources.Domain.NATURAL,
    constraints=("rounds >= 0",),
)
oracle_gates = qm.resources.ResourceRule(
    identity="tutorial.oracle.gates.v1",
    callable=oracle,
    metric=qm.resources.LOGICAL_GATES,
    relation=qm.resources.Relation.EQUALITY,
    expression=qm.resources.multiply(qm.resources.literal(2), oracle_rounds),
    assumptions=("rounds >= 0",),
    validity_domain=("rounds is an integer",),
    provenance="tutorial.oracle.contract.v1",
)

# %%
oracle_program = qm.specialize(use_oracle, rounds=2)
oracle_study = qm.resources.logical(
    oracle_program,
    rules=oracle_gates,
    strict=True,
)

# %% [markdown]
#
# 次の関数は、カーネルの呼び出し回数やオラクルへの問い合わせ回数を表す指標を返します。
#
# - `qm.resources.algorithmic_call_count(oracle.symbol)`：`oracle`の呼び出し回数。
# - `qm.resources.algorithmic_query_count(oracle.symbol)`：実装のない量子カーネル`oracle`への問い合わせ回数。
#
# `oracle.symbol`は、カーネル`oracle`を識別する名前です。各関数が返す指標を`oracle_study.metric(...)`に渡すと、対応する推定結果を取り出せます。
#
# `rounds=2`なので`oracle`を2回呼び出します。この例では1回の呼び出しを1回の問い合わせとして数えるため、どちらの指標も上下限は`(2, 2)`です。これらの呼び出しで使われるゲート数は、`LOGICAL_GATES`で別に集計します。

# %%
oracle_calls = qm.resources.algorithmic_call_count(oracle.symbol)
oracle_queries = qm.resources.algorithmic_query_count(oracle.symbol)

print(
    "logical gates:",
    oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure(),
)
print("oracle calls:", oracle_study.metric(oracle_calls).numeric_enclosure())

assert oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (8, 8)
assert oracle_study.metric(oracle_calls).numeric_enclosure() == (2, 2)
assert oracle_study.metric(oracle_queries).numeric_enclosure() == (2, 2)
assert qm.resources.EvidenceAuthority.TRUSTED_AXIOM in {
    item.authority for item in oracle_study.metric(qm.resources.LOGICAL_GATES).evidence
}

# %% [markdown]
# ## 指標の具体例
#
# ここからは、具体例を使って量子ビット数、演算回数、深さの違いを確認します。以下のカーネルは、すべての量子ビットが0の状態と、すべてが1の状態を等しく重ね合わせたGHZ（Greenberger–Horne–Zeilinger）状態を準備します。まず`width=4`で解析し、その後に指標の特徴がわかる小さなカーネルを取り上げます。
#


# %%
@qm.qkernel
def ghz_state(
    width: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    state = qm.h(state, target=0)
    for index in range(width - 1):
        state = qm.cx(state, control=index, target=index + 1)
    return qm.measure(state)


# %% [markdown]
# ### 量子ビット数
#
# `width=4`の`ghz_state`について、4種類の量子ビット数を比較します。

# %%
ghz_program = qm.specialize(ghz_state, width=4)
ghz_study = qm.resources.logical(ghz_program)
input_qubits = ghz_study.metric(qm.resources.LOGICAL_INPUT_QUBITS)
allocated_qubits = ghz_study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS)
ghz_peak = ghz_study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
circuit_qubits = ghz_study.metric(qm.resources.LOGICAL_CIRCUIT_QUBITS)

print("input quantum bits:", input_qubits.numeric_enclosure())
print("allocated quantum bits:", allocated_qubits.numeric_enclosure())
print("peak quantum bits:", ghz_peak.numeric_enclosure())
print("static circuit width:", circuit_qubits.numeric_enclosure())

assert input_qubits.numeric_enclosure() == (0, 0)
assert allocated_qubits.numeric_enclosure() == (4, 4)
assert ghz_peak.numeric_enclosure() == (4, 4)
assert circuit_qubits.numeric_enclosure() == (4, 4)

# %% [markdown]
# このカーネルは、量子ビットを引数として受け取らず、内部で4つ確保します。4つすべてを同時に使用するため、確保する量子ビット数、同時に使用する数の最大値、静的な回路幅はいずれも4です。
#
# これらの値が常に一致するとは限りません。次のカーネルでは、最初の量子ビットを測定してから、別の量子ビットを確保します。


# %%
@qm.qkernel
def released_qubit_example() -> tuple[qm.Shot[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    first = qm.qubits(1)
    measured = qm.measure(first)
    second = qm.qubits(1)
    return measured, second


# %%
released_study = qm.resources.logical(released_qubit_example)
released_peak = released_study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
released_allocated = released_study.metric(qm.resources.LOGICAL_ALLOCATED_QUBITS)
released_circuit = released_study.metric(qm.resources.LOGICAL_CIRCUIT_QUBITS)

print("peak quantum bits:", released_peak.numeric_enclosure())
print("allocated quantum bits:", released_allocated.numeric_enclosure())
print("static circuit width:", released_circuit.numeric_enclosure())

assert released_peak.numeric_enclosure() == (1, 1)
assert released_allocated.numeric_enclosure() == (2, 2)
assert released_circuit.numeric_enclosure() == (2, 2)

# %% [markdown]
# 合計2つの量子ビットを確保しますが、それぞれの使用期間は重なりません。そのため、同時に使用する量子ビット数の最大値は1で、確保する量子ビット数と静的な回路幅はどちらも2です。
#
# ### ゲート数・測定回数・リセット回数
#
# 同じ4量子ビットの`ghz_state`について、各ゲートが作用する量子ビット数ごとにゲート数を確認します。

# %%
ghz_gates = ghz_study.metric(qm.resources.LOGICAL_GATES)
single_qubit_gates = ghz_study.metric(qm.resources.LOGICAL_SINGLE_QUBIT_GATES)
two_qubit_gates = ghz_study.metric(qm.resources.LOGICAL_TWO_QUBIT_GATES)
multi_qubit_gates = ghz_study.metric(qm.resources.LOGICAL_MULTI_QUBIT_GATES)
ghz_measurements = ghz_study.metric(qm.resources.LOGICAL_MEASUREMENTS)
ghz_resets = ghz_study.metric(qm.resources.LOGICAL_RESETS)

print("total gates:", ghz_gates.numeric_enclosure())
print("single-quantum-bit gates:", single_qubit_gates.numeric_enclosure())
print("two-quantum-bit gates:", two_qubit_gates.numeric_enclosure())
print("multi-quantum-bit gates:", multi_qubit_gates.numeric_enclosure())

assert ghz_gates.numeric_enclosure() == (4, 4)
assert single_qubit_gates.numeric_enclosure() == (1, 1)
assert two_qubit_gates.numeric_enclosure() == (3, 3)
assert multi_qubit_gates.numeric_enclosure() == (0, 0)

# %% [markdown]
# このカーネルは、1量子ビットに作用するHadamardゲートを1回、2量子ビットに作用するCXゲートを3回適用します。3つ以上の量子ビットに作用するゲートはありません。それぞれのゲート数は1、3、0で、合計は4です。
#
# 測定とリセットは、ゲート数とは別に数えます。

# %%
print("measurements:", ghz_measurements.numeric_enclosure())
print("resets:", ghz_resets.numeric_enclosure())

assert ghz_measurements.numeric_enclosure() == (4, 4)
assert ghz_resets.numeric_enclosure() == (0, 0)

# %% [markdown]
# 最後の`qm.measure(state)`は4つの量子ビットをすべて測定するため、測定回数は4です。リセットは行わないため、リセット回数は0です。これらはカーネル1回の実行に対応する値であり、`shots`を変更しても変わりません。
#
# ### 深さ
#
# ゲート数は、ゲートを適用する回数を表します。一方、ゲート深さは、演算の依存関係を守って実行するために必要なゲートの層数を表します。測定とリセットの深さも、それぞれ別に確認できます。

# %%
ghz_gate_depth = ghz_study.metric(qm.resources.LOGICAL_GATE_DEPTH)
measurement_depth = ghz_study.metric(qm.resources.LOGICAL_MEASUREMENT_DEPTH)
reset_depth = ghz_study.metric(qm.resources.LOGICAL_RESET_DEPTH)

print("gate depth:", ghz_gate_depth.numeric_enclosure())
print("measurement depth:", measurement_depth.numeric_enclosure())
print("reset depth:", reset_depth.numeric_enclosure())

assert ghz_gate_depth.numeric_enclosure() == (4, 4)
assert measurement_depth.numeric_enclosure() == (1, 1)
assert reset_depth.numeric_enclosure() == (0, 0)

# %% [markdown]
# この例では、Hadamardゲートと3つのCXゲートが順につながっています。最初のCXゲートはHadamardゲートに依存し、それ以降のCXゲートは、直前のCXゲートのターゲットを制御量子ビットとして使います。そのため、4つのゲートには順に4層が必要です。
#
# 4つの測定はそれぞれ異なる量子ビットに作用するため、測定の深さは1です。リセットは行わないため、リセットの深さは0です。

# %% [markdown]
# ### ゲートの種類別の深さ
#
# クリフォード深さは、依存関係を守ったときに必要なクリフォードゲートの層数です。同時に実行できる複数のクリフォードゲートは、1層と数えます。2つのクリフォードゲートの間に非クリフォードゲートが入ると、そのゲート自体は数えなくても、2つのクリフォードゲートに別々の層が必要になる場合があります。
#
# RZZゲートは、2つのビット値が一致する計算基底状態と、異なる計算基底状態との間の相対位相を変える、2量子ビットの回転ゲートです。角度が$\theta$のとき、$|00\rangle$と$|11\rangle$には$e^{-i\theta/2}$を、$|01\rangle$と$|10\rangle$には$e^{i\theta/2}$を掛けます。
#
# 次のカーネルでは、量子ビット0にHゲートを適用し、両方の量子ビットに角度0.25ラジアンのRZZゲートを適用してから、量子ビット1にHゲートを適用します。


# %%
@qm.qkernel
def category_depth_example() -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.rzz(state, 0.25, left=0, right=1)
    state = qm.h(state, target=1)
    return state


# %%
category_study = qm.resources.logical(category_depth_example)
category_gate_depth = category_study.metric(qm.resources.LOGICAL_GATE_DEPTH)
category_clifford_depth = category_study.metric(qm.resources.LOGICAL_CLIFFORD_DEPTH)
category_rotation_depth = category_study.metric(qm.resources.LOGICAL_ROTATION_DEPTH)
category_non_clifford_depth = category_study.metric(
    qm.resources.LOGICAL_NON_CLIFFORD_DEPTH
)

assert category_gate_depth.numeric_enclosure() == (3, 3)
assert category_clifford_depth.numeric_enclosure() == (2, 2)
assert category_rotation_depth.numeric_enclosure() == (1, 1)
assert category_non_clifford_depth.numeric_enclosure() == (1, 1)

# %% [markdown]
# RZZゲートは最初のHゲートに依存し、2つ目のHゲートはRZZゲートに依存します。そのため、2つのHゲートは異なる量子ビットに作用していても、別々のクリフォード層として数えられます。
#
# それぞれの深さは次のとおりです。
#
# - **ゲート深さ：3**。3つすべてのゲートを数えます。
# - **クリフォード深さ：2**。RZZゲートを介した依存関係を保ちながら、2つのHゲートを数えます。
# - **回転ゲート深さ：1**。RZZゲートを数えます。
# - **非クリフォード深さ：1**。こちらもRZZゲートを数えます。
#
# この例では、RZZゲートが回転ゲートと非クリフォードゲートの両方に分類されます。種類別の深さをすべて足すと、同じゲートを重複して数えることになります。

# %% [markdown]
# ## `ResourceStudy`を保存・復元する

# %% [markdown]
# `ResourceStudy`は、`to_bytes()`でバイト列に変換し、`ResourceStudy.from_bytes(...)`で復元できます。バイト列には、リソースの推定結果、その仮定や根拠、解析対象のプログラムの情報が含まれます。
#
# このバイト列をファイルに保存すれば、後から利用できます。次の例では、変換したバイト列からメモリ上で復元し、コンパイルして実行します。そのために、まず`bell_pair`を特殊化し、得られた`Program`を解析します。

# %%
bell_program = qm.specialize(bell_pair)
executable_bell_study = qm.resources.logical(bell_program)
restored_bell_study = qm.resources.ResourceStudy.from_bytes(
    executable_bell_study.to_bytes()
)
bell_artifact = qm.compile(restored_bell_study, to=qm.Reference())
bell_record = qm.run(bell_artifact, on=qm.Reference(), shots=4, seed=7)

assert restored_bell_study == executable_bell_study
assert restored_bell_study.digest == executable_bell_study.digest
assert sum(bell_record.result.counts.values()) == 4
assert all(left == right for left, right in bell_record.result.counts)

# %% [markdown]
# 最初の2つの`assert`は、復元したstudyが元のstudyと等しく、内容を識別する値`digest`も一致することを確認します。残りの`assert`は、4回分の測定結果が得られ、各結果の2ビットが一致していることを確認します。
#
# :::{note}
# 量子カーネルから直接作成した`ResourceStudy`も、保存・復元できます。ただし、`ResourceStudy`をコンパイルするには、特殊化済みの`Program`を保持している必要があります。カーネルに`Const`パラメータがなくても、この条件は同じです。また、`evaluate(...)`で得たstudyはコンパイルできません。
#
# この条件は、`ResourceStudy`を`qm.compile(...)`に渡す場合のものです。量子カーネルを直接渡す場合は、`qm.specialize(...)`を別途呼び出す必要はありません。
# :::

# %% [markdown]
# ## 物理リソースを推定する
#
# 量子誤り訂正では、1つの論理量子ビットの情報を複数の物理量子ビットに符号化します。表面符号（surface code）はその方式の一つで、保護する量子情報そのものを読み出さずに、誤りを調べるための測定を繰り返し、その結果を訂正に利用します。
#
# この保護には、追加の量子ビットや操作が必要です。Qamomileには、必要な物理量子ビット数や実行時間を見積もるための、簡略化した表面符号のモデルが用意されています。
#
# `qm.resources.surface_code(...)`は、このモデルを`ResourceStudy`に適用します。元の論理リソースの推定結果に、物理量子ビット数や秒単位の実行時間などの推定結果を追加した、新しいstudyを返します。
#
# ここでは、先ほどのRZZゲートの例で作成した`category_study`に、デフォルト設定でモデルを適用します。このモデルは、同時に使用する論理量子ビット数の最大値と非クリフォード演算数に加えて、誤り率や誤り訂正1サイクルにかかる時間などの仮定を使います。
#
# 結果の`evidence`は、推定の根拠を記録したタプルで、複数の要素を持つ場合があります。以下の`evidence[0]`は、先頭の要素を参照します。

# %%
category_physical = qm.resources.surface_code(category_study)

physical_qubits = category_physical.metric(qm.resources.PHYSICAL_QUBIT_COUNT)
physical_duration = category_physical.metric(qm.resources.PHYSICAL_DURATION)

print("estimated physical qubits:", physical_qubits.numeric_enclosure())
print("estimated duration (seconds):", physical_duration.numeric_enclosure())

assert physical_qubits.numeric_enclosure()[0] > 0
assert physical_duration.numeric_enclosure()[0] > 0
assert (
    physical_qubits.evidence[0].authority
    is qm.resources.EvidenceAuthority.HEURISTIC_MODEL
)
assert not physical_qubits.evidence[0].verified
assert category_physical.semantic_digest == category_study.semantic_digest

# %% [markdown]
# デフォルトの仮定では、物理量子ビット数は72、実行時間は3マイクロ秒と推定されます。この組み込みモデルは説明用に簡略化されており、物理的な実装に必要なすべてのコストを考慮しているわけではありません。
#
# 結果には、近似的な物理モデルに基づく根拠として`HEURISTIC_MODEL`が記録され、`verified`は`False`になっています。推定の仮定は、`physical_qubits.assumptions`や`physical_duration.assumptions`で確認できます。
#
# 最後の`assert`は、解析対象のプログラムを識別する値`semantic_digest`が変わらないことを確認します。モデルの適用によって推定結果が追加されますが、プログラム自体は変更されません。

# %% [markdown]
# ## 推定結果の根拠と前提を確認する
#
# 数値だけでは、推定結果がどのように得られ、どのような仮定に依存しているのかはわかりません。各`ResourceValue`には、結果を解釈するための情報が含まれています。
#
# - `evidence`：推定の根拠を記録します。各要素の`authority`は、カーネルの構造から導かれた結果か、指定されたルールやモデルを仮定として採用した結果か、近似モデルによる結果かを示します。
# - `relation`：結果が等式、上限・下限、範囲、未確定の量などのどれに当たるかを示します。
# - `assumptions`：推定が成り立つための条件を記録します。
# - `unknown_contributors`：ゲート数のルールがないオラクルなど、推定の未確定な部分を示します。
#
# このガイドと[リソース推定のチュートリアル](../tutorial/04_resource_estimation.ipynb)の例では、それぞれ次のようになります。
#
# - **ベル状態を作る例（`bell_pair`）**：カーネルの構造から、ゲート数が2と求められます。根拠の`authority`は`STRUCTURALLY_DERIVED`、`relation`は`EQUALITY`です。
# - **チュートリアルの固定コストのオラクル**：指定したルールの下で、ゲート数は5です。オラクルの実装を調べずにコストを仮定として受け入れるため、結果には`TRUSTED_AXIOM`の根拠が含まれます。
# - **チュートリアルの制御演算**：クリーンアンシラモデルでは、単一制御Hゲートを1ゲートと数えるなどの仮定に基づき、回路のゲート数を5と見積もります。この結果にも`TRUSTED_AXIOM`の根拠が含まれます。
# - **コストが未指定のオラクル**：ゲート数の上下限は`(0, None)`で、`relation`は`PARTIAL_ENCLOSURE`です。ルールが不足していることは、`unknown_contributors`に記録されます。
# - **物理リソースの推定**：表面符号のモデルでは、デフォルトの仮定の下で物理量子ビット数を72と見積もります。根拠の`authority`は`HEURISTIC_MODEL`です。
#
# このように、`EQUALITY`であっても、指定されたルールや近似モデルに依存する場合があります。表面符号の例では、モデルの仮定の下で特定の値が得られたことを表しており、実機で必要となる量を厳密に予測できたことを意味するわけではありません。

# %% [markdown]
# 未確定なコストの例を確認するため、ルールを指定せずに`oracle_program`を解析します。ゲート数は負にならない一方、上限は不明なので、上下限は`(0, None)`です。以下では、この結果と、他の例の根拠や仮定を確認します。

# %%
unpriced_study = qm.resources.logical(oracle_program)
unpriced_gates = unpriced_study.metric(qm.resources.LOGICAL_GATES)
assert unpriced_gates.numeric_enclosure() == (0, None)
assert unpriced_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE

# %%
assert (
    gate_estimate.evidence[0].authority
    is qm.resources.EvidenceAuthority.STRUCTURALLY_DERIVED
)
assert gate_estimate.relation is qm.resources.Relation.EQUALITY
assert not gate_estimate.assumptions
assert "rounds >= 0" in oracle_study.metric(qm.resources.LOGICAL_GATES).assumptions
assert unpriced_gates.unknown_contributors
assert (
    physical_qubits.evidence[0].authority
    is qm.resources.EvidenceAuthority.HEURISTIC_MODEL
)

# %% [markdown]
# ### 仮定を推定結果とともに示す
#
# 推定の仮定は、結果と一緒に保存されます。たとえば、パラメータに依存するオラクルのルールでは`rounds >= 0`という条件が記録され、表面符号のモデルでは、考慮していない実装上のコストが記録されます。
#
# 推定結果を報告・比較するときは、数値とともにこれらの仮定も示します。未確定の部分がある場合は、`unknown_contributors`の内容も併記します。

# %% [markdown]
# ## まとめ
#
# - 各指標の上下限に加え、`relation`が示す関係、仮定、根拠、未確定の部分も確認します。
# - 確保する量子ビットの総数、同時に使用する数、静的な回路幅を区別します。ゲート数と、順に実行する層の数も異なります。
# - `ResourceStudy`は`to_bytes()`で保存し、`from_bytes(...)`で復元できます。コンパイルするには、特殊化された`Program`を保持している必要があります。
# - 物理リソースの推定は、選んだモデルとその仮定に依存します。結果を報告するときは、数値とともに仮定も示します。
