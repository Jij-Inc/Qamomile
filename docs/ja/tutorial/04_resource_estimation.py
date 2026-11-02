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
# tags: [tutorial, resource-estimation]
# ---
#
# # リソース推定
#
# この章では、Qamomileを使って、量子カーネルを実行せずに必要な量子ビット数や量子演算の回数を推定する方法を説明します。主に、同時に使用する量子ビット数の最大値、ゲート数、ゲート深さといった論理リソースを扱います。ゲート深さは、演算の依存関係を守りながら、同時に実行できるゲートを一つの層にまとめたときの層数です。
#
# まず、ベル状態を準備する回路から始め、すべての量子ビットが0の状態と、すべてが1の状態を等しく重ね合わせたGHZ状態へと例を広げます。量子ビット数を変えながら、必要なリソースがどう変わるかを確認します。その過程で、推定値と上下限の読み方、パラメータを含む式で表された推定結果に値を代入する方法、合計に各演算がどのように寄与しているかを調べる方法を学びます。
#
# さらに、実装を記述していない量子カーネルに、必要なゲート数などのコストを指定する方法と、ゲートへの分解方法を定めたモデルを使って制御演算のリソースを推定する方法を紹介します。これらの例を通じて、カーネルの解析から得られる結果と、指定したコストやモデルの仮定に依存する結果の違いも確認します。

# %% [markdown]
# ## セットアップ
#
# 必要に応じてQamomileをインストールし、インポートします。論理リソースの推定では量子カーネルを実行しないため、実行用のエンジンは必要ありません。

# %%
# Install the latest Qamomile through pip.
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm

# %% [markdown]
# ## 最初のリソース推定
#
# まず、2つの量子ビットがエンタングルしたベル対を準備するカーネルを使います。最初の量子ビットにアダマール（H）ゲートを適用し、続いて、その量子ビットを制御側、もう一方をターゲット側として制御X（CX）ゲートを適用します。最後に、両方の量子ビットを測定します。


# %%
@qm.qkernel
def bell_pair() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %% [markdown]
# このカーネルを`qm.resources.logical(...)`に直接渡して、論理リソースを解析します。事前に`qm.specialize(...)`を呼び出したり、特定の量子コンピュータ実機向けにコンパイルしたりする必要はありません。
#
# 解析結果は`ResourceStudy`として返されます。その`metric(...)`メソッドを使うと、ゲート数や同時に使用する量子ビット数の最大値など、各指標の推定結果を`ResourceValue`として取り出せます。
#
# `numeric_enclosure()`は、数値で表された下限と上限を`(lower, upper)`の組で返します。上下限が等しければ、推定値はその値に定まります。たとえば、`(2, 2)`なら推定値は2です。

# %%
bell_study = qm.resources.logical(bell_pair)

peak = bell_study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
gates = bell_study.metric(qm.resources.LOGICAL_GATES)
measurements = bell_study.metric(qm.resources.LOGICAL_MEASUREMENTS)
resets = bell_study.metric(qm.resources.LOGICAL_RESETS)
gate_depth = bell_study.metric(qm.resources.LOGICAL_GATE_DEPTH)

print("peak logical qubits:", peak.numeric_enclosure())
print("logical gates:", gates.numeric_enclosure())
print("logical measurements:", measurements.numeric_enclosure())
print("logical resets:", resets.numeric_enclosure())
print("logical gate depth:", gate_depth.numeric_enclosure())

assert peak.numeric_enclosure() == (2, 2)
assert gates.numeric_enclosure() == (2, 2)
assert measurements.numeric_enclosure() == (2, 2)
assert resets.numeric_enclosure() == (0, 0)
assert gate_depth.numeric_enclosure() == (2, 2)
assert gates.relation is qm.resources.Relation.EQUALITY

# %% [markdown]
# この例では、5つの指標すべてについて正確な値が得られています。カーネルは2つの量子ビットを同時に使い、2つのゲートを適用して、両方の量子ビットを測定します。リセットは行いません。CXゲートは最初の量子ビットへのHゲートの後に実行する必要があるため、ゲート深さは2です。測定はゲートとは別に数えます。
#
# これらの値はカーネルの解析から得られたもので、量子計算や測定を実際に行った結果ではありません。

# %% [markdown]
# ## 各指標の意味
#
# 前の例では、量子ビットの使用数、演算の回数、ゲート深さを確認しました。それぞれ、カーネルに必要なリソースの異なる側面を表しています。
#
# ### 同時に使用する量子ビット数
#
# `LOGICAL_PEAK_QUBITS`は、同時に使用する論理量子ビット数の最大値です。`bell_pair`では、両方の量子ビットを測定まで使い続けるため、この値は2になります。
#
# これは、カーネル全体で確保する量子ビットの総数とは異なります。ある量子ビットの使用が終わってから別の量子ビットを確保する場合、両者を同時使用として数えることはありません。引数として受け取る量子ビット数、内部で確保する量子ビット数、静的な回路幅との関係は、[リソース推定の詳細ガイド](../usage/resource_estimation.ipynb)で説明します。
#
# ### ゲート数・測定回数・リセット回数
#
# `LOGICAL_GATES`は、論理ゲートの適用回数を表します。作用する量子ビット数にかかわらず、一つのゲートの適用を1回と数えます。`bell_pair`では、HゲートとCXゲートがそれぞれ1回ずつなので、合計は2です。
#
# 作用する量子ビット数ごとの内訳も確認できます。
#
# - `LOGICAL_SINGLE_QUBIT_GATES`：1つの量子ビットに作用するゲート数。
# - `LOGICAL_TWO_QUBIT_GATES`：2つの量子ビットに作用するゲート数。
# - `LOGICAL_MULTI_QUBIT_GATES`：3つ以上の量子ビットに作用するゲート数。
#
# 測定とリセットは、`LOGICAL_MEASUREMENTS`と`LOGICAL_RESETS`を使い、量子ビットごとに数えます。`bell_pair`での`qm.measure(state)`の呼び出しは1回ですが、2つの量子ビットを測定するため、測定回数は2です。リセットは行わないため、リセット回数は0です。
#
# これらの値はカーネル1回の実行に対応しており、ショット数を掛けた値ではありません。
#
# ### ゲート深さ
#
# ゲート数がゲートの適用回数を表すのに対し、`LOGICAL_GATE_DEPTH`は、演算の依存関係を守って実行するために必要なゲートの層数を表します。異なる量子ビットに作用し、互いに依存しないゲートは、同じ層にまとめられます。
#
# `bell_pair`では、Hゲートの後にCXゲートが同じ量子ビットを使うため、別々の層になります。したがって、ゲート数もゲート深さも2です。
#
# 比較として、各量子ビットに独立にHゲートを適用するカーネルを考えます。


# %%
@qm.qkernel
def parallel_hadamards() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    return qm.measure(state)


# %%
parallel_study = qm.resources.logical(parallel_hadamards)

parallel_gates = parallel_study.metric(qm.resources.LOGICAL_GATES)
parallel_depth = parallel_study.metric(qm.resources.LOGICAL_GATE_DEPTH)

print("logical gates:", parallel_gates.numeric_enclosure())
print("logical gate depth:", parallel_depth.numeric_enclosure())

assert parallel_gates.numeric_enclosure() == (2, 2)
assert parallel_depth.numeric_enclosure() == (1, 1)
assert parallel_study.metric(
    qm.resources.LOGICAL_MEASUREMENT_DEPTH
).numeric_enclosure() == (1, 1)

# %% [markdown]
# 2つのHゲートはコード上では別々の行に書かれていますが、互いに依存しないため、一つのゲート層にまとめられます。
#
# :::{note}
# 論理ゲート深さは、演算の依存関係に基づく層数です。特定のデバイスでの実行時間を表すものではありません。測定とリセットには、それぞれ専用の深さの指標があり、ゲート深さには加算されません。
# :::

# %% [markdown]
# ## 問題サイズによる比較
#
# 必要なリソースは、問題の規模によって変わります。Qamomileでは、カーネルの`Const`パラメータを含む式としてリソースを推定できるため、カーネルを実行せずに複数の問題サイズを比較できます。

# %% [markdown]
# ### パラメータを含む推定式に値を代入する
#
# $n=\texttt{width}$個の量子ビットからなるGHZ（Greenberger–Horne–Zeilinger）状態は、次のように表されます。
#
# $$
# |\mathrm{GHZ}_n\rangle
# =\frac{|0\rangle^{\otimes n}+|1\rangle^{\otimes n}}{\sqrt{2}}.
# $$
#
# たとえば、4量子ビットの場合は$(|0000\rangle+|1111\rangle)/\sqrt{2}$です。
#
# 以下のカーネルでは、最初の量子ビットにHゲートを適用してから、隣り合う量子ビットにCXゲートを順番に適用します。`width=2`では、先ほどの例と同じベル状態が得られます。
#
# `width`は量子ビット数、ループの繰り返し回数、測定結果のビット数を決めるため、`Const`パラメータとして宣言します。


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
# `width`を指定せずにこのカーネルを`qm.resources.logical(...)`に渡すと、`width`を含む推定式を保持した`ResourceStudy`が返されます。
#
# `study.evaluate(...)`を使うと、その式に具体的な値を代入できます。呼び出すたびに新しい`ResourceStudy`が返されるため、元の推定式を使って別の値も調べられます。

# %%
symbolic_study = qm.resources.logical(ghz_state)

# 量子ビット数を変えて、論理ゲート数とゲート深さを比較します。
for width in (2, 4, 8):
    estimate = symbolic_study.evaluate(width=width)
    width_gates = estimate.metric(qm.resources.LOGICAL_GATES)
    width_depth = estimate.metric(qm.resources.LOGICAL_GATE_DEPTH)

    print(
        f"width={width}: "
        f"gates={width_gates.numeric_enclosure()}, "
        f"depth={width_depth.numeric_enclosure()}"
    )

    assert width_gates.numeric_enclosure() == (width, width)
    assert width_depth.numeric_enclosure() == (width, width)
    assert estimate.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        width,
        width,
    )
    assert estimate.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        width,
        width,
    )

# %% [markdown]
# ここで比較したいずれの`width`でも、Hゲートを1回、CXゲートを`width - 1`回使うため、ゲート数は合計`width`です。
#
# 最初のCXゲートは、量子ビット0へのHゲートの後に適用する必要があります。それ以降のCXゲートも、直前のCXゲートのターゲット量子ビットを制御量子ビットとして使うため、直前のゲートより先には実行できません。このため、Hゲートと`width - 1`個のCXゲートは順番に実行する`width`個の層となり、ゲート深さも`width`になります。
#
# また、`width`個の量子ビットを同時に使い、最後にすべて測定するため、同時に使用する量子ビット数の最大値と測定回数も`width`です。これらの推定式を評価しても、量子回路は実行されません。

# %% [markdown]
# ### 特殊化によって、より正確に推定できる場合
#
# `study.evaluate(...)`は、すでに導出されたリソースの式に値を代入する処理です。カーネルを解析し直すわけではありません。
#
# 特定の`Const`値を固定してから解析するには、`qm.specialize(...)`が返す`Program`を`qm.resources.logical(...)`に渡します。その値によって演算の依存関係が決まる場合には、より正確な推定結果が得られることがあります。
#
# たとえば、以下のカーネルは量子ビット0にHゲートを、量子ビット`i`にXゲートを適用します。


# %%
@qm.qkernel
def symbolic_access(i: qm.Const[qm.Int]) -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.x(state, target=i)
    return state


# %% [markdown]
# `i=0`なら両方のゲートが同じ量子ビットに作用するため、2つの層が必要です。`i=1`なら異なる量子ビットに作用するため、一つの層にまとめられます。

# %%
access_symbolic = qm.resources.logical(symbolic_access)
access_depth = access_symbolic.metric(qm.resources.LOGICAL_GATE_DEPTH)

access_evaluated = access_symbolic.evaluate(i=1)
evaluated_depth = access_evaluated.metric(qm.resources.LOGICAL_GATE_DEPTH)

access_i0 = qm.resources.logical(qm.specialize(symbolic_access, i=0))
access_i1 = qm.resources.logical(qm.specialize(symbolic_access, i=1))

depth_i0 = access_i0.metric(qm.resources.LOGICAL_GATE_DEPTH)
depth_i1 = access_i1.metric(qm.resources.LOGICAL_GATE_DEPTH)

# 値の指定前、推定式の評価後、特殊化後でゲート深さの上下限を比較します。
print("gate depth before specifying i:", access_depth.numeric_enclosure())
print("gate depth after evaluate(i=1):", evaluated_depth.numeric_enclosure())
print(
    "gate depth after specializing with i=0:",
    depth_i0.numeric_enclosure(),
)
print(
    "gate depth after specializing with i=1:",
    depth_i1.numeric_enclosure(),
)

assert access_depth.numeric_enclosure() == (1, 2)
assert evaluated_depth.numeric_enclosure() == (1, 2)
assert depth_i0.numeric_enclosure() == (2, 2)
assert depth_i1.numeric_enclosure() == (1, 1)

# %% [markdown]
# `i`が未指定の場合、解析結果はゲート深さを1以上2以下と報告します。`evaluate(i=1)`を呼び出してもゲート間の依存関係は再解析されないため、この上下限は変わりません。一方、特殊化したプログラムを解析すると、Xゲートが作用する量子ビットが確定し、正確な深さが得られます。
#
# すでに得られた推定式の値を比較する場合は`evaluate(...)`を使います。特定のカーネルの設定を解析に反映したい場合は、特殊化してからリソースを解析します。

# %% [markdown]
# ## リソースの内訳を調べる
#
# ゲート数の合計から、カーネルが何個のゲートを必要とするかがわかります。その内訳を知るには、合計に含まれる個々の演算を調べます。
#
# `ResourceStudy`には、次の3つのメソッドがあります。
#
# - `contributions(...)`：各演算の寄与を、推定値とソースコード上の位置とともに返します。
# - `hotspots(...)`：数値で表された上限が大きい順に、デフォルトで最大10件の寄与を返します。数値の上限が得られないものは、その後に並びます。
# - `explain(...)`：指定した指標の合計と、その内訳となる演算の説明を文字列で返します。
#
# 先ほど作成した`bell_study`を使い、論理ゲート数の内訳を確認します。

# %%
gate_contributions = bell_study.contributions(qm.resources.LOGICAL_GATES)
gate_hotspots = bell_study.hotspots(qm.resources.LOGICAL_GATES)
explanation = bell_study.explain(qm.resources.LOGICAL_GATES)

print(explanation)

assert [item.operation for item in gate_contributions] == ["quantum.h", "quantum.cx"]
assert set(gate_hotspots) == set(gate_contributions)
assert all(item.value.numeric_enclosure() == (1, 1) for item in gate_contributions)
assert "quantum.h" in explanation
assert "quantum.cx" in explanation

# %% [markdown]
# 説明には、論理ゲート数の合計が2と表示されます。`quantum.h`と`quantum.cx`は、`bell_pair`内のHゲートとCXゲートに対応し、それぞれ1ゲート分を占めています。ここでは`LOGICAL_GATES`を指定しているため、別に数える測定は、この内訳には含まれません。
#
# この例では両方のゲートの寄与が同じですが、より大きなカーネルでは、`hotspots(...)`を使って、指定した指標への寄与が大きい演算を見つけられます。

# %% [markdown]
# ## 実装のないカーネルにコストを指定する
#
# アルゴリズムが呼び出すカーネルをすべて実装する前に、必要なリソースを見積もりたい場合があります。Qamomileでは、入力と出力を宣言し、実装を記述しない量子カーネルを使って、このような解析を行えます。このカーネルを、*opaque quantum kernel*と呼びます。
#
# `ResourceRule`を使うと、そのカーネルに必要なゲート数などについて、仮定を指定できます。Qamomileは、この仮定と呼び出し側のコードを組み合わせて、全体に必要なリソースを推定します。
#
# :::{note}
# 実装もリソースのルールも与えられていなければ、Qamomileはそのカーネルの内部で使うゲート数を推定できません。一方、呼び出し回数などは、呼び出し側のコードから求められます。また、ゲート数のルールを与えても、ゲート深さや使われるゲートの種類まで決まるわけではありません。
# :::
#
# ### 固定のコストを指定する
#
# 以下の`oracle`は、2量子ビットに作用するユニタリー演算を宣言しています。本体は`...`となっており、具体的なゲート列は記述していません。`use_oracle`は、このカーネルを`rounds`回呼び出した後、量子ビットを測定します。


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
# ここでは、`oracle`の呼び出し1回につき、論理ゲートをちょうど5個使うと仮定します。以下のルールは、**「`oracle`の呼び出し1回あたりの論理ゲート数 = 5」** という仮定を表します。`LOGICAL_GATES`が対象の指標、`Relation.EQUALITY`が等号、`literal(5)`が値の5に対応します。このルールを`qm.resources.logical(...)`に渡して解析に使います。
#
# `identity`はルールを識別する名前です。`provenance`は、指定したコストの根拠となる仮定や参照資料を記録しておくためのメモです。この例では、「5ゲートと仮定した」という記録にラベルを付けています。
#
# ここでの`strict=True`は、`oracle`のコストが未確定のままならエラーにします。与えたルールによってゲート数は決まりますが、すべてのリソース指標が正確に求まるわけではありません。この設定については、後ほど詳しく説明します。

# %%
fixed_oracle_gates = qm.resources.ResourceRule(
    identity="tutorial.oracle.fixed-gates.v1",
    callable=oracle,
    metric=qm.resources.LOGICAL_GATES,
    relation=qm.resources.Relation.EQUALITY,
    expression=qm.resources.literal(5),
    provenance="tutorial.oracle.fixed-contract.v1",
)

fixed_oracle_program = qm.specialize(use_oracle, rounds=1)
fixed_oracle_study = qm.resources.logical(
    fixed_oracle_program,
    rules=fixed_oracle_gates,
    strict=True,
)

print(
    "logical gates:",
    fixed_oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure(),
)

assert fixed_oracle_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
    5,
    5,
)

# %% [markdown]
# `rounds=1`では`oracle`を1回呼び出すため、指定したルールに基づく論理ゲート数は5です。最後の測定は別に数えます。
#
# この結果は、与えた仮定のもとで正確な値です。Qamomileが`oracle`の実装を調べて、5ゲートであると検証したわけではありません。

# %% [markdown]
# ### 引数に応じたコストを指定する
#
# コストを、カーネルに渡す引数によって変えることもできます。先ほどの固定コストに代えて、`oracle`の呼び出し1回に必要な論理ゲート数が、その呼び出しに渡される`rounds`の2倍になると仮定します。
#
# この式を記述するため、`rule_parameter(oracle, "rounds", ...)`を使って、引数`rounds`を式の中で表す`oracle_rounds`を作成します。これを使った`multiply(literal(2), oracle_rounds)`が、`2 * rounds`を表します。値の範囲と制約には、`rounds`が0以上の整数であることを指定します。


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

# %% [markdown]
# この例の`rounds`には、二つの役割があります。`use_oracle`から`oracle`を呼び出す回数を決めるとともに、各呼び出しの引数としても渡されます。`use_oracle`では`Const`パラメータなので、そのカーネルを特殊化するときに値を固定します。
#
# `rounds=2`なら、ルールによって1回あたりのゲート数は4となります。それを2回呼び出すため、合計は8ゲートです。この解析では、先ほどの固定5ゲートのルールに代えて、新しいルールを使います。

# %%
oracle_program = qm.specialize(use_oracle, rounds=2)
oracle_study = qm.resources.logical(
    oracle_program,
    rules=oracle_gates,
    strict=True,
)

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
# 呼び出し回数は、`use_oracle`のコードを解析して求められます。この実装のないユニタリーカーネルへの呼び出しは、オラクルへの問い合わせとしても数えるため、呼び出し回数と問い合わせ回数はいずれも2です。
#
# 一方、呼び出し1回あたりのゲート数は、指定したルールに基づいています。推定結果の根拠に含まれる`TRUSTED_AXIOM`は、Qamomileが解析したゲート列ではなく、与えられた仮定に依存していることを示します。
#
# :::{note}
# リソースのコストを指定しても、そのカーネルを実行できるようになるわけではありません。実行には実装が必要です。また、実装を解析して求められるコストを、リソースのルールで上書きすることはできません。
# :::

# %% [markdown]
# ## 制御演算のリソース推定
#
# 制御演算は、さまざまなゲート列で実現できます。特定の分解方法に必要なリソースを見積もるには、まずカーネルを解析し、その後で、分解方法とゲートの数え方を定めたモデルを適用します。
#
# ここでは、3つの量子ビットで制御するHゲートを例に使います。
#
# ### 論理解析から始める
#
# `qm.control`には、ユニタリーな量子カーネルを渡します。Hゲート自体はユニタリーですが、組み込み関数`qm.h`は量子カーネルではないため、現在のAPIでは直接渡せません。そこで、`qm.h`を呼び出す量子カーネル`apply_hadamard`を定義し、それを`qm.control`に渡します。`apply_hadamard`自体は通常のHゲートを適用し、制御は`qm.control`によって付けられます。
#
# `three_controlled_hadamard`では、3つの制御量子ビットと1つのターゲット量子ビットを渡します。3つの制御量子ビットがすべて$|1\rangle$のとき、ターゲットにHゲートが適用されます。


# %%
@qm.qkernel
def apply_hadamard(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)


@qm.qkernel
def three_controlled_hadamard() -> qm.Q[qm.Bits[4]]:
    controls = qm.qubits(3)
    target = qm.qubits(1)
    controls, target = qm.control(apply_hadamard, controls, target)
    return qm.concat(controls, target)


# %% [markdown]
# 最後の`qm.concat(controls, target)`は、3つの制御量子ビット、ターゲットの順に結合して、一つの4量子ビットのレジスタにまとめます。このカーネルは測定を行わず、量子状態を返します。
#
# このカーネルを`qm.resources.logical(...)`に直接渡して解析します。特定の値で特殊化したカーネルを解析したい場合には、`qm.specialize(...)`が返す`Program`を渡すこともできます。
#
# 制御付きカーネルの呼び出し回数を数えるには、`qm.resources.logical_operation_count("qcore.control_call")`という指標を使います。`"qcore.control_call"`は、`qm.control`で記述した演算に対応するQamomile内部の識別子です。この指標を`abstract_control_study.metric(...)`に渡して、推定された回数を取り出します。
#
# ここで数えるのは制御付きカーネルの呼び出し回数です。その制御演算を実現するためのゲート数は、この後で分解モデルを使って見積もります。

# %%
abstract_control_study = qm.resources.logical(three_controlled_hadamard)
abstract_control_calls = abstract_control_study.metric(
    qm.resources.logical_operation_count("qcore.control_call")
)

print("controlled calls:", abstract_control_calls.numeric_enclosure())
print(
    "logical gate-count bounds:",
    abstract_control_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure(),
)

assert abstract_control_calls.numeric_enclosure() == (1, 1)
assert abstract_control_study.metric(
    qm.resources.LOGICAL_PEAK_QUBITS
).numeric_enclosure() == (4, 4)
assert abstract_control_study.metric(
    qm.resources.LOGICAL_GATES
).numeric_enclosure() == (1, None)

# %% [markdown]
# 解析結果から、制御付き呼び出しが1回あり、4つの論理量子ビットを使うことがわかります。ゲート数の結果`(1, None)`は、下限が1で、上限は未確定であることを表します。この段階では、制御演算の分解方法をまだ選んでいません。
#
# ### 分解モデルを選ぶ
#
# `qm.resources.clean_ancilla_controls(...)`は、クリーンアンシラを使った分解に必要なリソースを推定します。クリーンアンシラとは、$|0\rangle$に初期化し、使用後にも$|0\rangle$へ戻す補助量子ビットです。
#
# この分解では、トフォリ（CCX）ゲートを使います。トフォリゲートは、2つの制御量子ビットが両方とも$|1\rangle$のときに、ターゲットを反転します。
#
# 制御量子ビットが3つの場合、このモデルでは2つのアンシラを使います。2つのトフォリゲートで制御条件を計算し、その条件に従って制御付きHゲートをターゲットに適用します。その後、同じ2つのトフォリゲートを逆順に適用し、アンシラを元に戻します。このモデルでは、各トフォリゲートと単一制御のHゲートをそれぞれ1ゲートと数えるため、合計は5ゲートです。詳しい導出は、章末の付録「クリーンアンシラ分解の導出」で説明します。

# %%
decomposed_control_study = qm.resources.clean_ancilla_controls(abstract_control_study)

print(
    "model gate count:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_VIRTUAL_GATES
    ).numeric_enclosure(),
)
print(
    "model Toffoli count:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_TOFFOLI_GATES
    ).numeric_enclosure(),
)
print(
    "additional clean ancillas:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_CLEAN_ANCILLA_QUBITS
    ).numeric_enclosure(),
)
print(
    "model gate depth:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_GATE_DEPTH
    ).numeric_enclosure(),
)
print(
    "model qubit count:",
    decomposed_control_study.metric(
        qm.resources.CIRCUIT_VIRTUAL_QUBITS
    ).numeric_enclosure(),
)

assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_VIRTUAL_GATES
).numeric_enclosure() == (5, 5)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_TOFFOLI_GATES
).numeric_enclosure() == (4, 4)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_CLEAN_ANCILLA_QUBITS
).numeric_enclosure() == (2, 2)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_GATE_DEPTH
).numeric_enclosure() == (5, 5)
assert decomposed_control_study.metric(
    qm.resources.CIRCUIT_VIRTUAL_QUBITS
).numeric_enclosure() == (6, 6)

# %% [markdown]
# 2つのアンシラを加えるため、量子ビット数は4から6になります。この例では、5つのゲートを依存関係に従って順番に実行する必要があるため、ゲート深さは5です。
#
# `clean_ancilla_controls(...)`は、これらの`CIRCUIT_*`指標を追加した新しい`ResourceStudy`を返します。元の論理リソースの指標は保持され、カーネルの変更やコンパイルは行いません。
#
# ### 具体的な回路を記述して推定結果を比較する
#
# この分解をコードに直接記述し、そのリソース推定結果をモデルの結果と比較することもできます。
#
# 以下のカーネルでは、量子ビット0～2を制御、3をターゲット、4～5をアンシラ$a_0,a_1$として使います。コード中の番号付きコメントは、5つのゲート操作を示しています。最後の2つの操作によってアンシラが$|0\rangle$に戻る理由は、付録で説明します。


# %%
@qm.qkernel
def three_controlled_hadamard_manually() -> qm.Q[qm.Bits[6]]:
    state = qm.qubits(6)

    # 量子ビット0～2：制御、3：ターゲット、4～5：アンシラa0、a1。

    # ステップ1：制御量子ビット0と1のANDを計算し、a0に格納します。
    state = qm.ccx(state, control1=0, control2=1, target=4)

    # ステップ2：制御量子ビット2とa0のANDを計算し、a1に格納します。
    state = qm.ccx(state, control1=2, control2=4, target=5)

    controls, remainder = qm.split_at(state, 3)
    target, ancillas = qm.split_at(remainder, 1)
    first_ancilla, second_ancilla = qm.split_at(ancillas, 1)

    # ステップ3：a1を制御として、ターゲットにHゲートを適用します。
    second_ancilla, target = qm.control(
        apply_hadamard,
        second_ancilla,
        target,
    )

    # 元のインデックス順でレジスタを結合します。
    state = qm.concat(
        controls,
        qm.concat(target, first_ancilla, second_ancilla),
    )

    # ステップ4：ステップ2を元に戻し、a1を|0>に戻します。
    state = qm.ccx(state, control1=2, control2=4, target=5)

    # ステップ5：ステップ1を元に戻し、a0を|0>に戻します。
    state = qm.ccx(state, control1=0, control2=1, target=4)

    return state


# %% [markdown]
# このカーネルでも、単一制御のHゲートは`qm.control`で表しています。これを1ゲートとして数えるため、同じ分解モデルを適用します。以下の`assert`で、ゲート数、トフォリゲート数、ゲート深さ、量子ビット数を比較します。

# %%
manual_control_study = qm.resources.logical(three_controlled_hadamard_manually)
manual_control_projection = qm.resources.clean_ancilla_controls(manual_control_study)

assert (
    manual_control_study.metric(qm.resources.LOGICAL_TOFFOLI_GATES).numeric_enclosure()[
        0
    ]
    == 4
)
assert (
    manual_control_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()[0] == 5
)
assert manual_control_study.metric(
    qm.resources.LOGICAL_PEAK_QUBITS
).numeric_enclosure() == (6, 6)

for metric in (
    qm.resources.CIRCUIT_VIRTUAL_GATES,
    qm.resources.CIRCUIT_TOFFOLI_GATES,
    qm.resources.CIRCUIT_GATE_DEPTH,
    qm.resources.CIRCUIT_VIRTUAL_QUBITS,
):
    assert manual_control_projection.metric(metric).numeric_enclosure() == (
        decomposed_control_study.metric(metric).numeric_enclosure()
    )

# %% [markdown]
# どちらの方法でも、ゲート数は5、そのうちトフォリゲートは4、ゲート深さは5、量子ビット数は6となります。直接記述したカーネルでは、コード内で2つのアンシラを確保しています。先ほどの推定では、分解モデルによって、その2つのアンシラに必要なリソースが追加されました。

# %% [markdown]
# ## 不完全な推定結果を扱う
#
# 先ほどのオラクルの例では、リソースのルールを使ってゲート数を指定しました。ここでは、そのようなルールがない場合の結果を確認します。
#
# コストの情報が不足していても、`qm.resources.logical(...)`はデフォルトではエラーにせず、求められた上下限と未確定の部分を保持した推定結果を返します。以下の例では、`unpriced_step`のコストを指定しません。


# %%
@qm.qkernel(kind="unitary")
def unpriced_step(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]: ...


@qm.qkernel
def with_unpriced_oracle() -> qm.Q[qm.Bits[1]]:
    state = qm.qubits(1)
    return unpriced_step(state)


# %%
unpriced_study = qm.resources.logical(with_unpriced_oracle)
unpriced_gates = unpriced_study.metric(qm.resources.LOGICAL_GATES)

print("gate-count bounds:", unpriced_gates.numeric_enclosure())
print("unresolved costs:", unpriced_gates.unknown_contributors)

assert unpriced_gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
assert unpriced_gates.numeric_enclosure() == (0, None)
assert unpriced_gates.unknown_contributors

# %% [markdown]
# ゲート数の上下限は`(0, None)`です。ゲート数は負にならないため下限は0ですが、上限はわかっていません。オラクルが実際に使うゲート数は、未指定のままです。
#
# `Relation.PARTIAL_ENCLOSURE`は、推定が不完全であることを表します。`unknown_contributors`には未確定の部分についての説明が記録されます。この例では、`unpriced_step`のゲート数を指定するルールがないことが示されます。
#
# 未確定のコストをエラーとして扱いたい場合は、`strict=True`を指定します。同じ解析でも、今度は`ValueError`が発生します。

# %%
try:
    qm.resources.logical(with_unpriced_oracle, strict=True)
except ValueError as error:
    assert "unknown contributors" in str(error)
    print("Strict analysis rejected the unresolved resource cost.")
else:
    raise AssertionError("strict analysis must reject unresolved work")

# %% [markdown]
# ここでは、ノートブックの実行を続けられるように例外を捕捉しています。未指定のコストを補うには、先ほどの例と同様に`ResourceRule`を渡します。コストが0であると仮定する場合も、ルールで明示的に指定します。

# %% [markdown]
# ## さらに詳しく
#
# 個々の指標の説明、追加の具体例、`ResourceStudy`の保存・復元、物理リソースの推定結果と根拠の読み方は、[リソース推定の詳細ガイド](../usage/resource_estimation.ipynb)で紹介します。このガイドは単独で実行できます。

# %% [markdown]
# ## まとめ
#
# - `qm.resources.logical(...)`を使うと、量子回路を実行せずに、論理量子ビットの使用数、ゲート数、ゲート深さを推定できます。
# - 各指標の結果は`ResourceValue`として取得します。値や上下限に加えて、推定の根拠、仮定、未確定の部分も確認します。
# - `study.evaluate(...)`は、すでに得られた推定式に値を代入します。特定の`Const`値でカーネルを解析し直す場合は、特殊化して得られた`Program`を解析します。
# - `contributions(...)`、`hotspots(...)`、`explain(...)`を使うと、どの演算が推定結果に寄与しているかを調べられます。
# - 実装のないカーネルのコストがわかっている場合や、コストを仮定する場合は、`ResourceRule`で指定します。情報が足りなければ、不完全な推定結果として扱うか、`strict=True`でエラーにします。
# - 回路の分解モデルや物理モデルを使った推定は、そのモデルの仮定に依存します。結果を報告・比較するときは、数値とともに仮定も示します。

# %% [markdown]
# ## 付録：クリーンアンシラ分解の導出
#
# 「制御演算のリソース推定」で使った分解は、2つのクリーンアンシラを使って、3つの量子ビットを制御とするHゲートを実現します。ここでは、3つの制御量子ビットの値がすべて1のときだけHゲートが適用され、両方のアンシラが$|0\rangle$に戻る理由を説明します。
#
# 計算基底の各成分について、制御量子ビットの値を$c_0,c_1,c_2\in\{0,1\}$とします。ターゲットの状態は$|t\rangle$、2つのアンシラ$a_0,a_1$の初期状態は$|00\rangle$です。
#
# ここで、
#
# $$
# p=c_0c_1,\qquad q=c_0c_1c_2.
# $$
#
# と定義します。0と1の値に対する掛け算はAND（論理積）を表すため、$q=1$となるのは、3つの制御量子ビットの値がすべて1の場合だけです。以下では、$\operatorname{CCX}(x,y;z)$の$x,y$を制御、$z$をターゲットとします。
#
# 以下の5つのステップは、`three_controlled_hadamard_manually`の番号付きコメントに対応しています。
#
# 1. **最初の2つの制御条件を$a_0$に計算します。**
#
#    $c_0$と$c_1$を制御とするトフォリゲートを適用します。
#
#    $$
#    \operatorname{CCX}(c_0,c_1;a_0):
#    \qquad a_0:\ 0\longrightarrow p.
#    $$
#
# 2. **3つ目の制御条件も合わせて、$a_1$に計算します。**
#
#    $c_2$と$a_0$を制御とするトフォリゲートを適用します。
#
#    $$
#    \operatorname{CCX}(c_2,a_0;a_1):
#    \qquad a_1:\ 0\longrightarrow c_2p=q.
#    $$
#
# 3. **$a_1$を制御量子ビットとして、ターゲットにHゲートを適用します。**
#
#    $$
#    |t\rangle\longrightarrow H^q|t\rangle
#    =
#    \begin{cases}
#    |t\rangle,&q=0,\\
#    H|t\rangle,&q=1.
#    \end{cases}
#    $$
#
#    この演算では、制御量子ビットと両方のアンシラの値は変わりません。
#
# 4. **ステップ2を元に戻し、$a_1$を$|0\rangle$に戻します。**
#
#    同じトフォリゲートをもう一度適用すると、次のようになります。
#
#    $$
#    a_1:\ q\longrightarrow q\oplus c_2p=q\oplus q=0.
#    $$
#
# 5. **ステップ1を元に戻し、$a_0$を$|0\rangle$に戻します。**
#
#    最初のトフォリゲートをもう一度適用すると、次のようになります。
#
#    $$
#    a_0:\ p\longrightarrow p\oplus c_0c_1=p\oplus p=0.
#    $$
#
# $\oplus$はXOR（排他的論理和）を表します。同じ値どうしのXORは必ず0になるため、2つのトフォリゲートを逆順にもう一度適用すると、両方のアンシラが元に戻ります。このように、一時的な計算を逆演算で元に戻すことをアンコンピュートと呼びます。
#
# したがって、全体の変換は次のようになります。
#
# $$
# |c_0c_1c_2\rangle|t\rangle|00\rangle_a
# \longrightarrow
# |c_0c_1c_2\rangle
# H^{c_0c_1c_2}|t\rangle
# |00\rangle_a.
# $$
#
# 3つの制御量子ビットの値がすべて1のときだけターゲットにHゲートが適用され、両方のアンシラは$|0\rangle$に戻ります。量子演算の線形性により、重ね合わせ状態についても各成分にこの式を適用できます。制御量子ビットの測定は行いません。
#
# このモデルでは、各トフォリゲートと、制御量子ビットが1つのHゲートを、それぞれ1ゲートと数えます。
#
# $$
# \underbrace{2\ \mathrm{CCX}}_{\text{compute}}
# +\underbrace{1\ \mathrm{controlled}\ H}_{\text{apply}}
# +\underbrace{2\ \mathrm{CCX}}_{\text{uncompute}}
# =5\text{ gates}.
# $$
#
# 制御量子ビット3つ、ターゲット1つ、アンシラ2つで、必要な量子ビット数は6です。この分解では各ゲートが直前のゲートに依存するため、ゲート深さは5になります。

# %% [markdown]
# **次へ**：[チュートリアル05](05_execution_models.ipynb)では、量子カーネルの実行、測定結果の集計、観測量の期待値の推定を扱います。
