# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [usage, optimization]
# ---
#
# # `BinaryModel`の使い方
#
# `BinaryModel`はBINARY（0/1）変数またはSPIN（-1/1）変数の制約なし多項式目的関数を表します。このページでは係数mappingからモデルを作り、assignmentの評価、変数domainの変換、Hamiltonian・sample・OMMXとの連携を順に説明します。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools
import math

import ommx.v1

import qamomile as qm
from qamomile.optimization import BinaryModel, OMMXProblem, VarType

# %% [markdown]
# ## 係数からモデルを作る
#
# BINARY多項式には`BinaryModel.from_hubo`を使います。各キーは1つの項に含まれる変数IDのtupleで、値はその項の係数です。定数項は別に指定します。変数IDは0から始まる連番でなくても構いません。
#
# 例として$x_1+2x_3+3x_1x_3+5$を構築します。

# %%
COEFFICIENTS = {
    (1,): 1.0,
    (3,): 2.0,
    (1, 3): 3.0,
}

model = BinaryModel.from_hubo(
    COEFFICIENTS,
    constant=5.0,
    variables=(1, 3),
)

print("vartype:", model.vartype)
print("constant:", model.constant)
print("variables:", model.variables)
print("order:", model.order)
print("coefficients:", dict(model.coefficients))

assert model.vartype is VarType.BINARY
assert model.constant == 5.0
assert model.variables == (1, 3)
assert model.num_variables == 2
assert model.order == 2
assert model.coefficients == {
    (1,): 1.0,
    (1, 3): 3.0,
    (3,): 2.0,
}

# %% [markdown]
# 要素が1つのキー`(1,)`と`(3,)`は一次項、`(1, 3)`は二次項です。`model.variables == (1, 3)`はsequence assignment、Hamiltonianの量子ビット、decodeしたsampleで使う順序を表します。モデルは元の変数IDを保持します。

# %% [markdown]
# `to_ising()`を使うとBINARYモデルを等価なSPINモデルへ変換できます。逆変換には`to_binary()`を使います。

# %%
spin_equivalent = model.to_ising()

print("vartype:", spin_equivalent.vartype)
print("constant:", spin_equivalent.constant)
print("coefficients:", dict(spin_equivalent.coefficients))

assert spin_equivalent.vartype is VarType.SPIN
assert spin_equivalent.constant == 29.0 / 4.0
assert spin_equivalent.coefficients == {
    (1,): -5.0 / 4.0,
    (1, 3): 3.0 / 4.0,
    (3,): -7.0 / 4.0,
}

# %% [markdown]
# spin変数とbinary変数には$s=1-2x$の関係があります。この関係を例題へ代入すると、
#
# $$
# \begin{align*}
# x_1 + 2x_3 + 3x_1x_3 + 5
# &= \left(\frac{1-s_1}{2}\right)
# + 2\left(\frac{1-s_3}{2}\right)
# + 3\left(\frac{1-s_1}{2}\right)\left(\frac{1-s_3}{2}\right)+5 \\
# &= \frac{3}{4}s_1s_3-\frac{5}{4}s_1-\frac{7}{4}s_3+\frac{29}{4}
# \end{align*}
# $$
#
# これは上に表示した値と一致します。SPINモデルでも変数ID`(1, 3)`を保持します。

# %% [markdown]
# ## QUBO、HUBO、Isingから`BinaryModel`を構築する
#
# `from_hubo`は二次のQUBOモデルと高次のBINARY多項式も受け取ります。SPIN多項式には`from_ising`を使います。どちらも変数IDのtupleを係数のキーにします。
#
# ### QUBOから構築する
#
# QUBOは次数が二次以下のBINARY多項式なので、`from_hubo`で直接扱えます。対角成分はBINARYの恒等式$x_i^2=x_i$によって簡約されます。次のQUBO行列を考えます。
#
# $$
# \begin{bmatrix}
# 1 & 0.5 & 0 \\
# 0 & 2 & 1 \\
# 0 & 0 & 3
# \end{bmatrix}
# $$
#
# これは$x_0+2x_1+3x_2+0.5x_0x_1+x_1x_2$を表します。

# %%
qubo_model = BinaryModel.from_hubo(
    {
        (0, 0): 1.0,
        (1, 1): 2.0,
        (2, 2): 3.0,
        (0, 1): 0.5,
        (1, 2): 1.0,
    },
    variables=(0, 1, 2),
)
assert qubo_model.order == 2
assert qubo_model.coefficients == {
    (0,): 1.0,
    (0, 1): 0.5,
    (1,): 2.0,
    (1, 2): 1.0,
    (2,): 3.0,
}

print("vartype:", qubo_model.vartype)
print("constant:", qubo_model.constant)
print("coefficients:", dict(qubo_model.coefficients))

# %% [markdown]
# ### HUBOから構築する
#
# `from_hubo`は同じ係数mappingで三次以上の積も受け取ります。例えば、$x_0+2x_1+3x_2+0.5x_0x_1+x_1x_2+0.1x_0x_1x_2$を考えます。

# %%
hubo_model = BinaryModel.from_hubo(
    {
        (0,): 1.0,
        (1,): 2.0,
        (2,): 3.0,
        (0, 1): 0.5,
        (1, 2): 1.0,
        (0, 1, 2): 0.1,
    }
)
assert hubo_model.order == 3
assert hubo_model.coefficients[(0, 1, 2)] == 0.1

print("vartype:", hubo_model.vartype)
print("constant:", hubo_model.constant)
print("coefficients:", dict(hubo_model.coefficients))

# %% [markdown]
#
# ### Isingから構築する
#
# `from_ising`はSPIN変数について同じ係数キー形式を受け取り、高次の項も扱えます。

# %%
ising_model = BinaryModel.from_ising(
    {
        (0,): -1.0,
        (1,): 2.0,
        (2,): -3.0,
        (0, 1): 0.5,
        (1, 2): -1.0,
    }
)
assert ising_model.vartype is VarType.SPIN
assert ising_model.order == 2
assert ising_model.coefficients[(1, 2)] == -1.0

print("vartype:", ising_model.vartype)
print("constant:", ising_model.constant)
print("coefficients:", dict(ising_model.coefficients))

# %% [markdown]
# ## 正規化とenergy評価
#
# `BinaryModel`には`normalize_by_abs_max`と`normalize_by_rms`という2つの正規化methodと、energyを評価する`energy` methodがあります。順に見ていきます。
#
# ### 正規化
#
# Rustは、選択した代数に従って重複IDをcanonical化します。BINARY値では$x_i^n=x_i$です。SPIN値では$s_i^2=1$なので、重複IDは個数の偶奇だけが残ります。その後、同類項と空キーの定数を結合します。小さな科学計算用の係数をtoleranceで破棄することはありません。

# %%
binary_alias = BinaryModel.from_hubo(
    {
        (7, 7, 2): 3.0,
        (2, 7): -1.0,
        (): 2.0,
    },
    constant=0.5,
)
spin_alias = BinaryModel.from_ising(
    {
        (7, 7, 2): 3.0,
        (2,): -1.0,
    }
)

assert binary_alias.coefficients == {(2, 7): 2.0}
assert binary_alias.constant == 2.5
assert spin_alias.coefficients == {(2,): 2.0}
assert spin_alias.constant == 0.0

# %% [markdown]
# ### minimizerを変えずに係数をrescaleする
#
# `normalize_by_abs_max`は、すべての係数を絶対値が最大の係数で割ります。$x_1+2x_3+3x_1x_3+5$では最大の係数が3なので、正規化後のモデルは
#
# $$
# \frac{1}{3}x_1+\frac{2}{3}x_3+x_1x_3+\frac{5}{3}
# $$
#
# です。`normalize_by_rms`は次数別のRMS scaleを使います。
#
# $$
# W=\sqrt{\sum_k \frac{1}{|E_k|}\sum_{e\in E_k}w_e^2}.
# $$
#
# この例には線形項が2つ、二次項が1つあります。
#
# - $|E_1|=2$、$1^2+2^2=5$
# - $|E_2|=1$、$3^2=9$
#
# したがって$W=\sqrt{5/2+9}=\sqrt{11.5}\approx3.391$であり、正規化後の目的関数はおよそ
#
# $$
# 0.295x_1+0.590x_3+0.884x_1x_3+1.475
# $$
#
# となります。
#
# どちらもimmutableなモデルを返します。元のモデル、値domain、変数ID、minimizerは変わらず、energy scaleだけが変化します。

# %%
normalization_model = model
abs_max_model = normalization_model.normalize_by_abs_max()
rms_model = normalization_model.normalize_by_rms()
rms_factor = math.sqrt(11.5)

assert normalization_model.coefficients == {(1,): 1.0, (1, 3): 3.0, (3,): 2.0}
assert normalization_model.constant == 5.0
assert abs_max_model.coefficients == {
    (1,): 1.0 / 3.0,
    (1, 3): 1.0,
    (3,): 2.0 / 3.0,
}
assert abs_max_model.constant == 5.0 / 3.0
assert rms_model.coefficients == {
    (1,): 1.0 / rms_factor,
    (1, 3): 3.0 / rms_factor,
    (3,): 2.0 / rms_factor,
}
assert rms_model.constant == 5.0 / rms_factor

# %% [markdown]
# ### 目的関数（energy）の評価
#
# `energy`は変数assignmentに対して目的関数を評価します。$x_1+2x_3+3x_1x_3+5$について$x_1=1$、$x_3=0$のenergyを計算すると、
#
# $$
# 1+2\cdot0+3\cdot1\cdot0+5=6
# $$
#
# です。mappingは元問題のID空間で記述します。sequenceは`model.variables`に表示された密な順序、ここでは`(1, 3)`を使います。どちらも同じ検査済みRust評価境界を通ります。

# %%
assignment = {3: 0, 1: 1}

energy_from_ids = model.energy(assignment)
energy_from_dense_order = model.energy((1, 0))

print("energy:", energy_from_ids)
assert energy_from_ids == energy_from_dense_order == 6.0

# %% [markdown]
# 値domainは型と実行時契約の一部です。BINARYモデルはSPIN値を別の問題として解釈せず、エラーにします。

# %%
try:
    model.energy((-1, 1))
except ValueError as error:
    print("expected domain error:", error)
else:
    raise AssertionError("a BINARY model accepted a SPIN assignment")

# %% [markdown]
# ## BINARYとSPINを正確に変換する
#
# `to_ising`は元のIDとすべての目的関数値を保持しながら$s_i=1-2x_i$を適用します。`to_binary`はその逆変換です。この例は小さいため、sampleではなく4通りすべてのassignmentを決定論的に確認します。

# %%
spin_model = model.to_ising()

assert spin_model.vartype is VarType.SPIN
assert spin_model.variables == model.variables
assert spin_model.to_binary() == model

for bits in itertools.product((0, 1), repeat=model.num_variables):
    spins = tuple(1 - 2 * bit for bit in bits)
    assert spin_model.energy(spins) == model.energy(bits)

# %% [markdown]
# ## 同じ目的関数をHamiltonianへ射影する
#
# `model.hamiltonian`は、可換なPauli-Z wordの和で表すimmutableかつbackend-neutralな値です。量子ビット位置0は変数ID `1`、位置1は`3`に対応し、Pythonが係数を再構築することはありません。

# %%
hamiltonian = model.hamiltonian

print("Hamiltonian terms:", len(hamiltonian.terms))
assert hamiltonian.num_qubits == model.num_variables == 2
assert hamiltonian.is_hermitian
assert hamiltonian.is_pairwise_commuting
assert max(len(word.factors) for word, _ in hamiltonian.terms) == 2

# %% [markdown]
# ## 論理sampleを元のID空間へdecodeする
#
# 各backendは、まず固有のbit順をQamomileの論理source順へ正規化します。続いて`BinaryModel.decode`が各unique rowを元のIDへ戻し、同じ多項式で評価してoccurrence countを保持します。このusageページではすでに正規化された`SampleResult`を直接構築するため、simulatorの起動は不要です。

# %%
result = qm.SampleResult.from_counts(
    backend="example",
    counts={
        (1, 0): 3,
        (0, 1): 1,
    },
)
decoded = model.decode(result)

for sample in decoded.samples:
    print(sample.values, "energy =", sample.energy, "count =", sample.occurrences)

assert decoded.vartype is VarType.BINARY
assert decoded.total_occurrences == 4
assert decoded.mean_energy == 6.25
assert decoded.lowest.assignment == ((1, 1), (3, 0))
assert decoded.lowest.energy == 6.0

# %% [markdown]
# ## 1つのcanonical problem identityを永続化する
#
# serialize形式はcanonicalかつ有界で、integrity checkを備え、Python callbackやbackend objectを含みません。したがって、同じcanonical modelは同じbyte列とSHA-256 semantic digestを持ちます。

# %%
payload = model.to_bytes()
restored = BinaryModel.from_bytes(payload)

print("semantic digest:", model.semantic_digest)
assert restored == model
assert restored.to_bytes() == payload
assert restored.semantic_digest == model.semantic_digest
assert len(model.semantic_digest) == 64

# %% [markdown]
# ## OMMXから`BinaryModel`を構築する
#
# 最後に、同じモデルをOMMX（Open Mathematical prograMming eXchange）から構築します。OMMXは数理最適化データをsoftware間・人間間で交換するためのopen data formatとSDKです。実際のapplicationではpackaged problemや数理modeling toolからinstanceを受け取ることが多いですが、ここでは例を自己完結させるためlow-level componentから作成します。
#
# `OMMXProblem.from_instance`は渡されたinstanceをcopyし、BINARY多項式を構築して、元の目的関数と制約を後の評価用に保持します。feasibilityや元の目的関数値が必要な場合に使います。

# %%
x_1 = ommx.v1.DecisionVariable.binary(1, name="x_1")
x_3 = ommx.v1.DecisionVariable.binary(3, name="x_3")
instance = ommx.v1.Instance.from_components(
    decision_variables=[x_1, x_3],
    objective=x_1 + 2 * x_3 + 3 * x_1 * x_3 + 5,
    constraints=[],
    sense=ommx.v1.Instance.MINIMIZE,
)
original_payload = instance.to_bytes()

problem = OMMXProblem.from_instance(instance)
evaluation = problem.evaluate(decoded)

assert instance.to_bytes() == original_payload
assert problem.model == model
assert evaluation.total_occurrences == decoded.total_occurrences
assert evaluation.objectives == {0: 7.0, 1: 6.0}
assert all(evaluation.feasible.values())
assert evaluation.best_feasible.objective == 6.0

# %% [markdown]
# ## まとめ
#
# 係数mappingがある場合は`from_hubo`または`from_ising`から始めます。`BinaryModel`は元の変数IDを保持し、energy評価、domain変換、Hamiltonian射影、型付きsample decode、canonical persistenceに使えます。元の制約と目的関数値も必要な場合は`OMMXProblem.from_instance`を使います。

# %% [markdown]
# ## 関連トピック
#
# - [QAOAでMaxCutを解く](../algorithm/qaoa_maxcut)では、graphから多項式を作りQAOAを適用します。
# - [QAOAによるgraph partitioning](../algorithm/qaoa_graph_partition)では、OMMX instanceへQAOAを適用し、元のmodelでfeasibilityを評価します。
