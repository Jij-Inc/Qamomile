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
# tags: [usage, resource-estimation, primitive]
# ---
#
# # Block-encodingのリソース推定
#
# block-encodingは、Hamiltonian表現とqubitized FTQC algorithmをつなぐinterfaceです。このnotebookでは、backend circuit decompositionにcommitする前に、PREPARE、SELECT、reflection、QPE readoutのcostを分けてmodel化する方法を示します。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

from qamomile.circuit.estimator.algorithmic import (
    BlockEncodingResource,
    FTQCCostModel,
    FTQCResourceQuantity,
    compare_ftqc_resource_estimates,
    estimate_qubitized_qpe_from_block_encoding,
)

# %% [markdown]
# ## Workflow
#
# qubitized walkは通常、再利用可能なsubroutineから構成されます。
#
# - PREPAREはamplitudeや係数をindex registerへloadします。
# - SELECTはindexされたunitaryやoracleを適用します。
# - reflectionはwalk operatorを完成させます。
#
# Qamomileはこれらをresource modelのfieldとして保持します。algorithmic costを比較するだけなら、circuit IRに特別なblock-encoding operationを追加する必要はありません。

# %% [markdown]
# ## 最小例
#
# 下の数値はsyntheticです。1回のqubitized walkが2回のPREPARE、1回のSELECT、1回のreflectionでcost計上されることを示します。

# %%
block = BlockEncodingResource(
    system_qubits=12,
    normalization=sp.Integer(240),
    prepare_cost_toffoli=30,
    select_cost_toffoli=120,
    reflection_cost_toffoli=8,
    ancilla_qubits=5,
    name="toy_lcu",
)

print(block.to_dict())

assert block.logical_qubits == 17
assert block.walk_cost_toffoli == 188
assert block.resource_values()[FTQCResourceQuantity.WALK_COST_TOFFOLI] == 188

# %% [markdown]
# ## Qubitized QPE
#
# QPEはqubitized walkを繰り返し呼び出します。energy precisionを$\epsilon$とすると、symbolicなcall proxyは$\alpha / \epsilon$です。ここで$\alpha$はblock-encoding normalizationです。

# %%
architecture = FTQCCostModel(
    physical_qubits_per_logical=100,
    logical_cycle_time_seconds=sp.Float("1e-6"),
    factory_qubits=2000,
    toffoli_throughput_per_second=sp.Float("5e5"),
)

estimate = estimate_qubitized_qpe_from_block_encoding(
    block,
    precision=sp.Integer(3),
    qpe_register_qubits=6,
    cost_model=architecture,
)

print("iterations:", estimate.qpe_iterations)
print("Toffoli gates:", estimate.toffoli_gates)
print("logical qubits:", estimate.logical_qubits)

assert estimate.qpe_iterations == 80
assert estimate.toffoli_gates == 15040
assert estimate.logical_qubits == 23
assert estimate.physical_qubits == 4300
assert estimate.assumptions["block_encoding"] == "toy_lcu"
assert any(reference.key == "arXiv:1610.06546" for reference in estimate.references)

# %% [markdown]
# ## 表現を比較する
#
# 新しいfactorizationやsymmetry shiftは、SELECT/PREPARE costを増やしながらnormalizationを下げることがあります。fieldを分けておくと、そのtrade-offが見えます。

# %%
compressed_block = BlockEncodingResource(
    system_qubits=12,
    normalization=sp.Integer(120),
    prepare_cost_toffoli=36,
    select_cost_toffoli=144,
    reflection_cost_toffoli=8,
    ancilla_qubits=7,
    name="compressed_toy_lcu",
)

compressed_estimate = estimate_qubitized_qpe_from_block_encoding(
    compressed_block,
    precision=sp.Integer(3),
    qpe_register_qubits=6,
    cost_model=architecture,
)

comparison = compare_ftqc_resource_estimates(
    estimate,
    compressed_estimate,
    quantities=("qpe_iterations", "toffoli_gates", "logical_qubits"),
)

for row in comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4))

assert comparison[0].ratio == sp.Rational(1, 2)
assert sp.simplify(comparison[1].ratio - sp.Rational(28, 47)) == 0
assert sp.simplify(comparison[2].ratio - sp.Rational(25, 23)) == 0

# %% [markdown]
# ## Notes
#
# :::{note}
# `BlockEncodingResource`はalgorithm designのためのsymbolic contractとして扱います。これはestimatorが消費する量を記録するものであり、特定backend向けのPREPAREやSELECT circuitがすでにsynthesize済みであるとは主張しません。
# :::

# %% [markdown]
# ## Summary
#
# このnotebookでは、次のことを学びました。
#
# - Block-encoding estimateでは、normalization、PREPARE、SELECT、reflection、ancilla、QPE readout costを分けて扱います。
# - Qubitized QPEは、block-encodingのwalk costとnormalization-over-precision iterationを合成します。
# - あるsubroutineが高価になっても、representation trade-offによって総Toffoli countが下がる場合があります。
