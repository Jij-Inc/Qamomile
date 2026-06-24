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
# # FTQCリソースレビューの設計
#
# このページでは、fault-tolerant algorithmの主張をQamomileのresource quantityへ写す方法を説明します。
# 論文レベルのworkload claimから始め、論理algorithm modelとhardware assumptionを分けたまま、canonical quantityで候補を比較します。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.resource_estimation as qre

# %% [markdown]
# ## 研究のシグナル
#
# 近年のFTQC量子化学論文は、Hamiltonian representation、QPE implementation、architecture assumptionのどれかを変えてresource estimateを改善することが多いです。
# Qamomileは、特定の論文の表をAPIへ直接埋め込みません。
# 代わりに、その表を監査するためのquantityを公開します。
#
# | 研究のシグナル | 例となる論文 | Qamomileで確認する面 |
# | --- | --- | --- |
# | qubitized QPEの前にHamiltonian normalizationを下げる。 | [Symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502)は、QPE iterationとToffoli costを動かす1-normを減らします。 | `lambda_norm`, `representation_error`, `walk_cost_toffoli`, `qpe_iterations`, `t_gates` |
# | block-encoding workloadをearly-FTQC向けのTrotter workloadへ置き換える。 | [Unitary weight concentration](https://arxiv.org/abs/2603.22778)は、single-ancillaでTrotter-basedなQPE設定におけるeffective Hamiltonian weightの削減を報告しています。 | `effective_lambda_norm`, `unitary_weight_factor`, `trotter_steps_per_sample`, `pauli_rotations`, `rotation_synthesis_t_gates` |
# | 論理resourceをarchitecture bottleneckへ変換する。 | [Gidney and Ekeråの素因数分解推定](https://arxiv.org/abs/1905.09749)のようなsurface-codeとmagic-state-factoryの推定では、runtimeがlogical depthとnon-Clifford throughputの両方に依存します。 | `physical_qubits`, `depth_limited_runtime_seconds`, `non_clifford_limited_runtime_seconds`, `physical_qubit_seconds` |
#
# この分離により、三つのcontractを明示できます。
#
# - **Problem contract**：Hamiltonianのサイズ、locality、normalization。
# - **Algorithm contract**：QPE iteration、oracleまたはproduct-formulaのwork、precision budget、non-Clifford work。
# - **Architecture contract**：code distance、cycle time、factory footprint、non-Clifford throughput。

# %% [markdown]
# ## Quantity profile
#
# `ResourceReviewProfile`は、Qamomileがreviewerに確認してほしいquantity setに名前を付けます。
# profileはscoreではありません。
# 安定したchecklistです。

# %%
profiles = {
    "qubitized workload": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.HAMILTONIAN_QPE_WORKLOAD
    ),
    "Trotter workload": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.TROTTER_QPE_WORKLOAD
    ),
    "logical outcomes": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.FTQC_LOGICAL_OUTCOMES
    ),
    "physical outcomes": qre.describe_resource_review_profile(
        qre.ResourceReviewProfile.FTQC_PHYSICAL_OUTCOMES
    ),
}

for name, profile in profiles.items():
    print(name, [quantity.value for quantity in profile.quantities])

assert qre.ResourceQuantity.LAMBDA_NORM in profiles["qubitized workload"].quantities
assert (
    qre.ResourceQuantity.EFFECTIVE_LAMBDA_NORM
    in profiles["Trotter workload"].quantities
)
assert (
    qre.ResourceQuantity.NON_CLIFFORD_COUNT in profiles["logical outcomes"].quantities
)
assert (
    qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS
    in profiles["physical outcomes"].quantities
)

# %% [markdown]
# ## 論理workloadの比較
#
# まず、論理algorithm contractを比較します。
# 次の例では、sparse Pauli-LCUのqubitized-QPE workloadと、unitary-weight-concentration風のTrotter-QPE workloadを比較します。
# 数値はページを実行しやすくするために小さくしていますが、確認しているquantityはpaper-scale estimateでも同じです。

# %%
summary = qre.PauliHamiltonianResource(
    n_qubits=4,
    n_pauli_terms=12,
    lambda_norm=12,
    max_locality=2,
)

qubitized_workload = qre.HamiltonianQPEWorkload(
    summary,
    walk_cost_toffoli=120,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    qpe_register_qubits=3,
    representation_error=sp.Rational(1, 10),
    description="sparse Pauli LCU baseline",
)
uwc_workload = qre.TrotterQPEWorkload.from_effective_lambda_norm(
    summary,
    effective_lambda_norm=2,
    trotter_steps_per_sample=2,
    samples=12,
    randomized_compilation_factor=sp.Rational(1, 2),
    rotation_synthesis_t_gates=2,
    description="unitary weight concentration toy model",
)

qubitized_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    qubitized_workload,
    precision=1,
)
uwc_logical = qre.estimate_trotter_qpe_resources_from_workload(
    uwc_workload,
    precision=1,
)

logical_rows = qre.compare_resource_values(
    qubitized_logical,
    uwc_logical,
    quantities=profiles["logical outcomes"].quantities,
)
for row in logical_rows:
    print(row.to_dict())

assert qubitized_workload.algorithmic_precision(1) == sp.Rational(9, 10)
assert uwc_workload.unitary_weight_factor == sp.Rational(1, 6)
assert uwc_logical.qubits < qubitized_logical.qubits
non_clifford_row = next(
    row
    for row in logical_rows
    if row.quantity == qre.ResourceQuantity.NON_CLIFFORD_COUNT
)
assert non_clifford_row.candidate < non_clifford_row.baseline

# %% [markdown]
# ## Architecture bottleneck
#
# 論理比較が明確になったら、architecture modelを使ってbottleneckを確認します。
# `SurfaceCodeCostModel`は意図的にcompactです。
# 明示的なassumptionを記録し、`FTQCCostModel`が使うgenericなFTQC knobを導きます。
# error budgetからcode distanceやfactory layoutを選ぶmodelではありません。

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=7,
    physical_cycle_time_seconds=sp.Rational(1, 1_000_000),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=2,
    physical_qubits_per_factory=500,
    factory_cycles_per_non_clifford=4,
)

qubitized_physical = qre.estimate_physical_resources(qubitized_logical, surface_code)
uwc_physical = qre.estimate_physical_resources(uwc_logical, surface_code)

physical_rows = qre.compare_resource_values(
    qubitized_physical,
    uwc_physical,
    quantities=profiles["physical outcomes"].quantities,
)
for row in physical_rows:
    print(row.to_dict())

uwc_values = uwc_physical.resource_values()
assert uwc_values["runtime_seconds"] == sp.Max(
    uwc_values["depth_limited_runtime_seconds"],
    uwc_values["non_clifford_limited_runtime_seconds"],
)
assert (
    uwc_values["physical_qubit_seconds"]
    < qubitized_physical.resource_values()["physical_qubit_seconds"]
)

# %% [markdown]
# ## Symbolic scenario
#
# early FTQCのレビューでは、hardware assumptionを選ぶまでarchitecture knobをsymbolicなまま残すことがあります。
# scenario rowを使うと、logical estimateを変えずにそのassumptionを見える形にできます。

# %%
distance = sp.symbols("distance", positive=True)
cycle_time = sp.symbols("cycle_time", positive=True)
symbolic_surface_code = qre.SurfaceCodeCostModel(
    code_distance=distance,
    physical_cycle_time_seconds=cycle_time,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=2,
    physical_qubits_per_factory=500,
    factory_cycles_per_non_clifford=4,
)
symbolic_physical = qre.estimate_physical_resources(
    uwc_logical,
    symbolic_surface_code,
)

driver_rows = qre.audit_resource_value_drivers(
    symbolic_physical,
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in driver_rows:
    print(row.to_dict())

scenario_rows = qre.evaluate_resource_value_scenarios(
    symbolic_physical,
    {
        "fast small distance": {"distance": 5, "cycle_time": sp.Rational(1, 1_000_000)},
        "slow large distance": {"distance": 9, "cycle_time": sp.Rational(2, 1_000_000)},
    },
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in scenario_rows:
    print(row.to_dict())

assert {row.symbol for row in driver_rows} == {"cycle_time", "distance"}
assert len(scenario_rows) == 6
assert all(row.is_resolved for row in scenario_rows)

# %% [markdown]
# ## Summary
#
# このnotebookでは、次のことを確認しました。
#
# - FTQC resource reviewでは、problem、algorithm、architectureのcontractを分けます。
# - 近年の量子化学resource paperは、Hamiltonian normalization、effective lambda、non-Clifford work、logical qubits、runtime bottleneck、physical qubit-secondsのようなcanonical quantityで監査できます。
# - Qamomileのresource-estimation APIは、report、snapshot、manifestの層を追加する前に、それらのquantityを比較できるように設計されています。
