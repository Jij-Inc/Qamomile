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
# tags: [usage, chemistry, resource-estimation]
# ---
#
# # FTQCリソースワークフロー
#
# このページでは、Qamomileのresource-estimation primitiveを使って、fault-tolerantな量子化学リソース推定を比較する方法を示します。
# 例は意図的に小さくしています。
# 論文規模の推定をレビューするときに必要なquantityとAPI境界を示すことが目的であり、特定の論文の再現ではありません。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.observable as qm_o
import qamomile.resource_estimation as qre

# %% [markdown]
# ## ワークフローの境界
#
# FTQCリソース推定は、アルゴリズムをbackend circuitへloweringする前の段階で比較することが多いです。
# Qamomileでは、この層をsymbolicなresource modelとして扱います。
#
# | 論文レベルの主張 | 確認するQamomile quantity |
# | --- | --- |
# | Hamiltonian compressionがQPEのwork signalを減らす | `lambda_norm`, `effective_lambda_norm`, `representation_error` |
# | walkまたはtime-evolution routineが安くなる | `walk_cost_toffoli`, `pauli_rotations`, `t_gates`, `non_clifford_count` |
# | workを減らすためにmemoryを使う | `logical_qubits`, `system_qubits`, `block_encoding_ancilla_qubits`, `qpe_register_qubits` |
# | architecture liftでbottleneckが変わる | `physical_qubits`, `runtime_seconds`, `physical_qubit_seconds`, `active_volume` |
#
# :::{note}
# 近年の例として、[symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502)、[unitary weight concentration](https://arxiv.org/abs/2603.22778)、[active-volume型のリソース推定](https://arxiv.org/abs/2501.06165)があります。
# 以下のコードでは、toy numberを使って同じresource quantityを確認します。
# :::

# %% [markdown]
# ## Hamiltonianの要約
#
# まずHamiltonianをresource quantityへ要約します。
# summaryには、encoded width、非identity Pauli term数、Hamiltonian normalization、最大localityが記録されます。

# %%
hamiltonian = 4 * qm_o.Z(0) + 3 * qm_o.Z(1) + 2 * qm_o.X(0) * qm_o.X(1)
summary = qre.summarize_pauli_hamiltonian(hamiltonian)

print(summary.to_dict())

assert summary.n_qubits == 2
assert summary.n_pauli_terms == 3
assert sp.Abs(summary.lambda_norm - 9) < sp.Float("1e-12")
assert summary.max_locality == 2

# %% [markdown]
# ## Qubitized QPEの候補
#
# block-encoding contractがあれば、2つのqubitized QPE候補を比較できます。
# baselineは元のHamiltonian normalizationを保ちます。
# compressed candidateは、小さいnormalizationと安いPREPARE/SELECT/reflection costを仮定します。
# その代わりに、ancilla量子ビットを1つ増やし、小さいrepresentation-error budgetを使います。

# %%
target_precision = sp.Integer(1)

baseline_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=summary.lambda_norm,
    prepare_cost_toffoli=20,
    select_cost_toffoli=70,
    reflection_cost_toffoli=10,
    ancilla_qubits=1,
    name="sparse Pauli LCU",
)
compressed_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=sp.Rational(2, 5) * summary.lambda_norm,
    prepare_cost_toffoli=15,
    select_cost_toffoli=45,
    reflection_cost_toffoli=5,
    ancilla_qubits=2,
    name="compressed factorization",
)

baseline_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    baseline_block,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    qpe_register_qubits=2,
)
compressed_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    compressed_block,
    representation=qre.HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
    second_factor_rank=4,
    qpe_register_qubits=2,
    representation_error=sp.Rational(1, 10),
)

baseline_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    baseline_workload,
    precision=target_precision,
)
compressed_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    compressed_workload,
    precision=target_precision,
)

logical_rows = qre.compare_resource_values(
    baseline_logical,
    compressed_logical,
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.LOGICAL_QUBITS,
        qre.ResourceQuantity.NON_CLIFFORD_COUNT,
    ),
)

for row in logical_rows:
    print(row.to_dict())

assert compressed_logical.qubits == baseline_logical.qubits + 1
assert (
    compressed_logical.gates.oracle_calls["qpe_iterations"]
    < (baseline_logical.gates.oracle_calls["qpe_iterations"])
)
assert compressed_logical.gates.multi_qubit < baseline_logical.gates.multi_qubit

# %% [markdown]
# ## Precision budget
#
# representation errorはtarget precisionと並べて見える必要があります。
# `resource_values_for_precision()`は、要求したbudgetとphase estimationに残るprecisionを公開します。

# %%
precision_rows = qre.compare_resource_values(
    baseline_workload.resource_values_for_precision(target_precision),
    compressed_workload.resource_values_for_precision(target_precision),
    quantities=(
        qre.ResourceQuantity.TARGET_PRECISION,
        qre.ResourceQuantity.ALGORITHMIC_PRECISION,
    ),
)
compressed_precision_values = compressed_workload.resource_values_for_precision(
    target_precision
)

for row in precision_rows:
    print(row.to_dict())

assert precision_rows[0].ratio == 1
assert sp.Abs(precision_rows[1].candidate - sp.Rational(9, 10)) < sp.Float("1e-12")
assert compressed_precision_values["representation_error"] == sp.Rational(1, 10)

# %% [markdown]
# ## Unitary-weight型のTrotter QPE
#
# 推定によっては、block-encoding normalizationではなく、concentration後のeffective Hamiltonian weightを報告します。
# `TrotterQPEWorkload.from_effective_lambda_norm()`は元のHamiltonian summaryを保ち、そこからmultiplicative weight factorを導きます。

# %%
uwc_workload = qre.TrotterQPEWorkload.from_effective_lambda_norm(
    summary,
    effective_lambda_norm=1,
    trotter_steps_per_sample=2,
    samples=10,
    randomized_compilation_factor=sp.Rational(1, 2),
    rotation_synthesis_t_gates=2,
    description="unitary-weight-style toy estimate",
)
uwc_logical = qre.estimate_trotter_qpe_resources_from_workload(
    uwc_workload,
    precision=target_precision,
)

uwc_rows = qre.compare_resource_values(
    baseline_logical,
    uwc_logical,
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.LOGICAL_QUBITS,
        qre.ResourceQuantity.NON_CLIFFORD_COUNT,
    ),
)

for row in uwc_rows:
    print(row.to_dict())

assert sp.Abs(uwc_workload.unitary_weight_factor - sp.Rational(1, 9)) < sp.Float(
    "1e-12"
)
assert uwc_logical.qubits == summary.n_qubits + 1
assert sp.Abs(uwc_logical.gates.oracle_calls["qpe_iterations"] - 1) < sp.Float("1e-12")
assert uwc_logical.gates.t_gates < baseline_logical.gates.multi_qubit

# %% [markdown]
# ## Architecture lift
#
# 論理推定は、明示的なarchitecture modelを通してliftできます。
# surface-code型のmodelは、物理量子ビット数、runtime component、physical qubit-secondsを報告します。
# active-volume modelは、active resourceでコストを読むアルゴリズムのoperation-volume accountingを別に示します。

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=sp.Float("1e-6"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=1,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)
compressed_physical = qre.estimate_physical_resources(
    compressed_logical,
    surface_code,
)

active_volume_model = qre.ActiveVolumeCostModel(
    active_volume_per_logical_gate=2,
    active_volume_per_non_clifford=1,
    active_volume_throughput_per_second=100,
)
uwc_active_volume = qre.estimate_active_volume_resources(
    uwc_logical,
    active_volume_model,
)

physical_values = compressed_physical.resource_values()
active_values = uwc_active_volume.resource_values()

print(
    {
        "physical_qubits": physical_values["physical_qubits"],
        "runtime_seconds": physical_values["runtime_seconds"],
        "physical_qubit_seconds": physical_values["physical_qubit_seconds"],
        "active_volume": active_values["active_volume"],
        "active_volume_runtime_seconds": active_values["active_volume_runtime_seconds"],
    }
)

assert physical_values["runtime_seconds"] == sp.Max(
    physical_values["depth_limited_runtime_seconds"],
    physical_values["non_clifford_limited_runtime_seconds"],
)
assert sp.Abs(active_values["active_volume"] - 180) < sp.Float("1e-12")
assert sp.Abs(
    active_values["active_volume_runtime_seconds"] - sp.Rational(9, 5)
) < sp.Float("1e-12")

# %% [markdown]
# ## まとめ
#
# このnotebookでは、次のことを確認しました。
#
# - `PauliHamiltonianResource`とworkload objectを使うと、FTQC推定をsymbolicかつarchitecture-independentな形に保てます。
# - 候補は`lambda_norm`、`qpe_iterations`、`logical_qubits`、`non_clifford_count`、`representation_error`などのcanonical quantityで比較できます。
# - algorithm-levelの比較を終えてから、同じarchitecture modelで物理リソースまたはactive-volume proxyへliftします。
