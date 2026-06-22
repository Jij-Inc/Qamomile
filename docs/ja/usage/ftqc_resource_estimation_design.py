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
# # FTQCリソース推定の設計
#
# このnotebookでは、Qamomileがfault-tolerant quantum chemistryのために追跡するリソース推定量を説明します。backend固有の回路へloweringする前に、問題メタデータ、アルゴリズム上の作業量、logicalリソース、architecture仮定をどう分けるかに焦点を当てます。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCResourceCategory,
    FTQCResourceQuantity,
    SurfaceCodeCostModel,
    compare_ftqc_resource_estimates,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    iter_ftqc_resource_quantity_specs,
    summarize_pauli_hamiltonian,
)

# %% [markdown]
# ## 設計上の境界
#
# FTQC化学計算の論文では、compiler IRを変えるのではなく、Hamiltonian表現を変えることでcostを下げることがよくあります。そのためQamomileでは、この層をアルゴリズム上のメタデータとして扱います。
#
# - Hamiltonian summaryは`lambda_norm`やPauli項数など、表現レベルの量を保持します。
# - アルゴリズム推定器はそれらの量をQPE反復回数、ToffoliまたはT count、logical量子ビット、logical depth、physical量子ビット、runtime proxyへ変換します。
# - 具体的な回路loweringはbackend emitterが担当します。
#
# これはQamomileの他の場所で使っているcompiler上の規則と一致します。IRは抽象的に保ち、具体化はできるだけ遅くまで押し下げます。

# %% [markdown]
# ## 研究上のシグナル
#
# 現在のquantity catalogは、近年のFTQC化学計算研究で使われているcost driverに合わせています。
#
# | Research direction | Cost signal for Qamomile |
# | --- | --- |
# | Symmetry-compressed double factorizationはqubitized chemistry QPEのHamiltonian 1-normとToffoli countを削減します([arXiv:2403.03502](https://arxiv.org/abs/2403.03502))。 | `lambda_norm`、QPE反復回数、walkあたりのToffoli cost、総Toffoli countを別々に追跡します。 |
# | Simultaneous symmetry shiftsとtensor factorizationsは、electronic Hamiltonianのblock-encoding scaling constantを削減します([arXiv:2412.01338](https://arxiv.org/abs/2412.01338))。 | Hamiltonian normalizationを、emit済み回路の性質ではなく表現メタデータとして扱います。 |
# | Unitary weight concentrationを使うearly-FTQC single-ancilla QPEは、より小さいphysical量子ビットbudgetと限られたdepthを目標にします([arXiv:2603.22778](https://arxiv.org/abs/2603.22778))。 | Toffoli-nativeなqubitization costに加えて、T gates、logical depth、physical量子ビット、runtime、architecture knobを追跡します。 |
#
# これらはmodeling上の量です。特定の分子についてリソースを主張する前に、それぞれの論文の仮定に照らして検証する必要があります。

# %% [markdown]
# ## Quantity Catalog
#
# Qamomileはcanonicalなquantity keyを公開しているため、reportやtutorialごとにad hocな列名を作る必要がありません。

# %%
catalog = [
    spec.to_dict()
    for spec in iter_ftqc_resource_quantity_specs()
    if spec.category
    in {
        FTQCResourceCategory.PROBLEM,
        FTQCResourceCategory.ALGORITHM,
        FTQCResourceCategory.LOGICAL,
        FTQCResourceCategory.PHYSICAL,
        FTQCResourceCategory.ARCHITECTURE,
    }
]

for row in catalog:
    print(row["quantity"], row["unit"], row["category"])

assert FTQCResourceQuantity.LAMBDA_NORM.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.TOFFOLI_GATES.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.RUNTIME_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.CODE_DISTANCE.value in {
    row["quantity"] for row in catalog
}

# %% [markdown]
# ## 最小例
#
# まずQamomile observableから始め、それを一度summaryにしてから、表現レベルのmodelを2つ構築します。下の例ではsyntheticなscaling値を使うため、特定の分子についての主張ではなくworkflowを示します。

# %%
toy_hamiltonian = 0.5 * qm_o.Z(0) + 0.25 * qm_o.X(1) * qm_o.X(2)
summary = summarize_pauli_hamiltonian(
    toy_hamiltonian,
    n_spin_orbitals=40,
    source="toy_pauli_lcu",
)
scaled_summary = summary.with_lambda_scale(
    sp.Float("2.0e5") / summary.lambda_norm,
    source="scaled_toy_pauli_lcu",
)

architecture = SurfaceCodeCostModel(
    code_distance=20,
    physical_cycle_time_seconds=sp.Float("5e-8"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=1,
    factory_count=4,
    physical_qubits_per_factory=5000,
    factory_cycles_per_toffoli=2,
)
cost_model = architecture.to_cost_model()

assert architecture.physical_qubits_per_logical == 800
assert architecture.factory_qubits == 20000
assert (
    sp.Abs(architecture.toffoli_throughput_per_second - sp.Float("2e6"))
    < sp.Float("1e-9")
)

baseline_model = ChemistryQPEModel(
    hamiltonian=scaled_summary,
    method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
    walk_cost_toffoli=sp.Integer(4_000),
)
compressed_model = ChemistryQPEModel(
    hamiltonian=scaled_summary.with_lambda_scale(
        sp.Float("0.5"),
        source="compressed_scaled_toy_pauli_lcu",
    ),
    method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
    walk_cost_toffoli=sp.Integer(4_400),
    second_factor_rank=9,
)

baseline = estimate_qubitized_chemistry_qpe_from_model(
    baseline_model,
    precision=sp.Float("0.0016"),
    cost_model=cost_model,
)
compressed = estimate_qubitized_chemistry_qpe_from_model(
    compressed_model,
    precision=sp.Float("0.0016"),
    cost_model=cost_model,
)

comparison = compare_ftqc_resource_estimates(
    baseline,
    compressed,
    quantities=("qpe_iterations", "toffoli_gates", "physical_qubits"),
)

for row in comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert comparison[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison[0].ratio == sp.Float("0.5")
assert comparison[1].ratio == sp.Float("0.55")

# %% [markdown]
# ## Early-FTQCのパターン
#
# Early-FTQC推定はToffoli-nativeとは限りません。同じ比較APIを、主にT gatesとlogical depthを報告するTrotter型modelにも使えます。

# %%
plain_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=sp.Float("0.0016"),
    trotter_steps_per_sample=8,
    samples=128,
    cost_model=cost_model,
)
uwc_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=sp.Float("0.0016"),
    trotter_steps_per_sample=8,
    samples=128,
    unitary_weight_factor=sp.Float("0.1"),
    randomized_compilation_factor=sp.Float("0.5"),
    rotation_synthesis_t_gates=3,
    cost_model=cost_model,
)

trotter_comparison = compare_ftqc_resource_estimates(
    plain_trotter,
    uwc_trotter,
    quantities=("qpe_iterations", "logical_depth", "t_gates"),
)

for row in trotter_comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert trotter_comparison[0].ratio == sp.Float("0.1")
assert trotter_comparison[1].ratio == sp.Float("0.05")

# %% [markdown]
# ## Notes
#
# :::{note}
# 上の数値はsyntheticです。Qamomileがcost driverを分けて比較する流れを示しています。publication-qualityの分子研究には、分子固有のintegral、factorization rank、truncation error、synthesis仮定、architecture calibrationが必要です。
# :::
#
# 新しいFTQC推定器を追加するときは、次の境界を意識してください。
#
# - 新しい問題メタデータはIR operationではなく、structured summaryとして追加します。
# - 新しい測定量を公開するときは、report列として出す前にcanonical catalogへ追加します。
# - 手書きのphysical resource knobへ直接落とす前に、`SurfaceCodeCostModel`のようなstructured architecture modelを使います。
# - architecture仮定を明示して、algorithm metadataを変えずにphysical量子ビットとruntime推定を差し替えられるようにします。

# %% [markdown]
# ## Summary
#
# このnotebookでは、次のことを学びました。
#
# - 近年のFTQC化学計算研究から、Hamiltonian normalization、QPE反復回数、non-Clifford count、logical depth、physical量子ビット、runtimeを分けて追跡する必要があることがわかります。
# - Qamomileはこれらの量をアルゴリズム上のメタデータとして保持するため、circuit IRはbackend-neutralに保たれます。
# - Surface-code仮定は別にmodel化し、chemistry推定器が使うcost modelへ変換できます。
# - `compare_ftqc_resource_estimates`を使うと、特定のchemistry factorizationをhard-codeせず、symbolicな推定をreviewしやすいsavings tableへ変換できます。
