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
# tags: [algorithm, chemistry, resource-estimation, simulation]
# ---
#
# # FTQC化学計算のリソース推定
#
# このノートブックでは、Qamomileでfault-tolerantな量子化学計算のリソースモデルを比較する方法を示します。完全な論理回路がまだない段階で重要になる、Hamiltonian normalization、phase-estimation iterations、ToffoliまたはT counts、論理量子ビット、物理量子ビット、logical space-time volume、physical qubit-seconds、runtime proxiesに注目します。

# %%
# 最新のQamomileをpipでインストールします！
# # !pip install qamomile

# %%
import sympy as sp
from openfermion import QubitOperator

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCAccuracyBudget,
    FTQCResourceProfile,
    FTQCResourceQuantity,
    HamiltonianResourceReduction,
    QPEStatePreparationBudget,
    SurfaceCodeDistanceBudget,
    block_encoding_from_chemistry_model,
    compare_ftqc_resource_estimates,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_qubitized_qpe_from_block_encoding,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    ftqc_resource_profile_quantities,
    iter_ftqc_research_signals,
    iter_ftqc_resource_profile_specs,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
    summarize_openfermion_qubit_operator,
    summarize_pauli_hamiltonian,
)

# %% [markdown]
# ## Background
#
# Fault-tolerantな化学計算アルゴリズムは、NISQのvariationalな例とは異なるリソース量で比較されることが多くあります。qubitized QPEでは、Hamiltonian block-encoding normalizationがwalk callsの回数を決め、per-walk implementationがToffoli countを決めます。近年の化学計算向け提案では、backend-levelのgate decompositionだけを変えるのではなく、Hamiltonian representationを変えることでこれらのコストを下げようとします。
#
# 例として、symmetry-compressed double factorization([arXiv:2403.03502](https://arxiv.org/abs/2403.03502))、simultaneous symmetry shifts and tensor factorizations([arXiv:2412.01338](https://arxiv.org/abs/2412.01338))、unitary weight concentrationを使うearly-FTQC single-ancilla QPE([arXiv:2603.22778](https://arxiv.org/abs/2603.22778))があります。symmetry-adapted filtering([arXiv:2601.08533](https://arxiv.org/abs/2601.08533))のようなstate-preparation改善も、期待されるQPE試行回数を変えます。これらの提案はalgorithmic work、logical depth、論理量子ビット、hardware runtimeをそれぞれ異なる形で交換するため、logical qubit-layersやphysical qubit-secondsのようなspace-time量がレビュー対象として役立ちます。以下のEstimatorは意図的にsymbolicです。具体的なHamiltonian-loading circuitに進む前に、提案したrepresentationがコストを支配する量をどう変えるかを確認できます。

# %% [markdown]
# ## Problem Settings
#
# 同じactive-space scaleについて、次の3つのシナリオを比較します。
#
# 1. tensor-hypercontraction-likeなqubitized QPE baseline、
# 2. symmetry-compressed double-factorization-styleなqubitized QPE estimate、
# 3. unitary-weight concentration factorを持つearly-FTQC single-ancilla Trotter QPE estimate。
#
# 数値は小さく、人工的に選んでいます。このノートブックを高速でレビューしやすくするためです。特定の分子についての主張ではなく、workflowのデモとして読んでください。

# %%
n_spin_orbitals = 40
target_precision = sp.Float("0.0016")  # Hartree単位のおおよそのchemical accuracy
accuracy_budget = FTQCAccuracyBudget(
    target_precision=target_precision,
    truncation_error=sp.Float("1e-4"),
)
qpe_precision = accuracy_budget.qpe_precision

distance_budget = SurfaceCodeDistanceBudget(
    physical_error_rate=sp.Float("1e-3"),
    threshold_error_rate=sp.Float("1e-2"),
    target_logical_failure_probability=sp.Float("1e-9"),
    logical_operation_budget=1000,
)
architecture = distance_budget.to_surface_code_cost_model(
    physical_cycle_time_seconds=sp.Float("5e-8"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=1,
    factory_count=4,
    physical_qubits_per_factory=5000,
    factory_cycles_per_toffoli=2,
)
cost_model = architecture

assert distance_budget.code_distance == 21
assert architecture.physical_qubits_per_logical == 882
assert architecture.factory_qubits == 20000
assert sp.Abs(qpe_precision - sp.Float("0.0015")) < sp.Float("1e-12")

# %% [markdown]
# ## Resource Quantities
#
# symbolicなFTQC estimatesでは、problem quantities、logical work、physical
# assumptionsを分けて扱うべきです。Qamomileはcanonical quantity catalogを公開
# しているので、downstream reportsは安定したkeyを使いつつ、読者向けのlabel、
# unit、modeling layerも表示できます。

# %%
quantity_catalog = [
    spec.to_dict()
    for spec in iter_ftqc_resource_quantity_specs()
    if spec.quantity.value
    in {
        "lambda_norm",
        "target_precision",
        "state_preparation_success_probability",
        "qpe_repetitions",
        "qpe_iterations",
        "toffoli_gates",
        "t_gates",
        "logical_qubits",
        "logical_spacetime_volume",
        "physical_qubits",
        "physical_qubit_seconds",
        "runtime_seconds",
        "logical_error_rate",
        "code_distance",
    }
]

for row in quantity_catalog:
    print(row["quantity"], row["unit"], row["category"])

assert {row["quantity"] for row in quantity_catalog} == {
    "lambda_norm",
    "target_precision",
    "state_preparation_success_probability",
    "qpe_repetitions",
    "qpe_iterations",
    "toffoli_gates",
    "t_gates",
    "logical_qubits",
    "logical_spacetime_volume",
    "physical_qubits",
    "physical_qubit_seconds",
    "runtime_seconds",
    "logical_error_rate",
    "code_distance",
}

# %% [markdown]
# Qamomileは標準review profileも提供します。これはよく使うaudit上の問いに対応する、小さなquantity bundleです。profileは新しいresourceを計算するものではなく、比較すべき列に名前を付け、comparison helperへ直接渡せます。

# %%
profile_catalog = {
    spec.profile: spec.to_dict() for spec in iter_ftqc_resource_profile_specs()
}
space_time_quantities = ftqc_resource_profile_quantities(
    FTQCResourceProfile.SPACETIME
)

print(profile_catalog[FTQCResourceProfile.SPACETIME]["quantities"])

assert space_time_quantities == (
    FTQCResourceQuantity.LOGICAL_QUBITS,
    FTQCResourceQuantity.LOGICAL_DEPTH,
    FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
    FTQCResourceQuantity.PHYSICAL_QUBITS,
    FTQCResourceQuantity.RUNTIME_SECONDS,
    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
)

# %% [markdown]
# research-signal catalogは、近年の論文と、Qamomile modelが公開すべき量、さらに最初に確認すべきreview profileを対応づけます。これにより、このチュートリアルは文章だけの主張ではなく、小さく確認可能なcontractに基づきます。

# %%
signal_by_key = {
    signal.reference_key: signal for signal in iter_ftqc_research_signals()
}
scdf_signal = signal_by_key["arXiv:2403.03502"]
early_ftqc_signal = signal_by_key["arXiv:2603.22778"]

print(scdf_signal.title)
print([quantity.value for quantity in scdf_signal.quantities])
print([profile.value for profile in scdf_signal.profiles])
print(early_ftqc_signal.title)
print([quantity.value for quantity in early_ftqc_signal.quantities])
print([profile.value for profile in early_ftqc_signal.profiles])

assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in scdf_signal.quantities
assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME in early_ftqc_signal.quantities
assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in early_ftqc_signal.quantities
assert FTQCResourceProfile.SPACETIME in early_ftqc_signal.profiles

# %% [markdown]
# 小さなQamomile observableから始めます。実際の化学計算pipelineと同じ形にするためです。Hamiltonianを作る、または読み込み、LCU量を要約して、そのsummaryをFTQC Estimatorへ渡します。下のrescalingは、toy Hamiltonianを大きなactive-space modelの代わりとして使うためのものです。この係数が特定の分子を表すという主張ではありません。

# %%
toy_hamiltonian = 0.5 * qm_o.Z(0) + 0.25 * qm_o.X(1) * qm_o.X(2) + 0.125
toy_summary = summarize_pauli_hamiltonian(
    toy_hamiltonian,
    n_spin_orbitals=n_spin_orbitals,
    source="toy_pauli_lcu",
)
scaled_summary = toy_summary.with_lambda_scale(
    sp.Float("2.0e5") / toy_summary.lambda_norm,
    source="scaled_toy_pauli_lcu",
)

assert toy_summary.n_pauli_terms == 2
assert toy_summary.constant == sp.Float("0.125")
assert sp.simplify(scaled_summary.lambda_norm - sp.Float("2.0e5")) == 0

# %% [markdown]
# chemistry preprocessingがすでにOpenFermionの`QubitOperator`を生成する場合も、
# 同じ境界でsummaryを作れます。これにより、electronic-structure toolchainは
# Qamomileのcompiler IRの外側に保ちながら、コストを支配するHamiltonian
# metadataを保持できます。

# %%
openfermion_hamiltonian = (
    QubitOperator("Z0", 0.5)
    + QubitOperator("X1 X2", 0.25)
    + QubitOperator((), 0.125)
)
openfermion_summary = summarize_openfermion_qubit_operator(
    openfermion_hamiltonian,
    n_spin_orbitals=n_spin_orbitals,
    source="openfermion_toy_pauli_lcu",
)

assert openfermion_summary.n_pauli_terms == toy_summary.n_pauli_terms
assert openfermion_summary.lambda_norm == toy_summary.lambda_norm
assert openfermion_summary.constant == toy_summary.constant

# %% [markdown]
# ## Qubitized QPE Comparison
#
# Qamomileはchemistry factorization costをIRの外側に保ちます。model-driven Estimatorは、Hamiltonian summaryとrepresentation-dependentなone-walk Toffoli costを入力として受け取ります。
#
# ```text
# qpe_iterations = lambda_norm / qpe_precision
# toffoli_gates = qpe_iterations * walk_cost_toffoli
# ```
#
# これによりモデルを明確に保てます。Hamiltonian normalizationを変えること、walk circuitを変えること、physical architectureを変えることは、それぞれ別の設計選択です。

# %%
thc_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary,
        method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
        walk_cost_toffoli=sp.Integer(4_000),
        description="THC-style scaled toy model",
    )
)
scdf_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary.with_lambda_scale(
            sp.Float("0.5"),
            source="SCDF-style scaled toy model",
        ),
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        walk_cost_toffoli=sp.Integer(4_400),
        second_factor_rank=9,
        description="SCDF-style scaled toy model",
    )
)

thc = estimate_qubitized_chemistry_qpe_from_model(
    thc_model,
    precision=qpe_precision,
    cost_model=cost_model,
)
scdf = estimate_qubitized_chemistry_qpe_from_model(
    scdf_model,
    precision=qpe_precision,
    cost_model=cost_model,
)

assert scdf.qpe_iterations < thc.qpe_iterations
assert scdf.toffoli_gates < thc.toffoli_gates
assert scdf.resource_values()[FTQCResourceQuantity.CODE_DISTANCE] == 21
assert scdf.to_dict()["architecture_values"]["code_distance"] == "21"

print("THC Toffoli gates:", sp.N(thc.toffoli_gates, 4))
print("SCDF-style Toffoli gates:", sp.N(scdf.toffoli_gates, 4))
print("Toy Pauli terms:", toy_summary.n_pauli_terms)
print("SCDF-style logical qubits:", scdf.logical_qubits)

qubitized_quantities = (
    FTQCResourceQuantity.QPE_ITERATIONS,
    FTQCResourceQuantity.TOFFOLI_GATES,
    *space_time_quantities,
)

for row in scdf.to_quantity_table():
    if row["quantity"] in {quantity.value for quantity in qubitized_quantities}:
        print(row["label"], row["value"], row["unit"])

qubitized_savings = compare_ftqc_resource_estimates(
    thc,
    scdf,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)
for row in qubitized_savings:
    print(
        row.label,
        "ratio:",
        sp.N(row.ratio, 4),
        "reduction:",
        sp.N(row.reduction, 4),
    )

assert qubitized_savings[0].quantity.value == "qpe_iterations"
assert qubitized_savings[0].ratio == sp.Float("0.5")
assert sp.Abs(qubitized_savings[1].ratio - sp.Float("0.55")) < sp.Float("1e-12")
assert qubitized_savings[4].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME

qubitized_summary = summarize_ftqc_resource_comparison(
    thc,
    scdf,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

assert qubitized_summary.smaller[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert qubitized_summary.larger[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
assert any(
    row.quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
    for row in qubitized_summary.larger
)

# %% [markdown]
# ## State-preparation success budget
#
# QPE costは、prepared stateがtarget eigenstateと十分なoverlapを持つかにも依存します。symmetry filterやより良いtrial-state preparationにより、1回あたりの小さなoverheadを追加しながらsuccess probabilityを高められる場合があります。`QPEStatePreparationBudget`はこの仮定を明示し、期待されるQPEの繰り返しworkをscaleします。

# %%
weak_overlap_budget = QPEStatePreparationBudget(
    success_probability=sp.Rational(1, 8),
    description="unfiltered trial state",
)
symmetry_filtered_budget = QPEStatePreparationBudget(
    success_probability=sp.Rational(1, 2),
    state_preparation_t_gates=sp.Integer(1_000_000),
    state_preparation_logical_depth=sp.Integer(1_000_000),
    description="symmetry-filtered trial state",
)

weak_overlap_scdf = weak_overlap_budget.apply(scdf)
symmetry_filtered_scdf = symmetry_filtered_budget.apply(scdf)

preparation_savings = compare_ftqc_resource_estimates(
    weak_overlap_scdf,
    symmetry_filtered_scdf,
    quantities=(
        FTQCResourceQuantity.QPE_REPETITIONS,
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

for row in preparation_savings:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert weak_overlap_scdf.resource_values()[FTQCResourceQuantity.QPE_REPETITIONS] == 8
assert (
    symmetry_filtered_scdf.resource_values()[FTQCResourceQuantity.QPE_REPETITIONS] == 2
)
assert symmetry_filtered_scdf.resource_values()[
    FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
] == (symmetry_filtered_scdf.logical_qubits * symmetry_filtered_scdf.logical_depth)
assert symmetry_filtered_scdf.qpe_iterations == scdf.qpe_iterations * 2
assert symmetry_filtered_scdf.toffoli_gates < weak_overlap_scdf.toffoli_gates
assert "state_preparation_success_probability" in symmetry_filtered_scdf.to_dict()[
    "algorithm_values"
]

# %% [markdown]
# 同じchemistry modelはblock-encoding contractにも変換できます。将来のloader実装では、chemistry summaryやcompiler IRを変えずに、PREPARE、SELECT、reflection、workspaceのcostを分けてreviewできます。

# %%
scdf_block = block_encoding_from_chemistry_model(
    scdf_model,
    prepare_cost_toffoli=sp.Integer(200),
    reflection_cost_toffoli=sp.Integer(50),
    name="scdf_block_contract",
)
scdf_block_estimate = estimate_qubitized_qpe_from_block_encoding(
    scdf_block,
    precision=qpe_precision,
    qpe_register_qubits=12,
    cost_model=cost_model,
)

print(scdf_block.to_dict())
assert scdf_block.walk_cost_toffoli == scdf_model.walk_cost_toffoli + 450
assert scdf_block_estimate.target_precision == qpe_precision
assert scdf_block_estimate.logical_qubits == scdf_block.logical_qubits + 12

# %% [markdown]
# ## Early-FTQC Trotter QPE
#
# Early-FTQCの提案では、量子ビット数とdepthが強く制限される場合、qubitized walksよりも浅いsingle-ancilla QPEとPauli rotationsが好まれることがあります。下のresource reductionは、unitary weight concentrationのようなspectrally invariantなHamiltonian transformationを表します。コストを支配するeffective weightを下げつつ、circuit loweringとは分けて保持します。

# %%
uwc_reduction = HamiltonianResourceReduction(
    lambda_norm_factor=sp.Float("0.1"),
    description="unitary weight concentration",
)
plain_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=qpe_precision,
    trotter_steps_per_sample=8,
    samples=128,
    cost_model=cost_model,
)

uwc_trotter = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    scaled_summary,
    precision=qpe_precision,
    trotter_steps_per_sample=8,
    samples=128,
    randomized_compilation_factor=sp.Float("0.5"),
    rotation_synthesis_t_gates=3,
    resource_reduction=uwc_reduction,
    cost_model=cost_model,
)

assert uwc_trotter.qpe_iterations < plain_trotter.qpe_iterations
assert uwc_trotter.logical_depth < plain_trotter.logical_depth

print("Plain Trotter QPE depth proxy:", sp.N(plain_trotter.logical_depth, 4))
print("UWC-style Trotter QPE depth proxy:", sp.N(uwc_trotter.logical_depth, 4))
print("UWC-style T gates:", sp.N(uwc_trotter.t_gates, 4))

trotter_savings = compare_ftqc_resource_estimates(
    plain_trotter,
    uwc_trotter,
    quantities=(FTQCResourceQuantity.QPE_ITERATIONS,),
    profile=FTQCResourceProfile.SPACETIME,
)
for row in trotter_savings:
    print(
        row.label,
        "ratio:",
        sp.N(row.ratio, 4),
        "reduction:",
        sp.N(row.reduction, 4),
    )

assert trotter_savings[0].ratio == sp.Float("0.1")
assert trotter_savings[2].ratio == sp.Float("0.05")
assert trotter_savings[3].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
assert "lambda_norm=0.100000000000000" in uwc_trotter.assumptions[
    "resource_reduction_factors"
]

# %% [markdown]
# ## Result
#
# 推定結果を小さな表にまとめます。重要なのは、各列が別々の設計上の意味を持つことです。Hamiltonian representationの変更は`qpe_iterations`とper-step costに効くべきで、hardware modelの変更は`physical_qubits`、runtime、physical qubit-secondsに効くべきです。

# %%
rows = [
    ("THC qubitized QPE", thc),
    ("SCDF-style qubitized QPE", scdf),
    ("Plain Trotter QPE", plain_trotter),
    ("UWC-style Trotter QPE", uwc_trotter),
]

for name, estimate in rows:
    print(
        name,
        {
            "logical_qubits": sp.N(estimate.logical_qubits, 4),
            "physical_qubits": sp.N(estimate.physical_qubits, 4),
            "qpe_iterations": sp.N(estimate.qpe_iterations, 4),
            "toffoli_gates": sp.N(estimate.toffoli_gates, 4),
            "t_gates": sp.N(estimate.t_gates, 4),
            "logical_spacetime_volume": sp.N(
                estimate.resource_values()[FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME],
                4,
            ),
            "runtime_seconds": sp.N(estimate.runtime_seconds, 4),
            "physical_qubit_seconds": sp.N(
                estimate.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS],
                4,
            ),
        },
    )

assert scdf.physical_qubits > thc.physical_qubits
assert uwc_trotter.physical_qubits == plain_trotter.physical_qubits
assert uwc_trotter.resource_values()[
    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS
] < plain_trotter.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS]

# %% [markdown]
# ## Summary
#
# このノートブックでは、次のことを確認しました。
#
# - FTQC化学計算のリソース量を、Hamiltonian normalization、QPE iterations、non-Clifford counts、論理量子ビット、logical space-time volume、物理量子ビット、runtime proxies、physical qubit-secondsに分けて扱いました。
# - comparableなestimateを作る前に、total target precisionをtruncation errorとQPE precisionへ割り当てました。
# - state-preparation success probabilityを、QPE resourceの明示的な期待repetition factorとしてモデル化しました。
# - Qamomile IRをbackend-specificなchemistry loading circuitsへloweringせずに、qubitized QPE representationsを比較しました。
# - logical failure budgetからsurface-code distanceを選び、logical resourceをphysical resourceへliftしました。
# - code distanceなどのarchitecture quantityを各resource estimateに残し、後続のreportでauditできるようにしました。
# - chemistry QPE modelをblock-encoding contractへ変換し、PREPARE、SELECT、reflection、workspace costを分けてreviewできる形にしました。
# - unitary-weight concentration factorを、early-FTQC single-ancilla Trotter QPEのcost-driver reductionとしてモデル化する方法を示しました。
# - 近年のFTQC化学計算のresearch signalを、このチュートリアルで比較するcanonical quantitiesへ結びつけました。
# - 標準の`FTQCResourceProfile`を使い、space-time比較で同じquantity setを使うようにしました。
