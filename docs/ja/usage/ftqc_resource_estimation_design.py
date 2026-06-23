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
    BlockEncodingResource,
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCAccuracyBudget,
    FTQCResourceAggregationRule,
    FTQCResourceCategory,
    FTQCResourceConstraint,
    FTQCResourcePlan,
    FTQCResourcePlanStep,
    FTQCResourceProfile,
    FTQCResourceQuantity,
    SurfaceCodeCostModel,
    SurfaceCodeDistanceBudget,
    build_ftqc_resource_comparison_report,
    compare_ftqc_resource_estimates,
    default_ftqc_resource_aggregation_rule,
    evaluate_ftqc_resource_constraints,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    ftqc_resource_profile_quantities,
    iter_ftqc_research_signals,
    iter_ftqc_resource_profile_specs,
    iter_ftqc_resource_quantity_specs,
    plan_qubitized_qpe_from_block_encoding,
    summarize_ftqc_resource_comparison,
    summarize_pauli_hamiltonian,
)

# %% [markdown]
# ## 設計上の境界
#
# FTQC化学計算の論文では、compiler IRを変えるのではなく、Hamiltonian表現を変えることでcostを下げることがよくあります。そのためQamomileでは、この層をアルゴリズム上のメタデータとして扱います。
#
# - Hamiltonian summaryは`lambda_norm`やPauli項数など、表現レベルの量を保持します。
# - target precisionやtruncation errorのようなaccuracy budgetもfirst-classなquantityとして追跡します。これにより、2つのcost estimateを比較してよい前提が見えます。
# - アルゴリズム推定器はそれらの量をQPE反復回数、ToffoliまたはT count、logical量子ビット、logical depth、physical量子ビット、runtime proxyへ変換します。
# - 具体的な回路loweringはbackend emitterが担当します。
#
# これはQamomileの他の場所で使っているcompiler上の規則と一致します。IRは抽象的に保ち、具体化はできるだけ遅くまで押し下げます。

# %% [markdown]
# ## 研究上のシグナル
#
# 現在のquantity catalogは、近年のFTQC化学計算研究で使われているcost driverに合わせています。Qamomileはこの対応をstructured research signalとして公開しているため、reportで「なぜそのquantityを測るのか」を示せます。
#
# | Research direction | Cost signal for Qamomile |
# | --- | --- |
# | Symmetry-compressed double factorizationはqubitized chemistry QPEのHamiltonian 1-normとToffoli countを削減します([arXiv:2403.03502](https://arxiv.org/abs/2403.03502))。 | `lambda_norm`、QPE反復回数、walkあたりのToffoli cost、総Toffoli countを別々に追跡します。 |
# | Simultaneous symmetry shiftsとtensor factorizationsは、electronic Hamiltonianのblock-encoding scaling constantを削減します([arXiv:2412.01338](https://arxiv.org/abs/2412.01338))。 | Hamiltonian normalizationを、emit済み回路の性質ではなく表現メタデータとして扱います。 |
# | Symmetry-adapted filteringはQPE前のstate-preparation success probabilityを高める場合があります([arXiv:2601.08533](https://arxiv.org/abs/2601.08533))。 | success probability、期待QPE repetition、filtering overhead、T gates、runtimeを追跡します。 |
# | Unitary weight concentrationを使うearly-FTQC single-ancilla QPEは、より小さいphysical量子ビットbudgetと限られたdepthを目標にします([arXiv:2603.22778](https://arxiv.org/abs/2603.22778))。 | Toffoli-nativeなqubitization costに加えて、T gates、logical depth、logical space-time volume、physical量子ビット、runtime、physical qubit-seconds、architecture knobを追跡します。 |
#
# これらはmodeling上の量です。特定の分子についてリソースを主張する前に、それぞれの論文の仮定に照らして検証する必要があります。

# %%
research_signals = [signal.to_dict() for signal in iter_ftqc_research_signals()]
for signal in research_signals:
    print(signal["reference_key"], "->", ", ".join(signal["quantities"][:4]))

signal_by_key = {signal["reference_key"]: signal for signal in research_signals}
assert "lambda_norm" in signal_by_key["arXiv:2403.03502"]["quantities"]
assert "state_preparation_success_probability" in signal_by_key["arXiv:2601.08533"][
    "quantities"
]
assert "t_gates" in signal_by_key["arXiv:2603.22778"]["quantities"]

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

assert FTQCResourceQuantity.LAMBDA_NORM.value in {row["quantity"] for row in catalog}
assert FTQCResourceQuantity.TARGET_PRECISION.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.TRUNCATION_ERROR.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.TOFFOLI_GATES.value in {row["quantity"] for row in catalog}
assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.RUNTIME_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS.value in {
    row["quantity"] for row in catalog
}
assert FTQCResourceQuantity.CODE_DISTANCE.value in {row["quantity"] for row in catalog}

# %% [markdown]
# 標準review profileは、「space-time footprintは何か」のように繰り返し使う問いに対するquantity bundleです。reportごとにad hocな列listを手で写す必要を減らし、comparison helperへ直接渡せます。

# %%
profile_catalog = {
    spec.profile: spec.to_dict() for spec in iter_ftqc_resource_profile_specs()
}
space_time_quantities = ftqc_resource_profile_quantities(
    FTQCResourceProfile.SPACETIME
)

print(profile_catalog[FTQCResourceProfile.SPACETIME]["description"])
print(profile_catalog[FTQCResourceProfile.SPACETIME]["quantities"])

assert FTQCResourceProfile.CHEMISTRY_QPE in profile_catalog
assert space_time_quantities[-1] == FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS

# %% [markdown]
# ## Compositional Algorithm Plans
#
# 新しいFTQCアルゴリズムに具体的なQamomile回路がまだない段階では、抽象的なsubroutine列としてmodel化します。resource planはcanonical quantityを明示的なaggregation ruleで合成します。count、depth、runtime、space-time costは逐次stepで加算し、量子ビットfootprintはピークを取り、問題メタデータとarchitecture knobは一貫していることを要求します。

# %%
prepare_step = FTQCResourcePlanStep(
    "prepare_filter",
    {
        FTQCResourceQuantity.LOGICAL_QUBITS: 12,
        FTQCResourceQuantity.TOFFOLI_GATES: 7,
        FTQCResourceQuantity.LOGICAL_DEPTH: 11,
        FTQCResourceQuantity.TARGET_PRECISION: sp.Float("0.001"),
    },
    repetitions=3,
    label="State preparation and filtering",
)
qpe_step = FTQCResourcePlanStep(
    "phase_estimation",
    {
        "logical_qubits": 18,
        "toffoli_gates": 100,
        "logical_depth": 120,
        "runtime_seconds": 20,
        "physical_qubits": 4000,
        "target_precision": sp.Float("0.001"),
    },
    repetitions=2,
)
resource_plan = FTQCResourcePlan(
    (prepare_step, qpe_step),
    title="Filtered QPE plan",
)

for row in resource_plan.to_quantity_table():
    print(row["quantity"], row["aggregation"], row["value"])

plan_values = resource_plan.resource_values()
assert plan_values[FTQCResourceQuantity.TOFFOLI_GATES] == 221
assert plan_values[FTQCResourceQuantity.LOGICAL_DEPTH] == 273
assert plan_values[FTQCResourceQuantity.LOGICAL_QUBITS] == 18
assert default_ftqc_resource_aggregation_rule("logical_qubits") == (
    FTQCResourceAggregationRule.PEAK
)

# %% [markdown]
# 同じplan objectは`resource_values()`を公開するため、comparison helperやbudget helperは化学計算estimateと同じように扱えます。resourceをstep間のピークとして合成したい場合は、そのquantityに対してplan-level ruleをoverrideします。各step自身のrepetition scalingは先に適用されます。

# %%
parallel_runtime_plan = FTQCResourcePlan(
    (prepare_step, qpe_step),
    aggregation={"runtime_seconds": FTQCResourceAggregationRule.PEAK},
)
plan_budget = evaluate_ftqc_resource_constraints(
    resource_plan,
    (
        FTQCResourceConstraint("logical_qubits", 20),
        FTQCResourceConstraint("toffoli_gates", 200),
    ),
    title="Toy plan budget",
)

assert parallel_runtime_plan.resource_values()[FTQCResourceQuantity.RUNTIME_SECONDS] == 40
assert plan_budget.satisfied[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
assert plan_budget.violated[0].quantity == FTQCResourceQuantity.TOFFOLI_GATES

# %% [markdown]
# block-encoding modelから直接planを作ることもできます。このplanは再利用するblock-encoding contractと、繰り返されるqubitized-walk stepを分けます。そのため、loader circuitが存在しない段階でも、PREPARE、SELECT、reflection、walk cost、QPE反復回数をreviewできます。

# %%
block_encoding = BlockEncodingResource(
    system_qubits=40,
    normalization=sp.Float("2.0e5"),
    select_cost_toffoli=4_000,
    prepare_cost_toffoli=500,
    reflection_cost_toffoli=100,
    ancilla_qubits=80,
    name="toy_lcu_loader",
)
block_plan = plan_qubitized_qpe_from_block_encoding(
    block_encoding,
    precision=sp.Float("0.0015"),
    qpe_register_qubits=12,
)
block_plan_values = block_plan.resource_values()

for step in block_plan.to_dict()["steps"]:
    print(step["name"], step["repetitions"])

assert block_plan.steps[0].name == "block_encoding_contract"
assert block_plan.steps[1].name == "qubitized_walk_qpe"
assert block_plan_values[FTQCResourceQuantity.WALK_COST_TOFFOLI] == 5100
assert block_plan_values[FTQCResourceQuantity.LOGICAL_QUBITS] == 132
assert block_plan_values[FTQCResourceQuantity.TOFFOLI_GATES] == (
    block_plan_values[FTQCResourceQuantity.QPE_ITERATIONS] * 5100
)

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

distance_budget = SurfaceCodeDistanceBudget(
    physical_error_rate=sp.Float("1e-3"),
    threshold_error_rate=sp.Float("1e-2"),
    target_logical_failure_probability=sp.Float("1e-9"),
    logical_operation_budget=1000,
)

print(distance_budget.to_dict())
assert distance_budget.code_distance == 21
assert (
    sp.Abs(
        distance_budget.resource_values()[FTQCResourceQuantity.LOGICAL_ERROR_RATE]
        - distance_budget.logical_failure_probability_per_operation
    )
    < sp.Float("1e-24")
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

assert architecture.code_distance == 21
assert architecture.physical_qubits_per_logical == 882
assert architecture.factory_qubits == 20000
assert sp.Abs(
    architecture.toffoli_throughput_per_second - sp.Float("4e7") / 21
) < sp.Float(
    "1e-9",
)

accuracy_budget = FTQCAccuracyBudget(
    target_precision=sp.Float("0.0016"),
    truncation_error=sp.Float("1e-4"),
)
print(accuracy_budget.to_dict())
assert sp.Abs(accuracy_budget.qpe_precision - sp.Float("0.0015")) < sp.Float("1e-12")

baseline_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary,
        method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
        walk_cost_toffoli=sp.Integer(4_000),
    )
)
compressed_model = accuracy_budget.with_model(
    ChemistryQPEModel(
        hamiltonian=scaled_summary.with_lambda_scale(
            sp.Float("0.5"),
            source="compressed_scaled_toy_pauli_lcu",
        ),
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        walk_cost_toffoli=sp.Integer(4_400),
        second_factor_rank=9,
    )
)

baseline = estimate_qubitized_chemistry_qpe_from_model(
    baseline_model,
    precision=accuracy_budget.qpe_precision,
    cost_model=cost_model,
)
compressed = estimate_qubitized_chemistry_qpe_from_model(
    compressed_model,
    precision=accuracy_budget.qpe_precision,
    cost_model=cost_model,
)

assert compressed.resource_values()[FTQCResourceQuantity.CODE_DISTANCE] == 21
assert compressed.to_dict()["architecture_values"]["code_distance"] == "21"
assert compressed_model.resource_values()[FTQCResourceQuantity.TRUNCATION_ERROR] == (
    sp.Float("1e-4")
)

comparison = compare_ftqc_resource_estimates(
    baseline,
    compressed,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

for row in comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4), "reduction:", sp.N(row.reduction, 4))

assert comparison[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison[0].ratio == sp.Float("0.5")
assert sp.Abs(comparison[1].ratio - sp.Float("0.55")) < sp.Float("1e-12")
assert comparison[4].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
assert compressed.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS] == (
    compressed.physical_qubits * compressed.runtime_seconds
)

# %% [markdown]
# 設計レビューでは、summary helperを使うと、同じ行をcandidateが小さい、大きい、変わらない、または現在の仮定ではsymbolicなまま、というグループに分けて読めます。`smaller`の先頭には、数値化できる範囲で削減率が大きい行が来ます。

# %%
comparison_summary = summarize_ftqc_resource_comparison(
    baseline,
    compressed,
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)

for row in comparison_summary.smaller:
    print("smaller:", row.label, "by", sp.N(row.reduction, 4))
for row in comparison_summary.larger:
    print("larger:", row.label, "by", sp.N(-row.reduction, 4))

assert comparison_summary.smaller[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison_summary.larger[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
assert any(
    row.quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
    for row in comparison_summary.larger
)
assert comparison_summary.symbolic == ()

# %% [markdown]
# 設計レビュー用の自己完結したartifactが必要な場合は、reportを作ります。label、選択したprofile、行の順序、grouped summary countをまとめて保持できます。reportからは優先順位つきfindingも作れます。最初に大きな削減、次に大きなresource tradeoffが並びます。

# %%
comparison_report = build_ftqc_resource_comparison_report(
    baseline,
    compressed,
    title="Toy factorization comparison",
    baseline_label="THC-style",
    candidate_label="Compressed",
    quantities=(
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
    ),
    profile=FTQCResourceProfile.SPACETIME,
)
report_rows = comparison_report.to_row_table()
review_findings = comparison_report.to_review_findings(
    max_improvements=2,
    max_tradeoffs=2,
)

print(comparison_report.to_dict()["title"])
print(comparison_report.to_dict()["counts"])
for finding in review_findings:
    print(finding.direction, "-", finding.headline)

assert comparison_report.profile == FTQCResourceProfile.SPACETIME
assert report_rows[0]["baseline_label"] == "THC-style"
assert report_rows[0]["candidate_label"] == "Compressed"
assert report_rows[0]["quantity"] == "qpe_iterations"
assert review_findings[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
assert comparison_report.to_dict()["findings"][0]["direction"] == "smaller"

# %% [markdown]
# ## Reference provenance
#
# estimateはmethod-levelの研究referenceも保持します。これは数式とは意図的に分けています。reportは、上のsynthetic exampleを特定分子の再現とみなさずに、どの論文がmodelの動機になったかを示せます。

# %%
for reference in compressed.references:
    print(reference.key, "-", reference.title)

compressed_reference_keys = {reference.key for reference in compressed.references}
assert "arXiv:2403.03502" in compressed_reference_keys
assert "arXiv:2412.01338" in compressed_reference_keys
assert compressed.to_dict()["references"][0]["url"].startswith("https://arxiv.org/")

# %% [markdown]
# ## Formula provenance
#
# FTQC設計レビューでは値だけでは不十分です。重要なresourceをどのsymbolic formulaで計算したかも確認する必要があります。estimateは`resource_values()`と同じcanonical quantity keyを使うformula tableを公開します。

# %%
formula_rows = compressed.to_formula_table()
for row in formula_rows:
    if row["quantity"] in {"qpe_iterations", "toffoli_gates", "runtime_seconds"}:
        print(row["label"], "=", row["expression"])

formula_by_quantity = {row["quantity"]: row for row in formula_rows}
assert formula_by_quantity["qpe_iterations"]["expression"] == (
    "lambda_norm/target_precision"
)
assert formula_by_quantity["toffoli_gates"]["depends_on"] == [
    "qpe_iterations",
    "walk_cost_toffoli",
]
assert "formulas" in compressed.to_dict()

# %% [markdown]
# ## 共通の論理リソース形状
#
# FTQC estimateは、circuit-levelの`estimate_resources()`と同じ`ResourceEstimate`形状でlogical workを公開することもできます。このviewはphysical量子ビットとruntimeを意図的に含めません。これらはarchitecture仮定に依存するため、FTQC estimate側に残します。

# %%
logical_view = compressed.to_logical_resource_estimate()

print(logical_view)
assert logical_view.qubits == compressed.logical_qubits
assert logical_view.gates.total == compressed.toffoli_gates
assert logical_view.gates.multi_qubit == compressed.toffoli_gates
assert logical_view.gates.oracle_calls["qpe_iterations"] == compressed.qpe_iterations
assert "physical_qubits_per_logical" not in logical_view.parameters

# %% [markdown]
# ## Architecture感度
#
# estimateを作った後で別のarchitecture modelへreliftすると、algorithm modelを作り直さずにhardware仮定を調べられます。

# %%
faster_architecture = SurfaceCodeCostModel(
    code_distance=10,
    physical_cycle_time_seconds=sp.Float("5e-8"),
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=1,
    factory_count=8,
    physical_qubits_per_factory=2500,
    factory_cycles_per_toffoli=2,
)
relifted_baseline = baseline.with_cost_model(faster_architecture)

architecture_comparison = compare_ftqc_resource_estimates(
    baseline,
    relifted_baseline,
    quantities=("physical_qubits", "runtime_seconds"),
)

for row in architecture_comparison:
    print(row.label, "ratio:", sp.N(row.ratio, 4))

assert relifted_baseline.logical_qubits == baseline.logical_qubits
assert relifted_baseline.toffoli_gates == baseline.toffoli_gates
assert relifted_baseline.resource_values()[FTQCResourceQuantity.CODE_DISTANCE] == 10
assert sp.Abs(architecture_comparison[0].ratio - sp.Rational(350, 691)) < sp.Float(
    "1e-12"
)
assert sp.Abs(architecture_comparison[1].ratio - sp.Rational(10, 21)) < sp.Float(
    "1e-12"
)

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
# ## Budget constraints
#
# early-FTQCの研究では、別のreview questionも重要です。estimateがphysical量子ビットやruntimeのbudgetに収まるか、という問いです。budget reportを使うと、同じcanonical resource valueを明示的なconstraintと照合できます。architecture仮定がまだsymbolicな場合、marginは未判断のまま残ります。

# %%
budget_report = evaluate_ftqc_resource_constraints(
    uwc_trotter,
    (
        FTQCResourceConstraint(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            100_000,
            label="Early-FTQC physical-qubit budget",
        ),
        FTQCResourceConstraint(
            FTQCResourceQuantity.RUNTIME_SECONDS,
            60 * 60,
            label="One-hour runtime budget",
        ),
        FTQCResourceConstraint(
            FTQCResourceQuantity.LOGICAL_DEPTH,
            plain_trotter.logical_depth,
            label="No worse than plain Trotter depth",
        ),
    ),
    title="Synthetic early-FTQC budget",
)

for result in budget_report.results:
    print(result.status, result.label, "margin:", sp.N(result.margin, 4))

assert budget_report.satisfied[0].quantity == FTQCResourceQuantity.PHYSICAL_QUBITS
assert budget_report.violated[0].quantity == FTQCResourceQuantity.RUNTIME_SECONDS
assert budget_report.to_dict()["counts"] == {
    "satisfied": 2,
    "violated": 1,
    "symbolic": 0,
}

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
# - 近年のFTQC化学計算研究から、Hamiltonian normalization、target precision、truncation error、QPE反復回数、non-Clifford count、logical depth、logical space-time volume、physical量子ビット、runtime、physical qubit-secondsを分けて追跡する必要があることがわかります。
# - `iter_ftqc_research_signals`は、研究方向をQamomileがreportするcanonical quantityへ対応づけます。
# - `FTQCResourceProfile`を使うと、space-time profileのような再利用可能なquantity bundleをcomparison helperへ直接渡せます。
# - `FTQCResourcePlan`を使うと、具体的な回路実装ができる前に抽象的なFTQC subroutineを合成できます。
# - `plan_qubitized_qpe_from_block_encoding`は、block-encoding contractをPREPARE/SELECT/reflection/QPE resource planへ変換します。
# - Qamomileはこれらの量をアルゴリズム上のメタデータとして保持するため、circuit IRはbackend-neutralに保たれます。
# - accuracy budgetを使うと、estimateを比較する前にtotal target precisionをrepresentation truncation errorとQPE precisionへ分けられます。
# - Formula provenanceにより、重要なresource quantityの背後にあるsymbolic derivationを公開できます。
# - Surface-code仮定は別にmodel化し、chemistry推定器が使うcost modelへ変換できます。
# - Surface-code distanceは、logical failure budgetから選んだうえで、logical resourceをphysical量子ビットとruntimeへliftできます。
# - code distanceなどのarchitecture quantityは各estimateに残るため、reportでphysical resource仮定をauditできます。
# - estimateに研究referenceを保持し、どの論文がsymbolic modelの根拠になったかをreportでauditできるようにします。
# - reportでcircuit-level estimateと同じ形が必要な場合は、FTQC estimateを共通のlogical `ResourceEstimate` objectとして見られます。
# - 既存のlogical estimateは、algorithm estimateを作り直さずに新しいarchitecture仮定でreliftできます。
# - `compare_ftqc_resource_estimates`を使うと、特定のchemistry factorizationをhard-codeせず、symbolicな推定をreviewしやすいsavings tableへ変換できます。
# - `summarize_ftqc_resource_comparison`を使うと、その行を小さい、大きい、変わらない、symbolicな変化へ分けて設計レビューできます。
# - `build_ftqc_resource_comparison_report`を使うと、label、profile、行、優先順位つきfinding、grouped countをreview artifactとしてまとめられます。
# - `evaluate_ftqc_resource_constraints`を使うと、estimateをphysical量子ビット、runtime、depthなどの明示的なresource budgetと照合できます。
